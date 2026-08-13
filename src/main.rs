//! Test the Fan method from [1]
//!
//! Coordinate Systems:
//! - bcar-frame: car body (XYZ)
//! - bcam-frame: camera body (XYZ)
//! - c-frame: camera optical (XYZ)
//! - n-frame: horizontal reference (ENU)
//! - v-frame: observation point coordinate system (XYZ)
//!
//! The v-frame is defined by the observation direction, v_c, in the c-frame.
//! The z_v axis points along v_c.
//! The y_v axis is located in the x_c y_c plane.
//! The x_v axis is found by right-hand rule.
//!
//! I want to iterate over the results from the dataset.
//! The dataset contains information from three sensors.
//! All of the sensor data has been previously time-wise resampled.
//! Each row (or image) corresponds to a single unique frame.
//!
//! 1. Reference system (Novatel Oem7 INSPVA)
//!   - Lat, lon of car
//!   - Orientation of bcar-frame in n-frame
//!   - Stored in CSV file
//! 2. GPS Time (Novatel Oem7 Time)
//!   - Datetime
//!   - Stored in CSV file
//! 3. Polarization Camera (Lucid Vision PHX050S-P/Q)
//!   - Measurements of the skylight polarization pattern
//!   - Stored as png images named with the frame they correspond to
//!
//! TODO before running on the HPC:
//! - [x] Get better metric for error in the estimated solar azimuth
//!   - Want angular error
//! - [x] Figure out what data to dump: FrameResult structure
//!   - frame index
//!   - datetime at frame
//!   - psa solar vector, estimated solar vector (with and without rayleigh point filter)
//!   - * timing information
//! - [x] Get timing information for the algorithm
//! - [x] Also compute the s_c using all e-vectors (ignoring rayleigh points)
//! - [x] Move helper functions to src/lib.rs; add documentation comment and tests to each.
//! - [x] Dump the frame result structure as a CSV file
//!   - Ensure the file has a descriptive name
//!   - Unpack the vectors into NAME.x, NAME.y, NAME.z fields
//! - [ ] Add comments throughout that explain the implementation of the algorithm
//! - [ ] Ensure no crashes during long running HPC job
//!   - Checkpointing
//!   - Ensure all errors are handled: default to skip frame if problems arise
//!   - Optimize "low hanging fruit" without spending too much time or too many changes on it
//! - [ ] Good logging to check on HPC job progress
//! - [ ] Let user pass dataset path as argument (other arguments?)
//! - [ ] SLURM script for starting the runner on the hpc [2]
//!
//! [1] https://ieeexplore.ieee.org/document/11005588
//! [2] https://slurm.schedmd.com/overview.html

use std::{
    error::Error,
    f64::consts::{FRAC_PI_2, PI},
    path::{Path, PathBuf},
    time::Instant,
};

use chrono::{DateTime, Utc};
use fan_method::{
    aop_from_ev, aop_sensor_to_v, compute_azimuth, compute_rot_v_c, compute_s_c, elapsed_ms,
    ev_from_aop, optical_path, pixel, psa, rayleigh_ev, tait_bryan,
};
use nalgebra::{Rotation3, Vector3};

const ROWS: usize = 1024;
const COLS: usize = 1224;
const N_PIXELS: usize = ROWS * COLS;
const CENTER_ROW: usize = ROWS / 2;
const CENTER_COL: usize = COLS / 2;
const PIXEL_SIZE_MM: f64 = 0.0069;
const FOCAL_LENGTH_MM: f64 = 8.0;
const AOP_THRESHOLD_DEG: f64 = 5.0;
const TIME_CSV: &'static str =
    "/home/ben/git/research/polcam_dataset/2025-11-24/rmc/novatel_oem7_time/novatel_oem7_time.csv";
const INS_CSV: &'static str = "/home/ben/git/research/polcam_dataset/2025-11-24/rmc/novatel_oem7_inspva/novatel_oem7_inspva.csv";
const IMAGE_DIR: &'static str =
    "/home/ben/git/research/polcam_dataset/2025-11-24/rmc/camera_driver_gv_vis_image_raw";
const OUTPUT_CSV: &'static str = "frame_results.csv";
const N_FRAMES: usize = 1;

fn main() -> Result<(), Box<dyn Error>> {
    println!("Fan Method v0.1");
    println!("Implemented by Ben Potter in August 2026");
    println!("See original paper: https://ieeexplore.ieee.org/document/11005588");

    // Fixed rotations determined by experimental setup.
    let rot_bcam_bcar = tait_bryan(PI, 0., 0.);
    let rot_c_bcam = tait_bryan(0., 0., 0.);
    let rot_c_bcar = rot_c_bcam * rot_bcam_bcar;

    let n_pixels = N_PIXELS;
    let aop_threshold_rad = AOP_THRESHOLD_DEG.to_radians();
    let image_dir = PathBuf::new().join(IMAGE_DIR);

    let mut all_p_c = Vec::new();
    let mut all_v_c = Vec::new();

    // Compute the optical paths once at startup.
    let optical_paths_start = Instant::now();
    for row in 0..ROWS {
        for col in 0..COLS {
            let p_c = pixel(row, col, CENTER_ROW, CENTER_COL, PIXEL_SIZE_MM);
            let v_c = optical_path(p_c, FOCAL_LENGTH_MM);

            all_p_c.push(p_c);
            all_v_c.push(v_c);
        }
    }
    let optical_paths_duration = optical_paths_start.elapsed();
    println!(
        "precomputed optical paths in {:.3} ms",
        optical_paths_duration.as_secs_f64() * 1_000.0
    );

    let system = System {
        all_p_c,
        all_v_c,
        aop_threshold_rad,
        n_pixels,
    };

    let time_frame = fan_method::dataset::read_time(TIME_CSV).unwrap();
    let ins_frame = fan_method::dataset::read_ins(INS_CSV).unwrap();
    let mut results = Vec::new();
    for i in 0..N_FRAMES {
        // TODO: improve logging
        // println!("start frame {i}");

        // Given by the ins_frame; lets us determine n-frame to c-frame.
        // Azimuth is taken CW from North (likely need to negate it).
        // NovAtel azimuth is clockwise from North.  In an ENU n-frame, a
        // level car x-axis therefore points along (sin az, cos az, 0), which
        // is produced by Rz(pi/2 - az).  That matrix maps bcar-frame
        // components into n-frame components, so invert it before applying it
        // to the sun vector s_n.
        let rot_n_bcar = tait_bryan(
            FRAC_PI_2 - ins_frame[i].azimuth.to_radians(),
            ins_frame[i].pitch.to_radians(),
            ins_frame[i].roll.to_radians(),
        );
        let rot_c_n = rot_c_bcar * rot_n_bcar.inverse();

        // Compute measured aop from image.
        let image_file = format!("camera_driver_gv_vis_image_raw_{:04}.png", i);
        let image_file = image_dir.join(image_file);
        let aop = match fan_method::dataset::read_image(image_file) {
            Ok(aop) => aop,
            Err(e) => {
                eprintln!("failed to read image: {e}");
                continue;
            }
        };

        let frame = Frame {
            index: i,
            rot_c_n,
            time: time_frame[i],
            lat: ins_frame[i].lat,
            lon: ins_frame[i].lon,
            aop_s: aop,
        };

        let result = process_frame(&frame, &system);
        println!(
            "frame {} timing: total={:.3} ms, psa={:.3} ms, pixel_loop={:.3} ms, eigendecomp={:.3} ms",
            result.index,
            result.total_duration_ms,
            result.psa_duration_ms,
            result.pixel_loop_duration_ms,
            result.eigendecomp_duration_ms,
        );
        results.push(result);
    }

    write_frame_results(OUTPUT_CSV, &results)?;
    println!("wrote {} frame results to {OUTPUT_CSV}", results.len());

    Ok(())
}

/// Stores information which remains static across frames.
struct System {
    /// Pixel positions in the c-frame.
    all_p_c: Vec<Vector3<f64>>,
    /// Optical paths in the c-frame.
    all_v_c: Vec<Vector3<f64>>,
    /// Threshold where measured AoP values are considered to follow the Rayleigh model.
    aop_threshold_rad: f64,
    /// Number of pixels
    n_pixels: usize,
}

/// Stores information unique to each frame.
struct Frame {
    index: usize,
    rot_c_n: Rotation3<f64>,
    time: DateTime<Utc>,
    lat: f64,
    lon: f64,
    /// AoP in the sensor frame.
    aop_s: Vec<f64>,
}

/// Result of performing algorithm on one frame of a trajectory.
#[derive(Default, Debug)]
struct FrameResult {
    /// Frame index within a trajectory.
    index: usize,
    /// Datetime of the frame within a trajectory.
    time: DateTime<Utc>,
    /// Latitude of the frame within a trajectory.
    lat: f64,
    /// Longitude of the frame within a trajectory.
    lon: f64,

    /// Solar vector computed by PSA algorithm.
    psa_s_c: Vector3<f64>,
    /// Solar vector computed by eigendecomp. on all measured e-vectors.
    s_c: Vector3<f64>,
    /// Solar vector computed by eigendecomp. on measured e-vectors corresponding to rayleigh points.
    rayleigh_s_c: Vector3<f64>,

    /// Azimuth of solar vector computed by PSA algorithm in radians.
    psa_azimuth: f64,
    /// Azimuth of solar vector computed by eigendecomp. on all measured e-vectors in radians.
    azimuth: f64,
    /// Azimuth of solar vector computed by eigendecomp. on measured e-vectors corresponding to rayleigh points in radians.
    rayleigh_azimuth: f64,

    /// Number of pixels which correspond to rayleigh points.
    n_rayleigh_points: usize,
    /// Fraction of pixels which correspond to rayleigh points.
    frac_rayleigh_points: f64,

    /// Total wall-clock time spent processing this frame, in milliseconds.
    total_duration_ms: f64,
    /// Wall-clock time spent computing the PSA solar vector, in milliseconds.
    psa_duration_ms: f64,
    /// Wall-clock time spent iterating over pixels and computing e-vectors, in milliseconds.
    pixel_loop_duration_ms: f64,
    /// Wall-clock time spent computing solar vectors via eigendecomposition, in milliseconds.
    eigendecomp_duration_ms: f64,
}

/// The meat and potatoes of the algorithm.
fn process_frame(frame: &Frame, system: &System) -> FrameResult {
    let total_start = Instant::now();

    let psa_start = Instant::now();
    let psa_s_n = psa(frame.lat, frame.lon, frame.time);
    let psa_s_c = frame.rot_c_n * psa_s_n;
    let psa_duration_ms = elapsed_ms(psa_start);

    let mut aop_v = vec![0f64; system.n_pixels];
    let mut rayleigh_aop_v = vec![0f64; system.n_pixels];
    let mut n_rayleigh_points = 0usize;
    let mut rayleigh_point = vec![false; system.n_pixels];
    let mut e_c = vec![Vector3::zeros(); system.n_pixels];

    let pixel_loop_start = Instant::now();
    for i in 0..system.n_pixels {
        let v_c = &system.all_v_c[i];
        let rayleigh_e_c = rayleigh_ev(v_c, &psa_s_c);

        let rot_v_c = compute_rot_v_c(v_c);
        let rayleigh_e_v = rot_v_c * rayleigh_e_c;

        aop_v[i] = aop_sensor_to_v(frame.aop_s[i], system.all_p_c[i]);
        rayleigh_aop_v[i] = aop_from_ev(&rayleigh_e_v);
        rayleigh_point[i] =
            fan_method::aop_threshold(aop_v[i], rayleigh_aop_v[i], system.aop_threshold_rad);

        if rayleigh_point[i] {
            n_rayleigh_points += 1;
        }

        let rot_c_v = rot_v_c.inverse();
        let e_v = ev_from_aop(aop_v[i]);
        e_c[i] = rot_c_v * e_v;
    }
    let pixel_loop_duration_ms = elapsed_ms(pixel_loop_start);

    let eigendecomp_start = Instant::now();
    let s_c = compute_s_c(&e_c);
    let rayleigh_e_c: Vec<_> = e_c
        .iter()
        .zip(rayleigh_point.iter())
        .filter_map(|(e_c, is_rayleigh_point)| is_rayleigh_point.then_some(*e_c))
        .collect();
    let rayleigh_s_c = if rayleigh_e_c.is_empty() {
        eprintln!(
            "frame {} had no rayleigh points; falling back to all e-vectors for rayleigh_s_c",
            frame.index
        );
        s_c
    } else {
        compute_s_c(&rayleigh_e_c)
    };
    let eigendecomp_duration_ms = elapsed_ms(eigendecomp_start);
    let total_duration_ms = elapsed_ms(total_start);

    let psa_azimuth = compute_azimuth(&psa_s_c);
    let azimuth = compute_azimuth(&s_c);
    let rayleigh_azimuth = compute_azimuth(&rayleigh_s_c);
    let frac_rayleigh_points = n_rayleigh_points as f64 / system.n_pixels as f64;

    FrameResult {
        index: frame.index,
        time: frame.time,
        lat: frame.lat,
        lon: frame.lon,
        psa_s_c,
        s_c,
        rayleigh_s_c,
        psa_azimuth,
        azimuth,
        rayleigh_azimuth,
        n_rayleigh_points,
        frac_rayleigh_points,
        total_duration_ms,
        psa_duration_ms,
        pixel_loop_duration_ms,
        eigendecomp_duration_ms,
    }
}

/// Write frame results to a CSV file, unpacking vector fields into x/y/z columns.
fn write_frame_results(
    path: impl AsRef<Path>,
    results: &[FrameResult],
) -> Result<(), Box<dyn Error>> {
    let mut writer = csv::Writer::from_path(path)?;

    writer.write_record([
        "index",
        "time",
        "lat",
        "lon",
        "psa_s_c.x",
        "psa_s_c.y",
        "psa_s_c.z",
        "s_c.x",
        "s_c.y",
        "s_c.z",
        "rayleigh_s_c.x",
        "rayleigh_s_c.y",
        "rayleigh_s_c.z",
        "psa_azimuth",
        "azimuth",
        "rayleigh_azimuth",
        "n_rayleigh_points",
        "frac_rayleigh_points",
        "total_duration_ms",
        "psa_duration_ms",
        "pixel_loop_duration_ms",
        "eigendecomp_duration_ms",
    ])?;

    for result in results {
        writer.write_record([
            result.index.to_string(),
            result.time.to_rfc3339(),
            result.lat.to_string(),
            result.lon.to_string(),
            result.psa_s_c.x.to_string(),
            result.psa_s_c.y.to_string(),
            result.psa_s_c.z.to_string(),
            result.s_c.x.to_string(),
            result.s_c.y.to_string(),
            result.s_c.z.to_string(),
            result.rayleigh_s_c.x.to_string(),
            result.rayleigh_s_c.y.to_string(),
            result.rayleigh_s_c.z.to_string(),
            result.psa_azimuth.to_string(),
            result.azimuth.to_string(),
            result.rayleigh_azimuth.to_string(),
            result.n_rayleigh_points.to_string(),
            result.frac_rayleigh_points.to_string(),
            result.total_duration_ms.to_string(),
            result.psa_duration_ms.to_string(),
            result.pixel_loop_duration_ms.to_string(),
            result.eigendecomp_duration_ms.to_string(),
        ])?;
    }

    writer.flush()?;
    Ok(())
}
