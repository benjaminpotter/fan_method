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
//! - [x] Add comments throughout that explain the implementation of the algorithm
//! - [x] Optimize for HPC context -> parallelize?
//! - [x] Ensure no crashes during long running HPC job
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
    collections::HashSet,
    error::Error,
    f64::consts::{FRAC_PI_2, PI},
    fs::{File, OpenOptions},
    panic::{AssertUnwindSafe, catch_unwind},
    path::{Path, PathBuf},
    time::Instant,
};

use chrono::{DateTime, Utc};
use fan_method::{
    accumulate_e_vector, aop_from_ev, aop_sensor_to_v, compute_azimuth, compute_rot_v_c,
    compute_s_c_from_matrix, elapsed_ms, ev_from_aop, optical_path, pixel, psa, rayleigh_ev,
    tait_bryan,
};
use nalgebra::{Matrix3, Rotation3, Vector3};
use rayon::prelude::*;

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
const N_FRAMES: usize = 10;

macro_rules! log_info {
    ($event:literal, $($arg:tt)*) => {
        log_event("INFO", $event, format!($($arg)*))
    };
}

macro_rules! log_warn {
    ($event:literal, $($arg:tt)*) => {
        log_event("WARN", $event, format!($($arg)*))
    };
}

macro_rules! log_error {
    ($event:literal, $($arg:tt)*) => {
        log_event("ERROR", $event, format!($($arg)*))
    };
}

fn log_event(level: &str, event: &str, message: String) {
    println!(
        "ts={} level={} event={} msg={:?}",
        Utc::now().to_rfc3339(),
        level,
        event,
        message
    );
}

fn main() -> Result<(), Box<dyn Error>> {
    log_info!("startup", "Fan Method v0.1");
    log_info!("startup", "Implemented by Ben Potter in August 2026");
    log_info!(
        "startup",
        "See original paper: https://ieeexplore.ieee.org/document/11005588"
    );

    // Fixed rotations determined by experimental setup.
    //
    // The algorithm estimates the sun direction in the camera optical frame (c-frame),
    // but the reference INS attitude is provided for the car body frame (bcar-frame).
    // These fixed rotations describe how the camera was mounted in the vehicle, so
    // they let us move vectors between the car body, camera body, and camera optical
    // frames.
    let rot_bcam_bcar = tait_bryan(PI, 0., 0.);
    let rot_c_bcam = tait_bryan(0., 0., 0.);
    let rot_c_bcar = rot_c_bcam * rot_bcam_bcar;

    let n_pixels = N_PIXELS;
    // Pixels whose measured AoP is close to the PSA/Rayleigh-predicted AoP are
    // treated as Rayleigh points and used for the filtered sun estimate.
    let aop_threshold_rad = AOP_THRESHOLD_DEG.to_radians();
    let image_dir = PathBuf::new().join(IMAGE_DIR);

    let mut all_p_c = Vec::new();
    let mut all_v_c = Vec::new();

    // Compute camera-frame pixel locations and optical-path vectors once at startup.
    // These depend only on the camera geometry, not on the frame's time, pose, or image
    // data, so caching them avoids repeating this work for every image.
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
    log_info!(
        "precompute_optical_paths",
        "duration_ms={:.3}",
        optical_paths_duration.as_secs_f64() * 1_000.0
    );

    let system = System {
        all_p_c,
        all_v_c,
        aop_threshold_rad,
        n_pixels,
    };

    let time_frame = fan_method::dataset::read_time(TIME_CSV)?;
    let ins_frame = fan_method::dataset::read_ins(INS_CSV)?;
    let n_available_frames = time_frame.len().min(ins_frame.len());
    let n_frames_to_process = N_FRAMES.min(n_available_frames);

    if N_FRAMES > n_available_frames {
        log_warn!(
            "frame_count_truncated",
            "requested_frames={N_FRAMES} available_frames={n_available_frames} processing_frames={n_frames_to_process}"
        );
    }

    // Open the CSV before the long-running loop and flush after every successful frame.
    // This makes the output a useful checkpoint: if the job is killed, completed frames
    // are already on disk and do not need to be recomputed.
    let completed_frames = read_completed_frame_indices(OUTPUT_CSV)?;
    let mut writer = open_frame_results_writer(OUTPUT_CSV, !completed_frames.is_empty())?;
    if completed_frames.is_empty() {
        write_frame_results_header(&mut writer)?;
        writer.flush()?;
    } else {
        log_info!(
            "resume_output_csv",
            "completed_frames={} output_csv={OUTPUT_CSV}",
            completed_frames.len()
        );
    }

    let mut n_processed = 0usize;
    let mut n_skipped = 0usize;
    let mut n_already_completed = 0usize;
    for i in 0..n_frames_to_process {
        if completed_frames.contains(&i) {
            n_already_completed += 1;
            continue;
        }

        log_info!("frame_start", "frame={i}");

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

        // Load the measured angle of polarization (AoP) image for this frame. The
        // dataset reader returns one AoP value per pixel in sensor/image coordinates.
        let image_file = format!("camera_driver_gv_vis_image_raw_{:04}.png", i);
        let image_file = image_dir.join(image_file);
        let aop = match fan_method::dataset::read_image(&image_file) {
            Ok(aop) => aop,
            Err(e) => {
                n_skipped += 1;
                log_warn!(
                    "skip_frame_image_read_failed",
                    "frame={i} image_file={image_file:?} error={e}"
                );
                continue;
            }
        };

        if aop.len() != system.n_pixels {
            n_skipped += 1;
            log_warn!(
                "skip_frame_bad_aop_len",
                "frame={i} actual_aop_len={} expected_aop_len={}",
                aop.len(),
                system.n_pixels
            );
            continue;
        }

        let frame = Frame {
            index: i,
            rot_c_n,
            time: time_frame[i],
            lat: ins_frame[i].lat,
            lon: ins_frame[i].lon,
            aop_s: aop,
        };

        let result = match catch_unwind(AssertUnwindSafe(|| process_frame(&frame, &system))) {
            Ok(Ok(result)) => result,
            Ok(Err(e)) => {
                n_skipped += 1;
                log_warn!("skip_frame_processing_error", "frame={i} error={e}");
                continue;
            }
            Err(_) => {
                n_skipped += 1;
                log_error!("skip_frame_processing_panic", "frame={i}");
                continue;
            }
        };

        log_info!(
            "frame_processed",
            "frame={} total_duration_ms={:.3} psa_duration_ms={:.3} pixel_loop_duration_ms={:.3} eigendecomp_duration_ms={:.3} n_rayleigh_points={} frac_rayleigh_points={:.6}",
            result.index,
            result.total_duration_ms,
            result.psa_duration_ms,
            result.pixel_loop_duration_ms,
            result.eigendecomp_duration_ms,
            result.n_rayleigh_points,
            result.frac_rayleigh_points,
        );
        write_frame_result(&mut writer, &result)?;
        writer.flush()?;
        n_processed += 1;
    }

    log_info!(
        "finished",
        "processed_frames={n_processed} skipped_frames={n_skipped} already_completed_frames={n_already_completed} output_csv={OUTPUT_CSV}"
    );

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

/// Per-pixel contributions reduced across the image during frame processing.
#[derive(Clone)]
struct PixelAccum {
    /// Normal-equation matrix accumulated from all measured e-vectors.
    m_all: Matrix3<f64>,
    /// Normal-equation matrix accumulated only from Rayleigh-classified e-vectors.
    m_rayleigh: Matrix3<f64>,
    /// Number of finite AoP pixels included in the all-pixel estimate.
    n_valid_pixels: usize,
    /// Number of pixels classified as Rayleigh points.
    n_rayleigh_points: usize,
}

impl PixelAccum {
    fn zero() -> Self {
        Self {
            m_all: Matrix3::zeros(),
            m_rayleigh: Matrix3::zeros(),
            n_valid_pixels: 0,
            n_rayleigh_points: 0,
        }
    }

    fn combine(self, other: Self) -> Self {
        Self {
            m_all: self.m_all + other.m_all,
            m_rayleigh: self.m_rayleigh + other.m_rayleigh,
            n_valid_pixels: self.n_valid_pixels + other.n_valid_pixels,
            n_rayleigh_points: self.n_rayleigh_points + other.n_rayleigh_points,
        }
    }
}

/// Process a single trajectory frame with the Fan-method pipeline.
///
/// The high-level flow is:
/// 1. Use PSA as a reference solar direction for Rayleigh-point classification.
/// 2. Convert every measured pixel AoP into an e-vector in the camera frame.
/// 3. Classify pixels whose measured AoP agrees with the PSA/Rayleigh prediction.
/// 4. Estimate the solar direction from all e-vectors and from Rayleigh-filtered e-vectors.
fn process_frame(frame: &Frame, system: &System) -> Result<FrameResult, String> {
    if frame.aop_s.len() != system.n_pixels {
        return Err(format!(
            "frame has {} AoP values but system expects {} pixels",
            frame.aop_s.len(),
            system.n_pixels
        ));
    }

    let total_start = Instant::now();

    // PSA gives an independent reference sun vector in the local horizontal ENU
    // frame. Rotate it into the camera optical frame so it can be compared against
    // per-pixel optical paths and e-vectors.
    let psa_start = Instant::now();
    let psa_s_n = psa(frame.lat, frame.lon, frame.time);
    let psa_s_c = frame.rot_c_n * psa_s_n;
    let psa_duration_ms = elapsed_ms(psa_start);

    // Process pixels in parallel. Each worker builds local `3x3` normal-equation
    // matrices, then Rayon reduces those local matrices into frame-level totals. This
    // avoids locks, avoids storing one e-vector per pixel, and keeps eigendecomposition
    // limited to two tiny `3x3` matrices after the parallel pixel pass.
    let pixel_loop_start = Instant::now();
    let pixel_accum = (0..system.n_pixels)
        .into_par_iter()
        .map(|i| {
            let aop_s = frame.aop_s[i];
            if !aop_s.is_finite() {
                return PixelAccum::zero();
            }

            let v_c = &system.all_v_c[i];

            // Predict the Rayleigh e-vector for this optical path using the PSA sun
            // direction. This predicted polarization direction is used only to decide
            // whether the pixel behaves like a Rayleigh point.
            let rayleigh_e_c = rayleigh_ev(v_c, &psa_s_c);

            // The Fan method defines a local observation frame (v-frame) for each pixel,
            // with +Z along that pixel's optical path. AoP comparisons are made in this
            // frame because AoP is measured in the plane perpendicular to the observation
            // direction.
            let rot_v_c = compute_rot_v_c(v_c);
            let rayleigh_e_v = rot_v_c * rayleigh_e_c;

            // Convert the measured sensor AoP into the pixel's v-frame, convert the
            // predicted Rayleigh e-vector into its AoP, then compare the two axial angles.
            let aop_v = aop_sensor_to_v(aop_s, system.all_p_c[i]);
            let rayleigh_aop_v = aop_from_ev(&rayleigh_e_v);
            let is_rayleigh_point =
                fan_method::aop_threshold(aop_v, rayleigh_aop_v, system.aop_threshold_rad);

            // Convert the measured AoP into a unit e-vector in the v-frame, then rotate it
            // back into the camera frame. The eigendecomposition later uses these camera-
            // frame e-vectors to find the sun direction most nearly perpendicular to them.
            let rot_c_v = rot_v_c.inverse();
            let e_v = ev_from_aop(aop_v);
            let e_c = rot_c_v * e_v;
            if !(e_c.x.is_finite() && e_c.y.is_finite() && e_c.z.is_finite()) {
                return PixelAccum::zero();
            }

            let mut accum = PixelAccum::zero();
            accum.n_valid_pixels = 1;
            accumulate_e_vector(&mut accum.m_all, &e_c);
            if is_rayleigh_point {
                accumulate_e_vector(&mut accum.m_rayleigh, &e_c);
                accum.n_rayleigh_points = 1;
            }
            accum
        })
        .reduce(PixelAccum::zero, PixelAccum::combine);
    let pixel_loop_duration_ms = elapsed_ms(pixel_loop_start);

    if pixel_accum.n_valid_pixels == 0 {
        return Err("frame contains no finite AoP measurements".to_string());
    }

    let eigendecomp_start = Instant::now();

    // Estimate the sun direction from all measured e-vectors. For ideal Rayleigh
    // scattering, every valid e-vector is perpendicular to the solar vector; therefore
    // the best-fit sun vector is the unit vector that minimizes summed squared dot
    // products with all e-vectors, solved via eigendecomposition.
    let s_c = compute_s_c_from_matrix(pixel_accum.m_all);

    // Repeat the same eigendecomposition using only pixels classified as Rayleigh
    // points. This filtered estimate should be less affected by clouds, reflections,
    // saturation, or other pixels whose AoP does not follow the Rayleigh model.
    let rayleigh_s_c = if pixel_accum.n_rayleigh_points == 0 {
        log_warn!(
            "no_rayleigh_points",
            "frame={} fallback=all_e_vectors_for_rayleigh_s_c",
            frame.index
        );
        s_c
    } else {
        compute_s_c_from_matrix(pixel_accum.m_rayleigh)
    };
    let eigendecomp_duration_ms = elapsed_ms(eigendecomp_start);
    let total_duration_ms = elapsed_ms(total_start);

    // Collapse the three solar vectors to horizontal azimuth angles for easier
    // comparison in downstream analysis. The full vectors are still retained in the
    // CSV output.
    let psa_azimuth = compute_azimuth(&psa_s_c);
    let azimuth = compute_azimuth(&s_c);
    let rayleigh_azimuth = compute_azimuth(&rayleigh_s_c);
    let n_rayleigh_points = pixel_accum.n_rayleigh_points;
    let frac_rayleigh_points = n_rayleigh_points as f64 / system.n_pixels as f64;

    Ok(FrameResult {
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
    })
}

/// Read frame indices already present in an existing result CSV.
///
/// This enables simple restart/resume behavior for HPC jobs: if the process is killed
/// after writing some rows, a later run appends missing frames instead of recomputing
/// frames that are already checkpointed in the CSV.
fn read_completed_frame_indices(path: impl AsRef<Path>) -> Result<HashSet<usize>, Box<dyn Error>> {
    let path = path.as_ref();
    if !path.exists() || path.metadata()?.len() == 0 {
        return Ok(HashSet::new());
    }

    let mut completed = HashSet::new();
    let mut reader = csv::Reader::from_path(path)?;
    for record in reader.records() {
        let record = match record {
            Ok(record) => record,
            Err(e) => {
                log_warn!(
                    "malformed_existing_csv_row",
                    "path={path:?} error={e} action=ignore_row"
                );
                continue;
            }
        };
        if let Some(index) = record.get(0).and_then(|value| value.parse::<usize>().ok()) {
            completed.insert(index);
        }
    }

    Ok(completed)
}

/// Open the frame-result CSV either for append/resume or for a fresh run.
fn open_frame_results_writer(
    path: impl AsRef<Path>,
    append_existing: bool,
) -> Result<csv::Writer<File>, Box<dyn Error>> {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .append(append_existing)
        .truncate(!append_existing)
        .open(path)?;

    Ok(csv::Writer::from_writer(file))
}

/// Write the frame-result CSV header.
///
/// Vectors are flattened into scalar columns so the output can be consumed directly by
/// pandas, spreadsheets, plotting tools, or HPC post-processing scripts.
fn write_frame_results_header(writer: &mut csv::Writer<File>) -> Result<(), Box<dyn Error>> {
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
    Ok(())
}

/// Append one processed frame to the result CSV.
fn write_frame_result(
    writer: &mut csv::Writer<File>,
    result: &FrameResult,
) -> Result<(), Box<dyn Error>> {
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
    Ok(())
}
