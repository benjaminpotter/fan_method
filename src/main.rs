//! Test the Fan method from [1]
//!
//! Coordinate Systems:
//! - a-frame: car body (XYZ)
//! - b-frame: camera body (XYZ)
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
//!   - Orientation of a-frame in n-frame
//!   - Stored in CSV file
//! 2. GPS Time (Novatel Oem7 Time)
//!   - Datetime
//!   - Stored in CSV file
//! 3. Polarization Camera (Lucid Vision PHX050S-P/Q)
//!   - Measurements of the skylight polarization pattern
//!   - Stored as png images named with the frame they correspond to
//!
//! [1] https://ieeexplore.ieee.org/document/11005588

use std::io::{BufWriter, Write};

use chrono::{DateTime, Utc};
use nalgebra::{Matrix3, Rotation3, Vector3};

const ROWS: usize = 1024;
const COLS: usize = 1224;
const N_PIXELS: usize = ROWS * COLS;
const CENTER_ROW: usize = ROWS / 2;
const CENTER_COL: usize = COLS / 2;
const PIXEL_SIZE_MM: f64 = 0.0069;
const FOCAL_LENGTH_MM: f64 = 8.0;
const AOP_THRESHOLD_DEG: f64 = 5.0;

fn main() {
    println!("Fan Method v0.1");
    println!("Implemented by Ben Potter in August 2026");
    println!("See original paper: https://ieeexplore.ieee.org/document/11005588");

    // Compute the optical paths once at startup.
    let mut all_p_c = Vec::new();
    let mut all_v_c = Vec::new();

    for row in 0..ROWS {
        for col in 0..COLS {
            let p_c = pixel(row, col, CENTER_ROW, CENTER_COL, PIXEL_SIZE_MM);
            let v_c = optical_path(p_c, FOCAL_LENGTH_MM);

            all_p_c.push(p_c);
            all_v_c.push(v_c);
        }
    }

    let rot_b_n = Rotation3::<f64>::default();
    let n_pixels = N_PIXELS;
    let aop_threshold_rad = AOP_THRESHOLD_DEG.to_radians();

    let system = System {
        rot_b_n,
        all_p_c,
        all_v_c,
        aop_threshold_rad,
        n_pixels,
    };

    let index = 0;
    let rot_c_b = Rotation3::<f64>::default();
    let time = Utc::now();
    let lat = -45.;
    let lon = 45.;
    let aop = vec![0.; N_PIXELS];

    let frame = Frame {
        index,
        rot_c_b,
        time,
        lat,
        lon,
        aop,
    };

    process_frame(&frame, &system);
}

/// Stores information which remains static across frames.
struct System {
    /// Rotation between the n-frame and the b-frame.
    rot_b_n: Rotation3<f64>,
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
    rot_c_b: Rotation3<f64>,
    time: DateTime<Utc>,
    lat: f64,
    lon: f64,
    aop: Vec<f64>,
}

#[derive(Default)]
struct FrameResult {}

/// The meat and potatoes of the algorithm.
fn process_frame(frame: &Frame, system: &System) -> FrameResult {
    let mut result = FrameResult::default();

    let s_n = psa(frame.lat, frame.lon, frame.time);
    let s_c = frame.rot_c_b * system.rot_b_n * s_n;

    let mut rayleigh_aop = vec![0f64; system.n_pixels];
    let mut rayleigh_point = vec![0u8; system.n_pixels];

    for i in 0..system.n_pixels {
        let v_c = &system.all_v_c[i];
        let e_c = rayleigh_ev(v_c, &s_c);

        let rot_v_c = compute_rot_v_c(v_c);
        let e_v = rot_v_c * e_c;
        rayleigh_aop[i] = aop_from_ev(&e_v);

        rayleigh_point[i] =
            fan_method::aop_threshold(frame.aop[i], rayleigh_aop[i], system.aop_threshold_rad)
                as u8;
    }

    // Dump the raw rayleigh points for testing.
    std::fs::write(
        format!("rayleigh_point_{:04}.bin", frame.index),
        rayleigh_point,
    )
    .unwrap();

    // Dump the raw rayleigh aop for testing.
    let filename = format!("rayleigh_aop_{:04}.bin", frame.index);
    let file = std::fs::File::create(filename).unwrap();
    let mut writer = BufWriter::new(file);

    // Convert each f64 into bytes and write
    for &value in &rayleigh_aop {
        writer.write_all(&value.to_be_bytes()).unwrap();
    }

    result
}

/// Returns the rotation from the c-frame to the v-frame for an observation direction.
fn compute_rot_v_c(v_c: &Vector3<f64>) -> Rotation3<f64> {
    let z_v_c = v_c.normalize();

    // The y_v axis lies in the x_c-y_c plane and is orthogonal to z_v.
    let y_v_c = {
        let candidate = Vector3::new(-z_v_c.y, z_v_c.x, 0.0);

        if candidate.norm_squared() > f64::EPSILON {
            candidate.normalize()
        } else {
            Vector3::y()
        }
    };

    // Complete a right-handed frame: x_v × y_v = z_v.
    let x_v_c = y_v_c.cross(&z_v_c).normalize();

    Rotation3::from_matrix_unchecked(Matrix3::new(
        x_v_c.x, x_v_c.y, x_v_c.z, y_v_c.x, y_v_c.y, y_v_c.z, z_v_c.x, z_v_c.y, z_v_c.z,
    ))
}

/// Returns the direction vector to the sun from the observer based on the PSA algorithm.
fn psa(lat: f64, lon: f64, time: DateTime<Utc>) -> Vector3<f64> {
    let sp = spa::solar_position::<spa::StdFloatOps>(time, lat, lon)
        .expect("valid lat, lon, time in PSA algorithm");

    // spa uses degrees so we have to convert to radians.
    let zenith_angle = sp.zenith_angle.to_radians();
    let azimuth = sp.azimuth.to_radians();

    Vector3::new(
        zenith_angle.sin() * azimuth.sin(),
        zenith_angle.sin() * azimuth.cos(),
        zenith_angle.cos(),
    )
}

/// Returns the e-vector for the optical path and solar vector.
fn rayleigh_ev(v_c: &Vector3<f64>, s_c: &Vector3<f64>) -> Vector3<f64> {
    // The scattering angle, tau, is the smallest angle between v_c and s_c.
    let tau = v_c.angle(&s_c);
    let k = 1. / tau.sin();

    k * v_c.cross(s_c)
}

/// Returns the angle of polarization from the e-vector in the v-frame.
fn aop_from_ev(e_v: &Vector3<f64>) -> f64 {
    (e_v.y / e_v.x).atan()
}

/// Apply the Lagrange multiplier method to solve for the solar vector, s_c (3x1), given the optical paths, V_c (3xN),
/// the e-vectors, E_c (3xN), and the scalar factors, K (1xN).
///
/// N is the number of optical paths, where an optical path is a vector, v_c (3x1).
/// The scalar factors are computed from the reciprocal of the sin of the scattering angle.
/// The scattering angle is the angle between the solar vector, s_c, and the optical path, v_c.
/// The relationship between the solar vector, s_c, any single optical path, v_c, the corresponding
/// e-vector, e_c, and the scalar factor, k, is e_c = k * v_c cross s_c.
fn lagrange() -> Vector3<f64> {
    todo!()
}

/// Returns the physical location of a pixel in the c-frame from its row and column.
fn pixel(
    row: usize,
    col: usize,
    center_row: usize,
    center_col: usize,
    pixel_size: f64,
) -> Vector3<f64> {
    let x = (col as f64 - center_col as f64) * pixel_size;
    let y = (row as f64 - center_row as f64) * pixel_size;

    Vector3::new(x, y, 0.0)
}

/// Returns the optical path terminating at a pixel in the c-frame based on the focal length of the optical system.
fn optical_path(pixel: Vector3<f64>, focal_length: f64) -> Vector3<f64> {
    // This is NOT the implementation directly from the paper.
    // There may be some additional work to correct any mistaken reference frames.

    Vector3::new(pixel.x, pixel.y, focal_length)
}
