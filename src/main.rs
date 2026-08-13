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
//! [1] https://ieeexplore.ieee.org/document/11005588

use std::{
    f64::consts::{FRAC_PI_2, PI},
    io::{BufWriter, Write},
    path::PathBuf,
};

use chrono::{DateTime, Utc};
use nalgebra::{DMatrix, Matrix, Matrix3, Matrix3xX, Rotation3, Vector3};

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
const N_FRAMES: usize = 1;

fn main() {
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
    for row in 0..ROWS {
        for col in 0..COLS {
            let p_c = pixel(row, col, CENTER_ROW, CENTER_COL, PIXEL_SIZE_MM);
            let v_c = optical_path(p_c, FOCAL_LENGTH_MM);

            all_p_c.push(p_c);
            all_v_c.push(v_c);
        }
    }

    let system = System {
        all_p_c,
        all_v_c,
        aop_threshold_rad,
        n_pixels,
    };

    let time_frame = fan_method::dataset::read_time(TIME_CSV).unwrap();
    let ins_frame = fan_method::dataset::read_ins(INS_CSV).unwrap();
    for i in 0..N_FRAMES {
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

        process_frame(&frame, &system);
    }
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

#[derive(Default)]
struct FrameResult {}

/// The meat and potatoes of the algorithm.
fn process_frame(frame: &Frame, system: &System) -> FrameResult {
    let result = FrameResult::default();

    let s_n = psa(frame.lat, frame.lon, frame.time);
    let s_c = frame.rot_c_n * s_n;

    let mut aop_v = vec![0f64; system.n_pixels];
    let mut rayleigh_aop_v = vec![0f64; system.n_pixels];
    let mut rayleigh_point = vec![0u8; system.n_pixels];
    let mut e_c = Vec::new();

    for i in 0..system.n_pixels {
        let v_c = &system.all_v_c[i];
        let rayleigh_e_c = rayleigh_ev(v_c, &s_c);

        let rot_v_c = compute_rot_v_c(v_c);
        let rayleigh_e_v = rot_v_c * rayleigh_e_c;

        aop_v[i] = aop_sensor_to_v(frame.aop_s[i], system.all_p_c[i]);
        rayleigh_aop_v[i] = aop_from_ev(&rayleigh_e_v);

        let is_rayleigh_point =
            fan_method::aop_threshold(aop_v[i], rayleigh_aop_v[i], system.aop_threshold_rad);
        rayleigh_point[i] = is_rayleigh_point as u8;

        if is_rayleigh_point {
            let rot_c_v = rot_v_c.inverse();
            let e_v = ev_from_aop(aop_v[i]);
            e_c.push(rot_c_v * e_v);
        }
    }

    let a = Matrix3xX::from_columns(&e_c);
    let m = &a * a.transpose();
    let eig = m.symmetric_eigen();
    let (min_idx, _) = eig
        .eigenvalues
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN encountered in eigenvalues"))
        .expect("eigenvalues vector was empty");

    let optimal_s_c = eig.eigenvectors.column(min_idx).into_owned().normalize();

    let filename = format!("optimal_s_c_{:04}.bin", frame.index);
    let file = std::fs::File::create(filename).unwrap();
    let mut writer = BufWriter::new(file);

    for &value in optimal_s_c.iter() {
        writer.write_all(&value.to_be_bytes()).unwrap();
    }

    // Dump the raw rayleigh-point mask and AoP values for testing.
    let filename = format!("rayleigh_point_{:04}.bin", frame.index);
    let file = std::fs::File::create(filename).unwrap();
    let mut writer = BufWriter::new(file);

    writer.write_all(&rayleigh_point).unwrap();

    let filename = format!("rayleigh_aop_v_{:04}.bin", frame.index);
    let file = std::fs::File::create(filename).unwrap();
    let mut writer = BufWriter::new(file);

    // Convert each f64 into bytes and write
    for &value in &rayleigh_aop_v {
        writer.write_all(&value.to_be_bytes()).unwrap();
    }

    let filename = format!("aop_s_{:04}.bin", frame.index);
    let file = std::fs::File::create(filename).unwrap();
    let mut writer = BufWriter::new(file);

    // Convert each f64 into bytes and write
    for &value in &frame.aop_s {
        writer.write_all(&value.to_be_bytes()).unwrap();
    }

    let filename = format!("aop_v_{:04}.bin", frame.index);
    let file = std::fs::File::create(filename).unwrap();
    let mut writer = BufWriter::new(file);

    // Convert each f64 into bytes and write
    for &value in &aop_v {
        writer.write_all(&value.to_be_bytes()).unwrap();
    }

    result
}

/// Builds a rotation matrix from ZYX angles.
fn tait_bryan(yaw: f64, pitch: f64, roll: f64) -> Rotation3<f64> {
    let rot_z = Rotation3::from_axis_angle(&Vector3::z_axis(), yaw);
    let rot_y = Rotation3::from_axis_angle(&Vector3::y_axis(), pitch);
    let rot_x = Rotation3::from_axis_angle(&Vector3::x_axis(), roll);

    rot_z * rot_y * rot_x
}

fn aop_sensor_to_v(aop_s: f64, p_c: Vector3<f64>) -> f64 {
    // The v-frame x-axis projects onto the image plane in the radial direction
    // from the optical center to this pixel.  If sensor AoP is measured CCW
    // from +x_c, re-express it relative to +x_v by subtracting this radial
    // bearing.  If the camera SDK defines +y upward or reports clockwise AoP,
    // this is the one line to flip: use atan2(-p_c.y, p_c.x) and/or -aop_s.
    let radial_bearing = p_c.y.atan2(p_c.x);
    wrap_aop(aop_s - radial_bearing)
}

/// Ensure the aop falls on the interval (-pi/2, pi/2].
fn wrap_aop(aop: f64) -> f64 {
    let period = std::f64::consts::PI;
    let half_period = std::f64::consts::FRAC_PI_2;

    if !aop.is_finite() {
        return aop;
    }

    // AoP is axial, so angles separated by pi represent the same polarization
    // direction. This wraps onto (-pi/2, pi/2], mapping both endpoints to +pi/2.
    half_period - (half_period - aop).rem_euclid(period)
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

    // spa uses degrees. Its azimuth is clockwise from north. In this code's
    // ENU n-frame (x=east, y=north, z=up), the horizontal components are
    // east = sin(zenith) * sin(azimuth), north = sin(zenith) * cos(azimuth).
    enu_from_zenith_azimuth_cw_north(sp.zenith_angle.to_radians(), sp.azimuth.to_radians())
}

fn enu_from_zenith_azimuth_cw_north(zenith_angle: f64, azimuth_cw_from_north: f64) -> Vector3<f64> {
    Vector3::new(
        zenith_angle.sin() * azimuth_cw_from_north.sin(),
        zenith_angle.sin() * azimuth_cw_from_north.cos(),
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
    wrap_aop(e_v.y.atan2(e_v.x))
}

/// Returns the e-vector from the angle of polarization in the v-frame.
fn ev_from_aop(aop_v: f64) -> Vector3<f64> {
    Vector3::new(aop_v.cos(), aop_v.sin(), 0.)
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn enu_solar_azimuth_cardinal_directions_are_correct() {
        let zenith = FRAC_PI_2;
        let eps = 1e-12;

        let north = enu_from_zenith_azimuth_cw_north(zenith, 0.0);
        assert!(north.x.abs() < eps);
        assert!((north.y - 1.0).abs() < eps);

        let east = enu_from_zenith_azimuth_cw_north(zenith, FRAC_PI_2);
        assert!((east.x - 1.0).abs() < eps);
        assert!(east.y.abs() < eps);

        let south = enu_from_zenith_azimuth_cw_north(zenith, PI);
        assert!(south.x.abs() < eps);
        assert!((south.y + 1.0).abs() < eps);
    }
}
