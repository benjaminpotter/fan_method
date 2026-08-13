use chrono::{DateTime, Utc};
use nalgebra::{Matrix3, Matrix3xX, Rotation3, Vector3};

pub mod dataset;

/// Check whether two axial angle-of-polarization values are within `threshold` radians.
///
/// AoP is treated as an axial angle, so values separated by `pi` represent the same
/// polarization direction. Inputs are expected to be in radians.
pub fn aop_threshold(candidate: f64, target: f64, threshold: f64) -> bool {
    let period = std::f64::consts::PI;
    let half_period = std::f64::consts::FRAC_PI_2;

    // AoP is axial, so angles separated by pi represent the same polarization
    // direction. Wrap the signed difference onto (-pi/2, pi/2] and compare the
    // smallest equivalent separation against the threshold.
    let diff = (candidate - target + half_period).rem_euclid(period) - half_period;

    diff.abs() <= threshold
}

/// Convert a solar position from zenith/azimuth angles into an ENU unit vector.
///
/// `zenith_angle` is measured down from +up, and `azimuth_cw_from_north` is measured
/// clockwise from north. Both inputs are in radians.
pub fn enu_from_zenith_azimuth_cw_north(
    zenith_angle: f64,
    azimuth_cw_from_north: f64,
) -> Vector3<f64> {
    Vector3::new(
        zenith_angle.sin() * azimuth_cw_from_north.sin(),
        zenith_angle.sin() * azimuth_cw_from_north.cos(),
        zenith_angle.cos(),
    )
}

/// Return elapsed wall-clock time since `start`, in milliseconds.
pub fn elapsed_ms(start: std::time::Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1_000.0
}

/// Compute the azimuth angle of a vector, measured counter-clockwise from +X.
pub fn compute_azimuth(v: &Vector3<f64>) -> f64 {
    v.y.atan2(v.x)
}

/// Compute the optimal solar vector from e-vector measurements via eigendecomposition.
///
/// The returned vector is the normalized eigenvector corresponding to the smallest
/// eigenvalue of `E * E^T`, where each input e-vector is a column of `E`.
pub fn compute_s_c(e_c: &[Vector3<f64>]) -> Vector3<f64> {
    let a = Matrix3xX::from_columns(e_c);
    let m = &a * a.transpose();
    let eig = m.symmetric_eigen();
    let (min_idx, _) = eig
        .eigenvalues
        .iter()
        .enumerate()
        .min_by(|(_, a), (_, b)| a.partial_cmp(b).expect("NaN encountered in eigenvalues"))
        .expect("eigenvalues vector was empty");

    eig.eigenvectors.column(min_idx).into_owned().normalize()
}

/// Build a 3D rotation from ZYX Tait-Bryan angles.
///
/// The returned rotation is `Rz(yaw) * Ry(pitch) * Rx(roll)`. Inputs are in radians.
pub fn tait_bryan(yaw: f64, pitch: f64, roll: f64) -> Rotation3<f64> {
    let rot_z = Rotation3::from_axis_angle(&Vector3::z_axis(), yaw);
    let rot_y = Rotation3::from_axis_angle(&Vector3::y_axis(), pitch);
    let rot_x = Rotation3::from_axis_angle(&Vector3::x_axis(), roll);

    rot_z * rot_y * rot_x
}

/// Convert an AoP measured in the sensor frame into the observation-point `v` frame.
///
/// The conversion subtracts the radial bearing of the pixel in the camera frame and
/// wraps the result back onto the axial AoP interval.
pub fn aop_sensor_to_v(aop_s: f64, p_c: Vector3<f64>) -> f64 {
    let radial_bearing = p_c.y.atan2(p_c.x);
    wrap_aop(aop_s - radial_bearing)
}

/// Wrap an axial angle of polarization onto the interval `(-pi/2, pi/2]`.
pub fn wrap_aop(aop: f64) -> f64 {
    let period = std::f64::consts::PI;
    let half_period = std::f64::consts::FRAC_PI_2;

    if !aop.is_finite() {
        return aop;
    }

    half_period - (half_period - aop).rem_euclid(period)
}

/// Return the rotation from the camera `c` frame to an observation-point `v` frame.
///
/// The `v` frame's +Z axis points along `v_c`, +Y lies in the camera XY plane, and
/// +X completes a right-handed frame.
pub fn compute_rot_v_c(v_c: &Vector3<f64>) -> Rotation3<f64> {
    let z_v_c = v_c.normalize();

    let y_v_c = {
        let candidate = Vector3::new(-z_v_c.y, z_v_c.x, 0.0);

        if candidate.norm_squared() > f64::EPSILON {
            candidate.normalize()
        } else {
            Vector3::y()
        }
    };

    let x_v_c = y_v_c.cross(&z_v_c).normalize();

    Rotation3::from_matrix_unchecked(Matrix3::new(
        x_v_c.x, x_v_c.y, x_v_c.z, y_v_c.x, y_v_c.y, y_v_c.z, z_v_c.x, z_v_c.y, z_v_c.z,
    ))
}

/// Compute the direction vector to the sun from an observer using the PSA algorithm.
///
/// Latitude and longitude are degrees. The returned vector is expressed in the ENU
/// frame: +X east, +Y north, +Z up.
pub fn psa(lat: f64, lon: f64, time: DateTime<Utc>) -> Vector3<f64> {
    let sp = spa::solar_position::<spa::StdFloatOps>(time, lat, lon)
        .expect("valid lat, lon, time in PSA algorithm");

    enu_from_zenith_azimuth_cw_north(sp.zenith_angle.to_radians(), sp.azimuth.to_radians())
}

/// Compute the Rayleigh-model e-vector for an optical path and solar vector.
pub fn rayleigh_ev(v_c: &Vector3<f64>, s_c: &Vector3<f64>) -> Vector3<f64> {
    let tau = v_c.angle(s_c);
    let k = 1. / tau.sin();

    k * v_c.cross(s_c)
}

/// Compute the angle of polarization from an e-vector expressed in the `v` frame.
pub fn aop_from_ev(e_v: &Vector3<f64>) -> f64 {
    wrap_aop(e_v.y.atan2(e_v.x))
}

/// Compute a unit e-vector in the `v` frame from an angle of polarization.
pub fn ev_from_aop(aop_v: f64) -> Vector3<f64> {
    Vector3::new(aop_v.cos(), aop_v.sin(), 0.)
}

/// Return the physical location of a pixel in the camera frame.
///
/// `pixel_size` determines the physical distance represented by one row or column.
pub fn pixel(
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

/// Return the optical path terminating at a pixel for a pinhole camera model.
pub fn optical_path(pixel: Vector3<f64>, focal_length: f64) -> Vector3<f64> {
    Vector3::new(pixel.x, pixel.y, focal_length)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    fn deg(degrees: f64) -> f64 {
        degrees.to_radians()
    }

    fn assert_vec_close(actual: Vector3<f64>, expected: Vector3<f64>, eps: f64) {
        assert!(
            (actual - expected).norm() < eps,
            "actual={actual:?}, expected={expected:?}"
        );
    }

    #[test]
    fn returns_true_when_difference_is_less_than_threshold() {
        assert!(aop_threshold(deg(10.0), deg(12.0), deg(5.0)));
    }

    #[test]
    fn returns_true_when_difference_equals_threshold() {
        assert!(aop_threshold(deg(0.0), deg(5.0), deg(5.0)));
    }

    #[test]
    fn returns_false_when_difference_exceeds_threshold() {
        assert!(!aop_threshold(deg(10.0), deg(16.0), deg(5.0)));
    }

    #[test]
    fn handles_wrapping_across_aop_interval_boundary() {
        assert!(aop_threshold(deg(-89.0), deg(89.0), deg(3.0)));
    }

    #[test]
    fn handles_equivalent_angles_separated_by_pi() {
        assert!(aop_threshold(-FRAC_PI_2, FRAC_PI_2, deg(0.0),));
    }

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

    #[test]
    fn elapsed_ms_returns_nonnegative_milliseconds() {
        let elapsed = elapsed_ms(std::time::Instant::now());
        assert!(elapsed >= 0.0);
    }

    #[test]
    fn compute_azimuth_returns_angle_ccw_from_x() {
        assert!((compute_azimuth(&Vector3::x()) - 0.0).abs() < 1e-12);
        assert!((compute_azimuth(&Vector3::y()) - FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn compute_s_c_returns_normal_to_e_vector_plane() {
        let e_c = [Vector3::x(), Vector3::y()];
        let s_c = compute_s_c(&e_c);
        assert!(s_c.z.abs() > 1.0 - 1e-12);
    }

    #[test]
    fn tait_bryan_yaw_rotates_x_toward_y() {
        let rot = tait_bryan(FRAC_PI_2, 0.0, 0.0);
        assert_vec_close(rot * Vector3::x(), Vector3::y(), 1e-12);
    }

    #[test]
    fn aop_sensor_to_v_subtracts_pixel_radial_bearing() {
        let aop_v = aop_sensor_to_v(FRAC_PI_2, Vector3::new(0.0, 1.0, 0.0));
        assert!(aop_v.abs() < 1e-12);
    }

    #[test]
    fn wrap_aop_wraps_to_expected_interval() {
        assert!((wrap_aop(PI) - 0.0).abs() < 1e-12);
        assert!((wrap_aop(-FRAC_PI_2) - FRAC_PI_2).abs() < 1e-12);
        assert!((wrap_aop(3.0 * FRAC_PI_2) - FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn compute_rot_v_c_maps_observation_direction_to_v_frame_z() {
        let v_c = Vector3::new(1.0, 2.0, 8.0).normalize();
        let rot_v_c = compute_rot_v_c(&v_c);
        assert_vec_close(rot_v_c * v_c, Vector3::z(), 1e-12);
    }

    #[test]
    fn psa_returns_unit_vector() {
        let time = DateTime::parse_from_rfc3339("2025-06-01T12:00:00Z")
            .unwrap()
            .with_timezone(&Utc);
        let s_n = psa(40.0, -105.0, time);
        assert!((s_n.norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn rayleigh_ev_is_perpendicular_to_optical_path_and_sun_vector() {
        let v_c = Vector3::x();
        let s_c = Vector3::y();
        let e_c = rayleigh_ev(&v_c, &s_c);
        assert!(e_c.dot(&v_c).abs() < 1e-12);
        assert!(e_c.dot(&s_c).abs() < 1e-12);
        assert_vec_close(e_c, Vector3::z(), 1e-12);
    }

    #[test]
    fn aop_from_ev_returns_wrapped_vector_angle() {
        assert!((aop_from_ev(&Vector3::y()) - FRAC_PI_2).abs() < 1e-12);
        assert!((aop_from_ev(&-Vector3::y()) - FRAC_PI_2).abs() < 1e-12);
    }

    #[test]
    fn ev_from_aop_returns_unit_vector_in_xy_plane() {
        let e_v = ev_from_aop(FRAC_PI_2);
        assert_vec_close(e_v, Vector3::y(), 1e-12);
        assert!((e_v.norm() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn pixel_returns_centered_camera_frame_position() {
        assert_vec_close(pixel(6, 7, 5, 5, 0.5), Vector3::new(1.0, 0.5, 0.0), 1e-12);
    }

    #[test]
    fn optical_path_preserves_pixel_xy_and_sets_focal_length_z() {
        assert_vec_close(
            optical_path(Vector3::new(1.0, 2.0, 0.0), 8.0),
            Vector3::new(1.0, 2.0, 8.0),
            1e-12,
        );
    }
}
