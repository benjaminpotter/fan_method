use nalgebra::Vector3;

pub mod dataset;

/// Check whether the candidate is within threshold of target.
///
/// Ensure wrapping is handled.
/// Assume AoP on the interval (-pi/2, pi/2].
pub fn aop_threshold(candidate: f64, target: f64, threshold: f64) -> bool {
    let period = std::f64::consts::PI;
    let half_period = std::f64::consts::FRAC_PI_2;

    // AoP is axial, so angles separated by pi represent the same polarization
    // direction. Wrap the signed difference onto (-pi/2, pi/2] and compare the
    // smallest equivalent separation against the threshold.
    let diff = (candidate - target + half_period).rem_euclid(period) - half_period;

    diff.abs() <= threshold
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    fn deg(degrees: f64) -> f64 {
        degrees.to_radians()
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
        assert!(aop_threshold(
            -std::f64::consts::FRAC_PI_2,
            std::f64::consts::FRAC_PI_2,
            deg(0.0),
        ));
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
}
