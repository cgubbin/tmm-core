//! Validation and canonicalisation of caller-facing in-plane coordinates.
//!
//! The numerical backend represents every in-plane coordinate as a parallel
//! angular wavenumber in inverse centimetres.
//!
//! This module converts each supported caller-facing parameterisation into that
//! canonical representation after first validating the supplied values.
//!
//! Validation checks only properties of the caller input (for example,
//! finiteness and the principal interval for incident angles). Coordinate
//! transformations are performed separately so that validation and
//! canonicalisation remain independent concerns.
//!
//! Canonicalisation operates on seeded jet values rather than raw scalars.
//! Consequently, derivatives are taken with respect to the caller-facing
//! coordinate while the jet algebra automatically propagates the required chain
//! rule through the coordinate transformation.
//!
//! The supported transformations are:
//!
//! - parallel angular wavenumber → identity (with unit conversion);
//! - parallel wavenumber → parallel angular wavenumber;
//! - effective index → parallel angular wavenumber;
//! - incident angle → parallel angular wavenumber.

use nalgebra::ComplexField;
use ndarray::{ArrayBase, Data, Dimension};
use num_traits::{Float, FloatConst, FromPrimitive};
use thiserror::Error;

use crate::input::{InPlaneCoordinate, compile::coordinates::CanonicalCoordinateJet};

#[derive(Clone, Debug, PartialEq, Error)]
pub enum InPlaneInputError<R> {
    #[error(
        "in-plane coordinate contains a non-finite value \
         at flat index {index}: {value}"
    )]
    NonFinite { index: usize, value: R },

    #[error(
        "incident angle at flat index {index} is outside \
         the supported interval [-π/2, π/2]: {radians} rad"
    )]
    AngleOutsidePrincipalInterval { index: usize, radians: R },
}

/// Validate a caller-facing in-plane coordinate.
///
/// Every supplied value must be finite. Incident-angle coordinates are
/// additionally required to lie within the principal interval
/// `[-π/2, π/2]` after conversion to radians.
///
/// No unit conversion or canonicalisation is performed.
///
/// # Errors
///
/// Returns [`InPlaneInputError`] if any value is non-finite or, for incident
/// angles, lies outside the supported interval.
pub(crate) fn validate_in_plane<C, S, D>(
    values: &ArrayBase<S, D>,
    coordinate: InPlaneCoordinate,
) -> Result<(), InPlaneInputError<C>>
where
    C: ComplexField + Copy,
    C::RealField: Float + FloatConst + FromPrimitive,
    S: Data<Elem = C>,
    D: Dimension,
{
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(InPlaneInputError::NonFinite { index, value });
        }

        if let InPlaneCoordinate::IncidentAngle(unit) = coordinate {
            let radians = value * C::from_real(unit.scale_to_radians::<C::RealField>());

            let half_pi = C::from_real(<C::RealField as FloatConst>::PI()) / (C::one() + C::one());

            if radians.real() < -half_pi.real() || radians.real() > half_pi.real() {
                return Err(InPlaneInputError::AngleOutsidePrincipalInterval { index, radians });
            }
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum InPlaneCanonicalisationError {
    #[error(
        "incident-angle input requires the refractive \
         index of the incident medium"
    )]
    MissingIncidentIndex,
}

/// Convert a seeded caller-facing in-plane coordinate into the canonical
/// parallel angular wavenumber.
///
/// The returned quantity is always expressed as a parallel angular
/// wavenumber in inverse centimetres, regardless of the caller-facing
/// parameterisation.
///
/// Coordinate transformations are applied through the supplied jet algebra so
/// that derivatives with respect to the original caller-facing coordinate are
/// propagated automatically.
///
/// The transformations are:
///
/// ```text
/// β            → β
///
/// k∥           → 2π k∥
///
/// n_eff        → n_eff k₀
///
/// θ            → nᵢ k₀ sin θ
/// ```
///
/// where
///
/// - `β` is the parallel angular wavenumber;
/// - `k∥` is the parallel wavenumber;
/// - `k₀` is the vacuum angular wavenumber;
/// - `n_eff` is the effective index;
/// - `nᵢ` is the refractive index of the incident medium.
///
/// # Errors
///
/// Incident-angle coordinates require the refractive index of the incident
/// medium. If it is not supplied,
/// [`InPlaneCanonicalisationError::MissingIncidentIndex`] is returned.
pub(crate) fn canonicalise_in_plane<J>(
    value: J,
    coordinate: InPlaneCoordinate,
    vacuum_angular_wavenumber: &J,
    incident_index: Option<&J>,
) -> Result<J, InPlaneCanonicalisationError>
where
    J: CanonicalCoordinateJet,
    J::Scalar: ComplexField,
    <J::Scalar as ComplexField>::RealField: Float + FloatConst + FromPrimitive + Copy,
    J::Dimension: Dimension,
{
    let two_pi = <<J::Scalar as ComplexField>::RealField as FloatConst>::PI()
        + <<J::Scalar as ComplexField>::RealField as FloatConst>::PI();

    match coordinate {
        InPlaneCoordinate::ParallelAngularWavenumber(unit) => Ok(value.scale_real(
            unit.scale_to_inverse_centimetres::<<J::Scalar as ComplexField>::RealField>(),
        )),

        InPlaneCoordinate::ParallelWavenumber(unit) => Ok(value.scale_real(
            unit.scale_to_inverse_centimetres::<<J::Scalar as ComplexField>::RealField>() * two_pi,
        )),

        InPlaneCoordinate::EffectiveIndex => Ok(value.multiply(vacuum_angular_wavenumber.clone())),

        InPlaneCoordinate::IncidentAngle(unit) => {
            let incident_index =
                incident_index.ok_or(InPlaneCanonicalisationError::MissingIncidentIndex)?;

            let sine = value.scale_real(unit.scale_to_radians()).sin();

            Ok(incident_index
                .clone()
                .multiply(vacuum_angular_wavenumber.clone())
                .multiply(sine))
        }
    }
}

#[cfg(test)]
mod tests {
    use std::f64::consts::{FRAC_PI_2, PI};

    use ndarray::{Ix0, arr1, array};
    use num_complex::Complex64;
    use tmm_units::{AngleUnit, InverseLengthUnit};

    use super::*;

    const TOLERANCE: f64 = 1.0e-12;

    fn assert_close(actual: f64, expected: f64) {
        let scale = expected.abs().max(1.0);

        assert!(
            (actual - expected).abs() <= TOLERANCE * scale,
            "expected {expected:.16e}, got {actual:.16e}; \
             absolute error was {:.16e}",
            (actual - expected).abs(),
        );
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64) {
        let scale = expected.norm().max(1.0);

        assert!(
            (actual - expected).norm() <= TOLERANCE * scale,
            "expected {expected:?}, got {actual:?}; \
             absolute error was {:.16e}",
            (actual - expected).norm(),
        );
    }

    /// Minimal first-order jet used to test coordinate transformations.
    ///
    /// `derivative` is the derivative with respect to one caller-facing
    /// coordinate variable.
    #[derive(Clone, Debug, PartialEq)]
    struct RecordingJet {
        value: Complex64,
        derivative: Complex64,
    }

    impl RecordingJet {
        fn constant(value: f64) -> Self {
            Self {
                value: Complex64::new(value, 0.0),
                derivative: Complex64::new(0.0, 0.0),
            }
        }

        fn variable(value: f64) -> Self {
            Self {
                value: Complex64::new(value, 0.0),
                derivative: Complex64::new(1.0, 0.0),
            }
        }
    }

    impl crate::algebra::Jet for RecordingJet {
        type Scalar = Complex64;
        type Dimension = Ix0;
        type PointJet = Self;
    }

    impl CanonicalCoordinateJet for RecordingJet {
        fn scale_real(self, factor: f64) -> Self {
            Self {
                value: self.value * factor,
                derivative: self.derivative * factor,
            }
        }

        fn reciprocal(self) -> Self {
            Self {
                value: self.value.recip(),
                derivative: -self.derivative / self.value.powu(2),
            }
        }

        fn sin(self) -> Self {
            Self {
                value: self.value.sin(),
                derivative: self.value.cos() * self.derivative,
            }
        }

        fn multiply(self, rhs: Self) -> Self {
            Self {
                value: self.value * rhs.value,
                derivative: self.derivative * rhs.value + self.value * rhs.derivative,
            }
        }
    }

    fn canonicalise(
        value: RecordingJet,
        coordinate: InPlaneCoordinate,
        vacuum_angular_wavenumber: &RecordingJet,
        incident_index: Option<&RecordingJet>,
    ) -> Result<RecordingJet, InPlaneCanonicalisationError> {
        canonicalise_in_plane::<RecordingJet>(
            value,
            coordinate,
            vacuum_angular_wavenumber,
            incident_index,
        )
    }

    mod validation {
        use super::*;

        #[test]
        fn accepts_finite_parallel_angular_wavenumbers() {
            let values = arr1(&[-100.0, -1.0, 0.0, 1.0, 100.0]);

            let result = validate_in_plane(
                &values,
                InPlaneCoordinate::ParallelAngularWavenumber(InverseLengthUnit::PerCentimetre),
            );

            assert_eq!(result, Ok(()));
        }

        #[test]
        fn accepts_finite_parallel_wavenumbers() {
            let values = arr1(&[-100.0, -1.0, 0.0, 1.0, 100.0]);

            let result = validate_in_plane(
                &values,
                InPlaneCoordinate::ParallelWavenumber(InverseLengthUnit::PerCentimetre),
            );

            assert_eq!(result, Ok(()));
        }

        #[test]
        fn accepts_finite_effective_indices() {
            let values = arr1(&[-2.0, -1.0, 0.0, 1.0, 2.0]);

            let result = validate_in_plane(&values, InPlaneCoordinate::EffectiveIndex);

            assert_eq!(result, Ok(()));
        }

        #[test]
        fn rejects_nan() {
            let values = arr1(&[1.0, f64::NAN, 2.0]);

            let error = validate_in_plane(&values, InPlaneCoordinate::EffectiveIndex).unwrap_err();

            assert!(matches!(
                error,
                InPlaneInputError::NonFinite {
                    index: 1,
                    value,
                } if value.is_nan()
            ));
        }

        #[test]
        fn rejects_positive_infinity() {
            let values = arr1(&[1.0, f64::INFINITY]);

            let error = validate_in_plane(&values, InPlaneCoordinate::EffectiveIndex).unwrap_err();

            assert_eq!(
                error,
                InPlaneInputError::NonFinite {
                    index: 1,
                    value: f64::INFINITY,
                },
            );
        }

        #[test]
        fn rejects_negative_infinity() {
            let values = arr1(&[f64::NEG_INFINITY, 1.0]);

            let error = validate_in_plane(&values, InPlaneCoordinate::EffectiveIndex).unwrap_err();

            assert_eq!(
                error,
                InPlaneInputError::NonFinite {
                    index: 0,
                    value: f64::NEG_INFINITY,
                },
            );
        }

        #[test]
        fn reports_first_non_finite_value() {
            let values = arr1(&[1.0, f64::NAN, f64::INFINITY]);

            let error = validate_in_plane(&values, InPlaneCoordinate::EffectiveIndex).unwrap_err();

            assert!(matches!(
                error,
                InPlaneInputError::NonFinite {
                    index: 1,
                    value,
                } if value.is_nan()
            ));
        }

        #[test]
        fn accepts_zero_radians() {
            let values = arr1(&[0.0]);

            let result =
                validate_in_plane(&values, InPlaneCoordinate::IncidentAngle(AngleUnit::Radian));

            assert_eq!(result, Ok(()));
        }

        #[test]
        fn accepts_positive_pi_over_two() {
            let values = arr1(&[FRAC_PI_2]);

            let result =
                validate_in_plane(&values, InPlaneCoordinate::IncidentAngle(AngleUnit::Radian));

            assert_eq!(result, Ok(()));
        }

        #[test]
        fn accepts_negative_pi_over_two() {
            let values = arr1(&[-FRAC_PI_2]);

            let result =
                validate_in_plane(&values, InPlaneCoordinate::IncidentAngle(AngleUnit::Radian));

            assert_eq!(result, Ok(()));
        }

        #[test]
        fn rejects_radians_above_principal_interval() {
            let radians = FRAC_PI_2 + 1.0e-6;
            let values = arr1(&[radians]);

            let error =
                validate_in_plane(&values, InPlaneCoordinate::IncidentAngle(AngleUnit::Radian))
                    .unwrap_err();

            assert_eq!(
                error,
                InPlaneInputError::AngleOutsidePrincipalInterval { index: 0, radians },
            );
        }

        #[test]
        fn rejects_radians_below_principal_interval() {
            let radians = -FRAC_PI_2 - 1.0e-6;
            let values = arr1(&[radians]);

            let error =
                validate_in_plane(&values, InPlaneCoordinate::IncidentAngle(AngleUnit::Radian))
                    .unwrap_err();

            assert_eq!(
                error,
                InPlaneInputError::AngleOutsidePrincipalInterval { index: 0, radians },
            );
        }

        #[test]
        fn accepts_positive_ninety_degrees() {
            let values = arr1(&[90.0]);

            let result =
                validate_in_plane(&values, InPlaneCoordinate::IncidentAngle(AngleUnit::Degree));

            assert_eq!(result, Ok(()));
        }

        #[test]
        fn accepts_negative_ninety_degrees() {
            let values = arr1(&[-90.0]);

            let result =
                validate_in_plane(&values, InPlaneCoordinate::IncidentAngle(AngleUnit::Degree));

            assert_eq!(result, Ok(()));
        }

        #[test]
        fn rejects_angles_above_ninety_degrees() {
            let values = arr1(&[91.0]);

            let error =
                validate_in_plane(&values, InPlaneCoordinate::IncidentAngle(AngleUnit::Degree))
                    .unwrap_err();

            let expected_radians = 91.0_f64.to_radians();

            match error {
                InPlaneInputError::AngleOutsidePrincipalInterval { index, radians } => {
                    assert_eq!(index, 0);
                    assert_close(radians, expected_radians);
                }

                other => {
                    panic!(
                        "expected AngleOutsidePrincipalInterval, \
                         got {other:?}"
                    );
                }
            }
        }

        #[test]
        fn rejects_angles_below_negative_ninety_degrees() {
            let values = arr1(&[-91.0]);

            let error =
                validate_in_plane(&values, InPlaneCoordinate::IncidentAngle(AngleUnit::Degree))
                    .unwrap_err();

            let expected_radians = (-91.0_f64).to_radians();

            match error {
                InPlaneInputError::AngleOutsidePrincipalInterval { index, radians } => {
                    assert_eq!(index, 0);
                    assert_close(radians, expected_radians);
                }

                other => {
                    panic!(
                        "expected AngleOutsidePrincipalInterval, \
                         got {other:?}"
                    );
                }
            }
        }

        #[test]
        fn validates_each_angle_after_unit_conversion() {
            let values = arr1(&[0.0, 45.0, 91.0, 30.0]);

            let error =
                validate_in_plane(&values, InPlaneCoordinate::IncidentAngle(AngleUnit::Degree))
                    .unwrap_err();

            match error {
                InPlaneInputError::AngleOutsidePrincipalInterval { index, radians } => {
                    assert_eq!(index, 2);
                    assert_close(radians, 91.0_f64.to_radians());
                }

                other => {
                    panic!(
                        "expected AngleOutsidePrincipalInterval, \
                         got {other:?}"
                    );
                }
            }
        }

        #[test]
        fn checks_finiteness_before_angle_range() {
            let values = arr1(&[f64::NAN]);

            let error =
                validate_in_plane(&values, InPlaneCoordinate::IncidentAngle(AngleUnit::Radian))
                    .unwrap_err();

            assert!(matches!(
                error,
                InPlaneInputError::NonFinite {
                    index: 0,
                    value,
                } if value.is_nan()
            ));
        }

        #[test]
        fn in_plane_accepts_complex_intrinsic_coordinates() {
            let values = array![Complex64::new(1.0, 2.0), Complex64::new(-3.0, 4.0),];

            let result = validate_in_plane(
                &values,
                InPlaneCoordinate::ParallelAngularWavenumber(InverseLengthUnit::PerMetre),
            );

            assert!(result.is_ok());
        }

        #[test]
        fn in_plane_rejects_non_finite_imaginary_part() {
            let values = array![Complex64::new(1.0, 0.0), Complex64::new(2.0, f64::INFINITY),];

            let error = validate_in_plane(
                &values,
                InPlaneCoordinate::ParallelAngularWavenumber(InverseLengthUnit::PerMetre),
            )
            .unwrap_err();

            assert!(matches!(
                error,
                InPlaneInputError::NonFinite {
                    index: 1,
                    value
                } if value == Complex64::new(2.0, f64::INFINITY)
            ));
        }

        #[test]
        fn in_plane_accepts_complex_angle_inside_principal_interval() {
            let values = array![Complex64::new(0.25, 20.0)];

            let result =
                validate_in_plane(&values, InPlaneCoordinate::IncidentAngle(AngleUnit::Radian));

            assert!(result.is_ok());
        }

        #[test]
        fn in_plane_accepts_large_imaginary_angle_component() {
            let values = array![Complex64::new(0.25, 1.0e6)];

            let result =
                validate_in_plane(&values, InPlaneCoordinate::IncidentAngle(AngleUnit::Radian));

            assert!(result.is_ok());
        }

        #[test]
        fn in_plane_rejects_complex_angle_above_principal_interval() {
            let values = array![Complex64::new(std::f64::consts::FRAC_PI_2 + 1.0e-6, 4.0,)];

            let error =
                validate_in_plane(&values, InPlaneCoordinate::IncidentAngle(AngleUnit::Radian))
                    .unwrap_err();

            assert!(matches!(
                error,
                InPlaneInputError::AngleOutsidePrincipalInterval {
                    index: 0,
                    radians
                } if radians == values[0]
            ));
        }

        #[test]
        fn in_plane_rejects_complex_angle_below_principal_interval() {
            let values = array![Complex64::new(-std::f64::consts::FRAC_PI_2 - 1.0e-6, -3.0,)];

            let error =
                validate_in_plane(&values, InPlaneCoordinate::IncidentAngle(AngleUnit::Radian))
                    .unwrap_err();

            assert!(matches!(
                error,
                InPlaneInputError::AngleOutsidePrincipalInterval {
                    index: 0,
                    radians
                } if radians == values[0]
            ));
        }

        #[test]
        fn in_plane_accepts_complex_angle_on_principal_interval_boundary() {
            let values = array![
                Complex64::new(-std::f64::consts::FRAC_PI_2, 2.0),
                Complex64::new(std::f64::consts::FRAC_PI_2, -2.0),
            ];

            let result =
                validate_in_plane(&values, InPlaneCoordinate::IncidentAngle(AngleUnit::Radian));

            assert!(result.is_ok());
        }
    }

    mod canonicalisation {
        use super::*;

        #[test]
        fn parallel_angular_wavenumber_in_inverse_centimetres_is_unchanged() {
            let value = RecordingJet::constant(12.5);
            let vacuum_angular_wavenumber = RecordingJet::constant(20.0);

            let result = canonicalise(
                value,
                InPlaneCoordinate::ParallelAngularWavenumber(InverseLengthUnit::PerCentimetre),
                &vacuum_angular_wavenumber,
                None,
            )
            .unwrap();

            assert_complex_close(result.value, Complex64::new(12.5, 0.0));
        }

        #[test]
        fn parallel_angular_wavenumber_converts_inverse_micrometres() {
            let value = RecordingJet::constant(1.0);
            let vacuum_angular_wavenumber = RecordingJet::constant(20.0);

            let result = canonicalise(
                value,
                InPlaneCoordinate::ParallelAngularWavenumber(InverseLengthUnit::PerMicrometre),
                &vacuum_angular_wavenumber,
                None,
            )
            .unwrap();

            // 1 µm⁻¹ = 10⁴ cm⁻¹.
            assert_complex_close(result.value, Complex64::new(1.0e4, 0.0));
        }

        #[test]
        fn parallel_wavenumber_multiplies_by_two_pi() {
            let value = RecordingJet::constant(3.0);
            let vacuum_angular_wavenumber = RecordingJet::constant(20.0);

            let result = canonicalise(
                value,
                InPlaneCoordinate::ParallelWavenumber(InverseLengthUnit::PerCentimetre),
                &vacuum_angular_wavenumber,
                None,
            )
            .unwrap();

            assert_complex_close(result.value, Complex64::new(6.0 * PI, 0.0));
        }

        #[test]
        fn parallel_wavenumber_converts_units_and_multiplies_by_two_pi() {
            let value = RecordingJet::constant(1.0);
            let vacuum_angular_wavenumber = RecordingJet::constant(20.0);

            let result = canonicalise(
                value,
                InPlaneCoordinate::ParallelWavenumber(InverseLengthUnit::PerMicrometre),
                &vacuum_angular_wavenumber,
                None,
            )
            .unwrap();

            assert_complex_close(result.value, Complex64::new(2.0 * PI * 1.0e4, 0.0));
        }

        #[test]
        fn effective_index_multiplies_by_vacuum_angular_wavenumber() {
            let effective_index = RecordingJet::constant(1.5);

            let vacuum_angular_wavenumber = RecordingJet::constant(10.0);

            let result = canonicalise(
                effective_index,
                InPlaneCoordinate::EffectiveIndex,
                &vacuum_angular_wavenumber,
                None,
            )
            .unwrap();

            assert_complex_close(result.value, Complex64::new(15.0, 0.0));
        }

        #[test]
        fn incident_angle_in_radians_uses_sine_relationship() {
            let angle = RecordingJet::constant(PI / 6.0);

            let vacuum_angular_wavenumber = RecordingJet::constant(10.0);

            let incident_index = RecordingJet::constant(2.0);

            let result = canonicalise(
                angle,
                InPlaneCoordinate::IncidentAngle(AngleUnit::Radian),
                &vacuum_angular_wavenumber,
                Some(&incident_index),
            )
            .unwrap();

            assert_complex_close(result.value, Complex64::new(10.0, 0.0));
        }

        #[test]
        fn incident_angle_in_degrees_uses_sine_relationship() {
            let angle = RecordingJet::constant(30.0);

            let vacuum_angular_wavenumber = RecordingJet::constant(10.0);

            let incident_index = RecordingJet::constant(2.0);

            let result = canonicalise(
                angle,
                InPlaneCoordinate::IncidentAngle(AngleUnit::Degree),
                &vacuum_angular_wavenumber,
                Some(&incident_index),
            )
            .unwrap();

            assert_complex_close(result.value, Complex64::new(10.0, 0.0));
        }

        #[test]
        fn zero_incident_angle_produces_zero_parallel_wavenumber() {
            let angle = RecordingJet::constant(0.0);

            let vacuum_angular_wavenumber = RecordingJet::constant(15.0);

            let incident_index = RecordingJet::constant(1.75);

            let result = canonicalise(
                angle,
                InPlaneCoordinate::IncidentAngle(AngleUnit::Radian),
                &vacuum_angular_wavenumber,
                Some(&incident_index),
            )
            .unwrap();

            assert_complex_close(result.value, Complex64::new(0.0, 0.0));
        }

        #[test]
        fn ninety_degree_incident_angle_equals_n_times_k0() {
            let angle = RecordingJet::constant(90.0);

            let vacuum_angular_wavenumber = RecordingJet::constant(12.0);

            let incident_index = RecordingJet::constant(1.5);

            let result = canonicalise(
                angle,
                InPlaneCoordinate::IncidentAngle(AngleUnit::Degree),
                &vacuum_angular_wavenumber,
                Some(&incident_index),
            )
            .unwrap();

            assert_complex_close(result.value, Complex64::new(18.0, 0.0));
        }

        #[test]
        fn negative_incident_angle_produces_negative_parallel_wavenumber() {
            let angle = RecordingJet::constant(-30.0);

            let vacuum_angular_wavenumber = RecordingJet::constant(10.0);

            let incident_index = RecordingJet::constant(2.0);

            let result = canonicalise(
                angle,
                InPlaneCoordinate::IncidentAngle(AngleUnit::Degree),
                &vacuum_angular_wavenumber,
                Some(&incident_index),
            )
            .unwrap();

            assert_complex_close(result.value, Complex64::new(-10.0, 0.0));
        }

        #[test]
        fn incident_angle_requires_incident_index() {
            let angle = RecordingJet::constant(30.0);

            let vacuum_angular_wavenumber = RecordingJet::constant(10.0);

            let error = canonicalise(
                angle,
                InPlaneCoordinate::IncidentAngle(AngleUnit::Degree),
                &vacuum_angular_wavenumber,
                None,
            )
            .unwrap_err();

            assert_eq!(error, InPlaneCanonicalisationError::MissingIncidentIndex,);
        }

        #[test]
        fn non_angle_coordinates_do_not_require_incident_index() {
            let vacuum_angular_wavenumber = RecordingJet::constant(10.0);

            let cases = [
                (
                    RecordingJet::constant(2.0),
                    InPlaneCoordinate::ParallelAngularWavenumber(InverseLengthUnit::PerCentimetre),
                ),
                (
                    RecordingJet::constant(2.0),
                    InPlaneCoordinate::ParallelWavenumber(InverseLengthUnit::PerCentimetre),
                ),
                (
                    RecordingJet::constant(2.0),
                    InPlaneCoordinate::EffectiveIndex,
                ),
            ];

            for (value, coordinate) in cases {
                let result = canonicalise(value, coordinate, &vacuum_angular_wavenumber, None);

                assert!(
                    result.is_ok(),
                    "{coordinate:?} unexpectedly required \
                     an incident index",
                );
            }
        }
    }

    mod consistency {
        use super::*;

        #[test]
        fn equivalent_parameterisations_produce_same_canonical_value() {
            let vacuum_angular_wavenumber_value = 20.0;
            let incident_index_value = 1.5;
            let angle_degrees = 30.0;

            let expected_parallel_angular_wavenumber = incident_index_value
                * vacuum_angular_wavenumber_value
                * nalgebra::ComplexField::sin(angle_degrees.to_radians());

            let effective_index =
                expected_parallel_angular_wavenumber / vacuum_angular_wavenumber_value;

            let parallel_wavenumber = expected_parallel_angular_wavenumber / (2.0 * PI);

            let vacuum_angular_wavenumber = RecordingJet::constant(vacuum_angular_wavenumber_value);

            let incident_index = RecordingJet::constant(incident_index_value);

            let from_parallel_angular_wavenumber = canonicalise(
                RecordingJet::constant(expected_parallel_angular_wavenumber),
                InPlaneCoordinate::ParallelAngularWavenumber(InverseLengthUnit::PerCentimetre),
                &vacuum_angular_wavenumber,
                None,
            )
            .unwrap();

            let from_parallel_wavenumber = canonicalise(
                RecordingJet::constant(parallel_wavenumber),
                InPlaneCoordinate::ParallelWavenumber(InverseLengthUnit::PerCentimetre),
                &vacuum_angular_wavenumber,
                None,
            )
            .unwrap();

            let from_effective_index = canonicalise(
                RecordingJet::constant(effective_index),
                InPlaneCoordinate::EffectiveIndex,
                &vacuum_angular_wavenumber,
                None,
            )
            .unwrap();

            let from_incident_angle = canonicalise(
                RecordingJet::constant(angle_degrees),
                InPlaneCoordinate::IncidentAngle(AngleUnit::Degree),
                &vacuum_angular_wavenumber,
                Some(&incident_index),
            )
            .unwrap();

            let expected = Complex64::new(expected_parallel_angular_wavenumber, 0.0);

            assert_complex_close(from_parallel_angular_wavenumber.value, expected);

            assert_complex_close(from_parallel_wavenumber.value, expected);

            assert_complex_close(from_effective_index.value, expected);

            assert_complex_close(from_incident_angle.value, expected);
        }

        #[test]
        fn radian_and_degree_angles_produce_same_canonical_value() {
            let vacuum_angular_wavenumber = RecordingJet::constant(12.0);

            let incident_index = RecordingJet::constant(1.7);

            let from_degrees = canonicalise(
                RecordingJet::constant(45.0),
                InPlaneCoordinate::IncidentAngle(AngleUnit::Degree),
                &vacuum_angular_wavenumber,
                Some(&incident_index),
            )
            .unwrap();

            let from_radians = canonicalise(
                RecordingJet::constant(PI / 4.0),
                InPlaneCoordinate::IncidentAngle(AngleUnit::Radian),
                &vacuum_angular_wavenumber,
                Some(&incident_index),
            )
            .unwrap();

            assert_complex_close(from_degrees.value, from_radians.value);
        }

        #[test]
        fn inverse_centimetre_and_inverse_micrometre_inputs_agree() {
            let vacuum_angular_wavenumber = RecordingJet::constant(10.0);

            let from_inverse_centimetres = canonicalise(
                RecordingJet::constant(20_000.0),
                InPlaneCoordinate::ParallelAngularWavenumber(InverseLengthUnit::PerCentimetre),
                &vacuum_angular_wavenumber,
                None,
            )
            .unwrap();

            let from_inverse_micrometres = canonicalise(
                RecordingJet::constant(2.0),
                InPlaneCoordinate::ParallelAngularWavenumber(InverseLengthUnit::PerMicrometre),
                &vacuum_angular_wavenumber,
                None,
            )
            .unwrap();

            assert_complex_close(
                from_inverse_centimetres.value,
                from_inverse_micrometres.value,
            );
        }
    }

    mod derivatives {
        use super::*;

        #[test]
        fn parallel_angular_wavenumber_propagates_unit_jacobian() {
            let vacuum_angular_wavenumber = RecordingJet::constant(10.0);

            let result = canonicalise(
                RecordingJet::variable(3.0),
                InPlaneCoordinate::ParallelAngularWavenumber(InverseLengthUnit::PerCentimetre),
                &vacuum_angular_wavenumber,
                None,
            )
            .unwrap();

            assert_complex_close(result.value, Complex64::new(3.0, 0.0));

            assert_complex_close(result.derivative, Complex64::new(1.0, 0.0));
        }

        #[test]
        fn parallel_angular_wavenumber_propagates_length_unit_jacobian() {
            let vacuum_angular_wavenumber = RecordingJet::constant(10.0);

            let result = canonicalise(
                RecordingJet::variable(2.0),
                InPlaneCoordinate::ParallelAngularWavenumber(InverseLengthUnit::PerMicrometre),
                &vacuum_angular_wavenumber,
                None,
            )
            .unwrap();

            assert_complex_close(result.value, Complex64::new(2.0e4, 0.0));

            assert_complex_close(result.derivative, Complex64::new(1.0e4, 0.0));
        }

        #[test]
        fn parallel_wavenumber_propagates_two_pi_jacobian() {
            let vacuum_angular_wavenumber = RecordingJet::constant(10.0);

            let result = canonicalise(
                RecordingJet::variable(3.0),
                InPlaneCoordinate::ParallelWavenumber(InverseLengthUnit::PerCentimetre),
                &vacuum_angular_wavenumber,
                None,
            )
            .unwrap();

            assert_complex_close(result.value, Complex64::new(6.0 * PI, 0.0));

            assert_complex_close(result.derivative, Complex64::new(2.0 * PI, 0.0));
        }

        #[test]
        fn effective_index_derivative_is_vacuum_angular_wavenumber() {
            let vacuum_angular_wavenumber = RecordingJet::constant(12.0);

            let result = canonicalise(
                RecordingJet::variable(1.5),
                InPlaneCoordinate::EffectiveIndex,
                &vacuum_angular_wavenumber,
                None,
            )
            .unwrap();

            assert_complex_close(result.value, Complex64::new(18.0, 0.0));

            assert_complex_close(result.derivative, Complex64::new(12.0, 0.0));
        }

        #[test]
        fn effective_index_combines_derivatives_from_both_operands() {
            let effective_index = RecordingJet::variable(1.5);

            let vacuum_angular_wavenumber = RecordingJet::variable(12.0);

            let result = canonicalise(
                effective_index,
                InPlaneCoordinate::EffectiveIndex,
                &vacuum_angular_wavenumber,
                None,
            )
            .unwrap();

            // Both quantities depend on the same test variable:
            //
            // d(n_eff k0)/dx
            // = (dn_eff/dx) k0 + n_eff (dk0/dx)
            // = 1 * 12 + 1.5 * 1
            // = 13.5.
            assert_complex_close(result.derivative, Complex64::new(13.5, 0.0));
        }

        #[test]
        fn radian_angle_derivative_applies_sine_chain_rule() {
            let angle_radians = PI / 6.0;
            let incident_index_value = 2.0;
            let vacuum_angular_wavenumber_value = 10.0;

            let vacuum_angular_wavenumber = RecordingJet::constant(vacuum_angular_wavenumber_value);

            let incident_index = RecordingJet::constant(incident_index_value);

            let result = canonicalise(
                RecordingJet::variable(angle_radians),
                InPlaneCoordinate::IncidentAngle(AngleUnit::Radian),
                &vacuum_angular_wavenumber,
                Some(&incident_index),
            )
            .unwrap();

            let expected_derivative =
                incident_index_value * vacuum_angular_wavenumber_value * angle_radians.cos();

            assert_complex_close(result.value, Complex64::new(10.0, 0.0));

            assert_complex_close(result.derivative, Complex64::new(expected_derivative, 0.0));
        }

        #[test]
        fn degree_angle_derivative_includes_radians_per_degree() {
            let angle_degrees = 30.0;
            let angle_radians = angle_degrees.to_radians();

            let incident_index_value = 2.0;
            let vacuum_angular_wavenumber_value = 10.0;

            let vacuum_angular_wavenumber = RecordingJet::constant(vacuum_angular_wavenumber_value);

            let incident_index = RecordingJet::constant(incident_index_value);

            let result = canonicalise(
                RecordingJet::variable(angle_degrees),
                InPlaneCoordinate::IncidentAngle(AngleUnit::Degree),
                &vacuum_angular_wavenumber,
                Some(&incident_index),
            )
            .unwrap();

            let radians_per_degree = PI / 180.0;

            let expected_derivative = incident_index_value
                * vacuum_angular_wavenumber_value
                * angle_radians.cos()
                * radians_per_degree;

            assert_complex_close(result.derivative, Complex64::new(expected_derivative, 0.0));
        }

        #[test]
        fn zero_angle_has_maximal_angle_derivative() {
            let incident_index_value = 1.5;
            let vacuum_angular_wavenumber_value = 8.0;

            let vacuum_angular_wavenumber = RecordingJet::constant(vacuum_angular_wavenumber_value);

            let incident_index = RecordingJet::constant(incident_index_value);

            let result = canonicalise(
                RecordingJet::variable(0.0),
                InPlaneCoordinate::IncidentAngle(AngleUnit::Radian),
                &vacuum_angular_wavenumber,
                Some(&incident_index),
            )
            .unwrap();

            assert_complex_close(result.value, Complex64::new(0.0, 0.0));

            assert_complex_close(
                result.derivative,
                Complex64::new(incident_index_value * vacuum_angular_wavenumber_value, 0.0),
            );
        }

        #[test]
        fn pi_over_two_angle_has_zero_angle_derivative() {
            let vacuum_angular_wavenumber = RecordingJet::constant(8.0);

            let incident_index = RecordingJet::constant(1.5);

            let result = canonicalise(
                RecordingJet::variable(FRAC_PI_2),
                InPlaneCoordinate::IncidentAngle(AngleUnit::Radian),
                &vacuum_angular_wavenumber,
                Some(&incident_index),
            )
            .unwrap();

            assert_close(result.derivative.re, 0.0);

            assert_close(result.derivative.im, 0.0);
        }

        #[test]
        fn incident_angle_propagates_incident_index_derivative() {
            let angle = RecordingJet::constant(PI / 6.0);

            let vacuum_angular_wavenumber = RecordingJet::constant(10.0);

            let incident_index = RecordingJet::variable(2.0);

            let result = canonicalise(
                angle,
                InPlaneCoordinate::IncidentAngle(AngleUnit::Radian),
                &vacuum_angular_wavenumber,
                Some(&incident_index),
            )
            .unwrap();

            // d[n k0 sin(theta)]/dn = k0 sin(theta) = 5.
            assert_complex_close(result.derivative, Complex64::new(5.0, 0.0));
        }

        #[test]
        fn incident_angle_propagates_vacuum_wavenumber_derivative() {
            let angle = RecordingJet::constant(PI / 6.0);

            let vacuum_angular_wavenumber = RecordingJet::variable(10.0);

            let incident_index = RecordingJet::constant(2.0);

            let result = canonicalise(
                angle,
                InPlaneCoordinate::IncidentAngle(AngleUnit::Radian),
                &vacuum_angular_wavenumber,
                Some(&incident_index),
            )
            .unwrap();

            // d[n k0 sin(theta)]/dk0 = n sin(theta) = 1.
            assert_complex_close(result.derivative, Complex64::new(1.0, 0.0));
        }

        #[test]
        fn constant_inputs_produce_zero_derivative() {
            let vacuum_angular_wavenumber = RecordingJet::constant(10.0);

            let incident_index = RecordingJet::constant(2.0);

            let result = canonicalise(
                RecordingJet::constant(30.0),
                InPlaneCoordinate::IncidentAngle(AngleUnit::Degree),
                &vacuum_angular_wavenumber,
                Some(&incident_index),
            )
            .unwrap();

            assert_complex_close(result.derivative, Complex64::new(0.0, 0.0));
        }
    }
}
