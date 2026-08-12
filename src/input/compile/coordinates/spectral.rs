//! Validation and jet-aware compilation of caller-facing spectral coordinates.
//!
//! The numerical backend represents every spectral coordinate as vacuum
//! angular wavenumber in inverse centimetres.
//!
//! The intrinsic physical transformation from each supported spectral
//! representation is defined by `lamina-units`. This module validates the
//! supplied samples, seeds any requested derivative direction, and applies
//! that shared transformation through Lamina's jet algebra.
//!
//! Seeding occurs before transformation, so derivatives remain with respect
//! to the caller-facing spectral coordinate.

use lamina_units::{SpectralCoordinate, SpectralTransform};
use nalgebra::ComplexField;
use ndarray::{ArrayBase, Data, Dimension};
use num_traits::{Float, FloatConst, FromPrimitive};
use thiserror::Error;

use super::CanonicalCoordinateJet;

#[derive(Clone, Debug, PartialEq, Error)]
pub enum SpectralInputError<R> {
    #[error(
        "spectral coordinate contains a non-finite value \
         at flat index {index}: {value}"
    )]
    NonFinite { index: usize, value: R },

    #[error(
        "spectral coordinate must be strictly positive, \
         but flat index {index} contains {value}"
    )]
    NonPositive { index: usize, value: R },
}

/// Validate a caller-facing spectral coordinate.
///
/// Every supplied value must be finite. Real inputs must be strictly positive;
/// for complex inputs, the real component must be strictly positive.
///
/// No unit conversion or canonicalisation is performed.
///
/// # Errors
///
/// Returns [`SpectralInputError`] if any value is non-finite or has a real
/// component less than or equal to zero.
pub(crate) fn validate_spectral<C, S, D>(
    values: &ArrayBase<S, D>,
) -> Result<(), SpectralInputError<C>>
where
    C: ComplexField + Copy,
    S: Data<Elem = C>,
    D: Dimension,
{
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(SpectralInputError::NonFinite { index, value });
        }

        if value.real() <= C::zero().real() {
            return Err(SpectralInputError::NonPositive { index, value });
        }
    }

    Ok(())
}

/// Convert a seeded caller-facing spectral coordinate into vacuum angular
/// wavenumber in inverse centimetres.
///
/// The physical transformation is defined by `lamina-units`. This function
/// applies that transformation through Lamina's jet algebra so derivatives
/// remain with respect to the caller-facing coordinate.
pub(crate) fn canonicalise_spectral<J>(value: J, coordinate: SpectralCoordinate) -> J
where
    J: CanonicalCoordinateJet,
    J::Scalar: ComplexField,
    <J::Scalar as ComplexField>::RealField: Float + FloatConst + FromPrimitive + Copy,
    J::Dimension: Dimension,
{
    type Real<J> = <<J as crate::algebra::Jet>::Scalar as ComplexField>::RealField;

    match coordinate.transform::<Real<J>>() {
        SpectralTransform::Linear { scale } => value.scale_real(scale),

        SpectralTransform::Reciprocal {
            input_scale,
            numerator,
        } => value
            .scale_real(input_scale)
            .reciprocal()
            .scale_real(numerator),
    }
}

#[cfg(test)]
mod tests {
    use lamina_units::{AngularFrequencyUnit, FrequencyUnit, InverseLengthUnit, LengthUnit};
    use nalgebra::Complex;
    use ndarray::{Ix0, arr1, array};
    use num_complex::Complex64;

    use super::*;
    use crate::algebra::Jet;

    use std::f64::consts::PI;

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

    #[derive(Clone, Debug, PartialEq)]
    struct RecordingJet {
        value: Complex<f64>,
        derivative: Complex<f64>,
    }

    impl RecordingJet {
        fn constant(value: f64) -> Self {
            Self {
                value: Complex::new(value, 0.0),
                derivative: Complex::new(0.0, 0.0),
            }
        }

        fn variable(value: f64) -> Self {
            Self {
                value: Complex::new(value, 0.0),
                derivative: Complex::new(1.0, 0.0),
            }
        }
    }

    impl Jet for RecordingJet {
        type Scalar = Complex<f64>;
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

    fn canonicalise(value: RecordingJet, coordinate: SpectralCoordinate) -> RecordingJet {
        canonicalise_spectral::<RecordingJet>(value, coordinate)
    }

    mod validation {
        use super::*;

        #[test]
        fn accepts_positive_finite_values() {
            let values = arr1(&[f64::MIN_POSITIVE, 1.0e-12, 1.0, 1.0e12, f64::MAX]);

            let result = validate_spectral(&values);

            assert_eq!(result, Ok(()));
        }

        #[test]
        fn rejects_zero() {
            let values = arr1(&[1.0, 0.0, 2.0]);

            let error = validate_spectral(&values).unwrap_err();

            assert_eq!(
                error,
                SpectralInputError::NonPositive {
                    index: 1,
                    value: 0.0,
                },
            );
        }

        #[test]
        fn rejects_negative_value() {
            let values = arr1(&[1.0, -2.5, 3.0]);

            let error = validate_spectral(&values).unwrap_err();

            assert_eq!(
                error,
                SpectralInputError::NonPositive {
                    index: 1,
                    value: -2.5,
                },
            );
        }

        #[test]
        fn rejects_negative_zero() {
            let values = arr1(&[-0.0]);

            let error = validate_spectral(&values).unwrap_err();

            assert_eq!(
                error,
                SpectralInputError::NonPositive {
                    index: 0,
                    value: -0.0,
                },
            );
        }

        #[test]
        fn rejects_nan() {
            let values = arr1(&[1.0, f64::NAN, 2.0]);

            let error = validate_spectral(&values).unwrap_err();

            assert!(matches!(
                error,
                SpectralInputError::NonFinite {
                    index: 1,
                    value,
                } if value.is_nan()
            ));
        }

        #[test]
        fn rejects_positive_infinity() {
            let values = arr1(&[1.0, f64::INFINITY]);

            let error = validate_spectral(&values).unwrap_err();

            assert_eq!(
                error,
                SpectralInputError::NonFinite {
                    index: 1,
                    value: f64::INFINITY,
                },
            );
        }

        #[test]
        fn rejects_negative_infinity() {
            let values = arr1(&[f64::NEG_INFINITY, 1.0]);

            let error = validate_spectral(&values).unwrap_err();

            assert_eq!(
                error,
                SpectralInputError::NonFinite {
                    index: 0,
                    value: f64::NEG_INFINITY,
                },
            );
        }

        #[test]
        fn reports_first_invalid_value() {
            let values = arr1(&[1.0, -2.0, f64::NAN, 0.0]);

            let error = validate_spectral(&values).unwrap_err();

            assert_eq!(
                error,
                SpectralInputError::NonPositive {
                    index: 1,
                    value: -2.0,
                },
            );
        }

        #[test]
        fn reports_non_finite_before_non_positive_for_same_value() {
            let values = arr1(&[f64::NEG_INFINITY]);

            let error = validate_spectral(&values).unwrap_err();

            assert_eq!(
                error,
                SpectralInputError::NonFinite {
                    index: 0,
                    value: f64::NEG_INFINITY,
                },
            );
        }

        #[test]
        fn reports_flat_index_for_multidimensional_input() {
            let values = ndarray::arr2(&[[1.0, 2.0], [3.0, 0.0]]);

            let error = validate_spectral(&values).unwrap_err();

            assert_eq!(
                error,
                SpectralInputError::NonPositive {
                    index: 3,
                    value: 0.0,
                },
            );
        }

        #[test]
        fn accepts_empty_input() {
            let values = ndarray::Array1::<f64>::from_vec(Vec::new());

            let result = validate_spectral(&values);

            assert_eq!(result, Ok(()));
        }

        #[test]
        fn works_with_array_views() {
            let values = arr1(&[1.0, 2.0, 3.0]);

            let view = values.view();

            let result = validate_spectral(&view);

            assert_eq!(result, Ok(()));
        }

        #[test]
        fn spectral_accepts_positive_real_part_with_nonzero_imaginary_part() {
            let values = array![Complex64::new(1.0, 2.0), Complex64::new(3.0, -4.0),];

            assert!(validate_spectral(&values).is_ok());
        }

        #[test]
        fn spectral_rejects_negative_real_part_with_finite_imaginary_part() {
            let values = array![Complex64::new(1.0, 0.0), Complex64::new(-0.5, 10.0),];

            let error = validate_spectral(&values).unwrap_err();

            assert!(matches!(
                error,
                SpectralInputError::NonPositive {
                    index: 1,
                    value
                } if value == Complex64::new(-0.5, 10.0)
            ));
        }

        #[test]
        fn spectral_rejects_zero_real_part_with_nonzero_imaginary_part() {
            let values = array![Complex64::new(0.0, 3.0)];

            let error = validate_spectral(&values).unwrap_err();

            assert!(matches!(
                error,
                SpectralInputError::NonPositive {
                    index: 0,
                    value
                } if value == Complex64::new(0.0, 3.0)
            ));
        }

        #[test]
        fn spectral_rejects_non_finite_imaginary_part() {
            let values = array![Complex64::new(1.0, 0.0), Complex64::new(2.0, f64::INFINITY),];

            let error = validate_spectral(&values).unwrap_err();

            assert!(matches!(
                error,
                SpectralInputError::NonFinite {
                    index: 1,
                    value
                } if value == Complex64::new(2.0, f64::INFINITY)
            ));
        }

        #[test]
        fn spectral_rejects_nan_imaginary_part() {
            let values = array![Complex64::new(1.0, f64::NAN)];

            let error = validate_spectral(&values).unwrap_err();

            assert!(matches!(
                error,
                SpectralInputError::NonFinite { index: 0, .. }
            ));
        }
    }

    mod canonicalisation {
        use super::*;

        fn speed_of_light_cm_per_second<F>() -> F
        where
            F: FromPrimitive,
        {
            F::from_f64(lamina_units::SPEED_OF_LIGHT_CM_PER_SECOND).unwrap()
        }

        #[test]
        fn vacuum_angular_wavenumber_in_inverse_centimetres_is_unchanged() {
            let result = canonicalise(
                RecordingJet::constant(12.5),
                SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
            );

            assert_close(result.value.re, 12.5);
            assert_close(result.value.im, 0.0);
        }

        #[test]
        fn vacuum_angular_wavenumber_converts_inverse_micrometres() {
            let result = canonicalise(
                RecordingJet::constant(1.0),
                SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerMicrometre),
            );

            // 1 µm⁻¹ = 10⁴ cm⁻¹.
            assert_close(result.value.re, 1.0e4);
            assert_close(result.value.im, 0.0);
        }

        #[test]
        fn vacuum_wavenumber_multiplies_by_two_pi() {
            let result = canonicalise(
                RecordingJet::constant(3.0),
                SpectralCoordinate::VacuumWavenumber(InverseLengthUnit::PerCentimetre),
            );

            assert_close(result.value.re, 6.0 * PI);
            assert_close(result.value.im, 0.0);
        }

        #[test]
        fn vacuum_wavenumber_converts_units_and_multiplies_by_two_pi() {
            let result = canonicalise(
                RecordingJet::constant(1.0),
                SpectralCoordinate::VacuumWavenumber(InverseLengthUnit::PerMicrometre),
            );

            assert_close(result.value.re, 2.0 * PI * 1.0e4);
            assert_close(result.value.im, 0.0);
        }

        #[test]
        fn frequency_uses_two_pi_over_speed_of_light() {
            let frequency = 3.0e12;

            let result = canonicalise(
                RecordingJet::constant(frequency),
                SpectralCoordinate::Frequency(FrequencyUnit::Hertz),
            );

            let expected = 2.0 * PI * frequency / speed_of_light_cm_per_second::<f64>();

            assert_close(result.value.re, expected);
            assert_close(result.value.im, 0.0);
        }

        #[test]
        fn frequency_converts_terahertz_to_hertz() {
            let frequency_terahertz = 3.0;

            let result = canonicalise(
                RecordingJet::constant(frequency_terahertz),
                SpectralCoordinate::Frequency(FrequencyUnit::Terahertz),
            );

            let expected =
                2.0 * PI * frequency_terahertz * 1.0e12 / speed_of_light_cm_per_second::<f64>();

            assert_close(result.value.re, expected);
            assert_close(result.value.im, 0.0);
        }

        #[test]
        fn angular_frequency_uses_one_over_speed_of_light() {
            let angular_frequency = 4.0e13;

            let result = canonicalise(
                RecordingJet::constant(angular_frequency),
                SpectralCoordinate::AngularFrequency(AngularFrequencyUnit::RadiansPerSecond),
            );

            let expected = angular_frequency / speed_of_light_cm_per_second::<f64>();

            assert_close(result.value.re, expected);
            assert_close(result.value.im, 0.0);
        }

        #[test]
        fn angular_frequency_converts_terahertz_scale() {
            let angular_frequency_terahertz = 4.0;

            let result = canonicalise(
                RecordingJet::constant(angular_frequency_terahertz),
                SpectralCoordinate::AngularFrequency(AngularFrequencyUnit::TeraradiansPerSecond),
            );

            let expected =
                angular_frequency_terahertz * 1.0e12 / speed_of_light_cm_per_second::<f64>();

            assert_close(result.value.re, expected);
            assert_close(result.value.im, 0.0);
        }

        #[test]
        fn vacuum_wavelength_uses_two_pi_over_wavelength() {
            let wavelength_cm = 2.0;

            let result = canonicalise(
                RecordingJet::constant(wavelength_cm),
                SpectralCoordinate::VacuumWavelength(LengthUnit::Centimetre),
            );

            assert_close(result.value.re, 2.0 * PI / wavelength_cm);

            assert_close(result.value.im, 0.0);
        }

        #[test]
        fn vacuum_wavelength_converts_units_before_reciprocal() {
            let wavelength_micrometres = 2.0;

            let result = canonicalise(
                RecordingJet::constant(wavelength_micrometres),
                SpectralCoordinate::VacuumWavelength(LengthUnit::Micrometre),
            );

            let wavelength_cm = wavelength_micrometres * 1.0e-4;

            assert_close(result.value.re, 2.0 * PI / wavelength_cm);

            assert_close(result.value.im, 0.0);
        }

        #[test]
        fn wavelength_of_two_pi_centimetres_produces_unity() {
            let result = canonicalise(
                RecordingJet::constant(2.0 * PI),
                SpectralCoordinate::VacuumWavelength(LengthUnit::Centimetre),
            );

            assert_close(result.value.re, 1.0);
            assert_close(result.value.im, 0.0);
        }

        #[test]
        fn frequency_of_c_over_two_pi_produces_unity() {
            let frequency = speed_of_light_cm_per_second::<f64>() / (2.0 * PI);

            let result = canonicalise(
                RecordingJet::constant(frequency),
                SpectralCoordinate::Frequency(FrequencyUnit::Hertz),
            );

            assert_close(result.value.re, 1.0);
            assert_close(result.value.im, 0.0);
        }

        #[test]
        fn angular_frequency_equal_to_c_produces_unity() {
            let angular_frequency = speed_of_light_cm_per_second::<f64>();

            let result = canonicalise(
                RecordingJet::constant(angular_frequency),
                SpectralCoordinate::AngularFrequency(AngularFrequencyUnit::RadiansPerSecond),
            );

            assert_close(result.value.re, 1.0);
            assert_close(result.value.im, 0.0);
        }
    }

    mod consistency {
        use super::*;

        fn speed_of_light_cm_per_second<F>() -> F
        where
            F: FromPrimitive,
        {
            F::from_f64(lamina_units::SPEED_OF_LIGHT_CM_PER_SECOND).unwrap()
        }

        #[test]
        fn every_parameterisation_of_same_wave_agrees() {
            let wavelength_cm = 5.0e-5;

            let speed_of_light = speed_of_light_cm_per_second::<f64>();

            let frequency_hz = speed_of_light / wavelength_cm;

            let angular_frequency = 2.0 * PI * frequency_hz;

            let vacuum_wavenumber = 1.0 / wavelength_cm;

            let vacuum_angular_wavenumber = 2.0 * PI / wavelength_cm;

            let from_vacuum_angular_wavenumber = canonicalise(
                RecordingJet::constant(vacuum_angular_wavenumber),
                SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
            );

            let from_vacuum_wavenumber = canonicalise(
                RecordingJet::constant(vacuum_wavenumber),
                SpectralCoordinate::VacuumWavenumber(InverseLengthUnit::PerCentimetre),
            );

            let from_frequency = canonicalise(
                RecordingJet::constant(frequency_hz),
                SpectralCoordinate::Frequency(FrequencyUnit::Hertz),
            );

            let from_angular_frequency = canonicalise(
                RecordingJet::constant(angular_frequency),
                SpectralCoordinate::AngularFrequency(AngularFrequencyUnit::RadiansPerSecond),
            );

            let from_wavelength = canonicalise(
                RecordingJet::constant(wavelength_cm),
                SpectralCoordinate::VacuumWavelength(LengthUnit::Centimetre),
            );

            assert_close(
                from_vacuum_angular_wavenumber.value.re,
                vacuum_angular_wavenumber,
            );

            assert_close(from_vacuum_wavenumber.value.re, vacuum_angular_wavenumber);

            assert_close(from_frequency.value.re, vacuum_angular_wavenumber);

            assert_close(from_angular_frequency.value.re, vacuum_angular_wavenumber);

            assert_close(from_wavelength.value.re, vacuum_angular_wavenumber);
        }

        #[test]
        fn five_hundred_nanometre_wave_agrees_in_every_parameterisation() {
            let wavelength_nanometres = 500.0;
            let wavelength_cm = wavelength_nanometres * 1.0e-7;

            let speed_of_light = speed_of_light_cm_per_second::<f64>();

            let frequency_hz = speed_of_light / wavelength_cm;

            let angular_frequency = 2.0 * PI * frequency_hz;

            let vacuum_wavenumber_per_cm = 1.0 / wavelength_cm;

            let vacuum_angular_wavenumber_per_cm = 2.0 * PI / wavelength_cm;

            let from_vacuum_angular_wavenumber = canonicalise(
                RecordingJet::constant(vacuum_angular_wavenumber_per_cm),
                SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
            );

            let from_vacuum_wavenumber = canonicalise(
                RecordingJet::constant(vacuum_wavenumber_per_cm),
                SpectralCoordinate::VacuumWavenumber(InverseLengthUnit::PerCentimetre),
            );

            let from_frequency = canonicalise(
                RecordingJet::constant(frequency_hz),
                SpectralCoordinate::Frequency(FrequencyUnit::Hertz),
            );

            let from_angular_frequency = canonicalise(
                RecordingJet::constant(angular_frequency),
                SpectralCoordinate::AngularFrequency(AngularFrequencyUnit::RadiansPerSecond),
            );

            let from_wavelength = canonicalise(
                RecordingJet::constant(wavelength_nanometres),
                SpectralCoordinate::VacuumWavelength(LengthUnit::Nanometre),
            );

            let expected = vacuum_angular_wavenumber_per_cm;

            assert_close(from_vacuum_angular_wavenumber.value.re, expected);

            assert_close(from_vacuum_wavenumber.value.re, expected);

            assert_close(from_frequency.value.re, expected);

            assert_close(from_angular_frequency.value.re, expected);

            assert_close(from_wavelength.value.re, expected);
        }

        #[test]
        fn inverse_centimetres_and_inverse_micrometres_agree() {
            let per_micrometre = 2.5;
            let per_centimetre = per_micrometre * 1.0e4;

            let from_inverse_centimetres = canonicalise(
                RecordingJet::constant(per_centimetre),
                SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
            );

            let from_inverse_micrometres = canonicalise(
                RecordingJet::constant(per_micrometre),
                SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerMicrometre),
            );

            assert_close(
                from_inverse_centimetres.value.re,
                from_inverse_micrometres.value.re,
            );
        }

        #[test]
        fn centimetres_micrometres_and_nanometres_agree() {
            let wavelength_cm = 5.0e-5;
            let wavelength_micrometres = 0.5;
            let wavelength_nanometres = 500.0;

            let from_centimetres = canonicalise(
                RecordingJet::constant(wavelength_cm),
                SpectralCoordinate::VacuumWavelength(LengthUnit::Centimetre),
            );

            let from_micrometres = canonicalise(
                RecordingJet::constant(wavelength_micrometres),
                SpectralCoordinate::VacuumWavelength(LengthUnit::Micrometre),
            );

            let from_nanometres = canonicalise(
                RecordingJet::constant(wavelength_nanometres),
                SpectralCoordinate::VacuumWavelength(LengthUnit::Nanometre),
            );

            assert_close(from_centimetres.value.re, from_micrometres.value.re);

            assert_close(from_centimetres.value.re, from_nanometres.value.re);
        }

        #[test]
        fn hertz_and_terahertz_agree() {
            let frequency_terahertz = 3.0;
            let frequency_hertz = frequency_terahertz * 1.0e12;

            let from_hertz = canonicalise(
                RecordingJet::constant(frequency_hertz),
                SpectralCoordinate::Frequency(FrequencyUnit::Hertz),
            );

            let from_terahertz = canonicalise(
                RecordingJet::constant(frequency_terahertz),
                SpectralCoordinate::Frequency(FrequencyUnit::Terahertz),
            );

            assert_close(from_hertz.value.re, from_terahertz.value.re);
        }

        #[test]
        fn frequency_and_angular_frequency_agree() {
            let frequency_hz = 4.0e12;
            let angular_frequency = 2.0 * PI * frequency_hz;

            let from_frequency = canonicalise(
                RecordingJet::constant(frequency_hz),
                SpectralCoordinate::Frequency(FrequencyUnit::Hertz),
            );

            let from_angular_frequency = canonicalise(
                RecordingJet::constant(angular_frequency),
                SpectralCoordinate::AngularFrequency(AngularFrequencyUnit::RadiansPerSecond),
            );

            assert_close(from_frequency.value.re, from_angular_frequency.value.re);
        }

        #[test]
        fn wavelength_and_vacuum_wavenumber_are_reciprocal_parameterisations() {
            let wavelength_cm = 2.5e-4;
            let vacuum_wavenumber = 1.0 / wavelength_cm;

            let from_wavelength = canonicalise(
                RecordingJet::constant(wavelength_cm),
                SpectralCoordinate::VacuumWavelength(LengthUnit::Centimetre),
            );

            let from_vacuum_wavenumber = canonicalise(
                RecordingJet::constant(vacuum_wavenumber),
                SpectralCoordinate::VacuumWavenumber(InverseLengthUnit::PerCentimetre),
            );

            assert_close(from_wavelength.value.re, from_vacuum_wavenumber.value.re);
        }
    }

    mod derivatives {
        use super::*;

        fn speed_of_light_cm_per_second<F>() -> F
        where
            F: FromPrimitive,
        {
            F::from_f64(lamina_units::SPEED_OF_LIGHT_CM_PER_SECOND).unwrap()
        }

        #[test]
        fn vacuum_angular_wavenumber_preserves_derivative() {
            let result = canonicalise(
                RecordingJet::variable(3.0),
                SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
            );

            assert_close(result.value.re, 3.0);
            assert_close(result.derivative.re, 1.0);
        }

        #[test]
        fn vacuum_angular_wavenumber_unit_conversion_scales_derivative() {
            let result = canonicalise(
                RecordingJet::variable(2.0),
                SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerMicrometre),
            );

            assert_close(result.value.re, 2.0e4);
            assert_close(result.derivative.re, 1.0e4);
        }

        #[test]
        fn vacuum_wavenumber_derivative_is_two_pi() {
            let result = canonicalise(
                RecordingJet::variable(3.0),
                SpectralCoordinate::VacuumWavenumber(InverseLengthUnit::PerCentimetre),
            );

            assert_close(result.value.re, 6.0 * PI);
            assert_close(result.derivative.re, 2.0 * PI);
        }

        #[test]
        fn vacuum_wavenumber_unit_conversion_scales_derivative() {
            let result = canonicalise(
                RecordingJet::variable(1.0),
                SpectralCoordinate::VacuumWavenumber(InverseLengthUnit::PerMicrometre),
            );

            assert_close(result.derivative.re, 2.0 * PI * 1.0e4);
        }

        #[test]
        fn frequency_derivative_is_two_pi_over_c() {
            let result = canonicalise(
                RecordingJet::variable(5.0e12),
                SpectralCoordinate::Frequency(FrequencyUnit::Hertz),
            );

            let expected = 2.0 * PI / speed_of_light_cm_per_second::<f64>();

            assert_close(result.derivative.re, expected);
        }

        #[test]
        fn angular_frequency_derivative_is_one_over_c() {
            let result = canonicalise(
                RecordingJet::variable(8.0e13),
                SpectralCoordinate::AngularFrequency(AngularFrequencyUnit::RadiansPerSecond),
            );

            let expected = 1.0 / speed_of_light_cm_per_second::<f64>();

            assert_close(result.derivative.re, expected);
        }

        #[test]
        fn wavelength_derivative_matches_analytic_result() {
            let wavelength = 2.0;

            let result = canonicalise(
                RecordingJet::variable(wavelength),
                SpectralCoordinate::VacuumWavelength(LengthUnit::Centimetre),
            );

            let expected = -2.0 * PI / wavelength.powi(2);

            assert_close(result.derivative.re, expected);
        }

        #[test]
        fn wavelength_unit_conversion_scales_derivative() {
            let wavelength = 2.0;

            let result = canonicalise(
                RecordingJet::variable(wavelength),
                SpectralCoordinate::VacuumWavelength(LengthUnit::Micrometre),
            );

            let scale = 1.0e-4;

            let wavelength_cm = wavelength * scale;

            let expected = (-2.0 * PI / wavelength_cm.powi(2)) * scale;

            assert_close(result.derivative.re, expected);
        }

        #[test]
        fn wavelength_derivative_is_negative() {
            let result = canonicalise(
                RecordingJet::variable(1.0),
                SpectralCoordinate::VacuumWavelength(LengthUnit::Centimetre),
            );

            assert!(result.derivative.re < 0.0);
        }

        #[test]
        fn constant_input_has_zero_derivative() {
            let result = canonicalise(
                RecordingJet::constant(10.0),
                SpectralCoordinate::VacuumWavelength(LengthUnit::Centimetre),
            );

            assert_close(result.derivative.re, 0.0);
        }

        #[test]
        fn reciprocal_chain_rule_matches_multiple_wavelengths() {
            for wavelength in [0.2, 0.5, 1.0, 2.0, 10.0] {
                let result = canonicalise(
                    RecordingJet::variable(wavelength),
                    SpectralCoordinate::VacuumWavelength(LengthUnit::Centimetre),
                );

                let analytic = -2.0 * PI / wavelength.powi(2);

                assert_close(result.derivative.re, analytic);
            }
        }

        #[test]
        fn derivative_is_linear_under_scaling() {
            let a = canonicalise(
                RecordingJet::variable(1.0),
                SpectralCoordinate::Frequency(FrequencyUnit::Hertz),
            );

            let b = canonicalise(
                RecordingJet::variable(2.0),
                SpectralCoordinate::Frequency(FrequencyUnit::Hertz),
            );

            assert_close(a.derivative.re, b.derivative.re);
        }
    }
}
