//! Validation and canonicalisation of caller-facing spectral coordinates.
//!
//! The numerical backend represents every spectral coordinate as a vacuum
//! angular wavenumber expressed in inverse centimetres.
//!
//! This module validates caller-facing spectral values before converting them
//! into that canonical representation.
//!
//! Validation checks only properties of the supplied values themselves (for
//! example finiteness and positivity). Unit conversions and coordinate
//! transformations are performed separately so that validation remains
//! independent of canonicalisation.
//!
//! Canonicalisation operates on seeded jet values rather than raw scalars.
//! Consequently, derivatives are taken with respect to the caller-facing
//! spectral coordinate while the jet algebra automatically propagates the
//! required chain rule.
//!
//! The supported transformations are:
//!
//! - vacuum angular wavenumber → identity (with unit conversion);
//! - vacuum wavenumber → vacuum angular wavenumber;
//! - frequency → vacuum angular wavenumber;
//! - angular frequency → vacuum angular wavenumber;
//! - vacuum wavelength → vacuum angular wavenumber.

use nalgebra::ComplexField;
use ndarray::{ArrayBase, Data, Dimension};
use num_traits::{Float, FloatConst, FromPrimitive};
use thiserror::Error;

use crate::input::SpectralCoordinate;

use super::CanonicalCoordinateJet;

/// Exact speed of light expressed in centimetres per second.
///
/// This constant converts frequency- and angular-frequency-based
/// parameterisations into the canonical vacuum angular wavenumber.
fn speed_of_light_cm_per_second<R>() -> R
where
    R: Float,
{
    R::from(29_979_245_800.0).expect("speed of light must be representable")
}

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
/// Every supplied value must be finite and strictly positive.
///
/// No unit conversion or canonicalisation is performed.
///
/// # Errors
///
/// Returns [`SpectralInputError`] if any value is non-finite or is less than
/// or equal to zero.
pub(crate) fn validate_spectral<R, S, D>(
    values: &ArrayBase<S, D>,
) -> Result<(), SpectralInputError<R>>
where
    R: Float + Copy,
    S: Data<Elem = R>,
    D: Dimension,
{
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(SpectralInputError::NonFinite { index, value });
        }

        if value <= R::zero() {
            return Err(SpectralInputError::NonPositive { index, value });
        }
    }

    Ok(())
}

/// Convert a seeded caller-facing spectral coordinate into the canonical
/// vacuum angular wavenumber.
///
/// The returned quantity is always expressed as a vacuum angular wavenumber in
/// inverse centimetres, regardless of the caller-facing parameterisation.
///
/// Coordinate transformations are applied through the supplied jet algebra so
/// that derivatives with respect to the original caller-facing coordinate are
/// propagated automatically.
///
/// The transformations are:
///
/// ```text
/// β₀            → β₀
///
/// k₀            → 2π k₀
///
/// ν             → 2πν / c
///
/// ω             → ω / c
///
/// λ             → 2π / λ
/// ```
///
/// where
///
/// - `β₀` is the vacuum angular wavenumber;
/// - `k₀` is the vacuum wavenumber;
/// - `ν` is frequency;
/// - `ω` is angular frequency;
/// - `λ` is vacuum wavelength;
/// - `c` is the speed of light in vacuum.
pub(crate) fn canonicalise_spectral<C, D, J>(value: J, coordinate: SpectralCoordinate) -> J
where
    C: ComplexField,
    C::RealField: Float + FloatConst + FromPrimitive + Copy,
    J: CanonicalCoordinateJet<C, D>,
    D: Dimension,
{
    let two_pi = <C::RealField as FloatConst>::PI() + <C::RealField as FloatConst>::PI();

    match coordinate {
        SpectralCoordinate::VacuumAngularWavenumber(unit) => {
            value.scale_real(unit.scale_to_inverse_centimetres())
        }

        SpectralCoordinate::VacuumWavenumber(unit) => {
            value.scale_real(unit.scale_to_inverse_centimetres::<C::RealField>() * two_pi)
        }

        SpectralCoordinate::Frequency(unit) => {
            let frequency_scale = unit.scale_to_hertz();

            let factor = two_pi * frequency_scale / speed_of_light_cm_per_second();

            value.scale_real(factor)
        }

        SpectralCoordinate::AngularFrequency(unit) => {
            let angular_frequency_scale: C::RealField = unit.scale_to_radians_per_second();

            let factor = angular_frequency_scale / speed_of_light_cm_per_second();

            value.scale_real(factor)
        }

        SpectralCoordinate::VacuumWavelength(unit) => {
            let length_scale = unit.scale_to_centimetres();

            value
                .scale_real(length_scale)
                .reciprocal()
                .scale_real(two_pi)
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;

    use super::*;

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
    }
}
