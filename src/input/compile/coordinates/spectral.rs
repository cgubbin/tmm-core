use nalgebra::ComplexField;
use ndarray::{ArrayBase, Data, Dimension};
use num_traits::{Float, FloatConst, FromPrimitive};
use thiserror::Error;

use crate::input::SpectralCoordinate;

use super::CoordinateJet;

/// Speed of light in vacuum in centimetres per second.
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

pub(crate) fn canonicalise_spectral<C, D, J>(value: J, coordinate: SpectralCoordinate) -> J
where
    C: ComplexField,
    C::RealField: Float + FloatConst + FromPrimitive + Copy,
    J: CoordinateJet<C, D>,
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
