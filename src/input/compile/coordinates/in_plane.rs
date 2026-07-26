use nalgebra::ComplexField;
use ndarray::{ArrayBase, Data, Dimension};
use num_traits::{Float, FloatConst, FromPrimitive};
use thiserror::Error;

use crate::input::{InPlaneCoordinate, compile::coordinates::CoordinateJet};

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

pub(crate) fn validate_in_plane<R, S, D>(
    values: &ArrayBase<S, D>,
    coordinate: InPlaneCoordinate,
) -> Result<(), InPlaneInputError<R>>
where
    R: Float + FloatConst + FromPrimitive + Copy,
    S: Data<Elem = R>,
    D: Dimension,
{
    for (index, &value) in values.iter().enumerate() {
        if !value.is_finite() {
            return Err(InPlaneInputError::NonFinite { index, value });
        }

        if let InPlaneCoordinate::IncidentAngle(unit) = coordinate {
            let radians = value * unit.scale_to_radians();

            let half_pi = R::PI() / (R::one() + R::one());

            if radians < -half_pi || radians > half_pi {
                return Err(InPlaneInputError::AngleOutsidePrincipalInterval { index, radians });
            }
        }
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
pub enum CanonicaliseInPlaneError {
    #[error(
        "incident-angle input requires the refractive \
         index of the incident medium"
    )]
    MissingIncidentIndex,
}

pub(crate) fn canonicalise_in_plane<C, D, J>(
    value: J,
    coordinate: InPlaneCoordinate,
    vacuum_angular_wavenumber: &J,
    incident_index: Option<&J>,
) -> Result<J, CanonicaliseInPlaneError>
where
    C: ComplexField,
    C::RealField: Float + FloatConst + FromPrimitive + Copy,
    J: CoordinateJet<C, D>,
{
    let two_pi = <C::RealField as FloatConst>::PI() + <C::RealField as FloatConst>::PI();

    match coordinate {
        InPlaneCoordinate::ParallelAngularWavenumber(unit) => {
            Ok(value.scale_real(unit.scale_to_inverse_centimetres::<C::RealField>()))
        }

        InPlaneCoordinate::ParallelWavenumber(unit) => {
            Ok(value.scale_real(unit.scale_to_inverse_centimetres::<C::RealField>() * two_pi))
        }

        InPlaneCoordinate::EffectiveIndex => Ok(value.multiply(vacuum_angular_wavenumber.clone())),

        InPlaneCoordinate::IncidentAngle(unit) => {
            let incident_index =
                incident_index.ok_or(CanonicaliseInPlaneError::MissingIncidentIndex)?;

            let sine = value.scale_real(unit.scale_to_radians()).sin();

            Ok(incident_index
                .clone()
                .multiply(vacuum_angular_wavenumber.clone())
                .multiply(sine))
        }
    }
}
