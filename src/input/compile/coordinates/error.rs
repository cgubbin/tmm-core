use thiserror::Error;

use crate::algebra::UnsupportedDerivativeSlot;
use crate::input::compile::coordinates::{
    CoordinateVariable,
    in_plane::{InPlaneCanonicalisationError, InPlaneInputError},
    spectral::SpectralInputError,
};

#[derive(Debug, Error)]
pub enum CoordinateCompileError<R> {
    #[error(transparent)]
    Spectral(#[from] SpectralInputError<R>),

    #[error(transparent)]
    InPlane(#[from] InPlaneInputError<R>),

    #[error(transparent)]
    InPlaneCanonicalisation(#[from] InPlaneCanonicalisationError),

    #[error("failed to seed {variable:?}: {source}")]
    Seed {
        variable: CoordinateVariable,

        #[source]
        source: UnsupportedDerivativeSlot,
    },

    #[error("incident side must be specified to compile with an incident angle")]
    MissingIncidentSide,

    #[error("incident-angle coordinates are not supported for complex input")]
    ComplexIncidentAngleUnsupported,
}
