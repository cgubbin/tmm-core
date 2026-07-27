use thiserror::Error;

use crate::input::compile::{
    coordinates::{
        CoordinateVariable,
        in_plane::{CanonicaliseInPlaneError, InPlaneInputError},
        spectral::SpectralInputError,
    },
    seed::UnsupportedDerivativeSlot,
};

#[derive(Debug, Error)]
pub enum CoordinateCompileError<R> {
    #[error(transparent)]
    Spectral(#[from] SpectralInputError<R>),

    #[error(transparent)]
    InPlane(#[from] InPlaneInputError<R>),

    #[error(transparent)]
    CanonicaliseInPlane(#[from] CanonicaliseInPlaneError),

    #[error("failed to seed {variable:?}: {source}")]
    Seed {
        variable: CoordinateVariable,

        #[source]
        source: UnsupportedDerivativeSlot,
    },
}
