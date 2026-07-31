use thiserror::Error;

use crate::{input::ParameterAssignmentError, parameter::ThicknessSeedError};

/// An invalid high-level evaluation request.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum RequestError {
    #[error(transparent)]
    ThicknessSeed(#[from] ThicknessSeedError),

    #[error(transparent)]
    ParameterAssignment(#[from] ParameterAssignmentError),
}

/// Failure while preparing or solving a plane-wave problem.
///
/// `C` is the error returned by plane-wave compilation and `B` is the error
/// returned by the selected backend.
#[derive(Debug, Error)]
pub enum PlaneWaveEvaluationError<C, B> {
    #[error(transparent)]
    Request(#[from] RequestError),

    #[error("failed to compile the plane-wave problem")]
    Compile {
        #[source]
        source: C,
    },

    #[error("plane-wave backend failed")]
    Backend {
        #[source]
        source: B,
    },
}

impl<C, B> PlaneWaveEvaluationError<C, B> {
    pub(crate) fn compile(source: C) -> Self {
        Self::Compile { source }
    }

    pub(crate) fn backend(source: B) -> Self {
        Self::Backend { source }
    }
}
