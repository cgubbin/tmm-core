use thiserror::Error;

use crate::parameter::{DerivativeMappingError, ParameterValidationError};

/// Invalid real-axis evaluation request.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum SolveRequestError {
    #[error(transparent)]
    ParameterValidation(#[from] ParameterValidationError),

    #[error(transparent)]
    DerivativeMapping(#[from] DerivativeMappingError),
}

/// Failure while compiling or solving a real-axis problem.
#[derive(Debug, Error)]
pub enum RealAxisEvaluationError<C, B> {
    #[error(transparent)]
    Request(#[from] SolveRequestError),

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

impl<C, B> RealAxisEvaluationError<C, B> {
    pub(crate) fn compile(source: C) -> Self {
        Self::Compile { source }
    }

    pub(crate) fn backend(source: B) -> Self {
        Self::Backend { source }
    }
}
