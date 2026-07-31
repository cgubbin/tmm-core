use crate::parameter::{DerivativeMapping, DerivativeMappingError, Parameter, ThicknessSeedError};

/// The numerical information requested from the solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveRequest {
    /// Compute values only.
    Value,

    /// Compute the value and first derivative with respect to one
    /// caller-facing parameter.
    UnivariateFirst { parameter: Parameter },

    /// Compute the value, first derivative, and second derivative with respect
    /// to one caller-facing parameter.
    UnivariateSecond { parameter: Parameter },

    /// Compute first derivatives with respect to the supplied spectral and
    /// in-plane coordinates.
    BivariateFirst { axis0: Parameter, axis1: Parameter },

    /// Compute the full Hessian with respect to the supplied spectral and
    /// in-plane coordinates.
    BivariateSecond { axis0: Parameter, axis1: Parameter },
}

impl SolveRequest {
    pub(crate) fn parameters(&self) -> impl Iterator<Item = Parameter> {
        let parameters = match self {
            Self::Value => vec![],
            Self::UnivariateFirst { parameter } => vec![*parameter],
            Self::UnivariateSecond { parameter } => vec![*parameter],
            Self::BivariateFirst { axis0, axis1 } => vec![*axis0, *axis1],
            Self::BivariateSecond { axis0, axis1 } => vec![*axis0, *axis1],
        };

        parameters.into_iter()
    }

    pub(crate) fn assignment(&self) -> Result<DerivativeMapping, DerivativeMappingError> {
        DerivativeMapping::new(self.parameters())
    }

    pub(crate) fn validate(&self, finite_layer_count: usize) -> Result<(), ThicknessSeedError> {
        for parameter in self.parameters() {
            parameter.validate(finite_layer_count)?;
        }

        Ok(())
    }
}
