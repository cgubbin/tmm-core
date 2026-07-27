use crate::input::{ParameterAssignment, ParameterAssignmentError};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ThicknessSeedError {
    #[error(
        "requested thickness derivative for layer {index}, \
         but the stack contains {finite_layer_count} layers"
    )]
    LayerOutOfBounds {
        index: usize,
        finite_layer_count: usize,
    },
}

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

    pub(crate) fn assignment(&self) -> Result<ParameterAssignment, ParameterAssignmentError> {
        ParameterAssignment::new(self.parameters())
    }

    pub(crate) fn validate(&self, finite_layer_count: usize) -> Result<(), ThicknessSeedError> {
        for parameter in self.parameters() {
            parameter.validate(finite_layer_count)?;
        }

        Ok(())
    }
}

/// A caller-facing parameter with respect to which derivatives may be taken.
///
/// Coordinate derivatives are taken with respect to the supplied coordinate
/// representation and units, not necessarily the backend's canonical
/// coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Parameter {
    /// The supplied spectral coordinate.
    Spectral,

    /// The supplied in-plane coordinate.
    InPlane,

    /// The physical thickness of one finite layer.
    LayerThickness { layer: usize },
}

impl Parameter {
    pub(crate) fn validate(self, finite_layer_count: usize) -> Result<(), ThicknessSeedError> {
        match self {
            Parameter::LayerThickness { layer } if layer >= finite_layer_count => {
                Err(ThicknessSeedError::LayerOutOfBounds {
                    index: layer,
                    finite_layer_count,
                })
            }

            _ => Ok(()),
        }
    }
}
