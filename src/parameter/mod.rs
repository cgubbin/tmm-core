mod derivative_mapping;
mod layout;

pub(crate) use derivative_mapping::{DerivativeMapping, DerivativeMappingError};
pub(crate) use layout::{BivariateMapping, DirectionalMapping, ValueMapping};

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
    LayerThickness(FiniteLayerIndex),
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FiniteLayerIndex(pub(crate) usize);

impl Parameter {
    pub(crate) fn validate(self, finite_layer_count: usize) -> Result<(), ThicknessSeedError> {
        match self {
            Parameter::LayerThickness(FiniteLayerIndex(layer)) if layer >= finite_layer_count => {
                Err(ThicknessSeedError::LayerOutOfBounds {
                    index: layer,
                    finite_layer_count,
                })
            }

            _ => Ok(()),
        }
    }
}

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
