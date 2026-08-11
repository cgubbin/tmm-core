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
    /// The caller-supplied spectral coordinate, in its supplied representation
    /// and units.
    Spectral,

    /// The caller-supplied in-plane coordinate, in its supplied representation
    /// and units.
    InPlane,

    /// The physical thickness of one finite layer.
    LayerThickness(FiniteLayerIndex),
}

/// Zero-based index of a finite layer in left-to-right stack order.
///
/// Exterior media are not included in this index space.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub struct FiniteLayerIndex(usize);

impl FiniteLayerIndex {
    /// Construct an index identifying a finite layer.
    ///
    /// The index is validated against a particular stack when it is used.
    pub const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Return the zero-based finite-layer index.
    pub const fn get(self) -> usize {
        self.0
    }
}

impl From<usize> for FiniteLayerIndex {
    fn from(index: usize) -> Self {
        Self::new(index)
    }
}

impl From<FiniteLayerIndex> for usize {
    fn from(index: FiniteLayerIndex) -> Self {
        index.get()
    }
}

impl Parameter {
    pub(crate) fn validate(
        self,
        finite_layer_count: usize,
    ) -> Result<(), ParameterValidationError> {
        match self {
            Parameter::LayerThickness(FiniteLayerIndex(layer)) if layer >= finite_layer_count => {
                Err(ParameterValidationError::LayerOutOfBounds {
                    index: layer,
                    finite_layer_count,
                })
            }

            _ => Ok(()),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ParameterValidationError {
    #[error(
        "requested thickness derivative for layer {index}, \
         but the stack contains {finite_layer_count} layers"
    )]
    LayerOutOfBounds {
        index: usize,
        finite_layer_count: usize,
    },
}
