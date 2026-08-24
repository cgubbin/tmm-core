//! Boundary-resolved directional waves and canonical isotropic states.
//!
//! A finite homogeneous layer has two planar boundaries. At each boundary the
//! electromagnetic solution may be represented in two related forms:
//!
//! - [`BoundaryWaves`] stores forward- and backward-labelled directional
//!   amplitudes in the basis of the local medium;
//! - [`BoundaryState`] stores the canonical isotropic state used by the
//!   transfer and scattering formulations.
//!
//! Directional amplitudes are basis-dependent and are not generally
//! continuous across an interface. The canonical state is the appropriate
//! representation for continuity checks and subsequent physical projection.
//!
//! [`LayerBoundaryWaves`] and [`LayerBoundaryStates`] pair the left and right
//! boundaries of one finite layer. [`LayerBoundaries`] stores such quantities
//! for all finite layers in physical left-to-right order.

mod project;
mod state;
mod waves;

pub use project::BoundaryProjectionError;
pub use state::{BoundaryState, LayerBoundaryStates};
pub use waves::{BoundaryWaves, LayerBoundaryWaves};

pub(crate) use project::{
    project_layer_boundary_states, project_layer_boundary_waves, project_layer_mode_waves,
};

use crate::{FiniteLayerIndex, algebra::ScaleBy};

/// A quantity associated with every finite layer, ordered physically from
/// left to right.
///
/// The collection contains no exterior-medium records. Its length therefore
/// equals the number of finite layers in the stack.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerBoundaries<T> {
    layers: Vec<T>,
}

impl<T> LayerBoundaries<T> {
    pub(crate) fn new(layers: Vec<T>) -> Self {
        Self { layers }
    }

    /// Return the number of finite-layer records.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Return whether the stack contains no finite-layer records.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Return the record for `index`.
    pub fn get(&self, index: FiniteLayerIndex) -> Option<&T> {
        self.layers.get(index.get())
    }

    /// Iterate in physical left-to-right finite-layer order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &T> {
        self.layers.iter()
    }

    pub(crate) fn into_inner(self) -> Vec<T> {
        self.layers
    }
}

impl<S, T> ScaleBy<S> for LayerBoundaries<T>
where
    T: ScaleBy<S>,
{
    fn scale_by(self, scale: &S) -> Self {
        Self {
            layers: self
                .layers
                .into_iter()
                .map(|each| each.scale_by(scale))
                .collect(),
        }
    }
}
