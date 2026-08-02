mod project;
mod state;
mod waves;

pub use project::BoundaryProjectionError;
pub(crate) use project::{project_boundary_states, project_boundary_waves};
pub use state::{BoundaryState, LayerBoundaryStates};
pub use waves::{BoundaryWaves, LayerBoundaryWaves};

use crate::FiniteLayerIndex;

/// Boundary quantities for every finite layer, in physical left-to-right order.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerBoundaries<T> {
    layers: Vec<T>,
}

impl<T> LayerBoundaries<T> {
    pub(crate) fn new(layers: Vec<T>) -> Self {
        Self { layers }
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn get(&self, index: FiniteLayerIndex) -> Option<&T> {
        self.layers.get(index.0)
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &T> {
        self.layers.iter()
    }

    pub fn into_inner(self) -> Vec<T> {
        self.layers
    }

    pub fn map<U>(self, map: impl FnMut(T) -> U) -> LayerBoundaries<U> {
        LayerBoundaries::new(self.layers.into_iter().map(map).collect())
    }
}
