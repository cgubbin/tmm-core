//! Finite-layer observables and analytic homogeneous-layer integration.
//!
//! A stack containing `N` finite layers produces `N` layer records, ordered physically from left to
//! right.
//!
//! The layer pipeline has three stages:
//!
//! 1. retained boundary waves and homogeneous-layer quantities are assembled into internal layer
//!    data
//! 2. directional-wave and canonical-state products are integrated analytically through each
//!    homogeneous-layer
//! 3. physical projections produce layer power, dissipation or energy

mod dissipation;
mod energy;
mod integration;
mod power;
mod project;

pub use dissipation::LayerDissipation;
pub use energy::{LayerEnergy, LayerEnergyError};
pub use power::LayerPower;
pub use project::LayerProjectionError;

pub(crate) use energy::canonical_energy_normalization;
pub(crate) use integration::{
    IntegratedStateProducts, IntegratedWaveProducts, integrate_bilinear_wave_products,
    integrate_hermitian_wave_products,
};
pub(crate) use project::{
    IntegratedLayerData, LayerIntegrationInput, assemble_layer_integration_inputs,
};

use crate::FiniteLayerIndex;

/// A quantity associated with every finite-layer, in physical left-to-right order.
///
/// Exterior media are not represented. The collection length is therefore equal to the number of
/// finite layers in the stack;
#[derive(Clone, Debug, PartialEq)]
pub struct Layers<T> {
    values: Vec<T>,
}

impl<T> Layers<T> {
    pub(crate) fn new(values: Vec<T>) -> Self {
        Self { values }
    }

    /// Return the number of finite-layer records.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Return whether the stack contains non-finite layers
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Return the record associated with `index`.
    pub fn get(&self, index: FiniteLayerIndex) -> Option<&T> {
        self.values.get(index.0)
    }

    /// Return the leftmost finite-layer record.
    pub fn first(&self) -> Option<&T> {
        self.values.first()
    }

    /// Return the rightmost finite-layer record.
    pub fn last(&self) -> Option<&T> {
        self.values.last()
    }

    /// Iterate in physical left-to-right order over finite-layer records.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &T> {
        self.values.iter()
    }

    pub(crate) fn into_inner(self) -> Vec<T> {
        self.values
    }

    pub(crate) fn map<U>(self, map: impl FnMut(T) -> U) -> Layers<U> {
        Layers::new(self.values.into_iter().map(map).collect())
    }
}
