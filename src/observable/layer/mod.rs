//! Finite-layer observables and analytic homogeneous-layer integration.
//!
//! A stack containing `N` finite layers produces `N` layer records in
//! physical left-to-right order.
//!
//! The layer pipeline has three stages:
//!
//! 1. retained boundary waves and homogeneous-layer quantities are assembled
//!    into internal layer data;
//! 2. directional-wave and canonical-state products are integrated
//!    analytically through each homogeneous layer;
//! 3. physical projections produce layer power, dissipation, or energy.
//!
//! Real-input dissipation and energy use Hermitian products. Bilinear wave
//! products are exposed internally for complex modal overlap and
//! normalization.

mod aggregate;
mod confinement;
mod dissipation;
mod energy;
mod integration;
mod overlap;
mod participation;
mod power;
mod project;

pub use aggregate::{AggregateEnergy, LayerAggregateError};
pub use confinement::{EnergyConfinement, LayerConfinementError};
pub use dissipation::LayerDissipation;
pub use energy::{LayerEnergy, LayerEnergyError};
pub use overlap::{
    AggregateHermitianOverlap, HermitianLayerOverlapInput, NormalizedHermitianOverlap,
};
pub use participation::{LayerParticipation, LayerParticipationError};
pub use power::LayerPower;
pub use project::LayerProjectionError;

pub(crate) use integration::{HermitianOverlapError, PairOperand};
pub(crate) use integration::{
    IntegratedHermitianStateProducts, integrate_hermitian_wave_products,
    project_integrated_field_norms,
};
pub(crate) use overlap::{HermitianLayerOverlap, LayerOverlapOperand};
pub(crate) use project::{LayerIntegrationInput, assemble_layer_integration_inputs};

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

    /// Return whether there are no finite-layer records.
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

    pub fn iter_indexed(&self) -> impl ExactSizeIterator<Item = (FiniteLayerIndex, &T)> {
        self.values
            .iter()
            .enumerate()
            .map(|(index, value)| (FiniteLayerIndex(index), value))
    }

    pub(crate) fn into_inner(self) -> Vec<T> {
        self.values
    }

    pub(crate) fn map<U>(self, map: impl FnMut(T) -> U) -> Layers<U> {
        Layers::new(self.values.into_iter().map(map).collect())
    }
}
