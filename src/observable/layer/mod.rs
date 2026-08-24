//! Finite-layer observables and analytic homogeneous-layer integration.
//!
//! A stack containing `N` finite layers produces `N` layer records in
//! physical left-to-right order.
//!
//! The finite-layer pipeline has three stages:
//!
//! 1. retained boundary waves, layer geometry, and constitutive quantities are
//!    assembled into internal layer inputs;
//! 2. products of directional waves and canonical states are integrated
//!    analytically through each homogeneous layer;
//! 3. physical projections produce layer-resolved power, dissipation, energy,
//!    participation, and confinement quantities.
//!
//! Real-frequency energy and dissipation use Hermitian products. Bilinear
//! products are retained separately for complex modal overlap and
//! normalization, where holomorphic dependence must be preserved.
//!
//! Public types represent physical layer observables. Integration kernels and
//! overlap operands remain internal implementation details.

mod aggregate;
mod confinement;
mod dissipation;
mod energy;
pub(crate) mod integration;
mod overlap;
mod participation;
mod power;
mod project;

pub use aggregate::{AggregateEnergy, LayerAggregateError};
pub use confinement::{EnergyConfinement, LayerConfinementError};
pub use dissipation::LayerDissipation;
pub use energy::{LayerEnergy, LayerEnergyError};
pub use overlap::{
    AggregateBilinearNormalization, AggregateBilinearOverlap, AggregateHermitianOverlap,
};
pub use participation::{LayerParticipation, LayerParticipationError};
pub use power::LayerPower;
pub use project::LayerProjectionError;

pub(crate) use integration::IntegratedHermitianCrossStateProducts;
pub(crate) use overlap::OverlapError;
pub(crate) use overlap::{
    BilinearLayerOverlap, HermitianLayerOverlap, HermitianLayerOverlapInput, LayerOverlapInput,
    LayerOverlapOperand,
};
pub(crate) use project::{LayerIntegrationInput, assemble_layer_integration_inputs};

use crate::FiniteLayerIndex;

/// A quantity associated with every finite layer in physical left-to-right
/// order.
///
/// Exterior media are not represented. The collection length therefore equals
/// the number of finite layers in the stack.
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
        self.values.get(index.get())
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

    /// Iterate over finite-layer records together with their typed layer indices.
    pub fn iter_indexed(&self) -> impl ExactSizeIterator<Item = (FiniteLayerIndex, &T)> {
        self.values
            .iter()
            .enumerate()
            .map(|(index, value)| (FiniteLayerIndex::new(index), value))
    }

    pub(crate) fn into_inner(self) -> Vec<T> {
        self.values
    }

    pub(crate) fn map<U>(self, map: impl FnMut(T) -> U) -> Layers<U> {
        Layers::new(self.values.into_iter().map(map).collect())
    }
}
