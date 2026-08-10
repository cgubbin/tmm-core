//! Interface-resolved observables.
//!
//! A planar stack containing `N` finite layers has `N + 1` physical
//! interfaces:
//!
//! ```text
//! left exterior | layer 0 | ... | layer N - 1 | right exterior
//! ```
//!
//! Every interface is represented by quantities immediately on its left and
//! right. Both sides are retained even when continuity requires them to agree:
//! this preserves the provenance of each reconstruction and permits direct
//! continuity diagnostics.
//!
//! The module separates three stages:
//!
//! 1. [`InterfaceWaveData`] stores the directional waves and characteristic
//!    admittance associated with each interface side.
//! 2. [`InterfaceStates`] stores the corresponding canonical states.
//! 3. [`InterfacePower`] stores normalized signed power flux on each side.
//!
//! `InterfaceWaveData` is an internal projection type. Public callers receive
//! state or power observables through the retained evaluator.

mod power;
mod project;
mod state;
mod wave_data;

pub use power::{DirectedPower, InterfacePower};
pub use project::InterfaceProjectionError;
pub use state::InterfaceStates;

pub(crate) use project::{
    assemble_interface_wave_data, exterior_boundary_states, exterior_boundary_waves,
    project_layer_admittances,
};
pub(crate) use state::ExteriorBoundaryStates;
pub(crate) use wave_data::{ExteriorBoundaryWaves, InterfaceSide, InterfaceWaveData};

use crate::algebra::ScaleBy;

/// Interface-resolved quantities in physical left-to-right order.
///
/// A stack containing `N` finite layers has `N + 1` interfaces. For an empty
/// finite stack, this collection contains one interface between the two
/// exterior media.
#[derive(Clone, Debug, PartialEq)]
pub struct Interfaces<T> {
    values: Vec<T>,
}

impl<T> Interfaces<T> {
    pub(crate) fn new(values: Vec<T>) -> Self {
        Self { values }
    }

    /// Return the number of physical interfaces.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Return whether the interface sequence is empty.
    ///
    /// A valid assembled planar stack normally contains at least one
    /// interface, including a stack with no finite layers.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Return the interface at `index`.
    pub fn get(&self, index: usize) -> Option<&T> {
        self.values.get(index)
    }

    /// Return the leftmost physical interface.
    pub fn first(&self) -> Option<&T> {
        self.values.first()
    }

    /// Return the rightmost physical interface.
    pub fn last(&self) -> Option<&T> {
        self.values.last()
    }

    /// Iterate over interfaces in physical left-to-right order.
    pub fn iter(&self) -> impl ExactSizeIterator<Item = &T> {
        self.values.iter()
    }

    pub(crate) fn into_inner(self) -> Vec<T> {
        self.values
    }

    pub(crate) fn map<U>(self, map: impl FnMut(T) -> U) -> Interfaces<U> {
        Interfaces::new(self.values.into_iter().map(map).collect())
    }
}

impl<S, T> ScaleBy<S> for Interfaces<T>
where
    T: ScaleBy<S>,
{
    fn scale_by(self, scale: &S) -> Self {
        Self {
            values: self
                .values
                .into_iter()
                .map(|each| each.scale_by(scale))
                .collect(),
        }
    }
}
