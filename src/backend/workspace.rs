//! Backend-independent retained-workspace capabilities.
//!
//! Numerical backends may retain intermediate state required for field,
//! layer, and modal reconstruction. These traits expose only the common
//! capabilities needed by backend-independent evaluator and observable code.

use crate::backend::IsotropicLayerQuantities;

use super::{PlaneWaveSolution, PlaneWaveSolutionSource};

/// A retained backend workspace that can be consumed into its completed
/// plane-wave solution.
pub trait SolutionWorkspace: PlaneWaveSolutionSource {
    /// Consume the retained workspace, discarding reconstruction state and
    /// returning the completed solution.
    fn into_solution(self) -> PlaneWaveSolution<Self::Entries>;
}

/// Access to retained finite-layer material and geometric quantities.
///
/// `None` from [`Self::retained_layer_count`] indicates that the workspace was
/// evaluated without layer retention. When retention is available, valid
/// indices are `0..retained_layer_count`.
pub trait RetainedIsotropicLayers {
    type Algebra;

    /// Return the number of retained finite layers, or `None` if layer data
    /// were not retained.
    fn retained_layer_count(&self) -> Option<usize>;

    /// Return the evaluated quantities for finite layer `index`.
    fn layer_quantities(&self, index: usize) -> Option<&IsotropicLayerQuantities<Self::Algebra>>;

    /// Return the canonical thickness jet for finite layer `index`.
    fn layer_thickness(&self, index: usize) -> Option<&Self::Algebra>;
}
