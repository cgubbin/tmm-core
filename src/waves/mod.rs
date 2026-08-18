//! Backend-independent travelling-wave reconstruction.
//!
//! This module contains:
//!
//! - directional wave-amplitude containers;
//! - exterior and finite-layer boundary-wave reconstruction;
//! - spatial propagation of reconstructed amplitudes;
//! - sampling of retained solutions at canonical field positions.
//!
//! Backends are responsible only for reconstructing waves at reference
//! boundaries. Propagation and spatial sampling are backend-independent.

mod boundary;
mod propagation;
mod sampling;

pub(crate) use boundary::{
    BidirectionalWaves, BoundaryWaveSolution, ExteriorBoundaryWaves, LayerBoundaryWaves,
};
use ndarray::Dimension;
pub(crate) use propagation::{PropagateLayerWaves, PropagateWaves};
pub(crate) use sampling::{WaveSamplingContext, WaveSamplingError};

use crate::{
    ComplexScalar, IncidentSide,
    algebra::{Jet, ScalarAlgebra},
    backend::PlaneWaveSolutionSource,
    observable::{Amplitudes, ProjectAmplitudes},
};

/// Reconstruction of driven exterior directional waves.
///
/// Exterior waves depend only on the projected reflection/transmission
/// amplitudes and incident side, so this capability is provided automatically
/// for every compatible plane-wave solution source.
pub trait ReconstructExteriorBoundaryWaves: PlaneWaveSolutionSource
where
    Self::Entries: ProjectAmplitudes,
    <Self::Entries as ProjectAmplitudes>::Amplitudes: Amplitudes<Algebra = Self::Algebra>,
    Self::Algebra: ScalarAlgebra + Clone,
    <Self::Algebra as Jet>::Scalar: ComplexScalar,
    <Self::Algebra as Jet>::Dimension: Dimension,
{
    type Algebra;

    fn reconstruct_exterior_boundary_waves(
        &self,
        incident_side: IncidentSide,
    ) -> ExteriorBoundaryWaves<Self::Algebra> {
        let amplitudes = self.solution().amplitudes(incident_side);

        ExteriorBoundaryWaves::from_amplitudes(
            amplitudes.reflection(),
            amplitudes.transmission(),
            incident_side,
        )
    }
}

impl<T, A> ReconstructExteriorBoundaryWaves for T
where
    T: PlaneWaveSolutionSource,
    T::Entries: ProjectAmplitudes,
    <T::Entries as ProjectAmplitudes>::Amplitudes: Amplitudes<Algebra = A>,
    A: ScalarAlgebra + Clone,
    <A as Jet>::Scalar: ComplexScalar,
    <A as Jet>::Dimension: Dimension,
{
    type Algebra = A;
}

/// Reconstruction of driven directional waves at every finite-layer boundary.
///
/// Returns `None` when the underlying workspace did not retain the internal
/// state required for reconstruction.
pub trait ReconstructLayerBoundaryWaves {
    type Algebra;

    fn reconstruct_layer_boundary_waves(
        &self,
        incident_side: IncidentSide,
    ) -> Option<Vec<LayerBoundaryWaves<Self::Algebra>>>;
}
