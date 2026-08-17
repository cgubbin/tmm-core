use thiserror::Error;

use crate::{
    algebra::ScalarAlgebra,
    observable::BoundaryState,
    waves::{ExteriorBoundaryWaves, LayerBoundaryWaves},
};

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ModeReconstructionError {
    #[error("workspace does not retain the data required for modal reconstruction")]
    ModeDataNotRetained,

    #[error("the outgoing boundary system has no usable modal null vector")]
    NoUsableNullVector,

    #[error("the outgoing boundary system has a degenerate modal null space")]
    DegenerateNullSpace,

    #[error(
        "reconstructed modal boundary-wave count {wave_count} does not match \
         retained finite-layer count {layer_count}"
    )]
    LayerCountMismatch {
        wave_count: usize,
        layer_count: usize,
    },
}

/// Gauge used to select a representative vector from the one-dimensional
/// null space of the outgoing boundary system.
///
/// Different gauges represent the same physical mode up to an arbitrary
/// nonzero complex scale.
#[doc(hidden)]
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ModalGauge {
    FirstAdjugateColumn,
    SecondAdjugateColumn,
}

/// Unnormalised candidate solution of the outgoing homogeneous boundary
/// problem.
///
/// A candidate contains:
///
/// - a boundary state representing one chosen null-space gauge;
/// - the corresponding right-exterior outgoing amplitude;
/// - a projective residual measuring failure to satisfy the homogeneous
///   boundary condition exactly.
///
/// The overall complex scale is arbitrary. Physical modal normalisation is
/// applied only after reconstruction.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct PlaneWaveModeCandidate<A> {
    state: BoundaryState<A>,
    right_outgoing: A,
    residual: A,
}

impl<A> PlaneWaveModeCandidate<A> {
    pub(crate) fn new(state: BoundaryState<A>, right_outgoing: A, residual: A) -> Self {
        Self {
            state,
            right_outgoing,
            residual,
        }
    }

    pub(crate) fn state(&self) -> &BoundaryState<A> {
        &self.state
    }

    pub(crate) fn right_outgoing(&self) -> &A {
        &self.right_outgoing
    }

    pub(crate) fn residual(&self) -> &A {
        &self.residual
    }

    pub(crate) fn into_state(self) -> BoundaryState<A> {
        self.state
    }

    pub(crate) fn into_residual(self) -> A {
        self.residual
    }

    pub(crate) fn scaled(mut self, factor: &A) -> Self
    where
        A: ScalarAlgebra,
    {
        self.right_outgoing = self.right_outgoing.multiply(factor);
        self.residual = self.residual.multiply(factor);
        self.state = self.state.scaled(factor);

        self
    }
}

/// Reconstruct exterior directional waves for an outgoing homogeneous mode.
///
/// The mode has no incident side. The returned exterior waves must satisfy
/// outgoing boundary conditions on both sides:
///
/// - no right-going incident wave in the left exterior;
/// - no left-going incident wave in the right exterior.
///
/// The reconstructed mode retains the arbitrary complex amplitude carried by
/// `seed`. Modal normalization is applied later.
pub trait ReconstructExteriorModeWaves {
    type Algebra;

    fn reconstruct_exterior_mode_waves(
        &self,
        seed: &PlaneWaveModeCandidate<Self::Algebra>,
    ) -> Result<ExteriorBoundaryWaves<Self::Algebra>, ModeReconstructionError>;
}

/// Reconstruct directional waves in every finite layer for an outgoing
/// homogeneous modal solution.
///
/// The returned vector contains one entry per retained finite layer in
/// physical left-to-right order.
///
/// The reconstruction preserves the arbitrary complex scale carried by
/// `seed`; modal normalisation is applied later.
pub trait ReconstructLayerModeWaves {
    type Algebra;

    fn reconstruct_layer_mode_waves(
        &self,
        seed: &PlaneWaveModeCandidate<Self::Algebra>,
    ) -> Result<Vec<LayerBoundaryWaves<Self::Algebra>>, ModeReconstructionError>;
}

/// Source of an unnormalised outgoing homogeneous boundary solution.
///
/// Implemented by retained backend workspaces that contain sufficient data to
/// construct a representative vector in the outgoing boundary-system null
/// space.
pub trait ModalSolutionSource {
    type Algebra;

    /// Construct an arbitrary nonzero representative of the outgoing
    /// homogeneous boundary solution.
    fn modal_boundary_solution(
        &self,
    ) -> Result<PlaneWaveModeCandidate<Self::Algebra>, ModeReconstructionError>;
}
