use thiserror::Error;

use crate::{observable::BoundaryState, waves::LayerBoundaryWaves};

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ModeReconstructionError {
    #[error("workspace does not retain the data required for modal reconstruction")]
    ModeDataNotRetained,

    #[error("the outgoing boundary system has no usable modal null vector")]
    NoModalNullVector,

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

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ModalGauge {
    FirstAdjugateColumn,
    SecondAdjugateColumn,
}

#[derive(Clone, Debug)]
pub(crate) struct PlaneWaveModeCandidate<A> {
    state: BoundaryState<A>,
    residual: A,
}

impl<A> PlaneWaveModeCandidate<A> {
    pub(crate) fn new(state: BoundaryState<A>, residual: A) -> Self {
        Self { state, residual }
    }

    pub(crate) fn state(&self) -> &BoundaryState<A> {
        &self.state
    }

    pub(crate) fn projective_residual(&self) -> &A {
        &self.residual
    }

    pub(crate) fn into_state(self) -> BoundaryState<A> {
        self.state
    }

    pub(crate) fn into_projective_residual(self) -> A {
        self.residual
    }
}

/// Reconstruct directional waves for an outgoing homogeneous modal solution.
///
/// Unlike [`ReconstructLayerBoundaryWaves`], this operation has no incident
/// side. The workspace must construct a nonzero solution of the outgoing
/// homogeneous boundary system and propagate it through all finite layers.
///
/// The reconstructed mode retains an arbitrary complex amplitude. Modal
/// normalization is applied later.
pub(crate) trait ReconstructLayerModeWaves {
    type Algebra;

    fn reconstruct_layer_mode_waves(
        &self,
        seed: &PlaneWaveModeCandidate<Self::Algebra>,
    ) -> Result<Vec<LayerBoundaryWaves<Self::Algebra>>, ModeReconstructionError>;
}

pub(crate) trait ModalSolutionSource {
    type Algebra;

    /// Construct an arbitrary nonzero outgoing homogeneous boundary solution
    fn modal_boundary_solution(
        &self,
    ) -> Result<PlaneWaveModeCandidate<Self::Algebra>, ModeReconstructionError>;
}
