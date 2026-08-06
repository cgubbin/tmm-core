use crate::{
    IncidentSide,
    backend::{IsotropicLayerQuantities, LayerBoundaryWaves},
    evaluate::ModeReconstructionError,
};

use super::{PlaneWaveSolution, PlaneWaveSolutionSource};

pub trait SolutionWorkspace: PlaneWaveSolutionSource {
    fn into_solution(self) -> PlaneWaveSolution<Self::Entries>;
}

pub trait ReconstructLayerBoundaryWaves {
    type Algebra;

    fn reconstruct_layer_boundary_waves(
        &self,
        incident_side: IncidentSide,
    ) -> Option<Vec<LayerBoundaryWaves<Self::Algebra>>>;
}

pub trait RetainedIsotropicLayers {
    type Algebra;

    fn retained_layer_count(&self) -> Option<usize>;

    fn layer_quantities(&self, index: usize) -> Option<&IsotropicLayerQuantities<Self::Algebra>>;

    fn layer_thickness(&self, index: usize) -> Option<&Self::Algebra>;
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
    ) -> Result<Vec<LayerBoundaryWaves<Self::Algebra>>, ModeReconstructionError>;
}
