use crate::{
    IncidentSide,
    backend::{IsotropicLayerQuantities, LayerBoundaryWaves},
};

use super::{PlaneWaveEntries, PlaneWaveSolution, PlaneWaveSolutionSource};

pub trait SolutionWorkspace: PlaneWaveSolutionSource {
    fn into_solution(self) -> PlaneWaveSolution<Self::Entries>;
}

pub(crate) trait ReconstructLayerBoundaryWaves {
    type Algebra;

    fn reconstruct_layer_boundary_waves(
        &self,
        incident_side: IncidentSide,
    ) -> Option<Vec<LayerBoundaryWaves<Self::Algebra>>>;
}

pub(crate) trait RetainedIsotropicLayers {
    type Algebra;

    fn retained_layer_count(&self) -> Option<usize>;

    fn layer_quantities(&self, index: usize) -> Option<&IsotropicLayerQuantities<Self::Algebra>>;
}
