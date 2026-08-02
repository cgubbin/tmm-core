use crate::{IncidentSide, backend::LayerBoundaryWaves};

use super::{PlaneWaveSolution, PlaneWaveSolutionSource};

pub trait SolutionWorkspace: PlaneWaveSolutionSource {
    fn into_solution(self) -> PlaneWaveSolution<Self::Entries>;
}

pub(crate) trait ReconstructLayerBoundaryWaves: SolutionWorkspace {
    type Algebra;

    fn reconstruct_layer_boundary_waves(
        &self,
        incident_side: IncidentSide,
    ) -> Option<Vec<LayerBoundaryWaves<Self::Algebra>>>;
}
