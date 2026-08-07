mod boundary;
mod propagation;
mod sampling;

pub(crate) use boundary::{BidirectionalWaves, ExteriorBoundaryWaves, LayerBoundaryWaves};
pub(crate) use propagation::{PropagateLayerWaves, PropagateWaves};

use crate::IncidentSide;

pub trait ReconstructLayerBoundaryWaves {
    type Algebra;

    fn reconstruct_layer_boundary_waves(
        &self,
        incident_side: IncidentSide,
    ) -> Option<Vec<LayerBoundaryWaves<Self::Algebra>>>;
}
