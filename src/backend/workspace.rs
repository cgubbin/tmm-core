use crate::backend::IsotropicLayerQuantities;

use super::{PlaneWaveSolution, PlaneWaveSolutionSource};

pub trait SolutionWorkspace: PlaneWaveSolutionSource {
    fn into_solution(self) -> PlaneWaveSolution<Self::Entries>;
}

pub trait RetainedIsotropicLayers {
    type Algebra;

    fn retained_layer_count(&self) -> Option<usize>;

    fn layer_quantities(&self, index: usize) -> Option<&IsotropicLayerQuantities<Self::Algebra>>;

    fn layer_thickness(&self, index: usize) -> Option<&Self::Algebra>;
}
