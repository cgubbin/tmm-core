use crate::{
    ComplexScalar, PlaneWaveObservables, Polarisation, RealAxis,
    algebra::Jet,
    backend::solution::PlaneWaveSolutionView,
    input::{CanonicalBackendInput, CanonicalProblem, CanonicalSolverInput},
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};

use ndarray::Dimension;

mod isotropic;
mod plane_wave;
mod scatter2;
mod solution;
mod waves;

pub(crate) use solution::{PlaneWaveEntries, PlaneWaveSolution};
pub(crate) use waves::{BidirectionalWaves, ExteriorBoundaryWaves, LayerBoundaryWaves};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum RunMode {
    ResponseOnly,
    InternalFields,
}

impl RunMode {
    fn is_requested(&self) -> bool {
        *self == RunMode::InternalFields
    }
}

pub(crate) trait Backend<J, Domain>
where
    J: Jet,
    J::Scalar: ComplexScalar,
    J::Dimension: Dimension,
{
    type Entries: PlaneWaveEntries;
    type Workspace: SolutionWorkspace<Entries = Self::Entries>;
    type Error;

    fn solve<M>(
        &self,
        problem: &CanonicalProblem<M, J>,
        polarisation: Polarisation,
    ) -> Result<PlaneWaveSolution<Self::Entries>, Self::Error>
    where
        Domain: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
        J: ConstitutiveLift<Domain, M>;

    fn retain<M>(
        &self,
        problem: &CanonicalProblem<M, J>,
        polarisation: Polarisation,
    ) -> Result<Self::Workspace, Self::Error>
    where
        Domain: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
        J: ConstitutiveLift<Domain, M>;
}

pub trait SolutionWorkspace {
    type Entries: PlaneWaveEntries;

    fn solution(&self) -> PlaneWaveSolutionView<'_, Self::Entries>;

    fn into_solution(self) -> PlaneWaveSolution<Self::Entries>;
}
