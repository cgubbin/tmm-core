use crate::{
    ComplexScalar, Polarisation, RealAxis,
    algebra::Jet,
    input::{CanonicalBackendInput, CanonicalProblem, CanonicalSolverInput},
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};

use ndarray::Dimension;

mod isotropic;
mod plane_wave;
mod scatter2;
mod solution;
mod transfer2;
mod waves;

pub use scatter2::{Scatter2, Scatter2Error};
pub(crate) use scatter2::{Scatter2Entries, Scatter2ExteriorContext};
pub(crate) use solution::{PlaneWaveEntries, PlaneWaveSolution, PlaneWaveSolutionView};
pub(crate) use waves::{BidirectionalWaves, ExteriorBoundaryWaves, LayerBoundaryWaves};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum RunMode {
    ResponseOnly,
    InternalFields,
}

impl RunMode {
    fn is_requested(&self) -> bool {
        *self == RunMode::InternalFields
    }
}

#[doc(hidden)]
pub trait Backend<J, Domain>
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

pub trait PlaneWaveSolutionSource {
    type Entries: PlaneWaveEntries;

    fn solution(&self) -> PlaneWaveSolutionView<'_, Self::Entries>;
}

pub trait SolutionWorkspace: PlaneWaveSolutionSource {
    fn into_solution(self) -> PlaneWaveSolution<Self::Entries>;
}

impl<E: PlaneWaveEntries> PlaneWaveSolutionSource for PlaneWaveSolution<E> {
    type Entries = E;

    fn solution(&self) -> PlaneWaveSolutionView<'_, Self::Entries> {
        self.as_view()
    }
}
