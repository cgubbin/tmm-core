use crate::{
    ComplexScalar, PlaneWaveObservables, Polarisation, RealAxis,
    algebra::Jet,
    input::{CanonicalBackendInput, CanonicalProblem, CanonicalSolverInput},
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};

use ndarray::Dimension;

mod isotropic;
mod scatter2;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum RunMode {
    Evaluate,
    Accumulate,
}

impl RunMode {
    fn is_requested(&self) -> bool {
        *self == RunMode::Accumulate
    }
}

pub(crate) trait Backend<J, Domain>
where
    J: Jet,
    J::Scalar: ComplexScalar,
    J::Dimension: Dimension,
{
    type Entries;
    type Workspace: BackendWorkspace<Entries = Self::Entries>;
    type Error;

    fn solve<M>(
        &self,
        problem: &CanonicalProblem<M, J>,
        polarisation: Polarisation,
    ) -> Result<Self::Entries, Self::Error>
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

pub trait BackendWorkspace {
    type Entries;

    fn entries(&self) -> &Self::Entries;

    fn into_entries(self) -> Self::Entries;
}
