use crate::{
    ComplexScalar, Polarisation,
    algebra::Jet,
    input::CanonicalProblem,
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};

use ndarray::Dimension;

mod isotropic;
pub(crate) mod scatter2;
mod solution;
pub(crate) mod transfer2;
mod waves;
mod workspace;

pub(crate) use isotropic::IsotropicLayerQuantities;
pub use scatter2::{Scatter2, Scatter2Error};
pub(crate) use solution::{
    ExteriorAdmittanceProvider, PlaneWaveEntries, PlaneWaveSolution, PlaneWaveSolutionSource,
    PlaneWaveSolutionView,
};
pub use transfer2::{Transfer2, Transfer2Error};
pub(crate) use waves::{BidirectionalWaves, LayerBoundaryWaves};
pub(crate) use workspace::{
    ReconstructLayerBoundaryWaves, RetainedIsotropicLayers, SolutionWorkspace,
};

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
    type Error: std::fmt::Debug;

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
