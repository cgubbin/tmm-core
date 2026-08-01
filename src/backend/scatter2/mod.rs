mod backend;
mod entries;
mod error;
mod workspace;

use crate::{
    ComplexPlane, ComplexScalar, Polarisation, RealAxis,
    algebra::ScalarAlgebra,
    backend::{Backend, PlaneWaveSolution, RunMode, SolutionWorkspace},
    input::CanonicalProblem,
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};
pub(crate) use entries::{Scatter2Entries, Scatter2ExteriorContext};
pub use error::Scatter2Error;
pub(crate) use workspace::Scatter2Workspace;

use nalgebra::ComplexField;
use ndarray::Dimension;
use std::marker::PhantomData;

/// Scalar-channel isotropic 2×2 scattering backend.
#[derive(Copy, Clone, Debug, Default)]
pub struct Scatter2;

impl Scatter2 {
    /// Construct a scattering backend.
    pub const fn new() -> Self {
        Self
    }
}

impl<J, Domain> Backend<J, Domain> for Scatter2
where
    J: ScalarAlgebra + Clone,
    J::Scalar: ComplexScalar,
    J::Dimension: Dimension,
    <J::Scalar as ComplexField>::RealField: Copy,
{
    type Error = Scatter2Error;
    type Entries = Scatter2Entries<J>;
    type Workspace = Scatter2Workspace<J>;

    fn solve<M>(
        &self,
        problem: &CanonicalProblem<M, J>,
        polarisation: Polarisation,
    ) -> Result<PlaneWaveSolution<Self::Entries>, Self::Error>
    where
        Domain: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
        J: ConstitutiveLift<Domain, M>,
    {
        let workspace = self.accumulate::<J, Domain, M>(
            problem.coordinates(),
            problem.stack(),
            polarisation,
            RunMode::ResponseOnly,
        )?;

        Ok(workspace.into_solution())
    }

    fn retain<M>(
        &self,
        problem: &CanonicalProblem<M, J>,
        polarisation: Polarisation,
    ) -> Result<Self::Workspace, Self::Error>
    where
        Domain: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
        J: ConstitutiveLift<Domain, M>,
    {
        let workspace = self.accumulate::<J, Domain, M>(
            problem.coordinates(),
            problem.stack(),
            polarisation,
            RunMode::InternalFields,
        )?;

        Ok(workspace)
    }
}
