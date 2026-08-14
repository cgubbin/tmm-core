mod backend;
mod entries;
mod error;
mod projection;
mod workspace;

use crate::{
    CanonicalCoordinates, ComplexScalar, Polarisation,
    algebra::ScalarAlgebra,
    backend::{Backend, ExteriorWavevectors, PlaneWaveSolution, RunMode, SolutionWorkspace},
    input::CanonicalStack,
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};
pub(crate) use entries::{Scatter2Entries, Scatter2ExteriorContext};
pub use error::Scatter2Error;
pub(crate) use workspace::{RetainedScatterComponents, Scatter2Workspace};

pub(crate) use projection::{Scatter2ProjectiveEntries, cascade_projection};

use nalgebra::ComplexField;
use ndarray::Dimension;

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
    type Entries = Scatter2ProjectiveEntries<J>;
    type Workspace = Scatter2Workspace<J>;

    fn solve<M>(
        &self,
        coordinates: &CanonicalCoordinates<J>,
        stack: &CanonicalStack<M, J>,
        exterior: &ExteriorWavevectors<J>,
        polarisation: Polarisation,
    ) -> Result<PlaneWaveSolution<Self::Entries>, Self::Error>
    where
        Domain: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
        J: ConstitutiveLift<Domain, M>,
    {
        let workspace = self.accumulate::<J, Domain, M>(
            coordinates,
            stack,
            polarisation,
            exterior,
            RunMode::ResponseOnly,
        )?;

        Ok(workspace.into_solution())
    }

    fn retain<M>(
        &self,
        coordinates: &CanonicalCoordinates<J>,
        stack: &CanonicalStack<M, J>,
        exterior: &ExteriorWavevectors<J>,
        polarisation: Polarisation,
    ) -> Result<Self::Workspace, Self::Error>
    where
        Domain: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
        J: ConstitutiveLift<Domain, M>,
    {
        let workspace = self.accumulate::<J, Domain, M>(
            coordinates,
            stack,
            polarisation,
            exterior,
            RunMode::InternalFields,
        )?;

        Ok(workspace)
    }
}
