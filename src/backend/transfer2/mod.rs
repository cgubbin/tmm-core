//! Isotropic 2×2 transfer-matrix backend.
//!
//! [`Transfer2`] implements:
//!
//! - [`RawMatrixBackend`](crate::backend::RawMatrixBackend);
//! - [`PlaneWaveBackend`](crate::backend::PlaneWaveBackend);
//! - [`OutgoingModeResidualBackend`](crate::backend::OutgoingModeBackend).
//!
//! The backend is suitable for moderate optical thicknesses. For strongly
//! evanescent or optically thick stacks, prefer the scattering-matrix backend.

mod backend;
mod entries;
mod error;
mod workspace;

pub(crate) use entries::Transfer2Entries;
pub use error::Transfer2Error;
pub(crate) use workspace::Transfer2Workspace;

use nalgebra::ComplexField;
use ndarray::Dimension;
use std::marker::PhantomData;

use crate::{
    ComplexScalar, Polarisation,
    algebra::ScalarAlgebra,
    backend::{Backend, PlaneWaveSolution, RunMode, SolutionWorkspace},
    input::CanonicalProblem,
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};

/// Scalar-channel isotropic 2×2 transfer backend.
#[derive(Copy, Clone, Debug, Default)]
pub struct Transfer2<J>(PhantomData<J>);

impl<J> Transfer2<J> {
    /// Construct a transfer backend.
    pub const fn new() -> Self {
        Self(PhantomData)
    }
}

impl<J, Domain> Backend<J, Domain> for Transfer2<J>
where
    J: ScalarAlgebra + Clone,
    J::Scalar: ComplexScalar,
    J::Dimension: Dimension,
    <J::Scalar as ComplexField>::RealField: Copy,
{
    type Error = Transfer2Error;
    type Entries = Transfer2Entries<J>;
    type Workspace = Transfer2Workspace<J>;

    fn solve<M>(
        &self,
        problem: &CanonicalProblem<M, J>,
        polarisation: Polarisation,
    ) -> Result<PlaneWaveSolution<Self::Entries>, Self::Error>
    where
        Domain: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
        J: ConstitutiveLift<Domain, M>,
    {
        let workspace = self.accumulate::<Domain, M>(
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
        let workspace = self.accumulate::<Domain, M>(
            problem.coordinates(),
            problem.stack(),
            polarisation,
            RunMode::InternalFields,
        )?;

        Ok(workspace)
    }
}
