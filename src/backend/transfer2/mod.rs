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

/// Validation policy for non-finite transfer-matrix entries.
///
/// These checks detect overflow and invalid arithmetic. They do not detect all
/// forms of ill-conditioning or loss of precision.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransferStabilityCheck {
    /// Check only the completed transfer matrix.
    ///
    /// This adds one scan over the four final matrix entries and is the default.
    #[default]
    Final,

    /// Check each layer matrix and each intermediate accumulated matrix.
    ///
    /// This is useful for diagnostics but adds two scans per finite layer.
    PerLayer,

    /// Perform no explicit finiteness checks.
    Disabled,
}

/// Scalar-channel isotropic 2×2 transfer backend.
///
/// The transfer formulation may overflow for optically thick, strongly
/// absorbing, or strongly evanescent layers. The configured stability check
/// can detect non-finite matrix entries but cannot guarantee good conditioning.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Transfer2 {
    stability_check: TransferStabilityCheck,
}

impl Transfer2 {
    /// Construct a transfer backend using the default final-matrix check.
    pub const fn new() -> Self {
        Self {
            stability_check: TransferStabilityCheck::Final,
        }
    }

    /// Construct a transfer backend with the supplied stability policy.
    pub const fn with_stability_check(stability_check: TransferStabilityCheck) -> Self {
        Self { stability_check }
    }

    /// Return the configured stability policy.
    pub const fn stability_check(&self) -> TransferStabilityCheck {
        self.stability_check
    }
}

impl Default for Transfer2 {
    fn default() -> Self {
        Self::new()
    }
}

impl<J, Domain> Backend<J, Domain> for Transfer2
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
