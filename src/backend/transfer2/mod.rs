//! Isotropic 2×2 transfer-matrix backend.
//!
//! [`Transfer2`] evaluates scalar TE or TM channels using ordinary 2×2
//! transfer matrices.
//!
//! The backend operates on canonical coordinates and a canonical stack.
//! Exterior longitudinal wavevectors are supplied explicitly through
//! [`ExteriorWavevectors`](crate::backend::ExteriorWavevectors), allowing
//! complex-plane callers to control exterior branch selection independently
//! of finite-layer propagation.
//!
//! Transfer matrices are compact and convenient for moderate optical
//! thicknesses, but may overflow or become poorly conditioned for strongly
//! evanescent, absorbing, or optically thick stacks. In those regimes,
//! [`Scatter2`](crate::backend::Scatter2) is generally preferable.
//!
//! Optional stability checks detect non-finite matrix entries but do not
//! diagnose all forms of ill-conditioning or loss of precision.

mod backend;
mod entries;
mod error;
mod projection;
mod state;
mod workspace;

pub(crate) use entries::{Transfer2Entries, Transfer2ExteriorContext};
pub use error::Transfer2Error;

#[cfg(test)]
pub(crate) use state::{
    TransferState, bidirectional_waves_from_state, right_outgoing_transfer_state,
    transfer_state_from_waves, transfer_state_slope,
};

#[cfg(test)]
pub(crate) use workspace::right_exterior_waves;

pub(crate) use workspace::{RetainedTransferLayer, RetainedTransferLayers, Transfer2Workspace};

use nalgebra::ComplexField;
use ndarray::Dimension;

use crate::{
    CanonicalCoordinates, ComplexScalar, Polarisation,
    algebra::ScalarAlgebra,
    backend::{Backend, ExteriorWavevectors, PlaneWaveSolution, RunMode, SolutionWorkspace},
    input::CanonicalStack,
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};

/// Validation policy for non-finite transfer-matrix entries.
///
/// These checks detect overflow and invalid arithmetic. They do not detect all
/// forms of ill-conditioning or loss of precision.
///
/// Stability checks inspect the physical matrix values only. Derivative
/// components carried by jet algebras are not included in these checks.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TransferStabilityCheck {
    /// Check only the completed transfer matrix.
    ///
    /// This adds one scan over the four final matrix entries and is the default.
    #[default]
    Final,

    /// Check every finite-layer matrix and every accumulated transfer matrix.
    ///
    /// This is useful for locating the point at which overflow or invalid
    /// arithmetic first appears, but adds two finiteness scans per finite layer.
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
            exterior,
            polarisation,
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
            exterior,
            polarisation,
            RunMode::InternalFields,
        )?;

        Ok(workspace)
    }
}
