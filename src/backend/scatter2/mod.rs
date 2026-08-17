//! Isotropic 2×2 scattering-matrix backend.
//!
//! [`Scatter2`] evaluates scalar TE or TM channels using cascaded two-port
//! scattering matrices. The backend is intended to remain numerically stable
//! for structures where direct transfer-matrix multiplication may become
//! poorly conditioned.
//!
//! Ordinary plane-wave solutions are represented through projective scattering
//! entries, while retained workspaces additionally preserve the intermediate
//! components required for field and modal reconstruction.
//!
//! Exterior longitudinal wavevectors are supplied explicitly through
//! [`ExteriorWavevectors`], allowing complex-plane callers to control exterior
//! branch selection independently of the finite-layer propagation convention.

mod backend;
mod entries;
mod projection;
mod workspace;

use std::convert::Infallible;

use crate::{
    CanonicalCoordinates, ComplexScalar, Polarisation,
    algebra::ScalarAlgebra,
    backend::{Backend, ExteriorWavevectors, PlaneWaveSolution, RunMode, SolutionWorkspace},
    input::CanonicalStack,
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};
pub(crate) use entries::{Scatter2Entries, Scatter2ExteriorContext};
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
    J: ScalarAlgebra,
    J::Scalar: ComplexScalar,
    J::Dimension: Dimension,
    <J::Scalar as ComplexField>::RealField: Copy,
{
    type Error = Infallible;
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
            exterior,
            polarisation,
            RunMode::ResponseOnly,
        );

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
        );

        Ok(workspace)
    }
}
