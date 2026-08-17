//! Numerical backend interfaces and implementations.
//!
//! Backends consume canonical coordinates and canonical stacks and propagate
//! the selected jet algebra through transfer- or scattering-matrix
//! calculations.
//!
//!
//!
//! This module contains:
//!
//! - backend-independent solution and workspace contracts;
//! - exterior-wavevector and isotropic-layer quantities;
//! - isotropic 2×2 transfer- and scattering-matrix backends;
//! - mode-reconstruction support shared by those backends.
//!
//! Callers normally select a concrete backend such as [`Scatter2`] or
//! [`Transfer2`] through the evaluator API rather than invoking backend
//! methods directly.

use crate::{
    CanonicalCoordinates, ComplexScalar, Polarisation,
    algebra::Jet,
    input::CanonicalStack,
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};

use ndarray::Dimension;

mod exterior;
mod isotropic;
mod mode;
pub(crate) mod scatter2;
mod solution;
pub(crate) mod transfer2;
mod workspace;

pub use exterior::ExteriorWavevectors;
pub(crate) use exterior::evaluate_exterior_wavevectors;
pub(crate) use isotropic::IsotropicLayerQuantities;
pub(crate) use mode::{
    ModalSolutionSource, ModeReconstructionError, PlaneWaveModeCandidate,
    ReconstructExteriorModeWaves, ReconstructLayerModeWaves,
};
pub use scatter2::Scatter2;
pub(crate) use solution::{
    ExteriorContextProvider, PlaneWaveEntries, PlaneWaveSolution, PlaneWaveSolutionSource,
    PlaneWaveSolutionView,
};
pub use transfer2::{Transfer2, Transfer2Error};
pub(crate) use workspace::{RetainedIsotropicLayers, SolutionWorkspace};

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

/// Numerical backend for a canonical plane-wave problem.
///
/// `J` determines the value/derivative algebra propagated through the solve,
/// while `Domain` determines how constitutive material data are evaluated
/// (for example on the real spectral axis or in the complex plane).
///
/// Backends receive the canonical problem as two borrowed components:
///
/// - canonical spectral and in-plane coordinates, which may vary between
///   solves;
/// - a canonical stack, which may be compiled once and reused.
///
/// Exterior longitudinal wavevectors are supplied separately so callers such
/// as mode solvers may control their complex-plane branch choices.
///
/// This trait is public because it appears in bounds on public evaluator
/// implementations. Backend implementations are supplied by `lamina-core`;
/// downstream implementation is not currently a supported extension point.
#[doc(hidden)]
pub trait Backend<J, Domain>
where
    J: Jet,
    J::Scalar: ComplexScalar,
    J::Dimension: Dimension,
{
    /// Backend-specific entries stored in a completed plane-wave solution.
    type Entries: PlaneWaveEntries;

    /// Retained backend state used for field and mode reconstruction.
    type Workspace: SolutionWorkspace<Entries = Self::Entries>;

    /// Backend failure type.
    type Error: std::error::Error + Send + Sync + 'static;

    /// Solve the problem and return only the completed response state.
    ///
    /// No intermediate layer state is retained.
    fn solve<M>(
        &self,
        coordinates: &CanonicalCoordinates<J>,
        stack: &CanonicalStack<M, J>,
        exterior: &ExteriorWavevectors<J>,
        polarisation: Polarisation,
    ) -> Result<PlaneWaveSolution<Self::Entries>, Self::Error>
    where
        Domain: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
        J: ConstitutiveLift<Domain, M>;

    /// Solve the problem while retaining intermediate state required for
    /// subsequent field or mode reconstruction.
    fn retain<M>(
        &self,
        coordinates: &CanonicalCoordinates<J>,
        stack: &CanonicalStack<M, J>,
        exterior: &ExteriorWavevectors<J>,
        polarisation: Polarisation,
    ) -> Result<Self::Workspace, Self::Error>
    where
        Domain: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
        J: ConstitutiveLift<Domain, M>;
}
