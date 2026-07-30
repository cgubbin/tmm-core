//! Caller-facing descriptions of planar-wave solve problems.
//!
//! This module defines:
//!
//! - the physical coordinates used to parameterise a solve;
//! - the sampled coordinate values supplied by the caller;
//! - the incidence direction and polarisation;
//! - the numerical value and derivative information requested from the
//!   evaluator.
//!
//! Inputs are expressed in caller-selected units and coordinate systems.
//! Before evaluation, they are validated and compiled into the canonical
//! coordinates used by the numerical backends.
//!
//! The public types in this module describe a problem; they do not perform
//! coordinate conversion or numerical evaluation themselves.

mod canonical;
mod compile;
mod coordinate;
mod error;
mod parameter;
mod plane_wave;

pub(crate) use canonical::{
    CanonicalBackendInput, CanonicalCoordinates, CanonicalLayer, CanonicalProblem,
    CanonicalSolverInput, CanonicalStack,
};
pub(crate) use compile::{
    CompilationContext, CompileJet, CompilePlaneWaveError, ParameterAssignment,
    ParameterAssignmentError, SeedJet, compile_complex, compile_real,
};
pub(crate) use coordinate::{InPlaneCoordinate, PlaneWaveCoordinates, SpectralCoordinate};
pub use error::{PlaneWaveInputError, SpectralTransformError};
pub use parameter::SolveRequest;
pub(crate) use parameter::{Parameter, ThicknessSeedError};
pub use plane_wave::{PlaneWaveInput, PlaneWavePoint};

/// Polarisation supported by isotropic planar backends.
///
/// In an isotropic stratified system, transverse-electric and
/// transverse-magnetic fields decouple into independent scalar problems.
///
/// An anisotropic backend may require a different input representation because
/// this TE/TM decomposition does not generally remain valid.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Polarisation {
    /// Transverse-electric polarisation.
    ///
    /// The electric field is perpendicular to the plane of incidence.
    TransverseElectric,

    /// Transverse-magnetic polarisation.
    ///
    /// The magnetic field is perpendicular to the plane of incidence.
    TransverseMagnetic,
}

/// Side from which a plane wave is incident on a planar stack.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum IncidentSide {
    /// Incidence from the first exterior medium towards the final exterior
    /// medium.
    Left,

    /// Incidence from the final exterior medium towards the first exterior
    /// medium.
    Right,
}
