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
//! representation used internally by the numerical backends. Coordinate
//! transforms are performed through the selected jet algebra so that
//! derivatives remain with respect to the caller-supplied coordinates.
//!
//! The public types in this module describe a problem; they do not perform
//! coordinate conversion or numerical evaluation themselves.

pub(crate) mod canonical;
pub(crate) mod compile;
mod coordinate;
mod coordinate_input;
mod error;

pub(crate) use canonical::{CanonicalBackendInput, CanonicalProblem};
pub(crate) use compile::{
    CompilationContext, CompileJet, CompilePlaneWaveError, CoordinateContext, JetMapping,
    ProjectionConstraint, ProjectionConstraintError, ValidationConfig, compile_real,
};
pub(crate) use coordinate::ReferenceRequirement;
pub(crate) use coordinate_input::CoordinateValues;

pub use canonical::{CanonicalCoordinates, CanonicalStack};
pub use compile::{StackCompileError, StackThicknessJet, compile_canonical_constant_stack};
pub use coordinate::{Coordinates, InPlaneCoordinate, SpectralCoordinate};
pub use coordinate_input::{
    CoordinateGrid, CoordinateInput, CoordinateReference, CoordinateSamples,
};
pub use error::PlaneWaveInputError;

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
///
/// Finite layers are ordered from left to right. [`Left`](Self::Left)
/// therefore propagates initially in the positive stack direction, while
/// [`Right`](Self::Right) propagates initially in the negative stack
/// direction.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum IncidentSide {
    /// Incidence from the first exterior medium towards the final exterior
    /// medium.
    Left,

    /// Incidence from the final exterior medium towards the first exterior
    /// medium.
    Right,
}
