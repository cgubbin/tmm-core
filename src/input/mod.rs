mod canonical;
mod compile;
mod coordinate;
mod error;
mod parameter;
mod plane_wave;

pub(crate) use canonical::{
    CanonicalCoordinates, CanonicalPlaneWaveInput, CanonicalPlaneWaveProblem, CanonicalProblem,
    CanonicalSolverInput,
};
pub(crate) use compile::{
    CompilationPlan, CompileProblemError, ParameterAssignment, SeedJet, compile_problem,
    plan_compilation,
};
pub(crate) use coordinate::{InPlaneCoordinate, PlaneWaveCoordinates, SpectralCoordinate};
pub use error::{PlaneWaveInputError, SpectralTransformError};
pub use parameter::{DerivativeParameter, SolveRequest};
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
