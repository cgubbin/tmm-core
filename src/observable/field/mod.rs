mod constitutive;
mod dissipation;
mod electromagnetic;
mod energy;
mod material;
mod reconstruction;

pub use constitutive::ConstitutiveFields;
pub use dissipation::ElectromagneticDissipation;
pub use electromagnetic::{ElectromagneticFields, ElectromagneticIntensities};
pub use energy::ElectromagneticEnergy;
pub use material::ConstitutiveFieldReconstructionError;

pub use reconstruction::FieldReconstructionError;

pub(crate) use constitutive::{
    IsotropicConstitutiveParameters, IsotropicConstitutiveSpectralData,
    electromagnetic_dissipation_coefficients,
};
pub(crate) use material::{ConstitutiveSamplingContext, ConstitutiveSamplingError};
pub(crate) use reconstruction::FieldSamplingContext;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum FieldIndexError {
    #[error("expected {expected} excitation indices, received {actual}")]
    ExcitationRankMismatch { expected: usize, actual: usize },

    #[error(
        "excitation index {index} is out of bounds for axis {axis}, \
         which has length {length}"
    )]
    ExcitationIndexOutOfBounds {
        axis: usize,
        index: usize,
        length: usize,
    },

    #[error("spatial index {index} is out of bounds for an axis of length {length}")]
    SpatialIndexOutOfBounds { index: usize, length: usize },

    #[error("sampled field component does not have a final spatial axis")]
    MissingSpatialAxis,

    #[error("sampled field components do not have matching shapes")]
    ComponentShapeMismatch,

    #[error("requires static dimensional input")]
    DynamicInput,
}
