mod constitutive;
mod electromagnetic;

pub use constitutive::ConstitutiveFields;
pub use electromagnetic::ElectromagneticFields;


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
