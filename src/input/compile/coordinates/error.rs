use thiserror::Error;

use crate::input::compile::{
    coordinates::{
        CoordinateVariable,
        in_plane::{CanonicaliseInPlaneError, InPlaneInputError},
        spectral::SpectralInputError,
    },
    seed::UnsupportedDerivativeSlot,
};

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum CoordinateAssignmentError {
    #[error(
        "{variable:?} is assigned to jet slot {slot}, \
         but the selected jet algebra provides only \
         {available_slots} slot(s)"
    )]
    SlotOutOfBounds {
        variable: CoordinateVariable,
        slot: usize,
        available_slots: usize,
    },

    #[error(
        "{first:?} and {second:?} are both assigned \
         to jet slot {slot}"
    )]
    DuplicateSlot {
        first: CoordinateVariable,
        second: CoordinateVariable,
        slot: usize,
    },

    #[error("coordinate variable {variable:?} was assigned more than once")]
    DuplicateVariable { variable: CoordinateVariable },
}

#[derive(Debug, Error)]
pub enum CoordinateCompileError<R> {
    #[error(transparent)]
    Assignment(#[from] CoordinateAssignmentError),

    #[error(transparent)]
    Spectral(#[from] SpectralInputError<R>),

    #[error(transparent)]
    InPlane(#[from] InPlaneInputError<R>),

    #[error(transparent)]
    CanonicaliseInPlane(#[from] CanonicaliseInPlaneError),

    #[error("failed to seed {variable:?}: {source}")]
    Seed {
        variable: CoordinateVariable,

        #[source]
        source: UnsupportedDerivativeSlot,
    },
}
