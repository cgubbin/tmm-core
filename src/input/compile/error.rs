use thiserror::Error;

use crate::input::compile::{
    CoordinateCompileError, StackCompileError, assignment::ParameterAssignmentError,
};

#[derive(Debug, Error)]
pub enum CompileProblemError<R> {
    #[error(transparent)]
    Assignment(#[from] ParameterAssignmentError),

    #[error(transparent)]
    Coordinates(#[from] CoordinateCompileError<R>),

    #[error(transparent)]
    Stack(#[from] StackCompileError<R>),
    // #[error(
    //     "failed to evaluate the incident-medium \
    //      refractive index"
    // )]
    // IncidentIndex {
    //     #[source]
    //     source: E,
    // },
}
