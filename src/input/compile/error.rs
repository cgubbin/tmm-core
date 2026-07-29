use nalgebra::ComplexField;
use thiserror::Error;

use crate::input::compile::{
    CoordinateCompileError, StackCompileError, assignment::ParameterAssignmentError,
};

#[derive(Debug, Error)]
pub enum CompilePlaneWaveError<C: ComplexField> {
    #[error(transparent)]
    Assignment(#[from] ParameterAssignmentError),

    #[error(transparent)]
    Coordinates(#[from] CoordinateCompileError<C>),

    #[error(transparent)]
    Stack(#[from] StackCompileError<C::RealField>),
}
