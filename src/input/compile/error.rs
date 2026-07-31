use nalgebra::ComplexField;
use thiserror::Error;

use crate::input::compile::{CoordinateCompileError, StackCompileError};
use crate::parameter::DerivativeMappingError;

#[derive(Debug, Error)]
pub enum CompilePlaneWaveError<C: ComplexField> {
    #[error(transparent)]
    Mapping(#[from] MappingError),

    #[error(transparent)]
    Coordinates(#[from] CoordinateCompileError<C>),

    #[error(transparent)]
    Stack(#[from] StackCompileError<C::RealField>),
}

#[derive(Debug, Error)]
pub enum MappingError {
    #[error(transparent)]
    Mapping(#[from] DerivativeMappingError),

    #[error(
        "layer thickness variable refers to layer {layer}, \
         but the stack contains only {finite_layer_count} layer(s)"
    )]
    LayerOutOfBounds {
        layer: usize,
        finite_layer_count: usize,
    },
}
