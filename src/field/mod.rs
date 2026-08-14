//! Scalar- and vector-valued array fields.
//!
//! The types in this module describe the geometric rank of a sampled
//! quantity:
//!
//! - [`ScalarField`] stores one scalar array;
//! - [`VectorField`] stores three Cartesian component arrays.
//!
//! These types make no assumption about the physical meaning of an array
//! axis. In particular, they do not assume that the final axis represents
//! position. Spatial profile extraction is implemented separately by the
//! response sampling layer.

mod scalar;
mod vector;

pub use scalar::ScalarField;
pub use vector::{VectorField, VectorValue};

/// Error returned when the components of a field have inconsistent shapes.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "field component `{component}` has shape {actual:?}; \
     expected {expected:?}"
)]
pub struct FieldShapeError {
    component: &'static str,
    expected: Vec<usize>,
    actual: Vec<usize>,
}

impl FieldShapeError {
    pub(crate) fn new(component: &'static str, expected: &[usize], actual: &[usize]) -> Self {
        Self {
            component,
            expected: expected.to_vec(),
            actual: actual.to_vec(),
        }
    }

    /// Name of the component whose shape was inconsistent.
    pub fn component(&self) -> &'static str {
        self.component
    }

    /// Shape expected from the reference component.
    pub fn expected(&self) -> &[usize] {
        &self.expected
    }

    /// Actual shape of the inconsistent component.
    pub fn actual(&self) -> &[usize] {
        &self.actual
    }
}
