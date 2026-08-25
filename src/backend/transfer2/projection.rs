//! Boundary projections of the transfer matrix.
//!
//! The transfer matrix acts on field/slope states rather than directly on
//! travelling-wave amplitudes. This module applies the right-exterior
//! directional basis vectors and constructs the homogeneous outgoing-boundary
//! residual used for modal calculations.
//!
//! With `ξ = -iY`, the right-exterior basis states are:
//!
//! ```text
//! outgoing right-going: [1, -ξR]ᵀ
//! incoming left-going:  [1, +ξR]ᵀ.
//! ```

use ndarray::Dimension;

use crate::{ComplexScalar, algebra::ScalarAlgebra, backend::transfer2::Transfer2Entries};

/// Propagate the unit right-going basis state from the right exterior to the
/// left exterior.
///
/// The returned pair is:
///
/// ```text
/// p = m11 - ξR m12
/// q = m21 - ξR m22.
/// ```
pub(super) fn right_outgoing_column<A>(entries: &Transfer2Entries<A>, right_slope: &A) -> (A, A)
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    let field = entries.m11.subtract(&entries.m12.multiply(right_slope));

    let slope = entries.m21.subtract(&entries.m22.multiply(right_slope));

    (field, slope)
}

/// Propagate the unit left-going basis state from the right exterior to the
/// left exterior.
///
/// The returned pair is:
///
/// ```text
/// a = m11 + ξR m12
/// b = m21 + ξR m22.
/// ```
pub(super) fn right_incoming_column<A>(entries: &Transfer2Entries<A>, right_slope: &A) -> (A, A)
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    let field = entries.m11.add(&entries.m12.multiply(right_slope));

    let slope = entries.m21.add(&entries.m22.multiply(right_slope));

    (field, slope)
}
