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

/// Construct the outgoing-mode residual.
///
/// A nontrivial outgoing solution exists when a right-going state in the right
/// exterior propagates into a left-going state in the left exterior:
///
/// ```text
/// q = ξL p.
/// ```
///
/// The residual is normalized as:
///
/// ```text
/// D = ξL p - q,
/// ```
///
/// which is identical to `2 ξL / s21` for the corresponding scattering
/// representation.
pub(super) fn outgoing_residual<A>(
    entries: &Transfer2Entries<A>,
    left_slope: &A,
    right_slope: &A,
) -> A
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    let (field, slope) = right_outgoing_column(entries, right_slope);

    left_slope.multiply(&field).subtract(&slope)
}
