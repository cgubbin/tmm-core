use ndarray::Dimension;

use crate::ComplexScalar;

use super::ScatterMatrix2;

/// Compose two scalar-channel scattering matrices.
///
/// `left` is encountered first in propagation order and `right` second.
pub(crate) fn star_product<C, D>(
    left: &ScatterMatrix2<C, D>,
    right: &ScatterMatrix2<C, D>,
) -> ScatterMatrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    let one = left.s11().mapv(|_| C::one());

    let denominator = one - right.s11().clone() * left.s22().view();

    let s11 = left.s11().clone()
        + left.s12().clone() * right.s11().view() * left.s21().view() / denominator.view();

    let s12 = left.s12().clone() * right.s12().view() / denominator.view();

    let s21 = right.s21().clone() * left.s21().view() / denominator.view();

    let s22 = right.s22().clone()
        + right.s21().clone() * left.s22().view() * right.s12().view() / denominator;

    ScatterMatrix2::new(s11, s12, s21, s22)
}
