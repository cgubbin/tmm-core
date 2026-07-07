use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::ComplexScalar;

use super::Matrix2;

pub fn identity_matrix<C, D>(shape_source: &ArrayBase<OwnedRepr<C>, D>) -> Matrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    Matrix2::identity_like(shape_source)
}

pub fn multiply<C, D>(left: &Matrix2<C, D>, right: &Matrix2<C, D>) -> Matrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    Matrix2::new(
        left.m11().clone() * right.m11() + left.m12().clone() * right.m21(),
        left.m11().clone() * right.m12() + left.m12().clone() * right.m22(),
        left.m21().clone() * right.m11() + left.m22().clone() * right.m21(),
        left.m21().clone() * right.m12() + left.m22().clone() * right.m22(),
    )
}

pub fn multiply_first_derivative<C, D>(
    left: &Matrix2<C, D>,
    dleft: &Matrix2<C, D>,
    right: &Matrix2<C, D>,
    dright: &Matrix2<C, D>,
) -> Matrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    multiply(dleft, right).add(&multiply(left, dright))
}

pub fn multiply_second_derivative<C, D>(
    left: &Matrix2<C, D>,
    dleft: &Matrix2<C, D>,
    ddleft: &Matrix2<C, D>,
    right: &Matrix2<C, D>,
    dright: &Matrix2<C, D>,
    ddright: &Matrix2<C, D>,
) -> Matrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    let two = C::one() + C::one();

    multiply(ddleft, right)
        .add(&scale(&multiply(dleft, dright), two))
        .add(&multiply(left, ddright))
}

pub fn scale<C, D>(matrix: &Matrix2<C, D>, value: C) -> Matrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    Matrix2::new(
        matrix.m11().clone() * value,
        matrix.m12().clone() * value,
        matrix.m21().clone() * value,
        matrix.m22().clone() * value,
    )
}
