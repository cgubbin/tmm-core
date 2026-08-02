use super::{C, c};
use crate::algebra::{ArrayJet0, ArrayJet1, ArrayJet2, RealParameter, ScalarAlgebra};

use ndarray::{Array, ArrayBase, Dimension, Ix0, OwnedRepr, arr0};

pub type P = RealParameter;

pub type J0 = ArrayJet0<C, Ix0, P>;
pub type J1 = ArrayJet1<C, Ix0, P>;
pub type J2 = ArrayJet2<C, Ix0, P>;

pub fn unit_jet_like<D: Dimension>(source: &ArrayBase<OwnedRepr<C>, D>) -> ArrayJet0<C, D, P> {
    ArrayJet0::filled_constant_like(source, c(1.0))
}

pub fn zero_jet_from_real_value(value: f64) -> ArrayJet0<C, Ix0, RealParameter> {
    ArrayJet0::new(arr0(c(value)))
}

pub fn zero_jet_from_value(value: C) -> ArrayJet0<C, Ix0, RealParameter> {
    ArrayJet0::new(arr0(value))
}

pub fn zero_jet_from_array<D>(array: ArrayBase<OwnedRepr<C>, D>) -> ArrayJet0<C, D, RealParameter>
where
    D: Dimension,
{
    ArrayJet0::new(array)
}

pub fn independent_first<D>(value: Array<C, D>) -> ArrayJet1<C, D, P>
where
    D: Dimension,
{
    let first = Array::from_elem(value.raw_dim(), C::new(1.0, 0.0));

    ArrayJet1::from_parts(value, first)
}

pub fn independent_second<D>(value: Array<C, D>) -> ArrayJet2<C, D, P>
where
    D: Dimension,
{
    let first = Array::from_elem(value.raw_dim(), C::new(1.0, 0.0));
    let second = Array::from_elem(value.raw_dim(), C::new(0.0, 0.0));

    ArrayJet2::from_parts(value, first, second)
}

pub fn constant_first<D>(value: Array<C, D>) -> ArrayJet1<C, D, P>
where
    D: Dimension,
{
    let first = Array::from_elem(value.raw_dim(), C::new(0.0, 0.0));

    ArrayJet1::from_parts(value, first)
}

pub fn constant_second<D>(value: Array<C, D>) -> ArrayJet2<C, D, P>
where
    D: Dimension,
{
    let first = Array::from_elem(value.raw_dim(), C::new(0.0, 0.0));
    let second = Array::from_elem(value.raw_dim(), C::new(0.0, 0.0));

    ArrayJet2::from_parts(value, first, second)
}

pub fn affine_first<D>(value: Array<C, D>, first: Array<C, D>) -> ArrayJet1<C, D, P>
where
    D: Dimension,
{
    ArrayJet1::from_parts(value, first)
}

pub fn quadratic_second<D>(
    value: Array<C, D>,
    first: Array<C, D>,
    second: Array<C, D>,
) -> ArrayJet2<C, D, P>
where
    D: Dimension,
{
    ArrayJet2::from_parts(value, first, second)
}
