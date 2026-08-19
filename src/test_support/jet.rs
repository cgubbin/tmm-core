use super::{C, c};
use crate::algebra::{
    ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, HolomorphicParameter,
    RealParameter, ScalarAlgebra,
};

use ndarray::{Array, ArrayBase, Dimension, Ix0, OwnedRepr, arr0};

pub type P = RealParameter;
pub type H = HolomorphicParameter;

pub type RealJ0 = ArrayJet0<C, Ix0, RealParameter>;
pub type HoloJ0 = ArrayJet0<C, Ix0, HolomorphicParameter>;

pub type RealJ1 = ArrayJet1<C, Ix0, RealParameter>;
pub type HoloJ1 = ArrayJet1<C, Ix0, HolomorphicParameter>;

pub type RealJ2 = ArrayJet2<C, Ix0, RealParameter>;
pub type HoloJ2 = ArrayJet2<C, Ix0, HolomorphicParameter>;

pub type HoloJB1 = ArrayJetBivariate1<C, Ix0, HolomorphicParameter>;
pub type HoloJB2 = ArrayJetBivariate2<C, Ix0, HolomorphicParameter>;

pub fn real_j0(value: C) -> RealJ0 {
    ArrayJet0::new(arr0(value))
}

pub fn real_j0_from_real(value: f64) -> RealJ0 {
    real_j0(c(value))
}

pub fn real_j0_from_array<D>(array: ArrayBase<OwnedRepr<C>, D>) -> ArrayJet0<C, D, RealParameter>
where
    D: Dimension,
{
    ArrayJet0::new(array)
}

pub fn holo_j0(value: C) -> HoloJ0 {
    ArrayJet0::new(arr0(value))
}

pub fn holo_j0_from_real(value: f64) -> HoloJ0 {
    holo_j0(c(value))
}

pub fn holo_j0_from_array<D>(
    array: ArrayBase<OwnedRepr<C>, D>,
) -> ArrayJet0<C, D, HolomorphicParameter>
where
    D: Dimension,
{
    ArrayJet0::new(array)
}

pub fn unit_jet_like<D: Dimension>(source: &ArrayBase<OwnedRepr<C>, D>) -> ArrayJet0<C, D, P> {
    ArrayJet0::filled_constant_like(source, c(1.0))
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
