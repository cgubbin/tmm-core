use ndarray::{Array2, arr1};

use crate::ComplexScalar;

pub type Tensor3<C> = Array2<C>;

pub fn zero3<C>() -> Tensor3<C>
where
    C: ComplexScalar,
{
    Array2::from_elem((3, 3), C::zero())
}

pub fn diagonal3<C>(xx: C, yy: C, zz: C) -> Tensor3<C>
where
    C: ComplexScalar,
{
    Array2::from_diag(&arr1(&[xx, yy, zz]))
}
