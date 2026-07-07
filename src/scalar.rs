use nalgebra::{ComplexField, RealField};
use num_complex::Complex;

/// Complex scalar type used by this crate.
///
/// This extends `ComplexField` with constructors that are deliberately missing
/// from the upstream trait.
pub trait ComplexScalar: ComplexField + Copy {
    fn zero() -> Self;
    fn one() -> Self;
    fn i() -> Self;
    fn from_real_part(real: Self::RealField) -> Self;
    fn from_parts(real: Self::RealField, imag: Self::RealField) -> Self;
}

impl<R> ComplexScalar for Complex<R>
where
    R: RealField + Copy,
{
    fn zero() -> Self {
        Self::new(R::zero(), R::zero())
    }

    fn one() -> Self {
        Self::new(R::one(), R::zero())
    }

    fn i() -> Self {
        Self::new(R::zero(), R::one())
    }

    fn from_real_part(real: R) -> Self {
        Self::new(real, R::zero())
    }

    fn from_parts(real: R, imag: R) -> Self {
        Self::new(real, imag)
    }
}
