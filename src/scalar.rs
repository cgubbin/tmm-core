use nalgebra::{ComplexField, RealField};
use ndarray::ScalarOperand;
use num_complex::Complex;

/// Complex scalar type used by this crate.
///
/// This extends `ComplexField` with constructors that are deliberately missing
/// from the upstream trait.
pub trait ComplexScalar: ComplexField + ScalarOperand + Copy {
    fn i() -> Self;
    fn from_parts(real: Self::RealField, imag: Self::RealField) -> Self;
}

impl<R> ComplexScalar for Complex<R>
where
    R: RealField + Copy,
    Complex<R>: ScalarOperand,
{
    fn i() -> Self {
        Self::new(R::zero(), R::one())
    }

    fn from_parts(real: R, imag: R) -> Self {
        Self::new(real, imag)
    }
}
