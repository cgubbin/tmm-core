use nalgebra::{ComplexField, RealField};
use ndarray::ScalarOperand;
use num_complex::Complex;

/// Complex scalar type supported by Lamina.
///
/// This extends [`nalgebra::ComplexField`] with constructors required by
/// Lamina's generic complex-valued algebra but not provided by the upstream
/// trait.
///
/// The [`nalgebra::ComplexField`] trait can be implemented generally by either real or complex
/// numbers, a [`ComplexScalar`] is guaranteed to be complex.
pub trait ComplexScalar: ComplexField + ScalarOperand + Copy {
    /// Return the imaginary unit.
    fn i() -> Self;

    /// Construct a scalar from its real and imaginary parts.
    fn from_parts(real: Self::RealField, imaginary: Self::RealField) -> Self;
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

#[cfg(test)]
mod tests {
    use num_complex::Complex64;

    use super::ComplexScalar;

    #[test]
    fn imaginary_unit_has_expected_value() {
        assert_eq!(<Complex64 as ComplexScalar>::i(), Complex64::new(0.0, 1.0),);
    }

    #[test]
    fn from_parts_preserves_components() {
        assert_eq!(
            <Complex64 as ComplexScalar>::from_parts(2.0, -3.0),
            Complex64::new(2.0, -3.0),
        );
    }
}
