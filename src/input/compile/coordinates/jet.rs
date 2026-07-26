use crate::algebra::ScalarAlgebra;

use nalgebra::ComplexField;
use ndarray::Dimension;

/// Operations required to convert caller-facing coordinates into canonical
/// plane-wave coordinates.
pub trait CoordinateJet<C: ComplexField, D>: Sized + Clone {
    /// Multiply every coefficient by a real scalar.
    fn scale_real(self, factor: C::RealField) -> Self;

    /// Return the multiplicative reciprocal.
    fn reciprocal(self) -> Self;

    /// Apply sine coefficient-wise through the jet algebra.
    fn sin(self) -> Self;

    fn multiply(self, rhs: Self) -> Self;
}

impl<C, D, M> CoordinateJet<C, D> for M
where
    C: ComplexField,
    D: Dimension,
    M: ScalarAlgebra<C, D> + Sized + Clone,
{
    fn scale_real(self, factor: C::RealField) -> Self {
        ScalarAlgebra::scale(&self, C::from_real(factor))
    }

    fn reciprocal(self) -> Self {
        ScalarAlgebra::reciprocal(&self)
    }

    fn sin(self) -> Self {
        ScalarAlgebra::sin(&self)
    }

    fn multiply(self, rhs: Self) -> Self {
        ScalarAlgebra::multiply(&self, &rhs)
    }
}
