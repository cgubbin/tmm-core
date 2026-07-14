//! Shape-preserving 2×2 transfer matrices.
//!
//! [`Matrix2`] stores four owned `ndarray` arrays with a common sampled shape.
//! Matrix operations are ordinary 2×2 algebra, evaluated independently at
//! every point in that sample grid.
//!
//! Transfer matrices compose by ordinary matrix multiplication. If a new layer
//! matrix `L` is encountered in propagation order, the accumulated matrix is
//! updated as:
//!
//! ```text
//! M_total <- L M_total
//! ```
//!
//! This module contains only matrix representation and algebra. Layer-specific
//! constructors are implemented in the private `layer` module. Plane-wave
//! amplitudes and outgoing-mode residuals are constructed elsewhere because
//! they additionally depend on exterior-media conventions.

mod layer;

use std::ops::{Add, Mul, Neg, Sub};

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::ComplexScalar;

/// Shape-preserving 2×2 transfer matrix.
///
/// Every entry has the same sampled dimension `D`. The type does not encode a
/// physical boundary convention; it represents only the backend's native
/// transfer matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Matrix2<C, D>
where
    D: Dimension,
{
    m11: ArrayBase<OwnedRepr<C>, D>,
    m12: ArrayBase<OwnedRepr<C>, D>,
    m21: ArrayBase<OwnedRepr<C>, D>,
    m22: ArrayBase<OwnedRepr<C>, D>,
}

impl<C, D> Matrix2<C, D>
where
    D: Dimension,
{
    /// Construct a 2×2 matrix from four equally shaped sampled entries.
    ///
    /// This constructor does not validate that the arrays have matching
    /// shapes. Internal callers must preserve that invariant.
    pub fn new(
        m11: ArrayBase<OwnedRepr<C>, D>,
        m12: ArrayBase<OwnedRepr<C>, D>,
        m21: ArrayBase<OwnedRepr<C>, D>,
        m22: ArrayBase<OwnedRepr<C>, D>,
    ) -> Self {
        debug_assert_eq!(m11.raw_dim(), m12.raw_dim());
        debug_assert_eq!(m11.raw_dim(), m21.raw_dim());
        debug_assert_eq!(m11.raw_dim(), m22.raw_dim());

        Self { m11, m12, m21, m22 }
    }

    /// Return entry `(1, 1)`.
    pub fn m11(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.m11
    }

    /// Return entry `(1, 2)`.
    pub fn m12(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.m12
    }

    /// Return entry `(2, 1)`.
    pub fn m21(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.m21
    }

    /// Return entry `(2, 2)`.
    pub fn m22(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.m22
    }

    /// Consume the matrix and return its four entries in row-major order.
    #[allow(clippy::type_complexity)]
    pub(crate) fn into_parts(
        self,
    ) -> (
        ArrayBase<OwnedRepr<C>, D>,
        ArrayBase<OwnedRepr<C>, D>,
        ArrayBase<OwnedRepr<C>, D>,
        ArrayBase<OwnedRepr<C>, D>,
    ) {
        (self.m11, self.m12, self.m21, self.m22)
    }
}

impl<C, D> Matrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Construct the zero matrix with the sampled shape of `shape_source`.
    pub fn zeros_like(shape_source: &ArrayBase<OwnedRepr<C>, D>) -> Self {
        let zero = shape_source.mapv(|_| C::zero());

        Self::new(zero.clone(), zero.clone(), zero.clone(), zero)
    }

    /// Construct the identity matrix with the sampled shape of `shape_source`.
    pub fn identity_like(shape_source: &ArrayBase<OwnedRepr<C>, D>) -> Self {
        let one = shape_source.mapv(|_| C::one());
        let zero = shape_source.mapv(|_| C::zero());

        Self::new(one.clone(), zero.clone(), zero, one)
    }

    /// Compute the determinant at every sampled point.
    pub fn determinant(&self) -> ArrayBase<OwnedRepr<C>, D> {
        self.m11.clone() * self.m22.view() - self.m12.clone() * self.m21.view()
    }

    /// Multiply every matrix entry by a sampled coefficient array.
    ///
    /// `values` must have the same sampled shape as the matrix entries.
    pub fn scale_by_array(&self, values: &ArrayBase<OwnedRepr<C>, D>) -> Self {
        Self::new(
            self.m11.clone() * values.view(),
            self.m12.clone() * values.view(),
            self.m21.clone() * values.view(),
            self.m22.clone() * values.view(),
        )
    }
}

impl<C, D> Add for &Matrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    type Output = Matrix2<C, D>;

    fn add(self, rhs: Self) -> Self::Output {
        Matrix2::new(
            self.m11.clone() + rhs.m11.view(),
            self.m12.clone() + rhs.m12.view(),
            self.m21.clone() + rhs.m21.view(),
            self.m22.clone() + rhs.m22.view(),
        )
    }
}

impl<C, D> Sub for &Matrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    type Output = Matrix2<C, D>;

    fn sub(self, rhs: Self) -> Self::Output {
        Matrix2::new(
            self.m11.clone() - rhs.m11.view(),
            self.m12.clone() - rhs.m12.view(),
            self.m21.clone() - rhs.m21.view(),
            self.m22.clone() - rhs.m22.view(),
        )
    }
}

impl<C, D> Neg for &Matrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    type Output = Matrix2<C, D>;

    fn neg(self) -> Self::Output {
        Matrix2::new(
            -self.m11.clone(),
            -self.m12.clone(),
            -self.m21.clone(),
            -self.m22.clone(),
        )
    }
}

impl<C, D> Mul for &Matrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    type Output = Matrix2<C, D>;

    fn mul(self, rhs: Self) -> Self::Output {
        Matrix2::new(
            self.m11.clone() * rhs.m11.view() + self.m12.clone() * rhs.m21.view(),
            self.m11.clone() * rhs.m12.view() + self.m12.clone() * rhs.m22.view(),
            self.m21.clone() * rhs.m11.view() + self.m22.clone() * rhs.m21.view(),
            self.m21.clone() * rhs.m12.view() + self.m22.clone() * rhs.m22.view(),
        )
    }
}

impl<C, D> Mul<C> for &Matrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    type Output = Matrix2<C, D>;

    fn mul(self, rhs: C) -> Self::Output {
        Matrix2::new(
            self.m11.clone() * rhs,
            self.m12.clone() * rhs,
            self.m21.clone() * rhs,
            self.m22.clone() * rhs,
        )
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Ix0, arr0, array};
    use num_complex::Complex64;

    use super::*;

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn scalar_matrix(m11: f64, m12: f64, m21: f64, m22: f64) -> Matrix2<C, Ix0> {
        Matrix2::new(arr0(c(m11)), arr0(c(m12)), arr0(c(m21)), arr0(c(m22)))
    }

    fn assert_close(actual: C, expected: C) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = 1e-12,
            max_relative = 1e-12,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = 1e-12,
            max_relative = 1e-12,
        );
    }

    #[test]
    fn constructor_and_accessors_preserve_entries() {
        let matrix = scalar_matrix(1.0, 2.0, 3.0, 4.0);

        assert_close(matrix.m11()[()], c(1.0));
        assert_close(matrix.m12()[()], c(2.0));
        assert_close(matrix.m21()[()], c(3.0));
        assert_close(matrix.m22()[()], c(4.0));
    }

    #[test]
    fn into_parts_preserves_entries() {
        let matrix = scalar_matrix(1.0, 2.0, 3.0, 4.0);

        let (m11, m12, m21, m22) = matrix.into_parts();

        assert_close(m11[()], c(1.0));
        assert_close(m12[()], c(2.0));
        assert_close(m21[()], c(3.0));
        assert_close(m22[()], c(4.0));
    }

    #[test]
    fn zero_matrix_has_zero_entries() {
        let source = arr0(c(5.0));

        let matrix = Matrix2::zeros_like(&source);

        assert_close(matrix.m11()[()], c(0.0));
        assert_close(matrix.m12()[()], c(0.0));
        assert_close(matrix.m21()[()], c(0.0));
        assert_close(matrix.m22()[()], c(0.0));
    }

    #[test]
    fn identity_matrix_has_expected_entries() {
        let source = arr0(c(5.0));

        let matrix = Matrix2::identity_like(&source);

        assert_close(matrix.m11()[()], c(1.0));
        assert_close(matrix.m12()[()], c(0.0));
        assert_close(matrix.m21()[()], c(0.0));
        assert_close(matrix.m22()[()], c(1.0));
    }

    #[test]
    fn determinant_matches_two_by_two_formula() {
        let matrix = scalar_matrix(1.0, 2.0, 3.0, 4.0);

        assert_close(matrix.determinant()[()], c(1.0 * 4.0 - 2.0 * 3.0));
    }

    #[test]
    fn addition_is_entrywise() {
        let left = scalar_matrix(1.0, 2.0, 3.0, 4.0);
        let right = scalar_matrix(5.0, 6.0, 7.0, 8.0);

        let result = &left + &right;

        assert_close(result.m11()[()], c(6.0));
        assert_close(result.m12()[()], c(8.0));
        assert_close(result.m21()[()], c(10.0));
        assert_close(result.m22()[()], c(12.0));
    }

    #[test]
    fn subtraction_is_entrywise() {
        let left = scalar_matrix(5.0, 7.0, 11.0, 13.0);
        let right = scalar_matrix(2.0, 3.0, 5.0, 7.0);

        let result = &left - &right;

        assert_close(result.m11()[()], c(3.0));
        assert_close(result.m12()[()], c(4.0));
        assert_close(result.m21()[()], c(6.0));
        assert_close(result.m22()[()], c(6.0));
    }

    #[test]
    fn negation_is_entrywise() {
        let matrix = scalar_matrix(1.0, -2.0, 3.0, -4.0);

        let result = -&matrix;

        assert_close(result.m11()[()], c(-1.0));
        assert_close(result.m12()[()], c(2.0));
        assert_close(result.m21()[()], c(-3.0));
        assert_close(result.m22()[()], c(4.0));
    }

    #[test]
    fn multiplication_is_ordinary_matrix_product() {
        let left = scalar_matrix(1.0, 2.0, 3.0, 4.0);
        let right = scalar_matrix(5.0, 6.0, 7.0, 8.0);

        let result = &left * &right;

        assert_close(result.m11()[()], c(1.0 * 5.0 + 2.0 * 7.0));
        assert_close(result.m12()[()], c(1.0 * 6.0 + 2.0 * 8.0));
        assert_close(result.m21()[()], c(3.0 * 5.0 + 4.0 * 7.0));
        assert_close(result.m22()[()], c(3.0 * 6.0 + 4.0 * 8.0));
    }

    #[test]
    fn identity_is_left_and_right_multiplicative_identity() {
        let matrix = scalar_matrix(1.0, 2.0, 3.0, 4.0);
        let identity = Matrix2::identity_like(matrix.m11());

        assert_eq!(&identity * &matrix, matrix);
        assert_eq!(&matrix * &identity, matrix);
    }

    #[test]
    fn scalar_scale_multiplies_every_entry() {
        let matrix = scalar_matrix(1.0, 2.0, 3.0, 4.0);

        let result = &matrix * c(3.0);

        assert_close(result.m11()[()], c(3.0));
        assert_close(result.m12()[()], c(6.0));
        assert_close(result.m21()[()], c(9.0));
        assert_close(result.m22()[()], c(12.0));
    }

    #[test]
    fn sampled_scale_multiplies_each_sample_independently() {
        let matrix = Matrix2::new(
            array![c(1.0), c(2.0)],
            array![c(3.0), c(4.0)],
            array![c(5.0), c(6.0)],
            array![c(7.0), c(8.0)],
        );

        let coefficients = array![c(2.0), c(3.0)];

        let result = matrix.scale_by_array(&coefficients);

        assert_eq!(result.m11(), &array![c(2.0), c(6.0)],);
        assert_eq!(result.m12(), &array![c(6.0), c(12.0)],);
        assert_eq!(result.m21(), &array![c(10.0), c(18.0)],);
        assert_eq!(result.m22(), &array![c(14.0), c(24.0)],);
    }

    #[test]
    fn array1_matrix_multiplication_is_samplewise() {
        let left = Matrix2::new(
            array![c(1.0), c(2.0)],
            array![c(3.0), c(4.0)],
            array![c(5.0), c(6.0)],
            array![c(7.0), c(8.0)],
        );

        let right = Matrix2::new(
            array![c(2.0), c(3.0)],
            array![c(5.0), c(7.0)],
            array![c(11.0), c(13.0)],
            array![c(17.0), c(19.0)],
        );

        let result = &left * &right;

        assert_eq!(
            result.m11(),
            &array![c(1.0 * 2.0 + 3.0 * 11.0), c(2.0 * 3.0 + 4.0 * 13.0),],
        );

        assert_eq!(
            result.m12(),
            &array![c(1.0 * 5.0 + 3.0 * 17.0), c(2.0 * 7.0 + 4.0 * 19.0),],
        );

        assert_eq!(
            result.m21(),
            &array![c(5.0 * 2.0 + 7.0 * 11.0), c(6.0 * 3.0 + 8.0 * 13.0),],
        );

        assert_eq!(
            result.m22(),
            &array![c(5.0 * 5.0 + 7.0 * 17.0), c(6.0 * 7.0 + 8.0 * 19.0),],
        );
    }

    #[test]
    fn all_operations_preserve_sample_shape() {
        let matrix = Matrix2::new(
            array![c(1.0), c(2.0), c(3.0)],
            array![c(4.0), c(5.0), c(6.0)],
            array![c(7.0), c(8.0), c(9.0)],
            array![c(10.0), c(11.0), c(12.0)],
        );

        let identity = Matrix2::identity_like(matrix.m11());
        let sum = &matrix + &identity;
        let product = &matrix * &identity;
        let determinant = matrix.determinant();

        let expected = matrix.m11().raw_dim();

        assert_eq!(sum.m11().raw_dim(), expected);
        assert_eq!(sum.m12().raw_dim(), expected);
        assert_eq!(sum.m21().raw_dim(), expected);
        assert_eq!(sum.m22().raw_dim(), expected);

        assert_eq!(product.m11().raw_dim(), expected);
        assert_eq!(product.m12().raw_dim(), expected);
        assert_eq!(product.m21().raw_dim(), expected);
        assert_eq!(product.m22().raw_dim(), expected);

        assert_eq!(determinant.raw_dim(), expected);
    }
}
