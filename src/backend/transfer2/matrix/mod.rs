//! Matrix algebra for the 2×2 transfer-matrix backend.
//!
//! This module defines [`Matrix2`], a shape-preserving 2×2 matrix whose entries
//! are `ndarray` arrays. Each entry has the same sample shape, so matrix
//! operations are ordinary 2×2 algebra with elementwise array operations over
//! the sample grid.
//!
//! The backend accumulates layer matrices as:
//!
//! ```text
//! M_total <- L M_total
//! ```
//!
//! and derivative products as:
//!
//! ```text
//! d(LM)  = dL M + L dM
//! d²(LM) = d²L M + 2 dL dM + L d²M
//! ```

mod layer;

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{ComplexScalar, backend::input::IncidentSide};

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
    pub fn new(
        m11: ArrayBase<OwnedRepr<C>, D>,
        m12: ArrayBase<OwnedRepr<C>, D>,
        m21: ArrayBase<OwnedRepr<C>, D>,
        m22: ArrayBase<OwnedRepr<C>, D>,
    ) -> Self {
        Self { m11, m12, m21, m22 }
    }

    pub fn m11(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.m11
    }
    pub fn m12(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.m12
    }
    pub fn m21(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.m21
    }
    pub fn m22(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.m22
    }
    pub fn determinant(&self) -> ArrayBase<OwnedRepr<C>, D>
    where
        C: ComplexScalar,
    {
        self.m11.clone() * self.m22.view() - self.m12.clone() * self.m21.view()
    }

    pub fn zeros_like(shape_source: &ArrayBase<OwnedRepr<C>, D>) -> Self
    where
        C: ComplexScalar,
    {
        let zero = shape_source.mapv(|_| C::zero());
        Self::new(zero.clone(), zero.clone(), zero.clone(), zero)
    }

    pub fn identity_like(shape_source: &ArrayBase<OwnedRepr<C>, D>) -> Self
    where
        C: ComplexScalar,
    {
        let one = shape_source.mapv(|_| C::one());
        let zero = shape_source.mapv(|_| C::zero());
        Self::new(one.clone(), zero.clone(), zero, one)
    }

    /// Multiply every element of a 2x2 matrix by scalar `value`
    pub fn scale(&self, value: C) -> Self
    where
        C: ComplexScalar,
    {
        Matrix2::new(
            self.m11().clone() * value,
            self.m12().clone() * value,
            self.m21().clone() * value,
            self.m22().clone() * value,
        )
    }

    /// Multiply every entry of a 2×2 matrix by a sample-shaped array.
    ///
    /// This is used for chain-rule transformations where the derivative scaling
    /// factor varies over the input grid.
    pub fn scale_by_array(&self, values: &ArrayBase<OwnedRepr<C>, D>) -> Matrix2<C, D>
    where
        C: ComplexScalar,
    {
        Matrix2::new(
            self.m11().clone() * values.view(),
            self.m12().clone() * values.view(),
            self.m21().clone() * values.view(),
            self.m22().clone() * values.view(),
        )
    }

    pub fn multiply(&self, rhs: &Self) -> Self
    where
        C: ComplexScalar,
    {
        self * rhs
    }

    pub fn add(&self, rhs: &Self) -> Self
    where
        C: ComplexScalar,
    {
        self + rhs
    }

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

    pub(super) fn amplitudes(
        self,
        left_admittance: &ArrayBase<OwnedRepr<C>, D>,
        right_admittance: &ArrayBase<OwnedRepr<C>, D>,
        incident_side: IncidentSide,
    ) -> (ArrayBase<OwnedRepr<C>, D>, ArrayBase<OwnedRepr<C>, D>)
    where
        C: ComplexScalar,
        D: Dimension,
    {
        use std::ops::{Add, Div, Mul, Sub};

        let (a, b, c, d) = self.into_parts();

        let two = a.mapv(|_| C::one() + C::one());

        let b_yr = b.clone() * right_admittance;
        let d_yr = d.clone() * right_admittance;

        let u = a.clone() - &b_yr;
        let v = c.clone() - &d_yr;

        let denominator = left_admittance.mul(&u).sub(&v);

        match incident_side {
            IncidentSide::Left => {
                let reflection = left_admittance.mul(&u).add(&v).div(&denominator);

                let transmission = two.mul(left_admittance).div(&denominator);

                (reflection, transmission)
            }

            IncidentSide::Right => {
                let p = a.clone().add(&b_yr);
                let q = c.clone().add(&d_yr);

                let reflection = q.sub(&left_admittance.mul(&p)).div(&denominator);

                let determinant = a.mul(d).sub(&b.mul(c));

                let transmission = two
                    .mul(right_admittance)
                    .mul(&determinant)
                    .div(&denominator);

                (reflection, transmission)
            }
        }
    }
}

impl<C, D> std::ops::Add for &Matrix2<C, D>
where
    D: Dimension,
    C: ComplexScalar,
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

impl<C, D> std::ops::Sub for &Matrix2<C, D>
where
    D: Dimension,
    C: ComplexScalar,
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

impl<C, D> std::ops::Mul for &Matrix2<C, D>
where
    D: Dimension,
    C: ComplexScalar,
{
    type Output = Matrix2<C, D>;
    fn mul(self, rhs: Self) -> Self::Output {
        Matrix2::new(
            self.m11().clone() * rhs.m11() + self.m12().clone() * rhs.m21(),
            self.m11().clone() * rhs.m12() + self.m12().clone() * rhs.m22(),
            self.m21().clone() * rhs.m11() + self.m22().clone() * rhs.m21(),
            self.m21().clone() * rhs.m12() + self.m22().clone() * rhs.m22(),
        )
    }
}

#[cfg(test)]
mod matrix_math_tests {
    use approx::assert_relative_eq;
    use ndarray::{ArrayBase, Dimension, Ix0, OwnedRepr, arr0, arr1};
    use num_complex::Complex64;

    use super::*;

    const TOL: f64 = 1e-5;

    type C = Complex64;

    fn c(x: f64) -> C {
        C::new(x, 0.0)
    }

    fn assert_array_close<D>(
        actual: &ArrayBase<OwnedRepr<C>, D>,
        expected: &ArrayBase<OwnedRepr<C>, D>,
    ) where
        D: Dimension,
    {
        assert_eq!(actual.shape(), expected.shape());

        for (actual, expected) in actual.iter().zip(expected) {
            assert_relative_eq!(actual, expected, epsilon = TOL)
        }
    }

    fn assert_matrix_close<D>(actual: &Matrix2<C, D>, expected: &Matrix2<C, D>)
    where
        D: Dimension,
    {
        assert_array_close(actual.m11(), expected.m11());
        assert_array_close(actual.m12(), expected.m12());
        assert_array_close(actual.m21(), expected.m21());
        assert_array_close(actual.m22(), expected.m22());
    }

    fn scalar_matrix(a: f64, b: f64, c_: f64, d: f64) -> Matrix2<C, Ix0> {
        Matrix2::new(arr0(c(a)), arr0(c(b)), arr0(c(c_)), arr0(c(d)))
    }

    #[test]
    fn identity_left_multiplication_returns_matrix() {
        let m = scalar_matrix(1.0, 2.0, 3.0, 4.0);
        let id = Matrix2::identity_like(m.m11());

        assert_matrix_close(&(&id * &m), &m);
    }

    #[test]
    fn identity_right_multiplication_returns_matrix() {
        let m = scalar_matrix(1.0, 2.0, 3.0, 4.0);
        let id = Matrix2::identity_like(m.m11());

        assert_matrix_close(&(&m * &id), &m);
    }

    #[test]
    fn multiplication_matches_manual_scalar_result() {
        let a = scalar_matrix(1.0, 2.0, 3.0, 4.0);
        let b = scalar_matrix(5.0, 6.0, 7.0, 8.0);

        let expected = scalar_matrix(19.0, 22.0, 43.0, 50.0);

        assert_matrix_close(&(&a * &b), &expected);
    }

    #[test]
    fn multiplication_preserves_array_shape() {
        let a = Matrix2::new(
            arr1(&[c(1.0), c(2.0)]),
            arr1(&[c(0.0), c(1.0)]),
            arr1(&[c(1.0), c(0.0)]),
            arr1(&[c(2.0), c(3.0)]),
        );

        let b = Matrix2::identity_like(a.m11());

        let product = &a * &b;

        assert_eq!(product.m11().shape(), &[2]);
        assert_matrix_close(&product, &a);
    }

    #[test]
    fn determinant_is_m11_m22_minus_m12_m21() {
        let m = scalar_matrix(1.0, 2.0, 3.0, 4.0);

        assert_relative_eq!(m.determinant()[()], c(-2.0));
    }

    #[test]
    fn determinant_is_multiplicative_for_scalar_matrices() {
        let a = scalar_matrix(1.0, 2.0, 3.0, 4.0);
        let b = scalar_matrix(5.0, 6.0, 7.0, 8.0);

        let ab = &a * &b;

        assert_relative_eq!(
            ab.determinant()[()],
            a.determinant()[()] * b.determinant()[()],
            max_relative = 1e-12
        );
    }

    #[test]
    fn first_derivative_product_rule_matches_finite_difference() {
        let h = 1e-6;

        let left = |x: f64| scalar_matrix(1.0 + x, 2.0, 3.0, 4.0 - x);
        let right = |x: f64| scalar_matrix(5.0, 6.0 + x, 7.0 - x, 8.0);

        let l0 = left(0.0);
        let r0 = right(0.0);

        let dl = scalar_matrix(1.0, 0.0, 0.0, -1.0);
        let dr = scalar_matrix(0.0, 1.0, -1.0, 0.0);

        let analytical = multiply_first_derivative(&l0, &dl, &r0, &dr);

        let plus = &left(h) * &right(h);
        let minus = &left(-h) * &right(-h);

        let expected = (&plus.add(&(&minus).scale(c(-1.0)))).scale(c(1.0 / (2.0 * h)));

        assert_matrix_close(&analytical, &expected);
    }

    #[test]
    fn second_derivative_product_rule_matches_finite_difference() {
        let h = 1e-4;

        let left = |x: f64| scalar_matrix(1.0 + x + x * x, 2.0, 3.0, 4.0 - x + 0.5 * x * x);

        let right = |x: f64| scalar_matrix(5.0, 6.0 + x * x, 7.0 - x, 8.0 + 2.0 * x * x);

        let l0 = left(0.0);
        let r0 = right(0.0);

        let dl = scalar_matrix(1.0, 0.0, 0.0, -1.0);
        let dr = scalar_matrix(0.0, 0.0, -1.0, 0.0);

        let ddl = scalar_matrix(2.0, 0.0, 0.0, 1.0);
        let ddr = scalar_matrix(0.0, 2.0, 0.0, 4.0);

        let analytical = multiply_second_derivative(&l0, &dl, &ddl, &r0, &dr, &ddr);

        let plus = &left(h) * &right(h);
        let zero = &left(0.0) * &right(0.0);
        let minus = &left(-h) * &right(-h);

        let expected = (&plus.add(&(&zero).scale(c(-2.0))).add(&minus)).scale(c(1.0 / (h * h)));

        assert_matrix_close(&analytical, &expected);
    }
}
