//! Matrix accumulation for the 2×2 transfer-matrix backend.
//!
//! This module contains the running product used by the transfer-matrix backend.
//! It is intentionally much simpler than the old field-state machinery: it does
//! not store electromagnetic amplitudes, boundary conditions, or observables.
//!
//! Its only job is to accumulate
//!
//! ```text
//! M_total <- L M_total
//! ```
//!
//! and, when requested, the corresponding first and second derivatives:
//!
//! ```text
//! d(LM)  = dL M + L dM
//! d²(LM) = d²L M + 2 dL dM + L d²M
//! ```
//!
//! The finished accumulator is converted into a public [`TransferResult`].

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::ComplexScalar;

use super::{
    DerivativeVariable, Matrix2, TransferDerivatives, TransferResult, multiply_first_derivative,
    multiply_second_derivative,
};

pub(crate) struct MatrixAccumulator<C, D>
where
    D: Dimension,
{
    matrix: Matrix2<C, D>,
    first: Option<Matrix2<C, D>>,
    second: Option<Matrix2<C, D>>,
    variable: Option<DerivativeVariable>,
}

impl<C, D> MatrixAccumulator<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Create an accumulator initialized to the identity matrix.
    pub(crate) fn new(shape_source: &ArrayBase<OwnedRepr<C>, D>) -> Self {
        Self {
            matrix: Matrix2::identity_like(shape_source),
            first: None,
            second: None,
            variable: None,
        }
    }
    /// Accumulate one non-differentiated layer matrix.
    pub(crate) fn update(&mut self, layer: &Matrix2<C, D>) {
        self.matrix = layer * &self.matrix;
    }

    /// Accumulate one layer matrix and its first derivative.
    pub(crate) fn update_first(
        &mut self,
        dvariable: DerivativeVariable,
        layer: &Matrix2<C, D>,
        dlayer: &Matrix2<C, D>,
    ) {
        if let Some(variable) = self.variable {
            debug_assert!(variable == dvariable);
        }
        self.variable = Some(dvariable);

        let dcurrent = self
            .first
            .take()
            .unwrap_or_else(|| Matrix2::zeros_like(layer.m11()));

        self.first = Some(multiply_first_derivative(
            layer,
            dlayer,
            &self.matrix,
            &dcurrent,
        ));

        self.matrix = layer * &self.matrix;
    }

    /// Accumulate one layer matrix and its first and second derivatives.
    pub(crate) fn update_second(
        &mut self,
        dvariable: DerivativeVariable,
        layer: &Matrix2<C, D>,
        dlayer: &Matrix2<C, D>,
        ddlayer: &Matrix2<C, D>,
    ) {
        if let Some(variable) = self.variable {
            debug_assert!(variable == dvariable);
        }

        self.variable = Some(dvariable);

        let ddcurrent = self
            .second
            .take()
            .unwrap_or_else(|| Matrix2::zeros_like(layer.m11()));

        let dcurrent = self
            .first
            .take()
            .unwrap_or_else(|| Matrix2::zeros_like(layer.m11()));

        self.second = Some(multiply_second_derivative(
            layer,
            dlayer,
            ddlayer,
            &self.matrix,
            &dcurrent,
            &ddcurrent,
        ));

        self.first = Some(multiply_first_derivative(
            layer,
            dlayer,
            &self.matrix,
            &dcurrent,
        ));

        self.matrix = layer * &self.matrix;
    }

    /// Convert the accumulator into a public transfer result.
    pub(crate) fn finish(self) -> TransferResult<C, D> {
        match (self.variable, self.first, self.second) {
            (Some(variable), Some(first), Some(second)) => TransferResult::with_derivatives(
                self.matrix,
                TransferDerivatives::new(variable, first).with_second(second),
            ),
            (Some(variable), Some(first), None) => TransferResult::with_derivatives(
                self.matrix,
                TransferDerivatives::new(variable, first),
            ),
            _ => TransferResult::new(self.matrix),
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{ArrayBase, Dimension, OwnedRepr, arr0};
    use num_complex::Complex64;

    use std::ops::Add;

    use super::*;

    type C = Complex64;

    fn c(x: f64) -> C {
        C::new(x, 0.0)
    }

    fn scalar_matrix(a: f64, b: f64, c_: f64, d: f64) -> Matrix2<C, ndarray::Ix0> {
        Matrix2::new(arr0(c(a)), arr0(c(b)), arr0(c(c_)), arr0(c(d)))
    }

    fn assert_array_close<D>(
        actual: &ArrayBase<OwnedRepr<C>, D>,
        expected: &ArrayBase<OwnedRepr<C>, D>,
        tolerance: f64,
    ) where
        D: Dimension,
    {
        assert_eq!(actual.shape(), expected.shape());

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(
                actual.re,
                expected.re,
                max_relative = tolerance,
                epsilon = tolerance
            );
            assert_relative_eq!(
                actual.im,
                expected.im,
                max_relative = tolerance,
                epsilon = tolerance
            );
        }
    }

    fn assert_matrix_close<D>(actual: &Matrix2<C, D>, expected: &Matrix2<C, D>, tolerance: f64)
    where
        D: Dimension,
    {
        assert_array_close(actual.m11(), expected.m11(), tolerance);
        assert_array_close(actual.m12(), expected.m12(), tolerance);
        assert_array_close(actual.m21(), expected.m21(), tolerance);
        assert_array_close(actual.m22(), expected.m22(), tolerance);
    }

    #[test]
    fn empty_accumulator_finishes_as_identity() {
        let source = arr0(c(1.0));
        let result = MatrixAccumulator::new(&source).finish();

        let expected = Matrix2::identity_like(&source);

        assert_matrix_close(result.matrix(), &expected, 1e-12);
        assert!(result.derivatives().is_none());
    }

    #[test]
    fn single_update_returns_layer_matrix() {
        let source = arr0(c(1.0));
        let layer = scalar_matrix(1.0, 2.0, 3.0, 4.0);

        let mut accumulator = MatrixAccumulator::new(&source);
        accumulator.update(&layer);

        let result = accumulator.finish();

        assert_matrix_close(result.matrix(), &layer, 1e-12);
    }

    #[test]
    fn two_updates_accumulate_left_to_right_product() {
        let source = arr0(c(1.0));

        let first = scalar_matrix(1.0, 2.0, 3.0, 4.0);
        let second = scalar_matrix(5.0, 6.0, 7.0, 8.0);

        let mut accumulator = MatrixAccumulator::new(&source);
        accumulator.update(&first);
        accumulator.update(&second);

        let result = accumulator.finish();

        let expected = &second * &first;

        assert_matrix_close(result.matrix(), &expected, 1e-12);
    }

    #[test]
    fn first_derivative_accumulates_product_rule() {
        let source = arr0(c(1.0));

        let first = scalar_matrix(1.0, 2.0, 3.0, 4.0);
        let second = scalar_matrix(5.0, 6.0, 7.0, 8.0);

        let dfirst = scalar_matrix(0.1, 0.2, 0.3, 0.4);
        let dsecond = scalar_matrix(0.5, 0.6, 0.7, 0.8);

        let variable = DerivativeVariable::Thickness(0);

        let mut accumulator = MatrixAccumulator::new(&source);
        accumulator.update_first(variable, &first, &dfirst);
        accumulator.update_first(variable, &second, &dsecond);

        let result = accumulator.finish();

        let expected_matrix = &second * &first;
        let expected_derivative = &(&dsecond * &first) + &(&second * &dfirst);

        assert_matrix_close(result.matrix(), &expected_matrix, 1e-12);
        assert_matrix_close(
            result.derivatives().unwrap().first(),
            &expected_derivative,
            1e-12,
        );
    }

    #[test]
    fn first_derivative_matches_finite_difference_for_two_layer_product() {
        let h = 1e-6;
        let source = arr0(c(1.0));
        let variable = DerivativeVariable::Thickness(0);

        let first = |x: f64| scalar_matrix(1.0 + x, 2.0, 3.0, 4.0 - x);
        let second = |x: f64| scalar_matrix(5.0, 6.0 + x, 7.0 - x, 8.0);

        let dfirst = scalar_matrix(1.0, 0.0, 0.0, -1.0);
        let dsecond = scalar_matrix(0.0, 1.0, -1.0, 0.0);

        let mut accumulator = MatrixAccumulator::new(&source);
        accumulator.update_first(variable, &first(0.0), &dfirst);
        accumulator.update_first(variable, &second(0.0), &dsecond);

        let result = accumulator.finish();

        let plus = &second(h) * &first(h);
        let minus = &second(-h) * &first(-h);

        let expected = (&plus.add(&(&minus).scale(c(-1.0)))).scale(c(1.0 / (2.0 * h)));

        assert_matrix_close(result.derivatives().unwrap().first(), &expected, 1e-6);
    }

    #[test]
    fn second_derivative_accumulates_product_rule() {
        let source = arr0(c(1.0));
        let variable = DerivativeVariable::Thickness(0);

        let first = scalar_matrix(1.0, 2.0, 3.0, 4.0);
        let second = scalar_matrix(5.0, 6.0, 7.0, 8.0);

        let dfirst = scalar_matrix(0.1, 0.2, 0.3, 0.4);
        let dsecond = scalar_matrix(0.5, 0.6, 0.7, 0.8);

        let ddfirst = scalar_matrix(1.1, 1.2, 1.3, 1.4);
        let ddsecond = scalar_matrix(1.5, 1.6, 1.7, 1.8);

        let mut accumulator = MatrixAccumulator::new(&source);
        accumulator.update_second(variable, &first, &dfirst, &ddfirst);
        accumulator.update_second(variable, &second, &dsecond, &ddsecond);

        let result = accumulator.finish();

        let expected_second = &(&(&ddsecond * &first) + &(&(&dsecond * &dfirst)).scale(c(2.0)))
            + &(&second * &ddfirst);

        assert_matrix_close(
            result.derivatives().unwrap().second().unwrap(),
            &expected_second,
            1e-12,
        );
    }

    #[test]
    fn second_derivative_matches_finite_difference_for_two_layer_product() {
        let h = 1e-4;
        let source = arr0(c(1.0));
        let variable = DerivativeVariable::Thickness(0);

        let first = |x: f64| scalar_matrix(1.0 + x + x * x, 2.0, 3.0, 4.0 - x + 0.5 * x * x);

        let second = |x: f64| scalar_matrix(5.0, 6.0 + x * x, 7.0 - x, 8.0 + 2.0 * x * x);

        let dfirst = scalar_matrix(1.0, 0.0, 0.0, -1.0);
        let dsecond = scalar_matrix(0.0, 0.0, -1.0, 0.0);

        let ddfirst = scalar_matrix(2.0, 0.0, 0.0, 1.0);
        let ddsecond = scalar_matrix(0.0, 2.0, 0.0, 4.0);

        let mut accumulator = MatrixAccumulator::new(&source);
        accumulator.update_second(variable, &first(0.0), &dfirst, &ddfirst);
        accumulator.update_second(variable, &second(0.0), &dsecond, &ddsecond);

        let result = accumulator.finish();

        let plus = &second(h) * &first(h);
        let zero = &second(0.0) * &first(0.0);
        let minus = &second(-h) * &first(-h);

        let expected = (&plus.add(&(&zero).scale(c(-2.0))).add(&minus)).scale(c(1.0 / (h * h)));

        assert_matrix_close(
            result.derivatives().unwrap().second().unwrap(),
            &expected,
            1e-4,
        );
    }
}
