//! Scattering entries for a single isotropic interface.
//!
//! This module constructs the scalar-channel scattering response of a planar
//! interface from the physical characteristic admittances of the adjacent
//! media.
//!
//! The implementation is generic over [`ScalarAlgebra`], so the same formula
//! is used for:
//!
//! - sampled values;
//! - first-order jets;
//! - second-order jets.

use ndarray::Dimension;

use crate::{ComplexScalar, backend::algebra::ScalarAlgebra};

use super::super::entries::ScatterEntries;

/// Construct the scattering entries for an interface.
///
/// `left` and `right` are the physical characteristic admittances of the media
/// immediately to the left and right of the interface.
///
/// With the channel convention
///
/// ```text
/// [a_L^-]   [s11 s12] [a_L^+]
/// [a_R^+] = [s21 s22] [a_R^-],
/// ```
///
/// the entries are:
///
/// ```text
/// s11 = (Y_L - Y_R) / (Y_L + Y_R)
/// s12 = 2 Y_R / (Y_L + Y_R)
/// s21 = 2 Y_L / (Y_L + Y_R)
/// s22 = (Y_R - Y_L) / (Y_L + Y_R).
/// ```
///
/// The denominator must be nonzero at every sampled point.
pub(crate) fn interface<C, D, A>(left: &A, right: &A) -> ScatterEntries<A>
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D>,
{
    let two = A::constant_like(left.value(), C::one() + C::one());

    let denominator = left.add(right);

    ScatterEntries {
        s11: left.subtract(right).divide(&denominator),

        s12: two.multiply(right).divide(&denominator),

        s21: two.multiply(left).divide(&denominator),

        s22: right.subtract(left).divide(&denominator),
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array0, ArrayBase, Dimension, Ix0, OwnedRepr, arr0, array};
    use num_complex::Complex64;

    use super::*;
    use crate::backend::jet::{ArrayJet, ArrayJetFirst};

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn assert_complex_close(actual: C, expected: C, tolerance: f64) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = tolerance,
            max_relative = tolerance,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = tolerance,
            max_relative = tolerance,
        );
    }

    fn assert_array_close<D>(
        actual: &ArrayBase<OwnedRepr<C>, D>,
        expected: &ArrayBase<OwnedRepr<C>, D>,
        tolerance: f64,
    ) where
        D: Dimension,
    {
        assert_eq!(actual.raw_dim(), expected.raw_dim());

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_complex_close(actual, expected, tolerance);
        }
    }

    #[test]
    fn equal_admittances_produce_transparent_interface() {
        let left = arr0(c(2.5));
        let right = arr0(c(2.5));

        let entries = interface::<C, Ix0, _>(&left, &right);

        assert_complex_close(entries.s11[()], c(0.0), 1e-12);

        assert_complex_close(entries.s12[()], c(1.0), 1e-12);

        assert_complex_close(entries.s21[()], c(1.0), 1e-12);

        assert_complex_close(entries.s22[()], c(0.0), 1e-12);
    }

    #[test]
    fn interface_matches_fresnel_amplitudes() {
        let left_admittance = 2.0;
        let right_admittance = 3.0;

        let left = arr0(c(left_admittance));
        let right = arr0(c(right_admittance));

        let entries = interface::<C, Ix0, _>(&left, &right);

        let denominator = left_admittance + right_admittance;

        assert_complex_close(
            entries.s11[()],
            c((left_admittance - right_admittance) / denominator),
            1e-12,
        );

        assert_complex_close(
            entries.s12[()],
            c(2.0 * right_admittance / denominator),
            1e-12,
        );

        assert_complex_close(
            entries.s21[()],
            c(2.0 * left_admittance / denominator),
            1e-12,
        );

        assert_complex_close(
            entries.s22[()],
            c((right_admittance - left_admittance) / denominator),
            1e-12,
        );
    }

    #[test]
    fn reflection_changes_sign_when_interface_is_reversed() {
        let left = arr0(c(2.0));
        let right = arr0(c(3.0));

        let forward = interface::<C, Ix0, _>(&left, &right);

        let reversed = interface::<C, Ix0, _>(&right, &left);

        assert_complex_close(reversed.s11[()], -forward.s11[()], 1e-12);

        assert_complex_close(reversed.s22[()], -forward.s22[()], 1e-12);
    }

    #[test]
    fn interface_entries_satisfy_scalar_reciprocity_identity() {
        let left = arr0(c(2.0));
        let right = arr0(c(3.0));

        let entries = interface::<C, Ix0, _>(&left, &right);

        assert_complex_close(entries.s11[()] + entries.s22[()], c(0.0), 1e-12);

        assert_complex_close(
            entries.s12[()] * left[()],
            entries.s21[()] * right[()],
            1e-12,
        );
    }

    #[test]
    fn sampled_interface_preserves_shape() {
        let left = array![c(1.0), c(2.0), c(3.0)];

        let right = array![c(2.0), c(3.0), c(4.0)];

        let entries = interface::<C, ndarray::Ix1, _>(&left, &right);

        let expected = left.raw_dim();

        assert_eq!(entries.s11.raw_dim(), expected);
        assert_eq!(entries.s12.raw_dim(), expected);
        assert_eq!(entries.s21.raw_dim(), expected);
        assert_eq!(entries.s22.raw_dim(), expected);
    }

    fn admittances(x: f64) -> (Array0<C>, Array0<C>) {
        let left = 2.0 + 0.3 * x + 0.05 * x * x;

        let right = 3.0 - 0.2 * x + 0.08 * x * x;

        (arr0(c(left)), arr0(c(right)))
    }

    #[test]
    fn first_derivative_matches_finite_difference() {
        let (left, right) = admittances(0.0);

        let left = ArrayJetFirst::from_parts(left, arr0(c(0.3)));

        let right = ArrayJetFirst::from_parts(right, arr0(c(-0.2)));

        let analytic = interface::<C, Ix0, _>(&left, &right);

        let h = 1e-6;

        let (left_plus, right_plus) = admittances(h);

        let plus = interface::<C, Ix0, _>(&left_plus, &right_plus);

        let (left_minus, right_minus) = admittances(-h);

        let minus = interface::<C, Ix0, _>(&left_minus, &right_minus);

        let expected_s11 = (plus.s11[()] - minus.s11[()]) / (2.0 * h);

        let expected_s12 = (plus.s12[()] - minus.s12[()]) / (2.0 * h);

        let expected_s21 = (plus.s21[()] - minus.s21[()]) / (2.0 * h);

        let expected_s22 = (plus.s22[()] - minus.s22[()]) / (2.0 * h);

        assert_complex_close(analytic.s11.first()[()], expected_s11, 1e-8);

        assert_complex_close(analytic.s12.first()[()], expected_s12, 1e-8);

        assert_complex_close(analytic.s21.first()[()], expected_s21, 1e-8);

        assert_complex_close(analytic.s22.first()[()], expected_s22, 1e-8);
    }

    #[test]
    fn second_derivative_matches_finite_difference() {
        let (left, right) = admittances(0.0);

        let left = ArrayJet::from_parts(left, arr0(c(0.3)), arr0(c(0.1)));

        let right = ArrayJet::from_parts(right, arr0(c(-0.2)), arr0(c(0.16)));

        let analytic = interface::<C, Ix0, _>(&left, &right);

        let h = 1e-4;

        let (left_plus, right_plus) = admittances(h);

        let plus = interface::<C, Ix0, _>(&left_plus, &right_plus);

        let (left_zero, right_zero) = admittances(0.0);

        let zero = interface::<C, Ix0, _>(&left_zero, &right_zero);

        let (left_minus, right_minus) = admittances(-h);

        let minus = interface::<C, Ix0, _>(&left_minus, &right_minus);

        let h_squared = h * h;

        let expected_s11 = (plus.s11[()] - c(2.0) * zero.s11[()] + minus.s11[()]) / h_squared;

        let expected_s12 = (plus.s12[()] - c(2.0) * zero.s12[()] + minus.s12[()]) / h_squared;

        let expected_s21 = (plus.s21[()] - c(2.0) * zero.s21[()] + minus.s21[()]) / h_squared;

        let expected_s22 = (plus.s22[()] - c(2.0) * zero.s22[()] + minus.s22[()]) / h_squared;

        assert_complex_close(analytic.s11.second()[()], expected_s11, 2e-7);

        assert_complex_close(analytic.s12.second()[()], expected_s12, 2e-7);

        assert_complex_close(analytic.s21.second()[()], expected_s21, 2e-7);

        assert_complex_close(analytic.s22.second()[()], expected_s22, 2e-7);
    }

    #[test]
    fn first_order_zero_derivatives_reproduce_value_path() {
        let left_value = arr0(c(2.0));
        let right_value = arr0(c(3.0));

        let expected = interface::<C, Ix0, _>(&left_value, &right_value);

        let left = ArrayJetFirst::from_parts(left_value, arr0(c(0.0)));

        let right = ArrayJetFirst::from_parts(right_value, arr0(c(0.0)));

        let actual = interface::<C, Ix0, _>(&left, &right);

        assert_array_close(actual.s11.value(), &expected.s11, 1e-12);

        assert_array_close(actual.s12.value(), &expected.s12, 1e-12);

        assert_array_close(actual.s21.value(), &expected.s21, 1e-12);

        assert_array_close(actual.s22.value(), &expected.s22, 1e-12);

        assert_complex_close(actual.s11.first()[()], c(0.0), 1e-12);

        assert_complex_close(actual.s12.first()[()], c(0.0), 1e-12);

        assert_complex_close(actual.s21.first()[()], c(0.0), 1e-12);

        assert_complex_close(actual.s22.first()[()], c(0.0), 1e-12);
    }
}
