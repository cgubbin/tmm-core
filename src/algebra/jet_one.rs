//! First-order differential jets.
//!
//! This module provides generic containers for propagating first
//! derivatives through algebraic expressions.
//!
//! [`Jet1`] stores a value and its first derivative with respect to one
//! scalar parameter:
//!
//! ```text
//! (f, f′)
//! ```
//!
//! The parameter semantics are represented by a marker type:
//!
//! - [`RealParameter`] permits differentiation of complex-valued expressions
//!   with respect to a real parameter, including conjugation, real-part
//!   extraction, and Hermitian products;
//! - [`HolomorphicParameter`] represents complex differentiation and exposes
//!   only operations that preserve holomorphicity.
//!
//! Holomorphic jets are suitable for analytic or meromorphic characteristic
//! functions used in argument-principle mode finding. Real-parameter jets are
//! suitable for derivatives of complex fields and physical observables with
//! respect to real frequency, wavenumber, angle, or thickness.
//!
//! The underlying value type determines which algebraic operations are
//! available through capability traits:
//!
//! - [`JetAdditive`] supports addition, subtraction, and negation;
//! - [`JetBilinear`] supports a bilinear product and its product rules;
//! - [`JetField`] supports elementwise reciprocals and division.
//!
//! Arrays implement these capabilities elementwise. Transfer matrices may
//! implement additive and bilinear operations because ordinary matrix
//! multiplication is bilinear. Scattering matrices must not implement
//! [`JetBilinear`] because the Redheffer star product is rational rather than
//! bilinear.
//!
//! Value-only calculations should use the underlying value type directly.
//! [`Jet`] is reserved for calculations that request derivatives.
use crate::algebra::JetMultiplyByScalar;

use super::{
    HolomorphicParameter, JetAdditive, JetBilinear, JetConjugate, JetConstant, JetCrossProduct,
    JetField, JetHermitianProduct, JetOneLike, JetRealPart, JetScaleBy, JetZeroLike, RealParameter,
};

use nalgebra::ComplexField;
use ndarray::{Array, ArrayBase, Dimension, OwnedRepr};
use std::marker::PhantomData;

pub(crate) type ArrayJet1<C, D, P> = Jet1<ArrayBase<OwnedRepr<C>, D>, P>;

pub(crate) type PhysicalJet1<C, D> = ArrayJet1<C, D, RealParameter>;
pub(crate) type ModeJet1<C, D> = ArrayJet1<C, D, HolomorphicParameter>;

/// A value and its first derivative.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct Jet1<I, P = RealParameter> {
    value: I,
    first: I,
    parameter: PhantomData<P>,
}

impl<I, P> Jet1<I, P> {
    /// Construct a first-order jet.
    pub(crate) fn from_parts(value: I, first: I) -> Self {
        Self {
            value,
            first,
            parameter: PhantomData,
        }
    }

    /// Return the value.
    pub(crate) fn value(&self) -> &I {
        &self.value
    }

    /// Return the first derivative.
    pub(crate) fn first(&self) -> &I {
        &self.first
    }

    /// Consume the jet and return its components.
    pub(crate) fn into_parts(self) -> (I, I) {
        (self.value, self.first)
    }
}

impl<I, P> Jet1<I, P>
where
    I: JetAdditive,
{
    pub(crate) fn add(&self, rhs: &Self) -> Self {
        Self::from_parts(
            self.value.jet_add(&rhs.value),
            self.first.jet_add(&rhs.first),
        )
    }

    pub(crate) fn subtract(&self, rhs: &Self) -> Self {
        Self::from_parts(
            self.value.jet_subtract(&rhs.value),
            self.first.jet_subtract(&rhs.first),
        )
    }

    pub(crate) fn negate(&self) -> Self {
        Self::from_parts(self.value.jet_negate(), self.first.jet_negate())
    }
}

impl<I, P> Jet1<I, P>
where
    I: JetBilinear,
{
    /// Multiply two first-order jets using the product rule.
    pub(crate) fn multiply(&self, rhs: &Self) -> Self {
        Self::from_parts(
            self.value.jet_multiply(&rhs.value),
            self.first
                .jet_multiply(&rhs.value)
                .jet_add(&self.value.jet_multiply(&rhs.first)),
        )
    }
}

/// Samples of a scalar function and its first derivative.
///
/// For a function `f`, this stores:
///
/// - `value = f(x)`
/// - `first = f'(x)`
#[derive(Clone, Debug)]
pub(crate) struct FirstOrderExpansion<I> {
    value: I,
    first: I,
}

impl<I> FirstOrderExpansion<I> {
    pub(crate) fn new(value: I, first: I) -> Self {
        Self { value, first }
    }

    pub(crate) fn value(&self) -> &I {
        &self.value
    }

    pub(crate) fn first(&self) -> &I {
        &self.first
    }

    pub(crate) fn into_parts(self) -> (I, I) {
        (self.value, self.first)
    }
}

impl<C, D, P> ArrayJet1<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    pub(crate) fn compose_sampled_function(
        argument: &Self,
        expansion: FirstOrderExpansion<Array<C, D>>,
    ) -> Self {
        let (value, function_first) = expansion.into_parts();

        let first = function_first * argument.first().view();
        Self::from_parts(value, first)
    }
}

impl<I, P> Jet1<I, P>
where
    I: JetConstant + JetZeroLike,
{
    /// Construct a constant first-order jet with zero derivative.
    pub(crate) fn constant_like(source: &I, value: I::Scalar) -> Self {
        Self::from_parts(source.jet_constant_like(value), I::jet_zeros_like(source))
    }
}

impl<I, P> Jet1<I, P>
where
    I: JetZeroLike,
{
    /// Construct a first-order jet whose derivative is zero.
    pub(crate) fn constant(value: I) -> Self {
        let first = I::jet_zeros_like(&value);

        Self::from_parts(value, first)
    }
}

impl<I, P> Jet1<I, P>
where
    I: JetScaleBy,
{
    /// Scale the value and derivative by a constant scalar.
    pub(crate) fn scale_by(&self, value: I::Scalar) -> Self {
        Self::from_parts(
            self.value.jet_scale_by(value),
            self.first.jet_scale_by(value),
        )
    }
}

impl<I, P> Jet1<I, P>
where
    I: JetField,
{
    /// Compute the elementwise reciprocal and its first derivative.
    pub(crate) fn reciprocal(&self) -> Self {
        let inverse = self.value.jet_elementwise_reciprocal();
        let inverse_squared = inverse.jet_multiply(&inverse);

        Self::from_parts(
            inverse,
            self.first.jet_negate().jet_multiply(&inverse_squared),
        )
    }

    /// Divide two first-order jets elementwise.
    pub(crate) fn divide(&self, rhs: &Self) -> Self {
        self.multiply(&rhs.reciprocal())
    }
}

impl<V, P> Jet1<V, P> {
    pub fn multiply_by_scalar<S>(&self, scalar: &Jet1<S, P>) -> Self
    where
        V: JetAdditive + JetMultiplyByScalar<S>,
        P: Clone,
    {
        Self::from_parts(
            self.value().jet_multiply_by_scalar(scalar.value()),
            self.first()
                .jet_multiply_by_scalar(scalar.value())
                .jet_add(&self.value().jet_multiply_by_scalar(scalar.first())),
        )
    }
}

impl<I, P> Jet1<I, P>
where
    I: JetCrossProduct + JetAdditive,
{
    /// Compute the cross product of two first-order jets.
    ///
    /// The derivative is evaluated using the bilinear product rule.
    pub(crate) fn cross(&self, rhs: &Self) -> Self {
        let value = self.value.jet_cross(&rhs.value);

        let first = self
            .first
            .jet_cross(&rhs.value)
            .jet_add(&self.value.jet_cross(&rhs.first));

        Self::from_parts(value, first)
    }
}

impl<I> Jet1<I, RealParameter>
where
    I: JetHermitianProduct,
    I::Output: JetAdditive,
{
    /// This operation does not preserve holomorphicity and is therefore not
    /// available for holomorphic-parameter jets.
    /// Compute the Hermitian product of two jets differentiated with respect
    /// to a real scalar parameter.
    ///
    /// The Hermitian product is assumed to be conjugate-linear in its first
    /// operand and linear in its second operand:
    ///
    /// ```text
    /// h      = ⟨f, g⟩
    /// h′     = ⟨f′, g⟩ + ⟨f, g′⟩
    /// ```
    ///
    /// This operation is intentionally unavailable for holomorphic-parameter
    /// jets because conjugating the first operand does not preserve
    /// holomorphicity.
    pub(crate) fn hermitian_dot_product(&self, rhs: &Self) -> Jet1<I::Output, RealParameter> {
        let value = self.value().jet_hermitian_product(rhs.value());

        let first = self
            .first()
            .jet_hermitian_product(rhs.value())
            .jet_add(&self.value().jet_hermitian_product(rhs.first()));

        Jet1::from_parts(value, first)
    }
}

impl<I> Jet1<I, RealParameter>
where
    I: JetConjugate,
{
    /// Conjugate a jet differentiated with respect to a real parameter.
    ///
    /// For real `x`,
    ///
    /// ```text
    /// d(conj(f))/dx  = conj(df/dx)
    /// ```
    ///
    /// This operation is intentionally unavailable for holomorphic-parameter
    /// jets because complex conjugation does not preserve holomorphicity.
    pub(crate) fn conjugated(&self) -> Self {
        Self::from_parts(self.value.jet_conjugate(), self.first.jet_conjugate())
    }
}

impl<I> Jet1<I, RealParameter>
where
    I: JetRealPart,
{
    /// Extract the real parts of a jet differentiated with respect to a real
    /// parameter.
    ///
    /// This operation is intentionally unavailable for holomorphic-parameter
    /// jets.
    pub(crate) fn real(&self) -> Jet1<I::RealOutput, RealParameter> {
        Jet1::from_parts(self.value.jet_real(), self.first.jet_real())
    }
}

impl<C, D, P> ArrayJet1<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    pub(crate) fn exp(self) -> Self {
        let Self { value, first, .. } = self;

        let result = value.mapv(|x| x.exp());

        let derivative = first * &result;

        Self::from_parts(result, derivative)
    }

    pub(crate) fn sin(self) -> Self {
        let Self { value, first, .. } = self;

        let result = value.mapv(|x| x.sin());
        let derivative = value.mapv(|x| x.cos()) * first;

        Self::from_parts(result, derivative)
    }

    pub(crate) fn cos(self) -> Self {
        let Self { value, first, .. } = self;

        let result = value.mapv(|x| x.cos());
        let derivative = -value.mapv(|x| x.sin()) * first;

        Self::from_parts(result, derivative)
    }

    /// Compute the elementwise principal square root and its derivative.
    ///
    /// The derivative is `f' / (2 sqrt(f))`. It is singular where the value is
    /// zero. For complex values, the branch convention is that of
    /// [`ComplexField::sqrt`].
    pub(crate) fn sqrt(self) -> Self {
        let Self { value, first, .. } = self;
        let result = value.mapv(|x| x.sqrt());
        let two = C::one() + C::one();
        let derivative = first / result.mapv(|y| two * y);

        Self::from_parts(result, derivative)
    }
}

impl<I, P> Jet1<I, P> {
    /// Apply the same representation transformation independently to the value
    /// and derivative.
    ///
    /// This does not apply the differential chain rule and must not be used as a
    /// general function map.
    pub(crate) fn map_components<O, F>(self, mut f: F) -> Jet1<O, P>
    where
        F: FnMut(I) -> O,
    {
        Jet1::from_parts(f(self.value), f(self.first))
    }

    pub(crate) fn variable(value: I) -> Self
    where
        I: JetOneLike,
    {
        let first = I::jet_ones_like(&value);

        Self::from_parts(value, first)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use ndarray::{Array0, Array1, arr0, array};
    use num_complex::Complex64;

    type C = Complex64;
    type A0 = Array0<C>;
    type A1 = Array1<C>;

    const EPSILON: f64 = 1.0e-9;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn real(value: f64) -> C {
        c(value, 0.0)
    }

    fn a0(real: f64, imaginary: f64) -> A0 {
        arr0(c(real, imaginary))
    }

    fn assert_complex_close(actual: C, expected: C) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = EPSILON,
            max_relative = EPSILON,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = EPSILON,
            max_relative = EPSILON,
        );
    }

    fn assert_array0_close(actual: &A0, expected: C) {
        assert_complex_close(actual[()], expected);
    }

    fn assert_array1_close(actual: &A1, expected: &A1) {
        assert_eq!(actual.raw_dim(), expected.raw_dim());

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_complex_close(actual, expected);
        }
    }

    fn centred_difference<F>(function: F, step: f64) -> C
    where
        F: Fn(f64) -> C,
    {
        (function(step) - function(-step)) / (2.0 * step)
    }

    // ---------------------------------------------------------------------
    // Construction and accessors
    // ---------------------------------------------------------------------

    #[test]
    fn from_parts_preserves_value_and_derivative() {
        let value = a0(2.0, 3.0);
        let first = a0(5.0, 7.0);

        let jet: ArrayJet1<_, _, RealParameter> =
            ArrayJet1::from_parts(value.clone(), first.clone());

        assert_eq!(jet.value(), &value);
        assert_eq!(jet.first(), &first);
    }

    #[test]
    fn into_parts_returns_value_and_derivative() {
        let value = a0(2.0, 3.0);
        let first = a0(5.0, 7.0);

        let jet: ArrayJet1<_, _, RealParameter> =
            ArrayJet1::from_parts(value.clone(), first.clone());
        let (actual_value, actual_first) = jet.into_parts();

        assert_eq!(actual_value, value);
        assert_eq!(actual_first, first);
    }

    #[test]
    fn constant_has_zero_derivative() {
        let value = array![c(1.0, 2.0), c(3.0, 4.0), c(5.0, 6.0),];

        let jet: ArrayJet1<_, _, RealParameter> = ArrayJet1::constant(value.clone());

        assert_eq!(jet.value(), &value);
        assert_eq!(jet.first().raw_dim(), value.raw_dim());

        for &derivative in jet.first() {
            assert_complex_close(derivative, C::new(0.0, 0.0));
        }
    }

    #[test]
    fn constant_like_preserves_shape_and_sets_requested_value() {
        let source = array![c(1.0, 2.0), c(3.0, 4.0), c(5.0, 6.0),];

        let constant = c(7.0, -2.0);
        let jet: ArrayJet1<_, _, RealParameter> = ArrayJet1::constant_like(&source, constant);

        assert_eq!(jet.value().raw_dim(), source.raw_dim());
        assert_eq!(jet.first().raw_dim(), source.raw_dim());

        for &value in jet.value() {
            assert_complex_close(value, constant);
        }

        for &derivative in jet.first() {
            assert_complex_close(derivative, C::new(0.0, 0.0));
        }
    }

    #[test]
    fn variable_preserves_shape_and_has_unit_derivative() {
        let value = array![c(1.0, 2.0), c(3.0, 4.0), c(5.0, 6.0),];

        let jet: ArrayJet1<_, _, RealParameter> = ArrayJet1::variable(value.clone());

        assert_eq!(jet.value(), &value);
        assert_eq!(jet.first().raw_dim(), value.raw_dim());

        for &derivative in jet.first() {
            assert_complex_close(derivative, C::new(1.0, 0.0));
        }
    }

    // ---------------------------------------------------------------------
    // Additive operations
    // ---------------------------------------------------------------------

    #[test]
    fn add_is_componentwise() {
        let left: ArrayJet1<_, _, RealParameter> =
            ArrayJet1::from_parts(a0(2.0, 3.0), a0(5.0, 7.0));

        let right = ArrayJet1::from_parts(a0(11.0, 13.0), a0(17.0, 19.0));

        let result = left.add(&right);

        assert_array0_close(result.value(), c(13.0, 16.0));
        assert_array0_close(result.first(), c(22.0, 26.0));
    }

    #[test]
    fn subtract_is_componentwise() {
        let left: ArrayJet1<_, _, RealParameter> =
            ArrayJet1::from_parts(a0(11.0, 13.0), a0(17.0, 19.0));

        let right = ArrayJet1::from_parts(a0(2.0, 3.0), a0(5.0, 7.0));

        let result = left.subtract(&right);

        assert_array0_close(result.value(), c(9.0, 10.0));
        assert_array0_close(result.first(), c(12.0, 12.0));
    }

    #[test]
    fn negate_negates_value_and_derivative() {
        let jet: ArrayJet1<_, _, RealParameter> =
            ArrayJet1::from_parts(a0(2.0, -3.0), a0(-5.0, 7.0));

        let result = jet.negate();

        assert_array0_close(result.value(), c(-2.0, 3.0));
        assert_array0_close(result.first(), c(5.0, -7.0));
    }

    #[test]
    fn scale_by_scales_value_and_derivative() {
        let jet: ArrayJet1<_, _, RealParameter> =
            ArrayJet1::from_parts(a0(2.0, -3.0), a0(-5.0, 7.0));

        let scale = c(3.0, 2.0);
        let result = jet.scale_by(scale);

        assert_array0_close(result.value(), c(2.0, -3.0) * scale);

        assert_array0_close(result.first(), c(-5.0, 7.0) * scale);
    }

    // ---------------------------------------------------------------------
    // Scalar bilinear multiplication
    // ---------------------------------------------------------------------

    #[test]
    fn multiply_applies_first_order_product_rule() {
        let f: ArrayJet1<_, _, RealParameter> = ArrayJet1::from_parts(a0(2.0, 3.0), a0(5.0, 7.0));

        let g: ArrayJet1<_, _, RealParameter> =
            ArrayJet1::from_parts(a0(11.0, 13.0), a0(17.0, 19.0));

        let result = f.multiply(&g);

        let f_value = c(2.0, 3.0);
        let f_first = c(5.0, 7.0);
        let g_value = c(11.0, 13.0);
        let g_first = c(17.0, 19.0);

        assert_array0_close(result.value(), f_value * g_value);

        assert_array0_close(result.first(), f_first * g_value + f_value * g_first);
    }

    // ---------------------------------------------------------------------
    // Noncommutative multiplication
    // ---------------------------------------------------------------------

    #[derive(Clone, Debug, PartialEq)]
    struct Matrix2 {
        entries: [[f64; 2]; 2],
    }

    impl Matrix2 {
        fn new(a00: f64, a01: f64, a10: f64, a11: f64) -> Self {
            Self {
                entries: [[a00, a01], [a10, a11]],
            }
        }
    }

    impl JetAdditive for Matrix2 {
        fn jet_add(&self, rhs: &Self) -> Self {
            Self::new(
                self.entries[0][0] + rhs.entries[0][0],
                self.entries[0][1] + rhs.entries[0][1],
                self.entries[1][0] + rhs.entries[1][0],
                self.entries[1][1] + rhs.entries[1][1],
            )
        }

        fn jet_subtract(&self, rhs: &Self) -> Self {
            Self::new(
                self.entries[0][0] - rhs.entries[0][0],
                self.entries[0][1] - rhs.entries[0][1],
                self.entries[1][0] - rhs.entries[1][0],
                self.entries[1][1] - rhs.entries[1][1],
            )
        }

        fn jet_negate(&self) -> Self {
            Self::new(
                -self.entries[0][0],
                -self.entries[0][1],
                -self.entries[1][0],
                -self.entries[1][1],
            )
        }
    }

    impl JetBilinear for Matrix2 {
        fn jet_multiply(&self, rhs: &Self) -> Self {
            let a = &self.entries;
            let b = &rhs.entries;

            Self::new(
                a[0][0] * b[0][0] + a[0][1] * b[1][0],
                a[0][0] * b[0][1] + a[0][1] * b[1][1],
                a[1][0] * b[0][0] + a[1][1] * b[1][0],
                a[1][0] * b[0][1] + a[1][1] * b[1][1],
            )
        }
    }

    #[test]
    fn product_rule_preserves_noncommutative_operand_order() {
        let f: Jet1<Matrix2, RealParameter> = Jet1::from_parts(
            Matrix2::new(1.0, 2.0, 3.0, 4.0),
            Matrix2::new(0.0, 1.0, 2.0, 0.0),
        );

        let g = Jet1::from_parts(
            Matrix2::new(2.0, 0.0, 1.0, 3.0),
            Matrix2::new(1.0, 4.0, 0.0, 2.0),
        );

        let expected_value = f.value().jet_multiply(g.value());

        let expected_first = f
            .first()
            .jet_multiply(g.value())
            .jet_add(&f.value().jet_multiply(g.first()));

        let reversed_first = g
            .value()
            .jet_multiply(f.first())
            .jet_add(&g.first().jet_multiply(f.value()));

        let result = f.multiply(&g);

        assert_eq!(result.value(), &expected_value);
        assert_eq!(result.first(), &expected_first);

        // Ensure this example really detects an operand-order regression.
        assert_ne!(expected_first, reversed_first);
    }

    // ---------------------------------------------------------------------
    // Reciprocal and division
    // ---------------------------------------------------------------------

    #[test]
    fn reciprocal_matches_complex_formula() {
        let value = c(2.0, 3.0);
        let first = c(5.0, -7.0);

        let jet: ArrayJet1<_, _, RealParameter> = ArrayJet1::from_parts(arr0(value), arr0(first));

        let result = jet.reciprocal();

        assert_array0_close(result.value(), C::new(1.0, 0.0) / value);

        assert_array0_close(result.first(), -first / (value * value));
    }

    #[test]
    fn division_matches_complex_quotient_rule() {
        let f_value = c(3.0, 2.0);
        let f_first = c(5.0, -1.0);

        let g_value = c(2.0, -4.0);
        let g_first = c(7.0, 3.0);

        let f: ArrayJet1<_, _, RealParameter> = ArrayJet1::from_parts(arr0(f_value), arr0(f_first));

        let g: ArrayJet1<_, _, RealParameter> = ArrayJet1::from_parts(arr0(g_value), arr0(g_first));

        let result = f.divide(&g);

        assert_array0_close(result.value(), f_value / g_value);

        assert_array0_close(
            result.first(),
            (f_first * g_value - f_value * g_first) / (g_value * g_value),
        );
    }

    // ---------------------------------------------------------------------
    // Elementary functions
    // ---------------------------------------------------------------------

    #[test]
    fn exp_applies_chain_rule() {
        let value = c(0.7, -0.4);
        let first = c(1.3, 0.2);

        let jet: ArrayJet1<_, _, RealParameter> = ArrayJet1::from_parts(arr0(value), arr0(first));

        let result = jet.exp();
        let expected_value = value.exp();

        assert_array0_close(result.value(), expected_value);

        assert_array0_close(result.first(), expected_value * first);
    }

    #[test]
    fn sin_applies_chain_rule() {
        let value = c(0.7, -0.4);
        let first = c(1.3, 0.2);

        let jet: ArrayJet1<_, _, RealParameter> = ArrayJet1::from_parts(arr0(value), arr0(first));

        let result = jet.sin();

        assert_array0_close(result.value(), value.sin());

        assert_array0_close(result.first(), value.cos() * first);
    }

    #[test]
    fn cos_applies_chain_rule() {
        let value = c(0.7, -0.4);
        let first = c(1.3, 0.2);

        let jet: ArrayJet1<_, _, RealParameter> = ArrayJet1::from_parts(arr0(value), arr0(first));

        let result = jet.cos();

        assert_array0_close(result.value(), value.cos());

        assert_array0_close(result.first(), -value.sin() * first);
    }

    #[test]
    fn sqrt_applies_chain_rule_on_principal_branch() {
        let value = c(2.0, 0.5);
        let first = c(1.3, -0.2);

        let jet: ArrayJet1<_, _, RealParameter> = ArrayJet1::from_parts(arr0(value), arr0(first));

        let result = jet.sqrt();
        let expected_value = value.sqrt();

        assert_array0_close(result.value(), expected_value);

        assert_array0_close(result.first(), first / (real(2.0) * expected_value));
    }

    // ---------------------------------------------------------------------
    // Generic sampled-function composition
    // ---------------------------------------------------------------------

    #[test]
    fn compose_sampled_function_applies_chain_rule() {
        let argument_value = c(0.7, -0.4);
        let argument_first = c(1.3, 0.2);

        let argument: ArrayJet1<_, _, RealParameter> =
            ArrayJet1::from_parts(arr0(argument_value), arr0(argument_first));

        let sampled_value = arr0(argument_value.sin());
        let sampled_derivative = arr0(argument_value.cos());

        let result = ArrayJet1::compose_sampled_function(
            &argument,
            FirstOrderExpansion::new(sampled_value, sampled_derivative),
        );

        assert_array0_close(result.value(), argument_value.sin());

        assert_array0_close(result.first(), argument_value.cos() * argument_first);
    }

    // ---------------------------------------------------------------------
    // Composite finite-difference validation
    // ---------------------------------------------------------------------

    #[test]
    fn composite_expression_matches_centred_finite_difference() {
        let initial = c(0.8, 0.3);
        let direction = c(0.4, -0.2);

        let argument: ArrayJet1<_, _, RealParameter> =
            ArrayJet1::from_parts(arr0(initial), arr0(direction));

        // f(z) = exp(z) sin(z) / sqrt(z + 3)
        let numerator = argument.clone().exp().multiply(&argument.clone().sin());

        let shift = ArrayJet1::constant_like(argument.value(), real(3.0));

        let denominator = argument.add(&shift).sqrt();

        let result = numerator.divide(&denominator);

        let function = |parameter: f64| {
            let z = initial + parameter * direction;

            z.exp() * z.sin() / (z + real(3.0)).sqrt()
        };

        let finite_difference = centred_difference(function, 1.0e-6);

        assert_complex_close(result.value()[()], function(0.0));

        assert_relative_eq!(
            result.first()[()].re,
            finite_difference.re,
            epsilon = 2.0e-8,
            max_relative = 2.0e-8,
        );

        assert_relative_eq!(
            result.first()[()].im,
            finite_difference.im,
            epsilon = 2.0e-8,
            max_relative = 2.0e-8,
        );
    }

    // ---------------------------------------------------------------------
    // Real-parameter-only operations
    // ---------------------------------------------------------------------

    #[test]
    fn conjugation_conjugates_value_and_real_parameter_derivative() {
        let value = c(2.0, 3.0);
        let first = c(5.0, -7.0);

        let jet: ArrayJet1<_, _, RealParameter> = ArrayJet1::from_parts(arr0(value), arr0(first));

        let result = jet.conjugated();

        assert_array0_close(result.value(), value.conj());

        assert_array0_close(result.first(), first.conj());
    }

    #[test]
    fn conjugation_matches_real_parameter_finite_difference() {
        let initial = c(0.8, 0.3);
        let direction = c(0.4, -0.2);

        let jet: ArrayJet1<_, _, RealParameter> =
            ArrayJet1::from_parts(arr0(initial), arr0(direction));

        let result = jet.conjugated();

        let function = |parameter: f64| (initial + parameter * direction).conj();

        let finite_difference = centred_difference(function, 1.0e-6);

        assert_complex_close(result.first()[()], finite_difference);
    }

    #[test]
    fn real_part_extracts_real_value_and_derivative() {
        let value = c(2.0, 3.0);
        let first = c(5.0, -7.0);

        let jet: ArrayJet1<_, _, RealParameter> = ArrayJet1::from_parts(arr0(value), arr0(first));

        let result = jet.real();

        assert_relative_eq!(
            result.value()[()],
            2.0,
            epsilon = EPSILON,
            max_relative = EPSILON,
        );

        assert_relative_eq!(
            result.first()[()],
            5.0,
            epsilon = EPSILON,
            max_relative = EPSILON,
        );
    }

    // ---------------------------------------------------------------------
    // Cross product
    // ---------------------------------------------------------------------

    #[derive(Clone, Debug, PartialEq)]
    struct Vector3([f64; 3]);

    impl JetAdditive for Vector3 {
        fn jet_add(&self, rhs: &Self) -> Self {
            Self([
                self.0[0] + rhs.0[0],
                self.0[1] + rhs.0[1],
                self.0[2] + rhs.0[2],
            ])
        }

        fn jet_subtract(&self, rhs: &Self) -> Self {
            Self([
                self.0[0] - rhs.0[0],
                self.0[1] - rhs.0[1],
                self.0[2] - rhs.0[2],
            ])
        }

        fn jet_negate(&self) -> Self {
            Self([-self.0[0], -self.0[1], -self.0[2]])
        }
    }

    impl JetCrossProduct for Vector3 {
        fn jet_cross(&self, rhs: &Self) -> Self {
            let [ax, ay, az] = self.0;
            let [bx, by, bz] = rhs.0;

            Self([ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx])
        }
    }

    #[test]
    fn cross_product_applies_bilinear_product_rule() {
        let f: Jet1<Vector3, RealParameter> =
            Jet1::from_parts(Vector3([1.0, 2.0, 3.0]), Vector3([4.0, 5.0, 6.0]));

        let g = Jet1::from_parts(Vector3([7.0, 8.0, 9.0]), Vector3([10.0, 11.0, 12.0]));

        let expected_value = f.value().jet_cross(g.value());

        let expected_first = f
            .first()
            .jet_cross(g.value())
            .jet_add(&f.value().jet_cross(g.first()));

        let result = f.cross(&g);

        assert_eq!(result.value(), &expected_value);
        assert_eq!(result.first(), &expected_first);
    }

    // ---------------------------------------------------------------------
    // Hermitian product
    // ---------------------------------------------------------------------

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct ComplexScalar(C);

    impl JetAdditive for ComplexScalar {
        fn jet_add(&self, rhs: &Self) -> Self {
            Self(self.0 + rhs.0)
        }

        fn jet_subtract(&self, rhs: &Self) -> Self {
            Self(self.0 - rhs.0)
        }

        fn jet_negate(&self) -> Self {
            Self(-self.0)
        }
    }

    #[derive(Clone, Debug, PartialEq)]
    struct ComplexVector2([C; 2]);

    impl JetHermitianProduct for ComplexVector2 {
        type Output = ComplexScalar;

        fn jet_hermitian_product(&self, rhs: &Self) -> Self::Output {
            ComplexScalar(self.0[0].conj() * rhs.0[0] + self.0[1].conj() * rhs.0[1])
        }
    }

    #[test]
    fn hermitian_product_applies_real_parameter_product_rule() {
        let f = Jet1::from_parts(
            ComplexVector2([c(1.0, 2.0), c(3.0, -1.0)]),
            ComplexVector2([c(0.5, -0.3), c(-0.7, 0.2)]),
        );

        let g = Jet1::from_parts(
            ComplexVector2([c(2.0, -1.0), c(-0.5, 4.0)]),
            ComplexVector2([c(0.2, 0.6), c(1.1, -0.4)]),
        );

        let expected_value = f.value().jet_hermitian_product(g.value());

        let expected_first = f
            .first()
            .jet_hermitian_product(g.value())
            .jet_add(&f.value().jet_hermitian_product(g.first()));

        let result = f.hermitian_dot_product(&g);

        assert_complex_close(result.value().0, expected_value.0);

        assert_complex_close(result.first().0, expected_first.0);
    }

    #[test]
    fn hermitian_product_matches_real_parameter_finite_difference() {
        let f_value = [c(1.0, 2.0), c(3.0, -1.0)];

        let f_first = [c(0.5, -0.3), c(-0.7, 0.2)];

        let g_value = [c(2.0, -1.0), c(-0.5, 4.0)];

        let g_first = [c(0.2, 0.6), c(1.1, -0.4)];

        let f = Jet1::from_parts(ComplexVector2(f_value), ComplexVector2(f_first));

        let g = Jet1::from_parts(ComplexVector2(g_value), ComplexVector2(g_first));

        let result = f.hermitian_dot_product(&g);

        let function = |parameter: f64| {
            let f0 = f_value[0] + parameter * f_first[0];
            let f1 = f_value[1] + parameter * f_first[1];

            let g0 = g_value[0] + parameter * g_first[0];
            let g1 = g_value[1] + parameter * g_first[1];

            f0.conj() * g0 + f1.conj() * g1
        };

        let finite_difference = centred_difference(function, 1.0e-6);

        assert_relative_eq!(
            result.first().0.re,
            finite_difference.re,
            epsilon = 2.0e-9,
            max_relative = 2.0e-9,
        );

        assert_relative_eq!(
            result.first().0.im,
            finite_difference.im,
            epsilon = 2.0e-9,
            max_relative = 2.0e-9,
        );
    }

    // ---------------------------------------------------------------------
    // Representation mapping
    // ---------------------------------------------------------------------

    #[test]
    fn map_applies_same_transformation_to_both_components() {
        let jet: ArrayJet1<_, _, RealParameter> = ArrayJet1::from_parts(a0(2.0, 3.0), a0(5.0, 7.0));

        let mapped = jet.map_components(|array| array[()]);

        assert_complex_close(*mapped.value(), c(2.0, 3.0));

        assert_complex_close(*mapped.first(), c(5.0, 7.0));
    }

    // ---------------------------------------------------------------------
    // Multi-element array behaviour
    // ---------------------------------------------------------------------

    #[test]
    fn array_operations_are_elementwise_and_preserve_shape() {
        let values = array![c(1.0, 0.5), c(2.0, -0.5), c(3.0, 1.0),];

        let derivatives = array![c(0.1, 0.2), c(0.3, -0.1), c(-0.2, 0.4),];

        let jet: ArrayJet1<_, _, RealParameter> =
            ArrayJet1::from_parts(values.clone(), derivatives.clone());

        let result = jet.exp();

        let expected_values = values.mapv(|value| value.exp());

        let expected_derivatives = expected_values.clone() * derivatives;

        assert_array1_close(result.value(), &expected_values);

        assert_array1_close(result.first(), &expected_derivatives);
    }
}
