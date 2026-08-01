//! Second-order differential jets.
//!
//! This module provides generic containers for propagating first and second
//! derivatives through algebraic expressions.
//!
//! [`Jet`] stores a value and its first two derivatives with respect to one
//! scalar parameter:
//!
//! ```text
//! (f, f′, f″)
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

pub(crate) type ArrayJet2<C, D, P> = Jet2<ArrayBase<OwnedRepr<C>, D>, P>;

pub(crate) type PhysicalJet2<C, D> = ArrayJet2<C, D, RealParameter>;
pub(crate) type ModeJet2<C, D> = ArrayJet2<C, D, HolomorphicParameter>;

/// A value and its first and second derivatives.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct Jet2<I, P> {
    value: I,
    first: I,
    second: I,
    parameter: PhantomData<P>,
}

impl<I, P> Jet2<I, P> {
    /// Construct a second-order jet.
    pub(crate) fn from_parts(value: I, first: I, second: I) -> Self {
        Self {
            value,
            first,
            second,
            parameter: PhantomData,
        }
    }

    pub(crate) fn value(&self) -> &I {
        &self.value
    }

    pub(crate) fn first(&self) -> &I {
        &self.first
    }

    pub(crate) fn second(&self) -> &I {
        &self.second
    }

    pub(crate) fn into_parts(self) -> (I, I, I) {
        (self.value, self.first, self.second)
    }
}

impl<I, P> Jet2<I, P>
where
    I: JetAdditive,
{
    pub(crate) fn add(&self, rhs: &Self) -> Self {
        Self::from_parts(
            self.value.jet_add(&rhs.value),
            self.first.jet_add(&rhs.first),
            self.second.jet_add(&rhs.second),
        )
    }

    pub(crate) fn subtract(&self, rhs: &Self) -> Self {
        Self::from_parts(
            self.value.jet_subtract(&rhs.value),
            self.first.jet_subtract(&rhs.first),
            self.second.jet_subtract(&rhs.second),
        )
    }

    pub(crate) fn negate(&self) -> Self {
        Self::from_parts(
            self.value.jet_negate(),
            self.first.jet_negate(),
            self.second.jet_negate(),
        )
    }
}

impl<V, P> Jet2<V, P> {
    pub fn multiply_by_scalar<S>(&self, scalar: &Jet2<S, P>) -> Self
    where
        V: JetAdditive + JetMultiplyByScalar<S>,
        P: Clone,
    {
        let value = self.value().jet_multiply_by_scalar(scalar.value());

        let first = self
            .first()
            .jet_multiply_by_scalar(scalar.value())
            .jet_add(&self.value().jet_multiply_by_scalar(scalar.first()));

        let second = self
            .second()
            .jet_multiply_by_scalar(scalar.value())
            .jet_add(
                &self
                    .first()
                    .jet_multiply_by_scalar(scalar.first())
                    .jet_double(),
            )
            .jet_add(&self.value().jet_multiply_by_scalar(scalar.second()));

        Self::from_parts(value, first, second)
    }
}

impl<I, P> Jet2<I, P>
where
    I: JetBilinear,
{
    /// Multiply two second-order jets using the product rules.
    pub(crate) fn multiply(&self, rhs: &Self) -> Self {
        let cross = self.first.jet_multiply(&rhs.first).jet_double();

        Self::from_parts(
            self.value.jet_multiply(&rhs.value),
            self.first
                .jet_multiply(&rhs.value)
                .jet_add(&self.value.jet_multiply(&rhs.first)),
            self.second
                .jet_multiply(&rhs.value)
                .jet_add(&cross)
                .jet_add(&self.value.jet_multiply(&rhs.second)),
        )
    }
}

impl<I, P> Jet2<I, P>
where
    I: JetConstant + JetZeroLike,
{
    /// Construct a constant second-order jet with zero derivatives.
    pub(crate) fn constant_like(source: &I, value: I::Scalar) -> Self {
        let zero = I::jet_zeros_like(source);

        Self::from_parts(source.jet_constant_like(value), zero.clone(), zero)
    }
}

impl<I, P> Jet2<I, P>
where
    I: JetScaleBy,
{
    /// Scale the value and both derivatives by a constant scalar.
    pub(crate) fn scale_by(&self, value: I::Scalar) -> Self {
        Self::from_parts(
            self.value.jet_scale_by(value),
            self.first.jet_scale_by(value),
            self.second.jet_scale_by(value),
        )
    }
}

impl<I, P> Jet2<I, P>
where
    I: JetZeroLike,
{
    /// Construct a second-order jet whose first and second derivatives are zero.
    pub(crate) fn constant(value: I) -> Self {
        let zero = I::jet_zeros_like(&value);

        Self::from_parts(value, zero.clone(), zero)
    }
}

impl<I, P> Jet2<I, P>
where
    I: JetField,
{
    /// Compute the elementwise reciprocal and its first two derivatives.
    pub(crate) fn reciprocal(&self) -> Self {
        let inverse = self.value.jet_elementwise_reciprocal();
        let inverse_squared = inverse.jet_multiply(&inverse);
        let inverse_cubed = inverse_squared.jet_multiply(&inverse);

        let first = self.first.jet_negate().jet_multiply(&inverse_squared);

        let second = self
            .first
            .jet_multiply(&self.first)
            .jet_double()
            .jet_multiply(&inverse_cubed)
            .jet_subtract(&self.second.jet_multiply(&inverse_squared));

        Self::from_parts(inverse, first, second)
    }

    /// Divide two second-order jets elementwise.
    pub(crate) fn divide(&self, rhs: &Self) -> Self {
        self.multiply(&rhs.reciprocal())
    }
}

impl<I, P> Jet2<I, P>
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

        let mixed = self.first.jet_cross(&rhs.first);

        let second = self
            .second
            .jet_cross(&rhs.value)
            .jet_add(&mixed)
            .jet_add(&mixed)
            .jet_add(&self.value.jet_cross(&rhs.second));

        Self::from_parts(value, first, second)
    }
}

impl<I> Jet2<I, RealParameter>
where
    I: JetHermitianProduct,
    I::Output: JetAdditive,
{
    /// Compute the Hermitian product of two jets differentiated with respect
    /// to a real scalar parameter.
    ///
    /// The Hermitian product is assumed to be conjugate-linear in its first
    /// operand and linear in its second operand:
    ///
    /// ```text
    /// h      = ⟨f, g⟩
    /// h′     = ⟨f′, g⟩ + ⟨f, g′⟩
    /// h″     = ⟨f″, g⟩ + 2⟨f′, g′⟩ + ⟨f, g″⟩
    /// ```
    ///
    /// This operation is intentionally unavailable for holomorphic-parameter
    /// jets because conjugating the first operand does not preserve
    /// holomorphicity.
    pub(crate) fn hermitian_dot_product(&self, rhs: &Self) -> Jet2<I::Output, RealParameter> {
        let value = self.value().jet_hermitian_product(rhs.value());

        let first = self
            .first()
            .jet_hermitian_product(rhs.value())
            .jet_add(&self.value().jet_hermitian_product(rhs.first()));

        let mixed = self.first().jet_hermitian_product(rhs.first());

        let second = self
            .second()
            .jet_hermitian_product(rhs.value())
            .jet_add(&mixed)
            .jet_add(&mixed)
            .jet_add(&self.value().jet_hermitian_product(rhs.second()));

        Jet2::from_parts(value, first, second)
    }
}

impl<I> Jet2<I, RealParameter>
where
    I: JetConjugate,
{
    /// Conjugate a jet differentiated with respect to a real parameter.
    ///
    /// For real `x`,
    ///
    /// ```text
    /// d(conj(f))/dx  = conj(df/dx)
    /// d²(conj(f))/dx² = conj(d²f/dx²)
    /// ```
    ///
    /// This operation is intentionally unavailable for holomorphic-parameter
    /// jets because complex conjugation does not preserve holomorphicity.
    pub(crate) fn conjugated(&self) -> Self {
        Self::from_parts(
            self.value.jet_conjugate(),
            self.first.jet_conjugate(),
            self.second.jet_conjugate(),
        )
    }
}

impl<I> Jet2<I, RealParameter>
where
    I: JetRealPart,
{
    /// Extract the real parts of a jet differentiated with respect to a real
    /// parameter.
    ///
    /// This operation is intentionally unavailable for holomorphic-parameter
    /// jets.
    pub(crate) fn real(&self) -> Jet2<I::RealOutput, RealParameter> {
        Jet2::from_parts(
            self.value.jet_real(),
            self.first.jet_real(),
            self.second.jet_real(),
        )
    }
}

impl<I, P> Jet2<I, P> {
    pub(crate) fn variable(value: I) -> Self
    where
        I: JetOneLike + JetZeroLike,
    {
        let first = I::jet_ones_like(&value);
        let second = I::jet_zeros_like(&value);

        Self::from_parts(value, first, second)
    }
}

impl<C, D, P> ArrayJet2<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    pub(crate) fn exp(self) -> Self {
        let exp = self.value().mapv(|value| value.exp());

        let first = exp.clone() * self.first().view();

        let second =
            exp.clone() * (self.first().mapv(|value| value * value) + self.second().view());

        Self::from_parts(exp, first, second)
    }

    pub(crate) fn sin(self) -> Self {
        let sin = self.value().mapv(|x| x.sin());

        let cos = self.value().mapv(|x| x.cos());

        let first = cos.clone() * self.first();

        let second = -sin.clone() * self.first().jet_square() + cos * self.second();

        Self::from_parts(sin, first, second)
    }

    pub(crate) fn cos(self) -> Self {
        let sin = self.value().mapv(|x| x.sin());

        let cos = self.value().mapv(|x| x.cos());

        let first = -sin.clone() * self.first();

        let second = -cos.clone() * self.first().jet_square() - sin * self.second();

        Self::from_parts(cos, first, second)
    }

    /// Compute the elementwise principal square root and its derivative.
    ///
    /// The derivative is `f' / (2 sqrt(f))`. It is singular where the value is
    /// zero. For complex values, the branch convention is that of
    pub(crate) fn sqrt(self) -> Self {
        let value = self.value.mapv(|x| x.sqrt());

        let two = C::one() + C::one();
        let four = two + two;

        let first = self.first.clone() / value.mapv(|y| two * y);

        let second = self.second / value.mapv(|y| two * y)
            - self.first.mapv(|x| x * x) / value.mapv(|y| four * y * y * y);

        Self::from_parts(value, first, second)
    }
}

impl<I, P> Jet2<I, P> {
    pub(crate) fn map_components<O, F>(self, mut f: F) -> Jet2<O, P>
    where
        F: FnMut(I) -> O,
    {
        Jet2::from_parts(f(self.value), f(self.first), f(self.second))
    }
}

/// Samples of a scalar function and its first two derivatives.
///
/// For a function `f`, this stores:
///
/// - `value = f(x)`
/// - `first = f'(x)`
/// - `second = f''(x)`
#[derive(Clone, Debug)]
pub(crate) struct SecondOrderExpansion<I> {
    value: I,
    first: I,
    second: I,
}

impl<I> SecondOrderExpansion<I> {
    pub(crate) fn new(value: I, first: I, second: I) -> Self {
        Self {
            value,
            first,
            second,
        }
    }

    pub(crate) fn value(&self) -> &I {
        &self.value
    }

    pub(crate) fn first(&self) -> &I {
        &self.first
    }

    pub(crate) fn second(&self) -> &I {
        &self.second
    }

    pub(crate) fn into_parts(self) -> (I, I, I) {
        (self.value, self.first, self.second)
    }
}

impl<C, D, P> ArrayJet2<C, D, P>
where
    C: ComplexField,
    D: Dimension,
{
    pub(crate) fn compose_sampled_function(
        argument: &Self,
        expansion: SecondOrderExpansion<Array<C, D>>,
    ) -> Self {
        let (value, function_first, function_second) = expansion.into_parts();

        let first = &function_first * argument.first();

        let argument_first_squared = argument.first() * argument.first();

        let second =
            &function_second * &argument_first_squared + &function_first * argument.second();

        Self::from_parts(value, first, second)
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

    type RealJet2<I> = Jet2<I, RealParameter>;
    type HolomorphicJet2<I> = Jet2<I, HolomorphicParameter>;

    const EPSILON: f64 = 1.0e-11;
    const FINITE_DIFFERENCE_EPSILON: f64 = 5.0e-6;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn r(value: f64) -> C {
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

    fn assert_complex_fd_close(actual: C, expected: C) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = FINITE_DIFFERENCE_EPSILON,
            max_relative = FINITE_DIFFERENCE_EPSILON,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = FINITE_DIFFERENCE_EPSILON,
            max_relative = FINITE_DIFFERENCE_EPSILON,
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

    fn centred_first_difference<F>(function: &F, point: f64, step: f64) -> C
    where
        F: Fn(f64) -> C,
    {
        (function(point + step) - function(point - step)) / (2.0 * step)
    }

    fn centred_second_difference<F>(function: &F, point: f64, step: f64) -> C
    where
        F: Fn(f64) -> C,
    {
        (function(point + step) - r(2.0) * function(point) + function(point - step)) / (step * step)
    }

    // ---------------------------------------------------------------------
    // Construction and decomposition
    // ---------------------------------------------------------------------

    #[test]
    fn from_parts_preserves_all_components() {
        let value = a0(2.0, 3.0);
        let first = a0(5.0, 7.0);
        let second = a0(11.0, 13.0);

        let jet = RealJet2::from_parts(value.clone(), first.clone(), second.clone());

        assert_eq!(jet.value(), &value);
        assert_eq!(jet.first(), &first);
        assert_eq!(jet.second(), &second);
    }

    #[test]
    fn into_parts_returns_all_components() {
        let value = a0(2.0, 3.0);
        let first = a0(5.0, 7.0);
        let second = a0(11.0, 13.0);

        let jet = RealJet2::from_parts(value.clone(), first.clone(), second.clone());

        let (actual_value, actual_first, actual_second) = jet.into_parts();

        assert_eq!(actual_value, value);
        assert_eq!(actual_first, first);
        assert_eq!(actual_second, second);
    }

    #[test]
    fn constant_has_zero_first_and_second_derivatives() {
        let value = array![c(1.0, 2.0), c(3.0, 4.0), c(5.0, 6.0),];

        let jet: RealJet2<_> = Jet2::constant(value.clone());

        assert_eq!(jet.value(), &value);
        assert_eq!(jet.first().raw_dim(), value.raw_dim());
        assert_eq!(jet.second().raw_dim(), value.raw_dim());

        for &element in jet.first() {
            assert_complex_close(element, C::new(0.0, 0.0));
        }

        for &element in jet.second() {
            assert_complex_close(element, C::new(0.0, 0.0));
        }
    }

    #[test]
    fn constant_like_preserves_shape() {
        let source = array![c(1.0, 2.0), c(3.0, 4.0), c(5.0, 6.0),];

        let constant = c(7.0, -2.0);

        let jet: RealJet2<_> = Jet2::constant_like(&source, constant);

        assert_eq!(jet.value().raw_dim(), source.raw_dim());
        assert_eq!(jet.first().raw_dim(), source.raw_dim());
        assert_eq!(jet.second().raw_dim(), source.raw_dim());

        for &element in jet.value() {
            assert_complex_close(element, constant);
        }

        for &element in jet.first() {
            assert_complex_close(element, C::new(0.0, 0.0));
        }

        for &element in jet.second() {
            assert_complex_close(element, C::new(0.0, 0.0));
        }
    }

    #[test]
    fn variable_has_unit_first_and_zero_second_derivative() {
        let value = array![c(1.0, 2.0), c(3.0, 4.0), c(5.0, 6.0),];

        let jet: RealJet2<_> = Jet2::variable(value.clone());

        assert_eq!(jet.value(), &value);

        for &element in jet.first() {
            assert_complex_close(element, C::new(1.0, 0.0));
        }

        for &element in jet.second() {
            assert_complex_close(element, C::new(0.0, 0.0));
        }
    }

    #[test]
    fn holomorphic_variable_has_same_numerical_seed() {
        let value = a0(2.0, 3.0);

        let jet: HolomorphicJet2<_> = Jet2::variable(value.clone());

        assert_eq!(jet.value(), &value);
        assert_array0_close(jet.first(), r(1.0));
        assert_array0_close(jet.second(), r(0.0));
    }

    // ---------------------------------------------------------------------
    // Additive operations
    // ---------------------------------------------------------------------

    #[test]
    fn add_is_componentwise() {
        let left: RealJet2<_> = Jet2::from_parts(a0(2.0, 3.0), a0(5.0, 7.0), a0(11.0, 13.0));

        let right = Jet2::from_parts(a0(17.0, 19.0), a0(23.0, 29.0), a0(31.0, 37.0));

        let result = left.add(&right);

        assert_array0_close(result.value(), c(19.0, 22.0));
        assert_array0_close(result.first(), c(28.0, 36.0));
        assert_array0_close(result.second(), c(42.0, 50.0));
    }

    #[test]
    fn subtract_is_componentwise() {
        let left: RealJet2<_> = Jet2::from_parts(a0(17.0, 19.0), a0(23.0, 29.0), a0(31.0, 37.0));

        let right = Jet2::from_parts(a0(2.0, 3.0), a0(5.0, 7.0), a0(11.0, 13.0));

        let result = left.subtract(&right);

        assert_array0_close(result.value(), c(15.0, 16.0));
        assert_array0_close(result.first(), c(18.0, 22.0));
        assert_array0_close(result.second(), c(20.0, 24.0));
    }

    #[test]
    fn negate_negates_all_components() {
        let jet: RealJet2<_> = Jet2::from_parts(a0(2.0, -3.0), a0(-5.0, 7.0), a0(11.0, -13.0));

        let result = jet.negate();

        assert_array0_close(result.value(), c(-2.0, 3.0));
        assert_array0_close(result.first(), c(5.0, -7.0));
        assert_array0_close(result.second(), c(-11.0, 13.0));
    }

    #[test]
    fn scale_by_scales_all_components() {
        let jet: RealJet2<_> = Jet2::from_parts(a0(2.0, -3.0), a0(-5.0, 7.0), a0(11.0, -13.0));

        let scale = c(3.0, 2.0);
        let result = jet.scale_by(scale);

        assert_array0_close(result.value(), c(2.0, -3.0) * scale);

        assert_array0_close(result.first(), c(-5.0, 7.0) * scale);

        assert_array0_close(result.second(), c(11.0, -13.0) * scale);
    }

    // ---------------------------------------------------------------------
    // Bilinear multiplication
    // ---------------------------------------------------------------------

    #[test]
    fn multiply_applies_second_order_product_rule() {
        let f_value = c(2.0, 3.0);
        let f_first = c(5.0, 7.0);
        let f_second = c(11.0, 13.0);

        let g_value = c(17.0, 19.0);
        let g_first = c(23.0, 29.0);
        let g_second = c(31.0, 37.0);

        let f: RealJet2<_> = Jet2::from_parts(arr0(f_value), arr0(f_first), arr0(f_second));

        let g = Jet2::from_parts(arr0(g_value), arr0(g_first), arr0(g_second));

        let result = f.multiply(&g);

        assert_array0_close(result.value(), f_value * g_value);

        assert_array0_close(result.first(), f_first * g_value + f_value * g_first);

        assert_array0_close(
            result.second(),
            f_second * g_value + r(2.0) * f_first * g_first + f_value * g_second,
        );
    }

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
    fn product_rule_preserves_noncommutative_order() {
        let f: RealJet2<_> = Jet2::from_parts(
            Matrix2::new(1.0, 2.0, 3.0, 4.0),
            Matrix2::new(0.0, 1.0, 2.0, 0.0),
            Matrix2::new(1.0, 0.0, 0.0, -1.0),
        );

        let g = Jet2::from_parts(
            Matrix2::new(2.0, 0.0, 1.0, 3.0),
            Matrix2::new(1.0, 4.0, 0.0, 2.0),
            Matrix2::new(0.0, 2.0, 3.0, 1.0),
        );

        let expected_value = f.value().jet_multiply(g.value());

        let expected_first = f
            .first()
            .jet_multiply(g.value())
            .jet_add(&f.value().jet_multiply(g.first()));

        let expected_second = f
            .second()
            .jet_multiply(g.value())
            .jet_add(&f.first().jet_multiply(g.first()).jet_double())
            .jet_add(&f.value().jet_multiply(g.second()));

        let wrong_reversed_second = g
            .value()
            .jet_multiply(f.second())
            .jet_add(&g.first().jet_multiply(f.first()).jet_double())
            .jet_add(&g.second().jet_multiply(f.value()));

        let result = f.multiply(&g);

        assert_eq!(result.value(), &expected_value);
        assert_eq!(result.first(), &expected_first);
        assert_eq!(result.second(), &expected_second);
        assert_ne!(expected_second, wrong_reversed_second);
    }

    // ---------------------------------------------------------------------
    // Reciprocal and division
    // ---------------------------------------------------------------------

    #[test]
    fn reciprocal_matches_second_order_formula() {
        let value = c(2.0, 3.0);
        let first = c(5.0, -7.0);
        let second = c(11.0, 13.0);

        let jet: RealJet2<_> = Jet2::from_parts(arr0(value), arr0(first), arr0(second));

        let result = jet.reciprocal();

        assert_array0_close(result.value(), r(1.0) / value);

        assert_array0_close(result.first(), -first / value.powu(2));

        assert_array0_close(
            result.second(),
            r(2.0) * first.powu(2) / value.powu(3) - second / value.powu(2),
        );
    }

    #[test]
    fn reciprocal_of_reciprocal_recovers_jet() {
        let original: RealJet2<_> = Jet2::from_parts(a0(2.0, 0.5), a0(0.7, -0.3), a0(-0.2, 0.4));

        let result = original.reciprocal().reciprocal();

        assert_array0_close(result.value(), original.value()[()]);

        assert_array0_close(result.first(), original.first()[()]);

        assert_array0_close(result.second(), original.second()[()]);
    }

    #[test]
    fn division_matches_quotient_via_reciprocal() {
        let f: RealJet2<_> = Jet2::from_parts(a0(3.0, 2.0), a0(5.0, -1.0), a0(0.7, 0.3));

        let g = Jet2::from_parts(a0(2.0, -4.0), a0(7.0, 3.0), a0(-0.5, 1.2));

        let direct = f.divide(&g);
        let expanded = f.multiply(&g.reciprocal());

        assert_eq!(direct, expanded);
    }

    // ---------------------------------------------------------------------
    // Elementary functions
    // ---------------------------------------------------------------------

    #[test]
    fn exp_applies_second_order_chain_rule() {
        let value = c(0.7, -0.4);
        let first = c(1.3, 0.2);
        let second = c(-0.6, 0.8);

        let jet: HolomorphicJet2<_> = Jet2::from_parts(arr0(value), arr0(first), arr0(second));

        let result = jet.exp();
        let expected_value = value.exp();

        assert_array0_close(result.value(), expected_value);
        assert_array0_close(result.first(), expected_value * first);
        assert_array0_close(result.second(), expected_value * (first * first + second));
    }

    #[test]
    fn sin_applies_second_order_chain_rule() {
        let value = c(0.7, -0.4);
        let first = c(1.3, 0.2);
        let second = c(-0.6, 0.8);

        let jet: HolomorphicJet2<_> = Jet2::from_parts(arr0(value), arr0(first), arr0(second));

        let result = jet.sin();

        assert_array0_close(result.value(), value.sin());

        assert_array0_close(result.first(), value.cos() * first);

        assert_array0_close(
            result.second(),
            -value.sin() * first * first + value.cos() * second,
        );
    }

    #[test]
    fn cos_applies_second_order_chain_rule() {
        let value = c(0.7, -0.4);
        let first = c(1.3, 0.2);
        let second = c(-0.6, 0.8);

        let jet: HolomorphicJet2<_> = Jet2::from_parts(arr0(value), arr0(first), arr0(second));

        let result = jet.cos();

        assert_array0_close(result.value(), value.cos());

        assert_array0_close(result.first(), -value.sin() * first);

        assert_array0_close(
            result.second(),
            -value.cos() * first * first - value.sin() * second,
        );
    }

    #[test]
    fn sqrt_applies_second_order_chain_rule() {
        let value = c(2.0, 0.5);
        let first = c(1.3, -0.2);
        let second = c(-0.4, 0.7);

        let jet: HolomorphicJet2<_> = Jet2::from_parts(arr0(value), arr0(first), arr0(second));

        let result = jet.sqrt();
        let root = value.sqrt();

        assert_array0_close(result.value(), root);

        assert_array0_close(result.first(), first / (r(2.0) * root));

        assert_array0_close(
            result.second(),
            second / (r(2.0) * root) - first * first / (r(4.0) * root * root * root),
        );
    }

    // ---------------------------------------------------------------------
    // Sampled function composition
    // ---------------------------------------------------------------------

    #[test]
    fn sampled_function_composition_applies_chain_rule() {
        let argument_value = c(0.7, -0.4);
        let argument_first = c(1.3, 0.2);
        let argument_second = c(-0.6, 0.8);

        let argument: HolomorphicJet2<_> = Jet2::from_parts(
            arr0(argument_value),
            arr0(argument_first),
            arr0(argument_second),
        );

        let expansion = SecondOrderExpansion::new(
            arr0(argument_value.sin()),
            arr0(argument_value.cos()),
            arr0(-argument_value.sin()),
        );

        let result = HolomorphicJet2::compose_sampled_function(&argument, expansion);

        assert_array0_close(result.value(), argument_value.sin());

        assert_array0_close(result.first(), argument_value.cos() * argument_first);

        assert_array0_close(
            result.second(),
            -argument_value.sin() * argument_first * argument_first
                + argument_value.cos() * argument_second,
        );
    }

    // ---------------------------------------------------------------------
    // Composite finite differences
    // ---------------------------------------------------------------------

    #[test]
    fn composite_expression_matches_real_parameter_finite_differences() {
        let initial = c(0.8, 0.3);
        let direction = c(0.4, -0.2);
        let curvature = c(-0.15, 0.1);

        let argument: RealJet2<_> =
            Jet2::from_parts(arr0(initial), arr0(direction), arr0(curvature));

        let numerator = argument.clone().exp().multiply(&argument.clone().sin());

        let shift = RealJet2::constant_like(argument.value(), r(3.0));

        let denominator = argument.add(&shift).sqrt();

        let result = numerator.divide(&denominator);

        let path = |parameter: f64| {
            initial + parameter * direction + r(0.5 * parameter * parameter) * curvature
        };

        let function = |parameter: f64| {
            let z = path(parameter);
            z.exp() * z.sin() / (z + r(3.0)).sqrt()
        };

        let first_difference = centred_first_difference(&function, 0.0, 1.0e-5);

        let second_difference = centred_second_difference(&function, 0.0, 2.0e-4);

        assert_complex_close(result.value()[()], function(0.0));

        assert_complex_fd_close(result.first()[()], first_difference);

        assert_complex_fd_close(result.second()[()], second_difference);
    }

    // ---------------------------------------------------------------------
    // Real-parameter-only operations
    // ---------------------------------------------------------------------

    #[test]
    fn conjugation_conjugates_all_components() {
        let jet: RealJet2<_> = Jet2::from_parts(a0(2.0, 3.0), a0(5.0, -7.0), a0(-11.0, 13.0));

        let result = jet.conjugated();

        assert_array0_close(result.value(), c(2.0, -3.0));

        assert_array0_close(result.first(), c(5.0, 7.0));

        assert_array0_close(result.second(), c(-11.0, -13.0));
    }

    #[test]
    fn conjugation_matches_real_parameter_finite_differences() {
        let initial = c(0.8, 0.3);
        let direction = c(0.4, -0.2);
        let curvature = c(-0.15, 0.1);

        let jet: RealJet2<_> = Jet2::from_parts(arr0(initial), arr0(direction), arr0(curvature));

        let result = jet.conjugated();

        let function = |parameter: f64| {
            (initial + parameter * direction + r(0.5 * parameter * parameter) * curvature).conj()
        };

        assert_complex_fd_close(
            result.first()[()],
            centred_first_difference(&function, 0.0, 1.0e-5),
        );

        assert_complex_fd_close(
            result.second()[()],
            centred_second_difference(&function, 0.0, 2.0e-4),
        );
    }

    #[test]
    fn real_part_extracts_all_real_components() {
        let jet: RealJet2<_> = Jet2::from_parts(a0(2.0, 3.0), a0(5.0, -7.0), a0(-11.0, 13.0));

        let result = jet.real();

        assert_relative_eq!(result.value()[()], 2.0, epsilon = EPSILON,);

        assert_relative_eq!(result.first()[()], 5.0, epsilon = EPSILON,);

        assert_relative_eq!(result.second()[()], -11.0, epsilon = EPSILON,);
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
    fn cross_product_applies_second_order_product_rule() {
        let f: RealJet2<_> = Jet2::from_parts(
            Vector3([1.0, 2.0, 3.0]),
            Vector3([4.0, 5.0, 6.0]),
            Vector3([7.0, 8.0, 9.0]),
        );

        let g = Jet2::from_parts(
            Vector3([10.0, 11.0, 12.0]),
            Vector3([13.0, 14.0, 15.0]),
            Vector3([16.0, 17.0, 18.0]),
        );

        let expected_value = f.value().jet_cross(g.value());

        let expected_first = f
            .first()
            .jet_cross(g.value())
            .jet_add(&f.value().jet_cross(g.first()));

        let expected_second = f
            .second()
            .jet_cross(g.value())
            .jet_add(&f.first().jet_cross(g.first()).jet_double())
            .jet_add(&f.value().jet_cross(g.second()));

        let result = f.cross(&g);

        assert_eq!(result.value(), &expected_value);
        assert_eq!(result.first(), &expected_first);
        assert_eq!(result.second(), &expected_second);
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
    fn hermitian_product_applies_second_order_real_parameter_rule() {
        let f: RealJet2<_> = Jet2::from_parts(
            ComplexVector2([c(1.0, 2.0), c(3.0, -1.0)]),
            ComplexVector2([c(0.5, -0.3), c(-0.7, 0.2)]),
            ComplexVector2([c(-0.2, 0.4), c(0.1, -0.6)]),
        );

        let g = Jet2::from_parts(
            ComplexVector2([c(2.0, -1.0), c(-0.5, 4.0)]),
            ComplexVector2([c(0.2, 0.6), c(1.1, -0.4)]),
            ComplexVector2([c(0.8, -0.5), c(-0.3, 0.7)]),
        );

        let expected_value = f.value().jet_hermitian_product(g.value());

        let expected_first = f
            .first()
            .jet_hermitian_product(g.value())
            .jet_add(&f.value().jet_hermitian_product(g.first()));

        let expected_second = f
            .second()
            .jet_hermitian_product(g.value())
            .jet_add(&f.first().jet_hermitian_product(g.first()).jet_double())
            .jet_add(&f.value().jet_hermitian_product(g.second()));

        let result = f.hermitian_dot_product(&g);

        assert_complex_close(result.value().0, expected_value.0);

        assert_complex_close(result.first().0, expected_first.0);

        assert_complex_close(result.second().0, expected_second.0);
    }

    // ---------------------------------------------------------------------
    // Representation mapping
    // ---------------------------------------------------------------------

    #[test]
    fn map_components_transforms_all_components_independently() {
        let jet: RealJet2<_> = Jet2::from_parts(a0(2.0, 3.0), a0(5.0, 7.0), a0(11.0, 13.0));

        let mapped = jet.map_components(|array| array[()]);

        assert_complex_close(*mapped.value(), c(2.0, 3.0));
        assert_complex_close(*mapped.first(), c(5.0, 7.0));
        assert_complex_close(*mapped.second(), c(11.0, 13.0));
    }

    // ---------------------------------------------------------------------
    // Array shape and elementwise behaviour
    // ---------------------------------------------------------------------

    #[test]
    fn array_operations_preserve_shape_and_are_elementwise() {
        let values = array![c(1.0, 0.5), c(2.0, -0.5), c(3.0, 1.0),];

        let first = array![c(0.1, 0.2), c(0.3, -0.1), c(-0.2, 0.4),];

        let second = array![c(-0.4, 0.1), c(0.2, 0.5), c(0.7, -0.3),];

        let jet: HolomorphicJet2<_> =
            Jet2::from_parts(values.clone(), first.clone(), second.clone());

        let result = jet.exp();

        let expected_values = values.mapv(|value| value.exp());

        let expected_first = expected_values.clone() * &first;

        let expected_second =
            expected_values.clone() * (first.mapv(|value| value * value) + second);

        assert_array1_close(result.value(), &expected_values);

        assert_array1_close(result.first(), &expected_first);

        assert_array1_close(result.second(), &expected_second);
    }
}
