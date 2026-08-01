//! Bivariate first-order differential jets.
//!
//! [`JetBivariate`] propagates derivatives with respect to two independent
//! scalar coordinates. It stores a value, a two-component gradient, and the
//! three independent entries of a symmetric Hessian:
//!
//! ```text
//! (f, fₓ, fᵧ,)
//! ```
//!
//! The coordinates are abstractly named `axis0` and `axis1`. Application crates may
//! provide aliases or constructors assigning physical meanings to them.
//!
//! Parameter semantics are represented by a marker type:
//!
//! - [`RealParameter`] permits differentiation with respect to two real
//!   coordinates, including conjugation, real-part extraction, and Hermitian
//!   products;
//! - [`HolomorphicParameter`] represents differentiation with respect to
//!   complex coordinates and exposes only operations preserving
//!   holomorphicity.
//!
//! The stored Hessian assumes that mixed partial derivatives commute:
//!
//! ```text
//! ∂²f/∂x∂y = ∂²f/∂y∂x.
//! ```
//!
//! The payload type determines the available algebra through capability
//! traits such as [`JetAdditive`], [`JetBilinear`], and [`JetField`].

use crate::algebra::{FirstOrderExpansion, JetMultiplyByScalar};
use crate::differential::BivariateGradient;

use super::{
    HolomorphicParameter, JetAdditive, JetBilinear, JetConjugate, JetConstant, JetCrossProduct,
    JetHermitianProduct, JetOneLike, JetRealPart, JetScaleBy, JetZeroLike, RealParameter,
    SecondOrderExpansion,
};

use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, OwnedRepr};
use std::marker::PhantomData;

pub(crate) type ArrayJetBivariate1<C, D, P> = JetBivariate1<ArrayBase<OwnedRepr<C>, D>, P>;

pub(crate) type PhysicalJetBivariate1<C, D> = ArrayJetBivariate1<C, D, RealParameter>;
pub(crate) type ModeJetBivariate1<C, D> = ArrayJetBivariate1<C, D, HolomorphicParameter>;

#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct JetBivariate1<A, P> {
    value: A,

    first: BivariateGradient<A>,

    parameter: PhantomData<P>,
}

impl<I, P> JetBivariate1<I, P> {
    pub(crate) fn from_parts(value: I, first: BivariateGradient<I>) -> Self {
        Self {
            value,
            first,
            parameter: PhantomData,
        }
    }

    pub(crate) fn from_components(value: I, axis0: I, axis1: I) -> Self {
        Self::from_parts(value, BivariateGradient::new(axis0, axis1))
    }

    pub(crate) fn value(&self) -> &I {
        &self.value
    }

    pub(crate) fn axis0(&self) -> &I {
        self.first.axis0()
    }

    pub(crate) fn axis1(&self) -> &I {
        self.first.axis1()
    }

    pub(crate) fn first(&self) -> &BivariateGradient<I> {
        &self.first
    }

    pub(crate) fn into_parts(self) -> (I, BivariateGradient<I>) {
        (self.value, self.first)
    }
}

impl<I, P> JetBivariate1<I, P>
where
    I: JetOneLike + JetZeroLike,
{
    /// Construct the first independent coordinate.
    pub fn variable_axis0(value: I) -> Self {
        let zero = I::jet_zeros_like(&value);
        let one = I::jet_ones_like(&value);

        Self::from_components(value, one, zero.clone())
    }

    /// Construct the second independent coordinate.
    pub fn variable_axis1(value: I) -> Self {
        let zero = I::jet_zeros_like(&value);
        let one = I::jet_ones_like(&value);

        Self::from_components(value, zero.clone(), one)
    }
}

impl<I, P> JetBivariate1<I, P>
where
    I: JetZeroLike,
{
    pub fn from_x_derivatives(value: I, first: I) -> Self {
        let zero = I::jet_zeros_like(&value);

        Self::from_components(value, first, zero.clone())
    }

    pub fn from_y_derivatives(value: I, first: I) -> Self {
        let zero = I::jet_zeros_like(&value);

        Self::from_components(value, zero.clone(), first)
    }
}

impl<I, P> JetBivariate1<I, P>
where
    I: JetAdditive,
{
    pub(crate) fn add(&self, rhs: &Self) -> Self {
        Self::from_components(
            self.value().jet_add(rhs.value()),
            self.axis0().jet_add(rhs.axis0()),
            self.axis1().jet_add(rhs.axis1()),
        )
    }

    pub(crate) fn subtract(&self, rhs: &Self) -> Self {
        Self::from_components(
            self.value.jet_subtract(&rhs.value),
            self.axis0().jet_subtract(rhs.axis0()),
            self.axis1().jet_subtract(rhs.axis1()),
        )
    }

    pub(crate) fn negate(&self) -> Self {
        Self::from_components(
            self.value.jet_negate(),
            self.axis0().jet_negate(),
            self.axis1().jet_negate(),
        )
    }
}

impl<V, P> JetBivariate1<V, P> {
    /// Multiply this bivariate vector-valued jet by a scalar-valued jet.
    ///
    /// For `w = v s`, the derivatives are
    ///
    /// ```text
    /// w_x  = v_x s + v s_x
    /// w_y  = v_y s + v s_y
    /// ```
    pub(crate) fn multiply_by_scalar<S>(&self, scalar: &JetBivariate1<S, P>) -> Self
    where
        V: JetAdditive + JetMultiplyByScalar<S>,
        P: Clone,
    {
        let value = self.value().jet_multiply_by_scalar(scalar.value());

        let x = self
            .axis0()
            .jet_multiply_by_scalar(scalar.value())
            .jet_add(&self.value().jet_multiply_by_scalar(scalar.axis0()));

        let y = self
            .axis1()
            .jet_multiply_by_scalar(scalar.value())
            .jet_add(&self.value().jet_multiply_by_scalar(scalar.axis1()));

        Self::from_components(value, x, y)
    }
}

impl<I, P> JetBivariate1<I, P>
where
    I: JetBilinear,
{
    pub fn multiply(&self, rhs: &Self) -> Self {
        let value = self.value.jet_multiply(&rhs.value);

        let x = self
            .axis0()
            .jet_multiply(&rhs.value)
            .jet_add(&self.value.jet_multiply(rhs.axis0()));

        let y = self
            .axis1()
            .jet_multiply(&rhs.value)
            .jet_add(&self.value.jet_multiply(rhs.axis1()));

        Self::from_components(value, x, y)
    }
}

impl<I, P> JetBivariate1<I, P>
where
    I: JetZeroLike,
{
    pub fn constant(value: I) -> Self {
        let zero = I::jet_zeros_like(&value);

        Self::from_components(value, zero.clone(), zero.clone())
    }
}

impl<I, P> JetBivariate1<I, P>
where
    I: JetConstant + JetZeroLike,
{
    pub fn constant_like(source: &I, value: I::Scalar) -> Self {
        Self::constant(source.jet_constant_like(value))
    }
}

impl<I, P> JetBivariate1<I, P>
where
    I: JetScaleBy,
{
    /// Scale the value, gradient, and Hessian by a constant scalar.
    pub fn scale_by(&self, scalar: I::Scalar) -> Self {
        Self::from_components(
            self.value.jet_scale_by(scalar),
            self.axis0().jet_scale_by(scalar),
            self.axis1().jet_scale_by(scalar),
        )
    }
}

impl<I> JetBivariate1<I, RealParameter>
where
    I: JetConjugate,
{
    pub(crate) fn conjugated(&self) -> Self {
        Self::from_components(
            self.value.jet_conjugate(),
            self.axis0().jet_conjugate(),
            self.axis1().jet_conjugate(),
        )
    }
}

impl<I> JetBivariate1<I, RealParameter>
where
    I: JetRealPart,
{
    pub(crate) fn real(&self) -> JetBivariate1<I::RealOutput, RealParameter> {
        JetBivariate1::from_components(
            self.value.jet_real(),
            self.axis0().jet_real(),
            self.axis1().jet_real(),
        )
    }
}

impl<I, P> JetBivariate1<I, P>
where
    I: JetCrossProduct + JetAdditive,
{
    /// Compute the cross product of two bivariate second-order jets.
    pub(crate) fn cross(&self, rhs: &Self) -> Self {
        let value = self.value().jet_cross(rhs.value());

        let x = self
            .axis0()
            .jet_cross(rhs.value())
            .jet_add(&self.value().jet_cross(rhs.axis0()));

        let y = self
            .axis1()
            .jet_cross(rhs.value())
            .jet_add(&self.value().jet_cross(rhs.axis1()));

        Self::from_components(value, x, y)
    }
}

impl<I> JetBivariate1<I, RealParameter>
where
    I: JetHermitianProduct,
    I::Output: JetAdditive,
{
    /// Compute the Hermitian product of two bivariate second-order jets.
    pub(crate) fn hermitian_dot_product(
        &self,
        rhs: &Self,
    ) -> JetBivariate1<I::Output, RealParameter> {
        let value = self.value().jet_hermitian_product(rhs.value());

        let x = self
            .axis0()
            .jet_hermitian_product(rhs.value())
            .jet_add(&self.value().jet_hermitian_product(rhs.axis0()));

        let y = self
            .axis1()
            .jet_hermitian_product(rhs.value())
            .jet_add(&self.value().jet_hermitian_product(rhs.axis1()));

        JetBivariate1::from_components(value, x, y)
    }
}

impl<C, D, P> ArrayJetBivariate1<C, D, P>
where
    C: ComplexField,
    D: Dimension,
{
    pub(crate) fn compose_unary<F, F1>(&self, function: F, first: F1) -> Self
    where
        F: Fn(&C) -> C,
        F1: Fn(&C) -> C,
    {
        let value = self.value.mapv(|x| function(&x));
        let g1 = self.value.mapv(|x| first(&x));

        let x = &g1 * self.axis0();

        let y = &g1 * self.axis1();

        Self::from_components(value, x, y)
    }

    pub(crate) fn exp(&self) -> Self
    where
        C: Copy,
    {
        self.compose_unary(|x| x.exp(), |x| x.exp())
    }

    pub(crate) fn sin(&self) -> Self
    where
        C: Copy,
    {
        self.compose_unary(|x| x.sin(), |x| x.cos())
    }

    pub(crate) fn cos(&self) -> Self
    where
        C: Copy,
    {
        self.compose_unary(|x| x.cos(), |x| -x.sin())
    }

    pub(crate) fn sqrt(&self) -> Self
    where
        C: Copy,
    {
        self.compose_unary(
            |x| x.sqrt(),
            |x| C::one() / ((C::one() + C::one()) * x.sqrt()),
        )
    }

    pub(crate) fn reciprocal(&self) -> Self
    where
        C: Copy,
    {
        self.compose_unary(|x| C::one() / *x, |x| -C::one() / (*x * *x))
    }

    /// Divide two second-order jets elementwise.
    pub(crate) fn divide(&self, rhs: &Self) -> Self
    where
        C: Copy,
    {
        self.multiply(&rhs.reciprocal())
    }
}

impl<C, D, P> ArrayJetBivariate1<C, D, P>
where
    C: ComplexField,
    D: Dimension,
{
    pub(crate) fn compose_sampled_function(
        argument: &Self,
        expansion: FirstOrderExpansion<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self {
        let (value, function_first) = expansion.into_parts();

        let x = &function_first * argument.axis0();

        let y = &function_first * argument.axis1();

        Self::from_components(value, x, y)
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

    type RealJet = JetBivariate1<A0, RealParameter>;

    type HolomorphicJet = JetBivariate1<A0, HolomorphicParameter>;

    const EPSILON: f64 = 1.0e-11;
    const FD_EPSILON: f64 = 1.0e-5;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn r(value: f64) -> C {
        c(value, 0.0)
    }

    fn a(real: f64, imaginary: f64) -> A0 {
        arr0(c(real, imaginary))
    }

    fn assert_close(actual: C, expected: C) {
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

    fn assert_fd_close(actual: C, expected: C) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = FD_EPSILON,
            max_relative = FD_EPSILON,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = FD_EPSILON,
            max_relative = FD_EPSILON,
        );
    }

    fn finite_difference_x<F>(function: &F, x: f64, y: f64, step: f64) -> C
    where
        F: Fn(f64, f64) -> C,
    {
        (function(x + step, y) - function(x - step, y)) / (2.0 * step)
    }

    fn finite_difference_y<F>(function: &F, x: f64, y: f64, step: f64) -> C
    where
        F: Fn(f64, f64) -> C,
    {
        (function(x, y + step) - function(x, y - step)) / (2.0 * step)
    }

    #[test]
    fn multiplication_applies_bivariate_product_rules() {
        let f_value = c(1.0, 2.0);
        let fx = c(0.5, -0.2);
        let fy = c(-0.3, 0.7);

        let g_value = c(2.0, -1.0);
        let gx = c(-0.1, 0.4);
        let gy = c(0.9, -0.3);

        let f: RealJet = JetBivariate1::from_components(arr0(f_value), arr0(fx), arr0(fy));

        let g = JetBivariate1::from_components(arr0(g_value), arr0(gx), arr0(gy));

        let result = f.multiply(&g);

        assert_close(result.value()[()], f_value * g_value);

        assert_close(result.axis0()[()], fx * g_value + f_value * gx);

        assert_close(result.axis1()[()], fy * g_value + f_value * gy);
    }

    #[test]
    fn unary_composition_applies_bivariate_chain_rule() {
        let value = c(0.7, -0.4);
        let x = c(1.3, 0.2);
        let y = c(-0.6, 0.8);

        let jet: HolomorphicJet = JetBivariate1::from_components(arr0(value), arr0(x), arr0(y));

        let result = jet.sin();

        let first = value.cos();

        assert_close(result.value()[()], value.sin());
        assert_close(result.axis0()[()], first * x);
        assert_close(result.axis1()[()], first * y);
    }

    #[test]
    fn composite_expression_matches_finite_differences() {
        let function = |x: f64, y: f64| {
            let z = c(x, 0.2 * y);

            z.exp() * (z + c(0.3 * x * y, 0.0)).sin()
        };

        let x0 = 0.7;
        let y0 = -0.4;

        let x: RealJet = JetBivariate1::variable_axis0(arr0(r(x0)));

        let y: RealJet = JetBivariate1::variable_axis1(arr0(r(y0)));

        let imaginary_scale = c(0.0, 0.2);

        let z = x.add(&y.scale_by(imaginary_scale));

        let xy = x.multiply(&y).scale_by(r(0.3));

        let result = z.clone().exp().multiply(&z.add(&xy).sin());

        let h = 1.0e-4;

        let fx = (function(x0 + h, y0) - function(x0 - h, y0)) / (2.0 * h);

        let fy = (function(x0, y0 + h) - function(x0, y0 - h)) / (2.0 * h);

        assert_fd_close(result.value()[()], function(x0, y0));
        assert_fd_close(result.axis0()[()], fx);
        assert_fd_close(result.axis1()[()], fy);
    }

    #[test]
    fn conjugation_propagates_real_coordinate_derivatives() {
        let jet: RealJet = JetBivariate1::from_components(a(1.0, 2.0), a(3.0, -4.0), a(-5.0, 6.0));

        let result = jet.conjugated();

        assert_close(result.value()[()], c(1.0, -2.0));
        assert_close(result.axis0()[()], c(3.0, 4.0));
        assert_close(result.axis1()[()], c(-5.0, -6.0));
    }

    // ---------------------------------------------------------------------
    // Constructors
    // ---------------------------------------------------------------------

    #[test]
    fn variable_axis0_has_expected_seed() {
        let jet: HolomorphicJet = JetBivariate1::variable_axis0(a(2.0, 3.0));

        assert_close(jet.value()[()], c(2.0, 3.0));
        assert_close(jet.axis0()[()], r(1.0));
        assert_close(jet.axis1()[()], r(0.0));
    }

    #[test]
    fn variable_axis1_has_expected_seed() {
        let jet: HolomorphicJet = JetBivariate1::variable_axis1(a(2.0, 3.0));

        assert_close(jet.value()[()], c(2.0, 3.0));
        assert_close(jet.axis0()[()], r(0.0));
        assert_close(jet.axis1()[()], r(1.0));
    }

    #[test]
    fn constant_has_zero_gradient_and_hessian() {
        let jet: RealJet = JetBivariate1::constant(a(4.0, -2.0));

        assert_close(jet.value()[()], c(4.0, -2.0));
        assert_close(jet.axis0()[()], r(0.0));
        assert_close(jet.axis1()[()], r(0.0));
    }

    #[test]
    fn from_x_derivatives_sets_only_x_components() {
        let jet: RealJet = JetBivariate1::from_x_derivatives(a(2.0, 1.0), a(3.0, -1.0));

        assert_close(jet.value()[()], c(2.0, 1.0));
        assert_close(jet.axis0()[()], c(3.0, -1.0));
        assert_close(jet.axis1()[()], r(0.0));
    }

    #[test]
    fn from_y_derivatives_sets_only_y_components() {
        let jet: RealJet = JetBivariate1::from_y_derivatives(a(2.0, 1.0), a(3.0, -1.0));

        assert_close(jet.value()[()], c(2.0, 1.0));
        assert_close(jet.axis0()[()], r(0.0));
        assert_close(jet.axis1()[()], c(3.0, -1.0));
    }

    // ---------------------------------------------------------------------
    // Additive operations and scaling
    // ---------------------------------------------------------------------

    fn sample_jet() -> RealJet {
        JetBivariate1::from_components(a(1.0, 2.0), a(3.0, 4.0), a(5.0, 6.0))
    }

    #[test]
    fn addition_is_componentwise() {
        let left = sample_jet();

        let right = JetBivariate1::from_components(a(13.0, 14.0), a(15.0, 16.0), a(17.0, 18.0));

        let result = left.add(&right);

        assert_close(result.value()[()], c(14.0, 16.0));
        assert_close(result.axis0()[()], c(18.0, 20.0));
        assert_close(result.axis1()[()], c(22.0, 24.0));
    }

    #[test]
    fn subtraction_is_componentwise() {
        let left = sample_jet();
        let result = left.subtract(&left);

        assert_close(result.value()[()], r(0.0));
        assert_close(result.axis0()[()], r(0.0));
        assert_close(result.axis1()[()], r(0.0));
    }

    #[test]
    fn negation_negates_every_component() {
        let original = sample_jet();
        let result = original.negate();

        assert_close(result.value()[()], c(-1.0, -2.0));
        assert_close(result.axis0()[()], c(-3.0, -4.0));
        assert_close(result.axis1()[()], c(-5.0, -6.0));
    }

    #[test]
    fn scaling_scales_every_component() {
        let original = sample_jet();
        let scale = c(2.0, -1.0);

        let result = original.scale_by(scale);

        assert_close(result.value()[()], c(1.0, 2.0) * scale);

        assert_close(result.axis0()[()], c(3.0, 4.0) * scale);

        assert_close(result.axis1()[()], c(5.0, 6.0) * scale);
    }

    // ---------------------------------------------------------------------
    // Product rules
    // ---------------------------------------------------------------------

    #[test]
    fn multiplication_applies_all_bivariate_product_rules() {
        let f_value = c(1.0, 2.0);
        let fx = c(0.5, -0.2);
        let fy = c(-0.3, 0.7);

        let g_value = c(2.0, -1.0);
        let gx = c(-0.1, 0.4);
        let gy = c(0.9, -0.3);

        let f: RealJet = JetBivariate1::from_components(arr0(f_value), arr0(fx), arr0(fy));

        let g = JetBivariate1::from_components(arr0(g_value), arr0(gx), arr0(gy));

        let result = f.multiply(&g);

        assert_close(result.value()[()], f_value * g_value);

        assert_close(result.axis0()[()], fx * g_value + f_value * gx);

        assert_close(result.axis1()[()], fy * g_value + f_value * gy);
    }

    #[test]
    fn multiplying_by_constant_preserves_scaled_derivatives() {
        let jet = sample_jet();

        let constant: RealJet = JetBivariate1::constant(a(2.0, -1.0));

        let result = jet.multiply(&constant);
        let scale = c(2.0, -1.0);

        assert_close(result.value()[()], c(1.0, 2.0) * scale);

        assert_close(result.axis0()[()], c(3.0, 4.0) * scale);

        assert_close(result.axis1()[()], c(5.0, 6.0) * scale);
    }

    #[test]
    fn product_of_coordinate_variables_has_unit_mixed_derivative() {
        let x: RealJet = JetBivariate1::variable_axis0(a(2.0, 0.0));

        let y: RealJet = JetBivariate1::variable_axis1(a(3.0, 0.0));

        let result = x.multiply(&y);

        assert_close(result.value()[()], r(6.0));
        assert_close(result.axis0()[()], r(3.0));
        assert_close(result.axis1()[()], r(2.0));
    }

    #[test]
    fn square_of_x_has_expected_pure_second_derivative() {
        let x: RealJet = JetBivariate1::variable_axis0(a(3.0, 0.0));

        let result = x.multiply(&x);

        assert_close(result.value()[()], r(9.0));
        assert_close(result.axis0()[()], r(6.0));
        assert_close(result.axis1()[()], r(0.0));
    }

    // ---------------------------------------------------------------------
    // Reciprocal and division
    // ---------------------------------------------------------------------

    #[test]
    fn reciprocal_matches_bivariate_chain_rule() {
        let value = c(2.0, 0.5);
        let x = c(0.7, -0.2);
        let y = c(-0.3, 0.4);

        let jet: HolomorphicJet = JetBivariate1::from_components(arr0(value), arr0(x), arr0(y));

        let result = jet.reciprocal();

        let inverse = r(1.0) / value;
        let first = -r(1.0) / (value * value);

        assert_close(result.value()[()], inverse);
        assert_close(result.axis0()[()], first * x);
        assert_close(result.axis1()[()], first * y);
    }

    #[test]
    fn reciprocal_of_reciprocal_recovers_original() {
        let original: HolomorphicJet =
            JetBivariate1::from_components(a(2.0, 0.5), a(0.7, -0.2), a(-0.3, 0.4));

        let result = original.reciprocal().reciprocal();

        assert_close(result.value()[()], original.value()[()]);

        assert_close(result.axis0()[()], original.axis0()[()]);

        assert_close(result.axis1()[()], original.axis1()[()]);
    }

    #[test]
    fn division_agrees_with_multiplication_by_reciprocal() {
        let numerator: HolomorphicJet =
            JetBivariate1::from_components(a(2.0, 1.0), a(0.5, -0.3), a(-0.2, 0.7));

        let denominator: HolomorphicJet =
            JetBivariate1::from_components(a(3.0, -0.5), a(-0.4, 0.2), a(0.6, 0.1));

        let direct = numerator.divide(&denominator);

        let expanded = numerator.multiply(&denominator.reciprocal());

        assert_eq!(direct, expanded);
    }

    // ---------------------------------------------------------------------
    // Unary composition
    // ---------------------------------------------------------------------

    #[test]
    fn exponential_applies_bivariate_chain_rule() {
        let value = c(0.7, -0.4);
        let x = c(1.3, 0.2);
        let y = c(-0.6, 0.8);

        let jet: HolomorphicJet = JetBivariate1::from_components(arr0(value), arr0(x), arr0(y));

        let result = jet.exp();
        let exponential = value.exp();

        assert_close(result.value()[()], exponential);
        assert_close(result.axis0()[()], exponential * x);
        assert_close(result.axis1()[()], exponential * y);
    }

    #[test]
    fn sine_applies_bivariate_chain_rule() {
        let value = c(0.7, -0.4);
        let x = c(1.3, 0.2);
        let y = c(-0.6, 0.8);

        let jet: HolomorphicJet = JetBivariate1::from_components(arr0(value), arr0(x), arr0(y));

        let result = jet.sin();

        let first = value.cos();

        assert_close(result.value()[()], value.sin());
        assert_close(result.axis0()[()], first * x);
        assert_close(result.axis1()[()], first * y);
    }

    #[test]
    fn cosine_applies_bivariate_chain_rule() {
        let value = c(0.7, -0.4);
        let x = c(1.3, 0.2);
        let y = c(-0.6, 0.8);

        let jet: HolomorphicJet = JetBivariate1::from_components(arr0(value), arr0(x), arr0(y));

        let result = jet.cos();

        let first = -value.sin();

        assert_close(result.value()[()], value.cos());
        assert_close(result.axis0()[()], first * x);
        assert_close(result.axis1()[()], first * y);
    }

    #[test]
    fn sqrt_applies_bivariate_chain_rule() {
        let value = c(2.0, 0.5);
        let x = c(0.7, -0.2);
        let y = c(-0.3, 0.4);

        let jet: HolomorphicJet = JetBivariate1::from_components(arr0(value), arr0(x), arr0(y));

        let result = jet.sqrt();

        let root = value.sqrt();
        let first = r(1.0) / (r(2.0) * root);

        assert_close(result.value()[()], root);
        assert_close(result.axis0()[()], first * x);
        assert_close(result.axis1()[()], first * y);
    }

    // ---------------------------------------------------------------------
    // Sampled-function composition
    // ---------------------------------------------------------------------

    #[test]
    fn sampled_function_composition_matches_unary_composition() {
        let value = c(0.7, -0.4);

        let argument: HolomorphicJet =
            JetBivariate1::from_components(arr0(value), a(1.3, 0.2), a(-0.6, 0.8));

        let expansion = FirstOrderExpansion::new(arr0(value.sin()), arr0(value.cos()));

        let sampled = HolomorphicJet::compose_sampled_function(&argument, expansion);

        let direct = argument.sin();

        assert_close(sampled.value()[()], direct.value()[()]);
        assert_close(sampled.axis0()[()], direct.axis0()[()]);
        assert_close(sampled.axis1()[()], direct.axis1()[()]);
    }

    // ---------------------------------------------------------------------
    // Coordinate symmetry
    // ---------------------------------------------------------------------

    fn swap_coordinates(jet: &RealJet) -> RealJet {
        JetBivariate1::from_components(
            jet.value().clone(),
            jet.axis1().clone(),
            jet.axis0().clone(),
        )
    }

    #[test]
    fn multiplication_is_equivariant_under_coordinate_exchange() {
        let left = sample_jet();

        let right = JetBivariate1::from_components(a(2.0, -1.0), a(0.5, 0.3), a(-0.7, 0.1));

        let original = swap_coordinates(&left.multiply(&right));

        let swapped = swap_coordinates(&left).multiply(&swap_coordinates(&right));

        assert_eq!(original, swapped);
    }

    #[test]
    fn unary_composition_is_equivariant_under_coordinate_exchange() {
        let original = swap_coordinates(&sample_jet().sin());

        let swapped = swap_coordinates(&sample_jet()).sin();

        assert_eq!(original, swapped);
    }

    // ---------------------------------------------------------------------
    // Array behaviour
    // ---------------------------------------------------------------------

    #[test]
    fn array_operations_preserve_shape() {
        type ArrayJet = JetBivariate1<Array1<C>, RealParameter>;

        let value = array![c(1.0, 0.2), c(2.0, -0.4), c(3.0, 0.6),];

        let jet: ArrayJet = JetBivariate1::variable_axis0(value.clone());

        let result = jet.exp();

        assert_eq!(result.value().raw_dim(), value.raw_dim(),);

        assert_eq!(result.axis0().raw_dim(), value.raw_dim(),);

        assert_eq!(result.axis1().raw_dim(), value.raw_dim(),);
    }

    #[test]
    fn array_unary_operations_are_elementwise() {
        type ArrayJet = JetBivariate1<Array1<C>, HolomorphicParameter>;

        let values = array![c(1.0, 0.2), c(2.0, -0.4), c(3.0, 0.6),];

        let x = array![c(0.1, 0.2), c(-0.3, 0.4), c(0.5, -0.6),];

        let y = array![c(-0.2, 0.1), c(0.4, -0.3), c(-0.6, 0.5),];

        let jet: ArrayJet = JetBivariate1::from_components(values.clone(), x.clone(), y.clone());

        let result = jet.exp();

        for index in 0..values.len() {
            let exponential = values[index].exp();

            assert_close(result.value()[index], exponential);

            assert_close(result.axis0()[index], exponential * x[index]);

            assert_close(result.axis1()[index], exponential * y[index]);
        }
    }

    // ---------------------------------------------------------------------
    // Real-coordinate-only operations
    // ---------------------------------------------------------------------

    #[test]
    fn conjugation_conjugates_every_component() {
        let result = sample_jet().conjugated();

        assert_close(result.value()[()], c(1.0, -2.0));
        assert_close(result.axis0()[()], c(3.0, -4.0));
        assert_close(result.axis1()[()], c(5.0, -6.0));
    }

    #[test]
    fn real_extracts_every_component() {
        let result = sample_jet().real();

        assert_relative_eq!(result.value()[()], 1.0, epsilon = EPSILON,);

        assert_relative_eq!(result.axis0()[()], 3.0, epsilon = EPSILON,);

        assert_relative_eq!(result.axis1()[()], 5.0, epsilon = EPSILON,);
    }

    // ---------------------------------------------------------------------
    // Composite finite-difference validation
    // ---------------------------------------------------------------------

    #[test]
    fn composite_expression_matches_all_finite_differences() {
        let x0 = 0.7;
        let y0 = -0.4;

        let x: RealJet = JetBivariate1::variable_axis0(arr0(r(x0)));

        let y: RealJet = JetBivariate1::variable_axis1(arr0(r(y0)));

        let z = x.add(&y.scale_by(c(0.0, 0.2)));

        let coupling = x.multiply(&y).scale_by(r(0.3));

        let numerator = z.clone().exp().multiply(&z.add(&coupling).sin());

        let denominator = x
            .multiply(&x)
            .add(&y.multiply(&y))
            .add(&JetBivariate1::constant(arr0(r(2.0))))
            .sqrt();

        let result = numerator.divide(&denominator);

        let function = |x: f64, y: f64| {
            let z = c(x, 0.2 * y);
            let coupling = r(0.3 * x * y);

            z.exp() * (z + coupling).sin() / r(x * x + y * y + 2.0).sqrt()
        };

        let first_step = 1.0e-5;

        assert_fd_close(result.value()[()], function(x0, y0));

        assert_fd_close(
            result.axis0()[()],
            finite_difference_x(&function, x0, y0, first_step),
        );

        assert_fd_close(
            result.axis1()[()],
            finite_difference_y(&function, x0, y0, first_step),
        );
    }
}
