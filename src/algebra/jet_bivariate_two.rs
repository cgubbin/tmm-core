//! Bivariate second-order differential jets.
//!
//! [`JetBivariate2`] propagates first and second derivatives with respect to
//! two independent scalar coordinates. It stores a value, a two-component
//! gradient, and the three independent entries of a symmetric Hessian:
//!
//! ```text
//! (f, f₀, f₁, f₀₀, f₀₁, f₁₁)
//! ```
//!
//! The coordinates are represented abstractly as `axis0` and `axis1`.
//! Application-level code assigns their physical meanings.
//!
//! The Hessian representation assumes commuting mixed partial derivatives.
//!
//! Parameter semantics are represented by [`RealParameter`] and
//! [`HolomorphicParameter`], as for the other jet families.

use crate::algebra::{JetMultiplyByScalar, exprel, exprel_first, exprel_second};
use crate::differential::{BivariateGradient, BivariateHessian};

use super::{
    HolomorphicParameter, JetAdditive, JetBilinear, JetConjugate, JetConstant, JetCrossProduct,
    JetHermitianProduct, JetOneLike, JetRealPart, JetScaleBy, JetZeroLike, RealParameter,
    SecondOrderExpansion,
};

use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, OwnedRepr};
use num_traits::FromPrimitive;
use num_traits::float::FloatCore;
use std::marker::PhantomData;

pub(crate) type ArrayJetBivariate2<C, D, P> = JetBivariate2<ArrayBase<OwnedRepr<C>, D>, P>;

pub(crate) type PhysicalJetBivariate2<C, D> = ArrayJetBivariate2<C, D, RealParameter>;
pub(crate) type ModeJetBivariate2<C, D> = ArrayJetBivariate2<C, D, HolomorphicParameter>;

#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct JetBivariate2<A, P> {
    value: A,

    first: BivariateGradient<A>,

    second: BivariateHessian<A>,

    parameter: PhantomData<P>,
}

impl<I, P> JetBivariate2<I, P> {
    pub(crate) fn from_parts(
        value: I,
        first: BivariateGradient<I>,
        second: BivariateHessian<I>,
    ) -> Self {
        Self {
            value,
            first,
            second,
            parameter: PhantomData,
        }
    }

    pub(crate) fn from_components(
        value: I,
        axis0: I,
        axis1: I,
        axis0_axis0: I,
        axis0_axis1: I,
        axis1_axis1: I,
    ) -> Self {
        Self::from_parts(
            value,
            BivariateGradient::new(axis0, axis1),
            BivariateHessian::new(axis0_axis0, axis0_axis1, axis1_axis1),
        )
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

    pub(crate) fn axis0_axis0(&self) -> &I {
        self.second.axis0_axis0()
    }

    pub(crate) fn axis0_axis1(&self) -> &I {
        self.second.axis0_axis1()
    }

    pub(crate) fn axis1_axis1(&self) -> &I {
        self.second.axis1_axis1()
    }

    pub(crate) fn first(&self) -> &BivariateGradient<I> {
        &self.first
    }

    pub(crate) fn second(&self) -> &BivariateHessian<I> {
        &self.second
    }

    pub(crate) fn into_parts(self) -> (I, BivariateGradient<I>, BivariateHessian<I>) {
        (self.value, self.first, self.second)
    }
}

impl<I, P> JetBivariate2<I, P>
where
    I: JetOneLike + JetZeroLike,
{
    /// Construct the first independent coordinate.
    pub(crate) fn variable_axis0(value: I) -> Self {
        let zero = I::jet_zeros_like(&value);
        let one = I::jet_ones_like(&value);

        Self::from_components(value, one, zero.clone(), zero.clone(), zero.clone(), zero)
    }

    /// Construct the second independent coordinate.
    pub(crate) fn variable_axis1(value: I) -> Self {
        let zero = I::jet_zeros_like(&value);
        let one = I::jet_ones_like(&value);

        Self::from_components(value, zero.clone(), one, zero.clone(), zero.clone(), zero)
    }
}

impl<I, P> JetBivariate2<I, P>
where
    I: JetZeroLike,
{
    pub(crate) fn from_axis0_derivatives(value: I, first: I, second: I) -> Self {
        let zero = I::jet_zeros_like(&value);

        Self::from_components(value, first, zero.clone(), second, zero.clone(), zero)
    }

    pub(crate) fn from_axis1_derivatives(value: I, first: I, second: I) -> Self {
        let zero = I::jet_zeros_like(&value);

        Self::from_components(value, zero.clone(), first, zero.clone(), zero, second)
    }
}

impl<I, P> JetBivariate2<I, P>
where
    I: JetAdditive,
{
    pub(crate) fn add(&self, rhs: &Self) -> Self {
        Self::from_components(
            self.value().jet_add(rhs.value()),
            self.axis0().jet_add(rhs.axis0()),
            self.axis1().jet_add(rhs.axis1()),
            self.axis0_axis0().jet_add(rhs.axis0_axis0()),
            self.axis0_axis1().jet_add(rhs.axis0_axis1()),
            self.axis1_axis1().jet_add(rhs.axis1_axis1()),
        )
    }

    pub(crate) fn subtract(&self, rhs: &Self) -> Self {
        Self::from_components(
            self.value.jet_subtract(&rhs.value),
            self.axis0().jet_subtract(rhs.axis0()),
            self.axis1().jet_subtract(rhs.axis1()),
            self.axis0_axis0().jet_subtract(rhs.axis0_axis0()),
            self.axis0_axis1().jet_subtract(rhs.axis0_axis1()),
            self.axis1_axis1().jet_subtract(rhs.axis1_axis1()),
        )
    }

    pub(crate) fn negate(&self) -> Self {
        Self::from_components(
            self.value.jet_negate(),
            self.axis0().jet_negate(),
            self.axis1().jet_negate(),
            self.axis0_axis0().jet_negate(),
            self.axis0_axis1().jet_negate(),
            self.axis1_axis1().jet_negate(),
        )
    }
}

impl<V, P> JetBivariate2<V, P> {
    /// Multiply this bivariate vector-valued jet by a scalar-valued jet.
    ///
    /// For `w = v s`, the derivatives are
    ///
    /// ```text
    /// w_x  = v_x s + v s_x
    /// w_y  = v_y s + v s_y
    ///
    /// w_axis0_axis0 = v_xx s + 2 v_x s_x + v s_xx
    /// w_axis0_axis1 = v_xy s + v_x s_y + v_y s_x + v s_xy
    /// w_axis1_axis1 = v_yy s + 2 v_y s_y + v s_yy
    /// ```
    pub(crate) fn multiply_by_scalar<S>(&self, scalar: &JetBivariate2<S, P>) -> Self
    where
        V: JetAdditive + JetMultiplyByScalar<S>,
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

        let axis0_axis0 = self
            .axis0_axis0()
            .jet_multiply_by_scalar(scalar.value())
            .jet_add(
                &self
                    .axis0()
                    .jet_multiply_by_scalar(scalar.axis0())
                    .jet_double(),
            )
            .jet_add(&self.value().jet_multiply_by_scalar(scalar.axis0_axis0()));

        let axis0_axis1 = self
            .axis0_axis1()
            .jet_multiply_by_scalar(scalar.value())
            .jet_add(&self.axis0().jet_multiply_by_scalar(scalar.axis1()))
            .jet_add(&self.axis1().jet_multiply_by_scalar(scalar.axis0()))
            .jet_add(&self.value().jet_multiply_by_scalar(scalar.axis0_axis1()));

        let axis1_axis1 = self
            .axis1_axis1()
            .jet_multiply_by_scalar(scalar.value())
            .jet_add(
                &self
                    .axis1()
                    .jet_multiply_by_scalar(scalar.axis1())
                    .jet_double(),
            )
            .jet_add(&self.value().jet_multiply_by_scalar(scalar.axis1_axis1()));

        Self::from_components(value, x, y, axis0_axis0, axis0_axis1, axis1_axis1)
    }
}

impl<I, P> JetBivariate2<I, P>
where
    I: JetBilinear,
{
    pub(crate) fn multiply(&self, rhs: &Self) -> Self {
        let value = self.value.jet_multiply(&rhs.value);

        let x = self
            .axis0()
            .jet_multiply(&rhs.value)
            .jet_add(&self.value.jet_multiply(rhs.axis0()));

        let y = self
            .axis1()
            .jet_multiply(&rhs.value)
            .jet_add(&self.value.jet_multiply(rhs.axis1()));

        let axis0_axis0 = self
            .axis0_axis0()
            .jet_multiply(&rhs.value)
            .jet_add(&self.axis0().jet_multiply(rhs.axis0()).jet_double())
            .jet_add(&self.value.jet_multiply(rhs.axis0_axis0()));

        let axis0_axis1 = self
            .axis0_axis1()
            .jet_multiply(&rhs.value)
            .jet_add(&self.axis0().jet_multiply(rhs.axis1()))
            .jet_add(&self.axis1().jet_multiply(rhs.axis0()))
            .jet_add(&self.value.jet_multiply(rhs.axis0_axis1()));

        let axis1_axis1 = self
            .axis1_axis1()
            .jet_multiply(&rhs.value)
            .jet_add(&self.axis1().jet_multiply(rhs.axis1()).jet_double())
            .jet_add(&self.value.jet_multiply(rhs.axis1_axis1()));

        Self::from_components(value, x, y, axis0_axis0, axis0_axis1, axis1_axis1)
    }
}

impl<I, P> JetBivariate2<I, P>
where
    I: JetZeroLike,
{
    pub(crate) fn constant(value: I) -> Self {
        let zero = I::jet_zeros_like(&value);

        Self::from_components(
            value,
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero.clone(),
            zero,
        )
    }
}

impl<I, P> JetBivariate2<I, P>
where
    I: JetConstant + JetZeroLike,
{
    pub(crate) fn constant_like(source: &I, value: I::Scalar) -> Self {
        Self::constant(source.jet_constant_like(value))
    }
}

impl<I, P> JetBivariate2<I, P>
where
    I: JetScaleBy,
{
    /// Scale the value, gradient, and Hessian by a constant scalar.
    pub(crate) fn scale_by(&self, scalar: I::Scalar) -> Self {
        Self::from_components(
            self.value.jet_scale_by(scalar),
            self.axis0().jet_scale_by(scalar),
            self.axis1().jet_scale_by(scalar),
            self.axis0_axis0().jet_scale_by(scalar),
            self.axis0_axis1().jet_scale_by(scalar),
            self.axis1_axis1().jet_scale_by(scalar),
        )
    }
}

impl<I> JetBivariate2<I, RealParameter>
where
    I: JetConjugate,
{
    pub(crate) fn conjugated(&self) -> Self {
        Self::from_components(
            self.value.jet_conjugate(),
            self.axis0().jet_conjugate(),
            self.axis1().jet_conjugate(),
            self.axis0_axis0().jet_conjugate(),
            self.axis0_axis1().jet_conjugate(),
            self.axis1_axis1().jet_conjugate(),
        )
    }
}

impl<I> JetBivariate2<I, RealParameter>
where
    I: JetRealPart,
{
    pub(crate) fn real(&self) -> JetBivariate2<I::RealOutput, RealParameter> {
        JetBivariate2::from_components(
            self.value.jet_real(),
            self.axis0().jet_real(),
            self.axis1().jet_real(),
            self.axis0_axis0().jet_real(),
            self.axis0_axis1().jet_real(),
            self.axis1_axis1().jet_real(),
        )
    }

    pub(crate) fn imaginary(&self) -> JetBivariate2<I::RealOutput, RealParameter> {
        JetBivariate2::from_components(
            self.value.jet_imaginary(),
            self.axis0().jet_imaginary(),
            self.axis1().jet_imaginary(),
            self.axis0_axis0().jet_imaginary(),
            self.axis0_axis1().jet_imaginary(),
            self.axis1_axis1().jet_imaginary(),
        )
    }
}

impl<I, P> JetBivariate2<I, P>
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

        let mixed_axis0 = self.axis0().jet_cross(rhs.axis0());

        let axis0_axis0 = self
            .axis0_axis0()
            .jet_cross(rhs.value())
            .jet_add(&mixed_axis0)
            .jet_add(&mixed_axis0)
            .jet_add(&self.value().jet_cross(rhs.axis0_axis0()));

        let mixed_kx = self.axis1().jet_cross(rhs.axis1());

        let axis1_axis1 = self
            .axis1_axis1()
            .jet_cross(rhs.value())
            .jet_add(&mixed_kx)
            .jet_add(&mixed_kx)
            .jet_add(&self.value().jet_cross(rhs.axis1_axis1()));

        let axis0_axis1 = self
            .axis0_axis1()
            .jet_cross(rhs.value())
            .jet_add(&self.axis0().jet_cross(rhs.axis1()))
            .jet_add(&self.axis1().jet_cross(rhs.axis0()))
            .jet_add(&self.value().jet_cross(rhs.axis0_axis1()));

        Self::from_components(value, x, y, axis0_axis0, axis0_axis1, axis1_axis1)
    }
}

impl<I> JetBivariate2<I, RealParameter>
where
    I: JetHermitianProduct,
    I::Output: JetAdditive,
{
    /// Compute the Hermitian product of two bivariate second-order jets.
    pub(crate) fn hermitian_dot_product(
        &self,
        rhs: &Self,
    ) -> JetBivariate2<I::Output, RealParameter> {
        let value = self.value().jet_hermitian_product(rhs.value());

        let axis0 = self
            .axis0()
            .jet_hermitian_product(rhs.value())
            .jet_add(&self.value().jet_hermitian_product(rhs.axis0()));

        let axis1 = self
            .axis1()
            .jet_hermitian_product(rhs.value())
            .jet_add(&self.value().jet_hermitian_product(rhs.axis1()));

        let mixed_axis0 = self.axis0().jet_hermitian_product(rhs.axis0());

        let axis0_axis0 = self
            .axis0_axis0()
            .jet_hermitian_product(rhs.value())
            .jet_add(&mixed_axis0)
            .jet_add(&mixed_axis0)
            .jet_add(&self.value().jet_hermitian_product(rhs.axis0_axis0()));

        let mixed_kx = self.axis1().jet_hermitian_product(rhs.axis1());

        let axis1_axis1 = self
            .axis1_axis1()
            .jet_hermitian_product(rhs.value())
            .jet_add(&mixed_kx)
            .jet_add(&mixed_kx)
            .jet_add(&self.value().jet_hermitian_product(rhs.axis1_axis1()));

        let axis0_axis1 = self
            .axis0_axis1()
            .jet_hermitian_product(rhs.value())
            .jet_add(&self.axis0().jet_hermitian_product(rhs.axis1()))
            .jet_add(&self.axis1().jet_hermitian_product(rhs.axis0()))
            .jet_add(&self.value().jet_hermitian_product(rhs.axis0_axis1()));

        JetBivariate2::from_components(value, axis0, axis1, axis0_axis0, axis0_axis1, axis1_axis1)
    }
}

impl<C, D, P> ArrayJetBivariate2<C, D, P>
where
    C: ComplexField,
    D: Dimension,
{
    pub(crate) fn compose_unary<F, F1, F2>(&self, function: F, first: F1, second: F2) -> Self
    where
        F: Fn(&C) -> C,
        F1: Fn(&C) -> C,
        F2: Fn(&C) -> C,
    {
        let value = self.value.mapv(|x| function(&x));
        let g1 = self.value.mapv(|x| first(&x));
        let g2 = self.value.mapv(|x| second(&x));

        let axis0 = &g1 * self.axis0();

        let axis1 = &g1 * self.axis1();

        let axis0_axis0 = &g2 * self.axis0() * self.axis0() + &g1 * self.axis0_axis0();

        let axis0_axis1 = &g2 * self.axis0() * self.axis1() + &g1 * self.axis0_axis1();

        let axis1_axis1 = &g2 * self.axis1() * self.axis1() + &g1 * self.axis1_axis1();

        Self::from_components(value, axis0, axis1, axis0_axis0, axis0_axis1, axis1_axis1)
    }

    pub(crate) fn exprel(&self) -> Self
    where
        C: Copy,
        C::RealField: FloatCore + FromPrimitive,
    {
        self.compose_unary(|x| exprel(*x), |x| exprel_first(*x), |x| exprel_second(*x))
    }

    pub(crate) fn exp(&self) -> Self
    where
        C: Copy,
    {
        self.compose_unary(|x| x.exp(), |x| x.exp(), |x| x.exp())
    }

    pub(crate) fn sin(&self) -> Self
    where
        C: Copy,
    {
        self.compose_unary(|x| x.sin(), |x| x.cos(), |x| -x.sin())
    }

    pub(crate) fn cos(&self) -> Self
    where
        C: Copy,
    {
        self.compose_unary(|x| x.cos(), |x| -x.sin(), |x| -x.cos())
    }

    pub(crate) fn sqrt(&self) -> Self
    where
        C: Copy,
    {
        self.compose_unary(
            |x| x.sqrt(),
            |x| C::one() / ((C::one() + C::one()) * x.sqrt()),
            |x| {
                let two = C::one() + C::one();
                let four = two * two;

                -C::one() / (four * x.sqrt() * *x)
            },
        )
    }

    pub(crate) fn reciprocal(&self) -> Self
    where
        C: Copy,
    {
        self.compose_unary(
            |x| C::one() / *x,
            |x| -C::one() / (*x * *x),
            |x| {
                let two = C::one() + C::one();
                two / (*x * *x * *x)
            },
        )
    }

    /// Divide two second-order jets elementwise.
    pub(crate) fn divide(&self, rhs: &Self) -> Self
    where
        C: Copy,
    {
        self.multiply(&rhs.reciprocal())
    }
}

impl<C, D, P> ArrayJetBivariate2<C, D, P>
where
    C: ComplexField,
    D: Dimension,
{
    pub(crate) fn compose_sampled_function(
        argument: &Self,
        expansion: SecondOrderExpansion<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self {
        let (value, function_first, function_second) = expansion.into_parts();

        let x = &function_first * argument.axis0();

        let y = &function_first * argument.axis1();

        let argument_axis0_squared = argument.axis0() * argument.axis0();

        let argument_axis1_squared = argument.axis1() * argument.axis1();

        let argument_axis0_axis1 = argument.axis0() * argument.axis1();

        let axis0_axis0 =
            &function_second * &argument_axis0_squared + &function_first * argument.axis0_axis0();

        let axis0_axis1 =
            &function_second * &argument_axis0_axis1 + &function_first * argument.axis0_axis1();

        let axis1_axis1 =
            &function_second * &argument_axis1_squared + &function_first * argument.axis1_axis1();

        Self::from_components(value, x, y, axis0_axis0, axis0_axis1, axis1_axis1)
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

    type RealJet = JetBivariate2<A0, RealParameter>;

    type HolomorphicJet = JetBivariate2<A0, HolomorphicParameter>;

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

    fn finite_difference_axis0<F>(function: &F, x: f64, y: f64, step: f64) -> C
    where
        F: Fn(f64, f64) -> C,
    {
        (function(x + step, y) - function(x - step, y)) / (2.0 * step)
    }

    fn finite_difference_axis1<F>(function: &F, x: f64, y: f64, step: f64) -> C
    where
        F: Fn(f64, f64) -> C,
    {
        (function(x, y + step) - function(x, y - step)) / (2.0 * step)
    }

    fn finite_difference_axis0_axis0<F>(function: &F, x: f64, y: f64, step: f64) -> C
    where
        F: Fn(f64, f64) -> C,
    {
        (function(x + step, y) - r(2.0) * function(x, y) + function(x - step, y)) / (step * step)
    }

    fn finite_difference_axis0_axis1<F>(function: &F, x: f64, y: f64, step: f64) -> C
    where
        F: Fn(f64, f64) -> C,
    {
        (function(x + step, y + step) - function(x + step, y - step) - function(x - step, y + step)
            + function(x - step, y - step))
            / (4.0 * step * step)
    }

    fn finite_difference_axis1_axis1<F>(function: &F, x: f64, y: f64, step: f64) -> C
    where
        F: Fn(f64, f64) -> C,
    {
        (function(x, y + step) - r(2.0) * function(x, y) + function(x, y - step)) / (step * step)
    }

    #[test]
    fn multiplication_applies_bivariate_product_rules() {
        let f_value = c(1.0, 2.0);
        let faxis0 = c(0.5, -0.2);
        let faxis1 = c(-0.3, 0.7);
        let faxis0_axis0 = c(0.8, 0.1);
        let faxis0_axis1 = c(-0.4, 0.6);
        let faxis1_axis1 = c(0.2, -0.5);

        let g_value = c(2.0, -1.0);
        let gaxis0 = c(-0.1, 0.4);
        let gaxis1 = c(0.9, -0.3);
        let gaxis0_axis0 = c(-0.7, 0.2);
        let gaxis0_axis1 = c(0.3, 0.8);
        let gaxis1_axis1 = c(0.6, -0.9);

        let f: RealJet = JetBivariate2::from_components(
            arr0(f_value),
            arr0(faxis0),
            arr0(faxis1),
            arr0(faxis0_axis0),
            arr0(faxis0_axis1),
            arr0(faxis1_axis1),
        );

        let g = JetBivariate2::from_components(
            arr0(g_value),
            arr0(gaxis0),
            arr0(gaxis1),
            arr0(gaxis0_axis0),
            arr0(gaxis0_axis1),
            arr0(gaxis1_axis1),
        );

        let result = f.multiply(&g);

        assert_close(result.value()[()], f_value * g_value);

        assert_close(result.axis0()[()], faxis0 * g_value + f_value * gaxis0);

        assert_close(result.axis1()[()], faxis1 * g_value + f_value * gaxis1);

        assert_close(
            result.axis0_axis0()[()],
            faxis0_axis0 * g_value + r(2.0) * faxis0 * gaxis0 + f_value * gaxis0_axis0,
        );

        assert_close(
            result.axis0_axis1()[()],
            faxis0_axis1 * g_value + faxis0 * gaxis1 + faxis1 * gaxis0 + f_value * gaxis0_axis1,
        );

        assert_close(
            result.axis1_axis1()[()],
            faxis1_axis1 * g_value + r(2.0) * faxis1 * gaxis1 + f_value * gaxis1_axis1,
        );
    }

    #[test]
    fn unaraxis1_composition_applies_bivariate_chain_rule() {
        let value = c(0.7, -0.4);
        let x = c(1.3, 0.2);
        let y = c(-0.6, 0.8);
        let axis0_axis0 = c(0.1, -0.5);
        let axis0_axis1 = c(0.3, 0.7);
        let axis1_axis1 = c(-0.2, 0.4);

        let jet: HolomorphicJet = JetBivariate2::from_components(
            arr0(value),
            arr0(x),
            arr0(y),
            arr0(axis0_axis0),
            arr0(axis0_axis1),
            arr0(axis1_axis1),
        );

        let result = jet.sin();

        let first = value.cos();
        let second = -value.sin();

        assert_close(result.value()[()], value.sin());
        assert_close(result.axis0()[()], first * x);
        assert_close(result.axis1()[()], first * y);

        assert_close(
            result.axis0_axis0()[()],
            second * x * x + first * axis0_axis0,
        );

        assert_close(
            result.axis0_axis1()[()],
            second * x * y + first * axis0_axis1,
        );

        assert_close(
            result.axis1_axis1()[()],
            second * y * y + first * axis1_axis1,
        );
    }

    #[test]
    fn composite_expression_matches_finite_differences() {
        let function = |x: f64, y: f64| {
            let z = c(x, 0.2 * y);

            z.exp() * (z + c(0.3 * x * y, 0.0)).sin()
        };

        let x0 = 0.7;
        let y0 = -0.4;

        let x: RealJet = JetBivariate2::variable_axis0(arr0(r(x0)));

        let y: RealJet = JetBivariate2::variable_axis1(arr0(r(y0)));

        let imaginary_scale = c(0.0, 0.2);

        let z = x.add(&y.scale_by(imaginary_scale));

        let axis0_axis1 = x.multiply(&y).scale_by(r(0.3));

        let result = z.clone().exp().multiply(&z.add(&axis0_axis1).sin());

        let h = 1.0e-4;

        let faxis0 = (function(x0 + h, y0) - function(x0 - h, y0)) / (2.0 * h);

        let faxis1 = (function(x0, y0 + h) - function(x0, y0 - h)) / (2.0 * h);

        let faxis0_axis0 =
            (function(x0 + h, y0) - r(2.0) * function(x0, y0) + function(x0 - h, y0)) / (h * h);

        let faxis1_axis1 =
            (function(x0, y0 + h) - r(2.0) * function(x0, y0) + function(x0, y0 - h)) / (h * h);

        let faxis0_axis1 =
            (function(x0 + h, y0 + h) - function(x0 + h, y0 - h) - function(x0 - h, y0 + h)
                + function(x0 - h, y0 - h))
                / (4.0 * h * h);

        assert_fd_close(result.value()[()], function(x0, y0));
        assert_fd_close(result.axis0()[()], faxis0);
        assert_fd_close(result.axis1()[()], faxis1);
        assert_fd_close(result.axis0_axis0()[()], faxis0_axis0);
        assert_fd_close(result.axis0_axis1()[()], faxis0_axis1);
        assert_fd_close(result.axis1_axis1()[()], faxis1_axis1);
    }

    #[test]
    fn conjugation_propagates_real_coordinate_derivatives() {
        let jet: RealJet = JetBivariate2::from_components(
            a(1.0, 2.0),
            a(3.0, -4.0),
            a(-5.0, 6.0),
            a(7.0, -8.0),
            a(-9.0, 10.0),
            a(11.0, -12.0),
        );

        let result = jet.conjugated();

        assert_close(result.value()[()], c(1.0, -2.0));
        assert_close(result.axis0()[()], c(3.0, 4.0));
        assert_close(result.axis1()[()], c(-5.0, -6.0));
        assert_close(result.axis0_axis0()[()], c(7.0, 8.0));
        assert_close(result.axis0_axis1()[()], c(-9.0, -10.0));
        assert_close(result.axis1_axis1()[()], c(11.0, 12.0));
    }

    // ---------------------------------------------------------------------
    // Constructors
    // ---------------------------------------------------------------------

    #[test]
    fn variable_axis0_has_expected_seed() {
        let jet: HolomorphicJet = JetBivariate2::variable_axis0(a(2.0, 3.0));

        assert_close(jet.value()[()], c(2.0, 3.0));
        assert_close(jet.axis0()[()], r(1.0));
        assert_close(jet.axis1()[()], r(0.0));
        assert_close(jet.axis0_axis0()[()], r(0.0));
        assert_close(jet.axis0_axis1()[()], r(0.0));
        assert_close(jet.axis1_axis1()[()], r(0.0));
    }

    #[test]
    fn variable_axis1_has_expected_seed() {
        let jet: HolomorphicJet = JetBivariate2::variable_axis1(a(2.0, 3.0));

        assert_close(jet.value()[()], c(2.0, 3.0));
        assert_close(jet.axis0()[()], r(0.0));
        assert_close(jet.axis1()[()], r(1.0));
        assert_close(jet.axis0_axis0()[()], r(0.0));
        assert_close(jet.axis0_axis1()[()], r(0.0));
        assert_close(jet.axis1_axis1()[()], r(0.0));
    }

    #[test]
    fn constant_has_zero_gradient_and_hessian() {
        let jet: RealJet = JetBivariate2::constant(a(4.0, -2.0));

        assert_close(jet.value()[()], c(4.0, -2.0));
        assert_close(jet.axis0()[()], r(0.0));
        assert_close(jet.axis1()[()], r(0.0));
        assert_close(jet.axis0_axis0()[()], r(0.0));
        assert_close(jet.axis0_axis1()[()], r(0.0));
        assert_close(jet.axis1_axis1()[()], r(0.0));
    }

    #[test]
    fn from_axis0_derivatives_sets_only_axis0_components() {
        let jet: RealJet =
            JetBivariate2::from_axis0_derivatives(a(2.0, 1.0), a(3.0, -1.0), a(5.0, 2.0));

        assert_close(jet.value()[()], c(2.0, 1.0));
        assert_close(jet.axis0()[()], c(3.0, -1.0));
        assert_close(jet.axis1()[()], r(0.0));
        assert_close(jet.axis0_axis0()[()], c(5.0, 2.0));
        assert_close(jet.axis0_axis1()[()], r(0.0));
        assert_close(jet.axis1_axis1()[()], r(0.0));
    }

    #[test]
    fn from_axis1_derivatives_sets_only_axis1_components() {
        let jet: RealJet =
            JetBivariate2::from_axis1_derivatives(a(2.0, 1.0), a(3.0, -1.0), a(5.0, 2.0));

        assert_close(jet.value()[()], c(2.0, 1.0));
        assert_close(jet.axis0()[()], r(0.0));
        assert_close(jet.axis1()[()], c(3.0, -1.0));
        assert_close(jet.axis0_axis0()[()], r(0.0));
        assert_close(jet.axis0_axis1()[()], r(0.0));
        assert_close(jet.axis1_axis1()[()], c(5.0, 2.0));
    }

    // ---------------------------------------------------------------------
    // Additive operations and scaling
    // ---------------------------------------------------------------------

    fn sample_jet() -> RealJet {
        JetBivariate2::from_components(
            a(1.0, 2.0),
            a(3.0, 4.0),
            a(5.0, 6.0),
            a(7.0, 8.0),
            a(9.0, 10.0),
            a(11.0, 12.0),
        )
    }

    #[test]
    fn addition_is_componentwise() {
        let left = sample_jet();

        let right = JetBivariate2::from_components(
            a(13.0, 14.0),
            a(15.0, 16.0),
            a(17.0, 18.0),
            a(19.0, 20.0),
            a(21.0, 22.0),
            a(23.0, 24.0),
        );

        let result = left.add(&right);

        assert_close(result.value()[()], c(14.0, 16.0));
        assert_close(result.axis0()[()], c(18.0, 20.0));
        assert_close(result.axis1()[()], c(22.0, 24.0));
        assert_close(result.axis0_axis0()[()], c(26.0, 28.0));
        assert_close(result.axis0_axis1()[()], c(30.0, 32.0));
        assert_close(result.axis1_axis1()[()], c(34.0, 36.0));
    }

    #[test]
    fn subtraction_is_componentwise() {
        let left = sample_jet();
        let result = left.subtract(&left);

        assert_close(result.value()[()], r(0.0));
        assert_close(result.axis0()[()], r(0.0));
        assert_close(result.axis1()[()], r(0.0));
        assert_close(result.axis0_axis0()[()], r(0.0));
        assert_close(result.axis0_axis1()[()], r(0.0));
        assert_close(result.axis1_axis1()[()], r(0.0));
    }

    #[test]
    fn negation_negates_everaxis1_component() {
        let original = sample_jet();
        let result = original.negate();

        assert_close(result.value()[()], c(-1.0, -2.0));
        assert_close(result.axis0()[()], c(-3.0, -4.0));
        assert_close(result.axis1()[()], c(-5.0, -6.0));
        assert_close(result.axis0_axis0()[()], c(-7.0, -8.0));
        assert_close(result.axis0_axis1()[()], c(-9.0, -10.0));
        assert_close(result.axis1_axis1()[()], c(-11.0, -12.0));
    }

    #[test]
    fn scaling_scales_everaxis1_component() {
        let original = sample_jet();
        let scale = c(2.0, -1.0);

        let result = original.scale_by(scale);

        assert_close(result.value()[()], c(1.0, 2.0) * scale);

        assert_close(result.axis0()[()], c(3.0, 4.0) * scale);

        assert_close(result.axis1()[()], c(5.0, 6.0) * scale);

        assert_close(result.axis0_axis0()[()], c(7.0, 8.0) * scale);

        assert_close(result.axis0_axis1()[()], c(9.0, 10.0) * scale);

        assert_close(result.axis1_axis1()[()], c(11.0, 12.0) * scale);
    }

    // ---------------------------------------------------------------------
    // Product rules
    // ---------------------------------------------------------------------

    #[test]
    fn multiplication_applies_all_bivariate_product_rules() {
        let f_value = c(1.0, 2.0);
        let faxis0 = c(0.5, -0.2);
        let faxis1 = c(-0.3, 0.7);
        let faxis0_axis0 = c(0.8, 0.1);
        let faxis0_axis1 = c(-0.4, 0.6);
        let faxis1_axis1 = c(0.2, -0.5);

        let g_value = c(2.0, -1.0);
        let gaxis0 = c(-0.1, 0.4);
        let gaxis1 = c(0.9, -0.3);
        let gaxis0_axis0 = c(-0.7, 0.2);
        let gaxis0_axis1 = c(0.3, 0.8);
        let gaxis1_axis1 = c(0.6, -0.9);

        let f: RealJet = JetBivariate2::from_components(
            arr0(f_value),
            arr0(faxis0),
            arr0(faxis1),
            arr0(faxis0_axis0),
            arr0(faxis0_axis1),
            arr0(faxis1_axis1),
        );

        let g = JetBivariate2::from_components(
            arr0(g_value),
            arr0(gaxis0),
            arr0(gaxis1),
            arr0(gaxis0_axis0),
            arr0(gaxis0_axis1),
            arr0(gaxis1_axis1),
        );

        let result = f.multiply(&g);

        assert_close(result.value()[()], f_value * g_value);

        assert_close(result.axis0()[()], faxis0 * g_value + f_value * gaxis0);

        assert_close(result.axis1()[()], faxis1 * g_value + f_value * gaxis1);

        assert_close(
            result.axis0_axis0()[()],
            faxis0_axis0 * g_value + r(2.0) * faxis0 * gaxis0 + f_value * gaxis0_axis0,
        );

        assert_close(
            result.axis0_axis1()[()],
            faxis0_axis1 * g_value + faxis0 * gaxis1 + faxis1 * gaxis0 + f_value * gaxis0_axis1,
        );

        assert_close(
            result.axis1_axis1()[()],
            faxis1_axis1 * g_value + r(2.0) * faxis1 * gaxis1 + f_value * gaxis1_axis1,
        );
    }

    #[test]
    fn multiplying_by_constant_preserves_scaled_derivatives() {
        let jet = sample_jet();

        let constant: RealJet = JetBivariate2::constant(a(2.0, -1.0));

        let result = jet.multiply(&constant);
        let scale = c(2.0, -1.0);

        assert_close(result.value()[()], c(1.0, 2.0) * scale);

        assert_close(result.axis0()[()], c(3.0, 4.0) * scale);

        assert_close(result.axis1()[()], c(5.0, 6.0) * scale);

        assert_close(result.axis0_axis0()[()], c(7.0, 8.0) * scale);

        assert_close(result.axis0_axis1()[()], c(9.0, 10.0) * scale);

        assert_close(result.axis1_axis1()[()], c(11.0, 12.0) * scale);
    }

    #[test]
    fn product_of_coordinate_variables_has_unit_mixed_derivative() {
        let x: RealJet = JetBivariate2::variable_axis0(a(2.0, 0.0));

        let y: RealJet = JetBivariate2::variable_axis1(a(3.0, 0.0));

        let result = x.multiply(&y);

        assert_close(result.value()[()], r(6.0));
        assert_close(result.axis0()[()], r(3.0));
        assert_close(result.axis1()[()], r(2.0));
        assert_close(result.axis0_axis0()[()], r(0.0));
        assert_close(result.axis0_axis1()[()], r(1.0));
        assert_close(result.axis1_axis1()[()], r(0.0));
    }

    #[test]
    fn square_of_x_has_expected_pure_second_derivative() {
        let x: RealJet = JetBivariate2::variable_axis0(a(3.0, 0.0));

        let result = x.multiply(&x);

        assert_close(result.value()[()], r(9.0));
        assert_close(result.axis0()[()], r(6.0));
        assert_close(result.axis1()[()], r(0.0));
        assert_close(result.axis0_axis0()[()], r(2.0));
        assert_close(result.axis0_axis1()[()], r(0.0));
        assert_close(result.axis1_axis1()[()], r(0.0));
    }

    // ---------------------------------------------------------------------
    // Reciprocal and division
    // ---------------------------------------------------------------------

    #[test]
    fn reciprocal_matches_bivariate_chain_rule() {
        let value = c(2.0, 0.5);
        let x = c(0.7, -0.2);
        let y = c(-0.3, 0.4);
        let axis0_axis0 = c(0.2, 0.1);
        let axis0_axis1 = c(-0.5, 0.3);
        let axis1_axis1 = c(0.6, -0.4);

        let jet: HolomorphicJet = JetBivariate2::from_components(
            arr0(value),
            arr0(x),
            arr0(y),
            arr0(axis0_axis0),
            arr0(axis0_axis1),
            arr0(axis1_axis1),
        );

        let result = jet.reciprocal();

        let inverse = r(1.0) / value;
        let first = -r(1.0) / (value * value);
        let second = r(2.0) / (value * value * value);

        assert_close(result.value()[()], inverse);
        assert_close(result.axis0()[()], first * x);
        assert_close(result.axis1()[()], first * y);

        assert_close(
            result.axis0_axis0()[()],
            second * x * x + first * axis0_axis0,
        );

        assert_close(
            result.axis0_axis1()[()],
            second * x * y + first * axis0_axis1,
        );

        assert_close(
            result.axis1_axis1()[()],
            second * y * y + first * axis1_axis1,
        );
    }

    #[test]
    fn reciprocal_of_reciprocal_recovers_original() {
        let original: HolomorphicJet = JetBivariate2::from_components(
            a(2.0, 0.5),
            a(0.7, -0.2),
            a(-0.3, 0.4),
            a(0.2, 0.1),
            a(-0.5, 0.3),
            a(0.6, -0.4),
        );

        let result = original.reciprocal().reciprocal();

        assert_close(result.value()[()], original.value()[()]);

        assert_close(result.axis0()[()], original.axis0()[()]);

        assert_close(result.axis1()[()], original.axis1()[()]);

        assert_close(result.axis0_axis0()[()], original.axis0_axis0()[()]);

        assert_close(result.axis0_axis1()[()], original.axis0_axis1()[()]);

        assert_close(result.axis1_axis1()[()], original.axis1_axis1()[()]);
    }

    #[test]
    fn division_agrees_with_multiplication_by_reciprocal() {
        let numerator: HolomorphicJet = JetBivariate2::from_components(
            a(2.0, 1.0),
            a(0.5, -0.3),
            a(-0.2, 0.7),
            a(0.1, 0.4),
            a(-0.6, 0.2),
            a(0.8, -0.5),
        );

        let denominator: HolomorphicJet = JetBivariate2::from_components(
            a(3.0, -0.5),
            a(-0.4, 0.2),
            a(0.6, 0.1),
            a(0.3, -0.7),
            a(0.2, 0.5),
            a(-0.1, 0.9),
        );

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
        let axis0_axis0 = c(0.1, -0.5);
        let axis0_axis1 = c(0.3, 0.7);
        let axis1_axis1 = c(-0.2, 0.4);

        let jet: HolomorphicJet = JetBivariate2::from_components(
            arr0(value),
            arr0(x),
            arr0(y),
            arr0(axis0_axis0),
            arr0(axis0_axis1),
            arr0(axis1_axis1),
        );

        let result = jet.exp();
        let exponential = value.exp();

        assert_close(result.value()[()], exponential);
        assert_close(result.axis0()[()], exponential * x);
        assert_close(result.axis1()[()], exponential * y);

        assert_close(
            result.axis0_axis0()[()],
            exponential * (x * x + axis0_axis0),
        );

        assert_close(
            result.axis0_axis1()[()],
            exponential * (x * y + axis0_axis1),
        );

        assert_close(
            result.axis1_axis1()[()],
            exponential * (y * y + axis1_axis1),
        );
    }

    #[test]
    fn sine_applies_bivariate_chain_rule() {
        let value = c(0.7, -0.4);
        let x = c(1.3, 0.2);
        let y = c(-0.6, 0.8);
        let axis0_axis0 = c(0.1, -0.5);
        let axis0_axis1 = c(0.3, 0.7);
        let axis1_axis1 = c(-0.2, 0.4);

        let jet: HolomorphicJet = JetBivariate2::from_components(
            arr0(value),
            arr0(x),
            arr0(y),
            arr0(axis0_axis0),
            arr0(axis0_axis1),
            arr0(axis1_axis1),
        );

        let result = jet.sin();

        let first = value.cos();
        let second = -value.sin();

        assert_close(result.value()[()], value.sin());
        assert_close(result.axis0()[()], first * x);
        assert_close(result.axis1()[()], first * y);

        assert_close(
            result.axis0_axis0()[()],
            second * x * x + first * axis0_axis0,
        );

        assert_close(
            result.axis0_axis1()[()],
            second * x * y + first * axis0_axis1,
        );

        assert_close(
            result.axis1_axis1()[()],
            second * y * y + first * axis1_axis1,
        );
    }

    #[test]
    fn cosine_applies_bivariate_chain_rule() {
        let value = c(0.7, -0.4);
        let x = c(1.3, 0.2);
        let y = c(-0.6, 0.8);
        let axis0_axis0 = c(0.1, -0.5);
        let axis0_axis1 = c(0.3, 0.7);
        let axis1_axis1 = c(-0.2, 0.4);

        let jet: HolomorphicJet = JetBivariate2::from_components(
            arr0(value),
            arr0(x),
            arr0(y),
            arr0(axis0_axis0),
            arr0(axis0_axis1),
            arr0(axis1_axis1),
        );

        let result = jet.cos();

        let first = -value.sin();
        let second = -value.cos();

        assert_close(result.value()[()], value.cos());
        assert_close(result.axis0()[()], first * x);
        assert_close(result.axis1()[()], first * y);

        assert_close(
            result.axis0_axis0()[()],
            second * x * x + first * axis0_axis0,
        );

        assert_close(
            result.axis0_axis1()[()],
            second * x * y + first * axis0_axis1,
        );

        assert_close(
            result.axis1_axis1()[()],
            second * y * y + first * axis1_axis1,
        );
    }

    #[test]
    fn sqrt_applies_bivariate_chain_rule() {
        let value = c(2.0, 0.5);
        let x = c(0.7, -0.2);
        let y = c(-0.3, 0.4);
        let axis0_axis0 = c(0.2, 0.1);
        let axis0_axis1 = c(-0.5, 0.3);
        let axis1_axis1 = c(0.6, -0.4);

        let jet: HolomorphicJet = JetBivariate2::from_components(
            arr0(value),
            arr0(x),
            arr0(y),
            arr0(axis0_axis0),
            arr0(axis0_axis1),
            arr0(axis1_axis1),
        );

        let result = jet.sqrt();

        let root = value.sqrt();
        let first = r(1.0) / (r(2.0) * root);
        let second = -r(1.0) / (r(4.0) * root * value);

        assert_close(result.value()[()], root);
        assert_close(result.axis0()[()], first * x);
        assert_close(result.axis1()[()], first * y);

        assert_close(
            result.axis0_axis0()[()],
            second * x * x + first * axis0_axis0,
        );

        assert_close(
            result.axis0_axis1()[()],
            second * x * y + first * axis0_axis1,
        );

        assert_close(
            result.axis1_axis1()[()],
            second * y * y + first * axis1_axis1,
        );
    }

    // ---------------------------------------------------------------------
    // Sampled-function composition
    // ---------------------------------------------------------------------

    #[test]
    fn sampled_function_composition_matches_unaraxis1_composition() {
        let value = c(0.7, -0.4);

        let argument: HolomorphicJet = JetBivariate2::from_components(
            arr0(value),
            a(1.3, 0.2),
            a(-0.6, 0.8),
            a(0.1, -0.5),
            a(0.3, 0.7),
            a(-0.2, 0.4),
        );

        let expansion =
            SecondOrderExpansion::new(arr0(value.sin()), arr0(value.cos()), arr0(-value.sin()));

        let sampled = HolomorphicJet::compose_sampled_function(&argument, expansion);

        let direct = argument.sin();

        assert_close(sampled.value()[()], direct.value()[()]);
        assert_close(sampled.axis0()[()], direct.axis0()[()]);
        assert_close(sampled.axis1()[()], direct.axis1()[()]);
        assert_close(sampled.axis0_axis0()[()], direct.axis0_axis0()[()]);
        assert_close(sampled.axis0_axis1()[()], direct.axis0_axis1()[()]);
        assert_close(sampled.axis1_axis1()[()], direct.axis1_axis1()[()]);
    }

    // ---------------------------------------------------------------------
    // Coordinate symmetry
    // ---------------------------------------------------------------------

    fn swap_coordinates(jet: &RealJet) -> RealJet {
        JetBivariate2::from_components(
            jet.value().clone(),
            jet.axis1().clone(),
            jet.axis0().clone(),
            jet.axis1_axis1().clone(),
            jet.axis0_axis1().clone(),
            jet.axis0_axis0().clone(),
        )
    }

    #[test]
    fn multiplication_is_equivariant_under_coordinate_exchange() {
        let left = sample_jet();

        let right = JetBivariate2::from_components(
            a(2.0, -1.0),
            a(0.5, 0.3),
            a(-0.7, 0.1),
            a(0.2, 0.8),
            a(-0.4, 0.6),
            a(0.9, -0.2),
        );

        let original = swap_coordinates(&left.multiply(&right));

        let swapped = swap_coordinates(&left).multiply(&swap_coordinates(&right));

        assert_eq!(original, swapped);
    }

    #[test]
    fn unaraxis1_composition_is_equivariant_under_coordinate_exchange() {
        let original = swap_coordinates(&sample_jet().sin());

        let swapped = swap_coordinates(&sample_jet()).sin();

        assert_eq!(original, swapped);
    }

    // ---------------------------------------------------------------------
    // Array behaviour
    // ---------------------------------------------------------------------

    #[test]
    fn array_operations_preserve_shape() {
        type ArrayJet = JetBivariate2<Array1<C>, RealParameter>;

        let value = array![c(1.0, 0.2), c(2.0, -0.4), c(3.0, 0.6),];

        let jet: ArrayJet = JetBivariate2::variable_axis0(value.clone());

        let result = jet.exp();

        assert_eq!(result.value().raw_dim(), value.raw_dim(),);

        assert_eq!(result.axis0().raw_dim(), value.raw_dim(),);

        assert_eq!(result.axis1().raw_dim(), value.raw_dim(),);

        assert_eq!(result.axis0_axis0().raw_dim(), value.raw_dim(),);

        assert_eq!(result.axis0_axis1().raw_dim(), value.raw_dim(),);

        assert_eq!(result.axis1_axis1().raw_dim(), value.raw_dim(),);
    }

    #[test]
    fn array_unary_operations_are_elementwise() {
        type ArrayJet = JetBivariate2<Array1<C>, HolomorphicParameter>;

        let values = array![c(1.0, 0.2), c(2.0, -0.4), c(3.0, 0.6),];

        let x = array![c(0.1, 0.2), c(-0.3, 0.4), c(0.5, -0.6),];

        let y = array![c(-0.2, 0.1), c(0.4, -0.3), c(-0.6, 0.5),];

        let zero = Array1::zeros(values.raw_dim());

        let jet: ArrayJet = JetBivariate2::from_components(
            values.clone(),
            x.clone(),
            y.clone(),
            zero.clone(),
            zero.clone(),
            zero,
        );

        let result = jet.exp();

        for index in 0..values.len() {
            let exponential = values[index].exp();

            assert_close(result.value()[index], exponential);

            assert_close(result.axis0()[index], exponential * x[index]);

            assert_close(result.axis1()[index], exponential * y[index]);

            assert_close(
                result.axis0_axis1()[index],
                exponential * x[index] * y[index],
            );
        }
    }

    // ---------------------------------------------------------------------
    // Real-coordinate-only operations
    // ---------------------------------------------------------------------

    #[test]
    fn conjugation_conjugates_everaxis1_component() {
        let result = sample_jet().conjugated();

        assert_close(result.value()[()], c(1.0, -2.0));
        assert_close(result.axis0()[()], c(3.0, -4.0));
        assert_close(result.axis1()[()], c(5.0, -6.0));
        assert_close(result.axis0_axis0()[()], c(7.0, -8.0));
        assert_close(result.axis0_axis1()[()], c(9.0, -10.0));
        assert_close(result.axis1_axis1()[()], c(11.0, -12.0));
    }

    #[test]
    fn real_extracts_everaxis1_component() {
        let result = sample_jet().real();

        assert_relative_eq!(result.value()[()], 1.0, epsilon = EPSILON,);

        assert_relative_eq!(result.axis0()[()], 3.0, epsilon = EPSILON,);

        assert_relative_eq!(result.axis1()[()], 5.0, epsilon = EPSILON,);

        assert_relative_eq!(result.axis0_axis0()[()], 7.0, epsilon = EPSILON,);

        assert_relative_eq!(result.axis0_axis1()[()], 9.0, epsilon = EPSILON,);

        assert_relative_eq!(result.axis1_axis1()[()], 11.0, epsilon = EPSILON,);
    }

    // ---------------------------------------------------------------------
    // Composite finite-difference validation
    // ---------------------------------------------------------------------

    #[test]
    fn composite_expression_matches_all_finite_differences() {
        let x0 = 0.7;
        let y0 = -0.4;

        let x: RealJet = JetBivariate2::variable_axis0(arr0(r(x0)));

        let y: RealJet = JetBivariate2::variable_axis1(arr0(r(y0)));

        let z = x.add(&y.scale_by(c(0.0, 0.2)));

        let coupling = x.multiply(&y).scale_by(r(0.3));

        let numerator = z.clone().exp().multiply(&z.add(&coupling).sin());

        let denominator = x
            .multiply(&x)
            .add(&y.multiply(&y))
            .add(&JetBivariate2::constant(arr0(r(2.0))))
            .sqrt();

        let result = numerator.divide(&denominator);

        let function = |x: f64, y: f64| {
            let z = c(x, 0.2 * y);
            let coupling = r(0.3 * x * y);

            z.exp() * (z + coupling).sin() / r(x * x + y * y + 2.0).sqrt()
        };

        let first_step = 1.0e-5;
        let second_step = 2.0e-4;

        assert_fd_close(result.value()[()], function(x0, y0));

        assert_fd_close(
            result.axis0()[()],
            finite_difference_axis0(&function, x0, y0, first_step),
        );

        assert_fd_close(
            result.axis1()[()],
            finite_difference_axis1(&function, x0, y0, first_step),
        );

        assert_fd_close(
            result.axis0_axis0()[()],
            finite_difference_axis0_axis0(&function, x0, y0, second_step),
        );

        assert_fd_close(
            result.axis0_axis1()[()],
            finite_difference_axis0_axis1(&function, x0, y0, second_step),
        );

        assert_fd_close(
            result.axis1_axis1()[()],
            finite_difference_axis1_axis1(&function, x0, y0, second_step),
        );
    }
}
