//! First- and second-order differential jets.
//!
//! This module provides generic containers for propagating analytic first and
//! second derivatives through algebraic expressions.
//!
//! [`JetFirst`] stores a value and its first derivative:
//!
//! ```text
//! (f, f′)
//! ```
//!
//! [`Jet`] stores a value and its first and second derivatives:
//!
//! ```text
//! (f, f′, f″)
//! ```
//!
//! The underlying value type determines which operations are available through
//! capability traits:
//!
//! - [`JetAdditive`] supports addition, subtraction, and negation;
//! - [`JetBilinear`] supports a bilinear product and its product rules;
//! - [`JetField`] supports elementwise reciprocals and division;
//! - [`ChainRuleScale`] supports multiplication by sampled chain-rule
//!   coefficients.
//!
//! Arrays implement all of these capabilities. Transfer matrices implement
//! additive and bilinear operations because ordinary matrix multiplication is
//! bilinear. Scattering matrices must not implement [`JetBilinear`], because
//! the Redheffer star product is rational rather than bilinear.
//!
//! Value-only calculations should use the underlying value type directly.
//! [`JetFirst`] and [`Jet`] are reserved for calculations that actually
//! request derivatives.

use ndarray::{Array, ArrayBase, Dimension, OwnedRepr};

use crate::{ComplexScalar, backend::derivative::ChainRule};

pub(crate) type ArrayJet<C, D> = Jet<ArrayBase<OwnedRepr<C>, D>>;
pub(crate) type ArrayJetFirst<C, D> = JetFirst<ArrayBase<OwnedRepr<C>, D>>;

/// A value and its first and second derivatives.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Jet<I> {
    value: I,
    first: I,
    second: I,
}

/// A value and its first derivative.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct JetFirst<I> {
    value: I,
    first: I,
}

/// Construct a zero value with the same representation and sampled shape.
pub(crate) trait JetZeroLike: Clone {
    fn jet_zeros_like(shape_source: &Self) -> Self;
}

/// Additive operations supported componentwise by jets.
pub(crate) trait JetAdditive: Sized {
    /// Add two values.
    fn jet_add(&self, rhs: &Self) -> Self;

    /// Subtract `rhs` from this value.
    fn jet_subtract(&self, rhs: &Self) -> Self;

    /// Negate this value.
    fn jet_negate(&self) -> Self;
}

/// A bilinear product for which the ordinary product rule applies.
pub(crate) trait JetBilinear: JetAdditive {
    /// Multiply or compose two values using a bilinear operation.
    fn jet_multiply(&self, rhs: &Self) -> Self;

    /// Return twice this value.
    fn jet_double(&self) -> Self;
}

/// Construct a scalar constant with the same sampled shape.
pub(crate) trait JetScaleBy: Clone {
    type Scalar: Copy;

    fn jet_scale_by(&self, value: Self::Scalar) -> Self;
}

/// Construct a scalar constant with the same sampled shape.
pub(crate) trait JetConstant: Clone {
    type Scalar: Copy;

    fn constant_like(&self, value: Self::Scalar) -> Self;
}

/// Elementwise field operations used by reciprocal and division.
///
/// This trait is appropriate for sampled scalar arrays, not matrices.
pub(crate) trait JetField: JetBilinear {
    fn elementwise_reciprocal(&self) -> Self;
}

/// Operations required of a chain-rule coefficient.
pub(crate) trait ChainRuleCoefficient: Clone {
    fn square(&self) -> Self;
}

/// Scale a differentiated value by a chain-rule coefficient.
pub(crate) trait ChainRuleScale<Rhs>: Clone {
    fn scale_by(&self, rhs: &Rhs) -> Self;
}

impl<I> JetFirst<I> {
    /// Construct a first-order jet.
    pub(crate) fn from_parts(value: I, first: I) -> Self {
        Self { value, first }
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

    /// Transform the derivative to a new independent coordinate.
    pub(crate) fn chain_rule<R>(self, rule: &ChainRule<R>) -> Self
    where
        I: ChainRuleScale<R>,
    {
        Self {
            value: self.value,
            first: self.first.scale_by(&rule.first),
        }
    }
}

impl<I> JetFirst<I>
where
    I: JetAdditive,
{
    pub(crate) fn add(&self, rhs: &Self) -> Self {
        Self {
            value: self.value.jet_add(&rhs.value),
            first: self.first.jet_add(&rhs.first),
        }
    }

    pub(crate) fn subtract(&self, rhs: &Self) -> Self {
        Self {
            value: self.value.jet_subtract(&rhs.value),
            first: self.first.jet_subtract(&rhs.first),
        }
    }

    pub(crate) fn negate(&self) -> Self {
        Self {
            value: self.value.jet_negate(),
            first: self.first.jet_negate(),
        }
    }
}

impl<I> JetFirst<I>
where
    I: JetBilinear,
{
    /// Multiply two first-order jets using the product rule.
    pub(crate) fn multiply(&self, rhs: &Self) -> Self {
        Self {
            value: self.value.jet_multiply(&rhs.value),
            first: self
                .first
                .jet_multiply(&rhs.value)
                .jet_add(&self.value.jet_multiply(&rhs.first)),
        }
    }
}

impl<I> JetFirst<I>
where
    I: JetConstant + JetZeroLike,
{
    /// Construct a constant first-order jet with zero derivative.
    pub(crate) fn constant_like(source: &I, value: I::Scalar) -> Self {
        Self {
            value: source.constant_like(value),
            first: I::jet_zeros_like(source),
        }
    }
}

impl<I> JetFirst<I>
where
    I: JetZeroLike,
{
    /// Construct a first-order jet whose derivative is zero.
    pub(crate) fn constant(value: I) -> Self {
        let first = I::jet_zeros_like(&value);

        Self { value, first }
    }
}

impl<I> JetFirst<I>
where
    I: JetScaleBy,
{
    /// Multiply two first-order jets using the product rule.
    pub(crate) fn scale_by(&self, value: I::Scalar) -> Self {
        Self {
            value: self.value.jet_scale_by(value),
            first: self.first.jet_scale_by(value),
        }
    }
}

impl<I> JetFirst<I>
where
    I: JetField,
{
    /// Compute the elementwise reciprocal and its first derivative.
    pub(crate) fn reciprocal(&self) -> Self {
        let inverse = self.value.elementwise_reciprocal();
        let inverse_squared = inverse.jet_multiply(&inverse);

        Self {
            value: inverse,
            first: self.first.jet_negate().jet_multiply(&inverse_squared),
        }
    }

    /// Divide two first-order jets elementwise.
    pub(crate) fn divide(&self, rhs: &Self) -> Self {
        self.multiply(&rhs.reciprocal())
    }
}

impl<I> Jet<I> {
    /// Construct a second-order jet.
    pub(crate) fn from_parts(value: I, first: I, second: I) -> Self {
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

    /// Transform derivatives to a new independent coordinate.
    ///
    /// If `x` is the primitive coordinate and `y` is the requested coordinate:
    ///
    /// ```text
    /// df/dy   = df/dx · dx/dy
    /// d²f/dy² = d²f/dx² · (dx/dy)² + df/dx · d²x/dy²
    /// ```
    pub(crate) fn chain_rule<R>(self, rule: &ChainRule<R>) -> Self
    where
        I: ChainRuleScale<R> + JetAdditive,
        R: ChainRuleCoefficient,
    {
        let first_squared = rule.first.square();

        let transformed_second = self
            .second
            .scale_by(&first_squared)
            .jet_add(&self.first.scale_by(&rule.second));

        Self {
            value: self.value,
            first: self.first.scale_by(&rule.first),
            second: transformed_second,
        }
    }
}

impl<I> Jet<I>
where
    I: JetAdditive,
{
    pub(crate) fn add(&self, rhs: &Self) -> Self {
        Self {
            value: self.value.jet_add(&rhs.value),
            first: self.first.jet_add(&rhs.first),
            second: self.second.jet_add(&rhs.second),
        }
    }

    pub(crate) fn subtract(&self, rhs: &Self) -> Self {
        Self {
            value: self.value.jet_subtract(&rhs.value),
            first: self.first.jet_subtract(&rhs.first),
            second: self.second.jet_subtract(&rhs.second),
        }
    }

    pub(crate) fn negate(&self) -> Self {
        Self {
            value: self.value.jet_negate(),
            first: self.first.jet_negate(),
            second: self.second.jet_negate(),
        }
    }
}

impl<I> Jet<I>
where
    I: JetBilinear,
{
    /// Multiply two second-order jets using the product rules.
    pub(crate) fn multiply(&self, rhs: &Self) -> Self {
        let cross = self.first.jet_multiply(&rhs.first).jet_double();

        Self {
            value: self.value.jet_multiply(&rhs.value),

            first: self
                .first
                .jet_multiply(&rhs.value)
                .jet_add(&self.value.jet_multiply(&rhs.first)),

            second: self
                .second
                .jet_multiply(&rhs.value)
                .jet_add(&cross)
                .jet_add(&self.value.jet_multiply(&rhs.second)),
        }
    }
}

impl<I> Jet<I>
where
    I: JetConstant + JetZeroLike,
{
    /// Construct a constant second-order jet with zero derivatives.
    pub(crate) fn constant_like(source: &I, value: I::Scalar) -> Self {
        let zero = I::jet_zeros_like(source);

        Self {
            value: source.constant_like(value),
            first: zero.clone(),
            second: zero,
        }
    }
}

impl<I> Jet<I>
where
    I: JetScaleBy,
{
    /// Construct a constant second-order jet with zero derivatives.
    pub(crate) fn scale_by(&self, value: I::Scalar) -> Self {
        Self {
            value: self.value.jet_scale_by(value),
            first: self.first.jet_scale_by(value),
            second: self.second.jet_scale_by(value),
        }
    }
}

impl<I> Jet<I>
where
    I: JetZeroLike,
{
    /// Construct a second-order jet whose first and second derivatives are zero.
    pub(crate) fn constant(value: I) -> Self {
        let zero = I::jet_zeros_like(&value);

        Self {
            value,
            first: zero.clone(),
            second: zero,
        }
    }
}

impl<I> Jet<I>
where
    I: JetField,
{
    /// Compute the elementwise reciprocal and its first two derivatives.
    pub(crate) fn reciprocal(&self) -> Self {
        let inverse = self.value.elementwise_reciprocal();
        let inverse_squared = inverse.jet_multiply(&inverse);
        let inverse_cubed = inverse_squared.jet_multiply(&inverse);

        let first = self.first.jet_negate().jet_multiply(&inverse_squared);

        let second = self
            .first
            .jet_multiply(&self.first)
            .jet_double()
            .jet_multiply(&inverse_cubed)
            .jet_subtract(&self.second.jet_multiply(&inverse_squared));

        Self {
            value: inverse,
            first,
            second,
        }
    }

    /// Divide two second-order jets elementwise.
    pub(crate) fn divide(&self, rhs: &Self) -> Self {
        self.multiply(&rhs.reciprocal())
    }
}

impl<C, D> JetZeroLike for ArrayBase<OwnedRepr<C>, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn jet_zeros_like(shape_source: &Self) -> Self {
        Array::zeros(shape_source.raw_dim())
    }
}

impl<C, D> JetAdditive for ArrayBase<OwnedRepr<C>, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn jet_add(&self, rhs: &Self) -> Self {
        self.clone() + rhs.view()
    }

    fn jet_subtract(&self, rhs: &Self) -> Self {
        self.clone() - rhs.view()
    }

    fn jet_negate(&self) -> Self {
        -self.clone()
    }
}

impl<C, D> JetBilinear for ArrayBase<OwnedRepr<C>, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn jet_multiply(&self, rhs: &Self) -> Self {
        self.clone() * rhs.view()
    }

    fn jet_double(&self) -> Self {
        self.mapv(|x| x + x)
    }
}

impl<C, D> JetScaleBy for ArrayBase<OwnedRepr<C>, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    type Scalar = C;

    fn jet_scale_by(&self, value: Self::Scalar) -> Self {
        self.mapv(|x| x * value)
    }
}

impl<C, D> JetConstant for ArrayBase<OwnedRepr<C>, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    type Scalar = C;

    fn constant_like(&self, value: C) -> Self {
        self.mapv(|_| value)
    }
}

impl<C, D> JetField for ArrayBase<OwnedRepr<C>, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn elementwise_reciprocal(&self) -> Self {
        self.mapv(|x| C::one() / x)
    }
}

impl<C, D> ChainRuleScale<ArrayBase<OwnedRepr<C>, D>> for ArrayBase<OwnedRepr<C>, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn scale_by(&self, rhs: &ArrayBase<OwnedRepr<C>, D>) -> Self {
        self * rhs
    }
}

impl<C, D> ChainRuleCoefficient for ArrayBase<OwnedRepr<C>, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn square(&self) -> Self {
        self.mapv(|x| x * x)
    }
}

impl<C, D> ArrayJet<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn exp(self) -> Self {
        let exp = self.value().mapv(|value| value.exp());

        let first = exp.clone() * self.first().view();

        let second =
            exp.clone() * (self.first().mapv(|value| value * value) + self.second().view());

        Self::from_parts(exp, first, second)
    }
}

impl<C, D> ArrayJetFirst<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn exp(self) -> Self {
        let exp = self.value().mapv(|value| value.exp());

        let first = exp.clone() * self.first().view();

        Self::from_parts(exp, first)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array0, arr0};
    use num_complex::Complex64;

    use super::*;

    type C = Complex64;
    type A = Array0<C>;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn a(value: f64) -> A {
        arr0(c(value))
    }

    fn assert_close(actual: C, expected: C) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }

    #[test]
    fn first_jet_adds_componentwise() {
        let left = ArrayJetFirst::from_parts(a(2.0), a(3.0));
        let right = ArrayJetFirst::from_parts(a(5.0), a(7.0));

        let result = left.add(&right);

        assert_close(result.value()[()], c(7.0));
        assert_close(result.first()[()], c(10.0));
    }

    #[test]
    fn second_jet_adds_componentwise() {
        let left = ArrayJet::from_parts(a(2.0), a(3.0), a(4.0));
        let right = ArrayJet::from_parts(a(5.0), a(7.0), a(11.0));

        let result = left.add(&right);

        assert_close(result.value()[()], c(7.0));
        assert_close(result.first()[()], c(10.0));
        assert_close(result.second()[()], c(15.0));
    }

    #[test]
    fn first_jet_product_rule_matches_formula() {
        let f = ArrayJetFirst::from_parts(a(2.0), a(3.0));
        let g = ArrayJetFirst::from_parts(a(5.0), a(7.0));

        let result = f.multiply(&g);

        assert_close(result.value()[()], c(10.0));
        assert_close(result.first()[()], c(3.0 * 5.0 + 2.0 * 7.0));
    }

    #[test]
    fn second_jet_product_rule_matches_formula() {
        let f = ArrayJet::from_parts(a(2.0), a(3.0), a(4.0));
        let g = ArrayJet::from_parts(a(5.0), a(7.0), a(11.0));

        let result = f.multiply(&g);

        assert_close(result.value()[()], c(10.0));
        assert_close(result.first()[()], c(3.0 * 5.0 + 2.0 * 7.0));
        assert_close(
            result.second()[()],
            c(4.0 * 5.0 + 2.0 * 3.0 * 7.0 + 2.0 * 11.0),
        );
    }

    #[test]
    fn first_jet_reciprocal_matches_formula() {
        let f = ArrayJetFirst::from_parts(a(2.0), a(3.0));

        let result = f.reciprocal();

        assert_close(result.value()[()], c(0.5));
        assert_close(result.first()[()], c(-3.0 / 4.0));
    }

    #[test]
    fn second_jet_reciprocal_matches_formula() {
        let f = ArrayJet::from_parts(a(2.0), a(3.0), a(5.0));

        let result = f.reciprocal();

        assert_close(result.value()[()], c(0.5));
        assert_close(result.first()[()], c(-3.0 / 4.0));
        assert_close(result.second()[()], c(2.0 * 3.0 * 3.0 / 8.0 - 5.0 / 4.0));
    }

    #[test]
    fn first_jet_division_matches_quotient_rule() {
        let f = ArrayJetFirst::from_parts(a(3.0), a(5.0));
        let g = ArrayJetFirst::from_parts(a(2.0), a(7.0));

        let result = f.divide(&g);

        assert_close(result.value()[()], c(1.5));
        assert_close(result.first()[()], c((5.0 * 2.0 - 3.0 * 7.0) / 4.0));
    }

    #[test]
    fn second_order_chain_rule_matches_formula() {
        // f_x = 3, f_xx = 5
        let jet = ArrayJet::from_parts(a(2.0), a(3.0), a(5.0));

        // x_y = 7, x_yy = 11
        let rule = ChainRule {
            first: a(7.0),
            second: a(11.0),
        };

        let result = jet.chain_rule(&rule);

        assert_close(result.value()[()], c(2.0));
        assert_close(result.first()[()], c(3.0 * 7.0));
        assert_close(result.second()[()], c(5.0 * 7.0 * 7.0 + 3.0 * 11.0));
    }

    #[test]
    fn first_order_chain_rule_uses_only_first_coefficient() {
        let jet = ArrayJetFirst::from_parts(a(2.0), a(3.0));

        let rule = ChainRule {
            first: a(7.0),
            second: a(11.0),
        };

        let result = jet.chain_rule(&rule);

        assert_close(result.value()[()], c(2.0));
        assert_close(result.first()[()], c(21.0));
    }

    #[test]
    fn constant_jets_have_zero_derivatives() {
        let source = a(9.0);

        let first = ArrayJetFirst::constant_like(&source, c(4.0));
        let second = ArrayJet::constant_like(&source, c(4.0));

        assert_close(first.value()[()], c(4.0));
        assert_close(first.first()[()], c(0.0));

        assert_close(second.value()[()], c(4.0));
        assert_close(second.first()[()], c(0.0));
        assert_close(second.second()[()], c(0.0));
    }

    #[test]
    fn complex_second_order_product_rule_is_preserved() {
        let f = ArrayJet::from_parts(
            arr0(C::new(1.0, 2.0)),
            arr0(C::new(0.5, -0.25)),
            arr0(C::new(-0.3, 0.4)),
        );

        let g = ArrayJet::from_parts(
            arr0(C::new(2.0, -1.0)),
            arr0(C::new(-0.2, 0.7)),
            arr0(C::new(0.6, -0.8)),
        );

        let result = f.multiply(&g);

        let expected_value = f.value()[()] * g.value()[()];
        let expected_first = f.first()[()] * g.value()[()] + f.value()[()] * g.first()[()];
        let expected_second = f.second()[()] * g.value()[()]
            + c(2.0) * f.first()[()] * g.first()[()]
            + f.value()[()] * g.second()[()];

        assert_close(result.value()[()], expected_value);
        assert_close(result.first()[()], expected_first);
        assert_close(result.second()[()], expected_second);
    }
}
