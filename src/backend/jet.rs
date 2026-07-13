use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{ComplexScalar, backend::derivative::ChainRule};

pub(crate) type ArrayJet<C, D> = Jet<ArrayBase<OwnedRepr<C>, D>>;
pub(crate) type ArrayJetFirst<C, D> = JetFirst<ArrayBase<OwnedRepr<C>, D>>;

#[derive(Clone, Debug)]
pub(crate) struct Jet<I> {
    value: I,
    first: I,
    second: I,
}

#[derive(Clone, Debug)]
pub(crate) struct JetFirst<I> {
    value: I,
    first: I,
}

pub(crate) trait JetZeroLike: Clone {
    fn zeros_like(shape_source: &Self) -> Self;
}

pub(crate) trait JetAdditive {
    /// Add two algebra values.
    fn jet_add(&self, rhs: &Self) -> Self;

    /// Subtract two algebra values.
    fn jet_subtract(&self, rhs: &Self) -> Self;

    /// Negate an algebra value
    fn jet_negate(&self) -> Self;
}

pub(crate) trait JetBilinear: JetAdditive {
    /// Compose or multiply two algebra values.
    ///
    /// The precise operation depends on the algebra:
    ///
    /// - arrays: elementwise multiplication,
    /// - transfer matrices: ordinary matrix multiplication,
    /// - scattering matrices: Redheffer star composition.
    fn jet_multiply(&self, rhs: &Self) -> Self;

    /// Multiply an algebra value by two
    fn jet_double(&self) -> Self;
}

pub(crate) trait JetField: JetBilinear {
    /// Construct an elementwise constant with the same shape as `self`.
    fn constant_like(&self, value: Self::Scalar) -> Self;

    /// Compute the elementwise multiplicative inverse.
    fn elementwise_reciprocal(&self) -> Self;

    type Scalar: Copy;
}

pub(crate) trait ChainRuleCoefficient: Clone {
    fn square(&self) -> Self;
}

pub(crate) trait ChainRuleScale<Rhs>: Clone {
    fn scale_by(&self, rhs: &Rhs) -> Self;
}

impl<I> Jet<I>
where
    I: JetZeroLike,
{
    pub(crate) fn value_only(value: I) -> Self {
        let zero = I::zeros_like(&value);

        Self {
            value,
            first: zero.clone(),
            second: zero,
        }
    }

    pub(crate) fn with_first(value: I, first: I) -> Self {
        let second = I::zeros_like(&value);

        Self {
            value,
            first,
            second,
        }
    }
}

impl<I> Jet<I> {
    pub(crate) fn with_second(value: I, first: I, second: I) -> Self {
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
    pub(crate) fn multiply(&self, rhs: &Self) -> Self {
        let value = self.value.jet_multiply(&rhs.value);

        let first = self
            .first
            .jet_multiply(&rhs.value)
            .jet_add(&self.value.jet_multiply(&rhs.first));

        let cross = self.first.jet_multiply(&rhs.first).jet_double();

        let second = self
            .second
            .jet_multiply(&rhs.value)
            .jet_add(&cross)
            .jet_add(&self.value.jet_multiply(&rhs.second));

        Self {
            value,
            first,
            second,
        }
    }
}

impl<I> Jet<I>
where
    I: JetZeroLike + JetField,
{
    pub(crate) fn constant_like(source: &I, value: I::Scalar) -> Self {
        Self::value_only(source.constant_like(value))
    }

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

    pub(crate) fn divide(&self, rhs: &Self) -> Self {
        self.multiply(&rhs.reciprocal())
    }
}

impl<I> Jet<I> {
    pub(crate) fn chain_rule<R>(self, rule: &ChainRule<R>) -> Self
    where
        I: ChainRuleScale<R> + JetAdditive,
        R: ChainRuleCoefficient,
    {
        let primitive_first = self.first;

        let transformed_first = primitive_first.scale_by(&rule.first);

        let first_squared = rule.first.square();

        let transformed_second = self
            .second
            .scale_by(&first_squared)
            .jet_add(&primitive_first.scale_by(&rule.second));

        Self {
            value: self.value,
            first: transformed_first,
            second: transformed_second,
        }
    }
}

impl<I> JetFirst<I>
where
    I: JetZeroLike,
{
    pub(crate) fn value_only(value: I) -> Self {
        let zero = I::zeros_like(&value);

        Self {
            value,
            first: zero.clone(),
        }
    }

    pub(crate) fn with_first(value: I, first: I) -> Self {
        Self { value, first }
    }
}

impl<I> JetFirst<I> {
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
    pub(crate) fn multiply(&self, rhs: &Self) -> Self {
        let value = self.value.jet_multiply(&rhs.value);

        let first = self
            .first
            .jet_multiply(&rhs.value)
            .jet_add(&self.value.jet_multiply(&rhs.first));

        Self { value, first }
    }
}

impl<I> JetFirst<I>
where
    I: JetZeroLike + JetField,
{
    pub(crate) fn constant_like(source: &I, value: I::Scalar) -> Self {
        Self::value_only(source.constant_like(value))
    }

    pub(crate) fn reciprocal(&self) -> Self {
        let inverse = self.value.elementwise_reciprocal();
        let inverse_squared = inverse.jet_multiply(&inverse);
        let inverse_cubed = inverse_squared.jet_multiply(&inverse);

        let first = self.first.jet_negate().jet_multiply(&inverse_squared);

        Self {
            value: inverse,
            first,
        }
    }

    pub(crate) fn divide(&self, rhs: &Self) -> Self {
        self.multiply(&rhs.reciprocal())
    }
}

impl<I> JetFirst<I> {
    pub(crate) fn chain_rule<R>(self, rule: &ChainRule<R>) -> Self
    where
        I: ChainRuleScale<R> + JetAdditive,
        R: ChainRuleCoefficient,
    {
        let primitive_first = self.first;

        let transformed_first = primitive_first.scale_by(&rule.first);

        let first_squared = rule.first.square();

        Self {
            value: self.value,
            first: transformed_first,
        }
    }
}

impl<C, D> JetZeroLike for ArrayBase<OwnedRepr<C>, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn zeros_like(shape_source: &Self) -> Self {
        ArrayBase::zeros_like(shape_source)
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

impl<C, D> JetField for ArrayBase<OwnedRepr<C>, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    type Scalar = C;

    fn constant_like(&self, value: C) -> Self {
        self.mapv(|_| value)
    }

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
