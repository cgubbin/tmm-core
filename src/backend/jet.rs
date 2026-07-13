use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{ComplexScalar, backend::derivative::ChainRule};

#[derive(Clone, Debug)]
pub(crate) struct ArrayJet<C, D>
where
    D: Dimension,
{
    value: ArrayBase<OwnedRepr<C>, D>,
    first: ArrayBase<OwnedRepr<C>, D>,
    second: ArrayBase<OwnedRepr<C>, D>,
}

impl<C, D> ArrayJet<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn value_only(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        let zero = value.mapv(|_| C::zero());

        Self {
            value,
            first: zero.clone(),
            second: zero,
        }
    }

    pub(crate) fn with_first(
        value: ArrayBase<OwnedRepr<C>, D>,
        first: ArrayBase<OwnedRepr<C>, D>,
    ) -> Self {
        let second = value.mapv(|_| C::zero());

        Self {
            value,
            first,
            second,
        }
    }

    pub(crate) fn with_second(
        value: ArrayBase<OwnedRepr<C>, D>,
        first: ArrayBase<OwnedRepr<C>, D>,
        second: ArrayBase<OwnedRepr<C>, D>,
    ) -> Self {
        Self {
            value,
            first,
            second,
        }
    }

    pub(crate) fn constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        Self::value_only(source.mapv(|_| value))
    }

    pub(crate) fn add(&self, rhs: &Self) -> Self {
        Self {
            value: self.value.clone() + rhs.value.view(),
            first: self.first.clone() + rhs.first.view(),
            second: self.second.clone() + rhs.second.view(),
        }
    }

    pub(crate) fn subtract(&self, rhs: &Self) -> Self {
        Self {
            value: self.value.clone() - rhs.value.view(),
            first: self.first.clone() - rhs.first.view(),
            second: self.second.clone() - rhs.second.view(),
        }
    }

    pub(crate) fn multiply(&self, rhs: &Self) -> Self {
        let two = C::one() + C::one();

        Self {
            value: self.value.clone() * rhs.value.view(),

            first: self.first.clone() * rhs.value.view() + self.value.clone() * rhs.first.view(),

            second: self.second.clone() * rhs.value.view()
                + (self.first.clone() * rhs.first.view()).mapv(|x| two * x)
                + self.value.clone() * rhs.second.view(),
        }
    }

    pub(crate) fn negate(&self) -> Self {
        Self {
            value: -self.value.clone(),
            first: -self.first.clone(),
            second: -self.second.clone(),
        }
    }

    pub(crate) fn reciprocal(&self) -> Self {
        let two = C::one() + C::one();

        let squared = self.value.mapv(|x| x * x);
        let cubed = self.value.mapv(|x| x * x * x);

        Self {
            value: self.value.mapv(|x| C::one() / x),

            first: -self.first.clone() / squared.view(),

            second: self.first.mapv(|x| two * x * x) / cubed.view() - self.second.clone() / squared,
        }
    }

    pub(crate) fn divide(&self, rhs: &Self) -> Self {
        self.multiply(&rhs.reciprocal())
    }

    pub(crate) fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.value
    }

    pub(crate) fn first(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.first
    }

    pub(crate) fn second(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.second
    }

    pub(crate) fn chain_rule(self, rule: &ChainRule<ArrayBase<OwnedRepr<C>, D>>) -> Self {
        let primitive_first = self.first;

        let transformed_first = primitive_first.clone() * rule.first.view();

        let transformed_second =
            self.second * rule.first.mapv(|x| x * x) + primitive_first * rule.second.view();

        Self {
            value: self.value,
            first: transformed_first,
            second: transformed_second,
        }
    }
}
