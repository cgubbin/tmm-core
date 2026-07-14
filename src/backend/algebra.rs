use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::jet::{ArrayJet, ArrayJetFirst},
};

pub(crate) trait ScalarAlgebra<C, D>: Sized
where
    D: Dimension,
{
    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D>;

    fn constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self;

    fn add(&self, rhs: &Self) -> Self;
    fn subtract(&self, rhs: &Self) -> Self;
    fn negate(&self) -> Self;
    fn multiply(&self, rhs: &Self) -> Self;
    fn reciprocal(&self) -> Self;

    /// Multiply the value and all derivative components by one constant.
    fn scale(&self, coefficient: C) -> Self;

    fn divide(&self, rhs: &Self) -> Self {
        self.multiply(&rhs.reciprocal())
    }
}

impl<C, D> ScalarAlgebra<C, D> for ArrayBase<OwnedRepr<C>, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn value(&self) -> &Self {
        self
    }

    fn constant_like(source: &Self, value: C) -> Self {
        source.mapv(|_| value)
    }

    fn add(&self, rhs: &Self) -> Self {
        self.clone() + rhs.view()
    }

    fn subtract(&self, rhs: &Self) -> Self {
        self.clone() - rhs.view()
    }

    fn negate(&self) -> Self {
        -self.clone()
    }

    fn multiply(&self, rhs: &Self) -> Self {
        self.clone() * rhs.view()
    }

    fn scale(&self, coefficient: C) -> Self {
        self.mapv(|x| x * coefficient)
    }

    fn reciprocal(&self) -> Self {
        self.mapv(|value| C::one() / value)
    }
}

impl<C, D> ScalarAlgebra<C, D> for ArrayJetFirst<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArrayJetFirst::value(self)
    }

    fn constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        ArrayJetFirst::constant_like(source, value)
    }

    fn add(&self, rhs: &Self) -> Self {
        ArrayJetFirst::add(self, rhs)
    }

    fn subtract(&self, rhs: &Self) -> Self {
        ArrayJetFirst::subtract(self, rhs)
    }

    fn negate(&self) -> Self {
        ArrayJetFirst::negate(self)
    }

    fn multiply(&self, rhs: &Self) -> Self {
        ArrayJetFirst::multiply(self, rhs)
    }

    fn scale(&self, coefficient: C) -> Self {
        ArrayJetFirst::scale_by(self, coefficient)
    }

    fn reciprocal(&self) -> Self {
        ArrayJetFirst::reciprocal(self)
    }
}

impl<C, D> ScalarAlgebra<C, D> for ArrayJet<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArrayJet::value(self)
    }

    fn constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        ArrayJet::constant_like(source, value)
    }

    fn add(&self, rhs: &Self) -> Self {
        ArrayJet::add(self, rhs)
    }

    fn subtract(&self, rhs: &Self) -> Self {
        ArrayJet::subtract(self, rhs)
    }

    fn negate(&self) -> Self {
        ArrayJet::negate(self)
    }

    fn multiply(&self, rhs: &Self) -> Self {
        ArrayJet::multiply(self, rhs)
    }

    fn scale(&self, coefficient: C) -> Self {
        ArrayJet::scale_by(self, coefficient)
    }

    fn reciprocal(&self) -> Self {
        ArrayJet::reciprocal(self)
    }
}
