use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::{
        field::{CartesianVector3, CartesianVectorAlgebra},
        jet::{ArrayJet, ArrayJetFirst, Jet, JetFirst},
    },
};

pub(crate) trait ScalarAlgebra<C, D>: Sized
where
    D: Dimension,
    C: ComplexScalar,
{
    type RealField;
    type Vector: CartesianVectorAlgebra<C, D>;

    fn into_cartesian_vector(x: Self, y: Self, z: Self) -> Self::Vector;

    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D>;
    fn conjugate(&self) -> Self;
    fn real_part(&self) -> Self::RealField;
    fn magnitude_squared(&self) -> Self::RealField;

    fn exp(&self) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;

    fn constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self;

    fn zero_like(&self) -> Self;

    fn add(&self, rhs: &Self) -> Self;
    fn subtract(&self, rhs: &Self) -> Self;
    fn negate(&self) -> Self;
    fn multiply(&self, rhs: &Self) -> Self;

    fn square(&self) -> Self {
        self.multiply(self)
    }
    fn reciprocal(&self) -> Self;

    fn sqrt(&self) -> Self;

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
    type RealField = ArrayBase<OwnedRepr<C::RealField>, D>;
    type Vector = CartesianVector3<C, D>;

    fn value(&self) -> &Self {
        self
    }

    fn into_cartesian_vector(x: Self, y: Self, z: Self) -> Self::Vector {
        CartesianVector3::new(x, y, z)
    }

    fn constant_like(source: &Self, value: C) -> Self {
        source.mapv(|_| value)
    }

    fn zero_like(&self) -> Self {
        self.mapv(|_| C::zero())
    }

    fn exp(&self) -> Self {
        self.mapv(|x| x.exp())
    }

    fn sin(&self) -> Self {
        self.mapv(|x| x.sin())
    }

    fn cos(&self) -> Self {
        self.mapv(|x| x.cos())
    }

    fn conjugate(&self) -> Self {
        self.mapv(|each| each.conjugate())
    }

    fn real_part(&self) -> Self::RealField {
        self.mapv(|each| each.real())
    }

    fn magnitude_squared(&self) -> Self::RealField {
        self.mapv(|each| each.modulus_squared())
    }

    fn add(&self, rhs: &Self) -> Self {
        self.clone() + rhs.view()
    }

    fn subtract(&self, rhs: &Self) -> Self {
        self.clone() - rhs.view()
    }

    fn sqrt(&self) -> Self {
        self.mapv(|each| each.sqrt())
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
    type RealField = ArrayJetFirst<C::RealField, D>;
    type Vector = JetFirst<CartesianVector3<C, D>>;

    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArrayJetFirst::value(self)
    }

    fn into_cartesian_vector(x: Self, y: Self, z: Self) -> Self::Vector {
        JetFirst::from_parts(
            CartesianVector3::new(x.value().clone(), y.value().clone(), z.value().clone()),
            CartesianVector3::new(x.first().clone(), y.first().clone(), z.first().clone()),
        )
    }

    fn constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        ArrayJetFirst::constant_like(source, value)
    }

    fn exp(&self) -> Self {
        ArrayJetFirst::exp(self.clone())
    }

    fn sin(&self) -> Self {
        ArrayJetFirst::sin(self.clone())
    }

    fn cos(&self) -> Self {
        ArrayJetFirst::cos(self.clone())
    }

    fn zero_like(&self) -> Self {
        let source = self.value();
        Self::constant_like(source, C::zero())
    }

    fn conjugate(&self) -> Self {
        ArrayJetFirst::conjugated(&self)
    }

    fn real_part(&self) -> Self::RealField {
        ArrayJetFirst::real(&self)
    }

    fn magnitude_squared(&self) -> Self::RealField {
        (self.multiply(&self.conjugated())).real_part()
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

    fn sqrt(&self) -> Self {
        ArrayJetFirst::sqrt(self.clone())
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
    type RealField = ArrayJet<C::RealField, D>;
    type Vector = Jet<CartesianVector3<C, D>>;

    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArrayJet::value(self)
    }

    fn into_cartesian_vector(x: Self, y: Self, z: Self) -> Self::Vector {
        Jet::from_parts(
            CartesianVector3::new(x.value().clone(), y.value().clone(), z.value().clone()),
            CartesianVector3::new(x.first().clone(), y.first().clone(), z.first().clone()),
            CartesianVector3::new(x.second().clone(), y.second().clone(), z.second().clone()),
        )
    }

    fn constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        ArrayJet::constant_like(source, value)
    }

    fn zero_like(&self) -> Self {
        let source = self.value();
        Self::constant_like(source, C::zero())
    }

    fn exp(&self) -> Self {
        ArrayJet::exp(self.clone())
    }

    fn sin(&self) -> Self {
        ArrayJet::sin(self.clone())
    }

    fn cos(&self) -> Self {
        ArrayJet::cos(self.clone())
    }

    fn conjugate(&self) -> Self {
        ArrayJet::conjugated(&self)
    }

    fn real_part(&self) -> Self::RealField {
        ArrayJet::real(&self)
    }

    fn magnitude_squared(&self) -> Self::RealField {
        (self.multiply(&self.conjugated())).real_part()
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

    fn sqrt(&self) -> Self {
        ArrayJet::sqrt(self.clone())
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
