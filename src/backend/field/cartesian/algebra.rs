use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::{
        field::CartesianVector3,
        jet::{Jet, JetFirst},
    },
};

pub(crate) trait CartesianVectorAlgebra<C, D>: Clone
where
    C: ComplexScalar,
    D: Dimension,
{
    type RealVector;
    type ComplexScalarField;
    type RealScalarField;

    fn cross(&self, rhs: &Self) -> Self;

    fn conjugate(&self) -> Self;

    fn scale_by(&self, factor: C) -> Self;

    fn real_part(&self) -> Self::RealVector;

    fn hermitian_dot(&self, rhs: &Self) -> Self::ComplexScalarField;

    fn scalar_real_part(value: Self::ComplexScalarField) -> Self::RealScalarField;
}

impl<C, D> CartesianVectorAlgebra<C, D> for CartesianVector3<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    type RealVector = CartesianVector3<C::RealField, D>;

    type ComplexScalarField = ArrayBase<OwnedRepr<C>, D>;

    type RealScalarField = ArrayBase<OwnedRepr<C::RealField>, D>;

    fn cross(&self, rhs: &Self) -> Self {
        CartesianVector3::cross(self, rhs)
    }

    fn conjugate(&self) -> Self {
        CartesianVector3::conjugate(self)
    }

    fn scale_by(&self, factor: C) -> Self {
        self.clone() * factor
    }

    fn real_part(&self) -> Self::RealVector {
        self.map(|value| value.real())
    }

    fn hermitian_dot(&self, rhs: &Self) -> Self::ComplexScalarField {
        CartesianVector3::hermitian_dot(self, rhs)
    }

    fn scalar_real_part(value: Self::ComplexScalarField) -> Self::RealScalarField {
        value.mapv(|value| value.real())
    }
}

impl<C, D> CartesianVectorAlgebra<C, D> for JetFirst<CartesianVector3<C, D>>
where
    C: ComplexScalar,
    D: Dimension,
{
    type RealVector = JetFirst<CartesianVector3<C::RealField, D>>;

    type ComplexScalarField = JetFirst<ArrayBase<OwnedRepr<C>, D>>;

    type RealScalarField = JetFirst<ArrayBase<OwnedRepr<C::RealField>, D>>;

    fn cross(&self, rhs: &Self) -> Self {
        JetFirst::cross(self, rhs)
    }

    fn conjugate(&self) -> Self {
        JetFirst::conjugated(self)
    }

    fn scale_by(&self, factor: C) -> Self {
        JetFirst::scale_by(self, factor)
    }

    fn real_part(&self) -> Self::RealVector {
        JetFirst::real(self)
    }

    fn hermitian_dot(&self, rhs: &Self) -> Self::ComplexScalarField {
        JetFirst::hermitian_dot_product(self, rhs)
    }

    fn scalar_real_part(value: Self::ComplexScalarField) -> Self::RealScalarField {
        JetFirst::real(&value)
    }
}

impl<C, D> CartesianVectorAlgebra<C, D> for Jet<CartesianVector3<C, D>>
where
    C: ComplexScalar,
    D: Dimension,
{
    type RealVector = Jet<CartesianVector3<C::RealField, D>>;

    type ComplexScalarField = Jet<ArrayBase<OwnedRepr<C>, D>>;

    type RealScalarField = Jet<ArrayBase<OwnedRepr<C::RealField>, D>>;

    fn cross(&self, rhs: &Self) -> Self {
        Jet::cross(self, rhs)
    }

    fn conjugate(&self) -> Self {
        Jet::conjugated(self)
    }

    fn scale_by(&self, factor: C) -> Self {
        Jet::scale_by(self, factor)
    }

    fn real_part(&self) -> Self::RealVector {
        Jet::real(self)
    }

    fn hermitian_dot(&self, rhs: &Self) -> Self::ComplexScalarField {
        Jet::hermitian_dot_product(self, rhs)
    }

    fn scalar_real_part(value: Self::ComplexScalarField) -> Self::RealScalarField {
        Jet::real(&value)
    }
}
