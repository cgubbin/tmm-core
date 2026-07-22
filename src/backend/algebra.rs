use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::backend::{
    field::{CartesianVector3, CartesianVectorAlgebra},
    jet::{ArrayJet, ArrayJetFirst, ArraySpectralJet, Jet, JetFirst, SpectralJet},
};

pub trait ScalarAlgebra<T, D>: Sized + std::fmt::Debug
where
    D: Dimension,
{
    type RealField;
    type Vector: CartesianVectorAlgebra<T, D>;

    fn into_cartesian_vector(x: Self, y: Self, z: Self) -> Self::Vector;

    fn value(&self) -> &ArrayBase<OwnedRepr<T>, D>;

    fn from_value(source: ArrayBase<OwnedRepr<T>, D>) -> Self;

    fn conjugate(&self) -> Self;
    fn real_part(&self) -> Self::RealField;
    fn magnitude_squared(&self) -> Self::RealField;

    fn exp(&self) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;

    fn constant_like(source: &ArrayBase<OwnedRepr<T>, D>, value: T) -> Self;
    fn scalar_constant_like(&self, value: T) -> Self;

    fn structural_like(source: &ArrayBase<OwnedRepr<T>, D>, value: T) -> Self;

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
    fn scale(&self, coefficient: T) -> Self;

    fn divide(&self, rhs: &Self) -> Self {
        self.multiply(&rhs.reciprocal())
    }

    fn all_finite(&self) -> bool;
}

impl<C, D> ScalarAlgebra<C, D> for ArrayBase<OwnedRepr<C>, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type RealField = ArrayBase<OwnedRepr<C::RealField>, D>;
    type Vector = CartesianVector3<C, D>;

    fn value(&self) -> &Self {
        self
    }

    fn from_value(source: ArrayBase<OwnedRepr<C>, D>) -> Self {
        source
    }

    fn into_cartesian_vector(x: Self, y: Self, z: Self) -> Self::Vector {
        CartesianVector3::new(x, y, z)
    }

    fn constant_like(source: &Self, value: C) -> Self {
        source.mapv(|_| value)
    }

    fn structural_like(source: &Self, value: C) -> Self {
        source.mapv(|_| value)
    }

    fn scalar_constant_like(&self, value: C) -> Self {
        self.mapv(|_| value)
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

    fn all_finite(&self) -> bool {
        self.iter().all(complex_is_finite)
    }
}

impl<C, D> ScalarAlgebra<C, D> for ArrayJetFirst<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type RealField = ArrayJetFirst<C::RealField, D>;
    type Vector = JetFirst<CartesianVector3<C, D>>;

    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArrayJetFirst::value(self)
    }

    fn from_value(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        let zero = value.mapv(|_| C::zero());
        ArrayJetFirst::from_parts(value, zero)
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

    fn scalar_constant_like(&self, value: C) -> Self {
        ArrayJetFirst::constant_like(self.value(), value)
    }

    fn structural_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        JetFirst::from_parts(
            ndarray::Array::from_elem(source.raw_dim(), value),
            ndarray::Array::from_elem(source.raw_dim(), C::one()),
        )
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

    fn all_finite(&self) -> bool {
        self.value().iter().all(complex_is_finite) && self.first().iter().all(complex_is_finite)
    }
}

impl<C, D> ScalarAlgebra<C, D> for ArrayJet<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type RealField = ArrayJet<C::RealField, D>;
    type Vector = Jet<CartesianVector3<C, D>>;

    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArrayJet::value(self)
    }

    fn from_value(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        let zero = value.mapv(|_| C::zero());
        ArrayJet::from_parts(value, zero.clone(), zero)
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

    fn scalar_constant_like(&self, value: C) -> Self {
        ArrayJet::constant_like(self.value(), value)
    }

    fn structural_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        Jet::from_parts(
            ndarray::Array::from_elem(source.raw_dim(), value),
            ndarray::Array::from_elem(source.raw_dim(), C::one()),
            ndarray::Array::from_elem(source.raw_dim(), C::zero()),
        )
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

    fn all_finite(&self) -> bool {
        self.value().iter().all(complex_is_finite)
            && self.first().iter().all(complex_is_finite)
            && self.second().iter().all(complex_is_finite)
    }
}

impl<C, D> ScalarAlgebra<C, D> for ArraySpectralJet<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type RealField = ArraySpectralJet<C::RealField, D>;
    type Vector = SpectralJet<CartesianVector3<C, D>>;

    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArraySpectralJet::value(self)
    }

    fn from_value(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArraySpectralJet::constant(value)
    }

    fn into_cartesian_vector(x: Self, y: Self, z: Self) -> Self::Vector {
        SpectralJet::from_parts(
            CartesianVector3::new(x.value().clone(), y.value().clone(), z.value().clone()),
            CartesianVector3::new(x.dk0().clone(), y.dk0().clone(), z.dk0().clone()),
            CartesianVector3::new(x.dkx().clone(), y.dkx().clone(), z.dkx().clone()),
            CartesianVector3::new(
                x.dk0_dk0().clone(),
                y.dk0_dk0().clone(),
                z.dk0_dk0().clone(),
            ),
            CartesianVector3::new(
                x.dk0_dkx().clone(),
                y.dk0_dkx().clone(),
                z.dk0_dkx().clone(),
            ),
            CartesianVector3::new(
                x.dkx_dkx().clone(),
                y.dkx_dkx().clone(),
                z.dkx_dkx().clone(),
            ),
        )
    }

    fn constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        ArraySpectralJet::constant_like(source, value)
    }

    fn scalar_constant_like(&self, value: C) -> Self {
        ArraySpectralJet::constant_like(self.value(), value)
    }

    fn structural_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        ArraySpectralJet::parallel_wavenumber(ndarray::Array::from_elem(source.raw_dim(), value))
    }

    fn zero_like(&self) -> Self {
        let source = self.value();
        Self::constant_like(source, C::zero())
    }

    fn exp(&self) -> Self {
        ArraySpectralJet::exp(&self)
    }

    fn sin(&self) -> Self {
        ArraySpectralJet::sin(&self)
    }

    fn cos(&self) -> Self {
        ArraySpectralJet::cos(&self)
    }

    fn conjugate(&self) -> Self {
        ArraySpectralJet::conjugate(&self)
    }

    fn real_part(&self) -> Self::RealField {
        ArraySpectralJet::real_part(&self)
    }

    fn magnitude_squared(&self) -> Self::RealField {
        (self.multiply(&self.conjugate())).real_part()
    }

    fn add(&self, rhs: &Self) -> Self {
        SpectralJet::add(self, rhs)
    }

    fn subtract(&self, rhs: &Self) -> Self {
        SpectralJet::subtract(self, rhs)
    }

    fn negate(&self) -> Self {
        SpectralJet::negate(self)
    }

    fn sqrt(&self) -> Self {
        ArraySpectralJet::sqrt(&self)
    }

    fn multiply(&self, rhs: &Self) -> Self {
        ArraySpectralJet::multiply(self, rhs)
    }

    fn scale(&self, coefficient: C) -> Self {
        ArraySpectralJet::scale_by(self, coefficient)
    }

    fn reciprocal(&self) -> Self {
        ArraySpectralJet::reciprocal(self)
    }

    fn all_finite(&self) -> bool {
        self.value().iter().all(complex_is_finite)
            && self.dkx().iter().all(complex_is_finite)
            && self.dk0().iter().all(complex_is_finite)
            && self.dk0_dk0().iter().all(complex_is_finite)
            && self.dkx_dkx().iter().all(complex_is_finite)
            && self.dk0_dkx().iter().all(complex_is_finite)
    }
}

fn complex_is_finite<C>(value: &C) -> bool
where
    C: ComplexField + Copy,
{
    value.real().is_finite() && value.imaginary().is_finite()
}

pub(crate) trait FirstOrderFunctionAlgebra<C, D>: ScalarAlgebra<C, D>
where
    C: ComplexField,
    D: Dimension,
{
    /// Lift `f(argument)` from sampled f, f′ and f″.
    fn compose_sampled_function(
        argument: &Self,
        value: ArrayBase<OwnedRepr<C>, D>,
        first: ArrayBase<OwnedRepr<C>, D>,
    ) -> Self;
}

pub(crate) trait SecondOrderFunctionAlgebra<C, D>: ScalarAlgebra<C, D>
where
    C: ComplexField,
    D: Dimension,
{
    /// Lift `f(argument)` from sampled f, f′ and f″.
    fn compose_sampled_function(
        argument: &Self,
        value: ArrayBase<OwnedRepr<C>, D>,
        first: ArrayBase<OwnedRepr<C>, D>,
        second: ArrayBase<OwnedRepr<C>, D>,
    ) -> Self;
}
