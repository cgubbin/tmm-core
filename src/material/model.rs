use nalgebra::ComplexField;
use num_traits::{Float, One, Zero};

use super::{DerivativeOrder, Material, Sampled};
use crate::{
    ComplexScalar,
    material::{DifferentiableMaterial, DifferentiableMeromorphicMaterial, MeromorphicMaterial},
};

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Constant<R> {
    epsilon: R,
    mu: R,
}

impl<R> Constant<R> {
    pub fn new(epsilon: R, mu: R) -> Self {
        Self { epsilon, mu }
    }

    pub fn dielectric(epsilon: R) -> Self
    where
        R: Float,
    {
        Self::new(epsilon, R::one())
    }

    pub fn magnetodielectric(epsilon: R, mu: R) -> Self {
        Self::new(epsilon, mu)
    }

    pub fn vacuum() -> Self
    where
        R: Float,
    {
        Self::dielectric(R::one())
    }
}

impl<R> Material for Constant<R>
where
    R: Float + Zero + One,
{
    type Real = R;

    fn relative_permittivity<I, C>(&self, wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
        C: ComplexScalar<RealField = R>,
    {
        wavenumber.map(|_| C::from_real(self.epsilon))
    }

    fn relative_permeability<I, C>(&self, wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
        C: ComplexScalar<RealField = R>,
    {
        wavenumber.map(|_| C::from_real(self.mu))
    }
}

impl<R> DifferentiableMaterial for Constant<R>
where
    R: Float + Zero + One,
{
    fn relative_permittivity_derivative<I, C>(
        &self,
        wavenumber: I,
        _order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
        C: ComplexScalar<RealField = R>,
    {
        wavenumber.map(|_| C::zero())
    }
}

impl<R> MeromorphicMaterial for Constant<R>
where
    R: Float + Zero + One,
{
    fn relative_permittivity_complex<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: Sampled<Elem = C>,
    {
        vacuum_wavenumber.map(|_| C::from_real(self.epsilon))
    }

    fn relative_permeability_complex<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: Sampled<Elem = C>,
    {
        vacuum_wavenumber.map(|_| C::from_real(self.mu))
    }
}

impl<R> DifferentiableMeromorphicMaterial for Constant<R>
where
    R: Float + Zero + One,
{
    fn relative_permittivity_complex_derivative<I, C>(
        &self,
        vacuum_wavenumber: I,
        _order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: Sampled<Elem = C>,
    {
        vacuum_wavenumber.map(|_| C::zero())
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Lossy<R> {
    epsilon_re: R,
    epsilon_im: R,
    mu: R,
}

impl<R> Lossy<R> {
    pub fn new<C: ComplexField<RealField = R> + Copy>(epsilon: C, mu: R) -> Self {
        Self {
            epsilon_re: epsilon.real(),
            epsilon_im: epsilon.imaginary(),
            mu,
        }
    }

    pub fn dielectric<C: ComplexField<RealField = R> + Copy>(epsilon: C) -> Self {
        Self::new(epsilon, C::one().real())
    }

    pub fn magnetodielectric<C: ComplexField<RealField = R> + Copy>(epsilon: C, mu: R) -> Self {
        Self::new(epsilon, mu)
    }
}

impl<R> Material for Lossy<R>
where
    R: Float + Zero + One,
{
    type Real = R;

    fn relative_permittivity<I, C>(&self, wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
        C: ComplexScalar<RealField = R>,
    {
        wavenumber.map(|_| C::from_parts(self.epsilon_re, self.epsilon_im))
    }

    fn relative_permeability<I, C>(&self, wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
        C: ComplexScalar<RealField = R>,
    {
        wavenumber.map(|_| C::from_real(self.mu))
    }
}

impl<R> DifferentiableMaterial for Lossy<R>
where
    R: Float + Zero + One,
{
    fn relative_permittivity_derivative<I, C>(
        &self,
        wavenumber: I,
        _order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
        C: ComplexScalar<RealField = R>,
    {
        wavenumber.map(|_| C::zero())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::material::Scalar;

    use approx::assert_relative_eq;
    use num_complex::Complex;

    type C = Complex<f64>;

    #[test]
    fn constant_material_derivative_is_zero() {
        let material = Constant::new(4.0, 1.0);

        let deps: C =
            material.relative_permittivity_derivative(Scalar(1000.0), DerivativeOrder::First);

        assert_relative_eq!(deps.re, 0.0);
        assert_relative_eq!(deps.im, 0.0);
    }
}
