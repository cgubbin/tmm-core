use num_traits::{Float, One, Zero};

use super::{DerivativeOrder, DrudeLorentzBuilder, Material, Sampled, SpectralVariable};
use crate::ComplexScalar;

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

    fn is_dispersive(&self) -> bool {
        false
    }

    fn static_permittivity(&self) -> R {
        self.epsilon
    }

    fn refractive_index<I, C>(&self, wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
        C: ComplexScalar<RealField = R>,
    {
        wavenumber.map(|_| C::from_real(self.epsilon.sqrt()))
    }

    fn relative_permittivity<I, C>(&self, wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
        C: ComplexScalar<RealField = R>,
    {
        wavenumber.map(|_| C::from_real(self.epsilon))
    }

    fn relative_permeability<I, C>(&self, wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
        C: ComplexScalar<RealField = R>,
    {
        wavenumber.map(|_| C::from_real(self.mu))
    }

    fn relative_permittivity_derivative<I, C>(
        &self,
        wavenumber: I,
        _order: DerivativeOrder,
        _variable: SpectralVariable,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
        C: ComplexScalar<RealField = R>,
    {
        wavenumber.map(|_| C::zero())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DrudeLorentz<R> {
    epsilon_infinity: R,
    drude: Option<Drude<R>>,
    lorentz: Vec<Lorentz<R>>,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Drude<R> {
    pub plasma_frequency: R,
    pub damping_frequency: R,
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Lorentz<R> {
    pub strength: R,
    pub transverse_frequency: R,
    pub damping_frequency: R,
}

impl<R> DrudeLorentz<R> {
    pub fn builder(epsilon_infinity: R) -> DrudeLorentzBuilder<R> {
        DrudeLorentzBuilder::new(epsilon_infinity)
    }

    pub(crate) fn from_parts(
        epsilon_infinity: R,
        drude: Option<Drude<R>>,
        lorentz: Vec<Lorentz<R>>,
    ) -> Self {
        Self {
            epsilon_infinity,
            drude,
            lorentz,
        }
    }
}

impl<R> Material for DrudeLorentz<R>
where
    R: Float + Zero + One,
{
    type Real = R;

    fn is_dispersive(&self) -> bool {
        self.drude.is_some() || !self.lorentz.is_empty()
    }

    fn static_permittivity(&self) -> R {
        self.epsilon_infinity
    }

    fn relative_permittivity<I, C>(&self, wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
        C: ComplexScalar<RealField = R> + Copy,
    {
        wavenumber.map(|w| self.relative_permittivity_at(w))
    }

    fn refractive_index<I, C>(&self, wavenumber: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: Sampled<Elem = C>,
    {
        wavenumber.map(|w| self.relative_permittivity_at(w).sqrt())
    }

    fn relative_permittivity_derivative<I, C>(
        &self,
        wavenumber: I,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
        C: ComplexScalar<RealField = R> + Copy,
    {
        wavenumber.map(|w| self.relative_permittivity_derivative_at(w, order, variable))
    }
}

impl<R> DrudeLorentz<R>
where
    R: Float,
{
    fn relative_permittivity_at<C>(&self, w: C) -> C
    where
        C: ComplexScalar<RealField = R> + Copy,
    {
        let mut eps = C::from_real(self.epsilon_infinity);

        if let Some(drude) = self.drude {
            eps += drude.epsilon(w);
        }

        for lorentz in &self.lorentz {
            eps += lorentz.epsilon(w);
        }

        eps
    }

    fn relative_permittivity_derivative_at<C>(
        &self,
        w: C,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> C
    where
        C: ComplexScalar<RealField = R> + Copy,
    {
        let mut deps = C::zero();

        if let Some(drude) = self.drude {
            deps += drude.epsilon_derivative(w, order, variable);
        }

        for lorentz in &self.lorentz {
            deps += lorentz.epsilon_derivative(w, order, variable);
        }

        deps
    }
}

impl<R> Drude<R>
where
    R: Float,
{
    fn epsilon<C>(&self, w: C) -> C
    where
        C: ComplexScalar<RealField = R>,
    {
        let wp2 = C::from_real(self.plasma_frequency).powi(2);
        let gamma = C::i() * C::from_real(self.damping_frequency);

        -wp2 / w / (w - gamma)
    }

    fn epsilon_derivative<C>(&self, w: C, order: DerivativeOrder, variable: SpectralVariable) -> C
    where
        C: ComplexScalar<RealField = R>,
    {
        let two = C::one() + C::one();
        let four = two + two;

        let wp2 = C::from_real(self.plasma_frequency).powi(2);
        let gamma = C::i() * C::from_real(self.damping_frequency);
        let denom = w - gamma;

        match order {
            DerivativeOrder::First => {
                let dedw = wp2 * (two * w - gamma) / w.powi(2) / denom.powi(2);

                match variable {
                    SpectralVariable::VacuumWavenumber => dedw,
                    SpectralVariable::VacuumWavenumberSquared => dedw / (two * w),
                }
            }
            DerivativeOrder::Second => {
                let d2edw2 =
                    -two * wp2 * (w * w + w * denom + denom.powi(2)) / w.powi(3) / denom.powi(3);

                match variable {
                    SpectralVariable::VacuumWavenumber => d2edw2,
                    SpectralVariable::VacuumWavenumberSquared => {
                        let dedw = self.epsilon_derivative(
                            w,
                            DerivativeOrder::First,
                            SpectralVariable::VacuumWavenumber,
                        );

                        d2edw2 / (four * w.powi(2)) - dedw / (four * w.powi(3))
                    }
                }
            }
        }
    }
}

impl<R> Lorentz<R>
where
    R: Float,
{
    pub fn from_frequencies(
        epsilon_infinity: R,
        longitudinal_frequency: R,
        transverse_frequency: R,
        damping_frequency: R,
    ) -> Self {
        let strength = epsilon_infinity
            * (longitudinal_frequency.powi(2) / transverse_frequency.powi(2) - R::one());

        Self {
            strength,
            transverse_frequency,
            damping_frequency,
        }
    }

    fn epsilon<C>(&self, w: C) -> C
    where
        C: ComplexScalar<RealField = R>,
    {
        let strength = C::from_real(self.strength);
        let wt2 = C::from_real(self.transverse_frequency).powi(2);
        let gamma = C::i() * C::from_real(self.damping_frequency);

        strength * wt2 / (wt2 - w * (w - gamma))
    }

    fn epsilon_derivative<C>(&self, w: C, order: DerivativeOrder, variable: SpectralVariable) -> C
    where
        C: ComplexScalar<RealField = R>,
    {
        let two = C::one() + C::one();
        let four = two + two;

        let strength = C::from_real(self.strength);
        let wt2 = C::from_real(self.transverse_frequency).powi(2);
        let gamma = C::i() * C::from_real(self.damping_frequency);

        let denom = wt2 - w * (w - gamma);
        let numerator_term = two * w - gamma;

        match order {
            DerivativeOrder::First => {
                let dedw = strength * wt2 * numerator_term / denom.powi(2);

                match variable {
                    SpectralVariable::VacuumWavenumber => dedw,
                    SpectralVariable::VacuumWavenumberSquared => dedw / (two * w),
                }
            }
            DerivativeOrder::Second => {
                let d2edw2 =
                    two * strength * wt2 * (denom + numerator_term.powi(2)) / denom.powi(3);

                match variable {
                    SpectralVariable::VacuumWavenumber => d2edw2,
                    SpectralVariable::VacuumWavenumberSquared => {
                        let dedw = self.epsilon_derivative(
                            w,
                            DerivativeOrder::First,
                            SpectralVariable::VacuumWavenumber,
                        );

                        d2edw2 / (four * w.powi(2)) - dedw / (four * w.powi(3))
                    }
                }
            }
        }
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
    fn constant_material_is_not_dispersive() {
        let material = Constant::new(4.0, 1.0);

        assert!(!material.is_dispersive());
        assert_eq!(material.static_permittivity(), 4.0);

        let eps: C = material.relative_permittivity(Scalar(C::new(1000.0, 0.0)));

        assert_relative_eq!(eps.re, 4.0);
        assert_relative_eq!(eps.im, 0.0);
    }

    #[test]
    fn constant_material_derivative_is_zero() {
        let material = Constant::new(4.0, 1.0);

        let deps: C = material.relative_permittivity_derivative(
            Scalar(C::new(1000.0, 0.0)),
            DerivativeOrder::First,
            SpectralVariable::VacuumWavenumber,
        );

        assert_relative_eq!(deps.re, 0.0);
        assert_relative_eq!(deps.im, 0.0);
    }

    #[test]
    fn drude_lorentz_reports_dispersion() {
        let material = DrudeLorentz::builder(1.0)
            .with_drude(68153.8, 2382.6)
            .build();

        assert!(material.is_dispersive());
        assert_eq!(material.static_permittivity(), 1.0);
    }

    #[test]
    fn drude_permittivity_matches_formula() {
        let wp = 68153.8;
        let gamma = 2382.6;
        let w = C::new(1000.0, 0.0);

        let material = DrudeLorentz::builder(1.0).with_drude(wp, gamma).build();

        let actual: C = material.relative_permittivity(Scalar(w));

        let expected = C::new(1.0, 0.0) - C::new(wp * wp, 0.0) / w / (w - C::new(0.0, gamma));

        assert_relative_eq!(actual.re, expected.re, max_relative = 1e-12);
        assert_relative_eq!(actual.im, expected.im, max_relative = 1e-12);
    }

    #[test]
    fn lorentz_permittivity_matches_formula() {
        let strength = 0.5;
        let wt = 800.0;
        let gamma = 4.0;
        let w = C::new(1000.0, 0.0);

        let material = DrudeLorentz::builder(2.0)
            .with_lorentz(strength, wt, gamma)
            .build();

        let actual: C = material.relative_permittivity(Scalar(w));

        let expected = C::new(2.0, 0.0)
            + C::new(strength * wt * wt, 0.0)
                / (C::new(wt * wt, 0.0) - w * (w - C::new(0.0, gamma)));

        assert_relative_eq!(actual.re, expected.re, max_relative = 1e-12);
        assert_relative_eq!(actual.im, expected.im, max_relative = 1e-12);
    }
}

#[cfg(test)]
mod ndarray_tests {
    use super::*;
    use crate::material::Scalar;

    use approx::assert_relative_eq;
    use ndarray::{Array2, arr1};
    use num_complex::Complex;

    type C = Complex<f64>;

    #[test]
    fn scalar_and_array_paths_agree() {
        let material = DrudeLorentz::builder(1.0)
            .with_drude(68153.8, 2382.6)
            .with_lorentz(0.2, 900.0, 5.0)
            .build();

        let values = arr1(&[C::new(500.0, 0.0), C::new(1000.0, 0.0), C::new(1500.0, 0.0)]);

        let array_eps = material.relative_permittivity(values.clone());

        for (w, eps_from_array) in values.iter().zip(array_eps.iter()) {
            let eps_from_scalar: C = material.relative_permittivity(Scalar(*w));

            assert_relative_eq!(eps_from_scalar.re, eps_from_array.re, max_relative = 1e-12);
            assert_relative_eq!(eps_from_scalar.im, eps_from_array.im, max_relative = 1e-12);
        }
    }

    #[test]
    fn array2_shape_is_preserved() {
        let material = Constant::new(3.0, 1.0);

        let values = Array2::from_shape_vec(
            (2, 3),
            vec![
                C::new(1.0, 0.0),
                C::new(2.0, 0.0),
                C::new(3.0, 0.0),
                C::new(4.0, 0.0),
                C::new(5.0, 0.0),
                C::new(6.0, 0.0),
            ],
        )
        .unwrap();

        let eps = material.relative_permittivity(values);

        assert_eq!(eps.shape(), &[2, 3]);

        for value in eps {
            assert_relative_eq!(value.re, 3.0);
            assert_relative_eq!(value.im, 0.0);
        }
    }

    #[test]
    fn permeability_reflects_value() {
        let material = Constant::new(4.0, 2.0);

        let mu: C = material.relative_permeability(Scalar(C::new(1000.0, 0.0)));

        assert_relative_eq!(mu.re, 2.0);
        assert_relative_eq!(mu.im, 0.0);
    }

    #[test]
    fn permeability_derivative_defaults_to_zero() {
        let material = Constant::new(4.0, 1.0);

        let dmu: C = material.relative_permeability_derivative(
            Scalar(C::new(1000.0, 0.0)),
            DerivativeOrder::First,
            SpectralVariable::VacuumWavenumber,
        );

        assert_relative_eq!(dmu.re, 0.0);
        assert_relative_eq!(dmu.im, 0.0);
    }
}
