use num_traits::{Float, One};

use crate::{
    ComplexScalar,
    algebra::{ComplexJet, Jet, RealScalarAlgebra, ScalarAlgebra},
};

/// Pointwise Cartesian electric displacement and magnetic induction phasor fields.
///
/// The field uses the electromagnetic normalization chosen by the producing
/// backend. The electric displacement and magnetic induction vectors share the same ndarray sampling
/// shape.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstitutiveFields<V> {
    electric_displacement: V,
    magnetic_induction: V,
}

impl<V> ConstitutiveFields<V> {
    pub(crate) fn new(electric_displacement: V, magnetic_induction: V) -> Self {
        Self {
            electric_displacement,
            magnetic_induction,
        }
    }

    pub fn electric_displacement(&self) -> &V {
        &self.electric_displacement
    }

    pub fn magnetic_induction(&self) -> &V {
        &self.magnetic_induction
    }

    pub fn into_parts(self) -> (V, V) {
        (self.electric_displacement, self.magnetic_induction)
    }

    pub fn map_vectors<U>(self, f: impl Fn(V) -> U) -> ConstitutiveFields<U> {
        ConstitutiveFields {
            electric_displacement: f(self.electric_displacement),
            magnetic_induction: f(self.magnetic_induction),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IsotropicConstitutiveParameters<A> {
    epsilon: A,
    mu: A,
}

impl<A> IsotropicConstitutiveParameters<A> {
    pub(crate) const fn new(epsilon: A, mu: A) -> Self {
        Self { epsilon, mu }
    }

    pub fn epsilon(&self) -> &A {
        &self.epsilon
    }

    pub fn mu(&self) -> &A {
        &self.mu
    }

    pub fn into_parts(self) -> (A, A) {
        (self.epsilon, self.mu)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IsotropicConstitutiveSpectralFirst<A> {
    epsilon: A,
    mu: A,
}

impl<A> IsotropicConstitutiveSpectralFirst<A> {
    pub(crate) const fn new(epsilon: A, mu: A) -> Self {
        Self { epsilon, mu }
    }

    #[cfg(test)]
    pub fn epsilon(&self) -> &A {
        &self.epsilon
    }

    #[cfg(test)]
    pub fn mu(&self) -> &A {
        &self.mu
    }

    pub fn into_parts(self) -> (A, A) {
        (self.epsilon, self.mu)
    }
}

pub(crate) struct IsotropicConstitutiveSpectralData<A> {
    parameters: IsotropicConstitutiveParameters<A>,
    spectral_first: IsotropicConstitutiveSpectralFirst<A>,
    vacuum_angular_wavenumber: A,
}

impl<A> IsotropicConstitutiveSpectralData<A> {
    pub(crate) fn new(
        parameters: IsotropicConstitutiveParameters<A>,
        spectral_first: IsotropicConstitutiveSpectralFirst<A>,
        vacuum_angular_wavenumber: A,
    ) -> Self {
        Self {
            parameters,
            spectral_first,
            vacuum_angular_wavenumber,
        }
    }

    #[cfg(test)]
    pub(crate) fn parameters(&self) -> &IsotropicConstitutiveParameters<A> {
        &self.parameters
    }

    #[cfg(test)]
    pub(crate) fn mu_spectral_first(&self) -> &A {
        self.spectral_first.mu()
    }

    #[cfg(test)]
    pub(crate) fn epsilon_spectral_first(&self) -> &A {
        self.spectral_first.epsilon()
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        IsotropicConstitutiveParameters<A>,
        IsotropicConstitutiveSpectralFirst<A>,
        A,
    ) {
        (
            self.parameters,
            self.spectral_first,
            self.vacuum_angular_wavenumber,
        )
    }

    pub(crate) fn into_brillouin_factors(self) -> IsotropicBrillouinFactors<A>
    where
        A: ScalarAlgebra,
    {
        let (parameters, spectral_first, vacuum_angular_wavenumber) = self.into_parts();

        let (epsilon, mu) = parameters.into_parts();
        let (epsilon_spectral_first, mu_spectral_first) = spectral_first.into_parts();

        let electric = epsilon.add(&vacuum_angular_wavenumber.multiply(&epsilon_spectral_first));

        let magnetic = mu.add(&vacuum_angular_wavenumber.multiply(&mu_spectral_first));

        IsotropicBrillouinFactors::new(electric, magnetic)
    }
}

/// Spectral Brillouin constitutive factors for an isotropic medium.
///
/// For the canonical vacuum angular wavenumber `k0`, these store
///
/// ```text
/// electric = ∂(k0 ε) / ∂k0
/// magnetic = ∂(k0 μ) / ∂k0
/// ```
///
/// before any Hermitian or bilinear field contraction is applied.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IsotropicBrillouinFactors<A> {
    electric: A,
    magnetic: A,
}

impl<A> IsotropicBrillouinFactors<A> {
    pub(crate) const fn new(electric: A, magnetic: A) -> Self {
        Self { electric, magnetic }
    }

    #[cfg(test)]
    pub(crate) fn electric(&self) -> &A {
        &self.electric
    }

    #[cfg(test)]
    pub(crate) fn magnetic(&self) -> &A {
        &self.magnetic
    }

    #[cfg(test)]
    pub(crate) fn into_parts(self) -> (A, A) {
        (self.electric, self.magnetic)
    }
}

impl<A> IsotropicBrillouinFactors<A>
where
    A: RealScalarAlgebra,
{
    pub(crate) fn into_hermitian_energy_coefficients(
        self,
    ) -> ElectromagneticEnergyCoefficients<A::RealJet> {
        ElectromagneticEnergyCoefficients::new(self.electric.real(), self.magnetic.real())
    }
}

pub(crate) struct ElectromagneticEnergyCoefficients<R> {
    electric: R,
    magnetic: R,
}

impl<A> ElectromagneticEnergyCoefficients<A> {
    pub(crate) const fn new(electric: A, magnetic: A) -> Self {
        Self { electric, magnetic }
    }

    pub(crate) fn electric(&self) -> &A {
        &self.electric
    }

    pub(crate) fn magnetic(&self) -> &A {
        &self.magnetic
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ElectromagneticDissipationCoefficients<R> {
    electric: R,
    magnetic: R,
}

impl<A> ElectromagneticDissipationCoefficients<A> {
    pub(crate) const fn new(electric: A, magnetic: A) -> Self {
        Self { electric, magnetic }
    }

    pub(crate) fn electric(&self) -> &A {
        &self.electric
    }

    pub(crate) fn magnetic(&self) -> &A {
        &self.magnetic
    }
}

pub(crate) fn electromagnetic_dissipation_coefficients<A>(
    constitutive: &IsotropicConstitutiveParameters<A>,
    vacuum_angular_wavenumber: &A,
) -> ElectromagneticDissipationCoefficients<A::RealJet>
where
    A: ComplexJet + RealScalarAlgebra,
    A::RealJet: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    <A::RealJet as Jet>::Scalar: Float,
{
    let half = <<A::RealJet as Jet>::Scalar as One>::one()
        / (<<A::RealJet as Jet>::Scalar as One>::one()
            + <<A::RealJet as Jet>::Scalar as One>::one());

    let electric = constitutive
        .epsilon()
        .imaginary()
        .multiply(&vacuum_angular_wavenumber.real())
        .scale(half);

    let magnetic = constitutive
        .mu()
        .imaginary()
        .multiply(&vacuum_angular_wavenumber.real())
        .scale(half);

    ElectromagneticDissipationCoefficients::new(electric, magnetic)
}

#[cfg(test)]
mod tests {
    use ndarray::{Ix1, arr1};
    use num_complex::Complex64;

    use crate::field::VectorField;

    use super::*;

    type C = Complex64;
    type D = Ix1;
    type Vector = VectorField<C, D>;
    type Field = ConstitutiveFields<Vector>;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn scalar_vector(value: f64) -> Vector {
        VectorField::new_unchecked(
            arr1(&[c(value, 0.0)]),
            arr1(&[c(value + 0.1, 0.0)]),
            arr1(&[c(value + 0.2, 0.0)]),
        )
    }

    #[test]
    fn construction_preserves_electric_and_magnetic_fields() {
        let electric_displacement = scalar_vector(1.0);
        let magnetic_induction = scalar_vector(2.0);

        let field = Field::new(electric_displacement.clone(), magnetic_induction.clone());

        assert_eq!(field.electric_displacement(), &electric_displacement,);

        assert_eq!(field.magnetic_induction(), &magnetic_induction,);

        assert_eq!(
            field.into_parts(),
            (electric_displacement, magnetic_induction),
        );
    }
}

#[cfg(test)]
mod brillouin_tests {
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use crate::{
        algebra::{ArrayJet0, ArrayJet1, Jet0, Jet1, RealParameter},
        observable::field::constitutive::IsotropicConstitutiveSpectralFirst,
        test_support::{TOLERANCE, assertions::assert_complex_close},
    };

    use super::{
        IsotropicBrillouinFactors, IsotropicConstitutiveParameters,
        IsotropicConstitutiveSpectralData,
    };

    type C = Complex64;

    type J0 = ArrayJet0<C, Ix0, RealParameter>;
    type J1 = ArrayJet1<C, Ix0, RealParameter>;

    fn c(re: f64, im: f64) -> C {
        C::new(re, im)
    }

    fn jet0(value: C) -> J0 {
        Jet0::new(arr0(value))
    }

    fn jet1(value: C, first: C) -> J1 {
        Jet1::from_parts(arr0(value), arr0(first))
    }

    fn assert_jet0_close(actual: &J0, expected: C) {
        assert_complex_close(actual.value()[()], expected, TOLERANCE);
    }

    fn assert_jet1_close(actual: &J1, expected_value: C, expected_first: C) {
        assert_complex_close(actual.value()[()], expected_value, TOLERANCE);

        assert_complex_close(actual.first()[()], expected_first, TOLERANCE);
    }

    #[test]
    fn brillouin_factors_store_both_components() {
        let electric = jet0(c(2.0, 0.5));
        let magnetic = jet0(c(3.0, -0.25));

        let factors = IsotropicBrillouinFactors::new(electric.clone(), magnetic.clone());

        assert_eq!(factors.electric(), &electric,);

        assert_eq!(factors.magnetic(), &magnetic,);

        assert_eq!(factors.into_parts(), (electric, magnetic),);
    }

    #[test]
    fn brillouin_factors_equal_parameter_plus_k0_times_spectral_first() {
        let epsilon = c(2.1, 0.3);
        let mu = c(1.4, -0.2);

        let epsilon_first = c(0.25, -0.1);
        let mu_first = c(-0.15, 0.07);

        let k0 = c(2.7, 0.0);

        let data = IsotropicConstitutiveSpectralData::new(
            IsotropicConstitutiveParameters::new(jet0(epsilon), jet0(mu)),
            IsotropicConstitutiveSpectralFirst::new(jet0(epsilon_first), jet0(mu_first)),
            jet0(k0),
        );

        let factors = data.into_brillouin_factors();

        assert_jet0_close(factors.electric(), epsilon + k0 * epsilon_first);

        assert_jet0_close(factors.magnetic(), mu + k0 * mu_first);
    }

    #[test]
    fn nondispersive_brillouin_factors_reduce_to_epsilon_and_mu() {
        let epsilon = c(2.3, 0.0);
        let mu = c(1.2, 0.0);

        let data = IsotropicConstitutiveSpectralData::new(
            IsotropicConstitutiveParameters::new(jet0(epsilon), jet0(mu)),
            IsotropicConstitutiveSpectralFirst::new(jet0(C::ZERO), jet0(C::ZERO)),
            jet0(c(3.1, 0.0)),
        );

        let factors = data.into_brillouin_factors();

        assert_jet0_close(factors.electric(), epsilon);

        assert_jet0_close(factors.magnetic(), mu);
    }

    #[test]
    fn zero_k0_brillouin_factors_reduce_to_epsilon_and_mu() {
        let epsilon = c(2.3, 0.4);
        let mu = c(1.2, -0.3);

        let data = IsotropicConstitutiveSpectralData::new(
            IsotropicConstitutiveParameters::new(jet0(epsilon), jet0(mu)),
            IsotropicConstitutiveSpectralFirst::new(jet0(c(0.7, -0.2)), jet0(c(-0.4, 0.1))),
            jet0(C::ZERO),
        );

        let factors = data.into_brillouin_factors();

        assert_jet0_close(factors.electric(), epsilon);

        assert_jet0_close(factors.magnetic(), mu);
    }

    #[test]
    fn first_order_brillouin_factors_obey_product_rule() {
        /*
         * Cε = ε + k0 ε_k
         *
         * Cε' =
         *     ε'
         *   + k0' ε_k
         *   + k0 ε_k'
         *
         * and likewise for μ.
         */
        let epsilon = c(2.0, 0.3);
        let epsilon_outer_first = c(0.11, -0.04);

        let mu = c(1.3, -0.2);
        let mu_outer_first = c(-0.07, 0.06);

        let k0 = c(2.8, 0.0);
        let k0_outer_first = c(0.35, 0.0);

        let epsilon_spectral_first = c(0.24, -0.09);
        let epsilon_spectral_outer_first = c(-0.05, 0.03);

        let mu_spectral_first = c(-0.12, 0.08);
        let mu_spectral_outer_first = c(0.04, -0.02);

        let data = IsotropicConstitutiveSpectralData::new(
            IsotropicConstitutiveParameters::new(
                jet1(epsilon, epsilon_outer_first),
                jet1(mu, mu_outer_first),
            ),
            IsotropicConstitutiveSpectralFirst::new(
                jet1(epsilon_spectral_first, epsilon_spectral_outer_first),
                jet1(mu_spectral_first, mu_spectral_outer_first),
            ),
            jet1(k0, k0_outer_first),
        );

        let factors = data.into_brillouin_factors();

        let expected_electric_value = epsilon + k0 * epsilon_spectral_first;

        let expected_electric_first = epsilon_outer_first
            + k0_outer_first * epsilon_spectral_first
            + k0 * epsilon_spectral_outer_first;

        let expected_magnetic_value = mu + k0 * mu_spectral_first;

        let expected_magnetic_first =
            mu_outer_first + k0_outer_first * mu_spectral_first + k0 * mu_spectral_outer_first;

        assert_jet1_close(
            factors.electric(),
            expected_electric_value,
            expected_electric_first,
        );

        assert_jet1_close(
            factors.magnetic(),
            expected_magnetic_value,
            expected_magnetic_first,
        );
    }

    #[test]
    fn constant_outer_coordinate_still_propagates_spectral_data_derivative() {
        /*
         * This is the common thickness-derivative situation:
         *
         *     k0' = 0
         *
         * while ε_k and μ_k remain independent constitutive
         * spectral derivatives.
         */
        let epsilon = c(2.0, 0.0);
        let epsilon_outer_first = c(0.0, 0.0);

        let mu = c(1.1, 0.0);
        let mu_outer_first = c(0.0, 0.0);

        let k0 = c(2.5, 0.0);

        let epsilon_spectral_first = c(0.3, 0.0);
        let epsilon_spectral_outer_first = c(0.07, 0.0);

        let mu_spectral_first = c(0.15, 0.0);
        let mu_spectral_outer_first = c(-0.04, 0.0);

        let data = IsotropicConstitutiveSpectralData::new(
            IsotropicConstitutiveParameters::new(
                jet1(epsilon, epsilon_outer_first),
                jet1(mu, mu_outer_first),
            ),
            IsotropicConstitutiveSpectralFirst::new(
                jet1(epsilon_spectral_first, epsilon_spectral_outer_first),
                jet1(mu_spectral_first, mu_spectral_outer_first),
            ),
            jet1(k0, C::ZERO),
        );

        let factors = data.into_brillouin_factors();

        assert_jet1_close(
            factors.electric(),
            epsilon + k0 * epsilon_spectral_first,
            epsilon_outer_first + k0 * epsilon_spectral_outer_first,
        );

        assert_jet1_close(
            factors.magnetic(),
            mu + k0 * mu_spectral_first,
            mu_outer_first + k0 * mu_spectral_outer_first,
        );
    }

    #[test]
    fn varying_outer_spectral_coordinate_contributes_through_k0() {
        /*
         * Conversely, for a spectral outer derivative the derivative
         * of k0 itself contributes.
         */
        let epsilon = c(2.0, 0.0);
        let epsilon_outer_first = c(0.4, 0.0);

        let epsilon_spectral_first = c(0.4, 0.0);
        let epsilon_spectral_outer_first = c(-0.08, 0.0);

        let k0 = c(3.0, 0.0);
        let k0_outer_first = c(1.0, 0.0);

        let data = IsotropicConstitutiveSpectralData::new(
            IsotropicConstitutiveParameters::new(
                jet1(epsilon, epsilon_outer_first),
                jet1(c(1.0, 0.0), C::ZERO),
            ),
            IsotropicConstitutiveSpectralFirst::new(
                jet1(epsilon_spectral_first, epsilon_spectral_outer_first),
                jet1(C::ZERO, C::ZERO),
            ),
            jet1(k0, k0_outer_first),
        );

        let factors = data.into_brillouin_factors();

        let expected_first = epsilon_outer_first
            + k0_outer_first * epsilon_spectral_first
            + k0 * epsilon_spectral_outer_first;

        assert_complex_close(factors.electric().first()[()], expected_first, TOLERANCE);
    }
}
