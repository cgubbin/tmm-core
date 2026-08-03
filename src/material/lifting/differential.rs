use ndarray::Dimension;
use std::fmt::Debug;

use crate::{
    ComplexScalar, DerivativeOrder,
    algebra::{
        ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2,
        FirstOrderExpansion, ScalarAlgebra, SecondOrderExpansion,
    },
    material::lifting::ConstitutiveDerivativeEvaluator,
};

pub(crate) trait ConstitutiveSpectralFirstLift<E, M>: ScalarAlgebra
where
    Self::Scalar: ComplexScalar,
    Self::Dimension: Dimension,
    E: ConstitutiveDerivativeEvaluator<Self::Scalar, Self::Dimension, M>,
{
    /// Lift ∂ε/∂s, where s is the backend spectral variable.
    fn relative_permittivity_spectral_first(material: &M, spectral: &Self) -> Self;

    /// Lift ∂μ/∂s, where s is the backend spectral variable.
    fn relative_permeability_spectral_first(material: &M, spectral: &Self) -> Self;
}

impl<C, D, E, M, P> ConstitutiveSpectralFirstLift<E, M> for ArrayJet0<C, D, P>
where
    C: ComplexScalar + Copy,
    D: Dimension,
    E: ConstitutiveDerivativeEvaluator<C, D, M>,
    P: Clone + Debug,
{
    fn relative_permittivity_spectral_first(material: &M, spectral: &Self) -> Self {
        Self::new(E::relative_permittivity_derivative(
            material,
            spectral.value(),
            DerivativeOrder::First,
        ))
    }

    fn relative_permeability_spectral_first(material: &M, spectral: &Self) -> Self {
        Self::new(E::relative_permeability_derivative(
            material,
            spectral.value(),
            DerivativeOrder::First,
        ))
    }
}

impl<C, D, E, M, P> ConstitutiveSpectralFirstLift<E, M> for ArrayJet1<C, D, P>
where
    C: ComplexScalar + Copy,
    D: Dimension,
    E: ConstitutiveDerivativeEvaluator<C, D, M>,
    P: Clone + Debug,
{
    fn relative_permittivity_spectral_first(material: &M, spectral: &Self) -> Self {
        let value =
            E::relative_permittivity_derivative(material, spectral.value(), DerivativeOrder::First);

        let first = E::relative_permittivity_derivative(
            material,
            spectral.value(),
            DerivativeOrder::Second,
        );

        Self::compose_sampled_function(spectral, FirstOrderExpansion::new(value, first))
    }

    fn relative_permeability_spectral_first(material: &M, spectral: &Self) -> Self {
        let value =
            E::relative_permeability_derivative(material, spectral.value(), DerivativeOrder::First);

        let first = E::relative_permeability_derivative(
            material,
            spectral.value(),
            DerivativeOrder::Second,
        );

        Self::compose_sampled_function(spectral, FirstOrderExpansion::new(value, first))
    }
}

impl<C, D, E, M, P> ConstitutiveSpectralFirstLift<E, M> for ArrayJet2<C, D, P>
where
    C: ComplexScalar + Copy,
    D: Dimension,
    E: ConstitutiveDerivativeEvaluator<C, D, M>,
    P: Clone + Debug,
{
    fn relative_permittivity_spectral_first(material: &M, spectral: &Self) -> Self {
        let value =
            E::relative_permittivity_derivative(material, spectral.value(), DerivativeOrder::First);

        let first = E::relative_permittivity_derivative(
            material,
            spectral.value(),
            DerivativeOrder::Second,
        );

        let second =
            E::relative_permittivity_derivative(material, spectral.value(), DerivativeOrder::Third);

        Self::compose_sampled_function(spectral, SecondOrderExpansion::new(value, first, second))
    }

    fn relative_permeability_spectral_first(material: &M, spectral: &Self) -> Self {
        let value =
            E::relative_permeability_derivative(material, spectral.value(), DerivativeOrder::First);

        let first = E::relative_permeability_derivative(
            material,
            spectral.value(),
            DerivativeOrder::Second,
        );

        let second =
            E::relative_permeability_derivative(material, spectral.value(), DerivativeOrder::Third);

        Self::compose_sampled_function(spectral, SecondOrderExpansion::new(value, first, second))
    }
}

impl<C, D, E, M, P> ConstitutiveSpectralFirstLift<E, M> for ArrayJetBivariate1<C, D, P>
where
    C: ComplexScalar + Copy,
    D: Dimension,
    E: ConstitutiveDerivativeEvaluator<C, D, M>,
    P: Clone + Debug,
{
    fn relative_permittivity_spectral_first(material: &M, spectral: &Self) -> Self {
        let value =
            E::relative_permittivity_derivative(material, spectral.value(), DerivativeOrder::First);

        let first = E::relative_permittivity_derivative(
            material,
            spectral.value(),
            DerivativeOrder::Second,
        );

        Self::compose_sampled_function(spectral, FirstOrderExpansion::new(value, first))
    }

    fn relative_permeability_spectral_first(material: &M, spectral: &Self) -> Self {
        let value =
            E::relative_permeability_derivative(material, spectral.value(), DerivativeOrder::First);

        let first = E::relative_permeability_derivative(
            material,
            spectral.value(),
            DerivativeOrder::Second,
        );

        Self::compose_sampled_function(spectral, FirstOrderExpansion::new(value, first))
    }
}

impl<C, D, E, M, P> ConstitutiveSpectralFirstLift<E, M> for ArrayJetBivariate2<C, D, P>
where
    C: ComplexScalar + Copy,
    D: Dimension,
    E: ConstitutiveDerivativeEvaluator<C, D, M>,
    P: Clone + Debug,
{
    fn relative_permittivity_spectral_first(material: &M, spectral: &Self) -> Self {
        let value =
            E::relative_permittivity_derivative(material, spectral.value(), DerivativeOrder::First);

        let first = E::relative_permittivity_derivative(
            material,
            spectral.value(),
            DerivativeOrder::Second,
        );

        let second =
            E::relative_permittivity_derivative(material, spectral.value(), DerivativeOrder::Third);

        Self::compose_sampled_function(spectral, SecondOrderExpansion::new(value, first, second))
    }

    fn relative_permeability_spectral_first(material: &M, spectral: &Self) -> Self {
        let value =
            E::relative_permeability_derivative(material, spectral.value(), DerivativeOrder::First);

        let first = E::relative_permeability_derivative(
            material,
            spectral.value(),
            DerivativeOrder::Second,
        );

        let second =
            E::relative_permeability_derivative(material, spectral.value(), DerivativeOrder::Third);

        Self::compose_sampled_function(spectral, SecondOrderExpansion::new(value, first, second))
    }
}

#[cfg(test)]
mod spectral_first_lift_tests {
    use ndarray::{Array, Dimension, Ix0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        DifferentiableMaterial, Material, Sampled,
        algebra::{
            ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, RealParameter,
        },
        differential::{BivariateGradient, BivariateHessian},
        domain::RealAxis,
        material::{DerivativeOrder, EvaluateDifferentiableMaterial, EvaluateMaterial},
    };

    type C = Complex64;

    type J0 = ArrayJet0<C, Ix0, RealParameter>;
    type J1 = ArrayJet1<C, Ix0, RealParameter>;
    type J2 = ArrayJet2<C, Ix0, RealParameter>;
    type JB1 = ArrayJetBivariate1<C, Ix0, RealParameter>;
    type JB2 = ArrayJetBivariate2<C, Ix0, RealParameter>;

    #[derive(Clone, Copy, Debug)]
    struct PolynomialMaterial;

    /*
     * epsilon(k) = 2 + 3k + 5k² + 7k³
     *
     * epsilon'(k)   = 3 + 10k + 21k²
     * epsilon''(k)  = 10 + 42k
     * epsilon'''(k) = 42
     *
     * mu(k) = 11 + 13k + 17k² + 19k³
     *
     * mu'(k)   = 13 + 34k + 57k²
     * mu''(k)  = 34 + 114k
     * mu'''(k) = 114
     */

    impl Material for PolynomialMaterial {
        type Real = f64;

        fn relative_permittivity<I, X>(&self, vacuum_wavenumber: I) -> I::Mapped<X>
        where
            I: Sampled<Elem = Self::Real>,
            X: ComplexScalar<RealField = f64>,
        {
            vacuum_wavenumber.map(|k| X::from_real(2.0 + 3.0 * k + 5.0 * k * k + 7.0 * k * k * k))
        }

        fn relative_permeability<I, X>(&self, vacuum_wavenumber: I) -> I::Mapped<X>
        where
            I: Sampled<Elem = Self::Real>,
            X: ComplexScalar<RealField = f64>,
        {
            vacuum_wavenumber
                .map(|k| X::from_real(11.0 + 13.0 * k + 17.0 * k * k + 19.0 * k * k * k))
        }
    }

    impl DifferentiableMaterial for PolynomialMaterial {
        fn relative_permittivity_derivative<I, X>(
            &self,
            vacuum_wavenumber: I,
            order: DerivativeOrder,
        ) -> I::Mapped<X>
        where
            I: Sampled<Elem = Self::Real>,
            X: ComplexScalar<RealField = f64>,
        {
            match order {
                DerivativeOrder::First => {
                    vacuum_wavenumber.map(|k| X::from_real(3.0 + 10.0 * k + 21.0 * k * k))
                }

                DerivativeOrder::Second => vacuum_wavenumber.map(|k| X::from_real(10.0 + 42.0 * k)),

                DerivativeOrder::Third => vacuum_wavenumber.map(|_| X::from_real(42.0)),
            }
        }

        fn relative_permeability_derivative<I, X>(
            &self,
            vacuum_wavenumber: I,
            order: DerivativeOrder,
        ) -> I::Mapped<X>
        where
            I: Sampled<Elem = Self::Real>,
            X: ComplexScalar<RealField = f64>,
        {
            match order {
                DerivativeOrder::First => {
                    vacuum_wavenumber.map(|k| X::from_real(13.0 + 34.0 * k + 57.0 * k * k))
                }

                DerivativeOrder::Second => {
                    vacuum_wavenumber.map(|k| X::from_real(34.0 + 114.0 * k))
                }

                DerivativeOrder::Third => vacuum_wavenumber.map(|_| X::from_real(114.0)),
            }
        }
    }

    fn epsilon_first(k: f64) -> C {
        C::new(3.0 + 10.0 * k + 21.0 * k * k, 0.0)
    }

    fn epsilon_second(k: f64) -> C {
        C::new(10.0 + 42.0 * k, 0.0)
    }

    fn epsilon_third() -> C {
        C::new(42.0, 0.0)
    }

    fn mu_first(k: f64) -> C {
        C::new(13.0 + 34.0 * k + 57.0 * k * k, 0.0)
    }

    fn mu_second(k: f64) -> C {
        C::new(34.0 + 114.0 * k, 0.0)
    }

    fn mu_third() -> C {
        C::new(114.0, 0.0)
    }

    #[test]
    fn zero_order_lifts_intrinsic_first_derivatives() {
        let material = PolynomialMaterial;
        let k = 2.0;

        let spectral = J0::new(arr0(C::new(k, 0.0)));

        let epsilon = <J0 as ConstitutiveSpectralFirstLift<RealAxis,_>>::relative_permittivity_spectral_first(&material, &spectral);

        let mu = <J0 as ConstitutiveSpectralFirstLift<RealAxis,_>>::relative_permeability_spectral_first(&material, &spectral);

        assert_eq!(epsilon.value()[()], epsilon_first(k));
        assert_eq!(mu.value()[()], mu_first(k));
    }

    #[test]
    fn first_order_lift_composes_intrinsic_derivative_with_outer_direction() {
        let material = PolynomialMaterial;
        let k = 2.0;
        let k_first = 3.0;

        let spectral = J1::from_parts(arr0(C::new(k, 0.0)), arr0(C::new(k_first, 0.0)));

        let epsilon = <J1 as ConstitutiveSpectralFirstLift<RealAxis,_>>::relative_permittivity_spectral_first(&material, &spectral);
        let mu = <J1 as ConstitutiveSpectralFirstLift<RealAxis,_>>::relative_permeability_spectral_first(&material, &spectral);

        assert_eq!(epsilon.value()[()], epsilon_first(k));
        assert_eq!(epsilon.first()[()], epsilon_second(k) * k_first,);

        assert_eq!(mu.value()[()], mu_first(k));
        assert_eq!(mu.first()[()], mu_second(k) * k_first,);
    }

    #[test]
    fn second_order_lift_uses_third_material_derivative() {
        let material = PolynomialMaterial;

        let k = 2.0;
        let k_first = 3.0;
        let k_second = 5.0;

        let spectral = J2::from_parts(
            arr0(C::new(k, 0.0)),
            arr0(C::new(k_first, 0.0)),
            arr0(C::new(k_second, 0.0)),
        );

        let epsilon = <J2 as ConstitutiveSpectralFirstLift<RealAxis,_>>::relative_permittivity_spectral_first(&material, &spectral);
        let mu = <J2 as ConstitutiveSpectralFirstLift<RealAxis,_>>::relative_permeability_spectral_first(&material, &spectral);

        /*
         * For g(k) = epsilon'(k):
         *
         * d²g/dp² = epsilon'''(k) k_p² + epsilon''(k) k_pp.
         */
        let expected_epsilon_second =
            epsilon_third() * k_first * k_first + epsilon_second(k) * k_second;

        let expected_mu_second = mu_third() * k_first * k_first + mu_second(k) * k_second;

        assert_eq!(epsilon.value()[()], epsilon_first(k));
        assert_eq!(epsilon.first()[()], epsilon_second(k) * k_first,);
        assert_eq!(epsilon.second()[()], expected_epsilon_second,);

        assert_eq!(mu.value()[()], mu_first(k));
        assert_eq!(mu.first()[()], mu_second(k) * k_first,);
        assert_eq!(mu.second()[()], expected_mu_second,);
    }

    #[test]
    fn bivariate_first_lift_preserves_both_outer_axes() {
        let material = PolynomialMaterial;

        let k = 2.0;
        let axis0 = 3.0;
        let axis1 = 5.0;

        let spectral = JB1::from_parts(
            arr0(C::new(k, 0.0)),
            BivariateGradient::new(arr0(C::new(axis0, 0.0)), arr0(C::new(axis1, 0.0))),
        );

        let epsilon = <JB1 as ConstitutiveSpectralFirstLift<RealAxis,_>>::relative_permittivity_spectral_first(&material, &spectral);

        assert_eq!(epsilon.value()[()], epsilon_first(k));
        assert_eq!(epsilon.axis0()[()], epsilon_second(k) * axis0,);
        assert_eq!(epsilon.axis1()[()], epsilon_second(k) * axis1,);
    }

    #[test]
    fn bivariate_second_lift_uses_third_derivative_on_all_hessian_branches() {
        let material = PolynomialMaterial;

        let k = 2.0;

        let axis0 = 3.0;
        let axis1 = 5.0;

        let axis0_axis0 = 7.0;
        let axis0_axis1 = 11.0;
        let axis1_axis1 = 13.0;

        let spectral = JB2::from_parts(
            arr0(C::new(k, 0.0)),
            BivariateGradient::new(arr0(C::new(axis0, 0.0)), arr0(C::new(axis1, 0.0))),
            BivariateHessian::new(
                arr0(C::new(axis0_axis0, 0.0)),
                arr0(C::new(axis0_axis1, 0.0)),
                arr0(C::new(axis1_axis1, 0.0)),
            ),
        );

        let epsilon = <JB2 as ConstitutiveSpectralFirstLift<RealAxis,_>>::relative_permittivity_spectral_first(&material, &spectral);

        let expected_axis0_axis0 =
            epsilon_third() * axis0 * axis0 + epsilon_second(k) * axis0_axis0;

        let expected_axis0_axis1 =
            epsilon_third() * axis0 * axis1 + epsilon_second(k) * axis0_axis1;

        let expected_axis1_axis1 =
            epsilon_third() * axis1 * axis1 + epsilon_second(k) * axis1_axis1;

        assert_eq!(epsilon.value()[()], epsilon_first(k));

        assert_eq!(epsilon.axis0()[()], epsilon_second(k) * axis0,);

        assert_eq!(epsilon.axis1()[()], epsilon_second(k) * axis1,);

        assert_eq!(epsilon.axis0_axis0()[()], expected_axis0_axis0,);

        assert_eq!(epsilon.axis0_axis1()[()], expected_axis0_axis1,);

        assert_eq!(epsilon.axis1_axis1()[()], expected_axis1_axis1,);
    }
}
