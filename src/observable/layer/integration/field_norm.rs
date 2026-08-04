//! Projection of integrated canonical-state products into complete vector
//! electric- and magnetic-field norms.
//!
//! The canonical scalar field represents:
//!
//! - `E_y` for transverse-electric polarization;
//! - `H_y` for transverse-magnetic polarization.
//!
//! The remaining vector components are reconstructed from the canonical
//! secondary state, the parallel angular wavenumber, the vacuum angular
//! wavenumber, and the active constitutive factor.
//!
//! This module implements Hermitian real-input analysis. It is not used for
//! bilinear complex-modal overlaps.

use num_traits::One;

use crate::{
    Polarisation,
    algebra::{Jet, RealScalarAlgebra, ScalarAlgebra},
    backend::IsotropicLayerQuantities,
};

use super::IntegratedHermitianStateProducts;

/// Spatially integrated squared electromagnetic field magnitudes.
///
/// Both components use the real-jet representation because Hermitian field
/// norms are real-valued for real-input analysis.
///
/// ```text
/// electric = ∫ |E|² dz
/// magnetic = ∫ |H|² dz
/// ```
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IntegratedFieldNorms<R> {
    electric: R,
    magnetic: R,
}

impl<R> IntegratedFieldNorms<R> {
    pub(crate) const fn new(electric: R, magnetic: R) -> Self {
        Self { electric, magnetic }
    }

    pub(crate) fn electric(&self) -> &R {
        &self.electric
    }

    pub(crate) fn magnetic(&self) -> &R {
        &self.magnetic
    }

    pub(crate) fn into_parts(self) -> (R, R) {
        (self.electric, self.magnetic)
    }
}

/// Project integrated Hermitian canonical-state products into complete
/// electric and magnetic field norms.
///
/// Let:
///
/// ```text
/// k0     = vacuum angular wavenumber
/// beta   = parallel angular wavenumber
/// factor = mu for TE, epsilon for TM
/// ```
///
/// The common reconstruction weights are:
///
/// ```text
/// transverse   = 1 / |k0|²
/// longitudinal = |beta / (k0 factor)|².
/// ```
///
/// For TE:
///
/// ```text
/// electric = ∫ |field|² dz
/// magnetic = transverse ∫ |secondary|² dz
///          + longitudinal ∫ |field|² dz.
/// ```
///
/// For TM, the electric and magnetic roles are exchanged.
pub(crate) fn project_integrated_field_norms<A>(
    state: &IntegratedHermitianStateProducts<A>,
    quantities: &IsotropicLayerQuantities<A>,
    vacuum_angular_wavenumber: &A,
    parallel_angular_wavenumber: &A,
) -> IntegratedFieldNorms<A::RealJet>
where
    A: RealScalarAlgebra,
    A::RealJet: ScalarAlgebra,
    <A::RealJet as Jet>::Scalar: One,
{
    let vacuum_angular_wavenumber_squared = vacuum_angular_wavenumber.magnitude_squared();

    let one = A::RealJet::filled_constant_like(
        vacuum_angular_wavenumber_squared.value(),
        <A::RealJet as Jet>::Scalar::one(),
    );

    let transverse = one.divide(&vacuum_angular_wavenumber_squared);

    let longitudinal = parallel_angular_wavenumber
        .divide(&vacuum_angular_wavenumber.multiply(quantities.factor()))
        .magnitude_squared();

    let field = state.field_field().real();
    let secondary = state.secondary_secondary().real();

    match quantities.polarisation() {
        Polarisation::TransverseElectric => {
            let magnetic = secondary
                .multiply(&transverse)
                .add(&field.multiply(&longitudinal));

            IntegratedFieldNorms::new(field, magnetic)
        }
        Polarisation::TransverseMagnetic => {
            let electric = secondary
                .multiply(&transverse)
                .add(&field.multiply(&longitudinal));

            IntegratedFieldNorms::new(electric, field)
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        Polarisation,
        algebra::{ArrayJet0, ArrayJet1, ArrayJetBivariate2, ComplexJet, Jet0, RealParameter},
        backend::IsotropicLayerQuantities,
        differential::{BivariateGradient, BivariateHessian},
    };

    type C = Complex64;

    type A0 = ArrayJet0<C, Ix0, RealParameter>;
    type R0 = <A0 as ComplexJet>::RealJet;

    type A1 = ArrayJet1<C, Ix0, RealParameter>;
    type R1 = <A1 as ComplexJet>::RealJet;

    type AB2 = ArrayJetBivariate2<C, Ix0, RealParameter>;
    type RB2 = <AB2 as ComplexJet>::RealJet;

    const TOLERANCE: f64 = 1.0e-12;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn jet(value: C) -> A0 {
        Jet0::new(arr0(value))
    }

    fn real_jet(value: f64) -> A0 {
        jet(c(value, 0.0))
    }

    fn scalar(value: &R0) -> f64 {
        value.value()[()]
    }

    fn jet1(value: C, first: C) -> A1 {
        A1::from_parts(arr0(value), arr0(first))
    }

    fn constant_jet1(value: C) -> A1 {
        jet1(value, C::new(0.0, 0.0))
    }

    fn scalar1_value(value: &R1) -> f64 {
        value.value()[()]
    }

    fn scalar1_first(value: &R1) -> f64 {
        value.first()[()]
    }

    fn bivariate2(
        value: C,
        axis0: C,
        axis1: C,
        axis0_axis0: C,
        axis0_axis1: C,
        axis1_axis1: C,
    ) -> AB2 {
        AB2::from_parts(
            arr0(value),
            BivariateGradient::new(arr0(axis0), arr0(axis1)),
            BivariateHessian::new(arr0(axis0_axis0), arr0(axis0_axis1), arr0(axis1_axis1)),
        )
    }

    fn constant_bivariate2(value: C) -> AB2 {
        bivariate2(
            value,
            C::new(0.0, 0.0),
            C::new(0.0, 0.0),
            C::new(0.0, 0.0),
            C::new(0.0, 0.0),
            C::new(0.0, 0.0),
        )
    }

    fn scalar_b2_value(value: &RB2) -> f64 {
        value.value()[()]
    }

    fn scalar_b2_axis0(value: &RB2) -> f64 {
        value.axis0()[()]
    }

    fn scalar_b2_axis1(value: &RB2) -> f64 {
        value.axis1()[()]
    }

    fn scalar_b2_axis0_axis0(value: &RB2) -> f64 {
        value.axis0_axis0()[()]
    }

    fn scalar_b2_axis0_axis1(value: &RB2) -> f64 {
        value.axis0_axis1()[()]
    }

    fn scalar_b2_axis1_axis1(value: &RB2) -> f64 {
        value.axis1_axis1()[()]
    }

    fn state_products(
        field_field: C,
        secondary_secondary: C,
    ) -> IntegratedHermitianStateProducts<A0> {
        IntegratedHermitianStateProducts::new(
            jet(field_field),
            jet(secondary_secondary),
            jet(c(0.0, 0.0)),
            jet(c(0.0, 0.0)),
        )
    }

    fn quantities(polarisation: Polarisation, epsilon: C, mu: C) -> IsotropicLayerQuantities<A0> {
        IsotropicLayerQuantities::test_fixture(
            jet(c(3.0, 0.2)),
            jet(epsilon),
            jet(mu),
            polarisation,
        )
    }

    #[test]
    fn integrated_field_norms_preserve_component_order() {
        let norms = IntegratedFieldNorms::new(1, 2);

        assert_eq!(&norms.electric, &1);
        assert_eq!(&norms.magnetic, &2);
        assert_eq!(norms.into_parts(), (1, 2));
    }

    #[test]
    fn te_projection_uses_field_as_electric_norm() {
        /*
         * k0 = 2
         * beta = 0.6
         * mu = 3
         *
         * transverse:
         *   1 / |k0|² = 1/4
         *
         * longitudinal:
         *   |beta / (k0 mu)|²
         *   = |0.6 / 6|²
         *   = 0.01
         *
         * field_field = 5
         * secondary_secondary = 7
         *
         * electric = 5
         * magnetic = 7/4 + 5/100 = 1.8
         */
        let norms = project_integrated_field_norms(
            &state_products(c(5.0, 0.0), c(7.0, 0.0)),
            &quantities(Polarisation::TransverseElectric, c(2.0, 0.0), c(3.0, 0.0)),
            &real_jet(2.0),
            &real_jet(0.6),
        );

        assert_relative_eq!(
            scalar(&norms.electric),
            5.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(&norms.magnetic),
            1.8,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn tm_projection_uses_field_as_magnetic_norm() {
        /*
         * k0 = 2
         * beta = 0.6
         * epsilon = 2
         *
         * transverse:
         *   1/4
         *
         * longitudinal:
         *   |0.6 / (2*2)|²
         *   = 0.0225
         *
         * field_field = 5
         * secondary_secondary = 7
         *
         * electric = 7/4 + 5*0.0225 = 1.8625
         * magnetic = 5
         */
        let norms = project_integrated_field_norms(
            &state_products(c(5.0, 0.0), c(7.0, 0.0)),
            &quantities(Polarisation::TransverseMagnetic, c(2.0, 0.0), c(3.0, 0.0)),
            &real_jet(2.0),
            &real_jet(0.6),
        );

        assert_relative_eq!(
            scalar(&norms.electric),
            1.8625,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(&norms.magnetic),
            5.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn te_longitudinal_term_uses_mu_not_epsilon() {
        let state = state_products(c(5.0, 0.0), c(0.0, 0.0));

        let first = project_integrated_field_norms(
            &state,
            &quantities(Polarisation::TransverseElectric, c(2.0, 0.0), c(3.0, 0.0)),
            &real_jet(2.0),
            &real_jet(0.6),
        );

        let changed_epsilon = project_integrated_field_norms(
            &state,
            &quantities(Polarisation::TransverseElectric, c(20.0, 0.0), c(3.0, 0.0)),
            &real_jet(2.0),
            &real_jet(0.6),
        );

        let changed_mu = project_integrated_field_norms(
            &state,
            &quantities(Polarisation::TransverseElectric, c(2.0, 0.0), c(6.0, 0.0)),
            &real_jet(2.0),
            &real_jet(0.6),
        );

        assert_relative_eq!(
            scalar(first.magnetic()),
            scalar(changed_epsilon.magnetic()),
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert!(
            (scalar(first.magnetic()) - scalar(changed_mu.magnetic())).abs() > 1.0e-6,
            "changing mu must change the TE longitudinal contribution",
        );
    }

    #[test]
    fn tm_longitudinal_term_uses_epsilon_not_mu() {
        let state = state_products(c(5.0, 0.0), c(0.0, 0.0));

        let first = project_integrated_field_norms(
            &state,
            &quantities(Polarisation::TransverseMagnetic, c(2.0, 0.0), c(3.0, 0.0)),
            &real_jet(2.0),
            &real_jet(0.6),
        );

        let changed_mu = project_integrated_field_norms(
            &state,
            &quantities(Polarisation::TransverseMagnetic, c(2.0, 0.0), c(30.0, 0.0)),
            &real_jet(2.0),
            &real_jet(0.6),
        );

        let changed_epsilon = project_integrated_field_norms(
            &state,
            &quantities(Polarisation::TransverseMagnetic, c(4.0, 0.0), c(3.0, 0.0)),
            &real_jet(2.0),
            &real_jet(0.6),
        );

        assert_relative_eq!(
            scalar(first.electric()),
            scalar(changed_mu.electric()),
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert!(
            (scalar(first.electric()) - scalar(changed_epsilon.electric())).abs() > 1.0e-6,
            "changing epsilon must change the TM longitudinal contribution",
        );
    }

    #[test]
    fn normal_incidence_removes_longitudinal_component() {
        let state = state_products(c(5.0, 0.0), c(7.0, 0.0));

        let te = project_integrated_field_norms(
            &state,
            &quantities(Polarisation::TransverseElectric, c(2.0, 0.0), c(3.0, 0.0)),
            &real_jet(2.0),
            &real_jet(0.0),
        );

        let tm = project_integrated_field_norms(
            &state,
            &quantities(Polarisation::TransverseMagnetic, c(2.0, 0.0), c(3.0, 0.0)),
            &real_jet(2.0),
            &real_jet(0.0),
        );

        assert_relative_eq!(
            scalar(te.electric()),
            5.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(te.magnetic()),
            7.0 / 4.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(tm.electric()),
            7.0 / 4.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(tm.magnetic()),
            5.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn complex_factor_uses_hermitian_magnitude_squared() {
        let mu = c(3.0, 4.0);

        let norms = project_integrated_field_norms(
            &state_products(c(5.0, 0.0), c(0.0, 0.0)),
            &quantities(Polarisation::TransverseElectric, c(2.0, 0.0), mu),
            &real_jet(2.0),
            &real_jet(1.0),
        );

        /*
         * |beta / (k0 mu)|²
         *
         * = 1 / |2(3 + 4i)|²
         * = 1 / 100.
         */
        assert_relative_eq!(
            scalar(&norms.magnetic),
            5.0 / 100.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn projection_propagates_first_derivatives() {
        /*
         * Keep all weights constant and differentiate only the integrated
         * state products.
         *
         * TE weights:
         *   transverse   = 1/4
         *   longitudinal = 0.01
         *
         * field:
         *   value = 5
         *   first = 11
         *
         * secondary:
         *   value = 7
         *   first = 13
         *
         * electric first:
         *   11
         *
         * magnetic first:
         *   13/4 + 11/100 = 3.36
         */
        let state = IntegratedHermitianStateProducts::new(
            jet1(c(5.0, 0.0), c(11.0, 0.0)),
            jet1(c(7.0, 0.0), c(13.0, 0.0)),
            jet1(c(0.0, 0.0), c(0.0, 0.0)),
            jet1(c(0.0, 0.0), c(0.0, 0.0)),
        );

        let quantities = IsotropicLayerQuantities::test_fixture(
            constant_jet1(c(3.0, 0.2)),
            constant_jet1(c(2.0, 0.0)),
            constant_jet1(c(3.0, 0.0)),
            Polarisation::TransverseElectric,
        );

        let norms = project_integrated_field_norms(
            &state,
            &quantities,
            &constant_jet1(c(2.0, 0.0)),
            &constant_jet1(c(0.6, 0.0)),
        );

        assert_relative_eq!(
            scalar1_value(&norms.electric),
            5.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar1_first(&norms.electric),
            11.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar1_value(&norms.magnetic),
            1.8,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar1_first(&norms.magnetic),
            13.0 / 4.0 + 11.0 / 100.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn projection_propagates_bivariate_second_derivatives() {
        /*
         * Again keep the weights constant. Every derivative branch should
         * therefore be the same linear combination of the corresponding
         * field and secondary branches.
         */
        let field = bivariate2(
            c(5.0, 0.0),
            c(11.0, 0.0),
            c(13.0, 0.0),
            c(17.0, 0.0),
            c(19.0, 0.0),
            c(23.0, 0.0),
        );

        let secondary = bivariate2(
            c(7.0, 0.0),
            c(29.0, 0.0),
            c(31.0, 0.0),
            c(37.0, 0.0),
            c(41.0, 0.0),
            c(43.0, 0.0),
        );

        let state = IntegratedHermitianStateProducts::new(
            field,
            secondary,
            constant_bivariate2(c(0.0, 0.0)),
            constant_bivariate2(c(0.0, 0.0)),
        );

        let quantities = IsotropicLayerQuantities::test_fixture(
            constant_bivariate2(c(3.0, 0.2)),
            constant_bivariate2(c(2.0, 0.0)),
            constant_bivariate2(c(3.0, 0.0)),
            Polarisation::TransverseElectric,
        );

        let norms = project_integrated_field_norms(
            &state,
            &quantities,
            &constant_bivariate2(c(2.0, 0.0)),
            &constant_bivariate2(c(0.6, 0.0)),
        );

        let magnetic_branch = |field: f64, secondary: f64| secondary / 4.0 + field / 100.0;

        assert_relative_eq!(
            scalar_b2_value(&norms.electric),
            5.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar_b2_axis0(&norms.electric),
            11.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar_b2_axis1(&norms.electric),
            13.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar_b2_axis0_axis0(&norms.electric),
            17.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar_b2_axis0_axis1(&norms.electric),
            19.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar_b2_axis1_axis1(&norms.electric),
            23.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar_b2_value(&norms.magnetic),
            magnetic_branch(5.0, 7.0),
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar_b2_axis0(&norms.magnetic),
            magnetic_branch(11.0, 29.0),
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar_b2_axis1(&norms.magnetic),
            magnetic_branch(13.0, 31.0),
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar_b2_axis0_axis0(&norms.magnetic),
            magnetic_branch(17.0, 37.0),
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar_b2_axis0_axis1(&norms.magnetic),
            magnetic_branch(19.0, 41.0),
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar_b2_axis1_axis1(&norms.magnetic),
            magnetic_branch(23.0, 43.0),
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }
}
