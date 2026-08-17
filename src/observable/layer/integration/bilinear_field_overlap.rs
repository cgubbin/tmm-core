use crate::{
    Polarisation, algebra::ScalarAlgebra, backend::IsotropicLayerQuantities,
    observable::layer::integration::bilinear_state_products::IntegratedBilinearCrossStateProducts,
};

pub(crate) struct IntegratedBilinearFieldOverlap<A> {
    electric: A,
    magnetic: A,
}

impl<A> IntegratedBilinearFieldOverlap<A> {
    pub(crate) const fn new(electric: A, magnetic: A) -> Self {
        Self { electric, magnetic }
    }

    pub(crate) fn electric(&self) -> &A {
        &self.electric
    }

    pub(crate) fn magnetic(&self) -> &A {
        &self.magnetic
    }

    pub(crate) fn into_parts(self) -> (A, A) {
        (self.electric, self.magnetic)
    }
}

pub(crate) fn project_integrated_bilinear_field_overlap<A>(
    state: &IntegratedBilinearCrossStateProducts<A>,
    reference_quantities: &IsotropicLayerQuantities<A>,
    comparison_quantities: &IsotropicLayerQuantities<A>,
    reference_vacuum_angular_wavenumber: &A,
    comparison_vacuum_angular_wavenumber: &A,
    reference_parallel_angular_wavenumber: &A,
    comparison_parallel_angular_wavenumber: &A,
) -> IntegratedBilinearFieldOverlap<A>
where
    A: ScalarAlgebra,
{
    debug_assert_eq!(
        reference_quantities.polarisation(),
        comparison_quantities.polarisation(),
    );

    /*
     * Cross-transverse coefficient:
     *
     *     1 / (k0_reference k0_comparison)
     */
    let transverse = reference_vacuum_angular_wavenumber
        .multiply(comparison_vacuum_angular_wavenumber)
        .reciprocal();

    /*
     * Cross-longitudinal coefficient:
     *
     *     [beta_reference / (k0_reference factor_reference)]
     *     [beta_comparison / (k0_comparison factor_comparison)]
     */
    let reference_longitudinal = reference_parallel_angular_wavenumber
        .divide(&reference_vacuum_angular_wavenumber.multiply(reference_quantities.factor()));

    let comparison_longitudinal = comparison_parallel_angular_wavenumber
        .divide(&comparison_vacuum_angular_wavenumber.multiply(comparison_quantities.factor()));

    let longitudinal = reference_longitudinal.multiply(&comparison_longitudinal);

    let field = state.field_field();
    let secondary = state.secondary_secondary();

    /*
     * The transverse reconstructed Cartesian component contains ±i:
     *
     *     TE: Hx =  i secondary / k0
     *     TM: Ex = -i secondary / k0
     *
     * In either case the bilinear cross product contributes
     *
     *     i*i     = -1
     *     (-i)(-i) = -1.
     *
     * The longitudinal component contains no factor of i, so it retains
     * the positive sign.
     */
    let reconstructed_overlap = secondary
        .multiply(&transverse)
        .negate()
        .add(&field.multiply(&longitudinal));

    match reference_quantities.polarisation() {
        Polarisation::TransverseElectric => {
            /*
             * Ey Ey is represented directly by field_field.
             *
             * H · H =
             *     Hx Hx + Hz Hz
             *   = - secondary_secondary / (k0_r k0_c)
             *     + longitudinal * field_field.
             */
            IntegratedBilinearFieldOverlap::new(field.clone(), reconstructed_overlap)
        }

        Polarisation::TransverseMagnetic => {
            /*
             * Hy Hy is represented directly by field_field.
             *
             * E · E =
             *     Ex Ex + Ez Ez
             *   = - secondary_secondary / (k0_r k0_c)
             *     + longitudinal * field_field.
             */
            IntegratedBilinearFieldOverlap::new(reconstructed_overlap, field.clone())
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
        algebra::{ArrayJet0, Jet0, RealParameter},
        backend::IsotropicLayerQuantities,
    };

    type A = ArrayJet0<Complex64, Ix0, RealParameter>;

    const TOLERANCE: f64 = 1.0e-12;

    fn jet(value: Complex64) -> A {
        Jet0::new(arr0(value))
    }

    fn scalar(value: &A) -> Complex64 {
        value.value()[()]
    }

    fn quantities(
        epsilon: Complex64,
        mu: Complex64,
        kappa: Complex64,
        polarisation: Polarisation,
    ) -> IsotropicLayerQuantities<A> {
        IsotropicLayerQuantities::from_parts(jet(kappa), jet(epsilon), jet(mu), polarisation)
    }

    fn state(
        field_field: Complex64,
        secondary_secondary: Complex64,
    ) -> IntegratedBilinearCrossStateProducts<A> {
        IntegratedBilinearCrossStateProducts::new(
            jet(field_field),
            jet(secondary_secondary),
            jet(Complex64::new(0.0, 0.0)),
            jet(Complex64::new(0.0, 0.0)),
        )
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn te_projection_routes_field_overlap_to_electric_component() {
        let field_field = Complex64::new(2.0, 1.0);

        let projected = project_integrated_bilinear_field_overlap(
            &state(field_field, Complex64::new(3.0, -1.0)),
            &quantities(
                Complex64::new(2.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(1.0, 0.0),
                Polarisation::TransverseElectric,
            ),
            &quantities(
                Complex64::new(3.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(1.0, 0.0),
                Polarisation::TransverseElectric,
            ),
            &jet(Complex64::new(2.0, 0.0)),
            &jet(Complex64::new(3.0, 0.0)),
            &jet(Complex64::new(0.0, 0.0)),
            &jet(Complex64::new(0.0, 0.0)),
        );

        assert_complex_close(scalar(projected.electric()), field_field);
    }

    #[test]
    fn tm_projection_routes_field_overlap_to_magnetic_component() {
        let field_field = Complex64::new(2.0, 1.0);

        let projected = project_integrated_bilinear_field_overlap(
            &state(field_field, Complex64::new(3.0, -1.0)),
            &quantities(
                Complex64::new(2.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(1.0, 0.0),
                Polarisation::TransverseMagnetic,
            ),
            &quantities(
                Complex64::new(3.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(1.0, 0.0),
                Polarisation::TransverseMagnetic,
            ),
            &jet(Complex64::new(2.0, 0.0)),
            &jet(Complex64::new(3.0, 0.0)),
            &jet(Complex64::new(0.0, 0.0)),
            &jet(Complex64::new(0.0, 0.0)),
        );

        assert_complex_close(scalar(projected.magnetic()), field_field);
    }

    #[test]
    fn normal_incidence_leaves_negative_transverse_bilinear_contribution() {
        let field_field = Complex64::new(2.0, 1.0);
        let secondary_secondary = Complex64::new(3.0, -1.0);

        let projected = project_integrated_bilinear_field_overlap(
            &state(field_field, secondary_secondary),
            &quantities(
                Complex64::new(2.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(1.0, 0.0),
                Polarisation::TransverseElectric,
            ),
            &quantities(
                Complex64::new(3.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(1.0, 0.0),
                Polarisation::TransverseElectric,
            ),
            &jet(Complex64::new(2.0, 0.0)),
            &jet(Complex64::new(3.0, 0.0)),
            &jet(Complex64::new(0.0, 0.0)),
            &jet(Complex64::new(0.0, 0.0)),
        );

        /*
         * At normal incidence the longitudinal contribution vanishes.
         *
         * For TE,
         *
         *     Hx = i secondary / k0
         *
         * so the unconjugated bilinear product contributes
         *
         *     i * i = -1.
         */
        let expected_magnetic = -secondary_secondary / Complex64::new(6.0, 0.0);

        assert_complex_close(scalar(projected.magnetic()), expected_magnetic);
    }

    #[test]
    fn te_projection_uses_mu_in_longitudinal_coefficient() {
        let field_field = Complex64::new(2.0, 1.0);
        let secondary_secondary = Complex64::new(3.0, -1.0);

        let left_k0 = Complex64::new(2.0, 0.0);
        let right_k0 = Complex64::new(3.0, 0.0);

        let left_beta = Complex64::new(0.7, 0.0);
        let right_beta = Complex64::new(0.9, 0.0);

        let left_mu = Complex64::new(4.0, 0.0);
        let right_mu = Complex64::new(5.0, 0.0);

        let projected = project_integrated_bilinear_field_overlap(
            &state(field_field, secondary_secondary),
            &quantities(
                Complex64::new(20.0, 0.0),
                left_mu,
                Complex64::new(1.0, 0.0),
                Polarisation::TransverseElectric,
            ),
            &quantities(
                Complex64::new(30.0, 0.0),
                right_mu,
                Complex64::new(1.0, 0.0),
                Polarisation::TransverseElectric,
            ),
            &jet(left_k0),
            &jet(right_k0),
            &jet(left_beta),
            &jet(right_beta),
        );

        /*
         * TE magnetic overlap:
         *
         *     Hx_r Hx_c
         *       = - secondary_r secondary_c / (k0_r k0_c)
         *
         *     Hz_r Hz_c
         *       = field_r field_c
         *         [beta_r / (k0_r mu_r)]
         *         [beta_c / (k0_c mu_c)]
         */
        let transverse = Complex64::new(1.0, 0.0) / (left_k0 * right_k0);

        let longitudinal = left_beta / (left_k0 * left_mu) * right_beta / (right_k0 * right_mu);

        let expected = -secondary_secondary * transverse + field_field * longitudinal;

        assert_complex_close(scalar(projected.magnetic()), expected);
    }

    #[test]
    fn tm_projection_uses_epsilon_in_longitudinal_coefficient() {
        let field_field = Complex64::new(2.0, 1.0);
        let secondary_secondary = Complex64::new(3.0, -1.0);

        let left_k0 = Complex64::new(2.0, 0.0);
        let right_k0 = Complex64::new(3.0, 0.0);

        let left_beta = Complex64::new(0.7, 0.0);
        let right_beta = Complex64::new(0.9, 0.0);

        let left_epsilon = Complex64::new(4.0, 0.0);
        let right_epsilon = Complex64::new(5.0, 0.0);

        let projected = project_integrated_bilinear_field_overlap(
            &state(field_field, secondary_secondary),
            &quantities(
                left_epsilon,
                Complex64::new(20.0, 0.0),
                Complex64::new(1.0, 0.0),
                Polarisation::TransverseMagnetic,
            ),
            &quantities(
                right_epsilon,
                Complex64::new(30.0, 0.0),
                Complex64::new(1.0, 0.0),
                Polarisation::TransverseMagnetic,
            ),
            &jet(left_k0),
            &jet(right_k0),
            &jet(left_beta),
            &jet(right_beta),
        );

        /*
         * TM electric overlap:
         *
         *     Ex_r Ex_c
         *       = - secondary_r secondary_c / (k0_r k0_c)
         *
         *     Ez_r Ez_c
         *       = field_r field_c
         *         [beta_r / (k0_r epsilon_r)]
         *         [beta_c / (k0_c epsilon_c)]
         */
        let transverse = Complex64::new(1.0, 0.0) / (left_k0 * right_k0);

        let longitudinal =
            left_beta / (left_k0 * left_epsilon) * right_beta / (right_k0 * right_epsilon);

        let expected = -secondary_secondary * transverse + field_field * longitudinal;

        assert_complex_close(scalar(projected.electric()), expected);
    }

    #[test]
    fn exchanging_operands_preserves_complete_field_projection() {
        let left_quantities = quantities(
            Complex64::new(2.0, 0.3),
            Complex64::new(4.0, -0.2),
            Complex64::new(1.0, 0.1),
            Polarisation::TransverseElectric,
        );

        let right_quantities = quantities(
            Complex64::new(3.0, -0.1),
            Complex64::new(5.0, 0.4),
            Complex64::new(1.2, -0.2),
            Polarisation::TransverseElectric,
        );

        let state = state(Complex64::new(2.0, 1.0), Complex64::new(3.0, -1.0));

        let left_right = project_integrated_bilinear_field_overlap(
            &state,
            &left_quantities,
            &right_quantities,
            &jet(Complex64::new(2.0, 0.1)),
            &jet(Complex64::new(3.0, -0.2)),
            &jet(Complex64::new(0.7, 0.1)),
            &jet(Complex64::new(0.9, -0.1)),
        );

        let right_left = project_integrated_bilinear_field_overlap(
            &state,
            &right_quantities,
            &left_quantities,
            &jet(Complex64::new(3.0, -0.2)),
            &jet(Complex64::new(2.0, 0.1)),
            &jet(Complex64::new(0.9, -0.1)),
            &jet(Complex64::new(0.7, 0.1)),
        );

        assert_complex_close(scalar(left_right.electric()), scalar(right_left.electric()));

        assert_complex_close(scalar(left_right.magnetic()), scalar(right_left.magnetic()));
    }

    #[test]
    #[should_panic]
    fn mismatched_polarisations_trigger_debug_assertion() {
        let _ = project_integrated_bilinear_field_overlap(
            &state(Complex64::new(2.0, 0.0), Complex64::new(3.0, 0.0)),
            &quantities(
                Complex64::new(2.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(1.0, 0.0),
                Polarisation::TransverseElectric,
            ),
            &quantities(
                Complex64::new(3.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(1.0, 0.0),
                Polarisation::TransverseMagnetic,
            ),
            &jet(Complex64::new(2.0, 0.0)),
            &jet(Complex64::new(3.0, 0.0)),
            &jet(Complex64::new(0.0, 0.0)),
            &jet(Complex64::new(0.0, 0.0)),
        );
    }
}
