//! Projection of canonical-state cross-products into complete Hermitian
//! electric- and magnetic-field overlaps.

use thiserror::Error;

use crate::{
    FiniteLayerIndex, Polarisation,
    algebra::{RealScalarAlgebra, ScalarAlgebra},
    backend::IsotropicLayerQuantities,
    observable::LayerAggregateError,
};

use super::hermitian_state_products::IntegratedHermitianCrossStateProducts;

/// Integrated Hermitian electric- and magnetic-field overlaps.
///
/// ```text
/// electric = ∫ E_reference* · E_comparison dz
/// magnetic = ∫ H_reference* · H_comparison dz
/// ```
///
/// The values are generally complex.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IntegratedHermitianFieldOverlap<A> {
    electric: A,
    magnetic: A,
}

impl<A> IntegratedHermitianFieldOverlap<A> {
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

/// Operand involved in a pairwise retained-state operation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PairOperand {
    Reference,
    Comparison,
}

impl std::fmt::Display for PairOperand {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reference => formatter.write_str("reference"),
            Self::Comparison => formatter.write_str("comparison"),
        }
    }
}

/// Failure to construct or evaluate a Hermitian pair of retained plane-wave
/// solutions.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum HermitianOverlapError {
    /// The compiled sampled array shapes differ.
    #[error(
        "reference sampled shape {reference:?} does not match comparison \
         sampled shape {comparison:?}"
    )]
    SampleShapeMismatch {
        reference: Vec<usize>,
        comparison: Vec<usize>,
    },

    /// The states were compiled with different polarizations.
    #[error(
        "Hermitian overlap requires matching polarizations; reference is \
         {reference:?}, comparison is {comparison:?}"
    )]
    PolarisationMismatch {
        reference: Polarisation,
        comparison: Polarisation,
    },

    /// The retained finite-layer counts differ.
    #[error(
        "reference finite-layer count {reference_count} does not match \
         comparison finite-layer count {comparison_count}"
    )]
    LayerCountMismatch {
        reference_count: usize,
        comparison_count: usize,
    },

    /// Corresponding finite layers do not occupy the same physical interval.
    #[error(
        "finite layer {index:?} has incompatible reference and comparison \
         thicknesses"
    )]
    LayerThicknessMismatch { index: FiniteLayerIndex },

    /// The two jet mappings do not assign the same meaning to derivative
    /// components.
    #[error("reference and comparison differential mappings are incompatible")]
    DifferentialMappingMismatch,

    /// A state does not retain the layer data required by pairwise
    /// observables.
    #[error("{operand} state does not retain finite-layer analysis data")]
    LayersNotRetained { operand: PairOperand },

    #[error("error in layer in aggregation {0}")]
    Aggregate(LayerAggregateError),
}

pub(crate) fn project_integrated_hermitian_field_overlap<A>(
    state: &IntegratedHermitianCrossStateProducts<A>,
    reference_quantities: &IsotropicLayerQuantities<A>,
    comparison_quantities: &IsotropicLayerQuantities<A>,
    reference_vacuum_angular_wavenumber: &A,
    comparison_vacuum_angular_wavenumber: &A,
    reference_parallel_angular_wavenumber: &A,
    comparison_parallel_angular_wavenumber: &A,
) -> Result<IntegratedHermitianFieldOverlap<A>, HermitianOverlapError>
where
    A: RealScalarAlgebra,
{
    let reference_polarisation = reference_quantities.polarisation();

    let comparison_polarisation = comparison_quantities.polarisation();

    if reference_polarisation != comparison_polarisation {
        return Err(HermitianOverlapError::PolarisationMismatch {
            reference: reference_polarisation,
            comparison: comparison_polarisation,
        });
    }

    /*
     * Cross-transverse coefficient:
     *
     * 1 / (k0_reference* k0_comparison)
     */
    let transverse = reference_vacuum_angular_wavenumber
        .conjugated()
        .multiply(comparison_vacuum_angular_wavenumber)
        .reciprocal();

    /*
     * Cross-longitudinal coefficient:
     *
     * [beta_reference / (k0_reference factor_reference)]*
     * [beta_comparison / (k0_comparison factor_comparison)]
     */
    let reference_longitudinal = reference_parallel_angular_wavenumber
        .divide(&reference_vacuum_angular_wavenumber.multiply(reference_quantities.factor()));

    let comparison_longitudinal = comparison_parallel_angular_wavenumber
        .divide(&comparison_vacuum_angular_wavenumber.multiply(comparison_quantities.factor()));

    let longitudinal = reference_longitudinal.hermitian_product(&comparison_longitudinal);

    let field = state.field_field();

    let secondary = state.secondary_secondary();

    let reconstructed = secondary
        .multiply(&transverse)
        .add(&field.multiply(&longitudinal));

    Ok(match reference_polarisation {
        Polarisation::TransverseElectric => {
            IntegratedHermitianFieldOverlap::new(field.clone(), reconstructed)
        }

        Polarisation::TransverseMagnetic => {
            IntegratedHermitianFieldOverlap::new(reconstructed, field.clone())
        }
    })
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        Polarisation,
        algebra::{ArrayJet0, ArrayJet1, Jet0, RealParameter},
        backend::IsotropicLayerQuantities,
    };

    type C = Complex64;

    type A0 = ArrayJet0<C, Ix0, RealParameter>;

    type A1 = ArrayJet1<C, Ix0, RealParameter>;

    const TOLERANCE: f64 = 1.0e-12;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn jet(value: C) -> A0 {
        Jet0::new(arr0(value))
    }

    fn scalar(value: &A0) -> C {
        value.value()[()]
    }

    fn jet1(value: C, first: C) -> A1 {
        A1::from_parts(arr0(value), arr0(first))
    }

    fn scalar1_value(value: &A1) -> C {
        value.value()[()]
    }

    fn scalar1_first(value: &A1) -> C {
        value.first()[()]
    }

    fn state_products(
        field_field: C,
        secondary_secondary: C,
    ) -> IntegratedHermitianCrossStateProducts<A0> {
        IntegratedHermitianCrossStateProducts::new(
            jet(field_field),
            jet(secondary_secondary),
            jet(c(0.0, 0.0)),
            jet(c(0.0, 0.0)),
        )
    }

    fn quantities(polarisation: Polarisation, epsilon: C, mu: C) -> IsotropicLayerQuantities<A0> {
        IsotropicLayerQuantities::test_fixture(
            jet(c(1.7, 0.2)),
            jet(epsilon),
            jet(mu),
            polarisation,
        )
    }

    fn assert_complex_relative_eq(actual: C, expected: C) {
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
    fn integrated_field_overlap_preserves_component_order() {
        let overlap = IntegratedHermitianFieldOverlap::new(1, 2);

        assert_eq!(overlap.electric(), &1);
        assert_eq!(overlap.magnetic(), &2);
        assert_eq!(overlap.into_parts(), (1, 2));
    }

    #[test]
    fn rejects_mismatched_polarisations() {
        let error = project_integrated_hermitian_field_overlap(
            &state_products(c(2.0, 0.3), c(5.0, -0.7)),
            &quantities(Polarisation::TransverseElectric, c(2.0, 0.0), c(3.0, 0.0)),
            &quantities(Polarisation::TransverseMagnetic, c(2.0, 0.0), c(3.0, 0.0)),
            &jet(c(2.0, 0.0)),
            &jet(c(2.0, 0.0)),
            &jet(c(0.4, 0.0)),
            &jet(c(0.4, 0.0)),
        )
        .expect_err("TE and TM fields must not be contracted together");

        assert_eq!(
            error,
            HermitianOverlapError::PolarisationMismatch {
                reference: Polarisation::TransverseElectric,
                comparison: Polarisation::TransverseMagnetic,
            },
        );
    }

    #[test]
    fn te_projection_uses_scalar_field_as_electric_overlap() {
        /*
         * Identical real coordinates:
         *
         * k0_reference = k0_comparison = 2
         * beta_reference = beta_comparison = 0.6
         * mu_reference = mu_comparison = 3
         *
         * transverse:
         *   1 / (2*2) = 1/4
         *
         * longitudinal:
         *   (0.6 / (2*3))* (0.6 / (2*3))
         *   = 0.01
         *
         * field overlap = 5 + 2i
         * secondary overlap = 7 - 3i
         *
         * electric = field
         * magnetic = secondary/4 + field/100
         */
        let field = c(5.0, 2.0);

        let secondary = c(7.0, -3.0);

        let overlap = project_integrated_hermitian_field_overlap(
            &state_products(field, secondary),
            &quantities(Polarisation::TransverseElectric, c(2.0, 0.0), c(3.0, 0.0)),
            &quantities(Polarisation::TransverseElectric, c(2.0, 0.0), c(3.0, 0.0)),
            &jet(c(2.0, 0.0)),
            &jet(c(2.0, 0.0)),
            &jet(c(0.6, 0.0)),
            &jet(c(0.6, 0.0)),
        )
        .unwrap();

        let expected_magnetic = secondary / 4.0 + field / 100.0;

        assert_complex_relative_eq(scalar(overlap.electric()), field);

        assert_complex_relative_eq(scalar(overlap.magnetic()), expected_magnetic);
    }

    #[test]
    fn tm_projection_uses_scalar_field_as_magnetic_overlap() {
        /*
         * epsilon_reference = epsilon_comparison = 2
         *
         * transverse = 1/4
         *
         * longitudinal:
         *   (0.6 / (2*2))² = 0.0225
         */
        let field = c(5.0, 2.0);

        let secondary = c(7.0, -3.0);

        let overlap = project_integrated_hermitian_field_overlap(
            &state_products(field, secondary),
            &quantities(Polarisation::TransverseMagnetic, c(2.0, 0.0), c(3.0, 0.0)),
            &quantities(Polarisation::TransverseMagnetic, c(2.0, 0.0), c(3.0, 0.0)),
            &jet(c(2.0, 0.0)),
            &jet(c(2.0, 0.0)),
            &jet(c(0.6, 0.0)),
            &jet(c(0.6, 0.0)),
        )
        .unwrap();

        let expected_electric = secondary / 4.0 + field * 0.0225;

        assert_complex_relative_eq(scalar(overlap.electric()), expected_electric);

        assert_complex_relative_eq(scalar(overlap.magnetic()), field);
    }

    #[test]
    fn cross_transverse_weight_conjugates_reference_wavenumber() {
        let reference_k0 = c(2.0, 0.5);

        let comparison_k0 = c(3.0, -0.2);

        let secondary = c(7.0, -3.0);

        let overlap = project_integrated_hermitian_field_overlap(
            &state_products(c(0.0, 0.0), secondary),
            &quantities(Polarisation::TransverseElectric, c(2.0, 0.0), c(3.0, 0.0)),
            &quantities(Polarisation::TransverseElectric, c(2.0, 0.0), c(3.0, 0.0)),
            &jet(reference_k0),
            &jet(comparison_k0),
            &jet(c(0.0, 0.0)),
            &jet(c(0.0, 0.0)),
        )
        .unwrap();

        let expected = secondary / (reference_k0.conj() * comparison_k0);

        assert_complex_relative_eq(scalar(overlap.magnetic()), expected);
    }

    #[test]
    fn longitudinal_weight_conjugates_complete_reference_factor() {
        let reference_k0 = c(2.0, 0.3);

        let comparison_k0 = c(2.5, -0.1);

        let reference_beta = c(0.6, 0.2);

        let comparison_beta = c(0.4, -0.3);

        let reference_mu = c(3.0, 0.7);

        let comparison_mu = c(2.0, -0.4);

        let field = c(5.0, -2.0);

        let overlap = project_integrated_hermitian_field_overlap(
            &state_products(field, c(0.0, 0.0)),
            &quantities(Polarisation::TransverseElectric, c(7.0, 0.0), reference_mu),
            &quantities(
                Polarisation::TransverseElectric,
                c(11.0, 0.0),
                comparison_mu,
            ),
            &jet(reference_k0),
            &jet(comparison_k0),
            &jet(reference_beta),
            &jet(comparison_beta),
        )
        .unwrap();

        let reference_longitudinal = reference_beta / (reference_k0 * reference_mu);

        let comparison_longitudinal = comparison_beta / (comparison_k0 * comparison_mu);

        let expected = field * reference_longitudinal.conj() * comparison_longitudinal;

        assert_complex_relative_eq(scalar(overlap.magnetic()), expected);
    }

    #[test]
    fn normal_incidence_removes_longitudinal_overlap() {
        let field = c(5.0, 2.0);

        let secondary = c(7.0, -3.0);

        let overlap = project_integrated_hermitian_field_overlap(
            &state_products(field, secondary),
            &quantities(Polarisation::TransverseElectric, c(2.0, 0.0), c(3.0, 5.0)),
            &quantities(Polarisation::TransverseElectric, c(2.0, 0.0), c(7.0, -4.0)),
            &jet(c(2.0, 0.0)),
            &jet(c(3.0, 0.0)),
            &jet(c(0.0, 0.0)),
            &jet(c(0.0, 0.0)),
        )
        .unwrap();

        assert_complex_relative_eq(scalar(overlap.electric()), field);

        assert_complex_relative_eq(scalar(overlap.magnetic()), secondary / 6.0);
    }

    #[test]
    fn identical_inputs_reduce_to_field_norm_projection() {
        let field = c(5.0, 0.0);

        let secondary = c(7.0, 0.0);

        let overlap = project_integrated_hermitian_field_overlap(
            &state_products(field, secondary),
            &quantities(Polarisation::TransverseElectric, c(2.0, 0.0), c(3.0, 0.0)),
            &quantities(Polarisation::TransverseElectric, c(2.0, 0.0), c(3.0, 0.0)),
            &jet(c(2.0, 0.0)),
            &jet(c(2.0, 0.0)),
            &jet(c(0.6, 0.0)),
            &jet(c(0.6, 0.0)),
        )
        .unwrap();

        assert_complex_relative_eq(scalar(overlap.electric()), c(5.0, 0.0));

        assert_complex_relative_eq(scalar(overlap.magnetic()), c(1.8, 0.0));
    }

    #[test]
    fn projection_propagates_first_derivatives() {
        /*
         * Keep all weights constant and differentiate only the state
         * products.
         *
         * TE:
         *
         * electric = field
         * magnetic = secondary/4 + field/100
         */
        let state = IntegratedHermitianCrossStateProducts::new(
            jet1(c(5.0, 2.0), c(11.0, -3.0)),
            jet1(c(7.0, -3.0), c(13.0, 5.0)),
            jet1(c(0.0, 0.0), c(0.0, 0.0)),
            jet1(c(0.0, 0.0), c(0.0, 0.0)),
        );

        let quantities = IsotropicLayerQuantities::test_fixture(
            jet1(c(1.7, 0.2), c(0.0, 0.0)),
            jet1(c(2.0, 0.0), c(0.0, 0.0)),
            jet1(c(3.0, 0.0), c(0.0, 0.0)),
            Polarisation::TransverseElectric,
        );

        let overlap = project_integrated_hermitian_field_overlap(
            &state,
            &quantities,
            &quantities,
            &jet1(c(2.0, 0.0), c(0.0, 0.0)),
            &jet1(c(2.0, 0.0), c(0.0, 0.0)),
            &jet1(c(0.6, 0.0), c(0.0, 0.0)),
            &jet1(c(0.6, 0.0), c(0.0, 0.0)),
        )
        .unwrap();

        let expected_electric_first = c(11.0, -3.0);

        let expected_magnetic_first = c(13.0, 5.0) / 4.0 + c(11.0, -3.0) / 100.0;

        assert_complex_relative_eq(scalar1_value(overlap.electric()), c(5.0, 2.0));

        assert_complex_relative_eq(scalar1_first(overlap.electric()), expected_electric_first);

        assert_complex_relative_eq(scalar1_first(overlap.magnetic()), expected_magnetic_first);
    }

    #[test]
    fn swapping_solutions_conjugates_field_overlap() {
        let reference_quantities =
            quantities(Polarisation::TransverseElectric, c(2.0, 0.0), c(3.0, 0.7));

        let comparison_quantities =
            quantities(Polarisation::TransverseElectric, c(5.0, 0.0), c(2.0, -0.4));

        let reference_comparison_state = state_products(c(5.0, 2.0), c(7.0, -3.0));

        let comparison_reference_state = state_products(c(5.0, -2.0), c(7.0, 3.0));

        let reference_comparison = project_integrated_hermitian_field_overlap(
            &reference_comparison_state,
            &reference_quantities,
            &comparison_quantities,
            &jet(c(2.0, 0.3)),
            &jet(c(2.5, -0.1)),
            &jet(c(0.6, 0.2)),
            &jet(c(0.4, -0.3)),
        )
        .unwrap();

        let comparison_reference = project_integrated_hermitian_field_overlap(
            &comparison_reference_state,
            &comparison_quantities,
            &reference_quantities,
            &jet(c(2.5, -0.1)),
            &jet(c(2.0, 0.3)),
            &jet(c(0.4, -0.3)),
            &jet(c(0.6, 0.2)),
        )
        .unwrap();

        assert_complex_relative_eq(
            scalar(reference_comparison.electric()),
            scalar(comparison_reference.electric()).conj(),
        );

        assert_complex_relative_eq(
            scalar(reference_comparison.magnetic()),
            scalar(comparison_reference.magnetic()).conj(),
        );
    }
}
