use num_traits::One;

use crate::{
    Polarisation,
    algebra::{Jet, RealScalarAlgebra, ScalarAlgebra},
    backend::IsotropicLayerQuantities,
};

use super::IntegratedStateProducts;

/// Spatially integrated squared electromagnetic field magnitudes.
///
/// Both entries retain the complex jet representation during intermediate
/// algebra, although Hermitian field norms are real mathematically.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IntegratedFieldNorms<A> {
    electric: A,
    magnetic: A,
}

impl<A> IntegratedFieldNorms<A> {
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

    pub(crate) fn map<B>(self, mut map: impl FnMut(A) -> B) -> IntegratedFieldNorms<B> {
        IntegratedFieldNorms {
            electric: map(self.electric),
            magnetic: map(self.magnetic),
        }
    }
}

pub(crate) fn project_integrated_field_norms<A>(
    state: &IntegratedStateProducts<A>,
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

// #[cfg(test)]
// mod tests {
//     use approx::assert_relative_eq;
//     use ndarray::{Ix0, arr0};
//     use num_complex::Complex64;

//     use super::*;

//     use crate::{
//         Polarisation,
//         algebra::{ArrayJet0, ArrayJet1, ArrayJetBivariate2, ComplexJet, Jet0, RealParameter},
//         backend::IsotropicLayerQuantities,
//         differential::{BivariateGradient, BivariateHessian},
//         observable::layer::IntegratedStateProducts,
//     };

//     type C = Complex64;
//     type A0 = ArrayJet0<C, Ix0, RealParameter>;
//     type R0 = <A0 as ComplexJet>::RealJet;

//     const TOLERANCE: f64 = 1.0e-12;

//     fn complex_jet(value: C) -> A0 {
//         Jet0::new(arr0(value))
//     }

//     fn real_complex_jet(value: f64) -> A0 {
//         complex_jet(C::new(value, 0.0))
//     }

//     fn real_jet(value: f64) -> R0 {
//         Jet0::new(arr0(value))
//     }

//     fn complex_scalar(value: &A0) -> C {
//         value.value()[()]
//     }

//     fn real_scalar(value: &R0) -> f64 {
//         value.value()[()]
//     }

//     fn assert_real_close(actual: f64, expected: f64) {
//         assert_relative_eq!(
//             actual,
//             expected,
//             epsilon = TOLERANCE,
//             max_relative = TOLERANCE,
//         );
//     }

//     fn state_products() -> IntegratedStateProducts<A0> {
//         IntegratedStateProducts::new(
//             complex_jet(C::new(2.0, 100.0)),
//             complex_jet(C::new(3.0, -200.0)),
//             complex_jet(C::new(4.0, 5.0)),
//             complex_jet(C::new(4.0, -5.0)),
//         )
//     }

//     fn weights() -> IsotropicFieldNormWeights<R0> {
//         IsotropicFieldNormWeights::new(real_jet(7.0), real_jet(11.0))
//     }

//     fn quantities(polarisation: Polarisation, epsilon: C, mu: C) -> IsotropicLayerQuantities<A0> {
//         let epsilon = complex_jet(epsilon);
//         let mu = complex_jet(mu);

//         let factor = match polarisation {
//             Polarisation::TransverseElectric => mu.clone(),
//             Polarisation::TransverseMagnetic => epsilon.clone(),
//         };

//         let kappa = complex_jet(C::new(3.0, 0.2));

//         IsotropicLayerQuantities::test_fixture(kappa, epsilon, mu, polarisation)
//     }

//     #[test]
//     fn integrated_field_norms_store_both_components() {
//         let norms = IntegratedFieldNorms::new(1, 2);

//         assert_eq!(norms.electric(), &1);
//         assert_eq!(norms.magnetic(), &2);
//     }

//     #[test]
//     fn integrated_field_norms_into_parts_preserves_order() {
//         let norms = IntegratedFieldNorms::new(1, 2);

//         assert_eq!(norms.into_parts(), (1, 2));
//     }

//     #[test]
//     fn integrated_field_norms_map_transforms_both_components() {
//         let norms = IntegratedFieldNorms::new(1, 2);

//         let mapped = norms.map(|value| value * 10);

//         assert_eq!(mapped.electric(), &10);
//         assert_eq!(mapped.magnetic(), &20);
//     }

//     #[test]
//     fn field_norm_weights_store_both_terms() {
//         let weights = IsotropicFieldNormWeights::new(1, 2);

//         assert_eq!(weights.transverse(), &1);
//         assert_eq!(weights.longitudinal(), &2);
//     }

//     #[test]
//     fn te_projection_uses_real_field_norm_as_electric_norm() {
//         let norms = project_te_integrated_field_norms(&state_products(), &weights());

//         assert_real_close(real_scalar(norms.electric()), 2.0);
//     }

//     #[test]
//     fn te_projection_builds_magnetic_norm_from_real_diagonal_products() {
//         let norms = project_te_integrated_field_norms(&state_products(), &weights());

//         /*
//          * magnetic
//          * = secondary_secondary * transverse
//          * + field_field * longitudinal
//          *
//          * = 3*7 + 2*11
//          * = 43
//          */
//         assert_real_close(real_scalar(norms.magnetic()), 43.0);
//     }

//     #[test]
//     fn tm_projection_uses_real_field_norm_as_magnetic_norm() {
//         let norms = project_tm_integrated_field_norms(&state_products(), &weights());

//         assert_real_close(real_scalar(norms.magnetic()), 2.0);
//     }

//     #[test]
//     fn tm_projection_builds_electric_norm_from_real_diagonal_products() {
//         let norms = project_tm_integrated_field_norms(&state_products(), &weights());

//         assert_real_close(real_scalar(norms.electric()), 43.0);
//     }

//     #[test]
//     fn te_projection_ignores_cross_products() {
//         let first = IntegratedStateProducts::new(
//             real_complex_jet(2.0),
//             real_complex_jet(3.0),
//             complex_jet(C::new(100.0, 200.0)),
//             complex_jet(C::new(-300.0, 400.0)),
//         );

//         let second = IntegratedStateProducts::new(
//             real_complex_jet(2.0),
//             real_complex_jet(3.0),
//             complex_jet(C::new(-7.0, 9.0)),
//             complex_jet(C::new(11.0, -13.0)),
//         );

//         let first = project_te_integrated_field_norms(&first, &weights());

//         let second = project_te_integrated_field_norms(&second, &weights());

//         assert_eq!(first, second);
//     }

//     #[test]
//     fn tm_projection_ignores_cross_products() {
//         let first = IntegratedStateProducts::new(
//             real_complex_jet(2.0),
//             real_complex_jet(3.0),
//             complex_jet(C::new(100.0, 200.0)),
//             complex_jet(C::new(-300.0, 400.0)),
//         );

//         let second = IntegratedStateProducts::new(
//             real_complex_jet(2.0),
//             real_complex_jet(3.0),
//             complex_jet(C::new(-7.0, 9.0)),
//             complex_jet(C::new(11.0, -13.0)),
//         );

//         let first = project_tm_integrated_field_norms(&first, &weights());

//         let second = project_tm_integrated_field_norms(&second, &weights());

//         assert_eq!(first, second);
//     }

//     #[test]
//     fn zero_te_weights_remove_reconstructed_magnetic_norm() {
//         let zero_weights = IsotropicFieldNormWeights::new(real_jet(0.0), real_jet(0.0));

//         let norms = project_te_integrated_field_norms(&state_products(), &zero_weights);

//         assert_real_close(real_scalar(norms.magnetic()), 0.0);

//         assert_real_close(real_scalar(norms.electric()), 2.0);
//     }

//     #[test]
//     fn zero_tm_weights_remove_reconstructed_electric_norm() {
//         let zero_weights = IsotropicFieldNormWeights::new(real_jet(0.0), real_jet(0.0));

//         let norms = project_tm_integrated_field_norms(&state_products(), &zero_weights);

//         assert_real_close(real_scalar(norms.electric()), 0.0);

//         assert_real_close(real_scalar(norms.magnetic()), 2.0);
//     }

//     #[test]
//     fn physical_weights_use_inverse_vacuum_wavenumber_squared() {
//         let quantities = quantities(
//             Polarisation::TransverseElectric,
//             C::new(2.0, 0.0),
//             C::new(3.0, 0.0),
//         );

//         let vacuum = complex_jet(C::new(2.0, 0.0));

//         let parallel = complex_jet(C::new(0.6, 0.0));

//         let weights = isotropic_field_norm_weights(&vacuum, &parallel, &quantities);

//         assert_real_close(real_scalar(weights.transverse()), 1.0 / 4.0);
//     }

//     #[test]
//     fn te_longitudinal_weight_uses_mu_as_factor() {
//         let quantities = quantities(
//             Polarisation::TransverseElectric,
//             C::new(2.0, 0.0),
//             C::new(3.0, 0.0),
//         );

//         let vacuum = complex_jet(C::new(2.0, 0.0));

//         let parallel = complex_jet(C::new(0.6, 0.0));

//         let weights = isotropic_field_norm_weights(&vacuum, &parallel, &quantities);

//         let expected = (0.6_f64 / (2.0 * 3.0)).powi(2);

//         assert_real_close(real_scalar(weights.longitudinal()), expected);
//     }

//     #[test]
//     fn tm_longitudinal_weight_uses_epsilon_as_factor() {
//         let quantities = quantities(
//             Polarisation::TransverseMagnetic,
//             C::new(2.0, 0.0),
//             C::new(3.0, 0.0),
//         );

//         let vacuum = complex_jet(C::new(2.0, 0.0));

//         let parallel = complex_jet(C::new(0.6, 0.0));

//         let weights = isotropic_field_norm_weights(&vacuum, &parallel, &quantities);

//         let expected = (0.6_f64 / (2.0 * 2.0)).powi(2);

//         assert_real_close(real_scalar(weights.longitudinal()), expected);
//     }

//     #[test]
//     fn physical_dispatch_uses_te_projection() {
//         let quantities = quantities(
//             Polarisation::TransverseElectric,
//             C::new(2.0, 0.0),
//             C::new(3.0, 0.0),
//         );

//         let vacuum = complex_jet(C::new(2.0, 0.0));

//         let parallel = complex_jet(C::new(0.6, 0.0));

//         let actual =
//             project_integrated_field_norms(&state_products(), &quantities, &vacuum, &parallel);

//         let weights = isotropic_field_norm_weights(&vacuum, &parallel, &quantities);

//         let expected = project_te_integrated_field_norms(&state_products(), &weights);

//         assert_eq!(actual, expected);
//     }

//     #[test]
//     fn physical_dispatch_uses_tm_projection() {
//         let quantities = quantities(
//             Polarisation::TransverseMagnetic,
//             C::new(2.0, 0.0),
//             C::new(3.0, 0.0),
//         );

//         let vacuum = complex_jet(C::new(2.0, 0.0));

//         let parallel = complex_jet(C::new(0.6, 0.0));

//         let actual =
//             project_integrated_field_norms(&state_products(), &quantities, &vacuum, &parallel);

//         let weights = isotropic_field_norm_weights(&vacuum, &parallel, &quantities);

//         let expected = project_tm_integrated_field_norms(&state_products(), &weights);

//         assert_eq!(actual, expected);
//     }

//     #[test]
//     fn te_projection_propagates_real_jet_derivatives() {
//         type A = ArrayJet1<C, Ix0, RealParameter>;

//         type R = <A as ComplexJet>::RealJet;

//         fn complex_first(value: C, first: C) -> A {
//             A::from_parts(arr0(value), arr0(first))
//         }

//         fn real_first(value: f64, first: f64) -> R {
//             R::from_parts(arr0(value), arr0(first))
//         }

//         let state = IntegratedStateProducts::new(
//             complex_first(C::new(2.0, 100.0), C::new(3.0, 101.0)),
//             complex_first(C::new(5.0, -200.0), C::new(7.0, -201.0)),
//             complex_first(C::new(0.0, 0.0), C::new(0.0, 0.0)),
//             complex_first(C::new(0.0, 0.0), C::new(0.0, 0.0)),
//         );

//         let weights =
//             IsotropicFieldNormWeights::new(real_first(11.0, 13.0), real_first(17.0, 19.0));

//         let norms = project_te_integrated_field_norms(&state, &weights);

//         assert_eq!(norms.electric().value()[()], 2.0,);

//         assert_eq!(norms.electric().first()[()], 3.0,);

//         /*
//          * H = s*t + f*l
//          *
//          * value = 5*11 + 2*17 = 89
//          *
//          * first
//          * = 7*11 + 5*13 + 3*17 + 2*19
//          * = 231
//          */
//         assert_eq!(norms.magnetic().value()[()], 89.0,);

//         assert_eq!(norms.magnetic().first()[()], 231.0,);
//     }

//     #[test]
//     fn tm_projection_propagates_mixed_derivatives() {
//         type A = ArrayJetBivariate2<C, Ix0, RealParameter>;

//         type R = <A as ComplexJet>::RealJet;

//         fn complex_bivariate(values: [f64; 6]) -> A {
//             A::from_parts(
//                 arr0(C::new(values[0], 0.0)),
//                 BivariateGradient::new(arr0(C::new(values[1], 0.0)), arr0(C::new(values[2], 0.0))),
//                 BivariateHessian::new(
//                     arr0(C::new(values[3], 0.0)),
//                     arr0(C::new(values[4], 0.0)),
//                     arr0(C::new(values[5], 0.0)),
//                 ),
//             )
//         }

//         fn real_bivariate(values: [f64; 6]) -> R {
//             R::from_parts(
//                 arr0(values[0]),
//                 BivariateGradient::new(arr0(values[1]), arr0(values[2])),
//                 BivariateHessian::new(arr0(values[3]), arr0(values[4]), arr0(values[5])),
//             )
//         }

//         let field = complex_bivariate([2.0, 3.0, 5.0, 7.0, 11.0, 13.0]);

//         let secondary = complex_bivariate([17.0, 19.0, 23.0, 29.0, 31.0, 37.0]);

//         let zero = complex_bivariate([0.0; 6]);

//         let state =
//             IntegratedStateProducts::new(field.clone(), secondary.clone(), zero.clone(), zero);

//         let transverse = real_bivariate([41.0, 43.0, 47.0, 53.0, 59.0, 61.0]);

//         let longitudinal = real_bivariate([67.0, 71.0, 73.0, 79.0, 83.0, 89.0]);

//         let weights = IsotropicFieldNormWeights::new(transverse, longitudinal);

//         let norms = project_tm_integrated_field_norms(&state, &weights);

//         let expected_mixed = 59.0 * 17.0
//             + 43.0 * 23.0
//             + 47.0 * 19.0
//             + 41.0 * 31.0
//             + 83.0 * 2.0
//             + 71.0 * 5.0
//             + 73.0 * 3.0
//             + 67.0 * 11.0;

//         assert_eq!(norms.electric().axis0_axis1()[()], expected_mixed,);

//         assert_eq!(norms.magnetic().axis0_axis1()[()], 11.0,);
//     }
// }
