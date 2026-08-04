//! Analytically integrated dissipation in homogeneous finite layers.

use num_traits::One;

use crate::{
    algebra::{Jet, RealScalarAlgebra, ScalarAlgebra},
    backend::IsotropicLayerQuantities,
};

use super::{Layers, integration::project_integrated_field_norms, project::IntegratedLayerData};

/// Normalized power dissipated within one finite layer.
///
/// The components are spatial integrals over the complete layer and are
/// normalized by the magnitude of the incident-wave power flux.
///
/// For isotropic media:
///
/// ```text
/// total = electric + magnetic
/// ```
///
/// Passive media should produce a non-negative `total`, subject to numerical
/// error. Active media may produce a negative value.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerDissipation<R> {
    electric: R,
    magnetic: R,
    total: R,
}

impl<R> LayerDissipation<R> {
    pub(crate) const fn new(electric: R, magnetic: R, total: R) -> Self {
        Self {
            electric,
            magnetic,
            total,
        }
    }

    /// Return the integrated electric dissipation.
    pub fn electric(&self) -> &R {
        &self.electric
    }

    /// Return the integrated magnetic dissipation.
    pub fn magnetic(&self) -> &R {
        &self.magnetic
    }

    /// Return the total integrated dissipation.
    pub fn total(&self) -> &R {
        &self.total
    }

    /// Consume the result and return `(electric, magnetic, total)`.
    pub fn into_parts(self) -> (R, R, R) {
        (self.electric, self.magnetic, self.total)
    }

    /// Transform every component.
    pub fn map<U>(self, mut map: impl FnMut(R) -> U) -> LayerDissipation<U> {
        LayerDissipation {
            electric: map(self.electric),
            magnetic: map(self.magnetic),
            total: map(self.total),
        }
    }
}

impl<A> IntegratedLayerData<A> {
    fn into_dissipation(
        self,
        vacuum_angular_wavenumber: &A,
        parallel_angular_wavenumber: &A,
        incident_flux_magnitude: &A::RealJet,
    ) -> LayerDissipation<A::RealJet>
    where
        A: RealScalarAlgebra,
        A::RealJet: ScalarAlgebra,
        <A::RealJet as Jet>::Scalar: One,
    {
        let (state, quantities) = self.into_parts();

        let field_norms = project_integrated_field_norms(
            &state,
            &quantities,
            vacuum_angular_wavenumber,
            parallel_angular_wavenumber,
        );

        let (electric_coeff, magnetic_coeff) = dissipation_coefficients(
            vacuum_angular_wavenumber,
            &quantities,
            incident_flux_magnitude,
        );

        let (electric_norm, magnetic_norm) = field_norms.into_parts();

        let electric = electric_norm.multiply(&electric_coeff);

        let magnetic = magnetic_norm.multiply(&magnetic_coeff);

        let total = electric.add(&magnetic);

        LayerDissipation::new(electric, magnetic, total)
    }
}

impl<A> Layers<IntegratedLayerData<A>> {
    pub(crate) fn into_dissipation(
        self,
        vacuum_angular_wavenumber: &A,
        parallel_angular_wavenumber: &A,
        incident_flux_magnitude: &A::RealJet,
    ) -> Layers<LayerDissipation<A::RealJet>>
    where
        A: RealScalarAlgebra,
        A::RealJet: ScalarAlgebra,
        <A::RealJet as Jet>::Scalar: One,
    {
        self.map(|each| {
            each.into_dissipation(
                vacuum_angular_wavenumber,
                parallel_angular_wavenumber,
                incident_flux_magnitude,
            )
        })
    }
}

/// Construct canonical normalized dissipation coefficients.
///
/// With:
///
/// ```text
/// flux = Im(field* secondary),
/// ```
///
/// the normalized integrated loss coefficients are:
///
/// ```text
/// electric = |k0|² Im(epsilon) / incident_flux
/// magnetic = |k0|² Im(mu)      / incident_flux.
/// ```
pub(crate) fn dissipation_coefficients<A>(
    vacuum_angular_wavenumber: &A,
    quantities: &IsotropicLayerQuantities<A>,
    incident_flux_magnitude: &A::RealJet,
) -> (A::RealJet, A::RealJet)
where
    A: RealScalarAlgebra,
    A::RealJet: ScalarAlgebra,
{
    let vacuum_squared = vacuum_angular_wavenumber.magnitude_squared();

    let electric = vacuum_squared
        .multiply(&quantities.epsilon().imaginary())
        .divide(incident_flux_magnitude);

    let magnetic = vacuum_squared
        .multiply(&quantities.mu().imaginary())
        .divide(incident_flux_magnitude);

    (electric, magnetic)
}

// #[cfg(test)]
// mod tests {
//     use super::LayerDissipation;

//     #[test]
//     fn stores_all_dissipation_components() {
//         let dissipation = LayerDissipation::new(1, 2, 3);

//         assert_eq!(dissipation.electric(), &1);
//         assert_eq!(dissipation.magnetic(), &2);
//         assert_eq!(dissipation.total(), &3);
//     }

//     #[test]
//     fn into_parts_preserves_component_order() {
//         let dissipation = LayerDissipation::new(1, 2, 3);

//         assert_eq!(dissipation.into_parts(), (1, 2, 3),);
//     }

//     #[test]
//     fn map_transforms_every_component() {
//         let dissipation = LayerDissipation::new(1, 2, 3);

//         let mapped = dissipation.map(|value| value * 10);

//         assert_eq!(mapped.electric(), &10);
//         assert_eq!(mapped.magnetic(), &20);
//         assert_eq!(mapped.total(), &30);
//     }

//     #[test]
//     fn map_supports_non_clone_storage() {
//         #[derive(Debug, PartialEq)]
//         struct NonClone(i32);

//         let dissipation = LayerDissipation::new(NonClone(1), NonClone(2), NonClone(3));

//         let mapped = dissipation.map(|value| value.0 * 10);

//         assert_eq!(mapped.electric(), &10);
//         assert_eq!(mapped.magnetic(), &20);
//         assert_eq!(mapped.total(), &30);
//     }
// }

// #[cfg(test)]
// mod projection_tests {
//     use approx::assert_relative_eq;
//     use ndarray::{Ix0, arr0};
//     use num_complex::Complex64;

//     use super::*;

//     use crate::{
//         Polarisation,
//         algebra::{ArrayJet0, ArrayJet1, ComplexJet, Jet0, RealParameter},
//         backend::IsotropicLayerQuantities,
//         observable::layer::IntegratedFieldNorms,
//     };

//     type C = Complex64;
//     type A0 = ArrayJet0<C, Ix0, RealParameter>;
//     type R0 = <A0 as ComplexJet>::RealJet;

//     const TOLERANCE: f64 = 1.0e-12;

//     fn complex_jet(value: C) -> A0 {
//         Jet0::new(arr0(value))
//     }

//     fn real_jet(value: f64) -> R0 {
//         Jet0::new(arr0(value))
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

//     fn quantities(epsilon: C, mu: C, polarisation: Polarisation) -> IsotropicLayerQuantities<A0> {
//         IsotropicLayerQuantities::test_fixture(
//             complex_jet(C::new(3.0, 0.2)),
//             complex_jet(epsilon),
//             complex_jet(mu),
//             polarisation,
//         )
//     }

//     #[test]
//     fn coefficients_store_electric_and_magnetic_terms() {
//         let coefficients = IsotropicDissipationCoefficients::new(1, 2);

//         assert_eq!(coefficients.electric(), &1);
//         assert_eq!(coefficients.magnetic(), &2);
//     }

//     #[test]
//     fn coefficients_into_parts_preserves_order() {
//         let coefficients = IsotropicDissipationCoefficients::new(1, 2);

//         assert_eq!(coefficients.into_parts(), (1, 2),);
//     }

//     #[test]
//     fn coefficients_map_transforms_both_components() {
//         let coefficients = IsotropicDissipationCoefficients::new(1, 2);

//         let mapped = coefficients.map(|value| value * 10);

//         assert_eq!(mapped.electric(), &10);
//         assert_eq!(mapped.magnetic(), &20);
//     }

//     #[test]
//     fn projection_applies_both_coefficients() {
//         let norms = IntegratedFieldNorms::new(real_jet(2.0), real_jet(3.0));

//         let coefficients = IsotropicDissipationCoefficients::new(real_jet(5.0), real_jet(7.0));

//         let dissipation = project_layer_dissipation(norms, &coefficients);

//         assert_real_close(real_scalar(dissipation.electric()), 10.0);

//         assert_real_close(real_scalar(dissipation.magnetic()), 21.0);

//         assert_real_close(real_scalar(dissipation.total()), 31.0);
//     }

//     #[test]
//     fn zero_electric_coefficient_removes_electric_term() {
//         let dissipation = project_layer_dissipation(
//             IntegratedFieldNorms::new(real_jet(2.0), real_jet(3.0)),
//             &IsotropicDissipationCoefficients::new(real_jet(0.0), real_jet(7.0)),
//         );

//         assert_real_close(real_scalar(dissipation.electric()), 0.0);

//         assert_real_close(real_scalar(dissipation.magnetic()), 21.0);

//         assert_real_close(real_scalar(dissipation.total()), 21.0);
//     }

//     #[test]
//     fn zero_magnetic_coefficient_removes_magnetic_term() {
//         let dissipation = project_layer_dissipation(
//             IntegratedFieldNorms::new(real_jet(2.0), real_jet(3.0)),
//             &IsotropicDissipationCoefficients::new(real_jet(5.0), real_jet(0.0)),
//         );

//         assert_real_close(real_scalar(dissipation.electric()), 10.0);

//         assert_real_close(real_scalar(dissipation.magnetic()), 0.0);

//         assert_real_close(real_scalar(dissipation.total()), 10.0);
//     }

//     #[test]
//     fn negative_coefficient_represents_gain() {
//         let dissipation = project_layer_dissipation(
//             IntegratedFieldNorms::new(real_jet(2.0), real_jet(3.0)),
//             &IsotropicDissipationCoefficients::new(real_jet(-5.0), real_jet(0.0)),
//         );

//         assert_real_close(real_scalar(dissipation.total()), -10.0);
//     }

//     #[test]
//     fn total_is_sum_of_components() {
//         let dissipation = project_layer_dissipation(
//             IntegratedFieldNorms::new(real_jet(2.5), real_jet(3.5)),
//             &IsotropicDissipationCoefficients::new(real_jet(4.0), real_jet(6.0)),
//         );

//         assert_real_close(
//             real_scalar(dissipation.total()),
//             real_scalar(dissipation.electric()) + real_scalar(dissipation.magnetic()),
//         );
//     }

//     #[test]
//     fn coefficient_constructor_uses_vacuum_wavenumber_squared() {
//         let quantities = quantities(
//             C::new(2.0, 0.5),
//             C::new(3.0, 0.25),
//             Polarisation::TransverseElectric,
//         );

//         let coefficients = isotropic_dissipation_coefficients(
//             &complex_jet(C::new(2.0, 0.0)),
//             &quantities,
//             &real_jet(5.0),
//         );

//         /*
//          * |k0|² = 4
//          *
//          * electric = 4*0.5/5 = 0.4
//          * magnetic = 4*0.25/5 = 0.2
//          */
//         assert_real_close(real_scalar(coefficients.electric()), 0.4);

//         assert_real_close(real_scalar(coefficients.magnetic()), 0.2);
//     }

//     #[test]
//     fn zero_material_loss_produces_zero_coefficients() {
//         let quantities = quantities(
//             C::new(2.0, 0.0),
//             C::new(3.0, 0.0),
//             Polarisation::TransverseMagnetic,
//         );

//         let coefficients = isotropic_dissipation_coefficients(
//             &complex_jet(C::new(2.0, 0.0)),
//             &quantities,
//             &real_jet(5.0),
//         );

//         assert_real_close(real_scalar(coefficients.electric()), 0.0);

//         assert_real_close(real_scalar(coefficients.magnetic()), 0.0);
//     }

//     #[test]
//     fn coefficient_constructor_preserves_gain_sign() {
//         let quantities = quantities(
//             C::new(2.0, -0.5),
//             C::new(3.0, 0.0),
//             Polarisation::TransverseElectric,
//         );

//         let coefficients = isotropic_dissipation_coefficients(
//             &complex_jet(C::new(2.0, 0.0)),
//             &quantities,
//             &real_jet(5.0),
//         );

//         assert_real_close(real_scalar(coefficients.electric()), -0.4);
//     }

//     #[test]
//     fn projection_propagates_first_derivatives() {
//         type A = ArrayJet1<C, Ix0, RealParameter>;

//         type R = <A as ComplexJet>::RealJet;

//         fn real_first(value: f64, first: f64) -> R {
//             R::from_parts(arr0(value), arr0(first))
//         }

//         let norms = IntegratedFieldNorms::new(real_first(2.0, 3.0), real_first(5.0, 7.0));

//         let coefficients =
//             IsotropicDissipationCoefficients::new(real_first(11.0, 13.0), real_first(17.0, 19.0));

//         let dissipation = project_layer_dissipation(norms, &coefficients);

//         assert_eq!(dissipation.electric().value()[()], 22.0,);

//         assert_eq!(dissipation.electric().first()[()], 3.0 * 11.0 + 2.0 * 13.0,);

//         assert_eq!(dissipation.magnetic().value()[()], 85.0,);

//         assert_eq!(dissipation.magnetic().first()[()], 7.0 * 17.0 + 5.0 * 19.0,);

//         assert_eq!(dissipation.total().value()[()], 107.0,);

//         assert_eq!(dissipation.total().first()[()], 273.0,);
//     }
// }
