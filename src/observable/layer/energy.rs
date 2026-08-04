use ndarray::Dimension;
use num_traits::{FromPrimitive, One};
use thiserror::Error;

use crate::{
    ComplexScalar,
    algebra::{Jet, RealScalarAlgebra, ScalarAlgebra},
    backend::IsotropicLayerQuantities,
    material::{ConstitutiveSpectralFirstLift, lifting::ConstitutiveDerivativeEvaluator},
};

use super::{
    LayerProjectionError, Layers, integration::project_integrated_field_norms,
    project::IntegratedLayerData,
};

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum LayerEnergyError {
    #[error(transparent)]
    Projection(#[from] LayerProjectionError),

    #[error(
        "integrated layer count {layer_count} does not match differentiable \
     material count {material_count}"
    )]
    MaterialCountMismatch {
        layer_count: usize,
        material_count: usize,
    },
}

/// Integrated electromagnetic energy associated with one finite layer.
///
/// The electric and magnetic components are calculated using either:
///
/// - nondispersive constitutive weights, `Re(epsilon)` and `Re(mu)`; or
/// - Brillouin weights,
///   `Re[d(k0 epsilon)/d k0]` and `Re[d(k0 mu)/d k0]`.
///
/// Evaluator methods determine which definition is used. The default
/// differentiable-material path uses Brillouin energy, while the explicit
/// fallback uses nondispersive energy.
///
/// Results are normalized per unit incident-wave power flux.
///
/// ```text
/// total = electric + magnetic
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct LayerEnergy<R> {
    electric: R,
    magnetic: R,
    total: R,
}

impl<R> LayerEnergy<R> {
    pub(crate) const fn new(electric: R, magnetic: R, total: R) -> Self {
        Self {
            electric,
            magnetic,
            total,
        }
    }

    /// Return the integrated electric-energy contribution.
    pub fn electric(&self) -> &R {
        &self.electric
    }

    /// Return the integrated magnetic-energy contribution.
    pub fn magnetic(&self) -> &R {
        &self.magnetic
    }

    /// Return the total integrated layer energy.
    pub fn total(&self) -> &R {
        &self.total
    }

    /// Consume the value and return `(electric, magnetic, total)`.
    pub fn into_parts(self) -> (R, R, R) {
        (self.electric, self.magnetic, self.total)
    }

    /// Transform all energy components.
    pub fn map<U>(self, mut map: impl FnMut(R) -> U) -> LayerEnergy<U> {
        LayerEnergy {
            electric: map(self.electric),
            magnetic: map(self.magnetic),
            total: map(self.total),
        }
    }
}

/// Constitutive values required for Brillouin layer energy.
///
/// The spectral derivative fields are intrinsic material derivatives with
/// respect to the canonical vacuum angular wavenumber `k0`. They retain the
/// outer jet representation independently.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BrillouinConstitutiveDerivatives<A> {
    epsilon_spectral_first: A,
    mu_spectral_first: A,
}

impl<A> BrillouinConstitutiveDerivatives<A> {
    pub(crate) const fn new(epsilon_spectral_first: A, mu_spectral_first: A) -> Self {
        Self {
            epsilon_spectral_first,
            mu_spectral_first,
        }
    }

    pub(crate) fn epsilon_spectral_first(&self) -> &A {
        &self.epsilon_spectral_first
    }

    pub(crate) fn mu_spectral_first(&self) -> &A {
        &self.mu_spectral_first
    }

    pub(crate) fn into_parts(self) -> (A, A) {
        (self.epsilon_spectral_first, self.mu_spectral_first)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BrillouinLayerInput<A> {
    integrated: IntegratedLayerData<A>,
    derivative: BrillouinConstitutiveDerivatives<A>,
}

impl<A> BrillouinLayerInput<A> {
    pub(crate) const fn new(
        integrated: IntegratedLayerData<A>,
        derivative: BrillouinConstitutiveDerivatives<A>,
    ) -> Self {
        Self {
            integrated,
            derivative,
        }
    }

    pub(crate) fn integrated(&self) -> &IntegratedLayerData<A> {
        &self.integrated
    }

    pub(crate) fn derivative(&self) -> &BrillouinConstitutiveDerivatives<A> {
        &self.derivative
    }

    pub(crate) fn into_parts(
        self,
    ) -> (IntegratedLayerData<A>, BrillouinConstitutiveDerivatives<A>) {
        (self.integrated, self.derivative)
    }
}

impl<A> Layers<IntegratedLayerData<A>> {
    pub(crate) fn into_brillouin_layers<'a, E, M>(
        self,
        materials: impl ExactSizeIterator<Item = &'a M>,
        vacuum_angular_wavenumber: &A,
    ) -> Result<Layers<BrillouinLayerInput<A>>, LayerEnergyError>
    where
        M: 'a,
        A: ScalarAlgebra + ConstitutiveSpectralFirstLift<E, M>,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
        E: ConstitutiveDerivativeEvaluator<A::Scalar, A::Dimension, M>,
    {
        if self.len() != materials.len() {
            return Err(LayerEnergyError::MaterialCountMismatch {
                layer_count: self.len(),
                material_count: materials.len(),
            });
        }

        Ok(Layers::new(
            self.into_inner()
                .into_iter()
                .zip(materials)
                .map(|(layer, material)| {
                    let epsilon_spectral_first = A::relative_permittivity_spectral_first(
                        material,
                        vacuum_angular_wavenumber,
                    );

                    let mu_spectral_first = A::relative_permeability_spectral_first(
                        material,
                        vacuum_angular_wavenumber,
                    );

                    let material = BrillouinConstitutiveDerivatives::new(
                        epsilon_spectral_first,
                        mu_spectral_first,
                    );
                    BrillouinLayerInput::new(layer, material)
                })
                .collect(),
        ))
    }
}

/// Common normalization for energy per unit incident power flux.
///
/// With canonical flux:
///
/// ```text
/// F = Im(field* secondary),
/// ```
///
/// the time-averaged energy normalization is:
///
/// ```text
/// k0 / (2 F_incident).
/// ```
pub(crate) fn canonical_energy_normalization<A>(
    vacuum_angular_wavenumber: &A,
    incident_flux_magnitude: &A::RealJet,
) -> A::RealJet
where
    A: RealScalarAlgebra,
    A::RealJet: ScalarAlgebra,
    <A::RealJet as Jet>::Scalar: FromPrimitive,
{
    let vacuum = vacuum_angular_wavenumber.real();

    let half_scalar =
        <A::RealJet as Jet>::Scalar::from_f64(0.5).expect("one half must be representable");

    let half = A::RealJet::filled_constant_like(vacuum.value(), half_scalar);

    let common = vacuum.multiply(&half).divide(incident_flux_magnitude);

    common
}

impl<A> IntegratedLayerData<A> {
    fn into_nondispersive_energy(
        self,
        vacuum_angular_wavenumber: &A,
        parallel_angular_wavenumber: &A,
        normalization: &A::RealJet,
    ) -> LayerEnergy<A::RealJet>
    where
        A: RealScalarAlgebra,
        A::RealJet: ScalarAlgebra,
        <A::RealJet as Jet>::Scalar: One,
    {
        let (state_products, quantities) = self.into_parts();

        let (electric_norm, magnetic_norm) = project_integrated_field_norms(
            &state_products,
            &quantities,
            vacuum_angular_wavenumber,
            parallel_angular_wavenumber,
        )
        .into_parts();

        let electric_coefficient = quantities.epsilon().real().multiply(normalization);

        let magnetic_coefficient = quantities.mu().real().multiply(normalization);

        project_energy(
            electric_norm,
            magnetic_norm,
            electric_coefficient,
            magnetic_coefficient,
        )
    }
}

impl<A> Layers<IntegratedLayerData<A>> {
    pub(crate) fn into_nondispersive_energy(
        self,
        vacuum_angular_wavenumber: &A,
        parallel_angular_wavenumber: &A,
        incident_flux_magnitude: &A::RealJet,
    ) -> Layers<LayerEnergy<A::RealJet>>
    where
        A: RealScalarAlgebra,
        A::RealJet: ScalarAlgebra,
        <A::RealJet as Jet>::Scalar: FromPrimitive + One,
    {
        let normalization =
            canonical_energy_normalization(vacuum_angular_wavenumber, incident_flux_magnitude);

        self.map(|layer| {
            layer.into_nondispersive_energy(
                vacuum_angular_wavenumber,
                parallel_angular_wavenumber,
                &normalization,
            )
        })
    }
}

/// Construct Brillouin constitutive energy coefficients.
///
/// Ordinary constitutive values are taken from the retained isotropic layer
/// quantities. `data` supplies the additional intrinsic derivatives.
///
/// The constitutive weights are:
///
/// ```text
/// electric = Re[epsilon + k0 d epsilon/d k0]
/// magnetic = Re[mu      + k0 d mu/d k0]
/// ```
/// The derivative is intrinsic to the constitutive material model. It is
/// distinct from the outer derivative coordinates represented by `A`.
fn brillouin_energy_coefficients<A>(
    vacuum_angular_wavenumber: &A,
    quantities: &IsotropicLayerQuantities<A>,
    data: &BrillouinConstitutiveDerivatives<A>,
    normalization: &A::RealJet,
) -> (A::RealJet, A::RealJet)
where
    A: RealScalarAlgebra,
    A::RealJet: ScalarAlgebra,
{
    let electric_weight = quantities
        .epsilon()
        .add(&vacuum_angular_wavenumber.multiply(data.epsilon_spectral_first()))
        .real();

    let magnetic_weight = quantities
        .mu()
        .add(&vacuum_angular_wavenumber.multiply(data.mu_spectral_first()))
        .real();

    let electric = electric_weight.multiply(normalization);

    let magnetic = magnetic_weight.multiply(normalization);

    (electric, magnetic)
}

impl<A> BrillouinLayerInput<A> {
    fn into_brillouin_energy(
        self,
        vacuum_angular_wavenumber: &A,
        parallel_angular_wavenumber: &A,
        normalization: &A::RealJet,
    ) -> LayerEnergy<A::RealJet>
    where
        A: RealScalarAlgebra,
        A::RealJet: ScalarAlgebra,
        <A::RealJet as Jet>::Scalar: One,
    {
        let (layer, data) = self.into_parts();
        let (state_products, quantities) = layer.into_parts();

        let (electric_norm, magnetic_norm) = project_integrated_field_norms(
            &state_products,
            &quantities,
            vacuum_angular_wavenumber,
            parallel_angular_wavenumber,
        )
        .into_parts();

        let (electric_coefficient, magnetic_coefficient) = brillouin_energy_coefficients(
            vacuum_angular_wavenumber,
            &quantities,
            &data,
            normalization,
        );

        project_energy(
            electric_norm,
            magnetic_norm,
            electric_coefficient,
            magnetic_coefficient,
        )
    }
}

fn project_energy<R>(
    electric_norm: R,
    magnetic_norm: R,
    electric_coefficient: R,
    magnetic_coefficient: R,
) -> LayerEnergy<R>
where
    R: ScalarAlgebra,
{
    let electric = electric_norm.multiply(&electric_coefficient);

    let magnetic = magnetic_norm.multiply(&magnetic_coefficient);

    let total = electric.add(&magnetic);

    LayerEnergy::new(electric, magnetic, total)
}

impl<A> Layers<BrillouinLayerInput<A>> {
    pub(crate) fn into_brillouin_energy(
        self,
        vacuum_angular_wavenumber: &A,
        parallel_angular_wavenumber: &A,
        incident_flux_magnitude: &A::RealJet,
    ) -> Layers<LayerEnergy<A::RealJet>>
    where
        A: RealScalarAlgebra,
        A::RealJet: ScalarAlgebra,
        <A::RealJet as Jet>::Scalar: FromPrimitive + One,
    {
        let normalization =
            canonical_energy_normalization(vacuum_angular_wavenumber, incident_flux_magnitude);

        self.map(|each| {
            each.into_brillouin_energy(
                vacuum_angular_wavenumber,
                parallel_angular_wavenumber,
                &normalization,
            )
        })
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
        algebra::{ArrayJet0, ArrayJet1, ComplexJet, Jet0, RealParameter},
        backend::IsotropicLayerQuantities,
        material::DifferentiableMaterialHandle,
        observable::layer::{
            IntegratedHermitianStateProducts, Layers, project::IntegratedLayerData,
        },
    };

    type C = Complex64;

    type A0 = ArrayJet0<C, Ix0, RealParameter>;

    type R0 = <A0 as ComplexJet>::RealJet;

    type A1 = ArrayJet1<C, Ix0, RealParameter>;

    type R1 = <A1 as ComplexJet>::RealJet;

    const TOLERANCE: f64 = 1.0e-12;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn jet(value: C) -> A0 {
        Jet0::new(arr0(value))
    }

    fn real_jet(value: f64) -> R0 {
        Jet0::new(arr0(value))
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

    fn real_jet1(value: f64, first: f64) -> R1 {
        R1::from_parts(arr0(value), arr0(first))
    }

    fn scalar1_value(value: &R1) -> f64 {
        value.value()[()]
    }

    fn scalar1_first(value: &R1) -> f64 {
        value.first()[()]
    }

    fn state_products(
        field_field: f64,
        secondary_secondary: f64,
    ) -> IntegratedHermitianStateProducts<A0> {
        IntegratedHermitianStateProducts::new(
            jet(c(field_field, 0.0)),
            jet(c(secondary_secondary, 0.0)),
            jet(c(0.0, 0.0)),
            jet(c(0.0, 0.0)),
        )
    }

    fn quantities(polarisation: Polarisation, epsilon: C, mu: C) -> IsotropicLayerQuantities<A0> {
        /*
         * Constructor order:
         *
         * kappa, epsilon, mu, polarisation
         */
        IsotropicLayerQuantities::test_fixture(
            jet(c(3.0, 0.0)),
            jet(epsilon),
            jet(mu),
            polarisation,
        )
    }

    fn integrated_layer(
        polarisation: Polarisation,
        epsilon: C,
        mu: C,
        field_field: f64,
        secondary_secondary: f64,
    ) -> IntegratedLayerData<A0> {
        IntegratedLayerData::new(
            state_products(field_field, secondary_secondary),
            quantities(polarisation, epsilon, mu),
        )
    }

    fn derivatives(epsilon_first: C, mu_first: C) -> BrillouinConstitutiveDerivatives<A0> {
        BrillouinConstitutiveDerivatives::new(jet(epsilon_first), jet(mu_first))
    }

    fn brillouin_layer(
        polarisation: Polarisation,
        epsilon: C,
        mu: C,
        epsilon_first: C,
        mu_first: C,
        field_field: f64,
        secondary_secondary: f64,
    ) -> BrillouinLayerInput<A0> {
        BrillouinLayerInput::new(
            integrated_layer(polarisation, epsilon, mu, field_field, secondary_secondary),
            derivatives(epsilon_first, mu_first),
        )
    }

    #[test]
    fn layer_energy_preserves_component_order() {
        let energy = LayerEnergy::new(1, 2, 3);

        assert_eq!(energy.electric(), &1);
        assert_eq!(energy.magnetic(), &2);
        assert_eq!(energy.total(), &3);

        assert_eq!(energy.into_parts(), (1, 2, 3),);
    }

    #[test]
    fn layer_energy_map_transforms_all_components() {
        let energy = LayerEnergy::new(1, 2, 3);

        let mapped = energy.map(|value| value * 10);

        assert_eq!(mapped.electric(), &10);
        assert_eq!(mapped.magnetic(), &20);
        assert_eq!(mapped.total(), &30);
    }

    #[test]
    fn layer_energy_map_supports_non_clone_storage() {
        #[derive(Debug, PartialEq)]
        struct NonClone(i32);

        let energy = LayerEnergy::new(NonClone(1), NonClone(2), NonClone(3));

        let mapped = energy.map(|value| value.0 * 10);

        assert_eq!(mapped.electric(), &10);
        assert_eq!(mapped.magnetic(), &20);
        assert_eq!(mapped.total(), &30);
    }

    #[test]
    fn canonical_normalization_is_k0_over_twice_incident_flux() {
        let normalization = canonical_energy_normalization(&jet(c(4.0, 0.0)), &real_jet(5.0));

        assert_relative_eq!(
            scalar(&normalization),
            0.4,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn canonical_normalization_propagates_first_derivatives() {
        /*
         * N = k0 / (2F)
         *
         * k0 = 4 + 3p
         * F  = 5 + 7p
         *
         * N' = k0'/(2F) - k0 F'/(2F²)
         *    = 3/10 - 28/50
         *    = -0.26
         */
        let vacuum = jet1(c(4.0, 0.0), c(3.0, 0.0));

        let incident_flux = real_jet1(5.0, 7.0);

        let normalization = canonical_energy_normalization(&vacuum, &incident_flux);

        assert_relative_eq!(
            scalar1_value(&normalization),
            0.4,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar1_first(&normalization),
            -0.26,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn project_energy_applies_both_coefficients() {
        let energy = project_energy(real_jet(2.0), real_jet(3.0), real_jet(5.0), real_jet(7.0));

        assert_relative_eq!(
            scalar(energy.electric()),
            10.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(energy.magnetic()),
            21.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(energy.total()),
            31.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn project_energy_total_is_exact_component_sum() {
        let energy = project_energy(real_jet(2.5), real_jet(3.5), real_jet(4.0), real_jet(6.0));

        assert_relative_eq!(
            scalar(energy.total()),
            scalar(energy.electric()) + scalar(energy.magnetic()),
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn nondispersive_energy_uses_real_constitutive_weights() {
        let layer = integrated_layer(
            Polarisation::TransverseElectric,
            c(2.0, 100.0),
            c(3.0, -200.0),
            5.0,
            7.0,
        );

        let energy =
            layer.into_nondispersive_energy(&jet(c(2.0, 0.0)), &jet(c(0.6, 0.0)), &real_jet(1.0));

        let electric_norm = 5.0;

        let magnetic_norm = 7.0 / 4.0 + 5.0 * (0.6 * 0.6 / (4.0 * (3.0 * 3.0 + 200.0 * 200.0)));

        let expected_electric = electric_norm * 2.0;

        let expected_magnetic = magnetic_norm * 3.0;

        assert_relative_eq!(
            scalar(energy.electric()),
            expected_electric,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(energy.magnetic()),
            expected_magnetic,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(energy.total()),
            expected_electric + expected_magnetic,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn nondispersive_energy_coefficients_use_only_real_constitutive_parts() {
        let first = integrated_layer(
            Polarisation::TransverseElectric,
            c(2.0, 100.0),
            c(3.0, -200.0),
            5.0,
            7.0,
        )
        .into_nondispersive_energy(&jet(c(2.0, 0.0)), &jet(c(0.0, 0.0)), &real_jet(1.0));

        let second = integrated_layer(
            Polarisation::TransverseElectric,
            c(2.0, -17.0),
            c(3.0, 29.0),
            5.0,
            7.0,
        )
        .into_nondispersive_energy(&jet(c(2.0, 0.0)), &jet(c(0.0, 0.0)), &real_jet(1.0));

        assert_eq!(first, second);
    }

    #[test]
    fn nondispersive_sequence_preserves_count_and_order() {
        let layers = Layers::new(vec![
            integrated_layer(
                Polarisation::TransverseElectric,
                c(2.0, 0.0),
                c(3.0, 0.0),
                5.0,
                7.0,
            ),
            integrated_layer(
                Polarisation::TransverseElectric,
                c(4.0, 0.0),
                c(3.0, 0.0),
                5.0,
                7.0,
            ),
        ]);

        /*
         * k0 = 2, incident flux = 1:
         *
         * normalization = 2 / 2 = 1.
         */
        let energy =
            layers.into_nondispersive_energy(&jet(c(2.0, 0.0)), &jet(c(0.6, 0.0)), &real_jet(1.0));

        assert_eq!(energy.len(), 2);

        assert!(
            scalar(energy.first().unwrap().electric()) < scalar(energy.last().unwrap().electric()),
            "larger epsilon must remain associated with the second layer",
        );
    }

    #[test]
    fn empty_nondispersive_sequence_remains_empty() {
        let layers: Layers<IntegratedLayerData<A0>> = Layers::new(Vec::new());

        let energy =
            layers.into_nondispersive_energy(&jet(c(2.0, 0.0)), &jet(c(0.6, 0.0)), &real_jet(1.0));

        assert!(energy.is_empty());
    }

    #[test]
    fn brillouin_derivatives_preserve_component_order() {
        let derivatives = BrillouinConstitutiveDerivatives::new(1, 2);

        assert_eq!(derivatives.epsilon_spectral_first(), &1,);

        assert_eq!(derivatives.mu_spectral_first(), &2,);

        assert_eq!(derivatives.into_parts(), (1, 2),);
    }

    #[test]
    fn brillouin_input_preserves_layer_and_derivative_data() {
        let integrated = integrated_layer(
            Polarisation::TransverseElectric,
            c(2.0, 0.0),
            c(3.0, 0.0),
            5.0,
            7.0,
        );

        let derivatives = derivatives(c(11.0, 0.0), c(13.0, 0.0));

        let input = BrillouinLayerInput::new(integrated, derivatives);

        assert_eq!(
            input.derivative().epsilon_spectral_first().value()[()],
            c(11.0, 0.0),
        );

        assert_eq!(
            input.derivative().mu_spectral_first().value()[()],
            c(13.0, 0.0),
        );
    }

    #[test]
    fn brillouin_coefficients_use_derivative_of_k0_times_constitutive_value() {
        let quantities = quantities(
            Polarisation::TransverseElectric,
            c(2.0, 100.0),
            c(3.0, -200.0),
        );

        let derivatives = derivatives(c(5.0, 101.0), c(7.0, -201.0));

        let (electric, magnetic) = brillouin_energy_coefficients(
            &jet(c(11.0, 0.0)),
            &quantities,
            &derivatives,
            &real_jet(13.0),
        );

        /*
         * electric:
         *   Re[2 + 11*5] * 13 = 741
         *
         * magnetic:
         *   Re[3 + 11*7] * 13 = 1040
         */
        assert_relative_eq!(
            scalar(&electric),
            741.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(&magnetic),
            1040.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn brillouin_coefficients_take_real_part_after_full_product() {
        let quantities = quantities(Polarisation::TransverseElectric, c(2.0, 3.0), c(5.0, 7.0));

        let derivatives = derivatives(c(11.0, 13.0), c(17.0, 19.0));

        let (electric, magnetic) = brillouin_energy_coefficients(
            &jet(c(23.0, 0.0)),
            &quantities,
            &derivatives,
            &real_jet(1.0),
        );

        assert_relative_eq!(
            scalar(&electric),
            2.0 + 23.0 * 11.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(&magnetic),
            5.0 + 23.0 * 17.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn zero_intrinsic_derivatives_reduce_to_nondispersive_coefficients() {
        let quantities = quantities(
            Polarisation::TransverseElectric,
            c(2.0, 100.0),
            c(3.0, -200.0),
        );

        let derivatives = derivatives(c(0.0, 0.0), c(0.0, 0.0));

        let (electric, magnetic) = brillouin_energy_coefficients(
            &jet(c(11.0, 0.0)),
            &quantities,
            &derivatives,
            &real_jet(5.0),
        );

        assert_relative_eq!(
            scalar(&electric),
            10.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(&magnetic),
            15.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn brillouin_layer_projects_total_as_component_sum() {
        let input = brillouin_layer(
            Polarisation::TransverseElectric,
            c(2.0, 0.0),
            c(3.0, 0.0),
            c(5.0, 0.0),
            c(7.0, 0.0),
            5.0,
            7.0,
        );

        let energy =
            input.into_brillouin_energy(&jet(c(2.0, 0.0)), &jet(c(0.6, 0.0)), &real_jet(1.0));

        assert_relative_eq!(
            scalar(energy.total()),
            scalar(energy.electric()) + scalar(energy.magnetic()),
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn zero_intrinsic_derivatives_match_nondispersive_energy() {
        let nondispersive = integrated_layer(
            Polarisation::TransverseMagnetic,
            c(2.0, 0.0),
            c(3.0, 0.0),
            5.0,
            7.0,
        )
        .into_nondispersive_energy(&jet(c(2.0, 0.0)), &jet(c(0.6, 0.0)), &real_jet(1.0));

        let brillouin = brillouin_layer(
            Polarisation::TransverseMagnetic,
            c(2.0, 0.0),
            c(3.0, 0.0),
            c(0.0, 0.0),
            c(0.0, 0.0),
            5.0,
            7.0,
        )
        .into_brillouin_energy(&jet(c(2.0, 0.0)), &jet(c(0.6, 0.0)), &real_jet(1.0));

        assert_eq!(brillouin, nondispersive);
    }

    #[test]
    fn brillouin_sequence_preserves_count_and_order() {
        let layers = Layers::new(vec![
            brillouin_layer(
                Polarisation::TransverseElectric,
                c(2.0, 0.0),
                c(3.0, 0.0),
                c(1.0, 0.0),
                c(0.0, 0.0),
                5.0,
                7.0,
            ),
            brillouin_layer(
                Polarisation::TransverseElectric,
                c(2.0, 0.0),
                c(3.0, 0.0),
                c(4.0, 0.0),
                c(0.0, 0.0),
                5.0,
                7.0,
            ),
        ]);

        /*
         * k0 = 2, incident flux = 1:
         *
         * normalization = 1.
         */
        let energy =
            layers.into_brillouin_energy(&jet(c(2.0, 0.0)), &jet(c(0.6, 0.0)), &real_jet(1.0));

        assert_eq!(energy.len(), 2);

        assert!(
            scalar(energy.first().unwrap().electric()) < scalar(energy.last().unwrap().electric()),
            "larger epsilon derivative must remain in the second layer",
        );
    }

    #[test]
    fn empty_brillouin_sequence_remains_empty() {
        let layers: Layers<BrillouinLayerInput<A0>> = Layers::new(Vec::new());

        let energy =
            layers.into_brillouin_energy(&jet(c(2.0, 0.0)), &jet(c(0.6, 0.0)), &real_jet(1.0));

        assert!(energy.is_empty());
    }

    #[test]
    fn brillouin_coefficients_propagate_first_derivatives() {
        /*
         * k0 = 2 + 0.5p
         *
         * epsilon = 3 + 0.7p
         * epsilon' = 5 + 1.1p
         *
         * W = epsilon + k0 epsilon'
         *
         * W(0)  = 3 + 2*5 = 13
         * W'(0) = 0.7 + 0.5*5 + 2*1.1 = 5.4
         *
         * The supplied normalization is constant and equal to 2.
         *
         * coefficient:
         *   value = 26
         *   first = 10.8
         */
        let quantities = IsotropicLayerQuantities::test_fixture(
            constant_jet1(c(3.0, 0.0)),
            jet1(c(3.0, 0.0), c(0.7, 0.0)),
            constant_jet1(c(4.0, 0.0)),
            Polarisation::TransverseElectric,
        );

        let derivatives = BrillouinConstitutiveDerivatives::new(
            jet1(c(5.0, 0.0), c(1.1, 0.0)),
            constant_jet1(c(0.0, 0.0)),
        );

        let (electric, _magnetic) = brillouin_energy_coefficients(
            &jet1(c(2.0, 0.0), c(0.5, 0.0)),
            &quantities,
            &derivatives,
            &real_jet1(2.0, 0.0),
        );

        assert_relative_eq!(
            scalar1_value(&electric),
            26.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar1_first(&electric),
            10.8,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    use crate::material::{DerivativeOrder, DifferentiableMaterial, Material, Sampled};

    #[derive(Clone, Copy, Debug)]
    struct LinearDifferentiableMaterial {
        epsilon_slope: f64,
        mu_slope: f64,
    }

    impl Material for LinearDifferentiableMaterial {
        type Real = f64;

        fn relative_permittivity<I, X>(&self, vacuum_wavenumber: I) -> I::Mapped<X>
        where
            I: Sampled<Elem = Self::Real>,
            X: ComplexScalar<RealField = f64>,
        {
            vacuum_wavenumber.map(|k| X::from_real(1.0 + self.epsilon_slope * k))
        }

        fn relative_permeability<I, X>(&self, vacuum_wavenumber: I) -> I::Mapped<X>
        where
            I: Sampled<Elem = Self::Real>,
            X: ComplexScalar<RealField = f64>,
        {
            vacuum_wavenumber.map(|k| X::from_real(1.0 + self.mu_slope * k))
        }
    }

    impl DifferentiableMaterial for LinearDifferentiableMaterial {
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
                    vacuum_wavenumber.map(|_| X::from_real(self.epsilon_slope))
                }

                DerivativeOrder::Second | DerivativeOrder::Third => {
                    vacuum_wavenumber.map(|_| X::from_real(0.0))
                }
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
                DerivativeOrder::First => vacuum_wavenumber.map(|_| X::from_real(self.mu_slope)),

                DerivativeOrder::Second | DerivativeOrder::Third => {
                    vacuum_wavenumber.map(|_| X::from_real(0.0))
                }
            }
        }
    }

    fn material_handle(epsilon_slope: f64, mu_slope: f64) -> DifferentiableMaterialHandle<C> {
        DifferentiableMaterialHandle::new(LinearDifferentiableMaterial {
            epsilon_slope,
            mu_slope,
        })
    }

    #[test]
    fn brillouin_pairing_rejects_too_few_materials() {
        let layers = Layers::new(vec![
            integrated_layer(
                Polarisation::TransverseElectric,
                c(2.0, 0.0),
                c(3.0, 0.0),
                5.0,
                7.0,
            ),
            integrated_layer(
                Polarisation::TransverseElectric,
                c(2.0, 0.0),
                c(3.0, 0.0),
                5.0,
                7.0,
            ),
        ]);

        let materials = [material_handle(2.0, 3.0)];

        let error = layers
            .into_brillouin_layers::<crate::domain::RealAxis, _>(
                materials.iter(),
                &jet(c(2.0, 0.0)),
            )
            .expect_err("each integrated layer requires one material");

        assert_eq!(
            error,
            LayerEnergyError::MaterialCountMismatch {
                layer_count: 2,
                material_count: 1,
            },
        );
    }

    #[test]
    fn brillouin_pairing_rejects_too_many_materials() {
        let layers = Layers::new(vec![integrated_layer(
            Polarisation::TransverseElectric,
            c(2.0, 0.0),
            c(3.0, 0.0),
            5.0,
            7.0,
        )]);

        let materials = [material_handle(2.0, 3.0), material_handle(5.0, 7.0)];

        let error = layers
            .into_brillouin_layers::<crate::domain::RealAxis, _>(
                materials.iter(),
                &jet(c(2.0, 0.0)),
            )
            .expect_err("each integrated layer requires one material");

        assert_eq!(
            error,
            LayerEnergyError::MaterialCountMismatch {
                layer_count: 1,
                material_count: 2,
            },
        );
    }

    #[test]
    fn brillouin_pairing_preserves_material_order() {
        let layers = Layers::new(vec![
            integrated_layer(
                Polarisation::TransverseElectric,
                c(2.0, 0.0),
                c(3.0, 0.0),
                5.0,
                7.0,
            ),
            integrated_layer(
                Polarisation::TransverseElectric,
                c(2.0, 0.0),
                c(3.0, 0.0),
                5.0,
                7.0,
            ),
        ]);

        let materials = [material_handle(2.0, 3.0), material_handle(5.0, 7.0)];

        let paired = layers
            .into_brillouin_layers::<crate::domain::RealAxis, _>(
                materials.iter(),
                &jet(c(11.0, 0.0)),
            )
            .unwrap();

        assert_eq!(paired.len(), 2);

        assert_eq!(
            paired
                .first()
                .unwrap()
                .derivative()
                .epsilon_spectral_first()
                .value()[()],
            c(2.0, 0.0),
        );

        assert_eq!(
            paired
                .first()
                .unwrap()
                .derivative()
                .mu_spectral_first()
                .value()[()],
            c(3.0, 0.0),
        );

        assert_eq!(
            paired
                .last()
                .unwrap()
                .derivative()
                .epsilon_spectral_first()
                .value()[()],
            c(5.0, 0.0),
        );

        assert_eq!(
            paired
                .last()
                .unwrap()
                .derivative()
                .mu_spectral_first()
                .value()[()],
            c(7.0, 0.0),
        );
    }

    #[test]
    fn empty_layers_pair_with_empty_materials() {
        let layers: Layers<IntegratedLayerData<A0>> = Layers::new(Vec::new());

        let materials: [DifferentiableMaterialHandle<C>; 0] = [];

        let paired = layers
            .into_brillouin_layers::<crate::domain::RealAxis, _>(
                materials.iter(),
                &jet(c(2.0, 0.0)),
            )
            .unwrap();

        assert!(paired.is_empty());
    }
}
