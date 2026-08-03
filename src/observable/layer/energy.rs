//! Analytically integrated electromagnetic energy in homogeneous finite
//! layers.
//!
//! This module separates:
//!
//! - integrated electric and magnetic field norms;
//! - the constitutive weights defining an energy convention;
//! - the final normalized layer-energy contribution.
//!
//! Energy-specific constitutive data are evaluated lazily when an analysis is
//! requested. They are not retained by every backend workspace.

use num_traits::FromPrimitive;

use crate::{
    algebra::{ComplexJet, Jet, RealScalarAlgebra, ScalarAlgebra},
    backend::IsotropicLayerQuantities,
    observable::layer::energy_data::{IsotropicBrillouinEnergyData, IsotropicLayerEnergyData},
};

use super::IntegratedFieldNorms;

/// Definition used to calculate real-frequency electromagnetic energy.
///
/// The definitions differ only in their constitutive weights. Spatial field
/// integration is shared.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub enum EnergyDefinition {
    /// Nondispersive electromagnetic energy.
    ///
    /// The electric and magnetic weights are respectively proportional to
    /// `Re(epsilon)` and `Re(mu)`.
    Nondispersive,

    /// Brillouin energy for a dispersive medium.
    ///
    /// The constitutive weights are:
    ///
    /// ```text
    /// d(omega epsilon) / d omega
    /// d(omega mu)      / d omega
    /// ```
    ///
    /// The precise treatment of appreciably lossy media must remain explicit
    /// in the constitutive analysis policy.
    #[default]
    Brillouin,
}

/// Constitutive weights applied to integrated electric and magnetic norms.
///
/// These coefficients include all factors associated with:
///
/// - the selected energy definition;
/// - canonical electromagnetic normalization;
/// - incident-flux normalization, when the result is normalized.
///
/// They do not include the integrated field norms themselves.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IsotropicEnergyCoefficients<R> {
    electric: R,
    magnetic: R,
}

impl<R> IsotropicEnergyCoefficients<R> {
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

    pub(crate) fn map<U>(self, mut map: impl FnMut(R) -> U) -> IsotropicEnergyCoefficients<U> {
        IsotropicEnergyCoefficients {
            electric: map(self.electric),
            magnetic: map(self.magnetic),
        }
    }
}

/// Integrated electromagnetic energy associated with one finite layer.
///
/// The quantity is normalized according to the coefficient construction used
/// by the caller. For plane-wave scattering analysis, the natural convention
/// is energy per unit incident power flux.
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

/// Apply constitutive energy weights to integrated field norms.
pub(crate) fn project_layer_energy<R>(
    field_norms: IntegratedFieldNorms<R>,
    coefficients: &IsotropicEnergyCoefficients<R>,
) -> LayerEnergy<R>
where
    R: ScalarAlgebra,
{
    let (electric_norm, magnetic_norm) = field_norms.into_parts();

    let electric = electric_norm.multiply(coefficients.electric());

    let magnetic = magnetic_norm.multiply(coefficients.magnetic());

    let total = electric.add(&magnetic);

    LayerEnergy::new(electric, magnetic, total)
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
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CanonicalEnergyNormalization<R> {
    common: R,
}

impl<R> CanonicalEnergyNormalization<R> {
    pub(crate) const fn new(common: R) -> Self {
        Self { common }
    }

    pub(crate) fn common(&self) -> &R {
        &self.common
    }
}

pub(crate) fn canonical_energy_normalization<A>(
    vacuum_angular_wavenumber: &A,
    incident_flux_magnitude: &A::RealJet,
) -> CanonicalEnergyNormalization<A::RealJet>
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

    CanonicalEnergyNormalization::new(common)
}

pub(crate) fn nondispersive_energy_coefficients<A>(
    data: &IsotropicLayerEnergyData<A>,
    normalization: &CanonicalEnergyNormalization<A::RealJet>,
) -> IsotropicEnergyCoefficients<A::RealJet>
where
    A: RealScalarAlgebra,
    A::RealJet: ScalarAlgebra,
{
    debug_assert_eq!(data.definition(), EnergyDefinition::Nondispersive,);

    let electric = data.epsilon().real().multiply(normalization.common());

    let magnetic = data.mu().real().multiply(normalization.common());

    IsotropicEnergyCoefficients::new(electric, magnetic)
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
pub(crate) fn brillouin_energy_coefficients<A>(
    vacuum_angular_wavenumber: &A,
    quantities: &IsotropicLayerQuantities<A>,
    data: &IsotropicBrillouinEnergyData<A>,
    normalization: &CanonicalEnergyNormalization<A::RealJet>,
) -> IsotropicEnergyCoefficients<A::RealJet>
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

    let electric = electric_weight.multiply(normalization.common());

    let magnetic = magnetic_weight.multiply(normalization.common());

    IsotropicEnergyCoefficients::new(electric, magnetic)
}

#[cfg(test)]
mod tests {
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        Polarisation,
        algebra::{ArrayJet0, ArrayJet1, Jet0, RealParameter},
    };

    type R0 = ArrayJet0<f64, Ix0, RealParameter>;

    fn real_jet(value: f64) -> R0 {
        Jet0::new(arr0(value))
    }

    fn complex_jet(re: f64, im: f64) -> ArrayJet0<Complex64, Ix0, RealParameter> {
        Jet0::new(arr0(Complex64::new(re, im)))
    }

    fn scalar(value: &R0) -> f64 {
        value.value()[()]
    }

    #[test]
    fn energy_definition_defaults_to_brillouin() {
        assert_eq!(EnergyDefinition::default(), EnergyDefinition::Brillouin,);
    }

    #[test]
    fn energy_coefficients_store_both_components() {
        let coefficients = IsotropicEnergyCoefficients::new(1, 2);

        assert_eq!(coefficients.electric(), &1);
        assert_eq!(coefficients.magnetic(), &2);
    }

    #[test]
    fn energy_coefficients_into_parts_preserves_order() {
        let coefficients = IsotropicEnergyCoefficients::new(1, 2);

        assert_eq!(coefficients.into_parts(), (1, 2));
    }

    #[test]
    fn energy_coefficients_map_transforms_both_components() {
        let coefficients = IsotropicEnergyCoefficients::new(1, 2);

        let mapped = coefficients.map(|value| value * 10);

        assert_eq!(mapped.electric(), &10);
        assert_eq!(mapped.magnetic(), &20);
    }

    #[test]
    fn layer_energy_stores_all_components() {
        let energy = LayerEnergy::new(1, 2, 3);

        assert_eq!(energy.electric(), &1);
        assert_eq!(energy.magnetic(), &2);
        assert_eq!(energy.total(), &3);
    }

    #[test]
    fn layer_energy_into_parts_preserves_order() {
        let energy = LayerEnergy::new(1, 2, 3);

        assert_eq!(energy.into_parts(), (1, 2, 3));
    }

    #[test]
    fn layer_energy_map_transforms_every_component() {
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
    fn energy_projection_applies_both_coefficients() {
        let norms = IntegratedFieldNorms::new(real_jet(2.0), real_jet(3.0));

        let coefficients = IsotropicEnergyCoefficients::new(real_jet(5.0), real_jet(7.0));

        let energy = project_layer_energy(norms, &coefficients);

        assert_eq!(scalar(energy.electric()), 10.0);
        assert_eq!(scalar(energy.magnetic()), 21.0);
        assert_eq!(scalar(energy.total()), 31.0);
    }

    #[test]
    fn zero_electric_weight_removes_electric_energy() {
        let energy = project_layer_energy(
            IntegratedFieldNorms::new(real_jet(2.0), real_jet(3.0)),
            &IsotropicEnergyCoefficients::new(real_jet(0.0), real_jet(7.0)),
        );

        assert_eq!(scalar(energy.electric()), 0.0);
        assert_eq!(scalar(energy.magnetic()), 21.0);
        assert_eq!(scalar(energy.total()), 21.0);
    }

    #[test]
    fn zero_magnetic_weight_removes_magnetic_energy() {
        let energy = project_layer_energy(
            IntegratedFieldNorms::new(real_jet(2.0), real_jet(3.0)),
            &IsotropicEnergyCoefficients::new(real_jet(5.0), real_jet(0.0)),
        );

        assert_eq!(scalar(energy.electric()), 10.0);
        assert_eq!(scalar(energy.magnetic()), 0.0);
        assert_eq!(scalar(energy.total()), 10.0);
    }

    #[test]
    fn total_is_exact_sum_of_components() {
        let energy = project_layer_energy(
            IntegratedFieldNorms::new(real_jet(2.5), real_jet(3.5)),
            &IsotropicEnergyCoefficients::new(real_jet(4.0), real_jet(6.0)),
        );

        assert_eq!(
            scalar(energy.total()),
            scalar(energy.electric()) + scalar(energy.magnetic()),
        );
    }

    #[test]
    fn projection_propagates_first_derivatives() {
        type R = ArrayJet1<f64, Ix0, RealParameter>;

        fn first(value: f64, derivative: f64) -> R {
            R::from_parts(arr0(value), arr0(derivative))
        }

        let energy = project_layer_energy(
            IntegratedFieldNorms::new(first(2.0, 3.0), first(5.0, 7.0)),
            &IsotropicEnergyCoefficients::new(first(11.0, 13.0), first(17.0, 19.0)),
        );

        assert_eq!(energy.electric().value()[()], 22.0);
        assert_eq!(energy.electric().first()[()], 3.0 * 11.0 + 2.0 * 13.0,);

        assert_eq!(energy.magnetic().value()[()], 85.0);
        assert_eq!(energy.magnetic().first()[()], 7.0 * 17.0 + 5.0 * 19.0,);

        assert_eq!(energy.total().value()[()], 107.0);
        assert_eq!(energy.total().first()[()], 273.0);
    }

    #[test]
    fn nondispersive_coefficients_apply_real_constitutive_parts() {
        type C = Complex64;
        type A = ArrayJet0<C, Ix0, RealParameter>;
        type R = <A as ComplexJet>::RealJet;

        let data = IsotropicLayerEnergyData::nondispersive(
            Jet0::new(arr0(C::new(2.0, 0.7))),
            Jet0::new(arr0(C::new(3.0, -0.4))),
        );

        let normalization = CanonicalEnergyNormalization::new(Jet0::new(arr0(5.0)));

        let coefficients = nondispersive_energy_coefficients(&data, &normalization);

        assert_eq!(coefficients.electric().value()[()], 10.0,);

        assert_eq!(coefficients.magnetic().value()[()], 15.0,);
    }

    #[test]
    fn nondispersive_coefficients_ignore_imaginary_constitutive_parts() {
        let first = IsotropicLayerEnergyData::nondispersive(
            complex_jet(2.0, 100.0),
            complex_jet(3.0, -200.0),
        );

        let second =
            IsotropicLayerEnergyData::nondispersive(complex_jet(2.0, -7.0), complex_jet(3.0, 11.0));

        let normalization = CanonicalEnergyNormalization::new(real_jet(5.0));

        assert_eq!(
            nondispersive_energy_coefficients(&first, &normalization,),
            nondispersive_energy_coefficients(&second, &normalization,),
        );
    }

    #[test]
    fn nondispersive_coefficients_propagate_first_derivatives() {
        type C = Complex64;
        type A = ArrayJet1<C, Ix0, RealParameter>;
        type R = <A as ComplexJet>::RealJet;

        let epsilon = A::from_parts(arr0(C::new(2.0, 0.4)), arr0(C::new(3.0, 101.0)));

        let mu = A::from_parts(arr0(C::new(5.0, -0.2)), arr0(C::new(7.0, -103.0)));

        let normalization =
            CanonicalEnergyNormalization::new(R::from_parts(arr0(11.0), arr0(13.0)));

        let coefficients = nondispersive_energy_coefficients(
            &IsotropicLayerEnergyData::nondispersive(epsilon, mu),
            &normalization,
        );

        assert_eq!(coefficients.electric().value()[()], 22.0,);

        assert_eq!(coefficients.electric().first()[()], 3.0 * 11.0 + 2.0 * 13.0,);

        assert_eq!(coefficients.magnetic().value()[()], 55.0,);

        assert_eq!(coefficients.magnetic().first()[()], 7.0 * 11.0 + 5.0 * 13.0,);
    }

    #[test]
    fn brillouin_coefficients_use_retained_values_and_intrinsic_derivatives() {
        let quantities = IsotropicLayerQuantities::test_fixture(
            complex_jet(3.0, 0.2),    // kappa
            complex_jet(2.0, 100.0),  // epsilon
            complex_jet(3.0, -200.0), // mu
            Polarisation::TransverseElectric,
        );

        let data =
            IsotropicBrillouinEnergyData::new(complex_jet(5.0, 101.0), complex_jet(7.0, -201.0));

        let coefficients = brillouin_energy_coefficients(
            &complex_jet(11.0, 0.0),
            &quantities,
            &data,
            &CanonicalEnergyNormalization::new(real_jet(13.0)),
        );

        /*
         * Electric:
         *   Re[2 + 11*5] * 13 = 741
         *
         * Magnetic:
         *   Re[3 + 11*7] * 13 = 1040
         */
        assert_eq!(coefficients.electric().value()[()], 741.0,);

        assert_eq!(coefficients.magnetic().value()[()], 1040.0,);
    }
}
