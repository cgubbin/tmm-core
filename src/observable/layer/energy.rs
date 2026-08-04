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
        "integrated layer count {layer_count} does not match \
             Brillouin material-data count {material_count}"
    )]
    MaterialCountMismatch {
        layer_count: usize,
        material_count: usize,
    },
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

/// Constitutive values required for Brillouin layer energy.
///
/// The spectral derivative fields are intrinsic material derivatives with
/// respect to the canonical vacuum angular wavenumber `k0`. They retain the
/// outer jet representation independently.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IsotropicBrillouinEnergyData<A> {
    epsilon_spectral_first: A,
    mu_spectral_first: A,
}

impl<A> IsotropicBrillouinEnergyData<A> {
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

pub(crate) struct BrillouinLayerInput<A> {
    integrated: IntegratedLayerData<A>,
    material: IsotropicBrillouinEnergyData<A>,
}

impl<A> BrillouinLayerInput<A> {
    pub(crate) const fn new(
        integrated: IntegratedLayerData<A>,
        material: IsotropicBrillouinEnergyData<A>,
    ) -> Self {
        Self {
            integrated,
            material,
        }
    }

    pub(crate) fn integrated(&self) -> &IntegratedLayerData<A> {
        &self.integrated
    }

    pub(crate) fn material(&self) -> &IsotropicBrillouinEnergyData<A> {
        &self.material
    }

    pub(crate) fn into_parts(self) -> (IntegratedLayerData<A>, IsotropicBrillouinEnergyData<A>) {
        (self.integrated, self.material)
    }
}

impl<A> Layers<IntegratedLayerData<A>> {
    pub(crate) fn into_brillouin_input<'a, E, M>(
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

                    let material = IsotropicBrillouinEnergyData::new(
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

impl<A> IntegratedLayerData<A> {
    fn into_nondispersive_energy(
        self,
        vacuum_angular_wavenumber: &A,
        parallel_angular_wavenumber: &A,
        normalization: &CanonicalEnergyNormalization<A::RealJet>,
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

        let electric_coefficient = quantities.epsilon().real().multiply(normalization.common());

        let magnetic_coefficient = quantities.mu().real().multiply(normalization.common());

        let electric = electric_norm.multiply(&electric_coefficient);

        let magnetic = magnetic_norm.multiply(&magnetic_coefficient);

        let total = electric.add(&magnetic);

        LayerEnergy::new(electric, magnetic, total)
    }
}

impl<A> Layers<IntegratedLayerData<A>> {
    pub(crate) fn into_nondispersive_energy(
        self,
        vacuum_angular_wavenumber: &A,
        parallel_angular_wavenumber: &A,
        normalization: &CanonicalEnergyNormalization<A::RealJet>,
    ) -> Layers<LayerEnergy<A::RealJet>>
    where
        A: RealScalarAlgebra,
        A::RealJet: ScalarAlgebra,
        <A::RealJet as Jet>::Scalar: One,
    {
        self.map(|layer| {
            layer.into_nondispersive_energy(
                vacuum_angular_wavenumber,
                parallel_angular_wavenumber,
                normalization,
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
fn brillouin_energy_coefficients<A>(
    vacuum_angular_wavenumber: &A,
    quantities: &IsotropicLayerQuantities<A>,
    data: &IsotropicBrillouinEnergyData<A>,
    normalization: &CanonicalEnergyNormalization<A::RealJet>,
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

    let electric = electric_weight.multiply(normalization.common());

    let magnetic = magnetic_weight.multiply(normalization.common());

    (electric, magnetic)
}

impl<A> BrillouinLayerInput<A> {
    fn into_brillouin_energy(
        self,
        vacuum_angular_wavenumber: &A,
        parallel_angular_wavenumber: &A,
        normalization: &CanonicalEnergyNormalization<A::RealJet>,
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

        let electric = electric_norm.multiply(&electric_coefficient);

        let magnetic = magnetic_norm.multiply(&magnetic_coefficient);

        let total = electric.add(&magnetic);

        LayerEnergy::new(electric, magnetic, total)
    }
}

impl<A> Layers<BrillouinLayerInput<A>> {
    pub(crate) fn into_brillouin_energy(
        self,
        vacuum_angular_wavenumber: &A,
        parallel_angular_wavenumber: &A,
        normalization: &CanonicalEnergyNormalization<A::RealJet>,
    ) -> Layers<LayerEnergy<A::RealJet>>
    where
        A: RealScalarAlgebra,
        A::RealJet: ScalarAlgebra,
        <A::RealJet as Jet>::Scalar: One,
    {
        self.map(|each| {
            each.into_brillouin_energy(
                vacuum_angular_wavenumber,
                parallel_angular_wavenumber,
                normalization,
            )
        })
    }
}
