use ndarray::Dimension;
use num_traits::One;
use thiserror::Error;

use crate::{
    ComplexScalar, InterfacePower, LayerPower, Polarisation,
    algebra::{Jet, RealScalarAlgebra, ScalarAlgebra, ScalarAlgebraExpRelExt},
    backend::{IsotropicLayerQuantities, RetainedIsotropicLayers},
    input::{CanonicalCoordinates, CanonicalStack},
    material::{ConstitutiveEvaluator, ConstitutiveLift},
    observable::{
        BoundaryProjectionError, Interfaces, LayerBoundaries, LayerBoundaryWaves,
        layer::{
            EnergyDefinition, Layers,
            dissipation::{isotropic_dissipation_coefficients, project_layer_dissipation},
            energy::{
                CanonicalEnergyNormalization, LayerEnergy, brillouin_energy_coefficients,
                nondispersive_energy_coefficients, project_layer_energy,
            },
            energy_data::{IsotropicBrillouinEnergyData, IsotropicLayerEnergyData},
            field_norm::{IntegratedFieldNorms, project_integrated_field_norms},
            state_overlap::{IntegratedStateProducts, project_integrated_state_products},
        },
    },
};

use super::{
    IntegratedLayerWaveData, IntegratedWaveProducts, LayerDissipation, LayerWaveData,
    integrate_hermitian_wave_products,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RetainedLayerDatum {
    Quantities,
    Thickness,
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum LayerProjectionError {
    #[error("error in boundary projection {0}")]
    Boundary(#[from] BoundaryProjectionError),

    #[error("the backend result does not retain finite-layer boundary data")]
    LayersNotRetained,

    #[error(
        "boundary-wave count {wave_count} does not match retained layer count \
         {layer_count}"
    )]
    LayerCountMismatch {
        wave_count: usize,
        layer_count: usize,
    },

    #[error(
        "retained quantities are unavailable for finite layer {requested}; \
         retained layer count is {layer_count}"
    )]
    LayerQuantitiesUnavailable {
        requested: usize,
        layer_count: usize,
    },

    #[error(
        "finite-layer boundary data are inconsistent: {wave_count} wave \
         records and {admittance_count} admittances"
    )]
    LayerDataCountMismatch {
        wave_count: usize,
        admittance_count: usize,
    },

    #[error("missing")]
    MissingRetainedLayerDatum {
        datum: RetainedLayerDatum,
        index: usize,
        layer_count: usize,
    },
}

pub(crate) fn project_layer_power<R>(
    interfaces: Interfaces<InterfacePower<R>>,
) -> Layers<LayerPower<R>>
where
    R: ScalarAlgebra,
{
    let interfaces = interfaces.into_inner();

    let mut layers = Vec::with_capacity(interfaces.len().saturating_sub(1));

    for pair in interfaces.windows(2) {
        let left_flux = pair[0].right_net_flux().clone();

        let right_flux = pair[1].left_net_flux().clone();

        let absorbed = left_flux.subtract(&right_flux);

        layers.push(LayerPower::new(left_flux, right_flux, absorbed));
    }

    Layers::new(layers)
}

/// Combine retained boundary waves, medium quantities, and thicknesses.
///
/// Records are returned in physical left-to-right finite-layer order.
pub(crate) fn assemble_layer_wave_data<W>(
    workspace: &W,
    boundary_waves: LayerBoundaries<LayerBoundaryWaves<W::Algebra>>,
) -> Result<Layers<LayerWaveData<W::Algebra>>, LayerProjectionError>
where
    W: RetainedIsotropicLayers,
    W::Algebra: Clone,
{
    let layer_count = workspace
        .retained_layer_count()
        .ok_or(LayerProjectionError::LayersNotRetained)?;

    if boundary_waves.len() != layer_count {
        return Err(LayerProjectionError::LayerCountMismatch {
            wave_count: boundary_waves.len(),
            layer_count,
        });
    }

    let mut layers = Vec::with_capacity(layer_count);

    for (index, boundaries) in boundary_waves.into_inner().into_iter().enumerate() {
        let quantities = workspace
            .layer_quantities(index)
            .ok_or(LayerProjectionError::MissingRetainedLayerDatum {
                datum: RetainedLayerDatum::Quantities,
                index,
                layer_count,
            })?
            .clone();

        let thickness = workspace
            .layer_thickness(index)
            .ok_or(LayerProjectionError::MissingRetainedLayerDatum {
                datum: RetainedLayerDatum::Thickness,
                index,
                layer_count,
            })?
            .clone();

        let (left, _right) = boundaries.into_parts();

        layers.push(LayerWaveData::new(left, quantities, thickness));
    }

    Ok(Layers::new(layers))
}

pub(crate) fn integrate_layer_wave_data<A>(layer: LayerWaveData<A>) -> IntegratedLayerWaveData<A>
where
    A: RealScalarAlgebra + ScalarAlgebraExpRelExt,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    let (waves, quantities, thickness) = layer.into_parts();

    let wave_products = integrate_hermitian_wave_products(&waves, quantities.kappa(), &thickness);

    let state_products = project_integrated_state_products(
        &wave_products,
        &quantities.clone().into_admittance().into_inner(),
    );

    IntegratedLayerWaveData::new(wave_products, state_products, quantities, thickness)
}

pub(crate) fn integrate_layer_wave_sequence<A>(
    layers: Layers<LayerWaveData<A>>,
) -> Layers<IntegratedLayerWaveData<A>>
where
    A: RealScalarAlgebra + ScalarAlgebraExpRelExt,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    layers.map(integrate_layer_wave_data)
}

pub(crate) fn project_integrated_layer_dissipation<A>(
    layer: IntegratedLayerWaveData<A>,
    vacuum_angular_wavenumber: &A,
    parallel_wavenumber: &A,
    incident_flux_magnitude: &A::RealJet,
) -> LayerDissipation<A::RealJet>
where
    A: RealScalarAlgebra,
    A::RealJet: ScalarAlgebra,
    <A::RealJet as Jet>::Scalar: One,
{
    let (_wave_products, state_products, quantities, _thickness) = layer.into_parts();

    let field_norms = project_integrated_field_norms(
        &state_products,
        &quantities,
        vacuum_angular_wavenumber,
        parallel_wavenumber,
    );

    let coefficients = isotropic_dissipation_coefficients(
        vacuum_angular_wavenumber,
        &quantities,
        incident_flux_magnitude,
    );

    project_layer_dissipation(field_norms, &coefficients)
}

pub(crate) fn project_layer_dissipation_sequence<A>(
    layers: Layers<IntegratedLayerWaveData<A>>,
    vacuum_angular_wavenumber: &A,
    parallel_wavenumber: &A,
    incident_flux_magnitude: &A::RealJet,
) -> Layers<LayerDissipation<A::RealJet>>
where
    A: RealScalarAlgebra,
    A::RealJet: ScalarAlgebra,
    <A::RealJet as Jet>::Scalar: One,
{
    layers.map(|layer| {
        project_integrated_layer_dissipation(
            layer,
            vacuum_angular_wavenumber,
            parallel_wavenumber,
            incident_flux_magnitude,
        )
    })
}

pub(crate) fn evaluate_nondispersive_layer_energy_data<Domain, M, J>(
    coordinates: &CanonicalCoordinates<J>,
    stack: &CanonicalStack<M, J>,
) -> Layers<IsotropicLayerEnergyData<J>>
where
    J: ScalarAlgebra + ConstitutiveLift<Domain, M> + Clone,
    Domain: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
    J::Dimension: Dimension,
    J::Scalar: ComplexScalar,
{
    let layers = stack
        .layers()
        .into_iter()
        .map(|layer| {
            let epsilon =
                J::relative_permittivity(layer.material(), coordinates.vacuum_angular_wavenumber());

            let mu =
                J::relative_permeability(layer.material(), coordinates.vacuum_angular_wavenumber());

            IsotropicLayerEnergyData::nondispersive(epsilon, mu)
        })
        .collect();

    Layers::new(layers)
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LayerEnergyError {
    #[error(transparent)]
    Boundary(#[from] LayerProjectionError),

    #[error(
        "integrated layer count {integrated_count} does not match \
         energy-data layer count {energy_data_count}"
    )]
    LayerCountMismatch {
        integrated_count: usize,
        energy_data_count: usize,
    },

    #[error("energy definition {definition:?} is not yet supported")]
    UnsupportedDefinition { definition: EnergyDefinition },
}

pub(crate) fn project_integrated_layer_energy<A>(
    layer: IntegratedLayerWaveData<A>,
    data: IsotropicLayerEnergyData<A>,
    vacuum_angular_wavenumber: &A,
    parallel_wavenumber: &A,
    normalization: &CanonicalEnergyNormalization<A::RealJet>,
) -> LayerEnergy<A::RealJet>
where
    A: RealScalarAlgebra,
    A::RealJet: ScalarAlgebra,
    <A::RealJet as Jet>::Scalar: One,
{
    let (_wave_products, state_products, quantities, _thickness) = layer.into_parts();

    let field_norms = project_integrated_field_norms(
        &state_products,
        &quantities,
        vacuum_angular_wavenumber,
        parallel_wavenumber,
    );

    let coefficients = nondispersive_energy_coefficients(&data, normalization);

    project_layer_energy(field_norms, &coefficients)
}

pub(crate) fn project_layer_energy_sequence<A>(
    layers: Layers<IntegratedLayerWaveData<A>>,
    data: Layers<IsotropicLayerEnergyData<A>>,
    vacuum_angular_wavenumber: &A,
    parallel_wavenumber: &A,
    normalization: &CanonicalEnergyNormalization<A::RealJet>,
) -> Result<Layers<LayerEnergy<A::RealJet>>, LayerEnergyError>
where
    A: RealScalarAlgebra,
    A::RealJet: ScalarAlgebra,
    <A::RealJet as Jet>::Scalar: One,
{
    if layers.len() != data.len() {
        return Err(LayerEnergyError::LayerCountMismatch {
            integrated_count: layers.len(),
            energy_data_count: data.len(),
        });
    }

    let projected = layers
        .into_inner()
        .into_iter()
        .zip(data.into_inner())
        .map(|(layer, data)| {
            project_integrated_layer_energy(
                layer,
                data,
                vacuum_angular_wavenumber,
                parallel_wavenumber,
                normalization,
            )
        })
        .collect();

    Ok(Layers::new(projected))
}

pub(crate) fn project_integrated_layer_brillouin_energy<A>(
    layer: IntegratedLayerWaveData<A>,
    data: IsotropicBrillouinEnergyData<A>,
    vacuum_angular_wavenumber: &A,
    parallel_wavenumber: &A,
    normalization: &CanonicalEnergyNormalization<A::RealJet>,
) -> LayerEnergy<A::RealJet>
where
    A: RealScalarAlgebra,
    A::RealJet: ScalarAlgebra,
    <A::RealJet as Jet>::Scalar: One,
{
    let (_wave_products, state_products, quantities, _thickness) = layer.into_parts();

    let field_norms = project_integrated_field_norms(
        &state_products,
        &quantities,
        vacuum_angular_wavenumber,
        parallel_wavenumber,
    );

    let coefficients =
        brillouin_energy_coefficients(vacuum_angular_wavenumber, &quantities, &data, normalization);

    project_layer_energy(field_norms, &coefficients)
}

pub(crate) fn project_layer_brillouin_energy_sequence<A>(
    layers: Layers<IntegratedLayerWaveData<A>>,
    data: Layers<IsotropicBrillouinEnergyData<A>>,
    vacuum_angular_wavenumber: &A,
    parallel_wavenumber: &A,
    normalization: &CanonicalEnergyNormalization<A::RealJet>,
) -> Result<Layers<LayerEnergy<A::RealJet>>, LayerEnergyError>
where
    A: RealScalarAlgebra,
    A::RealJet: ScalarAlgebra,
    <A::RealJet as Jet>::Scalar: One,
{
    if layers.len() != data.len() {
        return Err(LayerEnergyError::LayerCountMismatch {
            integrated_count: layers.len(),
            energy_data_count: data.len(),
        });
    }

    let energy = layers
        .into_inner()
        .into_iter()
        .zip(data.into_inner())
        .map(|(layer, data)| {
            project_integrated_layer_brillouin_energy(
                layer,
                data,
                vacuum_angular_wavenumber,
                parallel_wavenumber,
                normalization,
            )
        })
        .collect();

    Ok(Layers::new(energy))
}

#[cfg(test)]
mod tests {
    use ndarray::{Ix0, arr0};

    use super::*;
    use crate::{
        IncidentSide, Polarisation,
        algebra::{ArrayJet0, Jet0, RealParameter},
        backend::{
            IsotropicLayerQuantities, LayerBoundaryWaves as BackendLayerBoundaryWaves,
            ReconstructLayerBoundaryWaves,
        },
        observable::{BoundaryWaves, DirectedPower, InterfacePower, Interfaces},
        test_support::{C, TOLERANCE, assertions::assert_real_close, jet::J0},
    };

    type A = ArrayJet0<f64, Ix0, RealParameter>;
    type CA = ArrayJet0<C, Ix0, RealParameter>;

    fn jet(value: f64) -> A {
        Jet0::new(arr0(value))
    }

    fn complex_jet(value: C) -> J0 {
        Jet0::new(arr0(value))
    }

    fn scalar(value: &A) -> f64 {
        value.value()[()]
    }

    fn interface_power(left_net_flux: f64, right_net_flux: f64) -> InterfacePower<A> {
        InterfacePower::new(
            DirectedPower::new(
                jet(100.0 + left_net_flux),
                jet(200.0 + left_net_flux),
                jet(left_net_flux),
            ),
            DirectedPower::new(
                jet(300.0 + right_net_flux),
                jet(400.0 + right_net_flux),
                jet(right_net_flux),
            ),
        )
    }

    #[test]
    fn empty_interface_sequence_produces_no_layers() {
        let layers = project_layer_power::<A>(Interfaces::new(Vec::new()));

        assert!(layers.is_empty());
    }

    #[test]
    fn one_interface_produces_no_finite_layers() {
        let layers = project_layer_power(Interfaces::new(vec![interface_power(0.8, 0.8)]));

        assert!(layers.is_empty());
    }

    #[test]
    fn two_interfaces_produce_one_layer() {
        let layers = project_layer_power(Interfaces::new(vec![
            interface_power(0.9, 0.8),
            interface_power(0.6, 0.5),
        ]));

        assert_eq!(layers.len(), 1);

        let layer = layers.get(0).unwrap();

        /*
         * Use the finite-layer sides:
         *
         * left boundary  = interface 0 right
         * right boundary = interface 1 left
         */
        assert_real_close(scalar(layer.left_flux()), 0.8, TOLERANCE);
        assert_real_close(scalar(layer.right_flux()), 0.6, TOLERANCE);
        assert_real_close(scalar(layer.absorbed()), 0.2, TOLERANCE);
    }

    #[test]
    fn multiple_layers_preserve_physical_order() {
        let layers = project_layer_power(Interfaces::new(vec![
            interface_power(1.0, 0.9),
            interface_power(0.8, 0.7),
            interface_power(0.4, 0.3),
            interface_power(0.1, 0.0),
        ]));

        assert_eq!(layers.len(), 3);

        let first = layers.get(0).unwrap();
        let second = layers.get(1).unwrap();
        let third = layers.get(2).unwrap();

        assert_real_close(scalar(first.left_flux()), 0.9, TOLERANCE);
        assert_real_close(scalar(first.right_flux()), 0.8, TOLERANCE);
        assert_real_close(scalar(first.absorbed()), 0.1, TOLERANCE);

        assert_real_close(scalar(second.left_flux()), 0.7, TOLERANCE);
        assert_real_close(scalar(second.right_flux()), 0.4, TOLERANCE);
        assert_real_close(scalar(second.absorbed()), 0.3, TOLERANCE);

        assert_real_close(scalar(third.left_flux()), 0.3, TOLERANCE);
        assert_real_close(scalar(third.right_flux()), 0.1, TOLERANCE);
        assert_real_close(scalar(third.absorbed()), 0.2, TOLERANCE);
    }

    #[test]
    fn absorption_uses_global_flux_difference_for_negative_flux() {
        /*
         * This represents right incidence. Flux remains globally signed:
         *
         * left boundary  = -0.4
         * right boundary = -0.7
         *
         * absorbed = -0.4 - (-0.7) = +0.3
         */
        let layers = project_layer_power(Interfaces::new(vec![
            interface_power(-0.5, -0.4),
            interface_power(-0.7, -0.8),
        ]));

        let layer = layers.get(0).unwrap();

        assert_real_close(scalar(layer.left_flux()), -0.4, TOLERANCE);
        assert_real_close(scalar(layer.right_flux()), -0.7, TOLERANCE);
        assert_real_close(scalar(layer.absorbed()), 0.3, TOLERANCE);
    }

    #[test]
    fn lossless_layer_has_zero_absorption() {
        let layers = project_layer_power(Interfaces::new(vec![
            interface_power(0.75, 0.75),
            interface_power(0.75, 0.75),
        ]));

        let layer = layers.get(0).unwrap();

        assert_real_close(scalar(layer.absorbed()), 0.0, TOLERANCE);
    }

    #[test]
    fn projection_uses_finite_layer_sides_not_exterior_sides() {
        let layers = project_layer_power(Interfaces::new(vec![
            /*
             * Deliberately discontinuous marker values. The projection
             * must use the right side of the left interface.
             */
            interface_power(100.0, 0.8),
            /*
             * It must use the left side of the right interface.
             */
            interface_power(0.6, 200.0),
        ]));

        let layer = layers.get(0).unwrap();

        assert_real_close(scalar(layer.left_flux()), 0.8, TOLERANCE);
        assert_real_close(scalar(layer.right_flux()), 0.6, TOLERANCE);
        assert_real_close(scalar(layer.absorbed()), 0.2, TOLERANCE);
    }

    #[test]
    fn summed_layer_absorption_telescopes_when_interfaces_are_continuous() {
        let layers = project_layer_power(Interfaces::new(vec![
            interface_power(1.0, 1.0),
            interface_power(0.8, 0.8),
            interface_power(0.5, 0.5),
            interface_power(0.2, 0.2),
        ]));

        let total: f64 = layers.iter().map(|layer| scalar(layer.absorbed())).sum();

        assert_real_close(total, 0.8, TOLERANCE);
    }

    fn boundary_waves_fixture(offset: f64) -> BoundaryWaves<CA> {
        BoundaryWaves::new(
            complex_jet(C::new(offset + 0.8, offset * 0.01 + 0.3)),
            complex_jet(C::new(offset - 0.2, offset * 0.02 + 0.5)),
        )
    }

    #[test]
    fn integrate_layer_wave_data_uses_retained_kappa() {
        let waves = BoundaryWaves::new(
            complex_jet(C::new(0.8, 0.3)),
            complex_jet(C::new(-0.2, 0.5)),
        );

        let quantities = IsotropicLayerQuantities::test_fixture(
            complex_jet(C::new(2.4, 0.35)),
            complex_jet(C::new(2.0, 1.0)),
            complex_jet(C::new(1.0, 0.0)),
            crate::Polarisation::TransverseElectric,
        );

        let thickness = complex_jet(C::new(1.7, 0.0));

        let expected = integrate_hermitian_wave_products(&waves, quantities.kappa(), &thickness);

        let actual = integrate_layer_wave_data(LayerWaveData::new(waves, quantities, thickness));

        assert_eq!(actual.wave_products(), &expected);
    }
    #[test]
    fn integrate_layer_wave_data_preserves_metadata() {
        let waves = boundary_waves_fixture(1.0);
        let quantities = IsotropicLayerQuantities::test_fixture(
            complex_jet(C::new(2.4, 0.35)),
            complex_jet(C::new(2.0, 1.0)),
            complex_jet(C::new(1.0, 0.0)),
            crate::Polarisation::TransverseElectric,
        );

        let thickness = complex_jet(C::new(0.7, 0.0));

        let expected_quantities = quantities.clone();
        let expected_thickness = thickness.clone();

        let result = integrate_layer_wave_data(LayerWaveData::new(waves, quantities, thickness));

        assert_eq!(result.quantities(), &expected_quantities,);

        assert_eq!(result.thickness(), &expected_thickness,);
    }

    fn layer_wave_data_fixture(offset: f64) -> LayerWaveData<CA> {
        LayerWaveData::new(
            boundary_waves_fixture(offset),
            super::error_tests::isotropic_quantities_fixture(offset),
            complex_jet(C::new(0.4 + 0.01 * offset, 0.0)),
        )
    }
    #[test]
    fn integrating_layer_wave_sequence_preserves_count_and_order() {
        let first = layer_wave_data_fixture(0.0);

        let second = layer_wave_data_fixture(10.0);

        let expected_first = integrate_layer_wave_data(first.clone());

        let expected_second = integrate_layer_wave_data(second.clone());

        let actual = integrate_layer_wave_sequence(Layers::new(vec![first, second]));

        assert_eq!(actual.len(), 2);
        assert_eq!(actual.get(0), Some(&expected_first));
        assert_eq!(actual.get(1), Some(&expected_second));
    }

    #[test]
    fn integrate_layer_wave_data_uses_kappa_as_exponent_coefficient() {
        let layer = layer_wave_data_fixture(0.0);

        let expected = integrate_hermitian_wave_products(
            layer.waves(),
            layer.quantities().kappa(),
            layer.thickness(),
        );

        let actual = integrate_layer_wave_data(layer);

        assert_eq!(actual.wave_products(), &expected,);
    }

    #[test]
    fn integrate_layer_wave_data_preserves_quantities_and_thickness() {
        let layer = layer_wave_data_fixture(0.0);

        let expected_quantities = layer.quantities().clone();

        let expected_thickness = layer.thickness().clone();

        let integrated = integrate_layer_wave_data(layer);

        assert_eq!(integrated.quantities(), &expected_quantities,);

        assert_eq!(integrated.thickness(), &expected_thickness,);
    }

    #[test]
    fn integrate_layer_wave_sequence_preserves_count_and_order() {
        let first = layer_wave_data_fixture(0.0);

        let second = layer_wave_data_fixture(10.0);

        let expected_first = integrate_layer_wave_data(first.clone());

        let expected_second = integrate_layer_wave_data(second.clone());

        let actual = integrate_layer_wave_sequence(Layers::new(vec![first, second]));

        assert_eq!(actual.len(), 2);
        assert_eq!(actual.get(0), Some(&expected_first),);
        assert_eq!(actual.get(1), Some(&expected_second),);
    }
}

#[cfg(test)]
mod error_tests {
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        Polarisation,
        algebra::{ArrayJet0, Jet0, RealParameter},
        backend::{IsotropicLayerQuantities, RetainedIsotropicLayers},
        observable::{BoundaryWaves, LayerBoundaries, LayerBoundaryWaves},
    };

    type C = Complex64;
    type A = ArrayJet0<C, Ix0, RealParameter>;

    fn jet(value: C) -> A {
        Jet0::new(arr0(value))
    }

    fn real_jet(value: f64) -> A {
        jet(C::new(value, 0.0))
    }

    fn boundary_waves(offset: f64) -> BoundaryWaves<A> {
        BoundaryWaves::new(real_jet(offset + 1.0), real_jet(offset + 2.0))
    }

    fn layer_boundary_waves(offset: f64) -> LayerBoundaryWaves<A> {
        LayerBoundaryWaves::new(boundary_waves(offset), boundary_waves(offset + 10.0))
    }

    pub(super) fn isotropic_quantities_fixture(offset: f64) -> IsotropicLayerQuantities<A> {
        let polarisation = Polarisation::TransverseElectric;
        let kappa = jet(C::new(offset + 2.4, 0.35));

        let factor = jet(C::new(offset + 3.1, -0.2));

        let (mu, epsilon) = match polarisation {
            Polarisation::TransverseElectric => (factor, jet(C::new(1.0, 0.0))),
            Polarisation::TransverseMagnetic => (jet(C::new(1.0, 0.0)), factor),
        };

        IsotropicLayerQuantities::test_fixture(kappa, epsilon, mu, polarisation)
    }

    #[derive(Clone, Debug)]
    struct TestRetainedLayers {
        reported_count: Option<usize>,
        quantities: Vec<Option<IsotropicLayerQuantities<A>>>,
        thicknesses: Vec<Option<A>>,
    }

    impl TestRetainedLayers {
        fn new(
            reported_count: Option<usize>,
            quantities: Vec<Option<IsotropicLayerQuantities<A>>>,
            thicknesses: Vec<Option<A>>,
        ) -> Self {
            Self {
                reported_count,
                quantities,
                thicknesses,
            }
        }
    }

    impl RetainedIsotropicLayers for TestRetainedLayers {
        type Algebra = A;

        fn retained_layer_count(&self) -> Option<usize> {
            self.reported_count
        }

        fn layer_quantities(
            &self,
            index: usize,
        ) -> Option<&IsotropicLayerQuantities<Self::Algebra>> {
            self.quantities.get(index).and_then(Option::as_ref)
        }

        fn layer_thickness(&self, index: usize) -> Option<&Self::Algebra> {
            self.thicknesses.get(index).and_then(Option::as_ref)
        }
    }

    #[test]
    fn assemble_layer_wave_data_rejects_wave_count_mismatch() {
        let workspace = TestRetainedLayers::new(
            Some(2),
            vec![
                Some(isotropic_quantities_fixture(10.0)),
                Some(isotropic_quantities_fixture(20.0)),
            ],
            vec![Some(real_jet(0.3)), Some(real_jet(0.7))],
        );

        let boundary_waves = LayerBoundaries::new(vec![layer_boundary_waves(100.0)]);

        let error = assemble_layer_wave_data(&workspace, boundary_waves)
            .expect_err("one wave record must not satisfy a two-layer workspace");

        assert_eq!(
            error,
            LayerProjectionError::LayerCountMismatch {
                wave_count: 1,
                layer_count: 2,
            },
        );
    }

    #[test]
    fn assemble_layer_wave_data_rejects_missing_quantities() {
        let workspace = TestRetainedLayers::new(
            Some(2),
            vec![Some(isotropic_quantities_fixture(10.0)), None],
            vec![Some(real_jet(0.3)), Some(real_jet(0.7))],
        );

        let boundary_waves = LayerBoundaries::new(vec![
            layer_boundary_waves(100.0),
            layer_boundary_waves(200.0),
        ]);

        let error = assemble_layer_wave_data(&workspace, boundary_waves)
            .expect_err("missing quantities for a reported layer must be rejected");

        assert_eq!(
            error,
            LayerProjectionError::MissingRetainedLayerDatum {
                datum: RetainedLayerDatum::Quantities,
                index: 1,
                layer_count: 2,
            },
        );
    }

    #[test]
    fn assemble_layer_wave_data_rejects_missing_thickness() {
        let workspace = TestRetainedLayers::new(
            Some(2),
            vec![
                Some(isotropic_quantities_fixture(10.0)),
                Some(isotropic_quantities_fixture(20.0)),
            ],
            vec![Some(real_jet(0.3)), None],
        );

        let boundary_waves = LayerBoundaries::new(vec![
            layer_boundary_waves(100.0),
            layer_boundary_waves(200.0),
        ]);

        let error = assemble_layer_wave_data(&workspace, boundary_waves)
            .expect_err("missing thickness for a reported layer must be rejected");

        assert_eq!(
            error,
            LayerProjectionError::MissingRetainedLayerDatum {
                datum: RetainedLayerDatum::Thickness,
                index: 1,
                layer_count: 2,
            },
        );
    }

    type C0 = ArrayJet0<Complex64, Ix0, RealParameter>;
    type R0 = ArrayJet0<f64, Ix0, RealParameter>;

    fn complex_jet(re: f64, im: f64) -> C0 {
        Jet0::new(arr0(Complex64::new(re, im)))
    }

    fn real_data_jet(re: f64) -> R0 {
        Jet0::new(arr0(re))
    }
    fn state_products_fixture(offset: f64) -> IntegratedStateProducts<C0> {
        IntegratedStateProducts::new(
            complex_jet(offset + 11.0, 0.0),
            complex_jet(offset + 12.0, 0.0),
            complex_jet(offset + 13.0, 0.0),
            complex_jet(offset + 14.0, 0.0),
        )
    }
    fn overlap_fixture(offset: f64) -> IntegratedWaveProducts<C0> {
        IntegratedWaveProducts::new(
            complex_jet(offset + 1.0, 0.0),
            complex_jet(offset + 2.0, 0.0),
            complex_jet(offset + 3.0, 0.0),
            complex_jet(offset + 4.0, 0.0),
        )
    }

    fn integrated_layer_fixture(offset: f64) -> IntegratedLayerWaveData<C0> {
        IntegratedLayerWaveData::new(
            overlap_fixture(offset),
            state_products_fixture(offset),
            isotropic_quantities_fixture(offset),
            real_jet(1000.0 + offset),
        )
    }

    fn energy_data_fixture(offset: f64) -> IsotropicLayerEnergyData<C0> {
        IsotropicLayerEnergyData::nondispersive(
            complex_jet(offset + 2.0, 0.1),
            complex_jet(offset + 3.0, -0.2),
        )
    }

    #[test]
    fn layer_energy_sequence_rejects_mismatched_counts() {
        let error = project_layer_energy_sequence(
            Layers::new(vec![
                integrated_layer_fixture(0.0),
                integrated_layer_fixture(10.0),
            ]),
            Layers::new(vec![energy_data_fixture(0.0)]),
            &complex_jet(2.0, 0.0),
            &complex_jet(0.3, 0.0),
            &CanonicalEnergyNormalization::new(real_data_jet(1.0)),
        )
        .expect_err("unaligned sequences must be rejected");

        assert_eq!(
            error,
            LayerEnergyError::LayerCountMismatch {
                integrated_count: 2,
                energy_data_count: 1,
            },
        );
    }
}
