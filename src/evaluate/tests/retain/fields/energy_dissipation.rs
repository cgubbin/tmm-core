use approx::assert_relative_eq;
use ndarray::Array1;

use crate::{
    FiniteLayerIndex, IncidentSide, Parameter, PlaneWaveEvaluator, Polarisation,
    backend::{ExteriorContextProvider, scatter2::Scatter2, transfer2::Transfer2},
    spatial::{FieldSampling, LayerSampling},
    test_support::{
        finite_difference::{
            FIRST_DERIVATIVE_TOLERANCE, SECOND_DERIVATIVE_TOLERANCE, VALUE_TOLERANCE,
        },
        jet::J0,
        planar::{scalar_real_input, two_layer_stack},
        stack::{absorbing_two_layer_stack, two_layer_stack_with_lossless_first_layer},
    },
};

fn sampling() -> FieldSampling<f64> {
    FieldSampling::new()
        .layer(0, LayerSampling::uniform(4))
        .layer(1, LayerSampling::uniform(4))
}

fn assert_real_array_close(actual: &Array1<f64>, expected: &Array1<f64>, tolerance: f64) {
    assert_eq!(actual.raw_dim(), expected.raw_dim());

    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        assert_relative_eq!(
            actual,
            expected,
            epsilon = tolerance,
            max_relative = tolerance,
        );
    }
}

fn assert_all_finite(values: &Array1<f64>) {
    assert!(
        values.iter().all(|value| value.is_finite()),
        "all sampled values should be finite",
    );
}

fn assert_all_close_to_zero(values: &Array1<f64>, tolerance: f64) {
    for &value in values {
        assert_relative_eq!(value, 0.0, epsilon = tolerance, max_relative = tolerance,);
    }
}

fn assert_total_is_sum(
    electric: &Array1<f64>,
    magnetic: &Array1<f64>,
    total: &Array1<f64>,
    tolerance: f64,
) {
    assert_eq!(electric.raw_dim(), magnetic.raw_dim());
    assert_eq!(electric.raw_dim(), total.raw_dim());

    for ((&electric, &magnetic), &total) in electric.iter().zip(magnetic).zip(total) {
        assert_relative_eq!(
            total,
            electric + magnetic,
            epsilon = tolerance,
            max_relative = tolerance,
        );
    }
}

macro_rules! for_each_backend {
    ($evaluator:ident, $body:block) => {{
        {
            let $evaluator = PlaneWaveEvaluator::new(Scatter2::new());
            $body
        }

        {
            let $evaluator = PlaneWaveEvaluator::new(Transfer2::new());
            $body
        }
    }};
}

#[test]
fn energy_density_is_finite_and_total_is_sum_of_components() {
    let stack = two_layer_stack();
    let sampling = sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for side in [IncidentSide::Left, IncidentSide::Right] {
            for_each_backend!(evaluator, {
                let state = evaluator
                    .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
                    .unwrap();

                let point = state.project_point(&()).unwrap();

                let response = point
                    .excitation(side)
                    .unwrap()
                    .evaluate_energy_density(&sampling)
                    .unwrap();

                assert_eq!(
                    response.sampling().len(),
                    sampling.resolve(&stack).unwrap().len(),
                );

                assert_all_finite(response.value().electric());

                assert_all_finite(response.value().magnetic());

                assert_all_finite(response.value().total());

                assert_total_is_sum(
                    response.value().electric(),
                    response.value().magnetic(),
                    response.value().total(),
                    VALUE_TOLERANCE,
                );
            });
        }
    }
}

#[test]
fn nondispersive_lossless_energy_density_is_nonnegative() {
    let stack = two_layer_stack();
    let sampling = sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for side in [IncidentSide::Left, IncidentSide::Right] {
            for_each_backend!(evaluator, {
                let state = evaluator
                    .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
                    .unwrap();

                let point = state.project_point(&()).unwrap();

                let response = point
                    .excitation(side)
                    .unwrap()
                    .evaluate_energy_density(&sampling)
                    .unwrap();

                for &value in response.value().electric() {
                    assert!(
                        value >= -VALUE_TOLERANCE,
                        "electric energy density should be nonnegative: {value}",
                    );
                }

                for &value in response.value().magnetic() {
                    assert!(
                        value >= -VALUE_TOLERANCE,
                        "magnetic energy density should be nonnegative: {value}",
                    );
                }

                for &value in response.value().total() {
                    assert!(
                        value >= -VALUE_TOLERANCE,
                        "total energy density should be nonnegative: {value}",
                    );
                }
            });
        }
    }
}

#[test]
fn lossless_stack_has_zero_dissipation_density() {
    let stack = two_layer_stack();
    let sampling = sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for side in [IncidentSide::Left, IncidentSide::Right] {
            for_each_backend!(evaluator, {
                let state = evaluator
                    .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
                    .unwrap();

                let point = state.project_point(&()).unwrap();

                let response = point
                    .excitation(side)
                    .unwrap()
                    .evaluate_dissipation_density(&sampling)
                    .unwrap();

                assert_all_close_to_zero(response.value().electric(), VALUE_TOLERANCE);

                assert_all_close_to_zero(response.value().magnetic(), VALUE_TOLERANCE);

                assert_all_close_to_zero(response.value().total(), VALUE_TOLERANCE);
            });
        }
    }
}

#[test]
fn dissipation_density_is_localised_to_absorbing_layer() {
    let stack = two_layer_stack_with_lossless_first_layer();

    /*
     * Four samples in each layer:
     *
     *   indices 0..4 -> layer 0, lossless
     *   indices 4..8 -> layer 1, absorbing
     */
    let sampling = sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for side in [IncidentSide::Left, IncidentSide::Right] {
            for_each_backend!(evaluator, {
                let state = evaluator
                    .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
                    .unwrap();

                let point = state.project_point(&()).unwrap();

                let response = point
                    .excitation(side)
                    .unwrap()
                    .evaluate_dissipation_density(&sampling)
                    .unwrap();

                let total = response.value().total();

                assert_eq!(total.len(), 8);

                for &value in &total.as_slice().unwrap()[0..4] {
                    assert_relative_eq!(
                        value,
                        0.0,
                        epsilon = VALUE_TOLERANCE,
                        max_relative = VALUE_TOLERANCE,
                    );
                }

                assert!(
                    total.as_slice().unwrap()[4..8]
                        .iter()
                        .any(|&value| value > VALUE_TOLERANCE),
                    "absorbing layer should contain positive dissipation",
                );

                for &value in &total.as_slice().unwrap()[4..8] {
                    assert!(
                        value >= -VALUE_TOLERANCE,
                        "passive absorbing layer should not have negative dissipation: {value}",
                    );
                }
            });
        }
    }
}

#[test]
fn energy_density_total_identity_holds_through_second_order() {
    let stack = absorbing_two_layer_stack();
    let sampling = sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for side in [IncidentSide::Left, IncidentSide::Right] {
            for_each_backend!(evaluator, {
                let state = evaluator
                    .retain_second(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        polarisation,
                        Parameter::Spectral,
                    )
                    .unwrap();

                let point = state.project_point(&()).unwrap();

                let response = point
                    .excitation(side)
                    .unwrap()
                    .evaluate_energy_density(&sampling)
                    .unwrap();

                assert_total_is_sum(
                    response.value().electric(),
                    response.value().magnetic(),
                    response.value().total(),
                    VALUE_TOLERANCE,
                );

                assert_total_is_sum(
                    response.derivatives().first().electric(),
                    response.derivatives().first().magnetic(),
                    response.derivatives().first().total(),
                    FIRST_DERIVATIVE_TOLERANCE,
                );

                assert_total_is_sum(
                    response.derivatives().second().electric(),
                    response.derivatives().second().magnetic(),
                    response.derivatives().second().total(),
                    SECOND_DERIVATIVE_TOLERANCE,
                );
            });
        }
    }
}

#[test]
fn dissipation_density_total_identity_holds_through_second_order() {
    let stack = absorbing_two_layer_stack();
    let sampling = sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for side in [IncidentSide::Left, IncidentSide::Right] {
            for_each_backend!(evaluator, {
                let state = evaluator
                    .retain_second(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        polarisation,
                        Parameter::Spectral,
                    )
                    .unwrap();

                let point = state.project_point(&()).unwrap();

                let response = point
                    .excitation(side)
                    .unwrap()
                    .evaluate_dissipation_density(&sampling)
                    .unwrap();

                assert_total_is_sum(
                    response.value().electric(),
                    response.value().magnetic(),
                    response.value().total(),
                    VALUE_TOLERANCE,
                );

                assert_total_is_sum(
                    response.derivatives().first().electric(),
                    response.derivatives().first().magnetic(),
                    response.derivatives().first().total(),
                    FIRST_DERIVATIVE_TOLERANCE,
                );

                assert_total_is_sum(
                    response.derivatives().second().electric(),
                    response.derivatives().second().magnetic(),
                    response.derivatives().second().total(),
                    SECOND_DERIVATIVE_TOLERANCE,
                );
            });
        }
    }
}

#[test]
fn thickness_derivatives_survive_energy_and_dissipation_evaluation() {
    let stack = absorbing_two_layer_stack();
    let sampling = sampling();

    let parameter = Parameter::LayerThickness(FiniteLayerIndex::new(1));

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for side in [IncidentSide::Left, IncidentSide::Right] {
            for_each_backend!(evaluator, {
                let state = evaluator
                    .retain_first(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        polarisation,
                        parameter,
                    )
                    .unwrap();

                let point = state.project_point(&()).unwrap();
                let excitation = point.excitation(side).unwrap();

                let energy = excitation.evaluate_energy_density(&sampling).unwrap();

                let dissipation = excitation.evaluate_dissipation_density(&sampling).unwrap();

                assert_eq!(energy.derivatives().parameter(), parameter,);

                assert_eq!(dissipation.derivatives().parameter(), parameter,);

                assert_all_finite(energy.derivatives().first().electric());
                assert_all_finite(energy.derivatives().first().magnetic());
                assert_all_finite(energy.derivatives().first().total());

                assert_all_finite(dissipation.derivatives().first().electric());
                assert_all_finite(dissipation.derivatives().first().magnetic());
                assert_all_finite(dissipation.derivatives().first().total());

                assert_total_is_sum(
                    energy.derivatives().first().electric(),
                    energy.derivatives().first().magnetic(),
                    energy.derivatives().first().total(),
                    FIRST_DERIVATIVE_TOLERANCE,
                );

                assert_total_is_sum(
                    dissipation.derivatives().first().electric(),
                    dissipation.derivatives().first().magnetic(),
                    dissipation.derivatives().first().total(),
                    FIRST_DERIVATIVE_TOLERANCE,
                );
            });
        }
    }
}

#[test]
fn transfer_and_scatter_agree_on_energy_density() {
    let stack = absorbing_two_layer_stack();
    let sampling = sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for side in [IncidentSide::Left, IncidentSide::Right] {
            let scatter = PlaneWaveEvaluator::new(Scatter2::new())
                .retain_second(
                    scalar_real_input(2.5, 0.31),
                    &stack,
                    polarisation,
                    Parameter::Spectral,
                )
                .unwrap();

            let transfer = PlaneWaveEvaluator::new(Transfer2::new())
                .retain_second(
                    scalar_real_input(2.5, 0.31),
                    &stack,
                    polarisation,
                    Parameter::Spectral,
                )
                .unwrap();

            let scatter = scatter
                .project_point(&())
                .unwrap()
                .excitation(side)
                .unwrap()
                .evaluate_energy_density(&sampling)
                .unwrap();

            let transfer = transfer
                .project_point(&())
                .unwrap()
                .excitation(side)
                .unwrap()
                .evaluate_energy_density(&sampling)
                .unwrap();

            for (actual, expected) in [
                (scatter.value().electric(), transfer.value().electric()),
                (scatter.value().magnetic(), transfer.value().magnetic()),
                (scatter.value().total(), transfer.value().total()),
            ] {
                assert_real_array_close(actual, expected, VALUE_TOLERANCE);
            }

            for (actual, expected) in [
                (
                    scatter.derivatives().first().electric(),
                    transfer.derivatives().first().electric(),
                ),
                (
                    scatter.derivatives().first().magnetic(),
                    transfer.derivatives().first().magnetic(),
                ),
                (
                    scatter.derivatives().first().total(),
                    transfer.derivatives().first().total(),
                ),
            ] {
                assert_real_array_close(actual, expected, FIRST_DERIVATIVE_TOLERANCE);
            }

            for (actual, expected) in [
                (
                    scatter.derivatives().second().electric(),
                    transfer.derivatives().second().electric(),
                ),
                (
                    scatter.derivatives().second().magnetic(),
                    transfer.derivatives().second().magnetic(),
                ),
                (
                    scatter.derivatives().second().total(),
                    transfer.derivatives().second().total(),
                ),
            ] {
                assert_real_array_close(actual, expected, SECOND_DERIVATIVE_TOLERANCE);
            }
        }
    }
}

#[test]
fn transfer_and_scatter_agree_on_dissipation_density() {
    let stack = absorbing_two_layer_stack();
    let sampling = sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for side in [IncidentSide::Left, IncidentSide::Right] {
            let scatter = PlaneWaveEvaluator::new(Scatter2::new())
                .retain_second(
                    scalar_real_input(2.5, 0.31),
                    &stack,
                    polarisation,
                    Parameter::Spectral,
                )
                .unwrap();

            let transfer = PlaneWaveEvaluator::new(Transfer2::new())
                .retain_second(
                    scalar_real_input(2.5, 0.31),
                    &stack,
                    polarisation,
                    Parameter::Spectral,
                )
                .unwrap();

            let scatter = scatter
                .project_point(&())
                .unwrap()
                .excitation(side)
                .unwrap()
                .evaluate_dissipation_density(&sampling)
                .unwrap();

            let transfer = transfer
                .project_point(&())
                .unwrap()
                .excitation(side)
                .unwrap()
                .evaluate_dissipation_density(&sampling)
                .unwrap();

            for (actual, expected) in [
                (scatter.value().electric(), transfer.value().electric()),
                (scatter.value().magnetic(), transfer.value().magnetic()),
                (scatter.value().total(), transfer.value().total()),
            ] {
                assert_real_array_close(actual, expected, VALUE_TOLERANCE);
            }

            for (actual, expected) in [
                (
                    scatter.derivatives().first().electric(),
                    transfer.derivatives().first().electric(),
                ),
                (
                    scatter.derivatives().first().magnetic(),
                    transfer.derivatives().first().magnetic(),
                ),
                (
                    scatter.derivatives().first().total(),
                    transfer.derivatives().first().total(),
                ),
            ] {
                assert_real_array_close(actual, expected, FIRST_DERIVATIVE_TOLERANCE);
            }

            for (actual, expected) in [
                (
                    scatter.derivatives().second().electric(),
                    transfer.derivatives().second().electric(),
                ),
                (
                    scatter.derivatives().second().magnetic(),
                    transfer.derivatives().second().magnetic(),
                ),
                (
                    scatter.derivatives().second().total(),
                    transfer.derivatives().second().total(),
                ),
            ] {
                assert_real_array_close(actual, expected, SECOND_DERIVATIVE_TOLERANCE);
            }
        }
    }
}

const DISSIPATION_INTEGRATION_POINTS: usize = 2001;

fn integrate_uniform_density(density: &[f64], thickness: f64) -> f64 {
    assert!(density.len() >= 2);

    let dz = thickness / (density.len() - 1) as f64;

    let endpoints = 0.5 * (density[0] + density[density.len() - 1]);

    let interior: f64 = density[1..density.len() - 1].iter().sum();

    dz * (endpoints + interior)
}

fn physical_to_normalised_power_scale(
    exterior: &impl ExteriorContextProvider<Algebra = J0>,
    side: IncidentSide,
) -> f64 {
    let k0 = exterior.vacuum_angular_wavenumber().value()[()].re;

    let incident_admittance = match side {
        IncidentSide::Left => exterior.left_admittance().value()[()].re,
        IncidentSide::Right => exterior.right_admittance().value()[()].re,
    };

    /*
     * Cartesian field observables use the physical phasor convention
     *
     *     <S> = 1/2 Re(E × H*)
     *
     * while integrated layer observables are expressed relative to unit
     * incident power.
     */
    2.0 * k0 / incident_admittance
}

const DENSITY_INTEGRATION_POINTS: usize = 2001;
const DENSITY_INTEGRATION_TOLERANCE: f64 = 1.0e-6;

fn dense_layer_sampling(layer_count: usize) -> FieldSampling<f64> {
    let mut sampling = FieldSampling::new();

    for index in 0..layer_count {
        sampling = sampling.layer(index, LayerSampling::uniform(DENSITY_INTEGRATION_POINTS));
    }

    sampling
}

#[test]
fn interface_power_matches_cartesian_poynting_flux() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = absorbing_two_layer_stack();

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let point = state.project_point(&()).unwrap();

    let excitation = point.excitation(IncidentSide::Left).unwrap();

    let poynting = excitation
        .evaluate_time_averaged_poynting_vector(&FieldSampling::new().layer_interfaces())
        .unwrap();

    let interfaces = excitation.interface_power().unwrap();

    /*
     * layer_interfaces():
     *
     *   sample 0 = layer 0 left
     *   sample 1 = layer 0 right
     *   sample 2 = layer 1 left
     *   sample 3 = layer 1 right
     *
     * InterfacePower and the Cartesian Poynting-vector observable now use
     * the same physical time-averaged flux convention.
     */
    for (sz, flux) in [
        (
            poynting.value().z()[0],
            interfaces.value().get(0).unwrap().right_net_flux()[()],
        ),
        (
            poynting.value().z()[1],
            interfaces.value().get(1).unwrap().left_net_flux()[()],
        ),
        (
            poynting.value().z()[2],
            interfaces.value().get(1).unwrap().right_net_flux()[()],
        ),
        (
            poynting.value().z()[3],
            interfaces.value().get(2).unwrap().left_net_flux()[()],
        ),
    ] {
        assert_relative_eq!(
            flux,
            sz,
            epsilon = VALUE_TOLERANCE,
            max_relative = VALUE_TOLERANCE,
        );
    }
}

#[test]
fn integrated_dissipation_density_matches_layer_flux_loss() {
    let stack = absorbing_two_layer_stack();

    let sampling = FieldSampling::new()
        .layer(0, LayerSampling::uniform(DISSIPATION_INTEGRATION_POINTS))
        .layer(1, LayerSampling::uniform(DISSIPATION_INTEGRATION_POINTS));

    const INTEGRATION_TOLERANCE: f64 = 1.0e-6;

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for side in [IncidentSide::Left, IncidentSide::Right] {
            for_each_backend!(evaluator, {
                let state = evaluator
                    .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
                    .unwrap();

                let point = state.project_point(&()).unwrap();

                let excitation = point.excitation(side).expect("state should be projectable");

                let dissipation = excitation.evaluate_dissipation_density(&sampling).unwrap();

                let layer_power = excitation.layer_power().unwrap();

                let density = dissipation.value().total();

                assert_eq!(density.len(), 2 * DISSIPATION_INTEGRATION_POINTS,);

                for layer_index in 0..2 {
                    let start = layer_index * DISSIPATION_INTEGRATION_POINTS;

                    let end = start + DISSIPATION_INTEGRATION_POINTS;

                    let layer_density = &density
                        .as_slice()
                        .expect("sampled dissipation should be contiguous")
                        [start..end];

                    let thickness = stack
                        .layers_left_to_right()
                        .get(layer_index)
                        .expect("layer should exist")
                        .thickness()
                        .into_inner()
                        .into_canonical();

                    let integrated = integrate_uniform_density(layer_density, thickness);

                    let layer = layer_power
                        .value()
                        .get(FiniteLayerIndex::new(layer_index))
                        .expect("layer power should contain every finite layer");

                    let expected = layer.absorbed()[()];

                    assert_relative_eq!(
                        integrated,
                        expected,
                        epsilon = INTEGRATION_TOLERANCE,
                        max_relative = INTEGRATION_TOLERANCE,
                    );
                }
            });
        }
    }
}

#[test]
fn integrated_energy_density_matches_layer_energy() {
    let stack = absorbing_two_layer_stack();

    let layer_count = stack.layers_left_to_right().len();

    let sampling = dense_layer_sampling(layer_count);

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for side in [IncidentSide::Left, IncidentSide::Right] {
            for_each_backend!(evaluator, {
                let state = evaluator
                    .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
                    .unwrap();

                let point = state.project_point(&()).unwrap();

                let excitation = point.excitation(side).expect("state should be projectable");

                let density = excitation.evaluate_energy_density(&sampling).unwrap();

                let layer_energy = excitation.layer_energy_dispersive().unwrap();

                let electric_density = density.value().electric();

                let magnetic_density = density.value().magnetic();

                let total_density = density.value().total();

                assert_eq!(
                    total_density.len(),
                    layer_count * DENSITY_INTEGRATION_POINTS,
                );

                for layer_index in 0..layer_count {
                    let start = layer_index * DENSITY_INTEGRATION_POINTS;

                    let end = start + DENSITY_INTEGRATION_POINTS;

                    let thickness = stack
                        .layers_left_to_right()
                        .get(layer_index)
                        .expect("layer should exist")
                        .thickness()
                        .into_inner()
                        .into_canonical();

                    let electric_integrated = integrate_uniform_density(
                        &electric_density
                            .as_slice()
                            .expect("sampled energy should be contiguous")
                            [start..end],
                        thickness,
                    );

                    let magnetic_integrated = integrate_uniform_density(
                        &magnetic_density
                            .as_slice()
                            .expect("sampled energy should be contiguous")
                            [start..end],
                        thickness,
                    );

                    let total_integrated = integrate_uniform_density(
                        &total_density
                            .as_slice()
                            .expect("sampled energy should be contiguous")
                            [start..end],
                        thickness,
                    );

                    let expected = layer_energy
                        .value()
                        .get(FiniteLayerIndex::new(layer_index))
                        .expect("layer energy should contain every finite layer");

                    assert_relative_eq!(
                        electric_integrated,
                        expected.electric()[()],
                        epsilon = DENSITY_INTEGRATION_TOLERANCE,
                        max_relative = DENSITY_INTEGRATION_TOLERANCE,
                    );

                    assert_relative_eq!(
                        magnetic_integrated,
                        expected.magnetic()[()],
                        epsilon = DENSITY_INTEGRATION_TOLERANCE,
                        max_relative = DENSITY_INTEGRATION_TOLERANCE,
                    );

                    assert_relative_eq!(
                        total_integrated,
                        expected.total()[()],
                        epsilon = DENSITY_INTEGRATION_TOLERANCE,
                        max_relative = DENSITY_INTEGRATION_TOLERANCE,
                    );

                    assert_relative_eq!(
                        total_integrated,
                        electric_integrated + magnetic_integrated,
                        epsilon = DENSITY_INTEGRATION_TOLERANCE,
                        max_relative = DENSITY_INTEGRATION_TOLERANCE,
                    );
                }
            });
        }
    }
}
