use approx::assert_relative_eq;
use ndarray::{ArrayBase, Ix0, OwnedRepr};

use crate::{
    IncidentSide, Parameter, PlaneWaveEvaluator, Polarisation,
    backend::{ExteriorContextProvider, scatter2::Scatter2, transfer2::Transfer2},
    observable::{LayerDissipation, LayerPower, Layers},
    parameter::FiniteLayerIndex,
    test_support::{
        planar::{scalar_real_input, two_layer_stack},
        stack::{
            absorbing_two_layer_stack, asymmetric_absorbing_single_layer_stack,
            electric_loss_stack, magnetic_loss_stack,
        },
    },
};

type RealArray = ArrayBase<OwnedRepr<f64>, Ix0>;

const VALUE_TOLERANCE: f64 = 2.0e-10;
const FIRST_TOLERANCE: f64 = 2.0e-8;
const SECOND_TOLERANCE: f64 = 2.0e-5;

fn scalar(value: &RealArray) -> f64 {
    value[()]
}

fn assert_real_close(actual: &RealArray, expected: &RealArray, tolerance: f64) {
    assert_relative_eq!(
        scalar(actual),
        scalar(expected),
        epsilon = tolerance,
        max_relative = tolerance,
    );
}

fn assert_real_zero(actual: &RealArray, tolerance: f64) {
    assert_relative_eq!(
        scalar(actual),
        0.0,
        epsilon = tolerance,
        max_relative = tolerance,
    );
}

fn assert_layer_dissipation_matches_power(
    dissipation: &Layers<LayerDissipation<RealArray>>,
    power: &Layers<LayerPower<RealArray>>,
    tolerance: f64,
) {
    assert_eq!(dissipation.len(), power.len());

    for index in 0..dissipation.len() {
        let dissipation = dissipation.get(FiniteLayerIndex(index)).unwrap();

        let power = power.get(FiniteLayerIndex(index)).unwrap();

        assert_real_close(dissipation.total(), power.absorbed(), tolerance);

        assert_relative_eq!(
            scalar(dissipation.total()),
            scalar(dissipation.electric()) + scalar(dissipation.magnetic()),
            epsilon = tolerance,
            max_relative = tolerance,
        );
    }
}

fn summed_dissipation(layers: &Layers<LayerDissipation<RealArray>>) -> f64 {
    layers.iter().map(|layer| scalar(layer.total())).sum()
}

#[test]
fn layer_dissipation_returns_one_record_per_finite_layer() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &absorbing_two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let excitation = state
        .excitation(IncidentSide::Left)
        .expect("state should be projectable");

    let response = excitation.layer_dissipation().unwrap();

    assert_eq!(response.value().len(), 2);
}

#[test]
fn lossless_stack_has_zero_layer_dissipation() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for incident_side in [IncidentSide::Left, IncidentSide::Right] {
            let state = evaluator
                .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
                .unwrap();

            let excitation = state
                .excitation(incident_side)
                .expect("state should be projectable");

            let response = excitation.layer_dissipation().unwrap();

            for layer in response.value().iter() {
                assert_real_zero(layer.electric(), VALUE_TOLERANCE);

                assert_real_zero(layer.magnetic(), VALUE_TOLERANCE);

                assert_real_zero(layer.total(), VALUE_TOLERANCE);
            }
        }
    }
}

#[test]
fn layer_dissipation_matches_layer_power_flux_loss() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = absorbing_two_layer_stack();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for incident_side in [IncidentSide::Left, IncidentSide::Right] {
            let state = evaluator
                .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
                .unwrap();

            let excitation = state
                .excitation(incident_side)
                .expect("state should be projectable");

            let dissipation = excitation.layer_dissipation().unwrap();

            let power = excitation.layer_power().unwrap();

            assert_layer_dissipation_matches_power(
                dissipation.value(),
                power.value(),
                VALUE_TOLERANCE,
            );
        }
    }
}

#[test]
fn summed_layer_dissipation_matches_external_absorptance() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = absorbing_two_layer_stack();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for incident_side in [IncidentSide::Left, IncidentSide::Right] {
            let state = evaluator
                .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
                .unwrap();

            let excitation = state
                .excitation(incident_side)
                .expect("state should be projectable");

            let dissipation = excitation.layer_dissipation().unwrap();

            let external = excitation.power();

            assert_relative_eq!(
                summed_dissipation(dissipation.value(),),
                scalar(external.value().absorptance(),),
                epsilon = VALUE_TOLERANCE,
                max_relative = VALUE_TOLERANCE,
            );
        }
    }
}

#[test]
fn electric_loss_is_attributed_to_electric_dissipation() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for incident_side in [IncidentSide::Left, IncidentSide::Right] {
            let state = evaluator
                .retain(
                    scalar_real_input(2.5, 0.31),
                    &electric_loss_stack(),
                    polarisation,
                )
                .unwrap();

            let excitation = state
                .excitation(incident_side)
                .expect("state should be projectable");

            let response = excitation.layer_dissipation().unwrap();

            let layer = response.value().get(FiniteLayerIndex(0)).unwrap();

            assert_real_zero(layer.magnetic(), VALUE_TOLERANCE);

            assert_real_close(layer.electric(), layer.total(), VALUE_TOLERANCE);

            assert!(
                scalar(layer.total()) > 0.0,
                "passive electric loss should produce positive dissipation",
            );
        }
    }
}

#[test]
fn magnetic_loss_is_attributed_to_magnetic_dissipation() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for incident_side in [IncidentSide::Left, IncidentSide::Right] {
            let state = evaluator
                .retain(
                    scalar_real_input(2.5, 0.31),
                    &magnetic_loss_stack(),
                    polarisation,
                )
                .unwrap();

            let excitation = state
                .excitation(incident_side)
                .expect("state should be projectable");

            let response = excitation.layer_dissipation().unwrap();

            let layer = response.value().get(FiniteLayerIndex(0)).unwrap();

            assert_real_zero(layer.electric(), VALUE_TOLERANCE);

            assert_real_close(layer.magnetic(), layer.total(), VALUE_TOLERANCE);

            assert!(
                scalar(layer.total()) > 0.0,
                "passive magnetic loss should produce positive dissipation",
            );
        }
    }
}

#[test]
fn first_layer_dissipation_derivative_matches_layer_power_derivative() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    for incident_side in [IncidentSide::Left, IncidentSide::Right] {
        let state = evaluator
            .retain_first(
                scalar_real_input(2.5, 0.31),
                &absorbing_two_layer_stack(),
                Polarisation::TransverseElectric,
                Parameter::Spectral,
            )
            .unwrap();

        let excitation = state
            .excitation(incident_side)
            .expect("state should be projectable");

        let dissipation = excitation.layer_dissipation().unwrap();

        let power = excitation.layer_power().unwrap();

        assert_layer_dissipation_matches_power(dissipation.value(), power.value(), VALUE_TOLERANCE);

        assert_layer_dissipation_matches_power(
            dissipation.derivatives().first(),
            power.derivatives().first(),
            FIRST_TOLERANCE,
        );
    }
}

#[test]
fn thickness_dissipation_derivative_matches_flux_loss_derivative() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let parameter = Parameter::LayerThickness(FiniteLayerIndex(1));

    let state = evaluator
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &absorbing_two_layer_stack(),
            Polarisation::TransverseMagnetic,
            parameter,
        )
        .unwrap();

    let excitation = state
        .excitation(IncidentSide::Right)
        .expect("state should be projectable");

    let dissipation = excitation.layer_dissipation().unwrap();

    let power = excitation.layer_power().unwrap();

    assert_eq!(dissipation.derivatives().parameter(), parameter,);

    assert_layer_dissipation_matches_power(
        dissipation.derivatives().first(),
        power.derivatives().first(),
        FIRST_TOLERANCE,
    );
}

#[test]
fn second_layer_dissipation_derivatives_match_flux_loss_derivatives() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain_second(
            scalar_real_input(2.5, 0.31),
            &absorbing_two_layer_stack(),
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap();

    let excitation = state
        .excitation(IncidentSide::Left)
        .expect("state should be projectable");

    let dissipation = excitation.layer_dissipation().unwrap();

    let power = excitation.layer_power().unwrap();

    assert_layer_dissipation_matches_power(dissipation.value(), power.value(), VALUE_TOLERANCE);

    assert_layer_dissipation_matches_power(
        dissipation.derivatives().first(),
        power.derivatives().first(),
        FIRST_TOLERANCE,
    );

    assert_layer_dissipation_matches_power(
        dissipation.derivatives().second(),
        power.derivatives().second(),
        SECOND_TOLERANCE,
    );
}

#[test]
fn bivariate_layer_dissipation_matches_flux_loss_on_all_branches() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let axis0 = Parameter::Spectral;
    let axis1 = Parameter::LayerThickness(FiniteLayerIndex(1));

    let state = evaluator
        .retain_bivariate_second(
            scalar_real_input(2.5, 0.31),
            &absorbing_two_layer_stack(),
            Polarisation::TransverseMagnetic,
            axis0,
            axis1,
        )
        .unwrap();

    let excitation = state
        .excitation(IncidentSide::Right)
        .expect("state should be projectable");

    let dissipation = excitation.layer_dissipation().unwrap();

    let power = excitation.layer_power().unwrap();

    assert_layer_dissipation_matches_power(dissipation.value(), power.value(), VALUE_TOLERANCE);

    let dissipation_gradient = dissipation.derivatives().first();

    let power_gradient = power.derivatives().first();

    assert_layer_dissipation_matches_power(
        dissipation_gradient.axis0(),
        power_gradient.axis0(),
        FIRST_TOLERANCE,
    );

    assert_layer_dissipation_matches_power(
        dissipation_gradient.axis1(),
        power_gradient.axis1(),
        FIRST_TOLERANCE,
    );

    let dissipation_hessian = dissipation.derivatives().second();

    let power_hessian = power.derivatives().second();

    assert_layer_dissipation_matches_power(
        dissipation_hessian.axis0_axis0(),
        power_hessian.axis0_axis0(),
        SECOND_TOLERANCE,
    );

    assert_layer_dissipation_matches_power(
        dissipation_hessian.axis0_axis1(),
        power_hessian.axis0_axis1(),
        SECOND_TOLERANCE,
    );

    assert_layer_dissipation_matches_power(
        dissipation_hessian.axis1_axis1(),
        power_hessian.axis1_axis1(),
        SECOND_TOLERANCE,
    );
}

#[test]
fn transfer_backend_layer_dissipation_matches_layer_power() {
    let evaluator = PlaneWaveEvaluator::new(Transfer2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &absorbing_two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let excitation = state
        .excitation(IncidentSide::Left)
        .expect("state should be projectable");

    let dissipation = excitation.layer_dissipation().unwrap();

    let power = excitation.layer_power().unwrap();

    assert_layer_dissipation_matches_power(dissipation.value(), power.value(), VALUE_TOLERANCE);
}
