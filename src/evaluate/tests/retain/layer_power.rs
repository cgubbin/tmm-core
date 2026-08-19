use approx::assert_relative_eq;
use ndarray::{ArrayBase, Ix0, OwnedRepr};

use crate::{
    IncidentSide, LayerPower, Parameter, Polarisation, RealAxisEvaluator,
    backend::{scatter2::Scatter2, transfer2::Transfer2},
    observable::Layers,
    parameter::FiniteLayerIndex,
    test_support::{
        assertions::assert_real_array_close,
        finite_difference::{
            FIRST_DERIVATIVE_TOLERANCE, SECOND_DERIVATIVE_TOLERANCE, VALUE_TOLERANCE,
        },
        planar::{scalar_real_input, two_layer_stack},
        stack::{absorbing_two_layer_stack, two_layer_stack_with_lossless_first_layer},
    },
};

type RealArray = ArrayBase<OwnedRepr<f64>, Ix0>;

fn scalar(value: &RealArray) -> f64 {
    value[()]
}

fn summed_absorption<R>(
    layers: &crate::observable::Layers<crate::observable::LayerPower<R>>,
    scalar: impl Fn(&R) -> f64,
) -> f64 {
    layers.iter().map(|layer| scalar(layer.absorbed())).sum()
}

#[test]
fn layer_power_returns_one_record_per_finite_layer() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let excitation = state
        .excitation(IncidentSide::Left)
        .expect("state should be projectable");

    let response = excitation.layer_power().unwrap();

    assert_eq!(response.value().len(), 2);
}

#[test]
fn lossless_layers_have_zero_absorption() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for side in [IncidentSide::Left, IncidentSide::Right] {
            let state = evaluator
                .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
                .unwrap();

            let excitation = state.excitation(side).expect("state should be projectable");

            let response = excitation.layer_power().unwrap();

            for layer in response.value().iter() {
                assert_relative_eq!(
                    scalar(layer.absorbed()),
                    0.0,
                    epsilon = VALUE_TOLERANCE,
                    max_relative = VALUE_TOLERANCE,
                );
            }
        }
    }
}

#[test]
fn layer_fluxes_are_taken_from_adjacent_interface_sides() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

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

    let interfaces = excitation.interface_power().unwrap();

    let layers = excitation.layer_power().unwrap();

    assert_eq!(layers.value().len() + 1, interfaces.value().len(),);

    for index in 0..layers.value().len() {
        let layer = layers.value().get(FiniteLayerIndex::new(index)).unwrap();

        let left_interface = interfaces.value().get(index).unwrap();

        let right_interface = interfaces.value().get(index + 1).unwrap();

        assert_real_array_close(
            layer.left_flux(),
            left_interface.right_net_flux(),
            VALUE_TOLERANCE,
        );

        assert_real_array_close(
            layer.right_flux(),
            right_interface.left_net_flux(),
            VALUE_TOLERANCE,
        );
    }
}

#[test]
fn summed_layer_absorption_matches_external_absorptance() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let stack = absorbing_two_layer_stack();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for side in [IncidentSide::Left, IncidentSide::Right] {
            let state = evaluator
                .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
                .unwrap();

            let excitation = state.excitation(side).expect("state should be projectable");

            let external = excitation.power();

            let layers = excitation.layer_power().unwrap();

            let actual: f64 = layers
                .value()
                .iter()
                .map(|layer| scalar(layer.absorbed()))
                .sum();

            let expected = scalar(external.value().absorptance());

            assert_relative_eq!(
                actual,
                expected,
                epsilon = VALUE_TOLERANCE,
                max_relative = VALUE_TOLERANCE,
            );

            assert!(
                actual > 0.0,
                "absorbing stack should have positive total absorption",
            );
        }
    }
}

#[test]
fn absorption_is_attributed_to_the_absorbing_layer() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack_with_lossless_first_layer(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let excitation = state
        .excitation(IncidentSide::Left)
        .expect("state should be projectable");

    let response = excitation.layer_power().unwrap();

    assert_eq!(response.value().len(), 2);

    let lossless = response.value().get(FiniteLayerIndex::new(0)).unwrap();
    let absorbing = response.value().get(FiniteLayerIndex::new(1)).unwrap();

    assert_relative_eq!(
        scalar(lossless.absorbed()),
        0.0,
        epsilon = VALUE_TOLERANCE,
        max_relative = VALUE_TOLERANCE,
    );

    assert!(
        scalar(absorbing.absorbed()) > 0.0,
        "absorbing layer should have positive resolved absorption",
    );
}

#[test]
fn right_incidence_uses_same_left_minus_right_absorption_definition() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &absorbing_two_layer_stack(),
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let excitation = state
        .excitation(IncidentSide::Right)
        .expect("state should be projectable");

    let response = excitation.layer_power().unwrap();

    for layer in response.value().iter() {
        let expected = scalar(layer.left_flux()) - scalar(layer.right_flux());

        assert_relative_eq!(
            scalar(layer.absorbed()),
            expected,
            epsilon = VALUE_TOLERANCE,
            max_relative = VALUE_TOLERANCE,
        );
    }

    let total: f64 = response
        .value()
        .iter()
        .map(|layer| scalar(layer.absorbed()))
        .sum();

    assert!(total > 0.0);
}

#[test]
fn summed_layer_absorption_derivative_matches_external_absorptance_derivative() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    for side in [IncidentSide::Left, IncidentSide::Right] {
        let state = evaluator
            .retain_first(
                scalar_real_input(2.5, 0.31),
                &absorbing_two_layer_stack(),
                Polarisation::TransverseElectric,
                Parameter::Spectral,
            )
            .unwrap();

        let excitation = state.excitation(side).expect("state should be projectable");

        let external = excitation.power();

        let layers = excitation.layer_power().unwrap();

        let actual: f64 = layers
            .derivatives()
            .first()
            .iter()
            .map(|layer| scalar(layer.absorbed()))
            .sum();

        let expected = scalar(external.derivatives().first().absorptance());

        assert_relative_eq!(
            actual,
            expected,
            epsilon = FIRST_DERIVATIVE_TOLERANCE,
            max_relative = FIRST_DERIVATIVE_TOLERANCE,
        );
    }
}

#[test]
fn thickness_derivative_of_resolved_absorption_sums_to_external_derivative() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let parameter = Parameter::LayerThickness(FiniteLayerIndex::new(1));

    let state = evaluator
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &absorbing_two_layer_stack(),
            Polarisation::TransverseMagnetic,
            parameter,
        )
        .unwrap();

    let excitation = state
        .excitation(IncidentSide::Left)
        .expect("state should be projectable");

    let external = excitation.power();

    let layers = excitation.layer_power().unwrap();

    assert_eq!(layers.derivatives().parameter(), parameter,);

    let actual: f64 = layers
        .derivatives()
        .first()
        .iter()
        .map(|layer| scalar(layer.absorbed()))
        .sum();

    let expected = scalar(external.derivatives().first().absorptance());

    assert_relative_eq!(
        actual,
        expected,
        epsilon = FIRST_DERIVATIVE_TOLERANCE,
        max_relative = FIRST_DERIVATIVE_TOLERANCE,
    );
}

#[test]
fn summed_second_layer_absorption_derivative_matches_external_response() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

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

    let external = excitation.power();

    let layers = excitation.layer_power().unwrap();

    let actual_first: f64 = layers
        .derivatives()
        .first()
        .iter()
        .map(|layer| scalar(layer.absorbed()))
        .sum();

    let actual_second: f64 = layers
        .derivatives()
        .second()
        .iter()
        .map(|layer| scalar(layer.absorbed()))
        .sum();

    assert_relative_eq!(
        actual_first,
        scalar(external.derivatives().first().absorptance(),),
        epsilon = FIRST_DERIVATIVE_TOLERANCE,
        max_relative = FIRST_DERIVATIVE_TOLERANCE,
    );

    assert_relative_eq!(
        actual_second,
        scalar(external.derivatives().second().absorptance(),),
        epsilon = SECOND_DERIVATIVE_TOLERANCE,
        max_relative = SECOND_DERIVATIVE_TOLERANCE,
    );
}

#[test]
fn bivariate_layer_absorption_derivatives_sum_to_external_derivatives() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let axis0 = Parameter::Spectral;
    let axis1 = Parameter::LayerThickness(FiniteLayerIndex::new(1));

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

    let external = excitation.power();

    let layers = excitation.layer_power().unwrap();

    let sum = |layers: &Layers<LayerPower<RealArray>>| {
        layers
            .iter()
            .map(|layer| scalar(layer.absorbed()))
            .sum::<f64>()
    };

    let actual_gradient = layers.derivatives().first();

    let expected_gradient = external.derivatives().first();

    assert_relative_eq!(
        sum(actual_gradient.axis0()),
        scalar(expected_gradient.axis0().absorptance()),
        epsilon = FIRST_DERIVATIVE_TOLERANCE,
        max_relative = FIRST_DERIVATIVE_TOLERANCE,
    );

    assert_relative_eq!(
        sum(actual_gradient.axis1()),
        scalar(expected_gradient.axis1().absorptance()),
        epsilon = FIRST_DERIVATIVE_TOLERANCE,
        max_relative = FIRST_DERIVATIVE_TOLERANCE,
    );

    let actual_hessian = layers.derivatives().second();

    let expected_hessian = external.derivatives().second();

    assert_relative_eq!(
        sum(actual_hessian.axis0_axis0()),
        scalar(expected_hessian.axis0_axis0().absorptance(),),
        epsilon = SECOND_DERIVATIVE_TOLERANCE,
        max_relative = SECOND_DERIVATIVE_TOLERANCE,
    );

    assert_relative_eq!(
        sum(actual_hessian.axis0_axis1()),
        scalar(expected_hessian.axis0_axis1().absorptance(),),
        epsilon = SECOND_DERIVATIVE_TOLERANCE,
        max_relative = SECOND_DERIVATIVE_TOLERANCE,
    );

    assert_relative_eq!(
        sum(actual_hessian.axis1_axis1()),
        scalar(expected_hessian.axis1_axis1().absorptance(),),
        epsilon = SECOND_DERIVATIVE_TOLERANCE,
        max_relative = SECOND_DERIVATIVE_TOLERANCE,
    );
}

#[test]
fn transfer_backend_projects_layer_power() {
    let evaluator = RealAxisEvaluator::new(Transfer2::new());

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

    let external = excitation.power();

    let layers = excitation.layer_power().unwrap();

    assert_eq!(layers.value().len(), 2);

    let total: f64 = layers
        .value()
        .iter()
        .map(|layer| scalar(layer.absorbed()))
        .sum();

    assert_relative_eq!(
        total,
        scalar(external.value().absorptance()),
        epsilon = VALUE_TOLERANCE,
        max_relative = VALUE_TOLERANCE,
    );
}
