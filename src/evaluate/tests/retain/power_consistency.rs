use approx::assert_relative_eq;
use ndarray::{ArrayBase, Ix0, OwnedRepr};
use num_complex::Complex64;

use crate::{
    Constant, IncidentSide, MaterialStack, Parameter, PlaneWaveEvaluator, Polarisation, Stack,
    Thickness,
    backend::{scatter2::Scatter2, transfer2::Transfer2},
    observable::{InterfacePower, Interfaces},
    parameter::FiniteLayerIndex,
    test_support::planar::{scalar_real_input, two_layer_stack},
};

type RealArray = ArrayBase<OwnedRepr<f64>, Ix0>;

const VALUE_TOLERANCE: f64 = 1.0e-11;
const FIRST_TOLERANCE: f64 = 1.0e-9;
const SECOND_TOLERANCE: f64 = 1.0e-7;

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

fn assert_real_value_close(actual: &RealArray, expected: f64, tolerance: f64) {
    assert_relative_eq!(
        scalar(actual),
        expected,
        epsilon = tolerance,
        max_relative = tolerance,
    );
}

fn assert_real_zero(actual: &RealArray, tolerance: f64) {
    assert_real_value_close(actual, 0.0, tolerance);
}

fn assert_net_flux_continuity(interfaces: &Interfaces<InterfacePower<RealArray>>, tolerance: f64) {
    assert!(
        !interfaces.is_empty(),
        "a valid stack must contain at least one interface",
    );

    for (index, interface) in interfaces.iter().enumerate() {
        let left = scalar(interface.left_net_flux());
        let right = scalar(interface.right_net_flux());

        assert_relative_eq!(left, right, epsilon = tolerance, max_relative = tolerance,);

        assert!(
            (left - right).abs() <= tolerance * (1.0 + left.abs().max(right.abs())),
            "net-flux discontinuity at interface {index}: \
             left={left}, right={right}",
        );
    }
}

#[test]
fn left_incidence_interface_power_matches_external_power() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .expect("retained evaluation should succeed");

    let external = state.power(IncidentSide::Left);

    let interfaces = state
        .interface_power(IncidentSide::Left)
        .expect("interface power should project");

    let first = interfaces
        .value()
        .first()
        .expect("left exterior interface should exist");

    let last = interfaces
        .value()
        .last()
        .expect("right exterior interface should exist");

    let reflectance = scalar(external.value().reflectance());

    let transmittance = scalar(external.value().transmittance());

    assert_real_value_close(first.left().forward_flux(), 1.0, VALUE_TOLERANCE);

    assert_real_value_close(first.left().backward_flux(), -reflectance, VALUE_TOLERANCE);

    assert_real_value_close(first.left().net_flux(), 1.0 - reflectance, VALUE_TOLERANCE);

    assert_real_value_close(last.right().forward_flux(), transmittance, VALUE_TOLERANCE);

    assert_real_zero(last.right().backward_flux(), VALUE_TOLERANCE);

    assert_real_value_close(last.right().net_flux(), transmittance, VALUE_TOLERANCE);
}

#[test]
fn right_incidence_interface_power_matches_external_power() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseMagnetic,
        )
        .expect("retained evaluation should succeed");

    let external = state.power(IncidentSide::Right);

    let interfaces = state
        .interface_power(IncidentSide::Right)
        .expect("interface power should project");

    let first = interfaces
        .value()
        .first()
        .expect("left exterior interface should exist");

    let last = interfaces
        .value()
        .last()
        .expect("right exterior interface should exist");

    let reflectance = scalar(external.value().reflectance());

    let transmittance = scalar(external.value().transmittance());

    assert_real_zero(first.left().forward_flux(), VALUE_TOLERANCE);

    assert_real_value_close(
        first.left().backward_flux(),
        -transmittance,
        VALUE_TOLERANCE,
    );

    assert_real_value_close(first.left().net_flux(), -transmittance, VALUE_TOLERANCE);

    assert_real_value_close(last.right().forward_flux(), reflectance, VALUE_TOLERANCE);

    assert_real_value_close(last.right().backward_flux(), -1.0, VALUE_TOLERANCE);

    assert_real_value_close(last.right().net_flux(), reflectance - 1.0, VALUE_TOLERANCE);
}

#[test]
fn net_flux_is_continuous_across_every_interface() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for incident_side in [IncidentSide::Left, IncidentSide::Right] {
            let interfaces = evaluator
                .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
                .unwrap()
                .interface_power(incident_side)
                .unwrap();

            assert_net_flux_continuity(interfaces.value(), VALUE_TOLERANCE);
        }
    }
}

#[test]
fn exterior_flux_drop_matches_absorptance_from_both_sides() {
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

            let external = state.power(incident_side);

            let interfaces = state.interface_power(incident_side).unwrap();

            let first = interfaces.value().first().unwrap();
            let last = interfaces.value().last().unwrap();

            let flux_drop = scalar(first.left_net_flux()) - scalar(last.right_net_flux());

            let expected = scalar(external.value().absorptance());

            dbg!(&expected, &flux_drop);

            assert_relative_eq!(
                flux_drop,
                expected,
                epsilon = VALUE_TOLERANCE,
                max_relative = VALUE_TOLERANCE,
            );
        }
    }
}

#[test]
fn first_interface_power_derivative_is_continuous() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let response = evaluator
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap()
        .interface_power(IncidentSide::Left)
        .unwrap();

    assert_eq!(response.derivatives().parameter(), Parameter::Spectral,);

    assert_net_flux_continuity(response.value(), VALUE_TOLERANCE);

    assert_net_flux_continuity(response.derivatives().first(), FIRST_TOLERANCE);
}

#[test]
fn thickness_interface_power_derivative_is_continuous() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let parameter = Parameter::LayerThickness(FiniteLayerIndex(1));

    let response = evaluator
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseMagnetic,
            parameter,
        )
        .unwrap()
        .interface_power(IncidentSide::Right)
        .unwrap();

    assert_eq!(response.derivatives().parameter(), parameter,);

    assert_net_flux_continuity(response.value(), VALUE_TOLERANCE);

    assert_net_flux_continuity(response.derivatives().first(), FIRST_TOLERANCE);
}

#[test]
fn second_interface_power_derivatives_are_continuous() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let response = evaluator
        .retain_second(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap()
        .interface_power(IncidentSide::Left)
        .unwrap();

    assert_net_flux_continuity(response.value(), VALUE_TOLERANCE);

    assert_net_flux_continuity(response.derivatives().first(), FIRST_TOLERANCE);

    assert_net_flux_continuity(response.derivatives().second(), SECOND_TOLERANCE);
}

#[test]
fn bivariate_interface_power_derivatives_are_continuous() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let axis0 = Parameter::Spectral;
    let axis1 = Parameter::LayerThickness(FiniteLayerIndex(1));

    let response = evaluator
        .retain_bivariate_second(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseMagnetic,
            axis0,
            axis1,
        )
        .unwrap()
        .interface_power(IncidentSide::Right)
        .unwrap();

    assert_eq!(response.derivatives().parameters(), [axis0, axis1],);

    assert_net_flux_continuity(response.value(), VALUE_TOLERANCE);

    let gradient = response.derivatives().first();

    assert_net_flux_continuity(gradient.axis0(), FIRST_TOLERANCE);

    assert_net_flux_continuity(gradient.axis1(), FIRST_TOLERANCE);

    let hessian = response.derivatives().second();

    assert_net_flux_continuity(hessian.axis0_axis0(), SECOND_TOLERANCE);

    assert_net_flux_continuity(hessian.axis0_axis1(), SECOND_TOLERANCE);

    assert_net_flux_continuity(hessian.axis1_axis1(), SECOND_TOLERANCE);
}

#[test]
fn interface_exterior_flux_derivative_matches_external_power_derivative() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap();

    let external = state.power(IncidentSide::Left);

    let interfaces = state.interface_power(IncidentSide::Left).unwrap();

    let first_derivative = interfaces.derivatives().first();

    let first = first_derivative.first().unwrap();
    let last = first_derivative.last().unwrap();

    let external_first = external.derivatives().first();

    // d(1 - R)/dp = -dR/dp.
    assert_real_value_close(
        first.left_net_flux(),
        -scalar(external_first.reflectance()),
        FIRST_TOLERANCE,
    );

    // dT/dp.
    assert_real_close(
        last.right_net_flux(),
        external_first.transmittance(),
        FIRST_TOLERANCE,
    );
}

#[test]
fn transfer_backend_projects_interface_power() {
    let evaluator = PlaneWaveEvaluator::new(Transfer2::new());

    let response = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap()
        .interface_power(IncidentSide::Left)
        .unwrap();

    assert_eq!(response.value().len(), 3);

    assert_net_flux_continuity(response.value(), VALUE_TOLERANCE);
}

// fn absorbing_single_layer_stack() -> MaterialStack<Complex64> {
//     Stack::from_materials(Constant::vacuum(), Constant::vacuum())
//         .material_layer(
//             tmm_materials::Drude::new(1.0, 15000.0, 20.0).unwrap(),
//             Thickness::micrometres(100.0),
//         )
//         .finalise()
// }

// fn absorbing_two_layer_stack() -> MaterialStack<Complex64> {
//     Stack::from_materials(Constant::vacuum(), Constant::dielectric(1.7))
//         .material_layer(
//             tmm_materials::Drude::new(1.0, 15000.0, 20.0).unwrap(),
//             Thickness::micrometres(0.35),
//         )
//         .material_layer(
//             tmm_materials::Drude::new(1.0, 30000.0, 30.0).unwrap(),
//             Thickness::micrometres(0.7),
//         )
//         .finalise()
// }

// #[test]
// fn absorbing_stack_net_flux_is_continuous_across_interfaces() {
//     let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

//     let stack = absorbing_two_layer_stack();

//     for polarisation in [
//         Polarisation::TransverseElectric,
//         Polarisation::TransverseMagnetic,
//     ] {
//         for incident_side in [IncidentSide::Left, IncidentSide::Right] {
//             let response = evaluator
//                 .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
//                 .unwrap()
//                 .interface_power(incident_side)
//                 .unwrap();

//             assert_net_flux_continuity(response.value(), VALUE_TOLERANCE);
//         }
//     }
// }
