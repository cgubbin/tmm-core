use approx::assert_relative_eq;
use ndarray::Array0;

use crate::{
    FiniteLayerIndex, IncidentSide, Parameter, PlaneWavePower, Polarisation, RealAxisEvaluator,
    backend::{scatter2::Scatter2, transfer2::Transfer2},
    projection::PointProjectionError,
    test_support::{
        planar::{sampled_real_input, scalar_real_input, two_layer_stack},
        stack::differentiable_lossless_two_layer_stack,
    },
};

const VALUE_TOLERANCE: f64 = 1.0e-12;
const DERIVATIVE_TOLERANCE: f64 = 1.0e-10;

fn assert_power_equivalent(
    projected: &PlaneWavePower<Array0<f64>>,
    direct: &PlaneWavePower<Array0<f64>>,
    tolerance: f64,
) {
    assert_relative_eq!(
        projected.reflectance()[()],
        direct.reflectance()[()],
        epsilon = tolerance,
        max_relative = tolerance,
    );

    assert_relative_eq!(
        projected.transmittance()[()],
        direct.transmittance()[()],
        epsilon = tolerance,
        max_relative = tolerance,
    );

    assert_relative_eq!(
        projected.absorptance()[()],
        direct.absorptance()[()],
        epsilon = tolerance,
        max_relative = tolerance,
    );
}

#[test]
fn projected_scatter_state_matches_direct_scalar_evaluation() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let spectral = [2.1, 2.3, 2.5];

    let parallel = [0.21, 0.27, 0.31];

    let batch = evaluator
        .retain(
            sampled_real_input(&spectral, &parallel),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let projected = batch.project_point(&1).unwrap();

    let direct = evaluator
        .retain(
            scalar_real_input(spectral[1], parallel[1]),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    for side in [IncidentSide::Left, IncidentSide::Right] {
        let projected = projected.excitation(side).unwrap().power();
        let direct = direct.excitation(side).unwrap().power();

        assert_power_equivalent(projected.value(), direct.value(), VALUE_TOLERANCE);
    }
}

#[test]
fn projected_transfer_state_matches_direct_scalar_evaluation() {
    let evaluator = RealAxisEvaluator::new(Transfer2::new());

    let stack = two_layer_stack();

    let spectral = [2.1, 2.3, 2.5];

    let parallel = [0.21, 0.27, 0.31];

    let batch = evaluator
        .retain(
            sampled_real_input(&spectral, &parallel),
            &stack,
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let projected = batch.project_point(&2).unwrap();

    let direct = evaluator
        .retain(
            scalar_real_input(spectral[2], parallel[2]),
            &stack,
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    for side in [IncidentSide::Left, IncidentSide::Right] {
        let projected = projected.excitation(side).unwrap().power();
        let direct = direct.excitation(side).unwrap().power();

        assert_power_equivalent(projected.value(), direct.value(), VALUE_TOLERANCE);
    }
}

#[test]
fn projected_first_order_state_matches_direct_scalar_derivatives() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let stack = differentiable_lossless_two_layer_stack();

    let spectral = [2.1, 2.3, 2.5];

    let parallel = [0.21, 0.27, 0.31];

    let batch = evaluator
        .retain_first(
            sampled_real_input(&spectral, &parallel),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap();

    let projected = batch.project_point(&1).unwrap();

    let direct = evaluator
        .retain_first(
            scalar_real_input(spectral[1], parallel[1]),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap();

    for side in [IncidentSide::Left, IncidentSide::Right] {
        let projected = projected.excitation(side).unwrap().power();
        let direct = direct.excitation(side).unwrap().power();
        assert_power_equivalent(projected.value(), direct.value(), VALUE_TOLERANCE);
        assert_power_equivalent(projected.first(), direct.first(), DERIVATIVE_TOLERANCE);
    }
}

#[test]
fn projected_second_order_state_matches_direct_scalar_derivatives() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let stack = differentiable_lossless_two_layer_stack();

    let spectral = [2.1, 2.3, 2.5];

    let parallel = [0.21, 0.27, 0.31];

    let batch = evaluator
        .retain_second(
            sampled_real_input(&spectral, &parallel),
            &stack,
            Polarisation::TransverseMagnetic,
            Parameter::Spectral,
        )
        .unwrap();

    let projected = batch.project_point(&2).unwrap();

    let direct = evaluator
        .retain_second(
            scalar_real_input(spectral[2], parallel[2]),
            &stack,
            Polarisation::TransverseMagnetic,
            Parameter::Spectral,
        )
        .unwrap();

    for side in [IncidentSide::Left, IncidentSide::Right] {
        let projected = projected.excitation(side).unwrap().power();
        let direct = direct.excitation(side).unwrap().power();
        assert_power_equivalent(projected.value(), direct.value(), VALUE_TOLERANCE);
        assert_power_equivalent(projected.first(), direct.first(), DERIVATIVE_TOLERANCE);
        assert_power_equivalent(projected.second(), direct.second(), DERIVATIVE_TOLERANCE);
    }
}

#[test]
fn projection_preserves_polarisation() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            sampled_real_input(&[2.1, 2.3], &[0.21, 0.27]),
            &two_layer_stack(),
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let point = state.project_point(&0).unwrap();

    assert_eq!(point.polarisation(), Polarisation::TransverseMagnetic,);
}

#[test]
fn projection_preserves_mapping_and_constraint() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain_first(
            sampled_real_input(&[2.1, 2.3], &[0.21, 0.27]),
            &differentiable_lossless_two_layer_stack(),
            Polarisation::TransverseElectric,
            Parameter::LayerThickness(FiniteLayerIndex::new(0)),
        )
        .unwrap();

    let point = state.project_point(&1).unwrap();

    assert_eq!(point.context().mapping(), state.context().mapping(),);

    assert_eq!(
        point.context().projection_constraint(),
        state.context().projection_constraint(),
    );
}

#[test]
fn invalid_state_projection_reports_batch_shape() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            sampled_real_input(&[2.1, 2.3], &[0.21, 0.27]),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let error = state.project_point(&2).unwrap_err();

    assert_eq!(error, PointProjectionError::OutOfBounds { shape: vec![2] },);
}

#[test]
fn external_power_commutes_with_point_projection() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            sampled_real_input(&[2.1, 2.3, 2.5], &[0.21, 0.27, 0.31]),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let batch = state.excitation(IncidentSide::Left).unwrap().power();

    let point_state = state.project_point(&1).unwrap();
    let point = point_state.excitation(IncidentSide::Left).unwrap().power();

    assert_relative_eq!(
        point.value().reflectance()[()],
        batch.value().reflectance()[1],
        epsilon = VALUE_TOLERANCE,
        max_relative = VALUE_TOLERANCE,
    );

    assert_relative_eq!(
        point.value().transmittance()[()],
        batch.value().transmittance()[1],
        epsilon = VALUE_TOLERANCE,
        max_relative = VALUE_TOLERANCE,
    );

    assert_relative_eq!(
        point.value().absorptance()[()],
        batch.value().absorptance()[1],
        epsilon = VALUE_TOLERANCE,
        max_relative = VALUE_TOLERANCE,
    );
}

#[test]
fn interface_power_commutes_with_point_projection() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            sampled_real_input(&[2.1, 2.3, 2.5], &[0.21, 0.27, 0.31]),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let batch = state
        .excitation(IncidentSide::Left)
        .unwrap()
        .interface_power()
        .unwrap();

    let point_state = state.project_point(&1).unwrap();

    let point = point_state
        .excitation(IncidentSide::Left)
        .unwrap()
        .interface_power()
        .unwrap();

    assert_eq!(point.value().len(), batch.value().len());

    for (point, batch) in point.value().iter().zip(batch.value().iter()) {
        assert_relative_eq!(
            point.left_net_flux()[()],
            batch.left_net_flux()[1],
            epsilon = VALUE_TOLERANCE,
            max_relative = VALUE_TOLERANCE,
        );

        assert_relative_eq!(
            point.right_net_flux()[()],
            batch.right_net_flux()[1],
            epsilon = VALUE_TOLERANCE,
            max_relative = VALUE_TOLERANCE,
        );
    }
}

#[test]
fn layer_power_commutes_with_point_projection() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            sampled_real_input(&[2.1, 2.3, 2.5], &[0.21, 0.27, 0.31]),
            &two_layer_stack(),
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let batch = state
        .excitation(IncidentSide::Right)
        .unwrap()
        .layer_power()
        .unwrap();

    let point_state = state.project_point(&2).unwrap();

    let point = point_state
        .excitation(IncidentSide::Right)
        .unwrap()
        .layer_power()
        .unwrap();

    assert_eq!(point.value().len(), batch.value().len());

    for (point, batch) in point.value().iter().zip(batch.value().iter()) {
        assert_relative_eq!(
            point.left_flux()[()],
            batch.left_flux()[2],
            epsilon = VALUE_TOLERANCE,
            max_relative = VALUE_TOLERANCE,
        );

        assert_relative_eq!(
            point.right_flux()[()],
            batch.right_flux()[2],
            epsilon = VALUE_TOLERANCE,
            max_relative = VALUE_TOLERANCE,
        );

        assert_relative_eq!(
            point.absorbed()[()],
            batch.absorbed()[2],
            epsilon = VALUE_TOLERANCE,
            max_relative = VALUE_TOLERANCE,
        );
    }
}
