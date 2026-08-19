use approx::assert_relative_eq;

use crate::{
    IncidentSide, Parameter, Polarisation, RealAxisEvaluator,
    backend::scatter2::Scatter2,
    test_support::{planar::scalar_real_input, stack::differentiable_lossless_two_layer_stack},
};

const VALUE_TOLERANCE: f64 = 2.0e-10;
const FIRST_TOLERANCE: f64 = 2.0e-8;

#[test]
fn public_nondispersive_layer_participation_returns_one_record_per_layer() {
    let state = RealAxisEvaluator::new(Scatter2::new())
        .retain(
            scalar_real_input(2.5, 0.31),
            &differentiable_lossless_two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let participation = state
        .excitation(IncidentSide::Left)
        .expect("state should be projectable")
        .layer_participation_nondispersive()
        .unwrap();

    assert_eq!(participation.value().len(), 2);
}

#[test]
fn public_nondispersive_layer_participation_sums_to_unity() {
    let state = RealAxisEvaluator::new(Scatter2::new())
        .retain(
            scalar_real_input(2.5, 0.31),
            &differentiable_lossless_two_layer_stack(),
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let participation = state
        .excitation(IncidentSide::Right)
        .expect("state should be projectable")
        .layer_participation_nondispersive()
        .unwrap();

    let total: f64 = participation
        .value()
        .iter()
        .map(|layer| layer.total()[()])
        .sum();

    assert_relative_eq!(
        total,
        1.0,
        epsilon = VALUE_TOLERANCE,
        max_relative = VALUE_TOLERANCE,
    );
}

#[test]
fn public_nondispersive_participation_derivatives_sum_to_zero() {
    let state = RealAxisEvaluator::new(Scatter2::new())
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &differentiable_lossless_two_layer_stack(),
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap();

    let participation = state
        .excitation(IncidentSide::Left)
        .expect("state should be projectable")
        .layer_participation_nondispersive()
        .unwrap();

    let value_sum: f64 = participation
        .value()
        .iter()
        .map(|layer| layer.total()[()])
        .sum();

    let first_sum: f64 = participation
        .derivatives()
        .first()
        .iter()
        .map(|layer| layer.total()[()])
        .sum();

    assert_relative_eq!(
        value_sum,
        1.0,
        epsilon = VALUE_TOLERANCE,
        max_relative = VALUE_TOLERANCE,
    );

    assert_relative_eq!(
        first_sum,
        0.0,
        epsilon = FIRST_TOLERANCE,
        max_relative = FIRST_TOLERANCE,
    );
}

#[test]
fn public_dispersive_layer_participation_returns_one_record_per_layer() {
    let state = RealAxisEvaluator::new(Scatter2::new())
        .retain(
            scalar_real_input(2.5, 0.31),
            &differentiable_lossless_two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let participation = state
        .excitation(IncidentSide::Left)
        .expect("state should be projectable")
        .layer_participation_dispersive()
        .unwrap();

    assert_eq!(participation.value().len(), 2);
}

#[test]
fn public_dispersive_layer_participation_sums_to_unity() {
    let state = RealAxisEvaluator::new(Scatter2::new())
        .retain(
            scalar_real_input(2.5, 0.31),
            &differentiable_lossless_two_layer_stack(),
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let participation = state
        .excitation(IncidentSide::Right)
        .expect("state should be projectable")
        .layer_participation_dispersive()
        .unwrap();

    let total: f64 = participation
        .value()
        .iter()
        .map(|layer| layer.total()[()])
        .sum();

    assert_relative_eq!(
        total,
        1.0,
        epsilon = VALUE_TOLERANCE,
        max_relative = VALUE_TOLERANCE,
    );
}

#[test]
fn public_dispersive_participation_derivatives_sum_to_zero() {
    let state = RealAxisEvaluator::new(Scatter2::new())
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &differentiable_lossless_two_layer_stack(),
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap();

    let participation = state
        .excitation(IncidentSide::Left)
        .expect("state should be projectable")
        .layer_participation_dispersive()
        .unwrap();

    let value_sum: f64 = participation
        .value()
        .iter()
        .map(|layer| layer.total()[()])
        .sum();

    let first_sum: f64 = participation
        .derivatives()
        .first()
        .iter()
        .map(|layer| layer.total()[()])
        .sum();

    assert_relative_eq!(
        value_sum,
        1.0,
        epsilon = VALUE_TOLERANCE,
        max_relative = VALUE_TOLERANCE,
    );

    assert_relative_eq!(
        first_sum,
        0.0,
        epsilon = FIRST_TOLERANCE,
        max_relative = FIRST_TOLERANCE,
    );
}
