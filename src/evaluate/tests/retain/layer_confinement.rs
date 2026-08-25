use approx::assert_relative_eq;

use crate::{
    FiniteLayerIndex, IncidentSide, Parameter, Polarisation, RealAxisEvaluator,
    backend::scatter2::Scatter2,
    observable::{LayerAggregateError, LayerConfinementError},
    test_support::{planar::scalar_real_input, stack::differentiable_lossless_two_layer_stack},
};

const VALUE_TOLERANCE: f64 = 2.0e-10;
const FIRST_TOLERANCE: f64 = 2.0e-8;

#[test]
fn selecting_all_layers_through_public_api_gives_unit_confinement_nondispersive() {
    let state = RealAxisEvaluator::new(Scatter2::new())
        .retain(
            scalar_real_input(2.5, 0.31),
            &differentiable_lossless_two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let confinement = state
        .excitation(IncidentSide::Left)
        .expect("state should be projectable")
        .layer_confinement_by_nondispersive(|_| true)
        .unwrap();

    assert_relative_eq!(
        confinement.total()[()],
        1.0,
        epsilon = VALUE_TOLERANCE,
        max_relative = VALUE_TOLERANCE,
    );
}

#[test]
fn one_layer_nondispersive_confinement_matches_its_participation() {
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

    let confinement = state
        .excitation(IncidentSide::Right)
        .expect("state should be projectable")
        .layer_confinement_by_nondispersive(|index| index == FiniteLayerIndex::new(1))
        .unwrap();

    assert_relative_eq!(
        confinement.electric()[()],
        participation
            .get(FiniteLayerIndex::new(1))
            .unwrap()
            .electric()[()],
        epsilon = VALUE_TOLERANCE,
        max_relative = VALUE_TOLERANCE,
    );

    assert_relative_eq!(
        confinement.magnetic()[()],
        participation
            .get(FiniteLayerIndex::new(1))
            .unwrap()
            .magnetic()[()],
        epsilon = VALUE_TOLERANCE,
        max_relative = VALUE_TOLERANCE,
    );

    assert_relative_eq!(
        confinement.total()[()],
        participation.get(FiniteLayerIndex::new(1)).unwrap().total()[()],
        epsilon = VALUE_TOLERANCE,
        max_relative = VALUE_TOLERANCE,
    );
}

#[test]
fn nondispersive_unit_confinement_has_zero_first_derivative() {
    let state = RealAxisEvaluator::new(Scatter2::new())
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &differentiable_lossless_two_layer_stack(),
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap();

    let confinement = state
        .excitation(IncidentSide::Left)
        .expect("state should be projectable")
        .layer_confinement_by_nondispersive(|_| true)
        .unwrap();

    assert_relative_eq!(
        confinement.value().total()[()],
        1.0,
        epsilon = VALUE_TOLERANCE,
        max_relative = VALUE_TOLERANCE,
    );

    assert_relative_eq!(
        confinement.derivatives().first().total()[()],
        0.0,
        epsilon = FIRST_TOLERANCE,
        max_relative = FIRST_TOLERANCE,
    );
}

#[test]
fn nondispersive_public_confinement_rejects_empty_selection() {
    let state = RealAxisEvaluator::new(Scatter2::new())
        .retain(
            scalar_real_input(2.5, 0.31),
            &differentiable_lossless_two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let error = state
        .excitation(IncidentSide::Left)
        .expect("state should be projectable")
        .layer_confinement_by_nondispersive(|_| false)
        .expect_err("empty confinement selections must be rejected");

    assert!(matches!(
        error,
        LayerConfinementError::Aggregate(LayerAggregateError::EmptySelection,),
    ));
}

#[test]
fn selecting_all_layers_through_public_api_gives_unit_confinement_dispersive() {
    let state = RealAxisEvaluator::new(Scatter2::new())
        .retain(
            scalar_real_input(2.5, 0.31),
            &differentiable_lossless_two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let confinement = state
        .excitation(IncidentSide::Left)
        .expect("state should be projectable")
        .layer_confinement_by_dispersive(|_| true)
        .unwrap();

    assert_relative_eq!(
        confinement.total()[()],
        1.0,
        epsilon = VALUE_TOLERANCE,
        max_relative = VALUE_TOLERANCE,
    );
}

#[test]
fn one_layer_dispersive_confinement_matches_its_participation() {
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

    let confinement = state
        .excitation(IncidentSide::Right)
        .expect("state should be projectable")
        .layer_confinement_by_dispersive(|index| index == FiniteLayerIndex::new(1))
        .unwrap();

    assert_relative_eq!(
        confinement.electric()[()],
        participation
            .get(FiniteLayerIndex::new(1))
            .unwrap()
            .electric()[()],
        epsilon = VALUE_TOLERANCE,
        max_relative = VALUE_TOLERANCE,
    );

    assert_relative_eq!(
        confinement.magnetic()[()],
        participation
            .get(FiniteLayerIndex::new(1))
            .unwrap()
            .magnetic()[()],
        epsilon = VALUE_TOLERANCE,
        max_relative = VALUE_TOLERANCE,
    );

    assert_relative_eq!(
        confinement.total()[()],
        participation.get(FiniteLayerIndex::new(1)).unwrap().total()[()],
        epsilon = VALUE_TOLERANCE,
        max_relative = VALUE_TOLERANCE,
    );
}

#[test]
fn dispersive_unit_confinement_has_zero_first_derivative() {
    let state = RealAxisEvaluator::new(Scatter2::new())
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &differentiable_lossless_two_layer_stack(),
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap();

    let confinement = state
        .excitation(IncidentSide::Left)
        .expect("state should be projectable")
        .layer_confinement_by_dispersive(|_| true)
        .unwrap();

    assert_relative_eq!(
        confinement.value().total()[()],
        1.0,
        epsilon = VALUE_TOLERANCE,
        max_relative = VALUE_TOLERANCE,
    );

    assert_relative_eq!(
        confinement.derivatives().first().total()[()],
        0.0,
        epsilon = FIRST_TOLERANCE,
        max_relative = FIRST_TOLERANCE,
    );
}

#[test]
fn dispersive_public_confinement_rejects_empty_selection() {
    let state = RealAxisEvaluator::new(Scatter2::new())
        .retain(
            scalar_real_input(2.5, 0.31),
            &differentiable_lossless_two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let error = state
        .excitation(IncidentSide::Left)
        .expect("state should be projectable")
        .layer_confinement_by_dispersive(|_| false)
        .expect_err("empty confinement selections must be rejected");

    assert!(matches!(
        error,
        LayerConfinementError::Aggregate(LayerAggregateError::EmptySelection,),
    ));
}
