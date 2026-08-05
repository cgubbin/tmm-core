use crate::{
    FiniteLayerIndex, IncidentSide, Parameter, PlaneWaveEvaluator, Polarisation,
    backend::scatter2::Scatter2,
    evaluate::pair::PlaneWavePairError,
    test_support::{
        planar::{scalar_real_input, single_layer_stack, two_layer_stack},
        stack::differentiable_lossless_two_layer_stack,
    },
};

#[test]
fn compatible_scalar_excitations_form_a_pair() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let reference_state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let comparison_state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let reference = reference_state.excitation(IncidentSide::Left).unwrap();

    let comparison = comparison_state.excitation(IncidentSide::Right).unwrap();

    let pair = reference
        .pair_with(comparison)
        .expect("compatible scalar excitations must form a pair");

    assert_eq!(pair.reference().incident_side(), IncidentSide::Left,);

    assert_eq!(pair.comparison().incident_side(), IncidentSide::Right,);
}

#[test]
fn pair_into_parts_preserves_reference_and_comparison_order() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let reference = state.excitation(IncidentSide::Right).unwrap();

    let comparison = state.excitation(IncidentSide::Left).unwrap();

    let pair = reference.pair_with(comparison).unwrap();

    let (reference, comparison) = pair.into_parts();

    assert_eq!(reference.incident_side(), IncidentSide::Right,);

    assert_eq!(comparison.incident_side(), IncidentSide::Left,);
}

#[test]
fn pairing_rejects_mismatched_polarisations() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let reference_state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let comparison_state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let reference = reference_state.excitation(IncidentSide::Left).unwrap();

    let comparison = comparison_state.excitation(IncidentSide::Left).unwrap();

    let error = reference
        .pair_with(comparison)
        .expect_err("TE and TM excitations must not be paired");

    assert_eq!(
        error,
        PlaneWavePairError::PolarisationMismatch {
            reference: Polarisation::TransverseElectric,
            comparison: Polarisation::TransverseMagnetic,
        },
    );
}

#[test]
fn pairing_rejects_different_derivative_mappings() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = differentiable_lossless_two_layer_stack();

    let reference_state = evaluator
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap();

    let comparison_state = evaluator
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::InPlane,
        )
        .unwrap();

    let reference = reference_state.excitation(IncidentSide::Left).unwrap();

    let comparison = comparison_state.excitation(IncidentSide::Left).unwrap();

    let error = reference
        .pair_with(comparison)
        .expect_err("different derivative mappings must be rejected");

    assert_eq!(error, PlaneWavePairError::DifferentialMappingMismatch,);
}

#[test]
fn pairing_rejects_different_thickness_parameter_indices() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = differentiable_lossless_two_layer_stack();

    let reference_state = evaluator
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::LayerThickness(FiniteLayerIndex(0)),
        )
        .unwrap();

    let comparison_state = evaluator
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::LayerThickness(FiniteLayerIndex(1)),
        )
        .unwrap();

    let reference = reference_state.excitation(IncidentSide::Left).unwrap();

    let comparison = comparison_state.excitation(IncidentSide::Left).unwrap();

    let error = reference
        .pair_with(comparison)
        .expect_err("different thickness axes must be rejected");

    assert_eq!(error, PlaneWavePairError::DifferentialMappingMismatch,);
}

#[test]
fn pairing_rejects_different_finite_layer_counts() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let reference_stack = two_layer_stack();

    let comparison_stack = single_layer_stack(1.0, 1.0);

    let reference_state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &reference_stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let comparison_state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &comparison_stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let reference = reference_state.excitation(IncidentSide::Left).unwrap();

    let comparison = comparison_state.excitation(IncidentSide::Left).unwrap();

    let error = reference
        .pair_with(comparison)
        .expect_err("different layer counts must be rejected");

    assert_eq!(
        error,
        PlaneWavePairError::LayerCountMismatch {
            reference_count: 2,
            comparison_count: 1,
        },
    );
}

#[test]
fn pairing_rejects_different_layer_thickness_values() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let reference_stack = single_layer_stack(0.4, 0.7);

    let comparison_stack = single_layer_stack(0.4, 0.9);

    let reference_state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &reference_stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let comparison_state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &comparison_stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let reference = reference_state.excitation(IncidentSide::Left).unwrap();

    let comparison = comparison_state.excitation(IncidentSide::Left).unwrap();

    let error = reference
        .pair_with(comparison)
        .expect_err("different integration intervals must be rejected");

    assert_eq!(
        error,
        PlaneWavePairError::LayerThicknessMismatch {
            index: FiniteLayerIndex(0),
        },
    );
}
