use crate::{
    FiniteLayerIndex, Parameter, PlaneWaveEvaluator, Polarisation,
    backend::scatter2::Scatter2,
    observable::HermitianOverlapError,
    test_support::{
        planar::{sampled_real_input, scalar_real_input, two_layer_stack},
        stack::{absorbing_single_layer_stack, two_layer_stack_with_thicknesses},
    },
};

#[test]
fn compatible_states_form_a_pair() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let reference = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let comparison = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let pair = reference
        .pair_with(&comparison)
        .expect("identical compiled states must be compatible");

    assert!(std::ptr::eq(pair.reference(), &reference),);

    assert!(std::ptr::eq(pair.comparison(), &comparison),);
}

#[test]
fn pair_rejects_mismatched_polarisations() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let reference = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let comparison = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let error = reference
        .pair_with(&comparison)
        .expect_err("TE and TM states must not form a Hermitian pair");

    assert_eq!(
        error,
        HermitianOverlapError::PolarisationMismatch {
            reference: Polarisation::TransverseElectric,
            comparison: Polarisation::TransverseMagnetic,
        },
    );
}

#[test]
fn pair_rejects_different_finite_layer_counts() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let reference_stack = two_layer_stack();
    let comparison_stack = absorbing_single_layer_stack();

    let reference = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &reference_stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let comparison = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &comparison_stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let error = reference
        .pair_with(&comparison)
        .expect_err("different layer partitions must be rejected");

    assert_eq!(
        error,
        HermitianOverlapError::LayerCountMismatch {
            reference_count: 2,
            comparison_count: 1,
        },
    );
}

#[test]
fn pair_rejects_mismatched_layer_thicknesses() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let reference_stack = two_layer_stack_with_thicknesses(0.4, 0.7);

    let comparison_stack = two_layer_stack_with_thicknesses(0.4, 0.9);

    let reference = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &reference_stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let comparison = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &comparison_stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let error = reference
        .pair_with(&comparison)
        .expect_err("corresponding layers must have equal thicknesses");

    assert_eq!(
        error,
        HermitianOverlapError::LayerThicknessMismatch {
            index: FiniteLayerIndex(1),
        },
    );
}

#[test]
fn pair_rejects_different_sample_extents() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let reference = evaluator
        .retain(
            sampled_real_input(&[2.3, 2.4, 2.5], &[0.2, 0.3, 0.4]),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let comparison = evaluator
        .retain(
            sampled_real_input(&[2.3, 2.4], &[0.2, 0.3]),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let error = reference
        .pair_with(&comparison)
        .expect_err("aligned overlap requires equal sample extents");

    assert_eq!(
        error,
        HermitianOverlapError::SampleShapeMismatch {
            reference: vec![3],
            comparison: vec![2],
        },
    );
}

#[test]
fn pair_rejects_different_derivative_parameters() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let reference = evaluator
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap();

    let comparison = evaluator
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::InPlane,
        )
        .unwrap();

    let error = reference
        .pair_with(&comparison)
        .expect_err("jet derivatives must have the same semantic parameter");

    assert_eq!(error, HermitianOverlapError::DifferentialMappingMismatch,);
}
