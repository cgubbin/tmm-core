use approx::assert_relative_eq;
use ndarray::Array0;
use num_complex::Complex64;

use crate::{
    IncidentSide, Parameter, PlaneWaveEvaluator, Polarisation,
    backend::scatter2::Scatter2,
    test_support::{
        C,
        jet::J0,
        planar::{sampled_real_input, scalar_real_input, two_layer_stack},
        stack::differentiable_lossless_two_layer_stack,
    },
};

const VALUE_TOLERANCE: f64 = 2.0e-10;
const FIRST_TOLERANCE: f64 = 2.0e-8;

fn assert_complex_close(actual: Complex64, expected: Complex64, tolerance: f64) {
    assert_relative_eq!(
        actual.re,
        expected.re,
        epsilon = tolerance,
        max_relative = tolerance,
    );

    assert_relative_eq!(
        actual.im,
        expected.im,
        epsilon = tolerance,
        max_relative = tolerance,
    );
}

#[test]
fn layer_overlap_returns_one_record_per_finite_layer() {
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

    let overlap = reference
        .pair_with(&comparison)
        .unwrap()
        .layer_overlap(IncidentSide::Left, IncidentSide::Left)
        .unwrap();

    assert_eq!(overlap.value().len(), 2);
}

#[test]
fn self_overlap_is_real() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let overlap = state
        .pair_with(&state)
        .unwrap()
        .layer_overlap(IncidentSide::Left, IncidentSide::Left)
        .unwrap();

    for layer in overlap.value().iter() {
        for value in [
            layer.electric()[()],
            layer.magnetic()[()],
            layer.total()[()],
        ] {
            assert_relative_eq!(
                value.im,
                0.0,
                epsilon = VALUE_TOLERANCE,
                max_relative = VALUE_TOLERANCE,
            );
        }
    }
}

fn scalar(arr: &Array0<C>) -> C {
    arr[()]
}

fn scalar_real(arr: &Array0<f64>) -> f64 {
    arr[()]
}

#[test]
fn self_overlap_components_are_nonnegative() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let overlap = state
        .pair_with(&state)
        .unwrap()
        .layer_overlap(IncidentSide::Right, IncidentSide::Right)
        .unwrap();

    for layer in overlap.value().iter() {
        assert!(scalar(layer.electric()).re >= -VALUE_TOLERANCE,);

        assert!(scalar(layer.magnetic()).re >= -VALUE_TOLERANCE,);

        assert!(scalar(layer.total()).re >= -VALUE_TOLERANCE,);
    }
}

#[test]
fn self_overlap_matches_direct_integrated_field_norms() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let overlap = state
        .pair_with(&state)
        .unwrap()
        .raw_layer_overlap(IncidentSide::Left, IncidentSide::Left)
        .unwrap();

    let integrated = state
        .raw_layer_integration_inputs(IncidentSide::Left)
        .unwrap()
        .integrate();

    for (overlap, integrated) in overlap.iter().zip(integrated.iter()) {
        let norms = crate::observable::project_integrated_field_norms(
            integrated.state_products(),
            integrated.quantities(),
            state.problem().coordinates().vacuum_angular_wavenumber(),
            state.problem().coordinates().parallel_angular_wavenumber(),
        );

        assert_complex_close(
            scalar(overlap.electric()),
            Complex64::new(scalar_real(norms.electric().value()), 0.0),
            VALUE_TOLERANCE,
        );

        assert_complex_close(
            scalar(overlap.magnetic()),
            Complex64::new(scalar_real(norms.magnetic().value()), 0.0),
            VALUE_TOLERANCE,
        );
    }
}

#[test]
fn swapping_pair_operands_conjugates_overlap() {
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

    let reference_comparison = reference
        .pair_with(&comparison)
        .unwrap()
        .layer_overlap(IncidentSide::Left, IncidentSide::Right)
        .unwrap();

    let comparison_reference = comparison
        .pair_with(&reference)
        .unwrap()
        .layer_overlap(IncidentSide::Right, IncidentSide::Left)
        .unwrap();

    for (left_right, right_left) in reference_comparison
        .value()
        .iter()
        .zip(comparison_reference.value().iter())
    {
        assert_complex_close(
            scalar(left_right.electric()),
            scalar(right_left.electric()).conj(),
            VALUE_TOLERANCE,
        );

        assert_complex_close(
            scalar(left_right.magnetic()),
            scalar(right_left.magnetic()).conj(),
            VALUE_TOLERANCE,
        );

        assert_complex_close(
            scalar(left_right.total()),
            scalar(right_left.total()).conj(),
            VALUE_TOLERANCE,
        );
    }
}

#[test]
fn evaluator_overlap_total_is_component_sum() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let overlap = state
        .pair_with(&state)
        .unwrap()
        .layer_overlap(IncidentSide::Left, IncidentSide::Right)
        .unwrap();

    for layer in overlap.value().iter() {
        assert_complex_close(
            scalar(layer.total()),
            scalar(layer.electric()) + scalar(layer.magnetic()),
            VALUE_TOLERANCE,
        );
    }
}

#[test]
fn aggregate_overlap_matches_sum_of_layer_overlaps() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let pair = state.pair_with(&state).unwrap();

    let layers = pair
        .layer_overlap(IncidentSide::Left, IncidentSide::Right)
        .unwrap();

    let aggregate = pair
        .aggregate_overlap(IncidentSide::Left, IncidentSide::Right)
        .unwrap();

    let expected: Complex64 = layers
        .value()
        .iter()
        .map(|layer| scalar(layer.total()))
        .sum();

    assert_complex_close(scalar(aggregate.value().total()), expected, VALUE_TOLERANCE);
}

#[test]
fn first_derivative_respects_hermitian_swap_symmetry() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = differentiable_lossless_two_layer_stack();

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
            Parameter::Spectral,
        )
        .unwrap();

    let left_right = reference
        .pair_with(&comparison)
        .unwrap()
        .layer_overlap(IncidentSide::Left, IncidentSide::Right)
        .unwrap();

    let right_left = comparison
        .pair_with(&reference)
        .unwrap()
        .layer_overlap(IncidentSide::Right, IncidentSide::Left)
        .unwrap();

    for (left_right, right_left) in left_right
        .derivatives()
        .first()
        .iter()
        .zip(right_left.derivatives().first().iter())
    {
        assert_complex_close(
            scalar(left_right.total()),
            scalar(right_left.total()).conj(),
            FIRST_TOLERANCE,
        );
    }
}

#[test]
fn self_overlap_first_derivative_is_real() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = differentiable_lossless_two_layer_stack();

    let state = evaluator
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseMagnetic,
            Parameter::Spectral,
        )
        .unwrap();

    let overlap = state
        .pair_with(&state)
        .unwrap()
        .layer_overlap(IncidentSide::Left, IncidentSide::Left)
        .unwrap();

    for layer in overlap.derivatives().first().iter() {
        assert_relative_eq!(
            scalar(layer.total()).im,
            0.0,
            epsilon = FIRST_TOLERANCE,
            max_relative = FIRST_TOLERANCE,
        );
    }
}

#[test]
fn overlap_preserves_aligned_sample_shape() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let input = sampled_real_input(&[2.3, 2.4, 2.5], &[0.21, 0.26, 0.31]);

    let reference = evaluator
        .retain(input.clone(), &stack, Polarisation::TransverseElectric)
        .unwrap();

    let comparison = evaluator
        .retain(input, &stack, Polarisation::TransverseElectric)
        .unwrap();

    let overlap = reference
        .pair_with(&comparison)
        .unwrap()
        .layer_overlap(IncidentSide::Left, IncidentSide::Left)
        .unwrap();

    for layer in overlap.value().iter() {
        assert_eq!(layer.total().shape(), &[3],);
    }
}
