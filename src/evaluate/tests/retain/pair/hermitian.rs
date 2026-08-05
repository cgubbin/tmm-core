use approx::assert_relative_eq;
use nalgebra::ComplexField;
use num_complex::Complex64;

use crate::{
    IncidentSide, Parameter, PlaneWaveEvaluator, Polarisation,
    backend::scatter2::Scatter2,
    test_support::{
        assertions::assert_complex_close,
        planar::{sampled_real_input, scalar_real_input, two_layer_stack},
        stack::differentiable_lossless_two_layer_stack,
    },
};

const VALUE_TOLERANCE: f64 = 1.0e-10;
const FIRST_TOLERANCE: f64 = 1.0e-8;
const SECOND_TOLERANCE: f64 = 1.0e-7;

fn scalar<C>(value: &ndarray::Array<C, ndarray::Ix0>) -> C
where
    C: Copy,
{
    value[()]
}

#[test]
fn layer_hermitian_overlap_returns_one_record_per_finite_layer() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let reference = state.excitation(IncidentSide::Left).unwrap();

    let comparison = state.excitation(IncidentSide::Right).unwrap();

    let overlap = reference
        .pair_with(comparison)
        .unwrap()
        .layer_hermitian_overlap()
        .unwrap();

    assert_eq!(overlap.value().len(), 2);
}

#[test]
fn layer_hermitian_overlap_total_is_component_sum() {
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
        .excitation(IncidentSide::Left)
        .unwrap()
        .pair_with(state.excitation(IncidentSide::Right).unwrap())
        .unwrap()
        .layer_hermitian_overlap()
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
fn aggregate_hermitian_overlap_matches_sum_of_layers() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let reference = state.excitation(IncidentSide::Left).unwrap();

    let comparison = state.excitation(IncidentSide::Right).unwrap();

    let pair = reference.pair_with(comparison).unwrap();

    let layers = pair.layer_hermitian_overlap().unwrap();

    let aggregate = pair.aggregate_hermitian_overlap().unwrap();

    let expected_electric = layers
        .value()
        .iter()
        .map(|layer| scalar(layer.electric()))
        .sum::<Complex64>();

    let expected_magnetic = layers
        .value()
        .iter()
        .map(|layer| scalar(layer.magnetic()))
        .sum::<Complex64>();

    let expected_total = layers
        .value()
        .iter()
        .map(|layer| scalar(layer.total()))
        .sum::<Complex64>();

    assert_complex_close(
        scalar(aggregate.value().electric()),
        expected_electric,
        VALUE_TOLERANCE,
    );

    assert_complex_close(
        scalar(aggregate.value().magnetic()),
        expected_magnetic,
        VALUE_TOLERANCE,
    );

    assert_complex_close(
        scalar(aggregate.value().total()),
        expected_total,
        VALUE_TOLERANCE,
    );
}

#[test]
fn aggregate_hermitian_overlap_total_is_component_sum() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let aggregate = state
        .excitation(IncidentSide::Left)
        .unwrap()
        .pair_with(state.excitation(IncidentSide::Right).unwrap())
        .unwrap()
        .aggregate_hermitian_overlap()
        .unwrap();

    assert_complex_close(
        scalar(aggregate.value().total()),
        scalar(aggregate.value().electric()) + scalar(aggregate.value().magnetic()),
        VALUE_TOLERANCE,
    );
}

#[test]
fn swapping_excitations_conjugates_layer_hermitian_overlap() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let left_right = state
        .excitation(IncidentSide::Left)
        .unwrap()
        .pair_with(state.excitation(IncidentSide::Right).unwrap())
        .unwrap()
        .layer_hermitian_overlap()
        .unwrap();

    let right_left = state
        .excitation(IncidentSide::Right)
        .unwrap()
        .pair_with(state.excitation(IncidentSide::Left).unwrap())
        .unwrap()
        .layer_hermitian_overlap()
        .unwrap();

    for (left_right, right_left) in left_right.value().iter().zip(right_left.value().iter()) {
        assert_complex_close(
            scalar(left_right.electric()),
            scalar(right_left.electric()).conjugate(),
            VALUE_TOLERANCE,
        );

        assert_complex_close(
            scalar(left_right.magnetic()),
            scalar(right_left.magnetic()).conjugate(),
            VALUE_TOLERANCE,
        );

        assert_complex_close(
            scalar(left_right.total()),
            scalar(right_left.total()).conjugate(),
            VALUE_TOLERANCE,
        );
    }
}

#[test]
fn swapping_excitations_conjugates_aggregate_hermitian_overlap() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let left_right = state
        .excitation(IncidentSide::Left)
        .unwrap()
        .pair_with(state.excitation(IncidentSide::Right).unwrap())
        .unwrap()
        .aggregate_hermitian_overlap()
        .unwrap();

    let right_left = state
        .excitation(IncidentSide::Right)
        .unwrap()
        .pair_with(state.excitation(IncidentSide::Left).unwrap())
        .unwrap()
        .aggregate_hermitian_overlap()
        .unwrap();

    assert_complex_close(
        scalar(left_right.value().electric()),
        scalar(right_left.value().electric()).conjugate(),
        VALUE_TOLERANCE,
    );

    assert_complex_close(
        scalar(left_right.value().magnetic()),
        scalar(right_left.value().magnetic()).conjugate(),
        VALUE_TOLERANCE,
    );

    assert_complex_close(
        scalar(left_right.value().total()),
        scalar(right_left.value().total()).conjugate(),
        VALUE_TOLERANCE,
    );
}

#[test]
fn layer_hermitian_self_overlap_is_real() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let state = evaluator
            .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
            .unwrap();

        for side in [IncidentSide::Left, IncidentSide::Right] {
            let excitation = state.excitation(side).unwrap();
            let excitationb = state.excitation(side).unwrap();

            let overlap = excitation
                .pair_with(excitationb)
                .unwrap()
                .layer_hermitian_overlap()
                .unwrap();

            for layer in overlap.value().iter() {
                assert_relative_eq!(
                    scalar(layer.electric()).imaginary(),
                    0.0,
                    epsilon = VALUE_TOLERANCE,
                    max_relative = VALUE_TOLERANCE,
                );

                assert_relative_eq!(
                    scalar(layer.magnetic()).imaginary(),
                    0.0,
                    epsilon = VALUE_TOLERANCE,
                    max_relative = VALUE_TOLERANCE,
                );

                assert_relative_eq!(
                    scalar(layer.total()).imaginary(),
                    0.0,
                    epsilon = VALUE_TOLERANCE,
                    max_relative = VALUE_TOLERANCE,
                );
            }
        }
    }
}

#[test]
fn layer_hermitian_self_overlap_is_nonnegative() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let state = evaluator
            .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
            .unwrap();

        for side in [IncidentSide::Left, IncidentSide::Right] {
            let excitation = state.excitation(side).unwrap();
            let excitationb = state.excitation(side).unwrap();

            let overlap = excitation
                .pair_with(excitationb)
                .unwrap()
                .layer_hermitian_overlap()
                .unwrap();

            for layer in overlap.value().iter() {
                assert!(scalar(layer.electric()).real() >= -VALUE_TOLERANCE,);

                assert!(scalar(layer.magnetic()).real() >= -VALUE_TOLERANCE,);

                assert!(scalar(layer.total()).real() >= -VALUE_TOLERANCE,);
            }
        }
    }
}

#[test]
fn aggregate_hermitian_self_overlap_is_real_and_nonnegative() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let excitation = state.excitation(IncidentSide::Left).unwrap();
    let excitationb = state.excitation(IncidentSide::Left).unwrap();

    let overlap = excitation
        .pair_with(excitationb)
        .unwrap()
        .aggregate_hermitian_overlap()
        .unwrap();

    for value in [
        scalar(overlap.value().electric()),
        scalar(overlap.value().magnetic()),
        scalar(overlap.value().total()),
    ] {
        assert_relative_eq!(
            value.imaginary(),
            0.0,
            epsilon = VALUE_TOLERANCE,
            max_relative = VALUE_TOLERANCE,
        );

        assert!(value.real() >= -VALUE_TOLERANCE,);
    }
}

#[test]
fn first_derivative_obeys_hermitian_swap_symmetry() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = differentiable_lossless_two_layer_stack();

    let state = evaluator
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap();

    let left_right = state
        .excitation(IncidentSide::Left)
        .unwrap()
        .pair_with(state.excitation(IncidentSide::Right).unwrap())
        .unwrap()
        .aggregate_hermitian_overlap()
        .unwrap();

    let right_left = state
        .excitation(IncidentSide::Right)
        .unwrap()
        .pair_with(state.excitation(IncidentSide::Left).unwrap())
        .unwrap()
        .aggregate_hermitian_overlap()
        .unwrap();

    let left_right = left_right.derivatives().first();

    let right_left = right_left.derivatives().first();

    assert_complex_close(
        scalar(left_right.electric()),
        scalar(right_left.electric()).conjugate(),
        FIRST_TOLERANCE,
    );

    assert_complex_close(
        scalar(left_right.magnetic()),
        scalar(right_left.magnetic()).conjugate(),
        FIRST_TOLERANCE,
    );

    assert_complex_close(
        scalar(left_right.total()),
        scalar(right_left.total()).conjugate(),
        FIRST_TOLERANCE,
    );
}

#[test]
fn first_derivative_of_hermitian_self_overlap_is_real() {
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

    let excitation = state.excitation(IncidentSide::Left).unwrap();

    let excitationb = state.excitation(IncidentSide::Left).unwrap();

    let overlap = excitation
        .pair_with(excitationb)
        .unwrap()
        .aggregate_hermitian_overlap()
        .unwrap();

    let first = overlap.derivatives().first();

    for value in [
        scalar(first.electric()),
        scalar(first.magnetic()),
        scalar(first.total()),
    ] {
        assert_relative_eq!(
            value.imaginary(),
            0.0,
            epsilon = FIRST_TOLERANCE,
            max_relative = FIRST_TOLERANCE,
        );
    }
}

#[test]
fn second_derivative_obeys_hermitian_swap_symmetry() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = differentiable_lossless_two_layer_stack();

    let state = evaluator
        .retain_second(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap();

    let left_right = state
        .excitation(IncidentSide::Left)
        .unwrap()
        .pair_with(state.excitation(IncidentSide::Right).unwrap())
        .unwrap()
        .aggregate_hermitian_overlap()
        .unwrap();

    let right_left = state
        .excitation(IncidentSide::Right)
        .unwrap()
        .pair_with(state.excitation(IncidentSide::Left).unwrap())
        .unwrap()
        .aggregate_hermitian_overlap()
        .unwrap();

    let left_right = left_right.derivatives().second();

    let right_left = right_left.derivatives().second();

    assert_complex_close(
        scalar(left_right.electric()),
        scalar(right_left.electric()).conjugate(),
        SECOND_TOLERANCE,
    );

    assert_complex_close(
        scalar(left_right.magnetic()),
        scalar(right_left.magnetic()).conjugate(),
        SECOND_TOLERANCE,
    );

    assert_complex_close(
        scalar(left_right.total()),
        scalar(right_left.total()).conjugate(),
        SECOND_TOLERANCE,
    );
}

#[test]
fn second_derivative_of_hermitian_self_overlap_is_real() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = differentiable_lossless_two_layer_stack();

    let state = evaluator
        .retain_second(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseMagnetic,
            Parameter::Spectral,
        )
        .unwrap();

    let excitation = state.excitation(IncidentSide::Right).unwrap();
    let excitationb = state.excitation(IncidentSide::Right).unwrap();

    let overlap = excitation
        .pair_with(excitationb)
        .unwrap()
        .aggregate_hermitian_overlap()
        .unwrap();

    let second = overlap.derivatives().second();

    for value in [
        scalar(second.electric()),
        scalar(second.magnetic()),
        scalar(second.total()),
    ] {
        assert_relative_eq!(
            value.imaginary(),
            0.0,
            epsilon = SECOND_TOLERANCE,
            max_relative = SECOND_TOLERANCE,
        );
    }
}

#[test]
fn overlap_from_projected_batch_matches_direct_scalar_overlap() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

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

    let projected_overlap = projected
        .excitation(IncidentSide::Left)
        .unwrap()
        .pair_with(projected.excitation(IncidentSide::Right).unwrap())
        .unwrap()
        .aggregate_hermitian_overlap()
        .unwrap();

    let direct_overlap = direct
        .excitation(IncidentSide::Left)
        .unwrap()
        .pair_with(direct.excitation(IncidentSide::Right).unwrap())
        .unwrap()
        .aggregate_hermitian_overlap()
        .unwrap();

    assert_complex_close(
        scalar(projected_overlap.value().electric()),
        scalar(direct_overlap.value().electric()),
        VALUE_TOLERANCE,
    );

    assert_complex_close(
        scalar(projected_overlap.value().magnetic()),
        scalar(direct_overlap.value().magnetic()),
        VALUE_TOLERANCE,
    );

    assert_complex_close(
        scalar(projected_overlap.value().total()),
        scalar(direct_overlap.value().total()),
        VALUE_TOLERANCE,
    );
}

#[test]
fn pairing_allows_different_scalar_coordinates() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let reference_state = evaluator
        .retain(
            scalar_real_input(2.3, 0.21),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let comparison_state = evaluator
        .retain(
            scalar_real_input(2.7, 0.34),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let pair = reference_state
        .excitation(IncidentSide::Left)
        .unwrap()
        .pair_with(comparison_state.excitation(IncidentSide::Right).unwrap())
        .expect("distinct scalar coordinates are permitted");

    let overlap = pair.aggregate_hermitian_overlap().unwrap();

    assert!(scalar(overlap.value().total()).real().is_finite(),);

    assert!(scalar(overlap.value().total()).imaginary().is_finite(),);
}
