use approx::assert_relative_eq;
use nalgebra::ComplexField;
use num_complex::Complex64;

use crate::{
    IncidentSide, Parameter, PlaneWaveEvaluator, Polarisation,
    backend::{scatter2::Scatter2, transfer2::Transfer2},
    test_support::{
        assertions::assert_complex_close,
        c,
        planar::{scalar_complex_input, two_layer_stack},
        stack::differentiable_lossless_two_layer_stack,
    },
};

const VALUE_TOLERANCE: f64 = 1.0e-10;
const DERIVATIVE_TOLERANCE: f64 = 1.0e-8;

#[test]
fn layer_bilinear_overlap_returns_one_record_per_finite_layer() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain_modal(
            scalar_complex_input(c(2.5), c(0.31)),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let overlap = state
        .excitation(IncidentSide::Left)
        .unwrap()
        .pair_with(state.excitation(IncidentSide::Right).unwrap())
        .unwrap()
        .layer_bilinear_overlap()
        .unwrap();

    assert_eq!(overlap.value().len(), 2);
}

#[test]
fn aggregate_bilinear_overlap_matches_sum_of_layers() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain_modal(
            scalar_complex_input(c(2.5), c(0.31)),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let pair = state
        .excitation(IncidentSide::Left)
        .unwrap()
        .pair_with(state.excitation(IncidentSide::Right).unwrap())
        .unwrap();

    let layers = pair.layer_bilinear_overlap().unwrap();

    let aggregate = pair.aggregate_bilinear_overlap().unwrap();

    let expected_electric = layers
        .value()
        .iter()
        .map(|layer| layer.electric()[()])
        .sum::<Complex64>();

    let expected_magnetic = layers
        .value()
        .iter()
        .map(|layer| layer.magnetic()[()])
        .sum::<Complex64>();

    let expected_total = layers
        .value()
        .iter()
        .map(|layer| layer.total()[()])
        .sum::<Complex64>();

    assert_complex_close(
        aggregate.value().electric()[()],
        expected_electric,
        VALUE_TOLERANCE,
    );

    assert_complex_close(
        aggregate.value().magnetic()[()],
        expected_magnetic,
        VALUE_TOLERANCE,
    );

    assert_complex_close(
        aggregate.value().total()[()],
        expected_total,
        VALUE_TOLERANCE,
    );
}

#[test]
fn aggregate_bilinear_total_is_component_sum() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain_modal(
            scalar_complex_input(c(2.5), c(0.31)),
            &two_layer_stack(),
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let overlap = state
        .excitation(IncidentSide::Left)
        .unwrap()
        .pair_with(state.excitation(IncidentSide::Right).unwrap())
        .unwrap()
        .aggregate_bilinear_overlap()
        .unwrap();

    assert_complex_close(
        overlap.value().total()[()],
        overlap.value().electric()[()] + overlap.value().magnetic()[()],
        VALUE_TOLERANCE,
    );
}

#[test]
fn exchanging_operands_preserves_bilinear_overlap() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let state = evaluator
            .retain_modal(
                scalar_complex_input(c(2.5), c(0.31)),
                &two_layer_stack(),
                polarisation,
            )
            .unwrap();

        let left_right = state
            .excitation(IncidentSide::Left)
            .unwrap()
            .pair_with(state.excitation(IncidentSide::Right).unwrap())
            .unwrap()
            .aggregate_bilinear_overlap()
            .unwrap();

        let right_left = state
            .excitation(IncidentSide::Right)
            .unwrap()
            .pair_with(state.excitation(IncidentSide::Left).unwrap())
            .unwrap()
            .aggregate_bilinear_overlap()
            .unwrap();

        assert_complex_close(
            left_right.value().electric()[()],
            right_left.value().electric()[()],
            VALUE_TOLERANCE,
        );

        assert_complex_close(
            left_right.value().magnetic()[()],
            right_left.value().magnetic()[()],
            VALUE_TOLERANCE,
        );

        assert_complex_close(
            left_right.value().total()[()],
            right_left.value().total()[()],
            VALUE_TOLERANCE,
        );
    }
}

#[test]
fn first_derivative_preserves_bilinear_symmetry() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain_modal_first(
            scalar_complex_input(c(2.5), c(0.31)),
            &differentiable_lossless_two_layer_stack(),
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap();

    let left_right = state
        .excitation(IncidentSide::Left)
        .unwrap()
        .pair_with(state.excitation(IncidentSide::Right).unwrap())
        .unwrap()
        .aggregate_bilinear_overlap()
        .unwrap();

    let right_left = state
        .excitation(IncidentSide::Right)
        .unwrap()
        .pair_with(state.excitation(IncidentSide::Left).unwrap())
        .unwrap()
        .aggregate_bilinear_overlap()
        .unwrap();

    let left_right = left_right.derivatives().first();

    let right_left = right_left.derivatives().first();

    assert_complex_close(
        left_right.electric()[()],
        right_left.electric()[()],
        DERIVATIVE_TOLERANCE,
    );

    assert_complex_close(
        left_right.magnetic()[()],
        right_left.magnetic()[()],
        DERIVATIVE_TOLERANCE,
    );

    assert_complex_close(
        left_right.total()[()],
        right_left.total()[()],
        DERIVATIVE_TOLERANCE,
    );
}

#[test]
fn second_derivative_preserves_bilinear_symmetry() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain_modal_second(
            scalar_complex_input(c(2.5), c(0.31)),
            &differentiable_lossless_two_layer_stack(),
            Polarisation::TransverseMagnetic,
            Parameter::Spectral,
        )
        .unwrap();

    let left_right = state
        .excitation(IncidentSide::Left)
        .unwrap()
        .pair_with(state.excitation(IncidentSide::Right).unwrap())
        .unwrap()
        .aggregate_bilinear_overlap()
        .unwrap();

    let right_left = state
        .excitation(IncidentSide::Right)
        .unwrap()
        .pair_with(state.excitation(IncidentSide::Left).unwrap())
        .unwrap()
        .aggregate_bilinear_overlap()
        .unwrap();

    let left_right = left_right.derivatives().second();

    let right_left = right_left.derivatives().second();

    assert_complex_close(
        left_right.electric()[()],
        right_left.electric()[()],
        DERIVATIVE_TOLERANCE,
    );

    assert_complex_close(
        left_right.magnetic()[()],
        right_left.magnetic()[()],
        DERIVATIVE_TOLERANCE,
    );

    assert_complex_close(
        left_right.total()[()],
        right_left.total()[()],
        DERIVATIVE_TOLERANCE,
    );
}

#[test]
fn scatter_and_transfer_backends_agree_on_bilinear_overlap() {
    let scatter = PlaneWaveEvaluator::new(Scatter2::new());

    let transfer = PlaneWaveEvaluator::new(Transfer2::new());

    let stack = two_layer_stack();
    let input = scalar_complex_input(c(2.5), c(0.31));

    let scatter_state = scatter
        .retain_modal(input.clone(), &stack, Polarisation::TransverseElectric)
        .unwrap();

    let transfer_state = transfer
        .retain_modal(input, &stack, Polarisation::TransverseElectric)
        .unwrap();

    let scatter_overlap = scatter_state
        .excitation(IncidentSide::Left)
        .unwrap()
        .pair_with(scatter_state.excitation(IncidentSide::Right).unwrap())
        .unwrap()
        .aggregate_bilinear_overlap()
        .unwrap();

    let transfer_overlap = transfer_state
        .excitation(IncidentSide::Left)
        .unwrap()
        .pair_with(transfer_state.excitation(IncidentSide::Right).unwrap())
        .unwrap()
        .aggregate_bilinear_overlap()
        .unwrap();

    assert_complex_close(
        scatter_overlap.value().electric()[()],
        transfer_overlap.value().electric()[()],
        VALUE_TOLERANCE,
    );

    assert_complex_close(
        scatter_overlap.value().magnetic()[()],
        transfer_overlap.value().magnetic()[()],
        VALUE_TOLERANCE,
    );

    assert_complex_close(
        scatter_overlap.value().total()[()],
        transfer_overlap.value().total()[()],
        VALUE_TOLERANCE,
    );
}
