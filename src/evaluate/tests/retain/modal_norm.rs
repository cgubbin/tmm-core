use ndarray::Ix0;

use crate::{
    CoordinateInput, FiniteLayerIndex, Parameter, PlaneWaveEvaluator, Polarisation,
    algebra::{Jet0, ScalarAlgebra},
    backend::{ModalSolutionSource, ReconstructLayerModeWaves, Scatter2, Transfer2},
    evaluate::mode::raw_qnm_normalisation_unchecked,
    test_support::{
        C,
        assertions::{assert_complex_close, assert_holo_zero_jet_close},
        finite_difference::{
            FIRST_DERIVATIVE_TOLERANCE, SECOND_DERIVATIVE_TOLERANCE, VALUE_TOLERANCE,
        },
        planar::{scalar_complex_input, two_layer_stack},
    },
};

fn modal_input() -> CoordinateInput<C, Ix0> {
    scalar_complex_input(C::new(2.5, -0.05), C::new(0.31, 0.02))
}

macro_rules! for_each_modal_backend {
    ($evaluator:ident, $body:block) => {{
        {
            let $evaluator = PlaneWaveEvaluator::new(Scatter2::new());

            $body
        }

        {
            let $evaluator = PlaneWaveEvaluator::new(Transfer2::new());

            $body
        }
    }};
}

#[test]
fn first_order_modes_have_constant_unit_qnm_normalisation() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for parameter in [
            Parameter::Spectral,
            Parameter::LayerThickness(FiniteLayerIndex::new(0)),
            Parameter::LayerThickness(FiniteLayerIndex::new(1)),
        ] {
            for_each_modal_backend!(evaluator, {
                let state = evaluator
                    .retain_modal_first(modal_input(), &two_layer_stack(), polarisation, parameter)
                    .unwrap();

                let mode = state.mode().unwrap();

                let normalisation =
                    raw_qnm_normalisation_unchecked(mode.solution(), mode.state()).unwrap();

                let total = normalisation.total();

                assert_complex_close(total.value()[()], C::new(1.0, 0.0), VALUE_TOLERANCE);

                assert_complex_close(
                    total.first()[()],
                    C::new(0.0, 0.0),
                    FIRST_DERIVATIVE_TOLERANCE,
                );
            });
        }
    }
}

#[test]
fn modes_are_qnm_normalised_at_value_order() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for_each_modal_backend!(evaluator, {
            let state = evaluator
                .retain_modal(modal_input(), &two_layer_stack(), polarisation)
                .unwrap();

            let mode = state.mode().unwrap();

            let normalisation =
                raw_qnm_normalisation_unchecked(mode.solution(), mode.state()).unwrap();

            assert_complex_close(
                normalisation.total().value()[()],
                C::new(1.0, 0.0),
                VALUE_TOLERANCE,
            );
        });
    }
}

#[test]
fn second_order_modes_have_constant_unit_qnm_normalisation() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for parameter in [
            Parameter::Spectral,
            Parameter::LayerThickness(FiniteLayerIndex::new(0)),
            Parameter::LayerThickness(FiniteLayerIndex::new(1)),
        ] {
            for_each_modal_backend!(evaluator, {
                let state = evaluator
                    .retain_modal_second(modal_input(), &two_layer_stack(), polarisation, parameter)
                    .unwrap();

                let mode = state.mode().unwrap();

                let normalisation =
                    raw_qnm_normalisation_unchecked(mode.solution(), mode.state()).unwrap();

                let total = normalisation.total();

                assert_complex_close(total.value()[()], C::new(1.0, 0.0), VALUE_TOLERANCE);

                assert_complex_close(
                    total.first()[()],
                    C::new(0.0, 0.0),
                    FIRST_DERIVATIVE_TOLERANCE,
                );

                assert_complex_close(
                    total.second()[()],
                    C::new(0.0, 0.0),
                    SECOND_DERIVATIVE_TOLERANCE,
                );
            });
        }
    }
}

#[test]
fn scatter_layer_mode_reconstruction_is_linear_in_candidate_scale() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain_modal(
            modal_input(),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();
    let workspace = state.workspace();

    let candidate = workspace.modal_boundary_solution().unwrap();

    let scale = Jet0::filled_constant_like(candidate.state().field().value(), C::new(0.7, -0.4));

    let scaled_candidate = candidate.clone().scaled(&scale);

    let raw = workspace.reconstruct_layer_mode_waves(&candidate).unwrap();

    let scaled = workspace
        .reconstruct_layer_mode_waves(&scaled_candidate)
        .unwrap();

    for (raw, scaled) in raw.iter().zip(&scaled) {
        assert_holo_zero_jet_close(
            scaled.left().forward(),
            &raw.left().forward().multiply(&scale),
        );
        assert_holo_zero_jet_close(
            scaled.left().backward(),
            &raw.left().backward().multiply(&scale),
        );
        assert_holo_zero_jet_close(
            scaled.right().forward(),
            &raw.right().forward().multiply(&scale),
        );
        assert_holo_zero_jet_close(
            scaled.right().backward(),
            &raw.right().backward().multiply(&scale),
        );
    }
}

#[test]
fn qnm_normalisation_is_quadratic_in_candidate_scale() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain_modal(
            modal_input(),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let candidate = state.workspace().modal_boundary_solution().unwrap();

    let raw = raw_qnm_normalisation_unchecked(&candidate, &state).unwrap();

    let scale = Jet0::filled_constant_like(candidate.state().field().value(), C::new(0.7, -0.4));

    let scaled_candidate = candidate.scaled(&scale);

    let scaled = raw_qnm_normalisation_unchecked(&scaled_candidate, &state).unwrap();

    let expected = raw.total().multiply(&scale).multiply(&scale);

    eprintln!("raw      = {:?}", raw.total().value()[()]);
    eprintln!("scale    = {:?}", scale.value()[()]);
    eprintln!("scaled   = {:?}", scaled.total().value()[()]);
    eprintln!("expected = {:?}", expected.value()[()]);

    assert_complex_close(
        scaled.total().value()[()],
        expected.value()[()],
        VALUE_TOLERANCE,
    );
}

#[test]
fn bivariate_modes_have_constant_unit_qnm_normalisation() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for_each_modal_backend!(evaluator, {
            let state = evaluator
                .retain_modal_bivariate_second(
                    modal_input(),
                    &two_layer_stack(),
                    polarisation,
                    Parameter::Spectral,
                    Parameter::LayerThickness(FiniteLayerIndex::new(1)),
                )
                .unwrap();

            let mode = state.mode().unwrap();

            let normalisation =
                raw_qnm_normalisation_unchecked(mode.solution(), mode.state()).unwrap();

            let total = normalisation.total();

            assert_complex_close(total.value()[()], C::new(1.0, 0.0), VALUE_TOLERANCE);

            assert_complex_close(
                total.axis0()[()],
                C::new(0.0, 0.0),
                FIRST_DERIVATIVE_TOLERANCE,
            );

            assert_complex_close(
                total.axis1()[()],
                C::new(0.0, 0.0),
                FIRST_DERIVATIVE_TOLERANCE,
            );

            assert_complex_close(
                total.axis0_axis0()[()],
                C::new(0.0, 0.0),
                SECOND_DERIVATIVE_TOLERANCE,
            );

            assert_complex_close(
                total.axis0_axis1()[()],
                C::new(0.0, 0.0),
                SECOND_DERIVATIVE_TOLERANCE,
            );

            assert_complex_close(
                total.axis1_axis1()[()],
                C::new(0.0, 0.0),
                SECOND_DERIVATIVE_TOLERANCE,
            );
        });
    }
}
