use crate::{
    FiniteLayerIndex, IncidentSide, Parameter, PlaneWaveEvaluator, Polarisation, backend::Scatter2,
};

use crate::test_support::{
    assertions::{assert_complex_close, assert_real_close},
    planar::{FILM_THICKNESS_CM, scalar_complex_input, scalar_real_input, single_layer_stack},
};

use num_complex::Complex64;

const TOLERANCE: f64 = 1.0e-12;

fn evaluator() -> PlaneWaveEvaluator<Scatter2> {
    PlaneWaveEvaluator::new(Scatter2)
}

#[test]
fn solve_and_retain_value_paths_have_identical_amplitudes() {
    let evaluator = evaluator();

    let stack = single_layer_stack(1.7, FILM_THICKNESS_CM);

    let solved = evaluator
        .evaluate(
            scalar_real_input(2.0, 0.1),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let retained = evaluator
        .retain(
            scalar_real_input(2.0, 0.1),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    for side in [IncidentSide::Left, IncidentSide::Right] {
        let solved = solved.amplitudes(side);

        let retained = retained.amplitudes(side);

        assert_complex_close(
            solved.value().reflection()[()],
            retained.value().reflection()[()],
            TOLERANCE,
        );

        assert_complex_close(
            solved.value().transmission()[()],
            retained.value().transmission()[()],
            TOLERANCE,
        );
    }
}

#[test]
fn solve_and_retain_value_paths_have_identical_power() {
    let evaluator = evaluator();

    let stack = single_layer_stack(1.7, FILM_THICKNESS_CM);

    let solved = evaluator
        .evaluate(
            scalar_real_input(2.0, 0.1),
            &stack,
            Polarisation::TransverseMagnetic,
        )
        .unwrap()
        .power(IncidentSide::Left);

    let retained = evaluator
        .retain(
            scalar_real_input(2.0, 0.1),
            &stack,
            Polarisation::TransverseMagnetic,
        )
        .unwrap()
        .power(IncidentSide::Left);

    assert_real_close(
        solved.value().reflectance()[()],
        retained.value().reflectance()[()],
        TOLERANCE,
    );

    assert_real_close(
        solved.value().transmittance()[()],
        retained.value().transmittance()[()],
        TOLERANCE,
    );

    assert_real_close(
        solved.value().absorptance()[()],
        retained.value().absorptance()[()],
        TOLERANCE,
    );
}

#[test]
fn solve_and_retain_first_derivatives_are_identical() {
    let evaluator = evaluator();

    let stack = single_layer_stack(1.7, FILM_THICKNESS_CM);

    let parameter = Parameter::LayerThickness(FiniteLayerIndex(0));

    let solved = evaluator
        .evaluate_first(
            scalar_real_input(2.0, 0.1),
            &stack,
            Polarisation::TransverseElectric,
            parameter,
        )
        .unwrap()
        .amplitudes(IncidentSide::Left);

    let retained = evaluator
        .retain_first(
            scalar_real_input(2.0, 0.1),
            &stack,
            Polarisation::TransverseElectric,
            parameter,
        )
        .unwrap()
        .amplitudes(IncidentSide::Left);

    assert_eq!(solved.parameter(), parameter);
    assert_eq!(retained.parameter(), parameter);

    assert_complex_close(
        solved.value().reflection()[()],
        retained.value().reflection()[()],
        TOLERANCE,
    );

    assert_complex_close(
        solved.first().reflection()[()],
        retained.first().reflection()[()],
        TOLERANCE,
    );
}

#[test]
fn solve_and_retain_second_derivatives_are_identical() {
    let evaluator = evaluator();

    let stack = single_layer_stack(1.7, FILM_THICKNESS_CM);

    let solved = evaluator
        .evaluate_second(
            scalar_real_input(2.0, 0.1),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap()
        .amplitudes(IncidentSide::Left);

    let retained = evaluator
        .retain_second(
            scalar_real_input(2.0, 0.1),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap()
        .amplitudes(IncidentSide::Left);

    assert_complex_close(
        solved.value().reflection()[()],
        retained.value().reflection()[()],
        TOLERANCE,
    );

    assert_complex_close(
        solved.first().reflection()[()],
        retained.first().reflection()[()],
        TOLERANCE,
    );

    assert_complex_close(
        solved.second().reflection()[()],
        retained.second().reflection()[()],
        TOLERANCE,
    );
}

#[test]
fn solve_and_retain_bivariate_results_are_identical() {
    let evaluator = evaluator();

    let stack = single_layer_stack(1.7, FILM_THICKNESS_CM);

    let thickness = Parameter::LayerThickness(FiniteLayerIndex(0));

    let solved = evaluator
        .evaluate_bivariate_second(
            scalar_real_input(2.0, 0.1),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
            thickness,
        )
        .unwrap()
        .amplitudes(IncidentSide::Left);

    let retained = evaluator
        .retain_bivariate_second(
            scalar_real_input(2.0, 0.1),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
            thickness,
        )
        .unwrap()
        .amplitudes(IncidentSide::Left);

    assert_eq!(solved.parameters(), retained.parameters(),);

    assert_complex_close(
        solved.gradient().axis0().reflection()[()],
        retained.gradient().axis0().reflection()[()],
        TOLERANCE,
    );

    assert_complex_close(
        solved.gradient().axis1().reflection()[()],
        retained.gradient().axis1().reflection()[()],
        TOLERANCE,
    );

    assert_complex_close(
        solved.hessian().axis0_axis1().reflection()[()],
        retained.hessian().axis0_axis1().reflection()[()],
        TOLERANCE,
    );
}

#[test]
fn solve_and_retain_modal_determinants_are_identical() {
    let evaluator = evaluator();

    let stack = single_layer_stack(1.7, FILM_THICKNESS_CM);

    let input = || scalar_complex_input(Complex64::new(2.0, 0.2), Complex64::new(0.3, 0.1));

    let solved = evaluator
        .evaluate_modal_second(
            input(),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap()
        .determinant();

    let retained = evaluator
        .retain_modal_second(
            input(),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap()
        .determinant();

    assert_complex_close(
        solved.value().value()[()],
        retained.value().value()[()],
        TOLERANCE,
    );

    assert_complex_close(
        solved.first().value()[()],
        retained.first().value()[()],
        TOLERANCE,
    );

    assert_complex_close(
        solved.second().value()[()],
        retained.second().value()[()],
        TOLERANCE,
    );
}
