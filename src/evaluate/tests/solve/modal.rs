use crate::{Parameter, PlaneWaveEvaluator, Polarisation, backend::Scatter2};

use crate::test_support::{
    TOLERANCE,
    assertions::assert_complex_close,
    planar::{dielectric_interface, scalar_complex_input},
};

use num_complex::Complex64;

fn evaluator() -> PlaneWaveEvaluator<Scatter2> {
    PlaneWaveEvaluator::new(Scatter2)
}

#[test]
fn modal_value_evaluation_returns_finite_determinant() {
    let evaluator = evaluator();

    let result = evaluator
        .evaluate_modal(
            scalar_complex_input(Complex64::new(2.0, 0.2), Complex64::new(0.3, 0.1)),
            &dielectric_interface(2.0),
            Polarisation::TransverseElectric,
        )
        .expect("modal evaluation should succeed");

    let determinant = result.determinant();

    assert!(determinant.value().value()[()].is_finite());
}

#[test]
fn modal_first_derivative_attaches_spectral_parameter() {
    let evaluator = evaluator();

    let result = evaluator
        .evaluate_modal_first(
            scalar_complex_input(Complex64::new(2.0, 0.2), Complex64::new(0.3, 0.1)),
            &dielectric_interface(2.0),
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .expect("modal derivative should succeed");

    let determinant = result.determinant();

    assert_eq!(determinant.parameter(), Parameter::Spectral,);

    assert!(determinant.first().value()[()].is_finite());
}

#[test]
fn modal_first_derivative_matches_complex_central_difference() {
    let evaluator = evaluator();

    let stack = dielectric_interface(2.0);

    let k0 = Complex64::new(2.0, 0.2);
    let k_parallel = Complex64::new(0.3, 0.1);

    let differentiated = evaluator
        .evaluate_modal_first(
            scalar_complex_input(k0, k_parallel),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap()
        .determinant();

    let step = 1.0e-6;
    let delta = Complex64::new(step, 0.0);

    let below = evaluator
        .evaluate_modal(
            scalar_complex_input(k0 - delta, k_parallel),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap()
        .determinant()
        .value()
        .value()[()];

    let above = evaluator
        .evaluate_modal(
            scalar_complex_input(k0 + delta, k_parallel),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap()
        .determinant()
        .value()
        .value()[()];

    let finite_difference = (above - below) / (2.0 * step);

    assert_complex_close(differentiated.first().value()[()], finite_difference, 1e-9);
}

#[test]
fn modal_second_derivative_matches_complex_central_difference() {
    let evaluator = evaluator();

    let stack = dielectric_interface(2.0);

    let k0 = Complex64::new(2.0, 0.2);
    let k_parallel = Complex64::new(0.3, 0.1);

    let differentiated = evaluator
        .evaluate_modal_second(
            scalar_complex_input(k0, k_parallel),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap()
        .determinant();

    let step = 2.0e-5;
    let delta = Complex64::new(step, 0.0);

    let below = evaluator
        .evaluate_modal(
            scalar_complex_input(k0 - delta, k_parallel),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap()
        .determinant()
        .value()
        .value()[()];

    let centre = evaluator
        .evaluate_modal(
            scalar_complex_input(k0, k_parallel),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap()
        .determinant()
        .value()
        .value()[()];

    let above = evaluator
        .evaluate_modal(
            scalar_complex_input(k0 + delta, k_parallel),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap()
        .determinant()
        .value()
        .value()[()];

    let finite_difference = (above - 2.0 * centre + below) / (step * step);

    assert_complex_close(
        differentiated.second().value()[()],
        finite_difference,
        1.0e-5,
    );
}

#[test]
fn modal_in_plane_derivative_is_supported_holomorphically() {
    let evaluator = evaluator();

    let result = evaluator
        .evaluate_modal_first(
            scalar_complex_input(Complex64::new(2.0, 0.2), Complex64::new(0.3, 0.1)),
            &dielectric_interface(2.0),
            Polarisation::TransverseElectric,
            Parameter::InPlane,
        )
        .expect("in-plane modal derivative should succeed");

    let determinant = result.determinant();

    assert_eq!(determinant.parameter(), Parameter::InPlane,);

    assert!(determinant.first().value()[()].is_finite());
}
