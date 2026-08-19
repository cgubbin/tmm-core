use ndarray::arr0;
use num_complex::Complex64;

use crate::{
    ComplexPlane, ComplexPlaneEvaluator, Polarisation,
    backend::{Scatter2, evaluate_exterior_wavevectors},
    input::CanonicalCoordinates,
    test_support::{
        assertions::assert_complex_close,
        jet::{HoloJ0, HoloJ1, HoloJ2},
        planar::dielectric_interface,
    },
};

const K0: Complex64 = Complex64::new(2.0, 0.2);
const K_PARALLEL: Complex64 = Complex64::new(0.3, 0.1);

fn value_coordinates(k0: Complex64, k_parallel: Complex64) -> CanonicalCoordinates<HoloJ0> {
    CanonicalCoordinates::new(
        HoloJ0::constant(arr0(k0)),
        HoloJ0::constant(arr0(k_parallel)),
    )
}

fn first_spectral_coordinates(
    k0: Complex64,
    k_parallel: Complex64,
) -> CanonicalCoordinates<HoloJ1> {
    CanonicalCoordinates::new(
        HoloJ1::variable(arr0(k0)),
        HoloJ1::constant(arr0(k_parallel)),
    )
}

fn first_in_plane_coordinates(
    k0: Complex64,
    k_parallel: Complex64,
) -> CanonicalCoordinates<HoloJ1> {
    CanonicalCoordinates::new(
        HoloJ1::constant(arr0(k0)),
        HoloJ1::variable(arr0(k_parallel)),
    )
}

fn second_spectral_coordinates(
    k0: Complex64,
    k_parallel: Complex64,
) -> CanonicalCoordinates<HoloJ2> {
    CanonicalCoordinates::new(
        HoloJ2::variable(arr0(k0)),
        HoloJ2::constant(arr0(k_parallel)),
    )
}

#[test]
fn modal_value_evaluation_returns_finite_determinant() {
    let stack = dielectric_interface(2.0);

    let evaluator =
        ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, Scatter2::new()).unwrap();

    let polarisation = Polarisation::TransverseElectric;

    let coordinates = value_coordinates(K0, K_PARALLEL);

    let exterior = evaluate_exterior_wavevectors::<ComplexPlane, _, HoloJ0>(
        &coordinates,
        evaluator.stack().left_exterior(),
        evaluator.stack().right_exterior(),
    );

    let determinant = evaluator
        .determinant(&coordinates, &exterior, polarisation)
        .expect("complex-plane evaluation should succeed");

    assert!(determinant.value()[()].is_finite());
}

#[test]
fn modal_first_spectral_derivative_is_finite() {
    let stack = dielectric_interface(2.0);

    let evaluator =
        ComplexPlaneEvaluator::<HoloJ1, _, _>::compile(&stack, Scatter2::new()).unwrap();

    let polarisation = Polarisation::TransverseElectric;

    let coordinates = first_spectral_coordinates(K0, K_PARALLEL);

    let exterior = evaluate_exterior_wavevectors::<ComplexPlane, _, HoloJ1>(
        &coordinates,
        evaluator.stack().left_exterior(),
        evaluator.stack().right_exterior(),
    );

    let determinant = evaluator
        .determinant(&coordinates, &exterior, polarisation)
        .expect("complex-plane derivative should succeed")
        .into_inner();

    assert!(determinant.first()[()].is_finite());
}

#[test]
fn modal_first_derivative_matches_complex_central_difference() {
    let stack = dielectric_interface(2.0);

    let evaluator =
        ComplexPlaneEvaluator::<HoloJ1, _, _>::compile(&stack, Scatter2::new()).unwrap();

    let polarisation = Polarisation::TransverseElectric;

    let differentiated_coordinates = first_spectral_coordinates(K0, K_PARALLEL);

    let differentiated_exterior = evaluate_exterior_wavevectors::<ComplexPlane, _, HoloJ1>(
        &differentiated_coordinates,
        evaluator.stack().left_exterior(),
        evaluator.stack().right_exterior(),
    );

    let differentiated = evaluator
        .determinant(
            &differentiated_coordinates,
            &differentiated_exterior,
            polarisation,
        )
        .unwrap()
        .into_inner();

    let step = 1.0e-6;
    let delta = Complex64::new(step, 0.0);

    let evaluate = |k0: Complex64| {
        let coordinates = CanonicalCoordinates::new(
            HoloJ1::constant(arr0(k0)),
            HoloJ1::constant(arr0(K_PARALLEL)),
        );

        let exterior = evaluate_exterior_wavevectors::<ComplexPlane, _, HoloJ1>(
            &coordinates,
            evaluator.stack().left_exterior(),
            evaluator.stack().right_exterior(),
        );

        evaluator
            .determinant(&coordinates, &exterior, polarisation)
            .unwrap()
            .into_inner()
            .value()[()]
    };

    let below = evaluate(K0 - delta);
    let above = evaluate(K0 + delta);

    let finite_difference = (above - below) / (2.0 * step);

    assert_complex_close(differentiated.first()[()], finite_difference, 1.0e-9);
}

#[test]
fn modal_second_derivative_matches_complex_central_difference() {
    let stack = dielectric_interface(2.0);

    let evaluator =
        ComplexPlaneEvaluator::<HoloJ2, _, _>::compile(&stack, Scatter2::new()).unwrap();

    let polarisation = Polarisation::TransverseElectric;

    let differentiated_coordinates = second_spectral_coordinates(K0, K_PARALLEL);

    let differentiated_exterior = evaluate_exterior_wavevectors::<ComplexPlane, _, HoloJ2>(
        &differentiated_coordinates,
        evaluator.stack().left_exterior(),
        evaluator.stack().right_exterior(),
    );

    let differentiated = evaluator
        .determinant(
            &differentiated_coordinates,
            &differentiated_exterior,
            polarisation,
        )
        .unwrap()
        .into_inner();

    let step = 2.0e-5;
    let delta = Complex64::new(step, 0.0);

    let evaluate = |k0: Complex64| {
        let coordinates = CanonicalCoordinates::new(
            HoloJ2::constant(arr0(k0)),
            HoloJ2::constant(arr0(K_PARALLEL)),
        );

        let exterior = evaluate_exterior_wavevectors::<ComplexPlane, _, HoloJ2>(
            &coordinates,
            evaluator.stack().left_exterior(),
            evaluator.stack().right_exterior(),
        );

        evaluator
            .determinant(&coordinates, &exterior, polarisation)
            .unwrap()
            .into_inner()
            .value()[()]
    };

    let below = evaluate(K0 - delta);
    let centre = evaluate(K0);
    let above = evaluate(K0 + delta);

    let finite_difference = (above - 2.0 * centre + below) / (step * step);

    assert_complex_close(differentiated.second()[()], finite_difference, 1.0e-5);
}

#[test]
fn modal_in_plane_derivative_is_supported_holomorphically() {
    let stack = dielectric_interface(2.0);

    let evaluator =
        ComplexPlaneEvaluator::<HoloJ1, _, _>::compile(&stack, Scatter2::new()).unwrap();

    let polarisation = Polarisation::TransverseElectric;

    let coordinates = first_in_plane_coordinates(K0, K_PARALLEL);

    let exterior = evaluate_exterior_wavevectors::<ComplexPlane, _, HoloJ1>(
        &coordinates,
        evaluator.stack().left_exterior(),
        evaluator.stack().right_exterior(),
    );

    let determinant = evaluator
        .determinant(&coordinates, &exterior, polarisation)
        .expect("in-plane modal derivative should succeed")
        .into_inner();

    assert!(determinant.first()[()].is_finite());
}
