use crate::{
    ComplexPlane, ComplexPlaneEvaluator, Polarisation,
    algebra::ScalarAlgebra,
    backend::{
        ExteriorContextProvider, Scatter2, Transfer2, evaluate_exterior_wavevectors,
        transfer2::{right_outgoing_transfer_state, transfer_state_slope},
    },
    input::CanonicalCoordinates,
    test_support::{
        C, TOLERANCE, assertions::assert_array_close, jet::HoloJ0, planar::two_layer_stack,
    },
};

use ndarray::arr0;

fn modal_coordinates() -> CanonicalCoordinates<HoloJ0> {
    CanonicalCoordinates::new(
        HoloJ0::constant(arr0(C::new(2.5, -0.05))),
        HoloJ0::constant(arr0(C::new(0.31, 0.02))),
    )
}

#[test]
fn scatter_projective_modal_residual_matches_physical_s21_formula() {
    let stack = two_layer_stack();

    let evaluator =
        ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, Scatter2::new()).unwrap();

    let polarisation = Polarisation::TransverseElectric;

    let coordinates = modal_coordinates();

    let exterior = evaluate_exterior_wavevectors::<ComplexPlane, _, HoloJ0>(
        &coordinates,
        evaluator.stack().left_exterior(),
        evaluator.stack().right_exterior(),
    );

    let solution = evaluator
        .solve(&coordinates, &exterior, polarisation)
        .unwrap();

    let entries = solution.entries();
    let context = solution.context();

    let xi_left = transfer_state_slope(context.left_admittance());

    let denominator = entries.denominator();

    let transmission_numerator = entries.n21();

    let expected = xi_left
        .multiply(denominator)
        .scale(C::new(2.0, 0.0))
        .divide(transmission_numerator);

    let actual = solution.determinant();

    assert_array_close(actual.value(), expected.value(), TOLERANCE);
}

#[test]
fn transfer_modal_residual_matches_right_outgoing_boundary_mismatch() {
    let stack = two_layer_stack();

    let evaluator =
        ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, Transfer2::new()).unwrap();

    let polarisation = Polarisation::TransverseElectric;

    let coordinates = modal_coordinates();

    let exterior = evaluate_exterior_wavevectors::<ComplexPlane, _, HoloJ0>(
        &coordinates,
        evaluator.stack().left_exterior(),
        evaluator.stack().right_exterior(),
    );

    let solution = evaluator
        .solve(&coordinates, &exterior, polarisation)
        .unwrap();

    let one = HoloJ0::filled_constant_like(solution.entries().m11().value(), C::new(1.0, 0.0));

    let right = right_outgoing_transfer_state(&one, solution.context().right_admittance());

    let left = solution.entries().apply_state(&right);

    let xi_left = transfer_state_slope(solution.context().left_admittance());

    let expected = xi_left.multiply(left.field()).subtract(left.slope());

    let actual = solution.determinant();

    assert_array_close(actual.value(), expected.value(), TOLERANCE);
}
