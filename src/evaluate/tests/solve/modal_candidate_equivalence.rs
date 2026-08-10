use ndarray::Ix0;

use crate::{
    CoordinateInput, PlaneWaveEvaluator, Polarisation,
    algebra::ScalarAlgebra,
    backend::{
        ExteriorContextProvider, Scatter2, Transfer2,
        transfer2::{right_outgoing_transfer_state, transfer_state_slope},
    },
    test_support::{
        C, TOLERANCE,
        assertions::assert_array_close,
        jet::J0H,
        planar::{scalar_complex_input, two_layer_stack},
    },
};

fn modal_input() -> CoordinateInput<C, Ix0> {
    scalar_complex_input(C::new(2.5, -0.05), C::new(0.31, 0.02))
}

#[test]
fn scatter_projective_modal_residual_matches_physical_s21_formula() {
    let state = PlaneWaveEvaluator::new(Scatter2::new())
        .evaluate_modal(
            modal_input(),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let entries = state.solution().entries();

    let context = state.solution().context();

    let xi_left = transfer_state_slope(context.left_admittance());

    let denominator = entries.denominator();

    let transmission_numerator = entries.n21();

    let expected = xi_left
        .multiply(denominator)
        .scale(C::new(2.0, 0.0))
        .divide(transmission_numerator);

    let actual = state.determinant();

    eprintln!("D       = {:?}", denominator.value()[()],);
    eprintln!("T       = {:?}", transmission_numerator.value()[()],);
    eprintln!("2xi D/T = {:?}", expected.value()[()],);
    eprintln!("actual  = {:?}", actual.value().value()[()],);

    assert_array_close(actual.value().value(), expected.value(), TOLERANCE);
}

#[test]
fn transfer_modal_residual_matches_right_outgoing_boundary_mismatch() {
    let state = PlaneWaveEvaluator::new(Transfer2::new())
        .evaluate_modal(
            modal_input(),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let solution = state.solution();

    let one = J0H::filled_constant_like(solution.entries().m11().value(), C::new(1.0, 0.0));

    let right = right_outgoing_transfer_state(&one, solution.context().right_admittance());

    let left = solution.entries().apply_state(&right);

    let xi_left = transfer_state_slope(solution.context().left_admittance());

    let expected = xi_left.multiply(left.field()).subtract(left.slope());

    let actual = state.determinant();

    assert_array_close(actual.value().value(), expected.value(), TOLERANCE);
}
