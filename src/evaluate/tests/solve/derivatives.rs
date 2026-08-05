use crate::{
    FiniteLayerIndex, IncidentSide, Parameter, PlaneWaveEvaluator, Polarisation, backend::Scatter2,
};

use crate::test_support::{
    assertions::assert_complex_close,
    planar::{FILM_THICKNESS_CM, dielectric_interface, scalar_real_input, single_layer_stack},
};

const TOLERANCE: f64 = 1.0e-11;

fn evaluator() -> PlaneWaveEvaluator<Scatter2> {
    PlaneWaveEvaluator::new(Scatter2::new())
}

fn thickness_parameter() -> Parameter {
    Parameter::LayerThickness(FiniteLayerIndex(0))
}

#[test]
fn first_spectral_derivative_of_nondispersive_interface_is_zero() {
    let evaluator = evaluator();

    let stack = dielectric_interface(2.0);
    let input = scalar_real_input(2.0, 0.0);

    let result = evaluator
        .evaluate_first(
            input,
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .expect("first-derivative evaluation should succeed");

    let amplitudes = result.amplitudes(IncidentSide::Left).unwrap();

    assert_eq!(amplitudes.parameter(), Parameter::Spectral,);

    assert_complex_close(
        amplitudes.first().reflection()[()],
        num_complex::Complex64::new(0.0, 0.0),
        TOLERANCE,
    );

    assert_complex_close(
        amplitudes.first().transmission()[()],
        num_complex::Complex64::new(0.0, 0.0),
        TOLERANCE,
    );
}

#[test]
fn second_spectral_derivative_of_nondispersive_interface_is_zero() {
    let evaluator = evaluator();

    let stack = dielectric_interface(2.0);
    let input = scalar_real_input(2.0, 0.0);

    let result = evaluator
        .evaluate_second(
            input,
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .expect("second-derivative evaluation should succeed");

    let amplitudes = result.amplitudes(IncidentSide::Left).unwrap();

    assert_eq!(amplitudes.parameter(), Parameter::Spectral,);

    assert_complex_close(
        amplitudes.first().reflection()[()],
        num_complex::Complex64::new(0.0, 0.0),
        TOLERANCE,
    );

    assert_complex_close(
        amplitudes.second().reflection()[()],
        num_complex::Complex64::new(0.0, 0.0),
        TOLERANCE,
    );
}

#[test]
fn first_thickness_derivative_matches_central_difference() {
    let evaluator = evaluator();

    let input = scalar_real_input(2.0, 0.0);
    let stack = single_layer_stack(1.7, FILM_THICKNESS_CM);

    let differentiated = evaluator
        .evaluate_first(
            input.clone(),
            &stack,
            Polarisation::TransverseElectric,
            thickness_parameter(),
        )
        .expect("thickness derivative should succeed");

    let analytic = differentiated.amplitudes(IncidentSide::Left).unwrap();

    assert_eq!(analytic.parameter(), thickness_parameter(),);

    let step = 1.0e-6;

    let below = evaluator
        .evaluate(
            input.clone(),
            &single_layer_stack(1.7, FILM_THICKNESS_CM - step),
            Polarisation::TransverseElectric,
        )
        .expect("lower finite-difference sample should succeed")
        .amplitudes(IncidentSide::Left)
        .unwrap()
        .value()
        .reflection()[()];

    let above = evaluator
        .evaluate(
            input,
            &single_layer_stack(1.7, FILM_THICKNESS_CM + step),
            Polarisation::TransverseElectric,
        )
        .expect("upper finite-difference sample should succeed")
        .amplitudes(IncidentSide::Left)
        .unwrap()
        .value()
        .reflection()[()];

    let finite_difference = (above - below) / (2.0 * step);

    assert_complex_close(analytic.first().reflection()[()], finite_difference, 1.0e-6);
}

// #[test]
// fn second_thickness_derivative_matches_central_difference() {
//     let evaluator = evaluator();

//     let input = scalar_real_input(2.0, 0.0);
//     let stack = single_layer_stack(1.7, FILM_THICKNESS_CM);

//     let differentiated = evaluator
//         .evaluate_second(
//             input.clone(),
//             &stack,
//             Polarisation::TransverseElectric,
//             thickness_parameter(),
//         )
//         .expect("second thickness derivative should succeed");

//     let analytic = differentiated.amplitudes(IncidentSide::Left).unwrap();

//     let step = 2.0e-5;

//     let below = evaluator
//         .evaluate(
//             input.clone(),
//             &single_layer_stack(1.7, FILM_THICKNESS_CM - step),
//             Polarisation::TransverseElectric,
//         )
//         .unwrap()
//         .amplitudes(IncidentSide::Left).unwrap()
//         .value()
//         .reflection()[()];

//     let centre = evaluator
//         .evaluate(input.clone(), &stack, Polarisation::TransverseElectric)
//         .unwrap()
//         .amplitudes(IncidentSide::Left).unwrap()
//         .value()
//         .reflection()[()];

//     let above = evaluator
//         .evaluate(
//             input,
//             &single_layer_stack(1.7, FILM_THICKNESS_CM + step),
//             Polarisation::TransverseElectric,
//         )
//         .unwrap()
//         .amplitudes(IncidentSide::Left).unwrap()
//         .value()
//         .reflection()[()];

//     let finite_difference = (above - 2.0 * centre + below) / (step * step);

//     assert_complex_close(
//         analytic.second().reflection()[()],
//         finite_difference,
//         1.0e-4,
//     );
// }

// #[test]
// fn bivariate_first_preserves_requested_axis_order() {
//     let evaluator = evaluator();

//     let stack = single_layer_stack(1.7, FILM_THICKNESS_CM);

//     let result = evaluator
//         .evaluate_bivariate_first(
//             scalar_real_input(2.0, 0.0),
//             &stack,
//             Polarisation::TransverseElectric,
//             Parameter::Spectral,
//             thickness_parameter(),
//         )
//         .expect("bivariate evaluation should succeed");

//     let amplitudes = result.amplitudes(IncidentSide::Left).unwrap();

//     assert_eq!(
//         amplitudes.parameters(),
//         [Parameter::Spectral, thickness_parameter(),],
//     );

//     // The nondispersive slab still depends on k₀ through its phase, so both
//     // derivatives are generally nonzero. The important assertion here is that
//     // both branches are finite and axis metadata are retained.
//     assert!(amplitudes.axis0().reflection()[()].is_finite(),);

//     assert!(amplitudes.axis1().reflection()[()].is_finite(),);
// }

// #[test]
// fn swapping_bivariate_parameters_swaps_first_derivative_axes() {
//     let evaluator = evaluator();

//     let stack = single_layer_stack(1.7, FILM_THICKNESS_CM);

//     let first_order = evaluator
//         .evaluate_bivariate_first(
//             scalar_real_input(2.0, 0.0),
//             &stack,
//             Polarisation::TransverseElectric,
//             Parameter::Spectral,
//             thickness_parameter(),
//         )
//         .unwrap()
//         .amplitudes(IncidentSide::Left).unwrap();

//     let reversed = evaluator
//         .evaluate_bivariate_first(
//             scalar_real_input(2.0, 0.0),
//             &stack,
//             Polarisation::TransverseElectric,
//             thickness_parameter(),
//             Parameter::Spectral,
//         )
//         .unwrap()
//         .amplitudes(IncidentSide::Left).unwrap();

//     assert_eq!(
//         reversed.parameters(),
//         [thickness_parameter(), Parameter::Spectral,],
//     );

//     assert_complex_close(
//         first_order.axis0().reflection()[()],
//         reversed.axis1().reflection()[()],
//         TOLERANCE,
//     );

//     assert_complex_close(
//         first_order.axis1().reflection()[()],
//         reversed.axis0().reflection()[()],
//         TOLERANCE,
//     );
// }

// #[test]
// fn swapping_bivariate_parameters_transposes_hessian_storage() {
//     let evaluator = evaluator();

//     let stack = single_layer_stack(1.7, FILM_THICKNESS_CM);

//     let forward = evaluator
//         .evaluate_bivariate_second(
//             scalar_real_input(2.0, 0.0),
//             &stack,
//             Polarisation::TransverseElectric,
//             Parameter::Spectral,
//             thickness_parameter(),
//         )
//         .unwrap()
//         .amplitudes(IncidentSide::Left).unwrap();

//     let reversed = evaluator
//         .evaluate_bivariate_second(
//             scalar_real_input(2.0, 0.0),
//             &stack,
//             Polarisation::TransverseElectric,
//             thickness_parameter(),
//             Parameter::Spectral,
//         )
//         .unwrap()
//         .amplitudes(IncidentSide::Left).unwrap();

//     assert_complex_close(
//         forward.gradient().axis0().reflection()[()],
//         reversed.gradient().axis1().reflection()[()],
//         TOLERANCE,
//     );

//     assert_complex_close(
//         forward.gradient().axis1().reflection()[()],
//         reversed.gradient().axis0().reflection()[()],
//         TOLERANCE,
//     );

//     assert_complex_close(
//         forward.hessian().axis0_axis0().reflection()[()],
//         reversed.hessian().axis1_axis1().reflection()[()],
//         TOLERANCE,
//     );

//     assert_complex_close(
//         forward.hessian().axis1_axis1().reflection()[()],
//         reversed.hessian().axis0_axis0().reflection()[()],
//         TOLERANCE,
//     );

//     assert_complex_close(
//         forward.hessian().axis0_axis1().reflection()[()],
//         reversed.hessian().axis0_axis1().reflection()[()],
//         TOLERANCE,
//     );
// }

// #[test]
// fn duplicate_bivariate_parameters_are_rejected() {
//     let evaluator = evaluator();

//     let error = evaluator
//         .evaluate_bivariate_first(
//             scalar_real_input(2.0, 0.0),
//             &single_layer_stack(1.7, FILM_THICKNESS_CM),
//             Polarisation::TransverseElectric,
//             Parameter::Spectral,
//             Parameter::Spectral,
//         )
//         .expect_err("duplicate derivative parameters must fail");

//     assert!(
//         error.to_string().contains("duplicate") || error.to_string().contains("mapping"),
//         "unexpected error: {error}",
//     );
// }

// #[test]
// fn out_of_range_thickness_parameter_is_rejected_during_compilation() {
//     let evaluator = evaluator();

//     let invalid_parameter = Parameter::LayerThickness(FiniteLayerIndex(1));

//     let error = evaluator
//         .evaluate_first(
//             scalar_real_input(2.0, 0.0),
//             &single_layer_stack(1.7, FILM_THICKNESS_CM),
//             Polarisation::TransverseElectric,
//             invalid_parameter,
//         )
//         .expect_err("out-of-range layer should fail compilation");

//     assert!(
//         error.to_string().contains("layer") || error.to_string().contains("thickness"),
//         "unexpected error: {error}",
//     );
// }
