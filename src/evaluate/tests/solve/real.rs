use crate::{IncidentSide, Polarisation, RealAxisEvaluator, backend::Scatter2};

use crate::test_support::{
    TOLERANCE,
    assertions::{assert_complex_close, assert_real_close},
    planar::{dielectric_interface, fresnel_amplitudes, fresnel_power, scalar_real_input},
};

fn evaluator() -> RealAxisEvaluator<Scatter2> {
    RealAxisEvaluator::new(Scatter2::new())
}

#[test]
fn value_evaluation_matches_te_fresnel_interface() {
    let evaluator = evaluator();

    let stack = dielectric_interface(2.0);
    let input = scalar_real_input(2.0, 0.0);

    let result = evaluator
        .evaluate(input, &stack, Polarisation::TransverseElectric)
        .expect("evaluation should succeed");

    let amplitudes = result.amplitudes(IncidentSide::Left).unwrap();

    let (expected_r, expected_t) = fresnel_amplitudes(1.0, 2.0, Polarisation::TransverseElectric);

    assert_complex_close(amplitudes.reflection()[()], expected_r, TOLERANCE);

    assert_complex_close(amplitudes.transmission()[()], expected_t, TOLERANCE);
}

#[test]
fn value_evaluation_matches_tm_fresnel_interface() {
    let evaluator = evaluator();

    let stack = dielectric_interface(2.0);
    let input = scalar_real_input(2.0, 0.0);

    let result = evaluator
        .evaluate(input, &stack, Polarisation::TransverseMagnetic)
        .expect("evaluation should succeed");

    let amplitudes = result.amplitudes(IncidentSide::Left).unwrap();

    let (expected_r, expected_t) = fresnel_amplitudes(1.0, 2.0, Polarisation::TransverseMagnetic);

    assert_complex_close(amplitudes.reflection()[()], expected_r, TOLERANCE);

    assert_complex_close(amplitudes.transmission()[()], expected_t, TOLERANCE);
}

#[test]
fn power_matches_fresnel_coefficients() {
    let evaluator = evaluator();

    let stack = dielectric_interface(2.0);
    let input = scalar_real_input(2.0, 0.0);

    let result = evaluator
        .evaluate(input, &stack, Polarisation::TransverseElectric)
        .expect("evaluation should succeed");

    let power = result.power(IncidentSide::Left).unwrap();

    let (expected_r, expected_t, expected_a) =
        fresnel_power(1.0, 2.0, Polarisation::TransverseElectric);

    assert_real_close(power.reflectance()[()], expected_r, TOLERANCE);

    assert_real_close(power.transmittance()[()], expected_t, TOLERANCE);

    assert_real_close(power.absorptance()[()], expected_a, TOLERANCE);
}

#[test]
fn lossless_interface_conserves_power() {
    let evaluator = evaluator();

    let stack = dielectric_interface(2.0);
    let input = scalar_real_input(2.0, 0.0);

    let result = evaluator
        .evaluate(input, &stack, Polarisation::TransverseElectric)
        .expect("evaluation should succeed");

    let power = result.power(IncidentSide::Left).unwrap();
    let power = power;

    let total = power.reflectance()[()] + power.transmittance()[()] + power.absorptance()[()];

    assert_real_close(total, 1.0, TOLERANCE);
    assert_real_close(power.absorptance()[()], 0.0, TOLERANCE);
}

#[test]
fn right_incidence_uses_reversed_exterior_normalisation() {
    let evaluator = evaluator();

    let stack = dielectric_interface(2.0);
    let input = scalar_real_input(2.0, 0.0);

    let result = evaluator
        .evaluate(input, &stack, Polarisation::TransverseElectric)
        .expect("evaluation should succeed");

    let amplitudes = result.amplitudes(IncidentSide::Right).unwrap();

    let (expected_r, expected_t) = fresnel_amplitudes(2.0, 1.0, Polarisation::TransverseElectric);

    assert_complex_close(amplitudes.reflection()[()], expected_r, TOLERANCE);

    assert_complex_close(amplitudes.transmission()[()], expected_t, TOLERANCE);

    let power = result.power(IncidentSide::Right).unwrap();

    let total = power.reflectance()[()] + power.transmittance()[()] + power.absorptance()[()];

    assert_real_close(total, 1.0, TOLERANCE);
}
