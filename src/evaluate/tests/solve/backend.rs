use approx::assert_relative_eq;
use ndarray::{ArrayBase, Dimension, Ix0, OwnedRepr};
use num_complex::Complex;

use crate::{
    ComplexScalar, FiniteLayerIndex, IncidentSide, Parameter, PlaneWaveAmplitudes,
    PlaneWaveDeterminant, PlaneWaveEvaluator, PlaneWavePower, Polarisation,
    algebra::{ArrayJet0, Jet0, RealParameter},
    backend::{Backend, PlaneWaveSolution, Scatter2, Transfer2},
    derivative_parts::DerivativePartsPolicy,
    differential::IntoDifferentialResponse,
    domain::{ComplexPlane, RealAxis},
    evaluate::PlaneWaveResult,
    input::{CoordinateInput, JetMapping},
    observable::ProjectAmplitudes,
    test_support::{
        C, TOLERANCE,
        assertions::{assert_complex_close, assert_real_close},
        finite_difference::FIRST_DERIVATIVE_TOLERANCE,
        planar::{
            dielectric_interface, scalar_complex_input, scalar_real_input, single_layer_stack,
            two_layer_stack,
        },
    },
};

type D = Ix0;
type ComplexArray = ArrayBase<OwnedRepr<C>, D>;
type RealArray = ArrayBase<OwnedRepr<f64>, D>;

const VALUE_TOLERANCE: f64 = 1.0e-11;
const FIRST_TOLERANCE: f64 = 1.0e-10;
const SECOND_TOLERANCE: f64 = 1.0e-9;

fn assert_amplitudes_equal(
    left: &PlaneWaveAmplitudes<ComplexArray>,
    right: &PlaneWaveAmplitudes<ComplexArray>,
    tolerance: f64,
) {
    assert_complex_close(left.reflection()[()], right.reflection()[()], tolerance);

    assert_complex_close(left.transmission()[()], right.transmission()[()], tolerance);
}

fn assert_power_equal(
    left: &PlaneWavePower<RealArray>,
    right: &PlaneWavePower<RealArray>,
    tolerance: f64,
) {
    assert_real_close(left.reflectance()[()], right.reflectance()[()], tolerance);

    assert_real_close(
        left.transmittance()[()],
        right.transmittance()[()],
        tolerance,
    );

    assert_real_close(left.absorptance()[()], right.absorptance()[()], tolerance);
}

fn assert_determinants_equal(
    left: &PlaneWaveDeterminant<ComplexArray>,
    right: &PlaneWaveDeterminant<ComplexArray>,
    tolerance: f64,
) {
    assert_complex_close(left.value()[()], right.value()[()], tolerance);
}

macro_rules! backend_equivalence_suite {
    (
        $module:ident,
        left = $left_backend:expr,
        right = $right_backend:expr $(,)?
    ) => {
        mod $module {
            use super::*;

            #[test]
            fn empty_interface_amplitudes_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = dielectric_interface(2.0);

                for polarisation in [
                    Polarisation::TransverseElectric,
                    Polarisation::TransverseMagnetic,
                ] {
                    let left_result = left
                        .evaluate(scalar_real_input(2.0, 0.3), &stack, polarisation)
                        .expect("left backend evaluation should succeed");

                    let right_result = right
                        .evaluate(scalar_real_input(2.0, 0.3), &stack, polarisation)
                        .expect("right backend evaluation should succeed");

                    for side in [IncidentSide::Left, IncidentSide::Right] {
                        let left_amplitudes = left_result.amplitudes(side);

                        let right_amplitudes = right_result.amplitudes(side);

                        assert_amplitudes_equal(
                            left_amplitudes.value(),
                            right_amplitudes.value(),
                            VALUE_TOLERANCE,
                        );
                    }
                }
            }

            #[test]
            fn single_layer_amplitudes_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = single_layer_stack(1.7, 0.23);

                for polarisation in [
                    Polarisation::TransverseElectric,
                    Polarisation::TransverseMagnetic,
                ] {
                    let left_result = left
                        .evaluate(scalar_real_input(2.4, 0.35), &stack, polarisation)
                        .expect("left backend evaluation should succeed");

                    let right_result = right
                        .evaluate(scalar_real_input(2.4, 0.35), &stack, polarisation)
                        .expect("right backend evaluation should succeed");

                    for side in [IncidentSide::Left, IncidentSide::Right] {
                        let left_amplitudes = left_result.amplitudes(side);

                        let right_amplitudes = right_result.amplitudes(side);

                        assert_amplitudes_equal(
                            left_amplitudes.value(),
                            right_amplitudes.value(),
                            VALUE_TOLERANCE,
                        );
                    }
                }
            }

            #[test]
            fn two_layer_amplitudes_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = two_layer_stack();

                for polarisation in [
                    Polarisation::TransverseElectric,
                    Polarisation::TransverseMagnetic,
                ] {
                    let left_result = left
                        .evaluate(scalar_real_input(2.8, 0.41), &stack, polarisation)
                        .expect("left backend evaluation should succeed");

                    let right_result = right
                        .evaluate(scalar_real_input(2.8, 0.41), &stack, polarisation)
                        .expect("right backend evaluation should succeed");

                    for side in [IncidentSide::Left, IncidentSide::Right] {
                        let left_amplitudes = left_result.amplitudes(side);

                        let right_amplitudes = right_result.amplitudes(side);

                        assert_amplitudes_equal(
                            left_amplitudes.value(),
                            right_amplitudes.value(),
                            VALUE_TOLERANCE,
                        );
                    }
                }
            }

            #[test]
            fn empty_interface_power_matches() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = dielectric_interface(2.0);

                for polarisation in [
                    Polarisation::TransverseElectric,
                    Polarisation::TransverseMagnetic,
                ] {
                    let left_result = left
                        .evaluate(scalar_real_input(2.0, 0.3), &stack, polarisation)
                        .unwrap();

                    let right_result = right
                        .evaluate(scalar_real_input(2.0, 0.3), &stack, polarisation)
                        .unwrap();

                    for side in [IncidentSide::Left, IncidentSide::Right] {
                        let left_power = left_result.power(side);

                        let right_power = right_result.power(side);

                        assert_power_equal(
                            left_power.value(),
                            right_power.value(),
                            VALUE_TOLERANCE,
                        );
                    }
                }
            }

            #[test]
            fn layered_stack_power_matches() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                for polarisation in [
                    Polarisation::TransverseElectric,
                    Polarisation::TransverseMagnetic,
                ] {
                    let left_result = left
                        .evaluate(scalar_real_input(2.2, 0.25), &stack, polarisation)
                        .unwrap();

                    let right_result = right
                        .evaluate(scalar_real_input(2.2, 0.25), &stack, polarisation)
                        .unwrap();

                    for side in [IncidentSide::Left, IncidentSide::Right] {
                        let left_power = left_result.power(side);

                        let right_power = right_result.power(side);

                        assert_power_equal(
                            left_power.value(),
                            right_power.value(),
                            VALUE_TOLERANCE,
                        );
                    }
                }
            }

            #[test]
            fn first_spectral_amplitude_derivatives_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let left_response = left
                    .evaluate_first(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left);

                let right_response = right
                    .evaluate_first(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left);

                assert_eq!(left_response.parameter(), Parameter::Spectral,);

                assert_eq!(right_response.parameter(), Parameter::Spectral,);

                assert_amplitudes_equal(
                    left_response.value(),
                    right_response.value(),
                    VALUE_TOLERANCE,
                );

                assert_amplitudes_equal(
                    left_response.first(),
                    right_response.first(),
                    FIRST_TOLERANCE,
                );
            }

            #[test]
            fn first_in_plane_amplitude_derivatives_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let left_response = left
                    .evaluate_first(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        Parameter::InPlane,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left);

                let right_response = right
                    .evaluate_first(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        Parameter::InPlane,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left);

                assert_amplitudes_equal(
                    left_response.value(),
                    right_response.value(),
                    VALUE_TOLERANCE,
                );

                assert_amplitudes_equal(
                    left_response.first(),
                    right_response.first(),
                    FIRST_TOLERANCE,
                );
            }

            #[test]
            fn first_thickness_amplitude_derivatives_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let parameter = Parameter::LayerThickness(FiniteLayerIndex(0));

                let left_response = left
                    .evaluate_first(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        parameter,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left);

                let right_response = right
                    .evaluate_first(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        parameter,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left);

                assert_eq!(left_response.parameter(), parameter,);

                assert_eq!(right_response.parameter(), parameter,);

                assert_amplitudes_equal(
                    left_response.value(),
                    right_response.value(),
                    VALUE_TOLERANCE,
                );

                assert_amplitudes_equal(
                    left_response.first(),
                    right_response.first(),
                    FIRST_TOLERANCE,
                );
            }

            #[test]
            fn first_power_derivatives_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let parameter = Parameter::LayerThickness(FiniteLayerIndex(0));

                let left_response = left
                    .evaluate_first(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        parameter,
                    )
                    .unwrap()
                    .power(IncidentSide::Left);

                let right_response = right
                    .evaluate_first(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        parameter,
                    )
                    .unwrap()
                    .power(IncidentSide::Left);

                assert_power_equal(
                    left_response.value(),
                    right_response.value(),
                    VALUE_TOLERANCE,
                );

                assert_power_equal(
                    left_response.first(),
                    right_response.first(),
                    FIRST_TOLERANCE,
                );
            }

            #[test]
            fn second_spectral_amplitude_derivatives_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let left_response = left
                    .evaluate_second(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left);

                let right_response = right
                    .evaluate_second(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left);

                assert_amplitudes_equal(
                    left_response.value(),
                    right_response.value(),
                    VALUE_TOLERANCE,
                );

                assert_amplitudes_equal(
                    left_response.first(),
                    right_response.first(),
                    FIRST_TOLERANCE,
                );

                assert_amplitudes_equal(
                    left_response.second(),
                    right_response.second(),
                    SECOND_TOLERANCE,
                );
            }

            #[test]
            fn second_thickness_amplitude_derivatives_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let parameter = Parameter::LayerThickness(FiniteLayerIndex(0));

                let left_response = left
                    .evaluate_second(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        parameter,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Right);

                let right_response = right
                    .evaluate_second(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        parameter,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Right);

                assert_amplitudes_equal(
                    left_response.value(),
                    right_response.value(),
                    VALUE_TOLERANCE,
                );

                assert_amplitudes_equal(
                    left_response.first(),
                    right_response.first(),
                    FIRST_TOLERANCE,
                );

                assert_amplitudes_equal(
                    left_response.second(),
                    right_response.second(),
                    SECOND_TOLERANCE,
                );
            }

            #[test]
            fn second_power_derivatives_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let left_response = left
                    .evaluate_second(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .power(IncidentSide::Left);

                let right_response = right
                    .evaluate_second(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .power(IncidentSide::Left);

                assert_power_equal(
                    left_response.value(),
                    right_response.value(),
                    VALUE_TOLERANCE,
                );

                assert_power_equal(
                    left_response.first(),
                    right_response.first(),
                    FIRST_TOLERANCE,
                );

                assert_power_equal(
                    left_response.second(),
                    right_response.second(),
                    SECOND_TOLERANCE,
                );
            }

            #[test]
            fn bivariate_first_amplitude_derivatives_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let axis0 = Parameter::Spectral;

                let axis1 = Parameter::LayerThickness(FiniteLayerIndex(0));

                let left_response = left
                    .evaluate_bivariate_first(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        axis0,
                        axis1,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left);

                let right_response = right
                    .evaluate_bivariate_first(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        axis0,
                        axis1,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left);

                assert_eq!(left_response.parameters(), [axis0, axis1],);

                assert_eq!(right_response.parameters(), [axis0, axis1],);

                assert_amplitudes_equal(
                    left_response.value(),
                    right_response.value(),
                    VALUE_TOLERANCE,
                );

                assert_amplitudes_equal(
                    left_response.axis0(),
                    right_response.axis0(),
                    FIRST_TOLERANCE,
                );

                assert_amplitudes_equal(
                    left_response.axis1(),
                    right_response.axis1(),
                    FIRST_TOLERANCE,
                );
            }

            #[test]
            fn bivariate_second_amplitude_derivatives_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let axis0 = Parameter::Spectral;

                let axis1 = Parameter::LayerThickness(FiniteLayerIndex(0));

                let left_response = left
                    .evaluate_bivariate_second(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        axis0,
                        axis1,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left);

                let right_response = right
                    .evaluate_bivariate_second(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        axis0,
                        axis1,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left);

                assert_eq!(left_response.parameters(), [axis0, axis1],);

                assert_eq!(right_response.parameters(), [axis0, axis1],);

                assert_amplitudes_equal(
                    left_response.value(),
                    right_response.value(),
                    VALUE_TOLERANCE,
                );

                assert_amplitudes_equal(
                    left_response.gradient().axis0(),
                    right_response.gradient().axis0(),
                    FIRST_TOLERANCE,
                );

                assert_amplitudes_equal(
                    left_response.gradient().axis1(),
                    right_response.gradient().axis1(),
                    FIRST_TOLERANCE,
                );

                assert_amplitudes_equal(
                    left_response.hessian().axis0_axis0(),
                    right_response.hessian().axis0_axis0(),
                    SECOND_TOLERANCE,
                );

                assert_amplitudes_equal(
                    left_response.hessian().axis0_axis1(),
                    right_response.hessian().axis0_axis1(),
                    SECOND_TOLERANCE,
                );

                assert_amplitudes_equal(
                    left_response.hessian().axis1_axis1(),
                    right_response.hessian().axis1_axis1(),
                    SECOND_TOLERANCE,
                );
            }

            #[test]
            fn bivariate_second_power_derivatives_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let axis0 = Parameter::Spectral;

                let axis1 = Parameter::LayerThickness(FiniteLayerIndex(0));

                let left_response = left
                    .evaluate_bivariate_second(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        axis0,
                        axis1,
                    )
                    .unwrap()
                    .power(IncidentSide::Left);

                let right_response = right
                    .evaluate_bivariate_second(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        axis0,
                        axis1,
                    )
                    .unwrap()
                    .power(IncidentSide::Left);

                assert_power_equal(
                    left_response.value(),
                    right_response.value(),
                    VALUE_TOLERANCE,
                );

                assert_power_equal(
                    left_response.gradient().axis0(),
                    right_response.gradient().axis0(),
                    FIRST_TOLERANCE,
                );

                assert_power_equal(
                    left_response.gradient().axis1(),
                    right_response.gradient().axis1(),
                    FIRST_TOLERANCE,
                );

                assert_power_equal(
                    left_response.hessian().axis0_axis0(),
                    right_response.hessian().axis0_axis0(),
                    SECOND_TOLERANCE,
                );

                assert_power_equal(
                    left_response.hessian().axis0_axis1(),
                    right_response.hessian().axis0_axis1(),
                    SECOND_TOLERANCE,
                );

                assert_power_equal(
                    left_response.hessian().axis1_axis1(),
                    right_response.hessian().axis1_axis1(),
                    SECOND_TOLERANCE,
                );
            }

            #[test]
            fn retained_value_external_responses_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = two_layer_stack();

                let left_state = left
                    .retain(
                        scalar_real_input(2.5, 0.3),
                        &stack,
                        Polarisation::TransverseElectric,
                    )
                    .unwrap();

                let right_state = right
                    .retain(
                        scalar_real_input(2.5, 0.3),
                        &stack,
                        Polarisation::TransverseElectric,
                    )
                    .unwrap();

                for side in [IncidentSide::Left, IncidentSide::Right] {
                    let left_amplitudes = left_state.amplitudes(side);

                    let right_amplitudes = right_state.amplitudes(side);

                    assert_amplitudes_equal(
                        left_amplitudes.value(),
                        right_amplitudes.value(),
                        VALUE_TOLERANCE,
                    );

                    let left_power = left_state.power(side);

                    let right_power = right_state.power(side);

                    assert_power_equal(left_power.value(), right_power.value(), VALUE_TOLERANCE);
                }
            }

            #[test]
            fn retained_second_order_external_responses_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = two_layer_stack();

                let left_state = left
                    .retain_second(
                        scalar_real_input(2.5, 0.3),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap();

                let right_state = right
                    .retain_second(
                        scalar_real_input(2.5, 0.3),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap();

                let left_response = left_state.amplitudes(IncidentSide::Left);

                let right_response = right_state.amplitudes(IncidentSide::Left);

                assert_amplitudes_equal(
                    left_response.value(),
                    right_response.value(),
                    VALUE_TOLERANCE,
                );

                assert_amplitudes_equal(
                    left_response.first(),
                    right_response.first(),
                    FIRST_TOLERANCE,
                );

                assert_amplitudes_equal(
                    left_response.second(),
                    right_response.second(),
                    SECOND_TOLERANCE,
                );
            }

            #[test]
            fn modal_determinants_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let left_response = left
                    .evaluate_modal(
                        scalar_complex_input(C::new(2.2, 0.15), C::new(0.3, -0.08)),
                        &stack,
                        Polarisation::TransverseElectric,
                    )
                    .unwrap()
                    .determinant();

                let right_response = right
                    .evaluate_modal(
                        scalar_complex_input(C::new(2.2, 0.15), C::new(0.3, -0.08)),
                        &stack,
                        Polarisation::TransverseElectric,
                    )
                    .unwrap()
                    .determinant();

                assert_determinants_equal(
                    left_response.value(),
                    right_response.value(),
                    VALUE_TOLERANCE,
                );
            }

            #[test]
            fn modal_first_derivatives_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let left_response = left
                    .evaluate_modal_first(
                        scalar_complex_input(C::new(2.2, 0.15), C::new(0.3, -0.08)),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .determinant();

                let right_response = right
                    .evaluate_modal_first(
                        scalar_complex_input(C::new(2.2, 0.15), C::new(0.3, -0.08)),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .determinant();

                assert_eq!(left_response.parameter(), Parameter::Spectral,);

                assert_eq!(right_response.parameter(), Parameter::Spectral,);

                assert_determinants_equal(
                    left_response.value(),
                    right_response.value(),
                    VALUE_TOLERANCE,
                );

                assert_determinants_equal(
                    left_response.first(),
                    right_response.first(),
                    FIRST_TOLERANCE,
                );
            }

            #[test]
            fn modal_second_derivatives_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let left_response = left
                    .evaluate_modal_second(
                        scalar_complex_input(C::new(2.2, 0.15), C::new(0.3, -0.08)),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .determinant();

                let right_response = right
                    .evaluate_modal_second(
                        scalar_complex_input(C::new(2.2, 0.15), C::new(0.3, -0.08)),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .determinant();

                assert_determinants_equal(
                    left_response.value(),
                    right_response.value(),
                    VALUE_TOLERANCE,
                );

                assert_determinants_equal(
                    left_response.first(),
                    right_response.first(),
                    FIRST_TOLERANCE,
                );

                assert_determinants_equal(
                    left_response.second(),
                    right_response.second(),
                    SECOND_TOLERANCE,
                );
            }

            #[test]
            fn retained_modal_determinants_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let left_response = left
                    .retain_modal_second(
                        scalar_complex_input(C::new(2.2, 0.15), C::new(0.3, -0.08)),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .determinant();

                let right_response = right
                    .retain_modal_second(
                        scalar_complex_input(C::new(2.2, 0.15), C::new(0.3, -0.08)),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .determinant();

                assert_determinants_equal(
                    left_response.value(),
                    right_response.value(),
                    VALUE_TOLERANCE,
                );

                assert_determinants_equal(
                    left_response.first(),
                    right_response.first(),
                    FIRST_TOLERANCE,
                );

                assert_determinants_equal(
                    left_response.second(),
                    right_response.second(),
                    SECOND_TOLERANCE,
                );
            }
        }
    };
}

backend_equivalence_suite!(
    transfer2_matches_scatter2,
    left = Transfer2::new(),
    right = Scatter2::new(),
);
