use ndarray::{ArrayBase, Ix0, OwnedRepr, arr0};

use crate::{
    CanonicalCoordinates, ComplexPlaneEvaluator, FiniteLayerIndex, IncidentSide, Parameter,
    PlaneWaveAmplitudes, PlaneWaveDeterminant, PlaneWavePower, Polarisation, RealAxisEvaluator,
    backend::{Scatter2, Transfer2},
    test_support::{
        C,
        assertions::{assert_complex_close, assert_real_close},
        jet::{HoloJ0, HoloJ1, HoloJ2},
        planar::{
            dielectric_interface, principal_exterior_wavevectors, scalar_real_input,
            single_layer_stack, two_layer_stack,
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

fn complex_value_coordinates() -> CanonicalCoordinates<HoloJ0> {
    CanonicalCoordinates::new(
        HoloJ0::constant(arr0(C::new(2.2, 0.15))),
        HoloJ0::constant(arr0(C::new(0.3, -0.08))),
    )
}

fn complex_first_spectral_coordinates() -> CanonicalCoordinates<HoloJ1> {
    CanonicalCoordinates::new(
        HoloJ1::variable(arr0(C::new(2.2, 0.15))),
        HoloJ1::constant(arr0(C::new(0.3, -0.08))),
    )
}

fn complex_second_spectral_coordinates() -> CanonicalCoordinates<HoloJ2> {
    CanonicalCoordinates::new(
        HoloJ2::variable(arr0(C::new(2.2, 0.15))),
        HoloJ2::constant(arr0(C::new(0.3, -0.08))),
    )
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
                let left = RealAxisEvaluator::new($left_backend);

                let right = RealAxisEvaluator::new($right_backend);

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
                        let left_amplitudes = left_result.amplitudes(side).unwrap();

                        let right_amplitudes = right_result.amplitudes(side).unwrap();

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
                let left = RealAxisEvaluator::new($left_backend);

                let right = RealAxisEvaluator::new($right_backend);

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
                        let left_amplitudes = left_result.amplitudes(side).unwrap();

                        let right_amplitudes = right_result.amplitudes(side).unwrap();

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
                let left = RealAxisEvaluator::new($left_backend);

                let right = RealAxisEvaluator::new($right_backend);

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
                        let left_amplitudes = left_result.amplitudes(side).unwrap();

                        let right_amplitudes = right_result.amplitudes(side).unwrap();

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
                let left = RealAxisEvaluator::new($left_backend);

                let right = RealAxisEvaluator::new($right_backend);

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
                        let left_power = left_result.power(side).unwrap();

                        let right_power = right_result.power(side).unwrap();

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
                let left = RealAxisEvaluator::new($left_backend);

                let right = RealAxisEvaluator::new($right_backend);

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
                        let left_power = left_result.power(side).unwrap();

                        let right_power = right_result.power(side).unwrap();

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
                let left = RealAxisEvaluator::new($left_backend);

                let right = RealAxisEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let left_response = left
                    .evaluate_first(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left)
                    .unwrap();

                let right_response = right
                    .evaluate_first(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left)
                    .unwrap();

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
                let left = RealAxisEvaluator::new($left_backend);

                let right = RealAxisEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let left_response = left
                    .evaluate_first(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        Parameter::InPlane,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left)
                    .unwrap();

                let right_response = right
                    .evaluate_first(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        Parameter::InPlane,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left)
                    .unwrap();

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
                let left = RealAxisEvaluator::new($left_backend);

                let right = RealAxisEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let parameter = Parameter::LayerThickness(FiniteLayerIndex::new(0));

                let left_response = left
                    .evaluate_first(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        parameter,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left)
                    .unwrap();

                let right_response = right
                    .evaluate_first(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        parameter,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left)
                    .unwrap();

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
                let left = RealAxisEvaluator::new($left_backend);

                let right = RealAxisEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let parameter = Parameter::LayerThickness(FiniteLayerIndex::new(0));

                let left_response = left
                    .evaluate_first(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        parameter,
                    )
                    .unwrap()
                    .power(IncidentSide::Left)
                    .unwrap();

                let right_response = right
                    .evaluate_first(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        parameter,
                    )
                    .unwrap()
                    .power(IncidentSide::Left)
                    .unwrap();

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
                let left = RealAxisEvaluator::new($left_backend);

                let right = RealAxisEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let left_response = left
                    .evaluate_second(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left)
                    .unwrap();

                let right_response = right
                    .evaluate_second(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left)
                    .unwrap();

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
                let left = RealAxisEvaluator::new($left_backend);

                let right = RealAxisEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let parameter = Parameter::LayerThickness(FiniteLayerIndex::new(0));

                let left_response = left
                    .evaluate_second(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        parameter,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Right)
                    .unwrap();

                let right_response = right
                    .evaluate_second(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        parameter,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Right)
                    .unwrap();

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
                let left = RealAxisEvaluator::new($left_backend);

                let right = RealAxisEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let left_response = left
                    .evaluate_second(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .power(IncidentSide::Left)
                    .unwrap();

                let right_response = right
                    .evaluate_second(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .power(IncidentSide::Left)
                    .unwrap();

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
                let left = RealAxisEvaluator::new($left_backend);

                let right = RealAxisEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let axis0 = Parameter::Spectral;

                let axis1 = Parameter::LayerThickness(FiniteLayerIndex::new(0));

                let left_response = left
                    .evaluate_bivariate_first(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        axis0,
                        axis1,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left)
                    .unwrap();

                let right_response = right
                    .evaluate_bivariate_first(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        axis0,
                        axis1,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left)
                    .unwrap();

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
                let left = RealAxisEvaluator::new($left_backend);

                let right = RealAxisEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let axis0 = Parameter::Spectral;

                let axis1 = Parameter::LayerThickness(FiniteLayerIndex::new(0));

                let left_response = left
                    .evaluate_bivariate_second(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        axis0,
                        axis1,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left)
                    .unwrap();

                let right_response = right
                    .evaluate_bivariate_second(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        axis0,
                        axis1,
                    )
                    .unwrap()
                    .amplitudes(IncidentSide::Left)
                    .unwrap();

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
                let left = RealAxisEvaluator::new($left_backend);

                let right = RealAxisEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.17);

                let axis0 = Parameter::Spectral;

                let axis1 = Parameter::LayerThickness(FiniteLayerIndex::new(0));

                let left_response = left
                    .evaluate_bivariate_second(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        axis0,
                        axis1,
                    )
                    .unwrap()
                    .power(IncidentSide::Left)
                    .unwrap();

                let right_response = right
                    .evaluate_bivariate_second(
                        scalar_real_input(2.2, 0.25),
                        &stack,
                        Polarisation::TransverseElectric,
                        axis0,
                        axis1,
                    )
                    .unwrap()
                    .power(IncidentSide::Left)
                    .unwrap();

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
                let left = RealAxisEvaluator::new($left_backend);

                let right = RealAxisEvaluator::new($right_backend);

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
                    let left_amplitudes = left_state.excitation(side).unwrap().amplitudes();

                    let right_amplitudes = right_state.excitation(side).unwrap().amplitudes();

                    assert_amplitudes_equal(
                        left_amplitudes.value(),
                        right_amplitudes.value(),
                        VALUE_TOLERANCE,
                    );

                    let left_power = left_state.excitation(side).unwrap().power();

                    let right_power = right_state.excitation(side).unwrap().power();

                    assert_power_equal(left_power.value(), right_power.value(), VALUE_TOLERANCE);
                }
            }

            #[test]
            fn retained_second_order_external_responses_match() {
                let left = RealAxisEvaluator::new($left_backend);

                let right = RealAxisEvaluator::new($right_backend);

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

                let left_response = left_state
                    .excitation(IncidentSide::Left)
                    .unwrap()
                    .amplitudes();

                let right_response = right_state
                    .excitation(IncidentSide::Left)
                    .unwrap()
                    .amplitudes();

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
            fn complex_plane_determinants_match() {
                let stack = single_layer_stack(1.8, 0.17);

                let left =
                    ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, $left_backend).unwrap();

                let right =
                    ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, $right_backend).unwrap();

                let polarisation = Polarisation::TransverseElectric;

                let left_coordinates = complex_value_coordinates();

                let right_coordinates = complex_value_coordinates();

                let left_exterior = principal_exterior_wavevectors(left.stack(), &left_coordinates);

                let right_exterior =
                    principal_exterior_wavevectors(right.stack(), &right_coordinates);

                let left_determinant = left
                    .determinant(&left_coordinates, &left_exterior, polarisation)
                    .unwrap();

                let right_determinant = right
                    .determinant(&right_coordinates, &right_exterior, polarisation)
                    .unwrap();

                assert_complex_close(
                    left_determinant.value()[()],
                    right_determinant.value()[()],
                    VALUE_TOLERANCE,
                );
            }

            #[test]
            fn complex_plane_first_spectral_determinant_derivatives_match() {
                let stack = single_layer_stack(1.8, 0.17);

                let left =
                    ComplexPlaneEvaluator::<HoloJ1, _, _>::compile(&stack, $left_backend).unwrap();

                let right =
                    ComplexPlaneEvaluator::<HoloJ1, _, _>::compile(&stack, $right_backend).unwrap();

                let polarisation = Polarisation::TransverseElectric;

                let left_coordinates = complex_first_spectral_coordinates();

                let right_coordinates = complex_first_spectral_coordinates();

                let left_exterior = principal_exterior_wavevectors(left.stack(), &left_coordinates);

                let right_exterior =
                    principal_exterior_wavevectors(right.stack(), &right_coordinates);

                let left_determinant = left
                    .determinant(&left_coordinates, &left_exterior, polarisation)
                    .unwrap()
                    .into_inner();

                let right_determinant = right
                    .determinant(&right_coordinates, &right_exterior, polarisation)
                    .unwrap()
                    .into_inner();

                assert_complex_close(
                    left_determinant.value()[()],
                    right_determinant.value()[()],
                    VALUE_TOLERANCE,
                );

                assert_complex_close(
                    left_determinant.first()[()],
                    right_determinant.first()[()],
                    FIRST_TOLERANCE,
                );
            }

            #[test]
            fn complex_plane_second_spectral_determinant_derivatives_match() {
                let stack = single_layer_stack(1.8, 0.17);

                let left =
                    ComplexPlaneEvaluator::<HoloJ2, _, _>::compile(&stack, $left_backend).unwrap();

                let right =
                    ComplexPlaneEvaluator::<HoloJ2, _, _>::compile(&stack, $right_backend).unwrap();

                let polarisation = Polarisation::TransverseMagnetic;

                let left_coordinates = complex_second_spectral_coordinates();

                let right_coordinates = complex_second_spectral_coordinates();

                let left_exterior = principal_exterior_wavevectors(left.stack(), &left_coordinates);

                let right_exterior =
                    principal_exterior_wavevectors(right.stack(), &right_coordinates);

                let left_determinant = left
                    .determinant(&left_coordinates, &left_exterior, polarisation)
                    .unwrap()
                    .into_inner();

                let right_determinant = right
                    .determinant(&right_coordinates, &right_exterior, polarisation)
                    .unwrap()
                    .into_inner();

                assert_complex_close(
                    left_determinant.value()[()],
                    right_determinant.value()[()],
                    VALUE_TOLERANCE,
                );

                assert_complex_close(
                    left_determinant.first()[()],
                    right_determinant.first()[()],
                    FIRST_TOLERANCE,
                );

                assert_complex_close(
                    left_determinant.second()[()],
                    right_determinant.second()[()],
                    SECOND_TOLERANCE,
                );
            }

            #[test]
            fn retained_complex_plane_second_order_determinants_match() {
                let stack = single_layer_stack(1.8, 0.17);

                let left =
                    ComplexPlaneEvaluator::<HoloJ2, _, _>::compile(&stack, $left_backend).unwrap();

                let right =
                    ComplexPlaneEvaluator::<HoloJ2, _, _>::compile(&stack, $right_backend).unwrap();

                let polarisation = Polarisation::TransverseElectric;

                let left_coordinates = complex_second_spectral_coordinates();

                let right_coordinates = complex_second_spectral_coordinates();

                let left_exterior = principal_exterior_wavevectors(left.stack(), &left_coordinates);

                let right_exterior =
                    principal_exterior_wavevectors(right.stack(), &right_coordinates);

                let left_state = left
                    .retain(left_coordinates, left_exterior, polarisation)
                    .unwrap();

                let right_state = right
                    .retain(right_coordinates, right_exterior, polarisation)
                    .unwrap();

                let left_determinant = left_state.determinant().into_inner();

                let right_determinant = right_state.determinant().into_inner();

                assert_complex_close(
                    left_determinant.value()[()],
                    right_determinant.value()[()],
                    VALUE_TOLERANCE,
                );

                assert_complex_close(
                    left_determinant.first()[()],
                    right_determinant.first()[()],
                    FIRST_TOLERANCE,
                );

                assert_complex_close(
                    left_determinant.second()[()],
                    right_determinant.second()[()],
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
