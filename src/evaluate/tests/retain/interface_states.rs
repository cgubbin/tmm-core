use crate::{
    IncidentSide, Parameter, Polarisation, RealAxisEvaluator,
    parameter::FiniteLayerIndex,
    test_support::{
        assertions::assert_interface_continuity,
        finite_difference::{
            FIRST_DERIVATIVE_TOLERANCE, SECOND_DERIVATIVE_TOLERANCE, VALUE_TOLERANCE,
        },
        planar::{dielectric_interface, scalar_real_input, single_layer_stack, two_layer_stack},
    },
};

macro_rules! interface_continuity_suite {
    (
        $module:ident,
        backend = $backend:expr $(,)?
    ) => {
        mod $module {
            use super::*;

            fn assert_value_continuity(
                stack: &crate::stack::Stack<crate::material::Constant<f64>, f64>,
                polarisation: Polarisation,
                incident_side: IncidentSide,
            ) {
                let evaluator = RealAxisEvaluator::new($backend);

                let state = evaluator
                    .retain(scalar_real_input(2.5, 0.31), stack, polarisation)
                    .expect("retained evaluation should succeed");

                let response = state
                    .excitation(incident_side)
                    .expect("state should be projectable")
                    .interface_states()
                    .expect("interface states should assemble");

                assert_interface_continuity(response.value(), VALUE_TOLERANCE);
            }

            #[test]
            fn empty_stack_is_continuous_for_left_incidence() {
                assert_value_continuity(
                    &dielectric_interface(2.0),
                    Polarisation::TransverseElectric,
                    IncidentSide::Left,
                );
            }

            #[test]
            fn empty_stack_is_continuous_for_right_incidence() {
                assert_value_continuity(
                    &dielectric_interface(2.0),
                    Polarisation::TransverseMagnetic,
                    IncidentSide::Right,
                );
            }

            #[test]
            fn one_layer_te_is_continuous_from_both_sides() {
                let stack = single_layer_stack(1.8, 0.23);

                for side in [IncidentSide::Left, IncidentSide::Right] {
                    assert_value_continuity(&stack, Polarisation::TransverseElectric, side);
                }
            }

            #[test]
            fn one_layer_tm_is_continuous_from_both_sides() {
                let stack = single_layer_stack(1.8, 0.23);

                for side in [IncidentSide::Left, IncidentSide::Right] {
                    assert_value_continuity(&stack, Polarisation::TransverseMagnetic, side);
                }
            }

            #[test]
            fn two_layer_te_is_continuous_from_both_sides() {
                let stack = two_layer_stack();

                for side in [IncidentSide::Left, IncidentSide::Right] {
                    assert_value_continuity(&stack, Polarisation::TransverseElectric, side);
                }
            }

            #[test]
            fn two_layer_tm_is_continuous_from_both_sides() {
                let stack = two_layer_stack();

                for side in [IncidentSide::Left, IncidentSide::Right] {
                    assert_value_continuity(&stack, Polarisation::TransverseMagnetic, side);
                }
            }

            #[test]
            fn first_spectral_derivative_is_continuous() {
                let evaluator = RealAxisEvaluator::new($backend);

                let stack = two_layer_stack();

                let state = evaluator
                    .retain_first(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap();

                let response = state
                    .excitation(IncidentSide::Left)
                    .expect("state should be projectable")
                    .interface_states()
                    .expect("interface states should assemble");

                assert_eq!(response.derivatives().parameter(), Parameter::Spectral,);

                assert_interface_continuity(response.value(), VALUE_TOLERANCE);

                assert_interface_continuity(
                    response.derivatives().first(),
                    FIRST_DERIVATIVE_TOLERANCE,
                );
            }

            #[test]
            fn first_thickness_derivative_is_continuous() {
                let evaluator = RealAxisEvaluator::new($backend);

                let stack = two_layer_stack();

                let parameter = Parameter::LayerThickness(FiniteLayerIndex::new(1));

                let state = evaluator
                    .retain_first(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        parameter,
                    )
                    .unwrap();

                let response = state
                    .excitation(IncidentSide::Right)
                    .expect("state should be projectable")
                    .interface_states()
                    .unwrap();

                assert_eq!(response.derivatives().parameter(), parameter,);

                assert_interface_continuity(response.value(), VALUE_TOLERANCE);

                assert_interface_continuity(
                    response.derivatives().first(),
                    FIRST_DERIVATIVE_TOLERANCE,
                );
            }

            #[test]
            fn second_spectral_derivatives_are_continuous() {
                let evaluator = RealAxisEvaluator::new($backend);

                let stack = two_layer_stack();

                let state = evaluator
                    .retain_second(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap();

                let response = state
                    .excitation(IncidentSide::Left)
                    .expect("state should be projectable")
                    .interface_states()
                    .unwrap();

                assert_interface_continuity(response.value(), VALUE_TOLERANCE);

                assert_interface_continuity(
                    response.derivatives().first(),
                    FIRST_DERIVATIVE_TOLERANCE,
                );

                assert_interface_continuity(
                    response.derivatives().second(),
                    SECOND_DERIVATIVE_TOLERANCE,
                );
            }

            #[test]
            fn bivariate_second_derivatives_are_continuous() {
                let evaluator = RealAxisEvaluator::new($backend);

                let stack = two_layer_stack();

                let axis0 = Parameter::Spectral;

                let axis1 = Parameter::LayerThickness(FiniteLayerIndex::new(1));

                let state = evaluator
                    .retain_bivariate_second(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        axis0,
                        axis1,
                    )
                    .unwrap();

                let response = state
                    .excitation(IncidentSide::Right)
                    .expect("state should be projectable")
                    .interface_states()
                    .unwrap();

                assert_eq!(response.derivatives().parameters(), [axis0, axis1],);

                assert_interface_continuity(response.value(), VALUE_TOLERANCE);

                let gradient = response.derivatives().first();

                assert_interface_continuity(gradient.axis0(), FIRST_DERIVATIVE_TOLERANCE);

                assert_interface_continuity(gradient.axis1(), FIRST_DERIVATIVE_TOLERANCE);

                let hessian = response.derivatives().second();

                assert_interface_continuity(hessian.axis0_axis0(), SECOND_DERIVATIVE_TOLERANCE);

                assert_interface_continuity(hessian.axis0_axis1(), SECOND_DERIVATIVE_TOLERANCE);

                assert_interface_continuity(hessian.axis1_axis1(), SECOND_DERIVATIVE_TOLERANCE);
            }
        }
    };
}

interface_continuity_suite!(transfer2, backend = crate::backend::Transfer2::new(),);

interface_continuity_suite!(scatter2, backend = crate::backend::Scatter2::new(),);
