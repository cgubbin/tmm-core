use ndarray::arr0;

use crate::{
    ComplexPlaneEvaluator, FiniteLayerIndex, Polarisation,
    algebra::{Jet0, ScalarAlgebra, SeedJet},
    backend::{ModalSolutionSource, ReconstructLayerModeWaves, Scatter2, Transfer2},
    evaluate::complex_plane::mode::raw_qnm_normalisation_unchecked,
    input::{CanonicalCoordinates, CanonicalStack, canonical::CanonicalLayer},
    test_support::{
        C,
        assertions::{assert_complex_close, assert_holo_zero_jet_close},
        finite_difference::{
            FIRST_DERIVATIVE_TOLERANCE, SECOND_DERIVATIVE_TOLERANCE, VALUE_TOLERANCE,
        },
        jet::{HoloJ0, HoloJ1, HoloJ2, HoloJB2},
        planar::{principal_exterior_wavevectors, two_layer_stack},
    },
};

const K0: C = C::new(2.5, -0.05);
const K_PARALLEL: C = C::new(0.31, 0.02);

fn value_coordinates() -> CanonicalCoordinates<HoloJ0> {
    CanonicalCoordinates::new(
        <HoloJ0 as SeedJet>::constant(arr0(K0)),
        <HoloJ0 as SeedJet>::constant(arr0(K_PARALLEL)),
    )
}

fn first_spectral_coordinates() -> CanonicalCoordinates<HoloJ1> {
    CanonicalCoordinates::new(
        <HoloJ1 as SeedJet>::variable(arr0(K0), 0).unwrap(),
        <HoloJ1 as SeedJet>::constant(arr0(K_PARALLEL)),
    )
}

fn second_spectral_coordinates() -> CanonicalCoordinates<HoloJ2> {
    CanonicalCoordinates::new(
        <HoloJ2 as SeedJet>::variable(arr0(K0), 0).unwrap(),
        <HoloJ2 as SeedJet>::constant(arr0(K_PARALLEL)),
    )
}

macro_rules! for_each_modal_backend {
    ($jet:ty, $name:ident, $stack:expr, $body:block) => {{
        {
            let $name =
                ComplexPlaneEvaluator::<$jet, _, _>::compile(&$stack, Scatter2::new()).unwrap();

            $body
        }

        {
            let $name =
                ComplexPlaneEvaluator::<$jet, _, _>::compile(&$stack, Transfer2::new()).unwrap();

            $body
        }
    }};
}

#[test]
fn modes_are_qnm_normalised_at_value_order() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let stack = two_layer_stack();

        for_each_modal_backend!(HoloJ0, evaluator, stack, {
            let coordinates = value_coordinates();

            let exterior = principal_exterior_wavevectors(evaluator.stack(), &coordinates);

            let state = evaluator
                .retain(coordinates, exterior, polarisation)
                .unwrap();

            let mode = state.mode().unwrap();

            let normalisation =
                raw_qnm_normalisation_unchecked(mode.solution(), mode.state()).unwrap();

            assert_complex_close(
                normalisation.total().value()[()],
                C::new(1.0, 0.0),
                VALUE_TOLERANCE,
            );
        });
    }
}

#[test]
fn first_order_spectral_modes_have_constant_unit_qnm_normalisation() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let stack = two_layer_stack();

        for_each_modal_backend!(HoloJ1, evaluator, stack, {
            let coordinates = first_spectral_coordinates();

            let exterior = principal_exterior_wavevectors(evaluator.stack(), &coordinates);

            let state = evaluator
                .retain(coordinates, exterior, polarisation)
                .unwrap();

            let mode = state.mode().unwrap();

            let normalisation =
                raw_qnm_normalisation_unchecked(mode.solution(), mode.state()).unwrap();

            let total = normalisation.total();

            assert_complex_close(total.value()[()], C::new(1.0, 0.0), VALUE_TOLERANCE);

            assert_complex_close(
                total.first()[()],
                C::new(0.0, 0.0),
                FIRST_DERIVATIVE_TOLERANCE,
            );
        });
    }
}

fn first_order_thickness_stack(
    layer_index: FiniteLayerIndex,
) -> CanonicalStack<crate::Constant<f64>, HoloJ1> {
    let physical = two_layer_stack();

    let layers = physical
        .layers_left_to_right()
        .iter()
        .enumerate()
        .map(|(index, layer)| {
            let thickness_cm = layer.thickness().as_centimetres();

            let thickness = if index == layer_index.get() {
                <HoloJ1 as SeedJet>::variable(arr0(C::new(thickness_cm, 0.0)), 0).unwrap()
            } else {
                <HoloJ1 as SeedJet>::constant(arr0(C::new(thickness_cm, 0.0)))
            };

            CanonicalLayer::new(*layer.material(), thickness)
        })
        .collect();

    CanonicalStack::new(
        *physical.left_exterior(),
        *physical.right_exterior(),
        layers,
    )
}

#[test]
fn first_order_geometry_modes_have_constant_unit_qnm_normalisation() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for layer_index in [FiniteLayerIndex::new(0), FiniteLayerIndex::new(1)] {
            {
                let stack = first_order_thickness_stack(layer_index);

                let evaluator = ComplexPlaneEvaluator::from_canonical_stack(stack, Scatter2::new());

                let coordinates = CanonicalCoordinates::new(
                    <HoloJ1 as SeedJet>::constant(arr0(K0)),
                    <HoloJ1 as SeedJet>::constant(arr0(K_PARALLEL)),
                );

                let exterior = principal_exterior_wavevectors(evaluator.stack(), &coordinates);

                let state = evaluator
                    .retain(coordinates, exterior, polarisation)
                    .unwrap();

                let mode = state.mode().unwrap();

                let normalisation =
                    raw_qnm_normalisation_unchecked(mode.solution(), mode.state()).unwrap();

                let total = normalisation.total();

                assert_complex_close(total.value()[()], C::new(1.0, 0.0), VALUE_TOLERANCE);

                assert_complex_close(
                    total.first()[()],
                    C::new(0.0, 0.0),
                    FIRST_DERIVATIVE_TOLERANCE,
                );
            }

            {
                let stack = first_order_thickness_stack(layer_index);

                let evaluator =
                    ComplexPlaneEvaluator::from_canonical_stack(stack, Transfer2::new());

                let coordinates = CanonicalCoordinates::new(
                    <HoloJ1 as SeedJet>::constant(arr0(K0)),
                    <HoloJ1 as SeedJet>::constant(arr0(K_PARALLEL)),
                );

                let exterior = principal_exterior_wavevectors(evaluator.stack(), &coordinates);

                let state = evaluator
                    .retain(coordinates, exterior, polarisation)
                    .unwrap();

                let mode = state.mode().unwrap();

                let normalisation =
                    raw_qnm_normalisation_unchecked(mode.solution(), mode.state()).unwrap();

                let total = normalisation.total();

                assert_complex_close(total.value()[()], C::new(1.0, 0.0), VALUE_TOLERANCE);

                assert_complex_close(
                    total.first()[()],
                    C::new(0.0, 0.0),
                    FIRST_DERIVATIVE_TOLERANCE,
                );
            }
        }
    }
}

#[test]
fn second_order_spectral_modes_have_constant_unit_qnm_normalisation() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let stack = two_layer_stack();

        for_each_modal_backend!(HoloJ2, evaluator, stack, {
            let coordinates = second_spectral_coordinates();

            let exterior = principal_exterior_wavevectors(evaluator.stack(), &coordinates);

            let state = evaluator
                .retain(coordinates, exterior, polarisation)
                .unwrap();

            let mode = state.mode().unwrap();

            let normalisation =
                raw_qnm_normalisation_unchecked(mode.solution(), mode.state()).unwrap();

            let total = normalisation.total();

            assert_complex_close(total.value()[()], C::new(1.0, 0.0), VALUE_TOLERANCE);

            assert_complex_close(
                total.first()[()],
                C::new(0.0, 0.0),
                FIRST_DERIVATIVE_TOLERANCE,
            );

            assert_complex_close(
                total.second()[()],
                C::new(0.0, 0.0),
                SECOND_DERIVATIVE_TOLERANCE,
            );
        });
    }
}

#[test]
fn scatter_layer_mode_reconstruction_is_linear_in_candidate_scale() {
    let stack = two_layer_stack();

    let evaluator =
        ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, Scatter2::new()).unwrap();

    let polarisation = Polarisation::TransverseElectric;

    let coordinates = value_coordinates();

    let exterior = principal_exterior_wavevectors(evaluator.stack(), &coordinates);

    let state = evaluator
        .retain(coordinates, exterior, polarisation)
        .unwrap();

    let workspace = state.workspace();

    let candidate = workspace.modal_boundary_solution().unwrap();

    let scale = Jet0::filled_constant_like(candidate.state().field().value(), C::new(0.7, -0.4));

    let scaled_candidate = candidate.clone().scaled(&scale);

    let raw = workspace.reconstruct_layer_mode_waves(&candidate).unwrap();

    let scaled = workspace
        .reconstruct_layer_mode_waves(&scaled_candidate)
        .unwrap();

    for (raw, scaled) in raw.iter().zip(&scaled) {
        assert_holo_zero_jet_close(
            scaled.left().forward(),
            &raw.left().forward().multiply(&scale),
        );

        assert_holo_zero_jet_close(
            scaled.left().backward(),
            &raw.left().backward().multiply(&scale),
        );

        assert_holo_zero_jet_close(
            scaled.right().forward(),
            &raw.right().forward().multiply(&scale),
        );

        assert_holo_zero_jet_close(
            scaled.right().backward(),
            &raw.right().backward().multiply(&scale),
        );
    }
}

#[test]
fn qnm_normalisation_is_quadratic_in_candidate_scale() {
    let stack = two_layer_stack();

    let evaluator =
        ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, Scatter2::new()).unwrap();

    let polarisation = Polarisation::TransverseElectric;

    let coordinates = value_coordinates();

    let exterior = principal_exterior_wavevectors(evaluator.stack(), &coordinates);

    let state = evaluator
        .retain(coordinates, exterior, polarisation)
        .unwrap();

    let candidate = state.workspace().modal_boundary_solution().unwrap();

    let raw = raw_qnm_normalisation_unchecked(&candidate, &state).unwrap();

    let scale = Jet0::filled_constant_like(candidate.state().field().value(), C::new(0.7, -0.4));

    let scaled_candidate = candidate.scaled(&scale);

    let scaled = raw_qnm_normalisation_unchecked(&scaled_candidate, &state).unwrap();

    let expected = raw.total().multiply(&scale).multiply(&scale);

    assert_complex_close(
        scaled.total().value()[()],
        expected.value()[()],
        VALUE_TOLERANCE,
    );
}

fn bivariate_stack() -> CanonicalStack<crate::Constant<f64>, HoloJB2> {
    let physical = two_layer_stack();

    let layers = physical
        .layers_left_to_right()
        .iter()
        .enumerate()
        .map(|(index, layer)| {
            let value = arr0(C::new(layer.thickness().as_centimetres(), 0.0));

            let thickness = if index == 1 {
                <HoloJB2 as SeedJet>::variable(value, 1).unwrap()
            } else {
                <HoloJB2 as SeedJet>::constant(value)
            };

            CanonicalLayer::new(*layer.material(), thickness)
        })
        .collect();

    CanonicalStack::new(
        *physical.left_exterior(),
        *physical.right_exterior(),
        layers,
    )
}

#[test]
fn bivariate_modes_have_constant_unit_qnm_normalisation() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        {
            let evaluator =
                ComplexPlaneEvaluator::from_canonical_stack(bivariate_stack(), Scatter2::new());

            let coordinates = CanonicalCoordinates::new(
                <HoloJB2 as SeedJet>::constant(arr0(K0)),
                <HoloJB2 as SeedJet>::constant(arr0(K_PARALLEL)),
            );

            let exterior = principal_exterior_wavevectors(evaluator.stack(), &coordinates);

            let state = evaluator
                .retain(coordinates, exterior, polarisation)
                .unwrap();

            let mode = state.mode().unwrap();

            let normalisation =
                raw_qnm_normalisation_unchecked(mode.solution(), mode.state()).unwrap();

            let total = normalisation.total();

            assert_complex_close(total.value()[()], C::new(1.0, 0.0), VALUE_TOLERANCE);

            assert_complex_close(
                total.axis0()[()],
                C::new(0.0, 0.0),
                FIRST_DERIVATIVE_TOLERANCE,
            );

            assert_complex_close(
                total.axis1()[()],
                C::new(0.0, 0.0),
                FIRST_DERIVATIVE_TOLERANCE,
            );

            assert_complex_close(
                total.axis0_axis0()[()],
                C::new(0.0, 0.0),
                SECOND_DERIVATIVE_TOLERANCE,
            );

            assert_complex_close(
                total.axis0_axis1()[()],
                C::new(0.0, 0.0),
                SECOND_DERIVATIVE_TOLERANCE,
            );

            assert_complex_close(
                total.axis1_axis1()[()],
                C::new(0.0, 0.0),
                SECOND_DERIVATIVE_TOLERANCE,
            );
        }

        {
            let evaluator =
                ComplexPlaneEvaluator::from_canonical_stack(bivariate_stack(), Transfer2::new());

            let coordinates = CanonicalCoordinates::new(
                <HoloJB2 as SeedJet>::constant(arr0(K0)),
                <HoloJB2 as SeedJet>::constant(arr0(K_PARALLEL)),
            );

            let exterior = principal_exterior_wavevectors(evaluator.stack(), &coordinates);

            let state = evaluator
                .retain(coordinates, exterior, polarisation)
                .unwrap();

            let mode = state.mode().unwrap();

            let normalisation =
                raw_qnm_normalisation_unchecked(mode.solution(), mode.state()).unwrap();

            let total = normalisation.total();

            assert_complex_close(total.value()[()], C::new(1.0, 0.0), VALUE_TOLERANCE);

            assert_complex_close(
                total.axis0()[()],
                C::new(0.0, 0.0),
                FIRST_DERIVATIVE_TOLERANCE,
            );

            assert_complex_close(
                total.axis1()[()],
                C::new(0.0, 0.0),
                FIRST_DERIVATIVE_TOLERANCE,
            );

            assert_complex_close(
                total.axis0_axis0()[()],
                C::new(0.0, 0.0),
                SECOND_DERIVATIVE_TOLERANCE,
            );

            assert_complex_close(
                total.axis0_axis1()[()],
                C::new(0.0, 0.0),
                SECOND_DERIVATIVE_TOLERANCE,
            );

            assert_complex_close(
                total.axis1_axis1()[()],
                C::new(0.0, 0.0),
                SECOND_DERIVATIVE_TOLERANCE,
            );
        }
    }
}
