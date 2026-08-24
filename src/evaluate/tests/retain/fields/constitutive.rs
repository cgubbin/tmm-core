use lamina_units::Length;
use ndarray::{Array1, Ix1, arr0};
use num_complex::Complex64;

use crate::{
    CanonicalCoordinates, ComplexPlane, ComplexPlaneEvaluator, IncidentSide, Parameter,
    Polarisation, RealAxisEvaluator, SeedJet,
    backend::{evaluate_exterior_wavevectors, scatter2::Scatter2, transfer2::Transfer2},
    field::VectorField,
    observable::ConstitutiveSamplingContext,
    parameter::FiniteLayerIndex,
    spatial::{ExteriorSampling, FieldSampling, LayerSampling},
    test_support::{
        assertions::assert_complex_close,
        finite_difference::{
            FIRST_DERIVATIVE_TOLERANCE, SECOND_DERIVATIVE_TOLERANCE, VALUE_TOLERANCE,
        },
        jet::{HoloJ0, HoloJ1},
        planar::{scalar_real_input, two_layer_stack},
    },
};

type C = Complex64;

fn sampling() -> FieldSampling<f64> {
    FieldSampling::new()
        .left_exterior(ExteriorSampling::point(Length::zero()))
        .layer(0, LayerSampling::uniform(3))
        .layer(1, LayerSampling::uniform(3))
        .right_exterior(ExteriorSampling::point(Length::zero()))
}

fn assert_complex_array_close(actual: &Array1<C>, expected: &Array1<C>, tolerance: f64) {
    assert_eq!(actual.raw_dim(), expected.raw_dim());

    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        assert_complex_close(actual, expected, tolerance);
    }
}

fn assert_vector_close(
    actual: &VectorField<C, Ix1>,
    expected: &VectorField<C, Ix1>,
    tolerance: f64,
) {
    assert_complex_array_close(actual.x(), expected.x(), tolerance);
    assert_complex_array_close(actual.y(), expected.y(), tolerance);
    assert_complex_array_close(actual.z(), expected.z(), tolerance);
}

fn multiply_vector(vector: &VectorField<C, Ix1>, scalar: &Array1<C>) -> VectorField<C, Ix1> {
    VectorField::new_unchecked(
        vector.x() * scalar,
        vector.y() * scalar,
        vector.z() * scalar,
    )
}

/*
 * (a v)' = a'v + av'
 */
fn multiply_vector_first(
    a: &Array1<C>,
    da: &Array1<C>,
    v: &VectorField<C, Ix1>,
    dv: &VectorField<C, Ix1>,
) -> VectorField<C, Ix1> {
    VectorField::new_unchecked(
        da * v.x() + a * dv.x(),
        da * v.y() + a * dv.y(),
        da * v.z() + a * dv.z(),
    )
}

/*
 * (a v)'' = a''v + 2a'v' + av''
 */
fn multiply_vector_second(
    a: &Array1<C>,
    da: &Array1<C>,
    dda: &Array1<C>,
    v: &VectorField<C, Ix1>,
    dv: &VectorField<C, Ix1>,
    ddv: &VectorField<C, Ix1>,
) -> VectorField<C, Ix1> {
    VectorField::new_unchecked(
        dda * v.x() + &(da * dv.x() * C::new(2.0, 0.0)) + &(a * ddv.x()),
        dda * v.y() + &(da * dv.y() * C::new(2.0, 0.0)) + &(a * ddv.y()),
        dda * v.z() + &(da * dv.z() * C::new(2.0, 0.0)) + &(a * ddv.z()),
    )
}

macro_rules! for_each_backend {
    ($evaluator:ident, $body:block) => {{
        {
            let $evaluator = RealAxisEvaluator::new(Scatter2::new());
            $body
        }

        {
            let $evaluator = RealAxisEvaluator::new(Transfer2::new());
            $body
        }
    }};
}

#[test]
fn excitation_constitutive_fields_equal_epsilon_e_and_mu_h() {
    let stack = two_layer_stack();
    let request = sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for side in [IncidentSide::Left, IncidentSide::Right] {
            for_each_backend!(evaluator, {
                let state = evaluator
                    .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
                    .unwrap();

                let point = state.project_point(&()).unwrap();

                let excitation = point.excitation(side).expect("state should be projectable");

                let fields = excitation.evaluate_fields(&request).unwrap();

                let constitutive = excitation.evaluate_constitutive_fields(&request).unwrap();

                let resolved = request.resolve(&stack).unwrap();

                let parameters = ConstitutiveSamplingContext::new(point.workspace())
                    .sample(&resolved)
                    .unwrap();

                assert_eq!(fields.sampling(), constitutive.sampling(),);

                let expected_d =
                    multiply_vector(fields.value().electric(), parameters.epsilon().value());

                let expected_b =
                    multiply_vector(fields.value().magnetic(), parameters.mu().value());

                assert_vector_close(
                    constitutive.value().electric_displacement(),
                    &expected_d,
                    VALUE_TOLERANCE,
                );

                assert_vector_close(
                    constitutive.value().magnetic_induction(),
                    &expected_b,
                    VALUE_TOLERANCE,
                );
            });
        }
    }
}

#[test]
fn excitation_constitutive_first_derivatives_obey_product_rule() {
    let stack = two_layer_stack();
    let request = sampling();

    for parameter in [
        Parameter::Spectral,
        Parameter::LayerThickness(FiniteLayerIndex::new(1)),
    ] {
        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            for side in [IncidentSide::Left, IncidentSide::Right] {
                for_each_backend!(evaluator, {
                    let state = evaluator
                        .retain_first(
                            scalar_real_input(2.5, 0.31),
                            &stack,
                            polarisation,
                            parameter,
                        )
                        .unwrap();

                    let point = state.project_point(&()).unwrap();

                    let excitation = point.excitation(side).expect("state should be projectable");

                    let fields = excitation.evaluate_fields(&request).unwrap();

                    let constitutive = excitation.evaluate_constitutive_fields(&request).unwrap();

                    let resolved = request.resolve(&stack).unwrap();

                    let parameters = ConstitutiveSamplingContext::new(point.workspace())
                        .sample(&resolved)
                        .unwrap();

                    let expected_d = multiply_vector_first(
                        parameters.epsilon().value(),
                        parameters.epsilon().first(),
                        fields.value().electric(),
                        fields.derivatives().first().electric(),
                    );

                    let expected_b = multiply_vector_first(
                        parameters.mu().value(),
                        parameters.mu().first(),
                        fields.value().magnetic(),
                        fields.derivatives().first().magnetic(),
                    );

                    assert_vector_close(
                        constitutive.derivatives().first().electric_displacement(),
                        &expected_d,
                        FIRST_DERIVATIVE_TOLERANCE,
                    );

                    assert_vector_close(
                        constitutive.derivatives().first().magnetic_induction(),
                        &expected_b,
                        FIRST_DERIVATIVE_TOLERANCE,
                    );
                });
            }
        }
    }
}

#[test]
fn excitation_constitutive_second_derivatives_obey_product_rule() {
    let stack = two_layer_stack();
    let request = sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for side in [IncidentSide::Left, IncidentSide::Right] {
            for_each_backend!(evaluator, {
                let state = evaluator
                    .retain_second(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        polarisation,
                        Parameter::Spectral,
                    )
                    .unwrap();

                let point = state.project_point(&()).unwrap();

                let excitation = point.excitation(side).expect("state should be projectable");

                let fields = excitation.evaluate_fields(&request).unwrap();

                let constitutive = excitation.evaluate_constitutive_fields(&request).unwrap();

                let resolved = request.resolve(&stack).unwrap();

                let parameters = ConstitutiveSamplingContext::new(point.workspace())
                    .sample(&resolved)
                    .unwrap();

                let expected_d = multiply_vector_second(
                    parameters.epsilon().value(),
                    parameters.epsilon().first(),
                    parameters.epsilon().second(),
                    fields.value().electric(),
                    fields.derivatives().first().electric(),
                    fields.derivatives().second().electric(),
                );

                let expected_b = multiply_vector_second(
                    parameters.mu().value(),
                    parameters.mu().first(),
                    parameters.mu().second(),
                    fields.value().magnetic(),
                    fields.derivatives().first().magnetic(),
                    fields.derivatives().second().magnetic(),
                );

                assert_vector_close(
                    constitutive.derivatives().second().electric_displacement(),
                    &expected_d,
                    SECOND_DERIVATIVE_TOLERANCE,
                );

                assert_vector_close(
                    constitutive.derivatives().second().magnetic_induction(),
                    &expected_b,
                    SECOND_DERIVATIVE_TOLERANCE,
                );
            });
        }
    }
}

#[test]
fn transfer_and_scatter_agree_on_excitation_constitutive_fields() {
    let stack = two_layer_stack();
    let request = sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for side in [IncidentSide::Left, IncidentSide::Right] {
            let scatter = RealAxisEvaluator::new(Scatter2::new())
                .retain_second(
                    scalar_real_input(2.5, 0.31),
                    &stack,
                    polarisation,
                    Parameter::Spectral,
                )
                .unwrap();

            let transfer = RealAxisEvaluator::new(Transfer2::new())
                .retain_second(
                    scalar_real_input(2.5, 0.31),
                    &stack,
                    polarisation,
                    Parameter::Spectral,
                )
                .unwrap();

            let scatter = scatter
                .project_point(&())
                .unwrap()
                .excitation(side)
                .unwrap()
                .evaluate_constitutive_fields(&request)
                .unwrap();

            let transfer = transfer
                .project_point(&())
                .unwrap()
                .excitation(side)
                .unwrap()
                .evaluate_constitutive_fields(&request)
                .unwrap();

            assert_vector_close(
                scatter.value().electric_displacement(),
                transfer.value().electric_displacement(),
                VALUE_TOLERANCE,
            );

            assert_vector_close(
                scatter.value().magnetic_induction(),
                transfer.value().magnetic_induction(),
                VALUE_TOLERANCE,
            );

            assert_vector_close(
                scatter.derivatives().first().electric_displacement(),
                transfer.derivatives().first().electric_displacement(),
                FIRST_DERIVATIVE_TOLERANCE,
            );

            assert_vector_close(
                scatter.derivatives().first().magnetic_induction(),
                transfer.derivatives().first().magnetic_induction(),
                FIRST_DERIVATIVE_TOLERANCE,
            );

            assert_vector_close(
                scatter.derivatives().second().electric_displacement(),
                transfer.derivatives().second().electric_displacement(),
                SECOND_DERIVATIVE_TOLERANCE,
            );

            assert_vector_close(
                scatter.derivatives().second().magnetic_induction(),
                transfer.derivatives().second().magnetic_induction(),
                SECOND_DERIVATIVE_TOLERANCE,
            );
        }
    }
}

#[test]
fn modal_constitutive_fields_equal_epsilon_e_and_mu_h() {
    let stack = two_layer_stack();
    let request = sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        {
            let evaluator =
                ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, Scatter2::new()).unwrap();

            let coordinates = CanonicalCoordinates::new(
                <HoloJ0 as SeedJet>::constant(arr0(C::new(2.5, -0.05))),
                <HoloJ0 as SeedJet>::constant(arr0(C::new(0.31, 0.02))),
            );

            let exterior = evaluate_exterior_wavevectors::<ComplexPlane, _, HoloJ0>(
                &coordinates,
                evaluator.stack().left_exterior(),
                evaluator.stack().right_exterior(),
            );

            let state = evaluator
                .retain(coordinates, exterior, polarisation)
                .unwrap();

            let mode = state.mode().unwrap();

            let fields = mode.fields(&request).unwrap();

            let constitutive = mode.constitutive_fields(&request).unwrap();

            let resolved = request.resolve_canonical(state.stack()).unwrap();

            let parameters = ConstitutiveSamplingContext::new(state.workspace())
                .sample(&resolved)
                .unwrap();

            /*
             * Complex-plane field results retain their jet structure.
             * Compare their primal components here.
             */
            let expected_d = multiply_vector(
                fields.quantity().electric().value(),
                parameters.epsilon().value(),
            );

            let expected_b = multiply_vector(
                fields.quantity().magnetic().value(),
                parameters.mu().value(),
            );

            assert_eq!(fields.sampling(), constitutive.sampling(),);

            assert_vector_close(
                constitutive.quantity().electric_displacement().value(),
                &expected_d,
                VALUE_TOLERANCE,
            );

            assert_vector_close(
                constitutive.quantity().magnetic_induction().value(),
                &expected_b,
                VALUE_TOLERANCE,
            );
        }

        {
            let evaluator =
                ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, Transfer2::new()).unwrap();

            let coordinates = CanonicalCoordinates::new(
                <HoloJ0 as SeedJet>::constant(arr0(C::new(2.5, -0.05))),
                <HoloJ0 as SeedJet>::constant(arr0(C::new(0.31, 0.02))),
            );

            let exterior = evaluate_exterior_wavevectors::<ComplexPlane, _, HoloJ0>(
                &coordinates,
                evaluator.stack().left_exterior(),
                evaluator.stack().right_exterior(),
            );

            let state = evaluator
                .retain(coordinates, exterior, polarisation)
                .unwrap();

            let mode = state.mode().unwrap();

            let fields = mode.fields(&request).unwrap();

            let constitutive = mode.constitutive_fields(&request).unwrap();

            let resolved = request.resolve_canonical(state.stack()).unwrap();

            let parameters = ConstitutiveSamplingContext::new(state.workspace())
                .sample(&resolved)
                .unwrap();

            /*
             * Complex-plane field results retain their jet structure.
             * Compare their primal components here.
             */
            let expected_d = multiply_vector(
                fields.quantity().electric().value(),
                parameters.epsilon().value(),
            );

            let expected_b = multiply_vector(
                fields.quantity().magnetic().value(),
                parameters.mu().value(),
            );

            assert_eq!(fields.sampling(), constitutive.sampling(),);

            assert_vector_close(
                constitutive.quantity().electric_displacement().value(),
                &expected_d,
                VALUE_TOLERANCE,
            );

            assert_vector_close(
                constitutive.quantity().magnetic_induction().value(),
                &expected_b,
                VALUE_TOLERANCE,
            );
        }
    }
}

#[test]
fn modal_constitutive_first_derivatives_obey_product_rule() {
    let stack = two_layer_stack();
    let request = sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        {
            let evaluator =
                ComplexPlaneEvaluator::<HoloJ1, _, _>::compile(&stack, Scatter2::new()).unwrap();

            let coordinates = CanonicalCoordinates::new(
                <HoloJ1 as SeedJet>::variable(arr0(C::new(2.5, -0.05)), 0).unwrap(),
                <HoloJ1 as SeedJet>::constant(arr0(C::new(0.31, 0.02))),
            );

            let exterior = evaluate_exterior_wavevectors::<ComplexPlane, _, HoloJ1>(
                &coordinates,
                evaluator.stack().left_exterior(),
                evaluator.stack().right_exterior(),
            );

            let state = evaluator
                .retain(coordinates, exterior, polarisation)
                .unwrap();

            let mode = state.mode().unwrap();

            let fields = mode.fields(&request).unwrap();

            let constitutive = mode.constitutive_fields(&request).unwrap();

            let resolved = request.resolve_canonical(state.stack()).unwrap();

            let parameters = ConstitutiveSamplingContext::new(state.workspace())
                .sample(&resolved)
                .unwrap();

            let epsilon = parameters.epsilon();
            let mu = parameters.mu();

            let electric = fields.quantity().electric();
            let magnetic = fields.quantity().magnetic();

            let expected_d = multiply_vector_first(
                epsilon.value(),
                epsilon.first(),
                electric.value(),
                electric.first(),
            );

            let expected_b =
                multiply_vector_first(mu.value(), mu.first(), magnetic.value(), magnetic.first());

            assert_vector_close(
                constitutive.quantity().electric_displacement().first(),
                &expected_d,
                FIRST_DERIVATIVE_TOLERANCE,
            );

            assert_vector_close(
                constitutive.quantity().magnetic_induction().first(),
                &expected_b,
                FIRST_DERIVATIVE_TOLERANCE,
            );
        }

        {
            let evaluator =
                ComplexPlaneEvaluator::<HoloJ1, _, _>::compile(&stack, Transfer2::new()).unwrap();

            let coordinates = CanonicalCoordinates::new(
                <HoloJ1 as SeedJet>::variable(arr0(C::new(2.5, -0.05)), 0).unwrap(),
                <HoloJ1 as SeedJet>::constant(arr0(C::new(0.31, 0.02))),
            );

            let exterior = evaluate_exterior_wavevectors::<ComplexPlane, _, HoloJ1>(
                &coordinates,
                evaluator.stack().left_exterior(),
                evaluator.stack().right_exterior(),
            );

            let state = evaluator
                .retain(coordinates, exterior, polarisation)
                .unwrap();

            let mode = state.mode().unwrap();

            let fields = mode.fields(&request).unwrap();

            let constitutive = mode.constitutive_fields(&request).unwrap();

            let resolved = request.resolve_canonical(state.stack()).unwrap();

            let parameters = ConstitutiveSamplingContext::new(state.workspace())
                .sample(&resolved)
                .unwrap();

            let epsilon = parameters.epsilon();
            let mu = parameters.mu();

            let electric = fields.quantity().electric();
            let magnetic = fields.quantity().magnetic();

            let expected_d = multiply_vector_first(
                epsilon.value(),
                epsilon.first(),
                electric.value(),
                electric.first(),
            );

            let expected_b =
                multiply_vector_first(mu.value(), mu.first(), magnetic.value(), magnetic.first());

            assert_vector_close(
                constitutive.quantity().electric_displacement().first(),
                &expected_d,
                FIRST_DERIVATIVE_TOLERANCE,
            );

            assert_vector_close(
                constitutive.quantity().magnetic_induction().first(),
                &expected_b,
                FIRST_DERIVATIVE_TOLERANCE,
            );
        }
    }
}
