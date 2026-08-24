use lamina_units::Length;
use nalgebra::ComplexField;
use ndarray::{Array1, Ix1};
use num_complex::Complex64;

use crate::{
    IncidentSide, Parameter, Polarisation, RealAxisEvaluator,
    algebra::Jet0,
    backend::{scatter2::Scatter2, transfer2::Transfer2},
    field::VectorField,
    parameter::FiniteLayerIndex,
    spatial::{ExteriorSampling, FieldSampling, LayerSampling},
    test_support::{
        assertions::assert_complex_close,
        finite_difference::{
            FIRST_DERIVATIVE_TOLERANCE, SECOND_DERIVATIVE_TOLERANCE, VALUE_TOLERANCE,
        },
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

fn assert_real_array_close(actual: &Array1<f64>, expected: &Array1<f64>, tolerance: f64) {
    assert_eq!(actual.raw_dim(), expected.raw_dim());

    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        approx::assert_relative_eq!(
            actual,
            expected,
            epsilon = tolerance,
            max_relative = tolerance,
        );
    }
}

fn assert_complex_array_close(actual: &Array1<C>, expected: &Array1<C>, tolerance: f64) {
    assert_eq!(actual.raw_dim(), expected.raw_dim());

    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        assert_complex_close(actual, expected, tolerance);
    }
}

fn assert_complex_vector_close(
    actual: &VectorField<C, Ix1>,
    expected: &VectorField<C, Ix1>,
    tolerance: f64,
) {
    assert_complex_array_close(actual.x(), expected.x(), tolerance);
    assert_complex_array_close(actual.y(), expected.y(), tolerance);
    assert_complex_array_close(actual.z(), expected.z(), tolerance);
}

fn assert_real_vector_close(
    actual: &VectorField<f64, Ix1>,
    expected: &VectorField<f64, Ix1>,
    tolerance: f64,
) {
    assert_real_array_close(actual.x(), expected.x(), tolerance);
    assert_real_array_close(actual.y(), expected.y(), tolerance);
    assert_real_array_close(actual.z(), expected.z(), tolerance);
}

fn assert_all_real_finite(values: &Array1<f64>) {
    assert!(values.iter().all(|value| value.is_finite()));
}

fn assert_all_complex_vector_finite(vector: &VectorField<C, Ix1>) {
    for component in [vector.x(), vector.y(), vector.z()] {
        assert!(
            component
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite())
        );
    }
}

fn assert_all_real_vector_finite(vector: &VectorField<f64, Ix1>) {
    for component in [vector.x(), vector.y(), vector.z()] {
        assert!(component.iter().all(|value| value.is_finite()));
    }
}

/*
 * For
 *
 *     q = v†v,
 *
 * the first derivative with respect to a real parameter is
 *
 *     q' = 2 Re(v†v').
 */
fn magnitude_squared_first(
    value: &VectorField<C, Ix1>,
    first: &VectorField<C, Ix1>,
) -> Array1<f64> {
    value
        .x()
        .iter()
        .zip(value.y())
        .zip(value.z())
        .zip(first.x())
        .zip(first.y())
        .zip(first.z())
        .map(|(((((&vx, &vy), &vz), &dx), &dy), &dz)| {
            2.0 * (vx.conj() * dx + vy.conj() * dy + vz.conj() * dz).re
        })
        .collect()
}

/*
 * For q = v†v,
 *
 *     q'' = 2 Re(v'†v' + v†v'').
 */
fn magnitude_squared_second(
    value: &VectorField<C, Ix1>,
    first: &VectorField<C, Ix1>,
    second: &VectorField<C, Ix1>,
) -> Array1<f64> {
    value
        .x()
        .iter()
        .zip(value.y())
        .zip(value.z())
        .zip(first.x())
        .zip(first.y())
        .zip(first.z())
        .zip(second.x())
        .zip(second.y())
        .zip(second.z())
        .map(
            |((((((((&vx, &vy), &vz), &dx), &dy), &dz), &ddx), &ddy), &ddz)| {
                2.0 * (dx.conj() * dx
                    + dy.conj() * dy
                    + dz.conj() * dz
                    + vx.conj() * ddx
                    + vy.conj() * ddy
                    + vz.conj() * ddz)
                    .re
            },
        )
        .collect()
}

fn assert_norm_values_match_fields(
    fields: &crate::observable::ElectromagneticFields<VectorField<C, Ix1>>,
    norms: &crate::observable::ElectromagneticIntensities<Array1<f64>>,
) {
    assert_real_array_close(
        norms.electric(),
        &fields.electric().magnitude_squared().into_values(),
        VALUE_TOLERANCE,
    );

    assert_real_array_close(
        norms.magnetic(),
        &fields.magnetic().magnitude_squared().into_values(),
        VALUE_TOLERANCE,
    );
}

macro_rules! for_each_backend {
    ($backend:ident, $body:block) => {{
        {
            let $backend = RealAxisEvaluator::new(Scatter2::new());
            $body
        }

        {
            let $backend = RealAxisEvaluator::new(Transfer2::new());
            $body
        }
    }};
}

#[test]
fn field_norm_values_match_electromagnetic_fields() {
    let stack = two_layer_stack();
    let sampling = sampling();

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

                let fields = excitation.evaluate_fields(&sampling).unwrap();

                let norms = excitation.evaluate_field_intensities(&sampling).unwrap();

                assert_eq!(fields.sampling(), norms.sampling());

                assert_norm_values_match_fields(fields.value(), norms.value());

                assert_all_real_finite(norms.value().electric());
                assert_all_real_finite(norms.value().magnetic());
            });
        }
    }
}

#[test]
fn complex_poynting_values_match_electromagnetic_fields() {
    let stack = two_layer_stack();
    let sampling = sampling();

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

                let fields = excitation.evaluate_fields(&sampling).unwrap();

                let poynting = excitation
                    .evaluate_complex_poynting_vector(&sampling)
                    .unwrap();

                let expected = fields
                    .value()
                    .clone()
                    .map_vectors(Jet0::new)
                    .complex_poynting_vector();

                assert_eq!(fields.sampling(), poynting.sampling());

                assert_complex_vector_close(poynting.value(), &expected, VALUE_TOLERANCE);

                assert_all_complex_vector_finite(poynting.value());
            });
        }
    }
}

#[test]
fn time_averaged_poynting_values_match_electromagnetic_fields() {
    let stack = two_layer_stack();
    let sampling = sampling();

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

                let fields = excitation.evaluate_fields(&sampling).unwrap();

                let poynting = excitation
                    .evaluate_time_averaged_poynting_vector(&sampling)
                    .unwrap();

                let expected = fields
                    .value()
                    .clone()
                    .map_vectors(Jet0::new)
                    .time_averaged_poynting_vector();

                assert_eq!(fields.sampling(), poynting.sampling());

                assert_real_vector_close(poynting.value(), &expected, VALUE_TOLERANCE);

                assert_all_real_vector_finite(poynting.value());
            });
        }
    }
}

#[test]
fn field_norm_first_derivatives_obey_product_rule() {
    let stack = two_layer_stack();
    let sampling = sampling();

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

                    let fields = excitation.evaluate_fields(&sampling).unwrap();

                    let norms = excitation.evaluate_field_intensities(&sampling).unwrap();

                    assert_eq!(norms.derivatives().parameter(), parameter,);

                    let expected_electric = magnitude_squared_first(
                        fields.value().electric(),
                        fields.derivatives().first().electric(),
                    );

                    let expected_magnetic = magnitude_squared_first(
                        fields.value().magnetic(),
                        fields.derivatives().first().magnetic(),
                    );

                    assert_real_array_close(
                        norms.derivatives().first().electric(),
                        &expected_electric,
                        FIRST_DERIVATIVE_TOLERANCE,
                    );

                    assert_real_array_close(
                        norms.derivatives().first().magnetic(),
                        &expected_magnetic,
                        FIRST_DERIVATIVE_TOLERANCE,
                    );
                });
            }
        }
    }
}

#[test]
fn field_norm_second_derivatives_obey_product_rule() {
    let stack = two_layer_stack();
    let sampling = sampling();

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

                let fields = excitation.evaluate_fields(&sampling).unwrap();

                let norms = excitation.evaluate_field_intensities(&sampling).unwrap();

                let expected_electric = magnitude_squared_second(
                    fields.value().electric(),
                    fields.derivatives().first().electric(),
                    fields.derivatives().second().electric(),
                );

                let expected_magnetic = magnitude_squared_second(
                    fields.value().magnetic(),
                    fields.derivatives().first().magnetic(),
                    fields.derivatives().second().magnetic(),
                );

                assert_real_array_close(
                    norms.derivatives().second().electric(),
                    &expected_electric,
                    SECOND_DERIVATIVE_TOLERANCE,
                );

                assert_real_array_close(
                    norms.derivatives().second().magnetic(),
                    &expected_magnetic,
                    SECOND_DERIVATIVE_TOLERANCE,
                );
            });
        }
    }
}

#[test]
fn time_averaged_poynting_is_real_part_of_complex_poynting_at_all_orders() {
    let stack = two_layer_stack();
    let sampling = sampling();

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

                let complex = excitation
                    .evaluate_complex_poynting_vector(&sampling)
                    .unwrap();

                let averaged = excitation
                    .evaluate_time_averaged_poynting_vector(&sampling)
                    .unwrap();

                assert_eq!(complex.sampling(), averaged.sampling());

                assert_real_vector_close(
                    averaged.value(),
                    &complex.value().clone().map(|v| v.re),
                    VALUE_TOLERANCE,
                );

                assert_real_vector_close(
                    averaged.derivatives().first(),
                    &complex.derivatives().first().clone().map(|v| v.re),
                    FIRST_DERIVATIVE_TOLERANCE,
                );

                assert_real_vector_close(
                    averaged.derivatives().second(),
                    &complex.derivatives().second().clone().map(|v| v.re),
                    SECOND_DERIVATIVE_TOLERANCE,
                );
            });
        }
    }
}

#[test]
fn transfer_and_scatter_agree_on_field_norms() {
    let stack = two_layer_stack();
    let sampling = sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for side in [IncidentSide::Left, IncidentSide::Right] {
            let scatter_state = RealAxisEvaluator::new(Scatter2::new())
                .retain_second(
                    scalar_real_input(2.5, 0.31),
                    &stack,
                    polarisation,
                    Parameter::Spectral,
                )
                .unwrap();

            let transfer_state = RealAxisEvaluator::new(Transfer2::new())
                .retain_second(
                    scalar_real_input(2.5, 0.31),
                    &stack,
                    polarisation,
                    Parameter::Spectral,
                )
                .unwrap();

            let scatter = scatter_state
                .project_point(&())
                .unwrap()
                .excitation(side)
                .unwrap()
                .evaluate_field_intensities(&sampling)
                .unwrap();

            let transfer = transfer_state
                .project_point(&())
                .unwrap()
                .excitation(side)
                .unwrap()
                .evaluate_field_intensities(&sampling)
                .unwrap();

            for (actual, expected) in [
                (scatter.value().electric(), transfer.value().electric()),
                (scatter.value().magnetic(), transfer.value().magnetic()),
            ] {
                assert_real_array_close(actual, expected, VALUE_TOLERANCE);
            }

            for (actual, expected) in [
                (
                    scatter.derivatives().first().electric(),
                    transfer.derivatives().first().electric(),
                ),
                (
                    scatter.derivatives().first().magnetic(),
                    transfer.derivatives().first().magnetic(),
                ),
            ] {
                assert_real_array_close(actual, expected, FIRST_DERIVATIVE_TOLERANCE);
            }

            for (actual, expected) in [
                (
                    scatter.derivatives().second().electric(),
                    transfer.derivatives().second().electric(),
                ),
                (
                    scatter.derivatives().second().magnetic(),
                    transfer.derivatives().second().magnetic(),
                ),
            ] {
                assert_real_array_close(actual, expected, SECOND_DERIVATIVE_TOLERANCE);
            }
        }
    }
}

#[test]
fn transfer_and_scatter_agree_on_poynting_vectors() {
    let stack = two_layer_stack();
    let sampling = sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for side in [IncidentSide::Left, IncidentSide::Right] {
            let scatter_state = RealAxisEvaluator::new(Scatter2::new())
                .retain_second(
                    scalar_real_input(2.5, 0.31),
                    &stack,
                    polarisation,
                    Parameter::Spectral,
                )
                .unwrap();

            let transfer_state = RealAxisEvaluator::new(Transfer2::new())
                .retain_second(
                    scalar_real_input(2.5, 0.31),
                    &stack,
                    polarisation,
                    Parameter::Spectral,
                )
                .unwrap();

            let scatter_point = scatter_state.project_point(&()).unwrap();

            let transfer_point = transfer_state.project_point(&()).unwrap();

            let scatter_excitation = scatter_point.excitation(side).unwrap();

            let transfer_excitation = transfer_point.excitation(side).unwrap();

            let scatter_complex = scatter_excitation
                .evaluate_complex_poynting_vector(&sampling)
                .unwrap();

            let transfer_complex = transfer_excitation
                .evaluate_complex_poynting_vector(&sampling)
                .unwrap();

            assert_complex_vector_close(
                scatter_complex.value(),
                transfer_complex.value(),
                VALUE_TOLERANCE,
            );

            assert_complex_vector_close(
                scatter_complex.derivatives().first(),
                transfer_complex.derivatives().first(),
                FIRST_DERIVATIVE_TOLERANCE,
            );

            assert_complex_vector_close(
                scatter_complex.derivatives().second(),
                transfer_complex.derivatives().second(),
                SECOND_DERIVATIVE_TOLERANCE,
            );

            let scatter_average = scatter_excitation
                .evaluate_time_averaged_poynting_vector(&sampling)
                .unwrap();

            let transfer_average = transfer_excitation
                .evaluate_time_averaged_poynting_vector(&sampling)
                .unwrap();

            assert_real_vector_close(
                scatter_average.value(),
                transfer_average.value(),
                VALUE_TOLERANCE,
            );

            assert_real_vector_close(
                scatter_average.derivatives().first(),
                transfer_average.derivatives().first(),
                FIRST_DERIVATIVE_TOLERANCE,
            );

            assert_real_vector_close(
                scatter_average.derivatives().second(),
                transfer_average.derivatives().second(),
                SECOND_DERIVATIVE_TOLERANCE,
            );
        }
    }
}
