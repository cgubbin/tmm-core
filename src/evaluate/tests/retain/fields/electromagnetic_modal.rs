use lamina_units::Length;
use ndarray::{ArrayBase, Ix0, Ix1, OwnedRepr};
use num_complex::Complex64;

use crate::{
    CoordinateInput, ElectromagneticFields, Parameter, PlaneWaveEvaluator, Polarisation,
    backend::{
        ExteriorContextProvider, ModalSolutionSource, ReconstructExteriorModeWaves,
        ReconstructLayerModeWaves, RetainedIsotropicLayers, Transfer2, scatter2::Scatter2,
    },
    field::VectorField,
    parameter::FiniteLayerIndex,
    spatial::{ExteriorSampling, FieldSampling, LayerSampling},
    test_support::{
        assertions::{assert_array_close, assert_complex_close},
        finite_difference::{
            FIRST_DERIVATIVE_TOLERANCE, SECOND_DERIVATIVE_TOLERANCE, VALUE_TOLERANCE,
        },
        planar::{scalar_complex_input, two_layer_stack},
    },
};

type C = Complex64;
type ComplexArray = ArrayBase<OwnedRepr<C>, Ix1>;

macro_rules! for_each_modal_backend {
    ($evaluator:ident, $body:block) => {{
        {
            let $evaluator = PlaneWaveEvaluator::new(Scatter2::new());

            $body
        }

        {
            let $evaluator = PlaneWaveEvaluator::new(Transfer2::new());

            $body
        }
    }};
}

fn modal_input() -> CoordinateInput<C, Ix0> {
    scalar_complex_input(C::new(2.5, -0.05), C::new(0.31, 0.02))
}

fn sampling() -> FieldSampling<f64> {
    FieldSampling::new()
        .left_exterior(ExteriorSampling::point(Length::zero()))
        .layer(0, LayerSampling::uniform(3))
        .layer(1, LayerSampling::uniform(3))
        .right_exterior(ExteriorSampling::point(Length::zero()))
}

fn interface_sampling() -> FieldSampling<f64> {
    FieldSampling::new()
        .left_exterior(ExteriorSampling::point(Length::zero()))
        .layer_interfaces()
        .right_exterior(ExteriorSampling::point(Length::zero()))
}

fn assert_zero(values: &ComplexArray, tolerance: f64) {
    for &value in values {
        assert!(
            value.norm() <= tolerance,
            "expected zero, got {value:?}; |value| = {:e}",
            value.norm(),
        );
    }
}

fn assert_vector_close(
    actual: &VectorField<C, Ix1>,
    expected: &VectorField<C, Ix1>,
    tolerance: f64,
) {
    assert_array_close(actual.x(), expected.x(), tolerance);
    assert_array_close(actual.y(), expected.y(), tolerance);
    assert_array_close(actual.z(), expected.z(), tolerance);
}

fn assert_fields_close(
    actual: &crate::observable::ElectromagneticFields<VectorField<C, Ix1>>,
    expected: &crate::observable::ElectromagneticFields<VectorField<C, Ix1>>,
    tolerance: f64,
) {
    assert_vector_close(actual.electric(), expected.electric(), tolerance);
    assert_vector_close(actual.magnetic(), expected.magnetic(), tolerance);
}

fn assert_te_structure(
    fields: &crate::observable::ElectromagneticFields<VectorField<C, Ix1>>,
    tolerance: f64,
) {
    assert_zero(fields.electric().x(), tolerance);
    assert_zero(fields.electric().z(), tolerance);
    assert_zero(fields.magnetic().y(), tolerance);
}

fn assert_tm_structure(
    fields: &crate::observable::ElectromagneticFields<VectorField<C, Ix1>>,
    tolerance: f64,
) {
    assert_zero(fields.electric().y(), tolerance);
    assert_zero(fields.magnetic().x(), tolerance);
    assert_zero(fields.magnetic().z(), tolerance);
}

fn assert_all_finite(fields: &crate::observable::ElectromagneticFields<VectorField<C, Ix1>>) {
    for component in [
        fields.electric().x(),
        fields.electric().y(),
        fields.electric().z(),
        fields.magnetic().x(),
        fields.magnetic().y(),
        fields.magnetic().z(),
    ] {
        assert!(
            component
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite()),
        );
    }
}

fn assert_complex_close_at_interface(
    quantity: &str,
    interface: &str,
    actual: C,
    expected: C,
    tolerance: f64,
) {
    let error = (actual - expected).norm();

    assert!(
        error <= tolerance,
        "{quantity} is discontinuous at {interface}: \
         left = {actual:?}, right = {expected:?}; \
         absolute error = {error:e}",
    );
}

fn assert_te_interface_continuity(
    fields: &crate::observable::ElectromagneticFields<VectorField<C, Ix1>>,
    tolerance: f64,
) {
    let electric = fields.electric();
    let magnetic = fields.magnetic();

    for (name, left, right) in [
        ("left exterior / layer 0", 0, 1),
        ("layer 0 / layer 1", 2, 3),
        ("layer 1 / right exterior", 4, 5),
    ] {
        assert_complex_close_at_interface(
            "Ey",
            name,
            electric.y()[left],
            electric.y()[right],
            tolerance,
        );

        assert_complex_close_at_interface(
            "Hx",
            name,
            magnetic.x()[left],
            magnetic.x()[right],
            tolerance,
        );
    }
}

fn assert_tm_interface_continuity(
    fields: &crate::observable::ElectromagneticFields<VectorField<C, Ix1>>,
    tolerance: f64,
) {
    let electric = fields.electric();
    let magnetic = fields.magnetic();

    for (name, left, right) in [
        ("left exterior / layer 0", 0, 1),
        ("layer 0 / layer 1", 2, 3),
        ("layer 1 / right exterior", 4, 5),
    ] {
        assert_complex_close_at_interface(
            "Ex",
            name,
            electric.x()[left],
            electric.x()[right],
            tolerance,
        );

        assert_complex_close_at_interface(
            "Hy",
            name,
            magnetic.y()[left],
            magnetic.y()[right],
            tolerance,
        );
    }
}

// -----------------------------------------------------------------------------
// Value reconstruction
// -----------------------------------------------------------------------------

#[test]
fn modal_te_fields_evaluate_from_retained_complex_solution() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain_modal(
            modal_input(),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let mode = state.mode().unwrap();

    let spatial_response = mode.evaluate_fields(&sampling()).unwrap();

    let response = spatial_response.quantity();

    /*
     * left exterior  1
     * layer 0        3
     * layer 1        3
     * right exterior 1
     */
    assert_eq!(response.value().electric().x().shape(), &[8]);
    assert_eq!(response.value().electric().y().shape(), &[8]);
    assert_eq!(response.value().electric().z().shape(), &[8]);

    assert_eq!(response.value().magnetic().x().shape(), &[8]);
    assert_eq!(response.value().magnetic().y().shape(), &[8]);
    assert_eq!(response.value().magnetic().z().shape(), &[8]);

    assert_te_structure(response.value(), VALUE_TOLERANCE);
    assert_all_finite(response.value());
}

#[test]
fn modal_tm_fields_evaluate_from_retained_complex_solution() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain_modal(
            modal_input(),
            &two_layer_stack(),
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let mode = state.mode().unwrap();

    let spatial_response = mode.evaluate_fields(&sampling()).unwrap();
    let response = spatial_response.quantity();

    assert_eq!(response.value().electric().x().shape(), &[8]);
    assert_eq!(response.value().magnetic().y().shape(), &[8]);

    assert_tm_structure(response.value(), VALUE_TOLERANCE);
    assert_all_finite(response.value());
}

#[test]
fn modal_fields_are_nonzero() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let state = evaluator
            .retain_modal(modal_input(), &two_layer_stack(), polarisation)
            .unwrap();

        let mode = state.mode().unwrap();

        let spatial_response = mode.evaluate_fields(&sampling()).unwrap();
        let response = spatial_response.quantity();

        let total = [
            response.value().electric().x(),
            response.value().electric().y(),
            response.value().electric().z(),
            response.value().magnetic().x(),
            response.value().magnetic().y(),
            response.value().magnetic().z(),
        ]
        .into_iter()
        .flat_map(|component| component.iter())
        .map(|value| value.norm_sqr())
        .sum::<f64>();

        assert!(
            total > VALUE_TOLERANCE,
            "modal field candidate must be nonzero",
        );
    }
}

// -----------------------------------------------------------------------------
// Differential response
// -----------------------------------------------------------------------------

#[test]
fn first_modal_field_derivative_survives_reconstruction() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let parameter = Parameter::Spectral;

    let state = evaluator
        .retain_modal_first(
            modal_input(),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
            parameter,
        )
        .unwrap();

    let mode = state.mode().unwrap();

    let spatial_response = mode.evaluate_fields(&sampling()).unwrap();
    let response = spatial_response.quantity();

    assert_eq!(response.derivatives().parameter(), parameter);

    assert_eq!(response.derivatives().first().electric().x().shape(), &[8],);
    assert_eq!(response.derivatives().first().magnetic().x().shape(), &[8],);

    assert_te_structure(response.value(), VALUE_TOLERANCE);

    assert_te_structure(response.derivatives().first(), FIRST_DERIVATIVE_TOLERANCE);

    assert_all_finite(response.value());
    assert_all_finite(response.derivatives().first());
}

#[test]
fn thickness_modal_field_derivative_survives_reconstruction() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let parameter = Parameter::LayerThickness(FiniteLayerIndex::new(1));

    let state = evaluator
        .retain_modal_first(
            modal_input(),
            &two_layer_stack(),
            Polarisation::TransverseMagnetic,
            parameter,
        )
        .unwrap();

    let mode = state.mode().unwrap();

    let spatial_response = mode.evaluate_fields(&sampling()).unwrap();
    let response = spatial_response.quantity();

    assert_eq!(response.derivatives().parameter(), parameter);

    assert_tm_structure(response.value(), VALUE_TOLERANCE);

    assert_tm_structure(response.derivatives().first(), FIRST_DERIVATIVE_TOLERANCE);

    assert_all_finite(response.derivatives().first());
}

#[test]
fn second_modal_field_derivative_survives_reconstruction() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain_modal_second(
            modal_input(),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap();

    let mode = state.mode().unwrap();

    let spatial_response = mode.evaluate_fields(&sampling()).unwrap();
    let response = spatial_response.quantity();

    assert_te_structure(response.value(), VALUE_TOLERANCE);

    assert_te_structure(response.derivatives().first(), FIRST_DERIVATIVE_TOLERANCE);

    assert_te_structure(response.derivatives().second(), SECOND_DERIVATIVE_TOLERANCE);

    assert_all_finite(response.derivatives().first());
    assert_all_finite(response.derivatives().second());
}

#[test]
fn bivariate_modal_field_derivatives_survive_reconstruction() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let axis0 = Parameter::Spectral;
    let axis1 = Parameter::LayerThickness(FiniteLayerIndex::new(1));

    let state = evaluator
        .retain_modal_bivariate_second(
            modal_input(),
            &two_layer_stack(),
            Polarisation::TransverseMagnetic,
            axis0,
            axis1,
        )
        .unwrap();

    let mode = state.mode().unwrap();

    let spatial_response = mode.evaluate_fields(&sampling()).unwrap();
    let response = spatial_response.quantity();

    let gradient = response.derivatives().first();

    assert_tm_structure(gradient.axis0(), FIRST_DERIVATIVE_TOLERANCE);
    assert_tm_structure(gradient.axis1(), FIRST_DERIVATIVE_TOLERANCE);

    let hessian = response.derivatives().second();

    assert_tm_structure(hessian.axis0_axis0(), SECOND_DERIVATIVE_TOLERANCE);
    assert_tm_structure(hessian.axis0_axis1(), SECOND_DERIVATIVE_TOLERANCE);
    assert_tm_structure(hessian.axis1_axis1(), SECOND_DERIVATIVE_TOLERANCE);

    assert_all_finite(gradient.axis0());
    assert_all_finite(gradient.axis1());

    assert_all_finite(hessian.axis0_axis0());
    assert_all_finite(hessian.axis0_axis1());
    assert_all_finite(hessian.axis1_axis1());
}

// -----------------------------------------------------------------------------
// Maxwell interface conditions
// -----------------------------------------------------------------------------

#[test]
fn modal_fields_satisfy_tangential_interface_continuity() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();
    let sampling = interface_sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let state = evaluator
            .retain_modal(modal_input(), &stack, polarisation)
            .unwrap();

        let mode = state.mode().unwrap();

        let spatial_response = mode.evaluate_fields(&sampling).unwrap();
        let response = spatial_response.quantity();

        assert_eq!(response.value().electric().x().shape(), &[6],);

        match polarisation {
            Polarisation::TransverseElectric => {
                assert_te_interface_continuity(response.value(), VALUE_TOLERANCE);
            }

            Polarisation::TransverseMagnetic => {
                assert_tm_interface_continuity(response.value(), VALUE_TOLERANCE);
            }
        }
    }
}

#[test]
fn first_modal_field_derivatives_satisfy_tangential_interface_continuity() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();
    let sampling = interface_sampling();

    for parameter in [
        Parameter::Spectral,
        Parameter::LayerThickness(FiniteLayerIndex::new(0)),
        Parameter::LayerThickness(FiniteLayerIndex::new(1)),
    ] {
        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            let state = evaluator
                .retain_modal_first(modal_input(), &stack, polarisation, parameter)
                .unwrap();

            let mode = state.mode().unwrap();

            let spatial_response = mode.evaluate_fields(&sampling).unwrap();
            let response = spatial_response.quantity();

            match polarisation {
                Polarisation::TransverseElectric => {
                    assert_te_interface_continuity(response.value(), VALUE_TOLERANCE);

                    assert_te_interface_continuity(
                        response.derivatives().first(),
                        FIRST_DERIVATIVE_TOLERANCE,
                    );
                }

                Polarisation::TransverseMagnetic => {
                    assert_tm_interface_continuity(response.value(), VALUE_TOLERANCE);

                    assert_tm_interface_continuity(
                        response.derivatives().first(),
                        FIRST_DERIVATIVE_TOLERANCE,
                    );
                }
            }
        }
    }
}

#[test]
fn second_modal_field_derivatives_satisfy_tangential_interface_continuity() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();
    let sampling = interface_sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let state = evaluator
            .retain_modal_second(modal_input(), &stack, polarisation, Parameter::Spectral)
            .unwrap();

        let mode = state.mode().unwrap();

        let spatial_response = mode.evaluate_fields(&sampling).unwrap();
        let response = spatial_response.quantity();

        match polarisation {
            Polarisation::TransverseElectric => {
                assert_te_interface_continuity(response.value(), VALUE_TOLERANCE);

                assert_te_interface_continuity(
                    response.derivatives().first(),
                    FIRST_DERIVATIVE_TOLERANCE,
                );

                assert_te_interface_continuity(
                    response.derivatives().second(),
                    SECOND_DERIVATIVE_TOLERANCE,
                );
            }

            Polarisation::TransverseMagnetic => {
                assert_tm_interface_continuity(response.value(), VALUE_TOLERANCE);

                assert_tm_interface_continuity(
                    response.derivatives().first(),
                    FIRST_DERIVATIVE_TOLERANCE,
                );

                assert_tm_interface_continuity(
                    response.derivatives().second(),
                    SECOND_DERIVATIVE_TOLERANCE,
                );
            }
        }
    }
}

#[test]
fn modal_last_layer_state_matches_right_exterior_state() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain_modal(
            modal_input(),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let workspace = state.workspace();

    let seed = workspace.modal_boundary_solution().unwrap();

    let layers = workspace.reconstruct_layer_mode_waves(&seed).unwrap();

    let exterior = workspace.reconstruct_exterior_mode_waves(&seed).unwrap();

    let last = layers.last().unwrap();

    let layer_admittance = workspace
        .layer_quantities(layers.len() - 1)
        .unwrap()
        .admittance()
        .into_inner();

    let right_admittance = workspace.solution().context().right_admittance();

    let layer_waves: crate::observable::BoundaryWaves<_> = last.right().clone().into();

    let exterior_waves: crate::observable::BoundaryWaves<_> = exterior.right().clone().into();

    let layer_state = layer_waves.into_state(&layer_admittance);
    let exterior_state = exterior_waves.into_state(right_admittance);

    eprintln!(
        "layer:    field={:?}, secondary={:?}",
        layer_state.field().value()[()],
        layer_state.secondary().value()[()],
    );

    eprintln!(
        "exterior: field={:?}, secondary={:?}",
        exterior_state.field().value()[()],
        exterior_state.secondary().value()[()],
    );

    crate::test_support::assertions::assert_complex_close(
        layer_state.field().value()[()],
        exterior_state.field().value()[()],
        VALUE_TOLERANCE,
    );

    crate::test_support::assertions::assert_complex_close(
        layer_state.secondary().value()[()],
        exterior_state.secondary().value()[()],
        VALUE_TOLERANCE,
    );
}

#[test]
fn transfer_modal_fields_satisfy_tangential_interface_continuity() {
    let evaluator = PlaneWaveEvaluator::new(Transfer2::new());

    let stack = two_layer_stack();
    let sampling = interface_sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let state = evaluator
            .retain_modal(modal_input(), &stack, polarisation)
            .unwrap();

        let mode = state.mode().unwrap();

        let spatial_response = mode.evaluate_fields(&sampling).unwrap();
        let response = spatial_response.quantity();

        match polarisation {
            Polarisation::TransverseElectric => {
                assert_te_interface_continuity(response.value(), VALUE_TOLERANCE);
            }

            Polarisation::TransverseMagnetic => {
                assert_tm_interface_continuity(response.value(), VALUE_TOLERANCE);
            }
        }
    }
}

#[test]
fn modal_field_response_preserves_resolved_sampling() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for_each_modal_backend!(evaluator, {
            let stack = two_layer_stack();

            let state = evaluator
                .retain_modal(modal_input(), &stack, polarisation)
                .unwrap();

            let requested = sampling();

            let expected = requested.resolve(&stack).unwrap();

            let mode = state.mode().unwrap();

            let response = mode.evaluate_fields(&requested).unwrap();

            assert_eq!(response.sampling(), &expected,);

            assert_eq!(
                response.quantity().value().electric().x().len(),
                response.sampling().len(),
            );

            assert_eq!(
                response.quantity().value().magnetic().x().len(),
                response.sampling().len(),
            );
        });
    }
}

#[test]
fn normalised_modal_fields_satisfy_tangential_interface_continuity() {
    let stack = two_layer_stack();
    let sampling = interface_sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for_each_modal_backend!(evaluator, {
            let state = evaluator
                .retain_modal(modal_input(), &stack, polarisation)
                .unwrap();

            let mode = state.mode().unwrap();

            let response = mode.evaluate_fields(&sampling).unwrap();

            match polarisation {
                Polarisation::TransverseElectric => {
                    assert_te_interface_continuity(response.quantity().value(), VALUE_TOLERANCE);
                }

                Polarisation::TransverseMagnetic => {
                    assert_tm_interface_continuity(response.quantity().value(), VALUE_TOLERANCE);
                }
            }
        });
    }
}

fn squared_field_difference(
    first: &ElectromagneticFields<VectorField<C, Ix1>>,
    second: &ElectromagneticFields<VectorField<C, Ix1>>,
    sign: C,
) -> f64 {
    [
        (first.electric().x(), second.electric().x()),
        (first.electric().y(), second.electric().y()),
        (first.electric().z(), second.electric().z()),
        (first.magnetic().x(), second.magnetic().x()),
        (first.magnetic().y(), second.magnetic().y()),
        (first.magnetic().z(), second.magnetic().z()),
    ]
    .into_iter()
    .flat_map(|(first, second)| {
        first
            .iter()
            .zip(second.iter())
            .map(move |(&first, &second)| (first - sign * second).norm_sqr())
    })
    .sum()
}

fn modal_relative_sign(
    first: &ElectromagneticFields<VectorField<C, Ix1>>,
    second: &ElectromagneticFields<VectorField<C, Ix1>>,
) -> C {
    let positive = C::new(1.0, 0.0);
    let negative = C::new(-1.0, 0.0);

    let positive_error = squared_field_difference(first, second, positive);

    let negative_error = squared_field_difference(first, second, negative);

    if positive_error <= negative_error {
        positive
    } else {
        negative
    }
}

fn assert_fields_close_with_sign(
    actual: &ElectromagneticFields<VectorField<C, Ix1>>,
    expected: &ElectromagneticFields<VectorField<C, Ix1>>,
    sign: C,
    tolerance: f64,
) {
    for (actual, expected) in [
        (actual.electric().x(), expected.electric().x()),
        (actual.electric().y(), expected.electric().y()),
        (actual.electric().z(), expected.electric().z()),
        (actual.magnetic().x(), expected.magnetic().x()),
        (actual.magnetic().y(), expected.magnetic().y()),
        (actual.magnetic().z(), expected.magnetic().z()),
    ] {
        assert_eq!(actual.shape(), expected.shape());

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_complex_close(actual, sign * expected, tolerance);
        }
    }
}

#[test]
fn normalised_modal_fields_agree_between_backends_up_to_global_sign() {
    let stack = two_layer_stack();
    let sampling = sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let scatter_state = PlaneWaveEvaluator::new(Scatter2::new())
            .retain_modal(modal_input(), &stack, polarisation)
            .unwrap();

        let transfer_state = PlaneWaveEvaluator::new(Transfer2::new())
            .retain_modal(modal_input(), &stack, polarisation)
            .unwrap();

        let scatter_mode = scatter_state.mode().unwrap();

        let transfer_mode = transfer_state.mode().unwrap();

        let scatter = scatter_mode.evaluate_fields(&sampling).unwrap();

        let transfer = transfer_mode.evaluate_fields(&sampling).unwrap();

        let scatter_fields = scatter.quantity().value();

        let transfer_fields = transfer.quantity().value();

        let sign = modal_relative_sign(scatter_fields, transfer_fields);

        assert_fields_close_with_sign(scatter_fields, transfer_fields, sign, VALUE_TOLERANCE);
    }
}
