use ndarray::{ArrayBase, Ix1, OwnedRepr};
use num_complex::Complex64;

use crate::{
    IncidentSide, Parameter, PlaneWaveEvaluator, Polarisation,
    algebra::ScalarAlgebra,
    backend::{
        ExteriorContextProvider, RetainedIsotropicLayers, scatter2::Scatter2, transfer2::Transfer2,
    },
    field::VectorField,
    parameter::FiniteLayerIndex,
    spatial::{CanonicalLayerPosition, ExteriorSampling, FieldSampling, LayerSampling, Length},
    test_support::{
        assertions::{assert_array_close, assert_bidirectional_waves_close, assert_zero_jet_close},
        finite_difference::{
            FIRST_DERIVATIVE_TOLERANCE, SECOND_DERIVATIVE_TOLERANCE, VALUE_TOLERANCE,
        },
        planar::{scalar_real_input, two_layer_stack},
    },
    waves::{PropagateLayerWaves, ReconstructExteriorBoundaryWaves, ReconstructLayerBoundaryWaves},
};

type C = Complex64;
type ComplexArray = ArrayBase<OwnedRepr<C>, Ix1>;

fn sampling() -> FieldSampling<f64> {
    FieldSampling::new()
        .left_exterior(ExteriorSampling::point(Length::zero()))
        .layer(0, LayerSampling::uniform(3))
        .layer(1, LayerSampling::uniform(3))
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

#[test]
fn point_projection_evaluates_te_fields() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    /*
     * Field reconstruction is a point operation. Go through the same
     * retained-state projection path used by callers of a sampled solve.
     */
    let point = state.project_point(&()).unwrap();

    let excitation = point
        .excitation(IncidentSide::Left)
        .expect("state should be projectable");

    let spatial_response = excitation.evaluate_fields(&sampling()).unwrap();
    let response = spatial_response.quantity();

    /*
     * Sampling contains:
     *
     *   left exterior       1
     *   finite layer 0      3
     *   finite layer 1      3
     *   right exterior      1
     *
     * giving eight points on the final spatial axis.
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
fn point_projection_evaluates_tm_fields() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let point = state.project_point(&()).unwrap();

    let excitation = point
        .excitation(IncidentSide::Right)
        .expect("state should be projectable");

    let spatial_response = excitation.evaluate_fields(&sampling()).unwrap();
    let response = spatial_response.quantity();

    assert_eq!(response.value().electric().x().shape(), &[8]);
    assert_eq!(response.value().magnetic().y().shape(), &[8]);

    assert_tm_structure(response.value(), VALUE_TOLERANCE);
    assert_all_finite(response.value());
}

#[test]
fn both_incident_sides_evaluate_fields() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let state = evaluator
            .retain(
                scalar_real_input(2.5, 0.31),
                &two_layer_stack(),
                polarisation,
            )
            .unwrap();

        let point = state.project_point(&()).unwrap();

        for side in [IncidentSide::Left, IncidentSide::Right] {
            let excitation = point.excitation(side).expect("state should be projectable");

            let spatial_response = excitation.evaluate_fields(&sampling()).unwrap();
            let response = spatial_response.quantity();

            assert_all_finite(response.value());

            match polarisation {
                Polarisation::TransverseElectric => {
                    assert_te_structure(response.value(), VALUE_TOLERANCE);
                }

                Polarisation::TransverseMagnetic => {
                    assert_tm_structure(response.value(), VALUE_TOLERANCE);
                }
            }
        }
    }
}

#[test]
fn first_field_derivative_survives_point_projection_and_reconstruction() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let parameter = Parameter::Spectral;

    let state = evaluator
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
            parameter,
        )
        .unwrap();

    let point = state.project_point(&()).unwrap();

    let excitation = point
        .excitation(IncidentSide::Left)
        .expect("state should be projectable");

    let spatial_response = excitation.evaluate_fields(&sampling()).unwrap();
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
fn thickness_field_derivative_survives_point_projection_and_reconstruction() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let parameter = Parameter::LayerThickness(FiniteLayerIndex::new(1));

    let state = evaluator
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseMagnetic,
            parameter,
        )
        .unwrap();

    let point = state.project_point(&()).unwrap();

    let excitation = point
        .excitation(IncidentSide::Right)
        .expect("state should be projectable");

    let spatial_response = excitation.evaluate_fields(&sampling()).unwrap();
    let response = spatial_response.quantity();

    assert_eq!(response.derivatives().parameter(), parameter);

    assert_tm_structure(response.value(), VALUE_TOLERANCE);
    assert_tm_structure(response.derivatives().first(), FIRST_DERIVATIVE_TOLERANCE);

    assert_all_finite(response.derivatives().first());
}

#[test]
fn second_field_derivative_survives_point_projection_and_reconstruction() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain_second(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap();

    let point = state.project_point(&()).unwrap();

    let excitation = point
        .excitation(IncidentSide::Left)
        .expect("state should be projectable");

    let spatial_response = excitation.evaluate_fields(&sampling()).unwrap();
    let response = spatial_response.quantity();

    assert_te_structure(response.value(), VALUE_TOLERANCE);

    assert_te_structure(response.derivatives().first(), FIRST_DERIVATIVE_TOLERANCE);

    assert_te_structure(response.derivatives().second(), SECOND_DERIVATIVE_TOLERANCE);

    assert_all_finite(response.derivatives().first());
    assert_all_finite(response.derivatives().second());
}

#[test]
fn bivariate_field_derivatives_survive_point_projection_and_reconstruction() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let axis0 = Parameter::Spectral;
    let axis1 = Parameter::LayerThickness(FiniteLayerIndex::new(1));

    let state = evaluator
        .retain_bivariate_second(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseMagnetic,
            axis0,
            axis1,
        )
        .unwrap();

    let point = state.project_point(&()).unwrap();

    let excitation = point
        .excitation(IncidentSide::Right)
        .expect("state should be projectable");

    let spatial_response = excitation.evaluate_fields(&sampling()).unwrap();
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

#[test]
fn transfer_and_scatter_backends_agree_on_fields() {
    let stack = two_layer_stack();
    let sampling = sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for side in [IncidentSide::Left, IncidentSide::Right] {
            let scatter_state = PlaneWaveEvaluator::new(Scatter2::new())
                .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
                .unwrap();

            let transfer_state = PlaneWaveEvaluator::new(Transfer2::new())
                .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
                .unwrap();

            let scatter_point = scatter_state.project_point(&()).unwrap();
            let transfer_point = transfer_state.project_point(&()).unwrap();

            let scatter_spatial = scatter_point
                .excitation(side)
                .expect("scatter state should be projectable")
                .evaluate_fields(&sampling)
                .unwrap();
            let scatter = scatter_spatial.quantity();

            let transfer_spatial = transfer_point
                .excitation(side)
                .expect("transfer state should be projectable")
                .evaluate_fields(&sampling)
                .unwrap();
            let transfer = transfer_spatial.quantity();

            assert_fields_close(scatter.value(), transfer.value(), VALUE_TOLERANCE);
        }
    }
}

#[test]
fn transfer_and_scatter_backends_agree_on_first_field_derivative() {
    let stack = two_layer_stack();
    let sampling = sampling();

    let parameter = Parameter::Spectral;

    let scatter_state = PlaneWaveEvaluator::new(Scatter2::new())
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
            parameter,
        )
        .unwrap();

    let transfer_state = PlaneWaveEvaluator::new(Transfer2::new())
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
            parameter,
        )
        .unwrap();

    let scatter_spatial = scatter_state
        .project_point(&())
        .unwrap()
        .excitation(IncidentSide::Left)
        .expect("scatter state should be projectable")
        .evaluate_fields(&sampling)
        .unwrap();

    let transfer_spatial = transfer_state
        .project_point(&())
        .unwrap()
        .excitation(IncidentSide::Left)
        .expect("transfer state should be projectable")
        .evaluate_fields(&sampling)
        .unwrap();

    let scatter = scatter_spatial.quantity();
    let transfer = transfer_spatial.quantity();

    assert_fields_close(scatter.value(), transfer.value(), VALUE_TOLERANCE);

    assert_fields_close(
        scatter.derivatives().first(),
        transfer.derivatives().first(),
        FIRST_DERIVATIVE_TOLERANCE,
    );
}

fn interface_sampling() -> FieldSampling<f64> {
    FieldSampling::new()
        .left_exterior(ExteriorSampling::point(Length::zero()))
        .layer_interfaces()
        .right_exterior(ExteriorSampling::point(Length::zero()))
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

    /*
     * Tangential TM fields are Ex and Hy.
     */
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

#[test]
fn reconstructed_fields_satisfy_tangential_interface_continuity() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();
    let sampling = interface_sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let state = evaluator
            .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
            .unwrap();

        let point = state.project_point(&()).unwrap();

        for side in [IncidentSide::Left, IncidentSide::Right] {
            let spatial_response = point
                .excitation(side)
                .expect("state should be projectable")
                .evaluate_fields(&sampling)
                .unwrap();

            let response = spatial_response.quantity();
            assert_eq!(response.value().electric().x().shape(), &[6]);

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
}

#[test]
fn first_field_derivatives_satisfy_tangential_interface_continuity() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();
    let sampling = interface_sampling();

    for parameter in [
        Parameter::Spectral,
        Parameter::LayerThickness(FiniteLayerIndex::new(0)),
        Parameter::LayerThickness(FiniteLayerIndex::new(1)),
    ] {
        eprintln!("testing continuity for {parameter:?}");
        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            let state = evaluator
                .retain_first(
                    scalar_real_input(2.5, 0.31),
                    &stack,
                    polarisation,
                    parameter,
                )
                .unwrap();

            let point = state.project_point(&()).unwrap();

            for side in [IncidentSide::Left, IncidentSide::Right] {
                let spatial_response = point
                    .excitation(side)
                    .expect("state should be projectable")
                    .evaluate_fields(&sampling)
                    .unwrap();
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
}

#[test]
fn second_field_derivatives_satisfy_tangential_interface_continuity() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();
    let sampling = interface_sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let state = evaluator
            .retain_second(
                scalar_real_input(2.5, 0.31),
                &stack,
                polarisation,
                Parameter::Spectral,
            )
            .unwrap();

        let point = state.project_point(&()).unwrap();

        for side in [IncidentSide::Left, IncidentSide::Right] {
            let spatial_response = point
                .excitation(side)
                .expect("state should be projectable")
                .evaluate_fields(&sampling)
                .unwrap();
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
}

#[test]
fn transfer_backend_fields_satisfy_tangential_interface_continuity() {
    let evaluator = PlaneWaveEvaluator::new(Transfer2::new());

    let stack = two_layer_stack();
    let sampling = interface_sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let state = evaluator
            .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
            .unwrap();

        let point = state.project_point(&()).unwrap();

        for side in [IncidentSide::Left, IncidentSide::Right] {
            let spatial_response = point
                .excitation(side)
                .expect("state should be projectable")
                .evaluate_fields(&sampling)
                .unwrap();
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
}

#[test]
fn reconstructed_exterior_and_first_layer_states_match_at_left_interface() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let point = state.project_point(&()).unwrap();

    let workspace = point.workspace();

    let exterior = workspace.reconstruct_exterior_boundary_waves(IncidentSide::Left);

    let layers = workspace
        .reconstruct_layer_boundary_waves(IncidentSide::Left)
        .unwrap();

    let solution = workspace.solution();

    let exterior_admittance = solution.context().left_admittance();

    let layer_admittance = workspace.layer_quantities(0).unwrap().admittance();

    let exterior_waves: crate::observable::BoundaryWaves<_> = exterior.left().clone().into();
    let exterior_state = exterior_waves.into_state(exterior_admittance);

    let layer_waves: crate::observable::BoundaryWaves<_> = layers[0].left().clone().into();
    let layer_state = layer_waves.into_state(&layer_admittance.clone().into_inner());

    crate::test_support::assertions::assert_complex_close(
        exterior_state.field().value()[()],
        layer_state.field().value()[()],
        VALUE_TOLERANCE,
    );

    crate::test_support::assertions::assert_complex_close(
        exterior_state.secondary().value()[()],
        layer_state.secondary().value()[()],
        VALUE_TOLERANCE,
    );
}

#[test]
fn reconstructed_forward_waves_obey_layer_propagation() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let point = state.project_point(&()).unwrap();
    let workspace = point.workspace();

    let layers = workspace
        .reconstruct_layer_boundary_waves(IncidentSide::Left)
        .unwrap();

    for (index, layer) in layers.iter().enumerate() {
        let quantities = workspace
            .layer_quantities(index)
            .expect("retained layer quantities should exist");

        let thickness = workspace
            .layer_thickness(index)
            .expect("retained layer thickness should exist");

        let phase = quantities.kappa().multiply(thickness).scale(C::i()).exp();

        let expected = layer.left().forward().multiply(&phase);

        assert_zero_jet_close(layer.right().forward(), &expected);
    }
}

#[test]
fn reconstructed_backward_waves_obey_layer_propagation() {
    let evaluator = PlaneWaveEvaluator::new(Transfer2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let point = state.project_point(&()).unwrap();
    let workspace = point.workspace();

    let layers = workspace
        .reconstruct_layer_boundary_waves(IncidentSide::Left)
        .unwrap();

    for (index, layer) in layers.iter().enumerate() {
        let quantities = workspace
            .layer_quantities(index)
            .expect("retained layer quantities should exist");

        let thickness = workspace
            .layer_thickness(index)
            .expect("retained layer thickness should exist");

        let phase = quantities.kappa().multiply(thickness).scale(C::i()).exp();

        let expected = layer.right().backward().multiply(&phase);

        assert_zero_jet_close(layer.left().backward(), &expected);
    }
}

#[test]
fn reconstructed_layer_waves_obey_propagation_for_both_incident_sides() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let point = state.project_point(&()).unwrap();
    let workspace = point.workspace();

    for side in [IncidentSide::Left, IncidentSide::Right] {
        let layers = workspace.reconstruct_layer_boundary_waves(side).unwrap();

        for (index, layer) in layers.iter().enumerate() {
            let quantities = workspace.layer_quantities(index).unwrap();

            let thickness = workspace.layer_thickness(index).unwrap();

            let phase = quantities.kappa().multiply(thickness).scale(C::i()).exp();

            let expected_right_forward = layer.left().forward().multiply(&phase);

            let expected_left_backward = layer.right().backward().multiply(&phase);

            assert_zero_jet_close(layer.right().forward(), &expected_right_forward);

            assert_zero_jet_close(layer.left().backward(), &expected_left_backward);
        }
    }
}

#[test]
fn reconstructed_layer_waves_are_propagation_consistent_at_left_boundary() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let point = state.project_point(&()).unwrap();
    let workspace = point.workspace();

    let layers = workspace
        .reconstruct_layer_boundary_waves(IncidentSide::Left)
        .unwrap();

    let quantities = workspace.layer_quantities(0).unwrap();
    let thickness = workspace.layer_thickness(0).unwrap();

    let propagated = layers[0].propagate_to_position(
        quantities.kappa(),
        thickness,
        CanonicalLayerPosition::FromLeft(0.0),
    );

    assert_bidirectional_waves_close(&propagated, layers[0].left(), VALUE_TOLERANCE);
}

#[test]
fn reconstructed_layer_waves_are_propagation_consistent_at_right_boundary() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let point = state.project_point(&()).unwrap();
    let workspace = point.workspace();

    let layers = workspace
        .reconstruct_layer_boundary_waves(IncidentSide::Left)
        .unwrap();

    let quantities = workspace.layer_quantities(0).unwrap();
    let thickness = workspace.layer_thickness(0).unwrap();

    let propagated = layers[0].propagate_to_position(
        quantities.kappa(),
        thickness,
        CanonicalLayerPosition::FromRight(0.0),
    );

    assert_bidirectional_waves_close(&propagated, layers[0].right(), VALUE_TOLERANCE);
}
