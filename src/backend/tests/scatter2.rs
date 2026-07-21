use crate::backend::field::generic_boundary_values;
use crate::backend::isotropic::IsotropicLayerQuantities;
use crate::backend::tests::support::{
    ABS_TOLERANCE, C, absorbing_stack, assert_canonical_state_close, assert_complex_array_close,
    assert_complex_close, assert_complex_derivative_close, assert_complex_second_derivative_close,
    assert_real_close, assert_real_derivative_close, boundary_samples, c, central_first_complex,
    central_first_real, central_second_complex, field_samples, first_derivative_step,
    lossless_stack, scalar_input, thickness_first_step, thickness_second_step,
    with_layer_thickness, with_vacuum_wavenumber,
};
use crate::backend::{
    ExteriorSampling, FieldSampling, IsotropicFieldState, field::generic_bidirectional_values,
};
use crate::backend::{FieldPosition, PlaneWaveFieldSample};
use crate::{
    DifferentiablePlaneWaveFieldBackend, PlaneWaveBackend, PlaneWaveFieldBackend, PlaneWaveInput,
    Stack, backend::PlaneWaveFieldResponse,
};
use crate::{
    EvaluateMaterial, IncidentSide, PlaneWaveResponse, Polarisation, Scatter2,
    SpectralDerivativeVariable, StructuralDerivativeVariable,
};

use ndarray::{Array0, Ix0};

fn solve_scatter2<M>(
    stack: &Stack<M, f64>,
    input: &PlaneWaveInput<Array0<f64>>,
) -> PlaneWaveResponse<C, Ix0>
where
    M: EvaluateMaterial<C, Real = f64>,
{
    let backend = Scatter2;

    backend
        .solve_plane_wave(stack, input)
        .expect("scatter2 should be solveable")
}

fn solve_scatter2_internal_fields<M>(
    stack: &Stack<M, f64>,
    input: &PlaneWaveInput<Array0<f64>>,
) -> PlaneWaveFieldResponse<C, Ix0>
where
    M: EvaluateMaterial<C, Real = f64>,
{
    let backend = Scatter2;

    backend
        .solve_plane_wave_internal_fields(stack, input)
        .expect("scatter2 should be solveable")
}

#[test]
fn diagnose_scatter2_first_layer_left_boundary() {
    let stack = lossless_stack();

    let input = scalar_input(Polarisation::TransverseElectric, IncidentSide::Left);

    let response = solve_scatter2_internal_fields(&stack, &input);

    let fields = response
        .sample_fields(&stack, &input, &boundary_samples())
        .unwrap();

    let generic = generic_boundary_values(response.boundary_waves().values());

    let complex_input = input.to_complex();

    let layer = &stack.layers_left_to_right()[0];

    let admittance = IsotropicLayerQuantities::real_axis(layer.material(), complex_input.planar())
        .into_admittance()
        .into_inner();

    let expected =
        IsotropicFieldState::from_waves::<C, Ix0>(generic.layer(0).unwrap().left(), &admittance);

    let actual = fields.sample(1).unwrap().canonical_state();

    assert_canonical_state_close(actual, &expected);
}

#[test]
fn scatter2_retained_layer_waves_obey_propagation_relations() {
    let stack = lossless_stack();

    let input = scalar_input(Polarisation::TransverseElectric, IncidentSide::Left);

    let response = solve_scatter2_internal_fields(&stack, &input);

    let waves = response.boundary_waves().values();

    let complex_input = input.to_complex();

    for (index, layer) in stack.layers_left_to_right().iter().enumerate() {
        let quantities =
            IsotropicLayerQuantities::real_axis(layer.material(), complex_input.planar());

        let kappa = quantities.kappa();
        let thickness = layer.thickness().as_cm();

        let retained = &waves.layers()[index];

        let phase = kappa.mapv(|kappa| (C::i() * kappa * c(thickness, 0.0)).exp());

        /*
         * Forward waves are referenced at the left boundary:
         *
         * a⁺_R = a⁺_L exp(-i κ d).
         */
        let expected_right_forward = retained.left().forward().clone() * phase.clone();

        /*
         * Backward waves are referenced at the right boundary:
         *
         * a⁻_L = a⁻_R exp(-i κ d).
         */
        let expected_left_backward = retained.right().backward().clone() * phase;

        assert_complex_array_close(retained.right().forward(), &expected_right_forward);

        assert_complex_array_close(retained.left().backward(), &expected_left_backward);
    }
}

#[test]
fn diagnose_scatter2_layer_zero_propagation() {
    let stack = lossless_stack();

    let input = scalar_input(Polarisation::TransverseElectric, IncidentSide::Left);

    let response = solve_scatter2_internal_fields(&stack, &input);

    let waves = response.boundary_waves().values();
    let retained = &waves.layers()[0];

    let complex_input = input.to_complex();

    let quantities = IsotropicLayerQuantities::real_axis(
        stack.layers_left_to_right()[0].material(),
        complex_input.planar(),
    );

    let kappa = quantities.kappa();
    let thickness = stack.layers_left_to_right()[0].thickness().as_cm();

    let phase = kappa.mapv(|kappa| (-C::i() * kappa * c(thickness, 0.0)).exp());

    let inverse_phase = kappa.mapv(|kappa| (C::i() * kappa * c(thickness, 0.0)).exp());

    let backward_from_right_expected = retained.right().backward().clone() * phase.clone();

    let backward_from_right_opposite_sign =
        retained.right().backward().clone() * inverse_phase.clone();

    let forward_from_left_expected = retained.left().forward().clone() * phase;

    let forward_from_left_opposite_sign = retained.left().forward().clone() * inverse_phase;

    dbg!(
        retained.left().forward(),
        retained.right().forward(),
        &forward_from_left_expected,
        &forward_from_left_opposite_sign,
        retained.left().backward(),
        retained.right().backward(),
        &backward_from_right_expected,
        &backward_from_right_opposite_sign,
    );
}

#[test]
fn scatter2_left_incidence_has_unit_left_incident_wave() {
    let stack = lossless_stack();
    let input = scalar_input(Polarisation::TransverseElectric, IncidentSide::Left);

    let response = solve_scatter2_internal_fields(&stack, &input);
    let waves = response.boundary_waves().values();

    assert_complex_close(waves.exterior().left().forward()[()], c(1.0, 0.0));

    assert_complex_close(waves.exterior().right().backward()[()], c(0.0, 0.0));
}

#[test]
fn scatter2_right_incidence_has_unit_right_incident_wave() {
    let stack = lossless_stack();
    let input = scalar_input(Polarisation::TransverseMagnetic, IncidentSide::Right);

    let response = solve_scatter2_internal_fields(&stack, &input);
    let waves = response.boundary_waves().values();

    assert_complex_close(waves.exterior().right().backward()[()], c(1.0, 0.0));

    assert_complex_close(waves.exterior().left().forward()[()], c(0.0, 0.0));
}

#[test]
fn scatter2_sampled_exterior_fields_match_boundary_waves() {
    use crate::backend::isotropic::IsotropicLayerQuantities;

    let stack = lossless_stack();
    let input = scalar_input(Polarisation::TransverseElectric, IncidentSide::Left);

    let response = solve_scatter2_internal_fields(&stack, &input);

    let sampling = FieldSampling::new()
        .left_exterior(ExteriorSampling::point(0.0))
        .right_exterior(ExteriorSampling::point(0.0));

    let fields = response.sample_fields(&stack, &input, &sampling).unwrap();

    let complex_input = input.to_complex();

    let left_admittance =
        IsotropicLayerQuantities::real_axis(stack.left_exterior(), complex_input.planar())
            .into_admittance()
            .into_inner();

    let right_admittance =
        IsotropicLayerQuantities::real_axis(stack.right_exterior(), complex_input.planar())
            .into_admittance()
            .into_inner();

    let waves = response.boundary_waves().values();

    let left_waves = generic_bidirectional_values(waves.exterior().left());

    let right_waves = generic_bidirectional_values(waves.exterior().right());

    let expected_left = IsotropicFieldState::from_waves::<C, Ix0>(&left_waves, &left_admittance);

    let expected_right = IsotropicFieldState::from_waves::<C, Ix0>(&right_waves, &right_admittance);

    assert_canonical_state_close(fields.sample(0).unwrap().canonical_state(), &expected_left);

    assert_canonical_state_close(fields.sample(1).unwrap().canonical_state(), &expected_right);
}

#[test]
fn scatter2_sampled_layer_boundary_fields_match_retained_waves() {
    use crate::backend::isotropic::IsotropicLayerQuantities;

    let stack = lossless_stack();
    let input = scalar_input(Polarisation::TransverseMagnetic, IncidentSide::Left);

    let response = solve_scatter2_internal_fields(&stack, &input);

    let samples = boundary_samples();

    let positions = samples.positions(&stack);
    dbg!(&positions);

    let fields = response.sample_fields(&stack, &input, &samples).unwrap();

    let complex_input = input.to_complex();
    let waves = response.boundary_waves().values();

    /*
     * Sample 0 is the left exterior. Finite-layer samples then occur in
     * pairs: [left boundary, right boundary].
     */
    for (index, layer) in stack.layers_left_to_right().iter().enumerate() {
        let admittance =
            IsotropicLayerQuantities::real_axis(layer.material(), complex_input.planar())
                .into_admittance()
                .into_inner();

        let left_waves = generic_bidirectional_values(waves.layers()[index].left());

        let right_waves = generic_bidirectional_values(waves.layers()[index].right());

        let expected_left = IsotropicFieldState::from_waves::<C, Ix0>(&left_waves, &admittance);

        let expected_right = IsotropicFieldState::from_waves::<C, Ix0>(&right_waves, &admittance);

        let left_sample = 1 + 2 * index;
        let right_sample = left_sample + 1;

        assert_canonical_state_close(
            fields.sample(left_sample).unwrap().canonical_state(),
            &expected_left,
        );

        assert_canonical_state_close(
            fields.sample(right_sample).unwrap().canonical_state(),
            &expected_right,
        );
    }
}

#[test]
fn scatter2_tangential_fields_are_continuous_at_internal_interfaces() {
    let stack = lossless_stack();
    let input = scalar_input(Polarisation::TransverseElectric, IncidentSide::Left);

    let response = solve_scatter2_internal_fields(&stack, &input);

    let fields = response
        .sample_fields(&stack, &input, &boundary_samples())
        .unwrap();

    let layer_count = stack.layers_left_to_right().len();

    for interface in 0..layer_count.saturating_sub(1) {
        let left_layer_right_sample = 1 + 2 * interface + 1;

        let right_layer_left_sample = 1 + 2 * (interface + 1);

        let left = fields
            .sample(left_layer_right_sample)
            .unwrap()
            .canonical_state();

        let right = fields
            .sample(right_layer_left_sample)
            .unwrap()
            .canonical_state();

        assert_canonical_state_close(left, right);
    }
}

#[test]
fn scatter2_tangential_fields_are_continuous_at_exterior_interfaces() {
    let stack = lossless_stack();
    let input = scalar_input(Polarisation::TransverseMagnetic, IncidentSide::Right);

    let response = solve_scatter2_internal_fields(&stack, &input);

    let fields = response
        .sample_fields(&stack, &input, &boundary_samples())
        .unwrap();

    let final_index = fields.len() - 1;

    assert_canonical_state_close(
        fields.sample(0).unwrap().canonical_state(),
        fields.sample(1).unwrap().canonical_state(),
    );

    assert_canonical_state_close(
        fields.sample(final_index - 1).unwrap().canonical_state(),
        fields.sample(final_index).unwrap().canonical_state(),
    );
}

#[test]
fn scatter2_normal_flux_is_constant_through_lossless_stack() {
    let stack = lossless_stack();
    let input = scalar_input(Polarisation::TransverseElectric, IncidentSide::Left);

    let response = solve_scatter2_internal_fields(&stack, &input);

    let fields = response
        .sample_fields(&stack, &input, &field_samples(&stack))
        .unwrap();

    let expected = fields.sample(0).unwrap().normal_flux()[()];

    for sample in fields.samples() {
        assert_real_close(sample.normal_flux()[()], expected);
    }
}

#[test]
fn scatter2_right_incident_lossless_flux_is_negative_and_constant() {
    let stack = lossless_stack();
    let input = scalar_input(Polarisation::TransverseMagnetic, IncidentSide::Right);

    let response = solve_scatter2_internal_fields(&stack, &input);

    let fields = response
        .sample_fields(&stack, &input, &field_samples(&stack))
        .unwrap();

    let expected = fields.sample(0).unwrap().normal_flux()[()];

    assert!(expected < 0.0);

    for sample in fields.samples() {
        assert_real_close(sample.normal_flux()[()], expected);
    }
}

#[test]
fn scatter2_lossless_stack_has_zero_layer_absorption() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for incident_side in [IncidentSide::Left, IncidentSide::Right] {
            let stack = lossless_stack();

            let input = scalar_input(polarisation, incident_side);

            let response = solve_scatter2_internal_fields(&stack, &input);

            let balance = response.power_balance(&stack, &input).unwrap();

            for layer in balance.layer_absorptance() {
                assert_real_close(layer[()], 0.0);
            }

            assert_real_close(balance.total_layer_absorptance()[()], 0.0);

            assert_real_close(balance.balance_residual()[()], 0.0);

            let r = response.response().reflectance()[()];

            let t = response.response().transmittance()[()];

            assert_real_close(r + t, 1.0);
        }
    }
}

#[test]
fn scatter2_absorbing_stack_has_positive_absorption_and_closes_balance() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for incident_side in [IncidentSide::Left, IncidentSide::Right] {
            let stack = absorbing_stack();

            let input = scalar_input(polarisation, incident_side);

            let response = solve_scatter2_internal_fields(&stack, &input);

            let balance = response.power_balance(&stack, &input).unwrap();

            for layer in balance.layer_absorptance() {
                assert!(
                    layer[()] >= -ABS_TOLERANCE,
                    "passive layer had negative absorption: {}",
                    layer[()],
                );
            }

            assert!(balance.total_layer_absorptance()[()] > ABS_TOLERANCE);

            assert_real_close(balance.balance_residual()[()], 0.0);

            let total = response.response().reflectance()[()]
                + response.response().transmittance()[()]
                + balance.total_layer_absorptance()[()];

            assert_real_close(total, 1.0);
        }
    }
}

#[test]
fn scatter2_layer_absorptance_matches_boundary_flux_drop() {
    let stack = absorbing_stack();

    let input = scalar_input(Polarisation::TransverseElectric, IncidentSide::Left);

    let response = solve_scatter2_internal_fields(&stack, &input);

    let fields = response
        .sample_fields(&stack, &input, &boundary_samples())
        .unwrap();

    let balance = response.power_balance(&stack, &input).unwrap();

    let incident_flux = balance.incident_flux()[()];

    for index in 0..stack.layers_left_to_right().len() {
        let left_sample = 1 + 2 * index;
        let right_sample = left_sample + 1;

        let left_flux = fields.sample(left_sample).unwrap().normal_flux()[()];

        let right_flux = fields.sample(right_sample).unwrap().normal_flux()[()];

        let expected = (left_flux - right_flux) / incident_flux;

        assert_real_close(balance.layer_absorptance()[index][()], expected);
    }
}

#[test]
fn scatter2_first_thickness_field_derivative_matches_finite_difference() {
    let backend = Scatter2::default();
    let stack = lossless_stack();
    let input = scalar_input(Polarisation::TransverseElectric, IncidentSide::Left);

    let layer_index = 1;
    let thickness = stack.layers_left_to_right()[layer_index]
        .thickness()
        .as_cm();

    let offset = 0.37 * thickness;
    let step = thickness_first_step(thickness);

    assert!(offset < thickness - step);

    let position = FieldPosition::Layer {
        index: layer_index,
        offset,
    };

    let differentiated = backend
        .solve_plane_wave_internal_fields_structural_first_derivative(
            &stack,
            &input,
            StructuralDerivativeVariable::Thickness(layer_index),
        )
        .unwrap();

    let analytic_fields = differentiated
        .sample_field_positions_structural_first_derivative(&stack, &input, [position])
        .unwrap();

    let analytic = analytic_fields
        .derivatives()
        .unwrap()
        .first_sample(0)
        .unwrap()
        .canonical_state()
        .primary()[()];

    let numerical = central_first_complex(
        |perturbed_thickness| {
            let perturbed_stack = with_layer_thickness(&stack, layer_index, perturbed_thickness);

            let response = backend
                .solve_plane_wave_internal_fields(&perturbed_stack, &input)
                .unwrap();

            response
                .sample_field_positions(&perturbed_stack, &input, [position])
                .unwrap()
                .sample(0)
                .unwrap()
                .canonical_state()
                .primary()[()]
        },
        thickness,
        step,
    );

    assert_complex_derivative_close(analytic, numerical);
}

#[test]
fn scatter2_second_thickness_field_derivative_matches_finite_difference() {
    let backend = Scatter2::default();
    let stack = lossless_stack();
    let input = scalar_input(Polarisation::TransverseElectric, IncidentSide::Left);

    let layer_index = 1;
    let thickness = stack.layers_left_to_right()[layer_index]
        .thickness()
        .as_cm();

    let step = thickness_second_step(thickness);

    let position = FieldPosition::Layer {
        index: layer_index,
        offset: 0.37 * thickness,
    };

    let differentiated = backend
        .solve_plane_wave_internal_fields_structural_second_derivative(
            &stack,
            &input,
            StructuralDerivativeVariable::Thickness(layer_index),
        )
        .unwrap();

    let sample = differentiated
        .sample_field_positions_structural_second_derivative(&stack, &input, [position])
        .unwrap();

    let analytic_second = sample
        .derivatives()
        .unwrap()
        .second_sample(0)
        .unwrap()
        .canonical_state()
        .primary()[()];

    let numerical_from_values = central_second_complex(
        |perturbed_thickness| {
            let perturbed_stack = with_layer_thickness(&stack, layer_index, perturbed_thickness);

            backend
                .solve_plane_wave_internal_fields(&perturbed_stack, &input)
                .unwrap()
                .sample_field_positions(&perturbed_stack, &input, [position])
                .unwrap()
                .sample(0)
                .unwrap()
                .canonical_state()
                .primary()[()]
        },
        thickness,
        step,
    );

    let numerical_from_first = central_first_complex(
        |perturbed_thickness| {
            let perturbed_stack = with_layer_thickness(&stack, layer_index, perturbed_thickness);

            backend
                .solve_plane_wave_internal_fields_structural_first_derivative(
                    &perturbed_stack,
                    &input,
                    StructuralDerivativeVariable::Thickness(layer_index),
                )
                .unwrap()
                .sample_field_positions_structural_first_derivative(
                    &perturbed_stack,
                    &input,
                    [position],
                )
                .unwrap()
                .derivatives()
                .unwrap()
                .first_sample(0)
                .unwrap()
                .canonical_state()
                .primary()[()]
        },
        thickness,
        step,
    );

    assert_complex_second_derivative_close(analytic_second, numerical_from_values);

    assert_complex_second_derivative_close(analytic_second, numerical_from_first);
}

#[test]
fn scatter2_first_vacuum_wavenumber_field_derivative_matches_finite_difference() {
    let backend = Scatter2::default();
    let stack = absorbing_stack();

    let input = scalar_input(Polarisation::TransverseMagnetic, IncidentSide::Left);

    let wavenumber = input.planar().vacuum_wavenumber()[()];
    let step = first_derivative_step(wavenumber);

    let position = FieldPosition::Layer {
        index: 1,
        offset: 0.04,
    };

    let differentiated = backend
        .solve_plane_wave_internal_fields_spectral_first_derivative(
            &stack,
            &input,
            SpectralDerivativeVariable::VacuumWavenumber,
        )
        .unwrap();

    let analytic = differentiated
        .sample_field_positions_spectral_first_derivative(&stack, &input, [position])
        .unwrap()
        .derivatives()
        .unwrap()
        .first_sample(0)
        .unwrap()
        .canonical_state()
        .primary()[()];

    let numerical_from_values = central_first_complex(
        |perturbed_wavenumber| {
            let perturbed_input = with_vacuum_wavenumber(&input, perturbed_wavenumber);

            backend
                .solve_plane_wave_internal_fields(&stack, &perturbed_input)
                .unwrap()
                .sample_field_positions(&stack, &perturbed_input, [position])
                .unwrap()
                .sample(0)
                .unwrap()
                .canonical_state()
                .primary()[()]
        },
        wavenumber,
        step,
    );

    assert_complex_derivative_close(analytic, numerical_from_values);
}

#[test]
fn scatter2_second_vacuum_wavenumber_field_derivative_matches_finite_difference() {
    let backend = Scatter2::default();
    let stack = absorbing_stack();

    let input = scalar_input(Polarisation::TransverseMagnetic, IncidentSide::Left);

    let wavenumber = input.planar().vacuum_wavenumber()[()];
    let step = first_derivative_step(wavenumber);

    let position = FieldPosition::Layer {
        index: 1,
        offset: 0.04,
    };

    let differentiated = backend
        .solve_plane_wave_internal_fields_spectral_second_derivative(
            &stack,
            &input,
            SpectralDerivativeVariable::VacuumWavenumber,
        )
        .unwrap();

    let analytic = differentiated
        .sample_field_positions_spectral_second_derivative(&stack, &input, [position])
        .unwrap()
        .derivatives()
        .unwrap()
        .second_sample(0)
        .unwrap()
        .canonical_state()
        .primary()[()];

    let numerical_from_values = central_second_complex(
        |perturbed_wavenumber| {
            let perturbed_input = with_vacuum_wavenumber(&input, perturbed_wavenumber);

            backend
                .solve_plane_wave_internal_fields(&stack, &perturbed_input)
                .unwrap()
                .sample_field_positions(&stack, &perturbed_input, [position])
                .unwrap()
                .sample(0)
                .unwrap()
                .canonical_state()
                .primary()[()]
        },
        wavenumber,
        step,
    );

    assert_complex_derivative_close(analytic, numerical_from_values);
}

#[test]
fn scatter2_first_thickness_power_derivative_matches_finite_difference() {
    let backend = Scatter2::default();
    let stack = absorbing_stack();
    let input = scalar_input(Polarisation::TransverseElectric, IncidentSide::Left);

    let layer_index = 1;
    let thickness = stack.layers_left_to_right()[layer_index]
        .thickness()
        .as_cm();

    let step = thickness_first_step(thickness);

    let differentiated = backend
        .solve_plane_wave_internal_fields_structural_first_derivative(
            &stack,
            &input,
            StructuralDerivativeVariable::Thickness(layer_index),
        )
        .unwrap();

    let analytic = differentiated
        .response()
        .derivatives()
        .expect("response derivative should be present")
        .first()
        .power()
        .reflectance()[()];

    let numerical = central_first_real(
        |perturbed_thickness| {
            let perturbed_stack = with_layer_thickness(&stack, layer_index, perturbed_thickness);

            backend
                .solve_plane_wave_internal_fields(&perturbed_stack, &input)
                .unwrap()
                .power()
                .reflectance()[()]
        },
        thickness,
        step,
    );

    assert_real_derivative_close(analytic, numerical);

    let analytic = differentiated
        .response()
        .derivatives()
        .expect("response derivative should be present")
        .first()
        .power()
        .transmittance()[()];

    let numerical = central_first_real(
        |perturbed_thickness| {
            let perturbed_stack = with_layer_thickness(&stack, layer_index, perturbed_thickness);

            backend
                .solve_plane_wave_internal_fields(&perturbed_stack, &input)
                .unwrap()
                .power()
                .transmittance()[()]
        },
        thickness,
        step,
    );

    assert_real_derivative_close(analytic, numerical);
}

#[test]
fn scatter2_first_thickness_power_derivatives_obey_balance() {
    let backend = Scatter2::default();
    let stack = absorbing_stack();
    let input = scalar_input(Polarisation::TransverseElectric, IncidentSide::Left);

    let layer_index = 1;

    let differentiated = backend
        .solve_plane_wave_internal_fields_structural_first_derivative(
            &stack,
            &input,
            StructuralDerivativeVariable::Thickness(layer_index),
        )
        .unwrap();

    let balance = differentiated
        .power_balance_structural_first_derivative(&stack, &input)
        .unwrap();

    let derivative = balance
        .derivatives()
        .expect("first power-balance derivative should be present");

    assert_real_close(derivative.first().balance_residual()[()], 0.0);
}

#[derive(Clone, Copy)]
pub(crate) struct ScalarFieldValues {
    pub primary: C,
    pub dual: C,

    pub electric_x: C,
    pub electric_y: C,
    pub electric_z: C,

    pub magnetic_x: C,
    pub magnetic_y: C,
    pub magnetic_z: C,

    pub normal_flux: f64,
}

pub(crate) fn scalar_field_values(sample: &PlaneWaveFieldSample<C, Ix0>) -> ScalarFieldValues {
    ScalarFieldValues {
        primary: sample.canonical_state().primary()[()],
        dual: sample.canonical_state().dual()[()],

        electric_x: sample.electric().x()[()],
        electric_y: sample.electric().y()[()],
        electric_z: sample.electric().z()[()],

        magnetic_x: sample.magnetic().x()[()],
        magnetic_y: sample.magnetic().y()[()],
        magnetic_z: sample.magnetic().z()[()],

        normal_flux: sample.normal_flux()[()],
    }
}
