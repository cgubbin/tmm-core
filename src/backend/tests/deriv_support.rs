use super::support::C;
use crate::{
    DifferentiablePlaneWaveFieldBackend, IncidentSide, PlaneWaveFieldBackend, Polarisation,
    Scatter2, SpectralDerivativeVariable, StructuralDerivativeVariable,
    backend::{
        FieldPosition, PlaneWaveFieldSampleOwned,
        tests::support::{
            assert_complex_derivative_close, assert_complex_second_derivative_close,
            assert_real_derivative_close, assert_real_second_derivative_close,
            central_first_complex, central_first_real, central_second_complex, central_second_real,
            first_derivative_step, lossless_stack, scalar_input, second_derivative_step,
            thickness_first_step, thickness_second_step, with_layer_thickness,
            with_parallel_wavenumber, with_vacuum_wavenumber,
        },
    },
};

use ndarray::Ix0;

#[derive(Clone, Copy, Debug)]
pub(crate) enum TestDerivativeOrder {
    First,
    Second,
}

pub(crate) fn check_complex_field_first_derivative_against_finite_difference<
    ValueSample,
    FirstSample,
    SolveValue,
    SolveFirst,
    ExtractValue,
    ExtractFirst,
>(
    parameter: f64,
    step: f64,
    mut solve_value: SolveValue,
    mut solve_first: SolveFirst,
    mut extract_value: ExtractValue,
    mut extract_first: ExtractFirst,
) where
    SolveValue: FnMut(f64) -> ValueSample,
    SolveFirst: FnMut(f64) -> FirstSample,
    ExtractValue: FnMut(&ValueSample) -> C,
    ExtractFirst: FnMut(&FirstSample) -> C,
{
    let analytic_sample = solve_first(parameter);
    let analytic = extract_first(&analytic_sample);

    let numerical = central_first_complex(
        |value| {
            let sample = solve_value(value);
            extract_value(&sample)
        },
        parameter,
        step,
    );

    assert_complex_derivative_close(analytic, numerical);
}

pub(crate) fn check_complex_field_second_derivative_against_finite_difference<
    ValueSample,
    FirstSample,
    SecondSample,
    SolveValue,
    SolveFirst,
    SolveSecond,
    ExtractValue,
    ExtractFirst,
    ExtractSecond,
>(
    parameter: f64,
    step: f64,
    mut solve_value: SolveValue,
    mut solve_first: SolveFirst,
    mut solve_second: SolveSecond,
    mut extract_value: ExtractValue,
    mut extract_first: ExtractFirst,
    mut extract_second: ExtractSecond,
) where
    SolveValue: FnMut(f64) -> ValueSample,
    SolveFirst: FnMut(f64) -> FirstSample,
    SolveSecond: FnMut(f64) -> SecondSample,
    ExtractValue: FnMut(&ValueSample) -> C,
    ExtractFirst: FnMut(&FirstSample) -> C,
    ExtractSecond: FnMut(&SecondSample) -> C,
{
    let analytic_sample = solve_second(parameter);
    let analytic = extract_second(&analytic_sample);

    let numerical_from_values = central_second_complex(
        |value| {
            let sample = solve_value(value);
            extract_value(&sample)
        },
        parameter,
        step,
    );

    let numerical_from_first = central_first_complex(
        |value| {
            let sample = solve_first(value);
            extract_first(&sample)
        },
        parameter,
        step,
    );

    assert_complex_second_derivative_close(analytic, numerical_from_values);

    assert_complex_second_derivative_close(analytic, numerical_from_first);
}

pub(crate) fn check_real_field_first_derivative_against_finite_difference<
    ValueSample,
    FirstSample,
    SolveValue,
    SolveFirst,
    ExtractValue,
    ExtractFirst,
>(
    parameter: f64,
    step: f64,
    mut solve_value: SolveValue,
    mut solve_first: SolveFirst,
    mut extract_value: ExtractValue,
    mut extract_first: ExtractFirst,
) where
    SolveValue: FnMut(f64) -> ValueSample,
    SolveFirst: FnMut(f64) -> FirstSample,
    ExtractValue: FnMut(&ValueSample) -> f64,
    ExtractFirst: FnMut(&FirstSample) -> f64,
{
    let analytic_sample = solve_first(parameter);
    let analytic = extract_first(&analytic_sample);

    let numerical = central_first_real(
        |value| {
            let sample = solve_value(value);
            extract_value(&sample)
        },
        parameter,
        step,
    );

    assert_real_derivative_close(analytic, numerical);
}

pub(crate) fn check_real_field_second_derivative_against_finite_difference<
    ValueSample,
    FirstSample,
    SecondSample,
    SolveValue,
    SolveFirst,
    SolveSecond,
    ExtractValue,
    ExtractFirst,
    ExtractSecond,
>(
    parameter: f64,
    step: f64,
    mut solve_value: SolveValue,
    mut solve_first: SolveFirst,
    mut solve_second: SolveSecond,
    mut extract_value: ExtractValue,
    mut extract_first: ExtractFirst,
    mut extract_second: ExtractSecond,
) where
    SolveValue: FnMut(f64) -> ValueSample,
    SolveFirst: FnMut(f64) -> FirstSample,
    SolveSecond: FnMut(f64) -> SecondSample,
    ExtractValue: FnMut(&ValueSample) -> f64,
    ExtractFirst: FnMut(&FirstSample) -> f64,
    ExtractSecond: FnMut(&SecondSample) -> f64,
{
    let analytic_sample = solve_second(parameter);
    let analytic = extract_second(&analytic_sample);

    let numerical_from_values = central_second_real(
        |value| {
            let sample = solve_value(value);
            extract_value(&sample)
        },
        parameter,
        step,
    );

    let numerical_from_first = central_first_real(
        |value| {
            let sample = solve_first(value);
            extract_first(&sample)
        },
        parameter,
        step,
    );

    assert_real_second_derivative_close(analytic, numerical_from_values);

    assert_real_second_derivative_close(analytic, numerical_from_first);
}

fn check_scatter2_thickness_field_first_derivative<ExtractValue, ExtractFirst>(
    polarisation: Polarisation,
    extract_value: ExtractValue,
    extract_first: ExtractFirst,
) where
    ExtractValue: FnMut(&PlaneWaveFieldSampleOwned<C, Ix0>) -> C,
    ExtractFirst: FnMut(&PlaneWaveFieldSampleOwned<C, Ix0>) -> C,
{
    let backend = Scatter2::default();
    let stack = lossless_stack();
    let input = scalar_input(polarisation, IncidentSide::Left);

    let layer_index = 1;

    let thickness = stack.layers_left_to_right()[layer_index]
        .thickness()
        .as_cm();

    let offset = 0.37 * thickness;

    let position = FieldPosition::Layer {
        index: layer_index,
        offset,
    };

    let step = thickness_first_step(thickness);

    check_complex_field_first_derivative_against_finite_difference(
        thickness,
        step,
        |perturbed_thickness| {
            let perturbed_stack = with_layer_thickness(&stack, layer_index, perturbed_thickness);

            backend
                .solve_plane_wave_internal_fields(&perturbed_stack, &input)
                .unwrap()
                .sample_field_positions(&perturbed_stack, &input, [position])
                .unwrap()
                .sample_view(0)
                .unwrap()
                .to_owned()
        },
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
                .sample_view(0)
                .unwrap()
                .to_owned()
        },
        extract_value,
        extract_first,
    );
}

fn check_scatter2_thickness_field_second_derivative<ExtractValue, ExtractFirst, ExtractSecond>(
    polarisation: Polarisation,
    extract_value: ExtractValue,
    extract_first: ExtractFirst,
    extract_second: ExtractSecond,
) where
    ExtractValue: FnMut(&PlaneWaveFieldSampleOwned<C, Ix0>) -> C,
    ExtractFirst: FnMut(&PlaneWaveFieldSampleOwned<C, Ix0>) -> C,
    ExtractSecond: FnMut(&PlaneWaveFieldSampleOwned<C, Ix0>) -> C,
{
    let backend = Scatter2::default();
    let stack = lossless_stack();
    let input = scalar_input(polarisation, IncidentSide::Left);

    let layer_index = 1;

    let thickness = stack.layers_left_to_right()[layer_index]
        .thickness()
        .as_cm();

    let offset = 0.37 * thickness;

    let position = FieldPosition::Layer {
        index: layer_index,
        offset,
    };

    let step = thickness_second_step(thickness);

    check_complex_field_second_derivative_against_finite_difference(
        thickness,
        step,
        |perturbed_thickness| {
            let perturbed_stack = with_layer_thickness(&stack, layer_index, perturbed_thickness);

            backend
                .solve_plane_wave_internal_fields(&perturbed_stack, &input)
                .unwrap()
                .sample_field_positions(&perturbed_stack, &input, [position])
                .unwrap()
                .sample_view(0)
                .unwrap()
                .to_owned()
        },
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
                .sample_view(0)
                .unwrap()
                .to_owned()
        },
        |perturbed_thickness| {
            let perturbed_stack = with_layer_thickness(&stack, layer_index, perturbed_thickness);

            backend
                .solve_plane_wave_internal_fields_structural_second_derivative(
                    &perturbed_stack,
                    &input,
                    StructuralDerivativeVariable::Thickness(layer_index),
                )
                .unwrap()
                .sample_field_positions_structural_second_derivative(
                    &perturbed_stack,
                    &input,
                    [position],
                )
                .unwrap()
                .sample_view(0)
                .unwrap()
                .to_owned()
        },
        extract_value,
        extract_first,
        extract_second,
    );
}

fn check_scatter2_parallel_wavenumber_field_first_derivative<ExtractValue, ExtractFirst>(
    polarisation: Polarisation,
    extract_value: ExtractValue,
    extract_first: ExtractFirst,
    square: bool,
) where
    ExtractValue: FnMut(&PlaneWaveFieldSampleOwned<C, Ix0>) -> C,
    ExtractFirst: FnMut(&PlaneWaveFieldSampleOwned<C, Ix0>) -> C,
{
    let backend = Scatter2::default();
    let stack = lossless_stack();
    let input = scalar_input(polarisation, IncidentSide::Left);

    let layer_index = 1;

    let thickness = stack.layers_left_to_right()[layer_index]
        .thickness()
        .as_cm();

    let offset = 0.37 * thickness;

    let position = FieldPosition::Layer {
        index: layer_index,
        offset,
    };

    let parameter = if square {
        input.planar().parallel_wavenumber()[()].powi(2)
    } else {
        input.planar().parallel_wavenumber()[()]
    };

    let step = first_derivative_step(parameter);

    check_complex_field_first_derivative_against_finite_difference(
        parameter,
        step,
        |value| {
            let perturbed_input =
                with_parallel_wavenumber(&input, if square { value.sqrt() } else { value });

            backend
                .solve_plane_wave_internal_fields(&stack, &perturbed_input)
                .unwrap()
                .sample_field_positions(&stack, &perturbed_input, [position])
                .unwrap()
                .sample_view(0)
                .unwrap()
                .to_owned()
        },
        |value| {
            let perturbed_input =
                with_parallel_wavenumber(&input, if square { value.sqrt() } else { value });

            backend
                .solve_plane_wave_internal_fields_structural_first_derivative(
                    &stack,
                    &perturbed_input,
                    if square {
                        StructuralDerivativeVariable::ParallelWavenumberSquared
                    } else {
                        StructuralDerivativeVariable::ParallelWavenumber
                    },
                )
                .unwrap()
                .sample_field_positions_structural_first_derivative(
                    &stack,
                    &perturbed_input,
                    [position],
                )
                .unwrap()
                .sample_view(0)
                .unwrap()
                .to_owned()
        },
        extract_value,
        extract_first,
    );
}

fn check_scatter2_parallel_wavenumber_field_second_derivative<
    ExtractValue,
    ExtractFirst,
    ExtractSecond,
>(
    polarisation: Polarisation,
    extract_value: ExtractValue,
    extract_first: ExtractFirst,
    extract_second: ExtractSecond,
    square: bool,
) where
    ExtractValue: FnMut(&PlaneWaveFieldSampleOwned<C, Ix0>) -> C,
    ExtractFirst: FnMut(&PlaneWaveFieldSampleOwned<C, Ix0>) -> C,
    ExtractSecond: FnMut(&PlaneWaveFieldSampleOwned<C, Ix0>) -> C,
{
    let backend = Scatter2::default();
    let stack = lossless_stack();
    let input = scalar_input(polarisation, IncidentSide::Left);

    let layer_index = 1;

    let thickness = stack.layers_left_to_right()[layer_index]
        .thickness()
        .as_cm();

    let offset = 0.37 * thickness;

    let position = FieldPosition::Layer {
        index: layer_index,
        offset,
    };

    let parameter = if square {
        input.planar().parallel_wavenumber()[()].powi(2)
    } else {
        input.planar().parallel_wavenumber()[()]
    };

    let step = second_derivative_step(parameter);

    check_complex_field_second_derivative_against_finite_difference(
        parameter,
        step,
        |value| {
            let perturbed_input =
                with_parallel_wavenumber(&input, if square { value.sqrt() } else { value });

            backend
                .solve_plane_wave_internal_fields(&stack, &perturbed_input)
                .unwrap()
                .sample_field_positions(&stack, &perturbed_input, [position])
                .unwrap()
                .sample_view(0)
                .unwrap()
                .to_owned()
        },
        |value| {
            let perturbed_input =
                with_parallel_wavenumber(&input, if square { value.sqrt() } else { value });

            backend
                .solve_plane_wave_internal_fields_structural_first_derivative(
                    &stack,
                    &perturbed_input,
                    if square {
                        StructuralDerivativeVariable::ParallelWavenumberSquared
                    } else {
                        StructuralDerivativeVariable::ParallelWavenumber
                    },
                )
                .unwrap()
                .sample_field_positions_structural_first_derivative(
                    &stack,
                    &perturbed_input,
                    [position],
                )
                .unwrap()
                .sample_view(0)
                .unwrap()
                .to_owned()
        },
        |value| {
            let perturbed_input =
                with_parallel_wavenumber(&input, if square { value.sqrt() } else { value });

            backend
                .solve_plane_wave_internal_fields_structural_second_derivative(
                    &stack,
                    &perturbed_input,
                    if square {
                        StructuralDerivativeVariable::ParallelWavenumberSquared
                    } else {
                        StructuralDerivativeVariable::ParallelWavenumber
                    },
                )
                .unwrap()
                .sample_field_positions_structural_second_derivative(
                    &stack,
                    &perturbed_input,
                    [position],
                )
                .unwrap()
                .sample_view(0)
                .unwrap()
                .to_owned()
        },
        extract_value,
        extract_first,
        extract_second,
    );
}

fn check_scatter2_vacuum_wavenumber_field_first_derivative<ExtractValue, ExtractFirst>(
    polarisation: Polarisation,
    extract_value: ExtractValue,
    extract_first: ExtractFirst,
    square: bool,
) where
    ExtractValue: FnMut(&PlaneWaveFieldSampleOwned<C, Ix0>) -> C,
    ExtractFirst: FnMut(&PlaneWaveFieldSampleOwned<C, Ix0>) -> C,
{
    let backend = Scatter2::default();
    let stack = lossless_stack();
    let input = scalar_input(polarisation, IncidentSide::Left);

    let layer_index = 1;

    let thickness = stack.layers_left_to_right()[layer_index]
        .thickness()
        .as_cm();

    let offset = 0.37 * thickness;

    let position = FieldPosition::Layer {
        index: layer_index,
        offset,
    };

    let parameter = if square {
        input.planar().vacuum_wavenumber()[()].powi(2)
    } else {
        input.planar().vacuum_wavenumber()[()]
    };

    let step = first_derivative_step(parameter);

    check_complex_field_first_derivative_against_finite_difference(
        parameter,
        step,
        |value| {
            let perturbed_input =
                with_vacuum_wavenumber(&input, if square { value.sqrt() } else { value });

            backend
                .solve_plane_wave_internal_fields(&stack, &perturbed_input)
                .unwrap()
                .sample_field_positions(&stack, &perturbed_input, [position])
                .unwrap()
                .sample_view(0)
                .unwrap()
                .to_owned()
        },
        |value| {
            let perturbed_input =
                with_vacuum_wavenumber(&input, if square { value.sqrt() } else { value });

            backend
                .solve_plane_wave_internal_fields_spectral_first_derivative(
                    &stack,
                    &perturbed_input,
                    if square {
                        SpectralDerivativeVariable::VacuumWavenumberSquared
                    } else {
                        SpectralDerivativeVariable::VacuumWavenumber
                    },
                )
                .unwrap()
                .sample_field_positions_spectral_first_derivative(
                    &stack,
                    &perturbed_input,
                    [position],
                )
                .unwrap()
                .sample_view(0)
                .unwrap()
                .to_owned()
        },
        extract_value,
        extract_first,
    );
}

fn check_scatter2_vacuum_wavenumber_field_second_derivative<
    ExtractValue,
    ExtractFirst,
    ExtractSecond,
>(
    polarisation: Polarisation,
    extract_value: ExtractValue,
    extract_first: ExtractFirst,
    extract_second: ExtractSecond,
    square: bool,
) where
    ExtractValue: FnMut(&PlaneWaveFieldSampleOwned<C, Ix0>) -> C,
    ExtractFirst: FnMut(&PlaneWaveFieldSampleOwned<C, Ix0>) -> C,
    ExtractSecond: FnMut(&PlaneWaveFieldSampleOwned<C, Ix0>) -> C,
{
    let backend = Scatter2::default();
    let stack = lossless_stack();
    let input = scalar_input(polarisation, IncidentSide::Left);

    let layer_index = 1;

    let thickness = stack.layers_left_to_right()[layer_index]
        .thickness()
        .as_cm();

    let offset = 0.37 * thickness;

    let position = FieldPosition::Layer {
        index: layer_index,
        offset,
    };

    let parameter = if square {
        input.planar().vacuum_wavenumber()[()].powi(2)
    } else {
        input.planar().vacuum_wavenumber()[()]
    };

    let step = second_derivative_step(parameter);

    check_complex_field_second_derivative_against_finite_difference(
        parameter,
        step,
        |value| {
            let perturbed_input =
                with_vacuum_wavenumber(&input, if square { value.sqrt() } else { value });

            backend
                .solve_plane_wave_internal_fields(&stack, &perturbed_input)
                .unwrap()
                .sample_field_positions(&stack, &perturbed_input, [position])
                .unwrap()
                .sample_view(0)
                .unwrap()
                .to_owned()
        },
        |value| {
            let perturbed_input =
                with_vacuum_wavenumber(&input, if square { value.sqrt() } else { value });

            backend
                .solve_plane_wave_internal_fields_spectral_first_derivative(
                    &stack,
                    &perturbed_input,
                    if square {
                        SpectralDerivativeVariable::VacuumWavenumberSquared
                    } else {
                        SpectralDerivativeVariable::VacuumWavenumber
                    },
                )
                .unwrap()
                .sample_field_positions_spectral_first_derivative(
                    &stack,
                    &perturbed_input,
                    [position],
                )
                .unwrap()
                .sample_view(0)
                .unwrap()
                .to_owned()
        },
        |value| {
            let perturbed_input =
                with_vacuum_wavenumber(&input, if square { value.sqrt() } else { value });

            backend
                .solve_plane_wave_internal_fields_spectral_second_derivative(
                    &stack,
                    &perturbed_input,
                    if square {
                        SpectralDerivativeVariable::VacuumWavenumberSquared
                    } else {
                        SpectralDerivativeVariable::VacuumWavenumber
                    },
                )
                .unwrap()
                .sample_field_positions_spectral_second_derivative(
                    &stack,
                    &perturbed_input,
                    [position],
                )
                .unwrap()
                .sample_view(0)
                .unwrap()
                .to_owned()
        },
        extract_value,
        extract_first,
        extract_second,
    );
}

#[test]
fn scatter2_first_thickness_primary_field_derivative_matches_fd() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        check_scatter2_thickness_field_first_derivative(
            polarisation,
            |sample| sample.value().canonical_state().primary()[()],
            |sample| sample.first().unwrap().canonical_state().primary()[()],
        );
    }
}

#[test]
fn scatter2_first_thickness_dual_field_derivative_matches_fd() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        check_scatter2_thickness_field_first_derivative(
            polarisation,
            |sample| sample.value().canonical_state().dual()[()],
            |sample| sample.first().unwrap().canonical_state().dual()[()],
        );
    }
}

#[test]
fn scatter2_second_thickness_primary_field_derivative_matches_fd() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        check_scatter2_thickness_field_second_derivative(
            polarisation,
            |sample| sample.value().canonical_state().primary()[()],
            |sample| sample.first().unwrap().canonical_state().primary()[()],
            |sample| sample.second().unwrap().canonical_state().primary()[()],
        );
    }
}

#[test]
fn scatter2_second_thickness_dual_field_derivative_matches_fd() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        check_scatter2_thickness_field_second_derivative(
            polarisation,
            |sample| sample.value().canonical_state().dual()[()],
            |sample| sample.first().unwrap().canonical_state().dual()[()],
            |sample| sample.second().unwrap().canonical_state().dual()[()],
        );
    }
}

#[test]
fn scatter2_first_parallel_wavenumber_primary_field_derivative_matches_fd() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        check_scatter2_parallel_wavenumber_field_first_derivative(
            polarisation,
            |sample| sample.value().canonical_state().primary()[()],
            |sample| sample.first().unwrap().canonical_state().primary()[()],
            false,
        );
    }
}

#[test]
fn scatter2_first_parallel_wavenumber_squared_primary_field_derivative_matches_fd() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        check_scatter2_parallel_wavenumber_field_first_derivative(
            polarisation,
            |sample| sample.value().canonical_state().primary()[()],
            |sample| sample.first().unwrap().canonical_state().primary()[()],
            true,
        );
    }
}

#[test]
fn scatter2_second_parallel_wavenumber_primary_field_derivative_matches_fd() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        check_scatter2_parallel_wavenumber_field_second_derivative(
            polarisation,
            |sample| sample.value().canonical_state().primary()[()],
            |sample| sample.first().unwrap().canonical_state().primary()[()],
            |sample| sample.second().unwrap().canonical_state().primary()[()],
            false,
        );
    }
}

#[test]
fn scatter2_second_parallel_wavenumber_squared_primary_field_derivative_matches_fd() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        check_scatter2_parallel_wavenumber_field_second_derivative(
            polarisation,
            |sample| sample.value().canonical_state().primary()[()],
            |sample| sample.first().unwrap().canonical_state().primary()[()],
            |sample| sample.second().unwrap().canonical_state().primary()[()],
            true,
        );
    }
}

#[test]
fn scatter2_first_vacuum_wavenumber_primary_field_derivative_matches_fd() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        check_scatter2_vacuum_wavenumber_field_first_derivative(
            polarisation,
            |sample| sample.value().canonical_state().primary()[()],
            |sample| sample.first().unwrap().canonical_state().primary()[()],
            false,
        );
    }
}

#[test]
fn scatter2_first_vacuum_wavenumber_squared_primary_field_derivative_matches_fd() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        check_scatter2_vacuum_wavenumber_field_first_derivative(
            polarisation,
            |sample| sample.value().canonical_state().primary()[()],
            |sample| sample.first().unwrap().canonical_state().primary()[()],
            true,
        );
    }
}

#[test]
fn scatter2_second_vacuum_wavenumber_primary_field_derivative_matches_fd() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        check_scatter2_vacuum_wavenumber_field_second_derivative(
            polarisation,
            |sample| sample.value().canonical_state().primary()[()],
            |sample| sample.first().unwrap().canonical_state().primary()[()],
            |sample| sample.second().unwrap().canonical_state().primary()[()],
            false,
        );
    }
}

#[test]
fn scatter2_second_vacuum_wavenumber_squared_primary_field_derivative_matches_fd() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        check_scatter2_vacuum_wavenumber_field_second_derivative(
            polarisation,
            |sample| sample.value().canonical_state().primary()[()],
            |sample| sample.first().unwrap().canonical_state().primary()[()],
            |sample| sample.second().unwrap().canonical_state().primary()[()],
            true,
        );
    }
}

#[test]
fn scatter2_first_thickness_te_ey_matches_fd() {
    check_scatter2_thickness_field_first_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().electric().y()[()],
        |sample| sample.first().unwrap().electric().y()[()],
    );
}

#[test]
fn scatter2_first_thickness_te_hx_matches_fd() {
    check_scatter2_thickness_field_first_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().magnetic().x()[()],
        |sample| sample.first().unwrap().magnetic().x()[()],
    );
}

#[test]
fn scatter2_first_thickness_te_hz_matches_fd() {
    check_scatter2_thickness_field_first_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().magnetic().z()[()],
        |sample| sample.first().unwrap().magnetic().z()[()],
    );
}

#[test]
fn scatter2_first_thickness_tm_ex_matches_fd() {
    check_scatter2_thickness_field_first_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().electric().x()[()],
        |sample| sample.first().unwrap().electric().x()[()],
    );
}

#[test]
fn scatter2_first_thickness_tm_ez_matches_fd() {
    check_scatter2_thickness_field_first_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().electric().z()[()],
        |sample| sample.first().unwrap().electric().z()[()],
    );
}

#[test]
fn scatter2_first_thickness_tm_hy_matches_fd() {
    check_scatter2_thickness_field_first_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().magnetic().y()[()],
        |sample| sample.first().unwrap().magnetic().y()[()],
    );
}

#[test]
fn scatter2_second_thickness_te_ey_matches_fd() {
    check_scatter2_thickness_field_second_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().electric().y()[()],
        |sample| sample.first().unwrap().electric().y()[()],
        |sample| sample.second().unwrap().electric().y()[()],
    );
}

#[test]
fn scatter2_second_thickness_te_hx_matches_fd() {
    check_scatter2_thickness_field_second_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().magnetic().x()[()],
        |sample| sample.first().unwrap().magnetic().x()[()],
        |sample| sample.second().unwrap().magnetic().x()[()],
    );
}

#[test]
fn scatter2_second_thickness_te_hz_matches_fd() {
    check_scatter2_thickness_field_second_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().magnetic().z()[()],
        |sample| sample.first().unwrap().magnetic().z()[()],
        |sample| sample.second().unwrap().magnetic().z()[()],
    );
}

#[test]
fn scatter2_second_thickness_tm_ex_matches_fd() {
    check_scatter2_thickness_field_second_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().electric().x()[()],
        |sample| sample.first().unwrap().electric().x()[()],
        |sample| sample.second().unwrap().electric().x()[()],
    );
}

#[test]
fn scatter2_second_thickness_tm_ez_matches_fd() {
    check_scatter2_thickness_field_second_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().electric().z()[()],
        |sample| sample.first().unwrap().electric().z()[()],
        |sample| sample.second().unwrap().electric().z()[()],
    );
}

#[test]
fn scatter2_second_thickness_tm_hy_matches_fd() {
    check_scatter2_thickness_field_second_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().magnetic().y()[()],
        |sample| sample.first().unwrap().magnetic().y()[()],
        |sample| sample.second().unwrap().magnetic().y()[()],
    );
}

#[test]
fn scatter2_first_parallel_wavenumber_te_ey_matches_fd() {
    check_scatter2_parallel_wavenumber_field_first_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().electric().y()[()],
        |sample| sample.first().unwrap().electric().y()[()],
        false,
    );
}

#[test]
fn scatter2_first_parallel_wavenumber_te_hx_matches_fd() {
    check_scatter2_parallel_wavenumber_field_first_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().magnetic().x()[()],
        |sample| sample.first().unwrap().magnetic().x()[()],
        false,
    );
}

#[test]
fn scatter2_first_parallel_wavenumber_te_hz_matches_fd() {
    check_scatter2_parallel_wavenumber_field_first_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().magnetic().z()[()],
        |sample| sample.first().unwrap().magnetic().z()[()],
        false,
    );
}

#[test]
fn scatter2_first_parallel_wavenumber_tm_ex_matches_fd() {
    check_scatter2_parallel_wavenumber_field_first_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().electric().x()[()],
        |sample| sample.first().unwrap().electric().x()[()],
        false,
    );
}

#[test]
fn scatter2_first_parallel_wavenumber_tm_ez_matches_fd() {
    check_scatter2_parallel_wavenumber_field_first_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().electric().z()[()],
        |sample| sample.first().unwrap().electric().z()[()],
        false,
    );
}

#[test]
fn scatter2_first_parallel_wavenumber_tm_hy_matches_fd() {
    check_scatter2_parallel_wavenumber_field_first_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().magnetic().y()[()],
        |sample| sample.first().unwrap().magnetic().y()[()],
        false,
    );
}

#[test]
fn scatter2_second_parallel_wavenumber_te_ey_matches_fd() {
    check_scatter2_parallel_wavenumber_field_second_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().electric().y()[()],
        |sample| sample.first().unwrap().electric().y()[()],
        |sample| sample.second().unwrap().electric().y()[()],
        false,
    );
}

#[test]
fn scatter2_second_parallel_wavenumber_te_hx_matches_fd() {
    check_scatter2_parallel_wavenumber_field_second_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().magnetic().x()[()],
        |sample| sample.first().unwrap().magnetic().x()[()],
        |sample| sample.second().unwrap().magnetic().x()[()],
        false,
    );
}

#[test]
fn scatter2_second_parallel_wavenumber_te_hz_matches_fd() {
    check_scatter2_parallel_wavenumber_field_second_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().magnetic().z()[()],
        |sample| sample.first().unwrap().magnetic().z()[()],
        |sample| sample.second().unwrap().magnetic().z()[()],
        false,
    );
}

#[test]
fn scatter2_second_parallel_wavenumber_tm_ex_matches_fd() {
    check_scatter2_parallel_wavenumber_field_second_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().electric().x()[()],
        |sample| sample.first().unwrap().electric().x()[()],
        |sample| sample.second().unwrap().electric().x()[()],
        false,
    );
}

#[test]
fn scatter2_second_parallel_wavenumber_tm_ez_matches_fd() {
    check_scatter2_parallel_wavenumber_field_second_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().electric().z()[()],
        |sample| sample.first().unwrap().electric().z()[()],
        |sample| sample.second().unwrap().electric().z()[()],
        false,
    );
}

#[test]
fn scatter2_second_parallel_wavenumber_tm_hy_matches_fd() {
    check_scatter2_parallel_wavenumber_field_second_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().magnetic().y()[()],
        |sample| sample.first().unwrap().magnetic().y()[()],
        |sample| sample.second().unwrap().magnetic().y()[()],
        false,
    );
}

#[test]
fn scatter2_first_parallel_wavenumber_squared_te_ey_matches_fd() {
    check_scatter2_parallel_wavenumber_field_first_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().electric().y()[()],
        |sample| sample.first().unwrap().electric().y()[()],
        true,
    );
}

#[test]
fn scatter2_first_parallel_wavenumber_squared_te_hx_matches_fd() {
    check_scatter2_parallel_wavenumber_field_first_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().magnetic().x()[()],
        |sample| sample.first().unwrap().magnetic().x()[()],
        true,
    );
}

#[test]
fn scatter2_first_parallel_wavenumber_squared_te_hz_matches_fd() {
    check_scatter2_parallel_wavenumber_field_first_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().magnetic().z()[()],
        |sample| sample.first().unwrap().magnetic().z()[()],
        true,
    );
}

#[test]
fn scatter2_first_parallel_wavenumber_squared_tm_ex_matches_fd() {
    check_scatter2_parallel_wavenumber_field_first_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().electric().x()[()],
        |sample| sample.first().unwrap().electric().x()[()],
        true,
    );
}

#[test]
fn scatter2_first_parallel_wavenumber_squared_tm_ez_matches_fd() {
    check_scatter2_parallel_wavenumber_field_first_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().electric().z()[()],
        |sample| sample.first().unwrap().electric().z()[()],
        true,
    );
}

#[test]
fn scatter2_first_parallel_wavenumber_squared_tm_hy_matches_fd() {
    check_scatter2_parallel_wavenumber_field_first_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().magnetic().y()[()],
        |sample| sample.first().unwrap().magnetic().y()[()],
        true,
    );
}

#[test]
fn scatter2_second_parallel_wavenumber_squared_te_ey_matches_fd() {
    check_scatter2_parallel_wavenumber_field_second_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().electric().y()[()],
        |sample| sample.first().unwrap().electric().y()[()],
        |sample| sample.second().unwrap().electric().y()[()],
        true,
    );
}

#[test]
fn scatter2_second_parallel_wavenumber_squared_te_hx_matches_fd() {
    check_scatter2_parallel_wavenumber_field_second_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().magnetic().x()[()],
        |sample| sample.first().unwrap().magnetic().x()[()],
        |sample| sample.second().unwrap().magnetic().x()[()],
        true,
    );
}

#[test]
fn scatter2_second_parallel_wavenumber_squared_te_hz_matches_fd() {
    check_scatter2_parallel_wavenumber_field_second_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().magnetic().z()[()],
        |sample| sample.first().unwrap().magnetic().z()[()],
        |sample| sample.second().unwrap().magnetic().z()[()],
        true,
    );
}

#[test]
fn scatter2_second_parallel_wavenumber_squared_tm_ex_matches_fd() {
    check_scatter2_parallel_wavenumber_field_second_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().electric().x()[()],
        |sample| sample.first().unwrap().electric().x()[()],
        |sample| sample.second().unwrap().electric().x()[()],
        true,
    );
}

#[test]
fn scatter2_second_parallel_wavenumber_squared_tm_ez_matches_fd() {
    check_scatter2_parallel_wavenumber_field_second_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().electric().z()[()],
        |sample| sample.first().unwrap().electric().z()[()],
        |sample| sample.second().unwrap().electric().z()[()],
        true,
    );
}

#[test]
fn scatter2_second_parallel_wavenumber_squared_tm_hy_matches_fd() {
    check_scatter2_parallel_wavenumber_field_second_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().magnetic().y()[()],
        |sample| sample.first().unwrap().magnetic().y()[()],
        |sample| sample.second().unwrap().magnetic().y()[()],
        true,
    );
}

#[test]
fn scatter2_first_vacuum_wavenumber_te_ey_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_first_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().electric().y()[()],
        |sample| sample.first().unwrap().electric().y()[()],
        false,
    );
}

#[test]
fn scatter2_first_vacuum_wavenumber_te_hx_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_first_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().magnetic().x()[()],
        |sample| sample.first().unwrap().magnetic().x()[()],
        false,
    );
}

#[test]
fn scatter2_first_vacuum_wavenumber_te_hz_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_first_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().magnetic().z()[()],
        |sample| sample.first().unwrap().magnetic().z()[()],
        false,
    );
}

#[test]
fn scatter2_first_vacuum_wavenumber_tm_ex_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_first_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().electric().x()[()],
        |sample| sample.first().unwrap().electric().x()[()],
        false,
    );
}

#[test]
fn scatter2_first_vacuum_wavenumber_tm_ez_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_first_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().electric().z()[()],
        |sample| sample.first().unwrap().electric().z()[()],
        false,
    );
}

#[test]
fn scatter2_first_vacuum_wavenumber_tm_hy_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_first_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().magnetic().y()[()],
        |sample| sample.first().unwrap().magnetic().y()[()],
        false,
    );
}

#[test]
fn scatter2_second_vacuum_wavenumber_te_ey_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_second_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().electric().y()[()],
        |sample| sample.first().unwrap().electric().y()[()],
        |sample| sample.second().unwrap().electric().y()[()],
        false,
    );
}

#[test]
fn scatter2_second_vacuum_wavenumber_te_hx_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_second_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().magnetic().x()[()],
        |sample| sample.first().unwrap().magnetic().x()[()],
        |sample| sample.second().unwrap().magnetic().x()[()],
        false,
    );
}

#[test]
fn scatter2_second_vacuum_wavenumber_te_hz_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_second_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().magnetic().z()[()],
        |sample| sample.first().unwrap().magnetic().z()[()],
        |sample| sample.second().unwrap().magnetic().z()[()],
        false,
    );
}

#[test]
fn scatter2_second_vacuum_wavenumber_tm_ex_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_second_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().electric().x()[()],
        |sample| sample.first().unwrap().electric().x()[()],
        |sample| sample.second().unwrap().electric().x()[()],
        false,
    );
}

#[test]
fn scatter2_second_vacuum_wavenumber_tm_ez_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_second_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().electric().z()[()],
        |sample| sample.first().unwrap().electric().z()[()],
        |sample| sample.second().unwrap().electric().z()[()],
        false,
    );
}

#[test]
fn scatter2_second_vacuum_wavenumber_tm_hy_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_second_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().magnetic().y()[()],
        |sample| sample.first().unwrap().magnetic().y()[()],
        |sample| sample.second().unwrap().magnetic().y()[()],
        false,
    );
}

#[test]
fn scatter2_first_vacuum_wavenumber_squared_te_ey_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_first_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().electric().y()[()],
        |sample| sample.first().unwrap().electric().y()[()],
        true,
    );
}

#[test]
fn scatter2_first_vacuum_wavenumber_squared_te_hx_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_first_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().magnetic().x()[()],
        |sample| sample.first().unwrap().magnetic().x()[()],
        true,
    );
}

#[test]
fn scatter2_first_vacuum_wavenumber_squared_te_hz_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_first_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().magnetic().z()[()],
        |sample| sample.first().unwrap().magnetic().z()[()],
        true,
    );
}

#[test]
fn scatter2_first_vacuum_wavenumber_squared_tm_ex_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_first_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().electric().x()[()],
        |sample| sample.first().unwrap().electric().x()[()],
        true,
    );
}

#[test]
fn scatter2_first_vacuum_wavenumber_squared_tm_ez_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_first_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().electric().z()[()],
        |sample| sample.first().unwrap().electric().z()[()],
        true,
    );
}

#[test]
fn scatter2_first_vacuum_wavenumber_squared_tm_hy_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_first_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().magnetic().y()[()],
        |sample| sample.first().unwrap().magnetic().y()[()],
        true,
    );
}

#[test]
fn scatter2_second_vacuum_wavenumber_squared_te_ey_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_second_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().electric().y()[()],
        |sample| sample.first().unwrap().electric().y()[()],
        |sample| sample.second().unwrap().electric().y()[()],
        true,
    );
}

#[test]
fn scatter2_second_vacuum_wavenumber_squared_te_hx_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_second_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().magnetic().x()[()],
        |sample| sample.first().unwrap().magnetic().x()[()],
        |sample| sample.second().unwrap().magnetic().x()[()],
        true,
    );
}

#[test]
fn scatter2_second_vacuum_wavenumber_squared_te_hz_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_second_derivative(
        Polarisation::TransverseElectric,
        |sample| sample.value().magnetic().z()[()],
        |sample| sample.first().unwrap().magnetic().z()[()],
        |sample| sample.second().unwrap().magnetic().z()[()],
        true,
    );
}

#[test]
fn scatter2_second_vacuum_wavenumber_squared_tm_ex_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_second_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().electric().x()[()],
        |sample| sample.first().unwrap().electric().x()[()],
        |sample| sample.second().unwrap().electric().x()[()],
        true,
    );
}

#[test]
fn scatter2_second_vacuum_wavenumber_squared_tm_ez_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_second_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().electric().z()[()],
        |sample| sample.first().unwrap().electric().z()[()],
        |sample| sample.second().unwrap().electric().z()[()],
        true,
    );
}

#[test]
fn scatter2_second_vacuum_wavenumber_squared_tm_hy_matches_fd() {
    check_scatter2_vacuum_wavenumber_field_second_derivative(
        Polarisation::TransverseMagnetic,
        |sample| sample.value().magnetic().y()[()],
        |sample| sample.first().unwrap().magnetic().y()[()],
        |sample| sample.second().unwrap().magnetic().y()[()],
        true,
    );
}
