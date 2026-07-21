use ndarray::{Array0, Ix0, arr1};

use crate::{
    DifferentiablePlaneWaveFieldBackend, EvaluateMaterial, IncidentSide, PlanarInput,
    PlaneWaveBackend, PlaneWaveFieldBackend, PlaneWaveInput, PlaneWaveResponse, Polarisation,
    Scatter2, SpectralDerivativeVariable, Stack, StructuralDerivativeVariable, Transfer2,
    backend::{
        PlaneWaveFieldResponse,
        tests::support::{
            absorbing_stack, assert_boundary_waves_close, assert_complex_array_close,
            assert_differentiated_power_balance_close, assert_field_derivatives_close,
            assert_fields_close, assert_power_balance_close, assert_real_array_close,
            field_positions, field_samples, lossless_stack, scalar_input,
        },
    },
};

use super::support::C;

fn solve_both<M>(
    stack: &Stack<M, f64>,
    input: &PlaneWaveInput<Array0<f64>>,
) -> (PlaneWaveResponse<C, Ix0>, PlaneWaveResponse<C, Ix0>)
where
    M: EvaluateMaterial<C, Real = f64>,
{
    let scatter = Scatter2
        .solve_plane_wave(stack, input)
        .expect("Scatter2 solve should succeed");

    let transfer = Transfer2
        .solve_plane_wave(stack, input)
        .expect("Transfer2 solve should succeed");

    (scatter, transfer)
}

fn solve_both_fields<M>(
    stack: &Stack<M, f64>,
    input: &PlaneWaveInput<Array0<f64>>,
) -> (
    PlaneWaveFieldResponse<C, Ix0>,
    PlaneWaveFieldResponse<C, Ix0>,
)
where
    M: EvaluateMaterial<C, Real = f64>,
{
    let scatter = Scatter2
        .solve_plane_wave_internal_fields(stack, input)
        .expect("Scatter2 solve should succeed");

    let transfer = Transfer2
        .solve_plane_wave_internal_fields(stack, input)
        .expect("Transfer2 solve should succeed");

    (scatter, transfer)
}

#[test]
fn transfer2_and_scatter2_boundary_waves_are_equivalent() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for incident_side in [IncidentSide::Left, IncidentSide::Right] {
            for stack in [lossless_stack(), absorbing_stack()] {
                let input = scalar_input(polarisation, incident_side);

                let (scatter, transfer) = solve_both_fields(&stack, &input);

                assert_boundary_waves_close(
                    scatter.boundary_waves().values(),
                    transfer.boundary_waves().values(),
                );
            }
        }
    }
}

#[test]
fn transfer2_and_scatter2_plane_wave_responses_are_equivalent() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for incident_side in [IncidentSide::Left, IncidentSide::Right] {
            let stack = absorbing_stack();

            let input = scalar_input(polarisation, incident_side);

            let (scatter, transfer) = solve_both_fields(&stack, &input);

            assert_complex_array_close(
                scatter.response().reflection(),
                transfer.response().reflection(),
            );

            assert_complex_array_close(
                scatter.response().transmission(),
                transfer.response().transmission(),
            );

            assert_real_array_close(
                scatter.response().reflectance(),
                transfer.response().reflectance(),
            );

            assert_real_array_close(
                scatter.response().transmittance(),
                transfer.response().transmittance(),
            );

            assert_real_array_close(
                scatter.response().absorptance(),
                transfer.response().absorptance(),
            );
        }
    }
}

#[test]
fn transfer2_and_scatter2_sampled_fields_are_equivalent() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for incident_side in [IncidentSide::Left, IncidentSide::Right] {
            let stack = absorbing_stack();

            let input = scalar_input(polarisation, incident_side);

            let (scatter, transfer) = solve_both_fields(&stack, &input);

            let sampling = field_samples(&stack);

            let scatter_fields = scatter.sample_fields(&stack, &input, &sampling).unwrap();

            let transfer_fields = transfer.sample_fields(&stack, &input, &sampling).unwrap();

            assert_fields_close(&scatter_fields, &transfer_fields);
        }
    }
}

#[test]
fn transfer2_and_scatter2_power_balances_are_equivalent() {
    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for incident_side in [IncidentSide::Left, IncidentSide::Right] {
            for stack in [lossless_stack(), absorbing_stack()] {
                let input = scalar_input(polarisation, incident_side);

                let (scatter, transfer) = solve_both_fields(&stack, &input);

                let scatter_balance = scatter.power_balance(&stack, &input).unwrap();

                let transfer_balance = transfer.power_balance(&stack, &input).unwrap();

                assert_power_balance_close(&scatter_balance, &transfer_balance);
            }
        }
    }
}

#[test]
fn transfer2_and_scatter2_fields_are_equivalent_for_array_input() {
    use ndarray::arr1;

    let stack = absorbing_stack();

    let planar = PlanarInput::new(
        arr1(&[6.0, 7.5, 9.0, 10.5]),
        arr1(&[0.2, 0.8, 1.1, 1.5]),
        Polarisation::TransverseMagnetic,
    );
    let input = PlaneWaveInput::new(planar, IncidentSide::Left);

    let scatter = Scatter2
        .solve_plane_wave_internal_fields(&stack, &input)
        .expect("Scatter2 solve should succeed");

    let transfer = Transfer2
        .solve_plane_wave_internal_fields(&stack, &input)
        .expect("Transfer2 solve should succeed");

    assert_boundary_waves_close(
        scatter.boundary_waves().values(),
        transfer.boundary_waves().values(),
    );

    let sampling = field_samples(&stack);

    let scatter_fields = scatter.sample_fields(&stack, &input, &sampling).unwrap();

    let transfer_fields = transfer.sample_fields(&stack, &input, &sampling).unwrap();

    assert_fields_close(&scatter_fields, &transfer_fields);

    let scatter_balance = scatter.power_balance(&stack, &input).unwrap();

    let transfer_balance = transfer.power_balance(&stack, &input).unwrap();

    assert_power_balance_close(&scatter_balance, &transfer_balance);
}

#[test]
fn transfer2_and_scatter2_first_spectral_fields_are_equivalent() {
    let stack = absorbing_stack();

    let input = scalar_input(Polarisation::TransverseMagnetic, IncidentSide::Left);

    let scatter = Scatter2
        .solve_plane_wave_internal_fields_spectral_first_derivative(
            &stack,
            &input,
            SpectralDerivativeVariable::VacuumWavenumber,
        )
        .unwrap();

    let transfer = Transfer2
        .solve_plane_wave_internal_fields_spectral_first_derivative(
            &stack,
            &input,
            SpectralDerivativeVariable::VacuumWavenumber,
        )
        .unwrap();

    let sampling = field_samples(&stack);

    let scatter_fields = scatter
        .sample_fields_spectral_first_derivative(&stack, &input, &sampling)
        .unwrap();

    let transfer_fields = transfer
        .sample_fields_spectral_first_derivative(&stack, &input, &sampling)
        .unwrap();

    assert_field_derivatives_close(&scatter_fields, &transfer_fields);

    let scatter_balance = scatter.power_balance(&stack, &input).unwrap();

    let transfer_balance = transfer.power_balance(&stack, &input).unwrap();

    assert_differentiated_power_balance_close(&scatter_balance, &transfer_balance);
}

#[test]
fn transfer2_and_scatter2_second_spectral_fields_are_equivalent() {
    let stack = absorbing_stack();

    let input = scalar_input(Polarisation::TransverseMagnetic, IncidentSide::Left);

    let scatter = Scatter2
        .solve_plane_wave_internal_fields_spectral_second_derivative(
            &stack,
            &input,
            SpectralDerivativeVariable::VacuumWavenumber,
        )
        .unwrap();

    let transfer = Transfer2
        .solve_plane_wave_internal_fields_spectral_second_derivative(
            &stack,
            &input,
            SpectralDerivativeVariable::VacuumWavenumber,
        )
        .unwrap();

    let sampling = field_samples(&stack);

    let scatter_fields = scatter
        .sample_fields_spectral_second_derivative(&stack, &input, &sampling)
        .unwrap();

    let transfer_fields = transfer
        .sample_fields_spectral_second_derivative(&stack, &input, &sampling)
        .unwrap();

    assert_field_derivatives_close(&scatter_fields, &transfer_fields);

    let scatter_balance = scatter.power_balance(&stack, &input).unwrap();

    let transfer_balance = transfer.power_balance(&stack, &input).unwrap();

    assert_differentiated_power_balance_close(&scatter_balance, &transfer_balance);
}

#[test]
fn transfer2_and_scatter2_first_structural_fields_are_equivalent() {
    let stack = absorbing_stack();

    let input = scalar_input(Polarisation::TransverseMagnetic, IncidentSide::Left);

    let scatter = Scatter2
        .solve_plane_wave_internal_fields_structural_first_derivative(
            &stack,
            &input,
            StructuralDerivativeVariable::ParallelWavenumber,
        )
        .unwrap();

    let transfer = Transfer2
        .solve_plane_wave_internal_fields_structural_first_derivative(
            &stack,
            &input,
            StructuralDerivativeVariable::ParallelWavenumber,
        )
        .unwrap();

    let sampling = field_samples(&stack);

    let scatter_fields = scatter
        .sample_fields_structural_first_derivative(&stack, &input, &sampling)
        .unwrap();

    let transfer_fields = transfer
        .sample_fields_structural_first_derivative(&stack, &input, &sampling)
        .unwrap();

    assert_field_derivatives_close(&scatter_fields, &transfer_fields);

    let scatter_balance = scatter.power_balance(&stack, &input).unwrap();

    let transfer_balance = transfer.power_balance(&stack, &input).unwrap();

    assert_differentiated_power_balance_close(&scatter_balance, &transfer_balance);
}

#[test]
fn transfer2_and_scatter2_second_structural_fields_are_equivalent() {
    let stack = absorbing_stack();

    let input = scalar_input(Polarisation::TransverseMagnetic, IncidentSide::Left);

    let scatter = Scatter2
        .solve_plane_wave_internal_fields_structural_second_derivative(
            &stack,
            &input,
            StructuralDerivativeVariable::ParallelWavenumber,
        )
        .unwrap();

    let transfer = Transfer2
        .solve_plane_wave_internal_fields_structural_second_derivative(
            &stack,
            &input,
            StructuralDerivativeVariable::ParallelWavenumber,
        )
        .unwrap();

    let sampling = field_samples(&stack);

    let scatter_fields = scatter
        .sample_fields_structural_second_derivative(&stack, &input, &sampling)
        .unwrap();

    let transfer_fields = transfer
        .sample_fields_structural_second_derivative(&stack, &input, &sampling)
        .unwrap();

    assert_field_derivatives_close(&scatter_fields, &transfer_fields);

    let scatter_balance = scatter.power_balance(&stack, &input).unwrap();

    let transfer_balance = transfer.power_balance(&stack, &input).unwrap();

    assert_differentiated_power_balance_close(&scatter_balance, &transfer_balance);
}
