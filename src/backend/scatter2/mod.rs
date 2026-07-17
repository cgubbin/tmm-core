mod backend;
mod component;
mod entries;
mod error;
mod fields;
mod matrix;
mod mode;
mod plane_wave;
mod raw_matrix;
mod workspace;

pub use backend::Scatter2;
pub use error::Scatter2Error;
pub use matrix::ScatterMatrix2;

#[cfg(test)]
mod tests {
    use crate::{
        IncidentSide, PlanarInput, PlaneWaveInput, PlaneWaveResponse, Polarisation, Stack,
        Thickness, Transfer2, ValidationConfig,
        backend::{
            DifferentiablePlaneWaveBackend, OutgoingModeBackend, PlaneWaveBackend,
            derivative::{SpectralDerivativeVariable, StructuralDerivativeVariable},
            evaluator::RealAxis,
            scatter2::Scatter2,
        },
        material::{Constant, enums::IsotropicMaterial},
    };

    use approx::assert_relative_eq;
    use ndarray::{Array0, ArrayBase, Dimension, OwnedRepr, arr0, array};
    use num_complex::Complex64;

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn planar(
        vacuum_wavenumber: f64,
        parallel_wavenumber: f64,
        polarisation: Polarisation,
    ) -> PlanarInput<Array0<f64>> {
        PlanarInput::new(
            arr0(vacuum_wavenumber),
            arr0(parallel_wavenumber),
            polarisation,
        )
    }

    fn one_layer_stack(thickness_cm: f64) -> Stack<IsotropicMaterial<f64>, f64> {
        Stack::builder(Constant::new(1.0, 1.0), Constant::new(1.44, 1.0))
            .with_layer(
                Constant::new(2.25, 1.0),
                Thickness::from_cm(thickness_cm).unwrap(),
            )
            .validation(ValidationConfig::permissive())
            .build()
            .unwrap()
    }

    fn two_layer_stack(
        first_thickness_cm: f64,
        second_thickness_cm: f64,
    ) -> Stack<IsotropicMaterial<f64>, f64> {
        Stack::builder(Constant::new(1.0, 1.0), Constant::new(1.44, 1.0))
            .with_layer(
                Constant::new(2.25, 1.0),
                Thickness::from_cm(first_thickness_cm).unwrap(),
            )
            .with_layer(
                Constant::new(3.24, 1.0),
                Thickness::from_cm(second_thickness_cm).unwrap(),
            )
            .build()
            .unwrap()
    }

    fn assert_complex_close(actual: C, expected: C, tolerance: f64) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = tolerance,
            max_relative = tolerance,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = tolerance,
            max_relative = tolerance,
        );
    }

    fn assert_array_close<D>(
        actual: &ArrayBase<OwnedRepr<C>, D>,
        expected: &ArrayBase<OwnedRepr<C>, D>,
        tolerance: f64,
    ) where
        D: Dimension,
    {
        assert_eq!(actual.raw_dim(), expected.raw_dim());

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_complex_close(actual, expected, tolerance);
        }
    }

    #[test]
    fn thick_evanescent_layer_has_finite_scattering_response() {
        let stack = Stack::builder(Constant::new(1.0, 1.0), Constant::new(1.0, 1.0))
            .with_layer(Constant::new(1.0, 1.0), Thickness::from_cm(100.0).unwrap())
            .build()
            .unwrap();

        let input = PlaneWaveInput::new(
            PlanarInput::new(arr0(1.0), arr0(2.0), Polarisation::TransverseElectric),
            IncidentSide::Left,
        );

        let response: PlaneWaveResponse<C, ndarray::Ix0> =
            Scatter2::new().solve_plane_wave(&stack, &input).unwrap();

        for value in [response.reflection()[()], response.transmission()[()]] {
            assert!(value.re.is_finite());
            assert!(value.im.is_finite());
        }

        /*
         * The layer is strongly evanescent, so transmission should be very small.
         */
        assert!(response.transmission()[()].norm() < 1e-12);
    }

    #[test]
    fn first_spectral_derivatives_match_transfer_backend_from_both_sides() {
        let stack = two_layer_stack(0.17, 0.29);

        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            for side in [IncidentSide::Left, IncidentSide::Right] {
                let input = PlaneWaveInput::new(planar(3.0, 0.4, polarisation), side);

                for variable in [
                    SpectralDerivativeVariable::VacuumWavenumber,
                    SpectralDerivativeVariable::VacuumWavenumberSquared,
                ] {
                    let scatter: PlaneWaveResponse<C, ndarray::Ix0> = Scatter2::new()
                        .solve_plane_wave_spectral_first_derivative(&stack, &input, variable)
                        .unwrap();

                    let transfer: PlaneWaveResponse<C, ndarray::Ix0> = Transfer2::new()
                        .solve_plane_wave_spectral_first_derivative(&stack, &input, variable)
                        .unwrap();

                    assert_array_close(scatter.reflection(), transfer.reflection(), 1e-11);

                    assert_array_close(scatter.transmission(), transfer.transmission(), 1e-11);

                    let scatter_first = scatter.derivatives().unwrap().first();

                    let transfer_first = transfer.derivatives().unwrap().first();

                    assert_array_close(
                        scatter_first.reflection(),
                        transfer_first.reflection(),
                        1e-9,
                    );

                    assert_array_close(
                        scatter_first.transmission(),
                        transfer_first.transmission(),
                        1e-9,
                    );
                }
            }
        }
    }

    #[test]
    fn first_structural_derivatives_match_transfer_backend_from_both_sides() {
        let stack = two_layer_stack(0.17, 0.29);

        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            for side in [IncidentSide::Left, IncidentSide::Right] {
                let input = PlaneWaveInput::new(planar(3.0, 0.4, polarisation), side);

                for variable in [
                    StructuralDerivativeVariable::ParallelWavenumber,
                    StructuralDerivativeVariable::ParallelWavenumberSquared,
                    StructuralDerivativeVariable::Thickness(0),
                    StructuralDerivativeVariable::Thickness(1),
                ] {
                    let scatter: PlaneWaveResponse<C, ndarray::Ix0> = Scatter2::new()
                        .solve_plane_wave_structural_first_derivative(&stack, &input, variable)
                        .unwrap();

                    let transfer: PlaneWaveResponse<C, ndarray::Ix0> = Transfer2::new()
                        .solve_plane_wave_structural_first_derivative(&stack, &input, variable)
                        .unwrap();

                    assert_array_close(scatter.reflection(), transfer.reflection(), 1e-11);

                    assert_array_close(scatter.transmission(), transfer.transmission(), 1e-11);

                    let scatter_first = scatter.derivatives().unwrap().first();

                    let transfer_first = transfer.derivatives().unwrap().first();

                    assert_array_close(
                        scatter_first.reflection(),
                        transfer_first.reflection(),
                        1e-9,
                    );

                    assert_array_close(
                        scatter_first.transmission(),
                        transfer_first.transmission(),
                        1e-9,
                    );
                }
            }
        }
    }

    #[test]
    fn second_spectral_derivatives_match_transfer_backend_from_both_sides() {
        let stack = two_layer_stack(0.17, 0.29);

        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            for side in [IncidentSide::Left, IncidentSide::Right] {
                let input = PlaneWaveInput::new(planar(3.0, 0.4, polarisation), side);

                for variable in [
                    SpectralDerivativeVariable::VacuumWavenumber,
                    SpectralDerivativeVariable::VacuumWavenumberSquared,
                ] {
                    let scatter: PlaneWaveResponse<C, ndarray::Ix0> = Scatter2::new()
                        .solve_plane_wave_spectral_second_derivative(&stack, &input, variable)
                        .unwrap();

                    let transfer: PlaneWaveResponse<C, ndarray::Ix0> = Transfer2::new()
                        .solve_plane_wave_spectral_second_derivative(&stack, &input, variable)
                        .unwrap();

                    let scatter_derivatives = scatter.derivatives().unwrap();

                    let transfer_derivatives = transfer.derivatives().unwrap();

                    assert_array_close(
                        scatter_derivatives.first().reflection(),
                        transfer_derivatives.first().reflection(),
                        1e-9,
                    );

                    assert_array_close(
                        scatter_derivatives.first().transmission(),
                        transfer_derivatives.first().transmission(),
                        1e-9,
                    );

                    assert_array_close(
                        scatter_derivatives.second().unwrap().reflection(),
                        transfer_derivatives.second().unwrap().reflection(),
                        1e-8,
                    );

                    assert_array_close(
                        scatter_derivatives.second().unwrap().transmission(),
                        transfer_derivatives.second().unwrap().transmission(),
                        1e-8,
                    );
                }
            }
        }
    }

    #[test]
    fn second_structural_derivatives_match_transfer_backend_from_both_sides() {
        let stack = two_layer_stack(0.17, 0.29);

        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            for side in [IncidentSide::Left, IncidentSide::Right] {
                let input = PlaneWaveInput::new(planar(3.0, 0.4, polarisation), side);

                for variable in [
                    StructuralDerivativeVariable::ParallelWavenumber,
                    StructuralDerivativeVariable::ParallelWavenumberSquared,
                    StructuralDerivativeVariable::Thickness(0),
                    StructuralDerivativeVariable::Thickness(1),
                ] {
                    let scatter: PlaneWaveResponse<C, ndarray::Ix0> = Scatter2::new()
                        .solve_plane_wave_structural_second_derivative(&stack, &input, variable)
                        .unwrap();

                    let transfer: PlaneWaveResponse<C, ndarray::Ix0> = Transfer2::new()
                        .solve_plane_wave_structural_second_derivative(&stack, &input, variable)
                        .unwrap();

                    let scatter_derivatives = scatter.derivatives().unwrap();

                    let transfer_derivatives = transfer.derivatives().unwrap();

                    assert_array_close(
                        scatter_derivatives.first().reflection(),
                        transfer_derivatives.first().reflection(),
                        1e-9,
                    );

                    assert_array_close(
                        scatter_derivatives.first().transmission(),
                        transfer_derivatives.first().transmission(),
                        1e-9,
                    );

                    assert_array_close(
                        scatter_derivatives.second().unwrap().reflection(),
                        transfer_derivatives.second().unwrap().reflection(),
                        1e-8,
                    );

                    assert_array_close(
                        scatter_derivatives.second().unwrap().transmission(),
                        transfer_derivatives.second().unwrap().transmission(),
                        1e-8,
                    );
                }
            }
        }
    }

    #[test]
    fn sampled_second_order_plane_wave_response_preserves_shape() {
        let stack = two_layer_stack(0.17, 0.29);

        let vacuum_wavenumber = array![2.5, 3.0, 3.5,];

        let parallel_wavenumber = array![0.2, 0.3, 0.4,];

        let input = PlaneWaveInput::new(
            PlanarInput::new(
                vacuum_wavenumber.clone(),
                parallel_wavenumber,
                Polarisation::TransverseMagnetic,
            ),
            IncidentSide::Right,
        );

        let response: PlaneWaveResponse<C, ndarray::Ix1> = Scatter2::new()
            .solve_plane_wave_spectral_second_derivative(
                &stack,
                &input,
                SpectralDerivativeVariable::VacuumWavenumber,
            )
            .unwrap();

        let expected = vacuum_wavenumber.raw_dim();

        assert_eq!(response.reflection().raw_dim(), expected);

        assert_eq!(response.transmission().raw_dim(), expected);

        let derivatives = response.derivatives().unwrap();

        for amplitudes in [derivatives.first(), derivatives.second().unwrap()] {
            assert_eq!(amplitudes.reflection().raw_dim(), expected);

            assert_eq!(amplitudes.transmission().raw_dim(), expected);
        }
    }

    #[test]
    fn array2_plane_wave_response_preserves_shape() {
        let stack = one_layer_stack(0.2);

        let k0 = array![[2.0, 2.5], [3.0, 3.5],];

        let kp = array![[0.1, 0.2], [0.3, 0.4],];

        let input = PlaneWaveInput::new(
            PlanarInput::new(k0.clone(), kp, Polarisation::TransverseElectric),
            IncidentSide::Left,
        );

        let response: PlaneWaveResponse<C, ndarray::Ix2> =
            Scatter2::new().solve_plane_wave(&stack, &input).unwrap();

        assert_eq!(response.reflection().raw_dim(), k0.raw_dim(),);

        assert_eq!(response.transmission().raw_dim(), k0.raw_dim(),);
    }

    #[test]
    fn empty_stack_structural_derivatives_match_transfer_backend() {
        let stack = Stack::builder(Constant::new(1.0, 1.0), Constant::new(2.25, 1.0))
            .validation(ValidationConfig::permissive())
            .build()
            .unwrap();

        let input = PlaneWaveInput::new(
            planar(3.0, 0.4, Polarisation::TransverseMagnetic),
            IncidentSide::Left,
        );

        for variable in [
            StructuralDerivativeVariable::ParallelWavenumber,
            StructuralDerivativeVariable::ParallelWavenumberSquared,
        ] {
            let scatter: PlaneWaveResponse<C, ndarray::Ix0> = Scatter2::new()
                .solve_plane_wave_structural_second_derivative(&stack, &input, variable)
                .unwrap();

            let transfer: PlaneWaveResponse<C, ndarray::Ix0> = Transfer2::new()
                .solve_plane_wave_structural_second_derivative(&stack, &input, variable)
                .unwrap();

            assert_eq!(scatter.reflection(), transfer.reflection(),);

            assert_eq!(scatter.transmission(), transfer.transmission(),);

            assert_array_close(
                scatter.derivatives().unwrap().first().reflection(),
                transfer.derivatives().unwrap().first().reflection(),
                1e-10,
            );

            assert_array_close(
                scatter
                    .derivatives()
                    .unwrap()
                    .second()
                    .unwrap()
                    .reflection(),
                transfer
                    .derivatives()
                    .unwrap()
                    .second()
                    .unwrap()
                    .reflection(),
                1e-9,
            );
        }
    }

    #[test]
    fn empty_stack_spectral_derivatives_match_transfer_backend() {
        let stack = Stack::builder(Constant::new(1.0, 1.0), Constant::new(2.25, 1.0))
            .validation(ValidationConfig::permissive())
            .build()
            .unwrap();

        let input = PlaneWaveInput::new(
            planar(3.0, 0.4, Polarisation::TransverseMagnetic),
            IncidentSide::Left,
        );

        for variable in [
            SpectralDerivativeVariable::VacuumWavenumber,
            SpectralDerivativeVariable::VacuumWavenumberSquared,
        ] {
            let scatter: PlaneWaveResponse<C, ndarray::Ix0> = Scatter2::new()
                .solve_plane_wave_spectral_second_derivative(&stack, &input, variable)
                .unwrap();

            let transfer: PlaneWaveResponse<C, ndarray::Ix0> = Transfer2::new()
                .solve_plane_wave_spectral_second_derivative(&stack, &input, variable)
                .unwrap();

            assert_eq!(scatter.reflection(), transfer.reflection(),);

            assert_eq!(scatter.transmission(), transfer.transmission(),);

            assert_array_close(
                scatter.derivatives().unwrap().first().reflection(),
                transfer.derivatives().unwrap().first().reflection(),
                1e-10,
            );

            assert_array_close(
                scatter
                    .derivatives()
                    .unwrap()
                    .second()
                    .unwrap()
                    .reflection(),
                transfer
                    .derivatives()
                    .unwrap()
                    .second()
                    .unwrap()
                    .reflection(),
                1e-9,
            );
        }
    }

    #[test]
    fn empty_stack_rejects_thickness_derivative() {
        let stack = Stack::builder(Constant::new(1.0, 1.0), Constant::new(2.25, 1.0))
            .validation(ValidationConfig::permissive())
            .build()
            .unwrap();

        let input = PlanarInput::new(arr0(c(1.0)), arr0(c(2.0)), Polarisation::TransverseElectric);

        let error = Scatter2::new()
            .evaluate_structural_first_with::<RealAxis, _, _, _>(
                &stack,
                &input,
                StructuralDerivativeVariable::Thickness(0),
            )
            .unwrap_err();

        assert_eq!(
            error,
            super::Scatter2Error::ThicknessLayerOutOfBounds {
                requested: 0,
                layer_count: 0,
            },
        );
    }

    #[test]
    fn thick_evanescent_mode_residual_is_finite() {
        let stack = Stack::builder(Constant::new(1.0, 1.0), Constant::new(1.0, 1.0))
            .with_layer(Constant::new(1.0, 1.0), Thickness::from_cm(10.0).unwrap())
            .build()
            .unwrap();

        let input = PlanarInput::new(arr0(c(1.0)), arr0(c(2.0)), Polarisation::TransverseElectric);

        let residual = Scatter2::new()
            .outgoing_mode_residual(&stack, &input)
            .unwrap();

        assert!(residual.value()[()].re.is_finite());
        assert!(residual.value()[()].im.is_finite());
    }
}
