use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::{
        DerivativeVariable, PlaneWaveAmplitudes, PlaneWaveBackend, PlaneWaveInput,
        PlaneWaveResponse,
        scatter2::{Scatter2, Scatter2Error},
    },
    material::Material,
    stack::Stack,
};

impl<C, D, M> PlaneWaveBackend<C, D, Stack<M, C::RealField>> for Scatter2
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
    M: Material<Real = C::RealField>,
{
    type Error = Scatter2Error;

    fn solve_plane_wave(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error> {
        let matrix = self.evaluate(stack, input.planar())?;

        let entries = matrix.into_entries();

        let (reflection, transmission) = entries.amplitudes(input.incident_side());

        Ok(PlaneWaveResponse::new(PlaneWaveAmplitudes::new(
            reflection,
            transmission,
        )))
    }

    fn solve_plane_wave_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error> {
        let entries = self.evaluate_first(stack, input.planar(), variable)?;

        let (reflection, transmission) = entries.amplitudes(input.incident_side());

        Ok(PlaneWaveResponse::from_first_jets(
            reflection,
            transmission,
            variable,
        ))
    }

    fn solve_plane_wave_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error> {
        let entries = self.evaluate_second(stack, input.planar(), variable)?;

        let (reflection, transmission) = entries.amplitudes(input.incident_side());

        Ok(PlaneWaveResponse::from_second_jets(
            reflection,
            transmission,
            variable,
        ))
    }
}

#[cfg(test)]
mod plane_wave_backend_tests {
    use approx::assert_relative_eq;
    use ndarray::{Array0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        IncidentSide, PlanarInput, Polarisation, Thickness, ValidationConfig,
        backend::{
            RawMatrixBackend,
            isotropic::IsotropicLayerAdmittance,
            scatter2::ScatterMatrix2,
            transfer2::{Matrix2, Transfer2},
        },
        material::Constant,
    };

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

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn planar(
        vacuum_wavenumber: f64,
        parallel_wavenumber: f64,
        polarisation: Polarisation,
    ) -> PlanarInput<Array0<C>> {
        PlanarInput::new(
            arr0(c(vacuum_wavenumber)),
            arr0(c(parallel_wavenumber)),
            polarisation,
        )
    }

    fn empty_stack(left_epsilon: f64, right_epsilon: f64) -> Stack<Constant<f64>, f64> {
        Stack::builder(
            Constant::new(left_epsilon, 1.0),
            Constant::new(right_epsilon, 1.0),
        )
        .validation(ValidationConfig::permissive())
        .build()
        .unwrap()
    }

    fn one_layer_stack(thickness_cm: f64) -> Stack<Constant<f64>, f64> {
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
    ) -> Stack<Constant<f64>, f64> {
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

    fn transfer_to_scatter<C, D>(
        matrix: Matrix2<C, D>,
        left_admittance: &ArrayBase<OwnedRepr<C>, D>,
        right_admittance: &ArrayBase<OwnedRepr<C>, D>,
    ) -> ScatterMatrix2<C, D>
    where
        C: ComplexScalar,
        D: Dimension,
    {
        let (a, b, c, d) = matrix.into_parts();

        let left_slope = left_admittance.mapv(|y| -C::i() * y);

        let right_slope = right_admittance.mapv(|y| -C::i() * y);

        let u = a.clone() - b.clone() * right_slope.view();

        let v = c.clone() - d.clone() * right_slope.view();

        let p = a.clone() + b.clone() * right_slope.view();

        let q = c.clone() + d.clone() * right_slope.view();

        let denominator = left_slope.clone() * u.view() - v.view();

        let determinant = a.clone() * d.view() - b.clone() * c.view();

        let two = C::one() + C::one();

        let s11 = (left_slope.clone() * u + v) / denominator.view();

        let s21 = left_slope.mapv(|x| two * x) / denominator.view();

        let s22 = (q - left_slope * p) / denominator.view();

        let s12 = right_slope.mapv(|x| two * x) * determinant / denominator;

        ScatterMatrix2::new(s11, s12, s21, s22)
    }

    fn assert_matrix_close(
        actual: &ScatterMatrix2<C, ndarray::Ix0>,
        expected: &ScatterMatrix2<C, ndarray::Ix0>,
        tolerance: f64,
    ) {
        assert_complex_close(actual.s11()[()], expected.s11()[()], tolerance);
        assert_complex_close(actual.s12()[()], expected.s12()[()], tolerance);
        assert_complex_close(actual.s21()[()], expected.s21()[()], tolerance);
        assert_complex_close(actual.s22()[()], expected.s22()[()], tolerance);
    }

    #[test]
    fn empty_stack_scatter_matches_transfer_conversion() {
        let stack = empty_stack(1.0, 2.25);

        let input = planar(3.0, 0.4, Polarisation::TransverseElectric);

        let transfer = Transfer2::new().evaluate(&stack, &input).unwrap();

        let left = IsotropicLayerAdmittance::evaluate(stack.left_exterior(), &input).into_inner();

        let right = IsotropicLayerAdmittance::evaluate(stack.right_exterior(), &input).into_inner();

        let expected = transfer_to_scatter(transfer, &left, &right);

        let actual = Scatter2::new().evaluate(&stack, &input).unwrap();

        assert_matrix_close(&actual, &expected, 1e-11);
    }

    #[test]
    fn uniform_one_layer_scatter_matches_transfer_conversion() {
        let medium = Constant::new(2.25, 1.0);

        let stack = Stack::builder(medium.clone(), medium)
            .with_layer(medium.clone(), Thickness::from_cm(0.2).unwrap())
            .build()
            .unwrap();

        let input = planar(3.0, 0.4, Polarisation::TransverseElectric);

        let transfer = Transfer2::new().evaluate(&stack, &input).unwrap();

        let left = IsotropicLayerAdmittance::evaluate(stack.left_exterior(), &input).into_inner();

        let right = IsotropicLayerAdmittance::evaluate(stack.right_exterior(), &input).into_inner();

        let expected = transfer_to_scatter(transfer, &left, &right);

        let actual = Scatter2::new().evaluate(&stack, &input).unwrap();

        assert_matrix_close(&actual, &expected, 1e-11);
    }

    #[test]
    fn nonuniform_one_layer_scatter_matches_transfer_conversion() {
        let stack = one_layer_stack(0.2);
        let input = planar(3.0, 0.4, Polarisation::TransverseElectric);

        let transfer = Transfer2::new().evaluate(&stack, &input).unwrap();

        let left = IsotropicLayerAdmittance::evaluate(stack.left_exterior(), &input).into_inner();

        let right = IsotropicLayerAdmittance::evaluate(stack.right_exterior(), &input).into_inner();

        let expected = transfer_to_scatter(transfer, &left, &right);

        let actual = Scatter2::new().evaluate(&stack, &input).unwrap();

        assert_matrix_close(&actual, &expected, 1e-11);
    }

    #[test]
    fn scatter_matrix_matches_transfer_matrix_conversion() {
        let stack = two_layer_stack(0.17, 0.29);

        let input = planar(3.0, 0.4, Polarisation::TransverseElectric);

        let transfer = Transfer2::new().evaluate(&stack, &input).unwrap();

        let left = IsotropicLayerAdmittance::evaluate(stack.left_exterior(), &input).into_inner();

        let right = IsotropicLayerAdmittance::evaluate(stack.right_exterior(), &input).into_inner();

        let expected = transfer_to_scatter(transfer, &left, &right);

        let actual = Scatter2::new().evaluate(&stack, &input).unwrap();

        assert_matrix_close(&actual, &expected, 1e-11);
    }

    #[test]
    fn value_amplitudes_match_raw_scattering_channels() {
        let stack = two_layer_stack(0.17, 0.29);

        for side in [IncidentSide::Left, IncidentSide::Right] {
            let planar = planar(3.0, 0.4, Polarisation::TransverseElectric);

            let input = PlaneWaveInput::new(planar.clone(), side);

            let matrix = Scatter2::new().solve_matrix(&stack, &planar).unwrap();

            let response = Scatter2::new().solve_plane_wave(&stack, &input).unwrap();

            let (expected_r, expected_t) = match side {
                IncidentSide::Left => (matrix.matrix().s11(), matrix.matrix().s21()),

                IncidentSide::Right => (matrix.matrix().s22(), matrix.matrix().s12()),
            };

            assert_eq!(response.reflection(), expected_r,);

            assert_eq!(response.transmission(), expected_t,);
        }
    }

    #[test]
    fn plane_wave_values_match_transfer_backend() {
        let stack = two_layer_stack(0.17, 0.29);

        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            for side in [IncidentSide::Left, IncidentSide::Right] {
                let input = PlaneWaveInput::new(planar(3.0, 0.4, polarisation), side);

                let scatter = Scatter2::new().solve_plane_wave(&stack, &input).unwrap();

                let transfer = Transfer2::new().solve_plane_wave(&stack, &input).unwrap();

                assert_complex_close(scatter.reflection()[()], transfer.reflection()[()], 1e-11);

                assert_complex_close(
                    scatter.transmission()[()],
                    transfer.transmission()[()],
                    1e-11,
                );
            }
        }
    }

    #[test]
    fn first_derivatives_match_transfer_backend() {
        let stack = two_layer_stack(0.17, 0.29);

        let input = PlaneWaveInput::new(
            planar(3.0, 0.4, Polarisation::TransverseElectric),
            IncidentSide::Left,
        );

        for variable in [
            DerivativeVariable::VacuumWavenumber,
            DerivativeVariable::VacuumWavenumberSquared,
            DerivativeVariable::ParallelWavenumber,
            DerivativeVariable::ParallelWavenumberSquared,
            DerivativeVariable::Thickness(0),
            DerivativeVariable::Thickness(1),
        ] {
            let scatter = Scatter2::new()
                .solve_plane_wave_first_derivative(&stack, &input, variable)
                .unwrap();

            let transfer = Transfer2::new()
                .solve_plane_wave_first_derivative(&stack, &input, variable)
                .unwrap();

            let scatter_first = scatter.derivatives().unwrap().first();

            let transfer_first = transfer.derivatives().unwrap().first();

            assert_complex_close(
                scatter_first.reflection()[()],
                transfer_first.reflection()[()],
                1e-9,
            );

            assert_complex_close(
                scatter_first.transmission()[()],
                transfer_first.transmission()[()],
                1e-9,
            );
        }
    }

    #[test]
    fn second_derivatives_match_transfer_backend() {
        let stack = two_layer_stack(0.17, 0.29);

        let input = PlaneWaveInput::new(
            planar(3.0, 0.4, Polarisation::TransverseMagnetic),
            IncidentSide::Right,
        );

        for variable in [
            DerivativeVariable::VacuumWavenumber,
            DerivativeVariable::VacuumWavenumberSquared,
            DerivativeVariable::ParallelWavenumber,
            DerivativeVariable::ParallelWavenumberSquared,
            DerivativeVariable::Thickness(0),
            DerivativeVariable::Thickness(1),
        ] {
            let scatter = Scatter2::new()
                .solve_plane_wave_second_derivative(&stack, &input, variable)
                .unwrap();

            let transfer = Transfer2::new()
                .solve_plane_wave_second_derivative(&stack, &input, variable)
                .unwrap();

            let scatter_derivatives = scatter.derivatives().unwrap();

            let transfer_derivatives = transfer.derivatives().unwrap();

            let scatter_first = scatter_derivatives.first();

            let transfer_first = transfer_derivatives.first();

            let scatter_second = scatter_derivatives.second().unwrap();

            let transfer_second = transfer_derivatives.second().unwrap();

            assert_complex_close(
                scatter_first.reflection()[()],
                transfer_first.reflection()[()],
                1e-9,
            );

            assert_complex_close(
                scatter_first.transmission()[()],
                transfer_first.transmission()[()],
                1e-9,
            );

            assert_complex_close(
                scatter_second.reflection()[()],
                transfer_second.reflection()[()],
                1e-8,
            );

            assert_complex_close(
                scatter_second.transmission()[()],
                transfer_second.transmission()[()],
                1e-8,
            );
        }
    }

    #[test]
    fn first_order_response_records_variable() {
        let stack = one_layer_stack(0.2);

        let input = PlaneWaveInput::new(
            planar(3.0, 0.4, Polarisation::TransverseElectric),
            IncidentSide::Left,
        );

        let variable = DerivativeVariable::Thickness(0);

        let response = Scatter2::new()
            .solve_plane_wave_first_derivative(&stack, &input, variable)
            .unwrap();

        let derivatives = response.derivatives().unwrap();

        assert_eq!(derivatives.variable(), variable,);

        assert!(derivatives.second().is_none());
    }

    #[test]
    fn second_order_response_contains_second_derivative() {
        let stack = one_layer_stack(0.2);

        let input = PlaneWaveInput::new(
            planar(3.0, 0.4, Polarisation::TransverseMagnetic),
            IncidentSide::Right,
        );

        let response = Scatter2::new()
            .solve_plane_wave_second_derivative(
                &stack,
                &input,
                DerivativeVariable::ParallelWavenumberSquared,
            )
            .unwrap();

        assert!(response.derivatives().unwrap().second().is_some());
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
    fn transfer_plane_wave_matches_transfer_matrix_conversion() {
        let stack = two_layer_stack(0.17, 0.29);

        for side in [IncidentSide::Left, IncidentSide::Right] {
            let planar = planar(3.0, 0.4, Polarisation::TransverseElectric);

            let input = PlaneWaveInput::new(planar.clone(), side);

            let transfer_matrix = Transfer2::new().evaluate(&stack, &planar).unwrap();

            let left_admittance =
                IsotropicLayerAdmittance::evaluate(stack.left_exterior(), &planar).into_inner();

            let right_admittance =
                IsotropicLayerAdmittance::evaluate(stack.right_exterior(), &planar).into_inner();

            let equivalent_scatter =
                transfer_to_scatter(transfer_matrix, &left_admittance, &right_admittance);

            let response = Transfer2::new().solve_plane_wave(&stack, &input).unwrap();

            let (expected_r, expected_t) = match side {
                IncidentSide::Left => (equivalent_scatter.s11(), equivalent_scatter.s21()),
                IncidentSide::Right => (equivalent_scatter.s22(), equivalent_scatter.s12()),
            };

            assert_array_close(response.reflection(), expected_r, 1e-12);

            assert_array_close(response.transmission(), expected_t, 1e-12);
        }
    }
}
