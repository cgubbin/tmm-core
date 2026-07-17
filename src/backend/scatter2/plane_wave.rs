use ndarray::{ArrayBase, Dimension, OwnedRepr};
use num_traits::Float;

use crate::{
    ComplexScalar, IncidentSide, PlanarInput,
    backend::{
        PlaneWaveBackend, PlaneWaveInput, PlaneWaveResponse,
        derivative::{SpectralDerivativeVariable, StructuralDerivativeVariable},
        evaluator::RealAxis,
        isotropic::IsotropicLayerAdmittance,
        jet::{ArrayJet, ArrayJetFirst},
        plane_wave::DifferentiablePlaneWaveBackend,
        scatter2::{Scatter2, Scatter2Error},
    },
    material::{DifferentiableMaterial, Material},
    stack::Stack,
};

impl<C, D, M> PlaneWaveBackend<C, D, Stack<M, C::RealField>> for Scatter2
where
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
    M: Material<Real = C::RealField>,
{
    type Error = Scatter2Error;

    fn solve_plane_wave(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input();
        let matrix = self.evaluate_with::<RealAxis, _, _, _>(stack, &planar)?;

        let entries = matrix.into_entries();

        let (reflection, transmission) = entries.amplitudes(input.incident_side());

        let response = plane_wave_from_amplitudes(
            reflection,
            transmission,
            &planar,
            stack,
            input.incident_side(),
        );

        Ok(response)
    }

    fn solve_plane_wave_structural_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input();
        let entries =
            self.evaluate_structural_first_with::<RealAxis, _, _, _>(stack, &planar, variable)?;

        let (reflection, transmission) = entries.amplitudes(input.incident_side());

        let response = plane_wave_from_first_jet_amplitudes_structural(
            reflection,
            transmission,
            &planar,
            stack,
            input.incident_side(),
            variable,
        );

        Ok(response)
    }

    fn solve_plane_wave_structural_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<<C>::RealField>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input();
        let entries =
            self.evaluate_structural_second_with::<RealAxis, _, _, _>(stack, &planar, variable)?;

        let (reflection, transmission) = entries.amplitudes(input.incident_side());

        let response = plane_wave_from_second_jet_amplitudes_structural(
            reflection,
            transmission,
            &planar,
            stack,
            input.incident_side(),
            variable,
        );

        Ok(response)
    }
}

impl<C, D, M> DifferentiablePlaneWaveBackend<C, D, Stack<M, C::RealField>> for Scatter2
where
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
    M: DifferentiableMaterial<Real = C::RealField>,
{
    fn solve_plane_wave_spectral_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input();
        let entries =
            self.evaluate_spectral_first_with::<RealAxis, _, _, _>(stack, &planar, variable)?;

        let (reflection, transmission) = entries.amplitudes(input.incident_side());

        let response = plane_wave_from_first_jet_amplitudes_spectral(
            reflection,
            transmission,
            &planar,
            stack,
            input.incident_side(),
            variable,
        );

        Ok(response)
    }

    fn solve_plane_wave_spectral_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input();
        let entries =
            self.evaluate_spectral_second_with::<RealAxis, _, _, _>(stack, &planar, variable)?;

        let (reflection, transmission) = entries.amplitudes(input.incident_side());

        let response = plane_wave_from_second_jet_amplitudes_spectral(
            reflection,
            transmission,
            &planar,
            stack,
            input.incident_side(),
            variable,
        );

        Ok(response)
    }
}

pub(super) fn plane_wave_from_amplitudes<C, D, M>(
    reflection: ArrayBase<OwnedRepr<C>, D>,
    transmission: ArrayBase<OwnedRepr<C>, D>,
    planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    stack: &Stack<M, C::RealField>,
    incident_side: IncidentSide,
) -> PlaneWaveResponse<C, D>
where
    C: ComplexScalar,
    C::RealField: Float,
    D: Dimension,
    M: Material<Real = C::RealField>,
{
    // Construct complex exterior admittances using the lifted input.
    let left_admittance =
        IsotropicLayerAdmittance::evaluate_real_axis(stack.left_exterior(), planar).into_inner();

    let right_admittance =
        IsotropicLayerAdmittance::evaluate_real_axis(stack.right_exterior(), planar).into_inner();

    let (incident_normalisation, transmitted_normalisation) = match incident_side {
        IncidentSide::Left => (left_admittance, right_admittance),

        IncidentSide::Right => (right_admittance, left_admittance),
    };

    PlaneWaveResponse::from_values(
        reflection,
        transmission,
        incident_normalisation,
        transmitted_normalisation,
    )
}

pub(super) fn plane_wave_from_first_jet_amplitudes_structural<C, D, M>(
    reflection: ArrayJetFirst<C, D>,
    transmission: ArrayJetFirst<C, D>,
    planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    stack: &Stack<M, C::RealField>,
    incident_side: IncidentSide,
    variable: StructuralDerivativeVariable,
) -> PlaneWaveResponse<C, D>
where
    C: ComplexScalar,
    C::RealField: Float,
    D: Dimension,
    M: Material<Real = C::RealField>,
{
    // Construct complex exterior admittances using the lifted input.
    let left_admittance = IsotropicLayerAdmittance::evaluate_first_structural_real_axis(
        stack.left_exterior(),
        planar,
        variable,
    );

    let right_admittance = IsotropicLayerAdmittance::evaluate_first_structural_real_axis(
        stack.right_exterior(),
        planar,
        variable,
    );

    let (incident_normalisation, transmitted_normalisation) = match incident_side {
        IncidentSide::Left => (left_admittance, right_admittance),

        IncidentSide::Right => (right_admittance, left_admittance),
    };

    PlaneWaveResponse::from_first_jets(
        reflection,
        transmission,
        incident_normalisation,
        transmitted_normalisation,
        variable.into(),
    )
}

pub(super) fn plane_wave_from_first_jet_amplitudes_spectral<C, D, M>(
    reflection: ArrayJetFirst<C, D>,
    transmission: ArrayJetFirst<C, D>,
    planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    stack: &Stack<M, C::RealField>,
    incident_side: IncidentSide,
    variable: SpectralDerivativeVariable,
) -> PlaneWaveResponse<C, D>
where
    C: ComplexScalar,
    C::RealField: Float,
    D: Dimension,
    M: DifferentiableMaterial<Real = C::RealField>,
{
    // Construct complex exterior admittances using the lifted input.
    let left_admittance = IsotropicLayerAdmittance::evaluate_first_spectral_real_axis(
        stack.left_exterior(),
        planar,
        variable,
    );

    let right_admittance = IsotropicLayerAdmittance::evaluate_first_spectral_real_axis(
        stack.right_exterior(),
        planar,
        variable,
    );

    let (incident_normalisation, transmitted_normalisation) = match incident_side {
        IncidentSide::Left => (left_admittance, right_admittance),

        IncidentSide::Right => (right_admittance, left_admittance),
    };

    PlaneWaveResponse::from_first_jets(
        reflection,
        transmission,
        incident_normalisation,
        transmitted_normalisation,
        variable.into(),
    )
}

pub(super) fn plane_wave_from_second_jet_amplitudes_structural<C, D, M>(
    reflection: ArrayJet<C, D>,
    transmission: ArrayJet<C, D>,
    planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    stack: &Stack<M, C::RealField>,
    incident_side: IncidentSide,
    variable: StructuralDerivativeVariable,
) -> PlaneWaveResponse<C, D>
where
    C: ComplexScalar,
    C::RealField: Float,
    D: Dimension,
    M: Material<Real = C::RealField>,
{
    // Construct complex exterior admittances using the lifted input.
    let left_admittance = IsotropicLayerAdmittance::evaluate_second_structural_real_axis(
        stack.left_exterior(),
        planar,
        variable,
    );

    let right_admittance = IsotropicLayerAdmittance::evaluate_second_structural_real_axis(
        stack.right_exterior(),
        planar,
        variable,
    );

    let (incident_normalisation, transmitted_normalisation) = match incident_side {
        IncidentSide::Left => (left_admittance, right_admittance),

        IncidentSide::Right => (right_admittance, left_admittance),
    };

    PlaneWaveResponse::from_second_jets(
        reflection,
        transmission,
        incident_normalisation,
        transmitted_normalisation,
        variable.into(),
    )
}

pub(super) fn plane_wave_from_second_jet_amplitudes_spectral<C, D, M>(
    reflection: ArrayJet<C, D>,
    transmission: ArrayJet<C, D>,
    planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    stack: &Stack<M, C::RealField>,
    incident_side: IncidentSide,
    variable: SpectralDerivativeVariable,
) -> PlaneWaveResponse<C, D>
where
    C: ComplexScalar,
    C::RealField: Float,
    D: Dimension,
    M: DifferentiableMaterial<Real = C::RealField>,
{
    // Construct complex exterior admittances using the lifted input.
    let left_admittance = IsotropicLayerAdmittance::evaluate_second_spectral_real_axis(
        stack.left_exterior(),
        planar,
        variable,
    );

    let right_admittance = IsotropicLayerAdmittance::evaluate_second_spectral_real_axis(
        stack.right_exterior(),
        planar,
        variable,
    );

    let (incident_normalisation, transmitted_normalisation) = match incident_side {
        IncidentSide::Left => (left_admittance, right_admittance),

        IncidentSide::Right => (right_admittance, left_admittance),
    };

    PlaneWaveResponse::from_second_jets(
        reflection,
        transmission,
        incident_normalisation,
        transmitted_normalisation,
        variable.into(),
    )
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
        material::{Constant, enums::IsotropicMaterial},
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
    ) -> PlanarInput<Array0<f64>> {
        PlanarInput::new(
            arr0(vacuum_wavenumber),
            arr0(parallel_wavenumber),
            polarisation,
        )
    }

    fn c_planar(
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

    fn empty_stack(left_epsilon: f64, right_epsilon: f64) -> Stack<IsotropicMaterial<f64>, f64> {
        Stack::builder(
            Constant::new(left_epsilon, 1.0),
            Constant::new(right_epsilon, 1.0),
        )
        .validation(ValidationConfig::permissive())
        .build()
        .unwrap()
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

        let input = c_planar(3.0, 0.4, Polarisation::TransverseElectric);

        let transfer = Transfer2::new()
            .evaluate_with::<RealAxis, _, _, _>(&stack, &input)
            .unwrap();

        let left = IsotropicLayerAdmittance::evaluate_real_axis(stack.left_exterior(), &input)
            .into_inner();

        let right = IsotropicLayerAdmittance::evaluate_real_axis(stack.right_exterior(), &input)
            .into_inner();

        let expected = transfer_to_scatter(transfer, &left, &right);

        let actual = Scatter2::new()
            .evaluate_with::<RealAxis, _, _, _>(&stack, &input)
            .unwrap();

        assert_matrix_close(&actual, &expected, 1e-11);
    }

    #[test]
    fn uniform_one_layer_scatter_matches_transfer_conversion() {
        let medium = Constant::new(2.25, 1.0);

        let stack = Stack::builder(medium.clone(), medium)
            .with_layer(medium.clone(), Thickness::from_cm(0.2).unwrap())
            .build()
            .unwrap();

        let input = c_planar(3.0, 0.4, Polarisation::TransverseElectric);

        let transfer = Transfer2::new()
            .evaluate_with::<RealAxis, _, _, _>(&stack, &input)
            .unwrap();

        let left = IsotropicLayerAdmittance::evaluate_real_axis(stack.left_exterior(), &input)
            .into_inner();

        let right = IsotropicLayerAdmittance::evaluate_real_axis(stack.right_exterior(), &input)
            .into_inner();

        let expected = transfer_to_scatter(transfer, &left, &right);

        let actual = Scatter2::new()
            .evaluate_with::<RealAxis, _, _, _>(&stack, &input)
            .unwrap();

        assert_matrix_close(&actual, &expected, 1e-11);
    }

    #[test]
    fn nonuniform_one_layer_scatter_matches_transfer_conversion() {
        let stack = one_layer_stack(0.2);
        let input = c_planar(3.0, 0.4, Polarisation::TransverseElectric);

        let transfer = Transfer2::new()
            .evaluate_with::<RealAxis, _, _, _>(&stack, &input)
            .unwrap();

        let left = IsotropicLayerAdmittance::evaluate_real_axis(stack.left_exterior(), &input)
            .into_inner();

        let right = IsotropicLayerAdmittance::evaluate_real_axis(stack.right_exterior(), &input)
            .into_inner();

        let expected = transfer_to_scatter(transfer, &left, &right);

        let actual = Scatter2::new()
            .evaluate_with::<RealAxis, _, _, _>(&stack, &input)
            .unwrap();

        assert_matrix_close(&actual, &expected, 1e-11);
    }

    #[test]
    fn scatter_matrix_matches_transfer_matrix_conversion() {
        let stack = two_layer_stack(0.17, 0.29);

        let input = c_planar(3.0, 0.4, Polarisation::TransverseElectric);

        let transfer = Transfer2::new()
            .evaluate_with::<RealAxis, _, _, _>(&stack, &input)
            .unwrap();

        let left = IsotropicLayerAdmittance::evaluate_real_axis(stack.left_exterior(), &input)
            .into_inner();

        let right = IsotropicLayerAdmittance::evaluate_real_axis(stack.right_exterior(), &input)
            .into_inner();

        let expected = transfer_to_scatter(transfer, &left, &right);

        let actual = Scatter2::new()
            .evaluate_with::<RealAxis, _, _, _>(&stack, &input)
            .unwrap();

        assert_matrix_close(&actual, &expected, 1e-11);
    }

    #[test]
    fn value_amplitudes_match_raw_scattering_channels() {
        let stack = two_layer_stack(0.17, 0.29);

        for side in [IncidentSide::Left, IncidentSide::Right] {
            let planar = planar(3.0, 0.4, Polarisation::TransverseElectric);

            let input = PlaneWaveInput::new(planar.clone(), side);

            let matrix = Scatter2::new().solve_matrix(&stack, &planar).unwrap();

            let response: PlaneWaveResponse<C, ndarray::Ix0> =
                Scatter2::new().solve_plane_wave(&stack, &input).unwrap();

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
    fn first_spectral_derivatives_match_transfer_backend() {
        let stack = two_layer_stack(0.17, 0.29);

        let input = PlaneWaveInput::new(
            planar(3.0, 0.4, Polarisation::TransverseElectric),
            IncidentSide::Left,
        );

        for variable in [
            SpectralDerivativeVariable::VacuumWavenumber,
            SpectralDerivativeVariable::VacuumWavenumberSquared,
        ] {
            let scatter = Scatter2::new()
                .solve_plane_wave_spectral_first_derivative(&stack, &input, variable)
                .unwrap();

            let transfer = Transfer2::new()
                .solve_plane_wave_spectral_first_derivative(&stack, &input, variable)
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
    fn first_structural_derivatives_match_transfer_backend() {
        let stack = two_layer_stack(0.17, 0.29);

        let input = PlaneWaveInput::new(
            planar(3.0, 0.4, Polarisation::TransverseElectric),
            IncidentSide::Left,
        );

        for variable in [
            StructuralDerivativeVariable::ParallelWavenumber,
            StructuralDerivativeVariable::ParallelWavenumberSquared,
            StructuralDerivativeVariable::Thickness(0),
            StructuralDerivativeVariable::Thickness(1),
        ] {
            let scatter = Scatter2::new()
                .solve_plane_wave_structural_first_derivative(&stack, &input, variable)
                .unwrap();

            let transfer = Transfer2::new()
                .solve_plane_wave_structural_first_derivative(&stack, &input, variable)
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
    fn second_structural_derivatives_match_transfer_backend() {
        let stack = two_layer_stack(0.17, 0.29);

        let input = PlaneWaveInput::new(
            planar(3.0, 0.4, Polarisation::TransverseMagnetic),
            IncidentSide::Right,
        );

        for variable in [
            StructuralDerivativeVariable::ParallelWavenumber,
            StructuralDerivativeVariable::ParallelWavenumberSquared,
            StructuralDerivativeVariable::Thickness(0),
            StructuralDerivativeVariable::Thickness(1),
        ] {
            let scatter = Scatter2::new()
                .solve_plane_wave_structural_second_derivative(&stack, &input, variable)
                .unwrap();

            let transfer = Transfer2::new()
                .solve_plane_wave_structural_second_derivative(&stack, &input, variable)
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
    fn second_spectral_derivatives_match_transfer_backend() {
        let stack = two_layer_stack(0.17, 0.29);

        let input = PlaneWaveInput::new(
            planar(3.0, 0.4, Polarisation::TransverseMagnetic),
            IncidentSide::Right,
        );

        for variable in [
            SpectralDerivativeVariable::VacuumWavenumber,
            SpectralDerivativeVariable::VacuumWavenumberSquared,
        ] {
            let scatter = Scatter2::new()
                .solve_plane_wave_spectral_second_derivative(&stack, &input, variable)
                .unwrap();

            let transfer = Transfer2::new()
                .solve_plane_wave_spectral_second_derivative(&stack, &input, variable)
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

        let variable = StructuralDerivativeVariable::Thickness(0);

        let response: PlaneWaveResponse<C, ndarray::Ix0> = Scatter2::new()
            .solve_plane_wave_structural_first_derivative(&stack, &input, variable)
            .unwrap();

        let derivatives = response.derivatives().unwrap();

        assert_eq!(derivatives.variable(), variable.into(),);

        assert!(derivatives.second().is_none());
    }

    #[test]
    fn second_order_response_contains_second_derivative() {
        let stack = one_layer_stack(0.2);

        let input = PlaneWaveInput::new(
            planar(3.0, 0.4, Polarisation::TransverseMagnetic),
            IncidentSide::Right,
        );

        let response: PlaneWaveResponse<C, ndarray::Ix0> = Scatter2::new()
            .solve_plane_wave_structural_second_derivative(
                &stack,
                &input,
                StructuralDerivativeVariable::ParallelWavenumberSquared,
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
            let c_planar = c_planar(3.0, 0.4, Polarisation::TransverseElectric);
            let planar = planar(3.0, 0.4, Polarisation::TransverseElectric);

            let input = PlaneWaveInput::new(planar.clone(), side);

            let transfer_matrix = Transfer2::new()
                .evaluate_with::<RealAxis, _, _, _>(&stack, &c_planar)
                .unwrap();

            let left_admittance =
                IsotropicLayerAdmittance::evaluate_real_axis(stack.left_exterior(), &c_planar)
                    .into_inner();

            let right_admittance =
                IsotropicLayerAdmittance::evaluate_real_axis(stack.right_exterior(), &c_planar)
                    .into_inner();

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

    fn assert_real_close(actual: f64, expected: f64, tolerance: f64) {
        assert_relative_eq!(
            actual,
            expected,
            epsilon = tolerance,
            max_relative = tolerance,
        );
    }

    #[test]
    fn assert_lossless_power_balance_scatter2() {
        let stack = two_layer_stack(1.0, 2.0);

        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            for incident_side in [IncidentSide::Left, IncidentSide::Right] {
                let input = PlaneWaveInput::new(
                    PlanarInput::new(arr0(3.0), arr0(0.4), polarisation),
                    incident_side,
                );

                let response: PlaneWaveResponse<C, ndarray::Ix0> =
                    Scatter2::new().solve_plane_wave(&stack, &input).unwrap();

                assert_real_close(
                    response.power().reflectance()[()] + response.power().transmittance()[()],
                    1.0,
                    1e-10,
                );

                assert_real_close(response.power().absorptance()[()], 0.0, 1e-10);
            }
        }
    }

    #[test]
    fn reflectance_and_transmittance_derivatives_match_finite_difference() {
        let stack = two_layer_stack(1.0, 2.0);
        let backend = Scatter2::new();

        let k0 = 3.0;
        let h = 1e-4;

        let input = |k0| {
            PlaneWaveInput::new(
                PlanarInput::new(arr0(k0), arr0(0.4), Polarisation::TransverseElectric),
                IncidentSide::Left,
            )
        };

        let analytic: PlaneWaveResponse<C, ndarray::Ix0> = backend
            .solve_plane_wave_spectral_second_derivative(
                &stack,
                &input(k0),
                SpectralDerivativeVariable::VacuumWavenumber,
            )
            .unwrap();

        let plus: PlaneWaveResponse<C, ndarray::Ix0> =
            backend.solve_plane_wave(&stack, &input(k0 + h)).unwrap();

        let zero: PlaneWaveResponse<C, ndarray::Ix0> =
            backend.solve_plane_wave(&stack, &input(k0)).unwrap();

        let minus: PlaneWaveResponse<C, ndarray::Ix0> =
            backend.solve_plane_wave(&stack, &input(k0 - h)).unwrap();

        let expected_reflectance_first =
            (plus.power().reflectance()[()] - minus.power().reflectance()[()]) / (2.0 * h);

        let expected_transmittance_first =
            (plus.power().transmittance()[()] - minus.power().transmittance()[()]) / (2.0 * h);

        let expected_reflectance_second = (plus.power().reflectance()[()]
            - 2.0 * zero.power().reflectance()[()]
            + minus.power().reflectance()[()])
            / (h * h);

        let expected_transmittance_second = (plus.power().transmittance()[()]
            - 2.0 * zero.power().transmittance()[()]
            + minus.power().transmittance()[()])
            / (h * h);

        let derivatives = analytic.derivatives().unwrap();

        assert_real_close(
            derivatives.first().power().reflectance()[()],
            expected_reflectance_first,
            1e-6,
        );

        assert_real_close(
            derivatives.first().power().transmittance()[()],
            expected_transmittance_first,
            1e-6,
        );

        assert_real_close(
            derivatives.second().unwrap().power().reflectance()[()],
            expected_reflectance_second,
            1e-4,
        );

        assert_real_close(
            derivatives.second().unwrap().power().transmittance()[()],
            expected_transmittance_second,
            1e-4,
        );
    }
}
