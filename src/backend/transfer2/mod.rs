//! Isotropic 2×2 transfer-matrix backend.
//!
//! [`Transfer2`] implements:
//!
//! - [`RawMatrixBackend`](crate::backend::RawMatrixBackend);
//! - [`PlaneWaveBackend`](crate::backend::PlaneWaveBackend);
//! - [`OutgoingModeBackend`](crate::backend::OutgoingModeBackend).
//!
//! The backend is suitable for moderate optical thicknesses. For strongly
//! evanescent or optically thick stacks, prefer the scattering-matrix backend.

mod backend;
mod error;
mod jet;
mod matrix;
mod mode;
mod plane_wave;
mod response;

pub use backend::Transfer2;
pub use error::Transfer2Error;
pub use matrix::Matrix2;

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::{
        MatrixEvaluation, PlanarInput, RawMatrixBackend,
        evaluator::{ComplexPlane, RealAxis},
        matrix::{
            ComplexMatrixBackend, ComplexMatrixSpectralDerivativeBackend,
            ComplexMatrixStructuralDerivativeBackend, RawMatrixSpectralDerivativeBackend,
            RawMatrixStructuralDerivativeBackend,
        },
    },
    material::{
        DifferentiableMaterial, DifferentiableMeromorphicMaterial, Material, MeromorphicMaterial,
    },
    stack::Stack,
};

impl<C, D, M> RawMatrixBackend<C, D, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    D: Dimension,
    M: Material<Real = C::RealField>,
    C::RealField: Copy,
{
    type Matrix = Matrix2<C, D>;
    type Error = Transfer2Error;

    fn solve_matrix(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        let input = input.clone().to_complex();
        self.evaluate_with::<RealAxis, _, _, _>(stack, &input)
            .map(MatrixEvaluation::new)
    }
}

impl<C, D, M> RawMatrixStructuralDerivativeBackend<C, D, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    D: Dimension,
    M: Material<Real = C::RealField>,
    C::RealField: Copy,
{
    fn solve_matrix_structural_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<<C>::RealField>, D>>,
        variable: super::derivative::StructuralDerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        let input = input.clone().to_complex();
        self.evaluate_structural_first_with::<RealAxis, _, _, _>(stack, &input, variable)
            .map(|j| MatrixEvaluation::from_first_jet(j, variable.into()))
    }

    fn solve_matrix_structural_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<<C>::RealField>, D>>,
        variable: super::derivative::StructuralDerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        let input = input.clone().to_complex();
        self.evaluate_structural_second_with::<RealAxis, _, _, _>(stack, &input, variable)
            .map(|j| MatrixEvaluation::from_second_jet(j, variable.into()))
    }
}

impl<C, D, M> RawMatrixSpectralDerivativeBackend<C, D, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    D: Dimension,
    M: DifferentiableMaterial<Real = C::RealField>,
    C::RealField: Copy,
{
    fn solve_matrix_spectral_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<<C>::RealField>, D>>,
        variable: super::derivative::SpectralDerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        let input = input.clone().to_complex();
        self.evaluate_spectral_first_with::<RealAxis, _, _, _>(stack, &input, variable)
            .map(|j| MatrixEvaluation::from_first_jet(j, variable.into()))
    }

    fn solve_matrix_spectral_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<<C>::RealField>, D>>,
        variable: super::derivative::SpectralDerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        let input = input.clone().to_complex();
        self.evaluate_spectral_second_with::<RealAxis, _, _, _>(stack, &input, variable)
            .map(|j| MatrixEvaluation::from_second_jet(j, variable.into()))
    }
}

impl<C, D, M> ComplexMatrixBackend<C, D, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    D: Dimension,
    M: MeromorphicMaterial<Real = C::RealField>,
    C::RealField: Copy,
{
    type Matrix = Matrix2<C, D>;
    type Error = Transfer2Error;

    fn solve_analytic_matrix(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        self.evaluate_with::<ComplexPlane, _, _, _>(stack, input)
            .map(MatrixEvaluation::new)
    }
}

impl<C, D, M> ComplexMatrixStructuralDerivativeBackend<C, D, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    D: Dimension,
    M: MeromorphicMaterial<Real = C::RealField>,
    C::RealField: Copy,
{
    fn solve_complex_matrix_structural_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: super::derivative::StructuralDerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        self.evaluate_structural_first_with::<RealAxis, _, _, _>(stack, input, variable)
            .map(|j| MatrixEvaluation::from_first_jet(j, variable.into()))
    }

    fn solve_complex_matrix_structural_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: super::derivative::StructuralDerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        self.evaluate_structural_second_with::<RealAxis, _, _, _>(stack, input, variable)
            .map(|j| MatrixEvaluation::from_second_jet(j, variable.into()))
    }
}

impl<C, D, M> ComplexMatrixSpectralDerivativeBackend<C, D, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    D: Dimension,
    M: DifferentiableMeromorphicMaterial<Real = C::RealField>,
    C::RealField: Copy,
{
    fn solve_complex_matrix_spectral_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: super::derivative::SpectralDerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        self.evaluate_spectral_first_with::<RealAxis, _, _, _>(stack, input, variable)
            .map(|j| MatrixEvaluation::from_first_jet(j, variable.into()))
    }

    fn solve_complex_matrix_spectral_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: super::derivative::SpectralDerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        self.evaluate_spectral_second_with::<RealAxis, _, _, _>(stack, input, variable)
            .map(|j| MatrixEvaluation::from_second_jet(j, variable.into()))
    }
}

#[cfg(test)]
mod interface_consistency_tests {
    use approx::assert_relative_eq;
    use nalgebra::ComplexField;
    use ndarray::{Array0, ArrayBase, Dimension, OwnedRepr, arr0};
    use num_complex::Complex64;

    use crate::{
        PlaneWaveResponse,
        backend::{
            DerivativeVariable, OutgoingModeBackend, PlanarInput, PlaneWaveBackend, PlaneWaveInput,
            Polarisation, RawMatrixBackend,
            input::IncidentSide,
            isotropic::IsotropicLayerAdmittance,
            jet::{Jet, JetFirst},
            transfer2::{Matrix2, Transfer2, response::outgoing_residual},
        },
        material::{Constant, enums::IsotropicMaterial},
        stack::{Stack, Thickness, ValidationConfig},
    };

    type C = Complex64;
    type ScalarArray = Array0<C>;
    type RealScalarArray = Array0<f64>;
    type ScalarMatrix = Matrix2<C, ndarray::Ix0>;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
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

    fn planar_input() -> PlanarInput<ScalarArray> {
        PlanarInput::new(arr0(c(3.0)), arr0(c(0.4)), Polarisation::TransverseElectric)
    }

    fn plane_wave_input(side: IncidentSide) -> PlaneWaveInput<ScalarArray> {
        PlaneWaveInput::new(planar_input(), side)
    }

    fn planar_input_re() -> PlanarInput<RealScalarArray> {
        PlanarInput::new(arr0(3.0), arr0(0.4), Polarisation::TransverseElectric)
    }

    fn plane_wave_input_re(side: IncidentSide) -> PlaneWaveInput<RealScalarArray> {
        PlaneWaveInput::new(planar_input_re(), side)
    }

    fn material(epsilon: f64, mu: f64) -> Constant<f64> {
        Constant::new(epsilon, mu)
    }

    fn stack() -> Stack<IsotropicMaterial<f64>, f64> {
        // Adapt to the concrete Stack constructor.
        Stack::builder(Constant::new(1.0, 1.0), Constant::new(1.44, 1.0))
            .with_layer(Constant::new(2.25, 1.0), Thickness::from_cm(0.15).unwrap())
            .with_layer(Constant::new(3.24, 1.0), Thickness::from_cm(0.23).unwrap())
            .build()
            .unwrap()
    }

    fn empty_stack(left_epsilon: f64, right_epsilon: f64) -> Stack<IsotropicMaterial<f64>, f64> {
        // Adapt to the concrete Stack constructor.
        Stack::builder(material(left_epsilon, 1.0), material(right_epsilon, 1.0))
            .validation(ValidationConfig::permissive())
            .build()
            .unwrap()
    }

    fn exterior_admittances(
        stack: &Stack<IsotropicMaterial<f64>, f64>,
        planar: &PlanarInput<ScalarArray>,
    ) -> (ScalarArray, ScalarArray) {
        let left = IsotropicLayerAdmittance::evaluate_real_axis(stack.left_exterior(), planar)
            .into_inner();

        let right = IsotropicLayerAdmittance::evaluate_real_axis(stack.right_exterior(), planar)
            .into_inner();

        (left, right)
    }

    fn assert_close(actual: C, expected: C, tolerance: f64) {
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

    fn assert_matrix_close(
        actual: &Matrix2<C, ndarray::Ix0>,
        expected: &Matrix2<C, ndarray::Ix0>,
        tolerance: f64,
    ) {
        assert_close(actual.m11()[()], expected.m11()[()], tolerance);
        assert_close(actual.m12()[()], expected.m12()[()], tolerance);
        assert_close(actual.m21()[()], expected.m21()[()], tolerance);
        assert_close(actual.m22()[()], expected.m22()[()], tolerance);
    }

    #[test]
    fn raw_matrix_and_plane_wave_value_interfaces_agree() {
        let backend = Transfer2::new();
        let stack = stack();

        for side in [IncidentSide::Left, IncidentSide::Right] {
            let input = plane_wave_input_re(side);
            let input_re = plane_wave_input_re(side);
            let planar = input.planar();

            let raw = backend.solve_matrix(&stack, planar).unwrap().into_matrix();

            let (left, right) = exterior_admittances(&stack, planar);

            let (expected_r, expected_t) = raw.amplitudes(&left, &right, side);

            let response = backend.solve_plane_wave(&stack, &input_re).unwrap();

            assert_array_close(response.reflection(), &expected_r, 1e-12);

            assert_array_close(response.transmission(), &expected_t, 1e-12);
        }
    }

    #[test]
    fn raw_matrix_and_plane_wave_first_derivative_interfaces_agree() {
        let backend = Transfer2::new();
        let stack = stack();
        let input = plane_wave_input(IncidentSide::Right);
        let planar = input.planar();

        let input_re = plane_wave_input_re(IncidentSide::Right);

        let variable = DerivativeVariable::VacuumWavenumber;

        let raw = backend
            .solve_matrix_first_derivative(&stack, planar, variable)
            .unwrap();

        let (matrix, derivatives) = raw.into_parts();
        let derivatives = derivatives.unwrap();

        let matrix_jet = JetFirst::from_parts(matrix, derivatives.into_first());

        let left =
            IsotropicLayerAdmittance::evaluate_first(stack.left_exterior(), planar, variable);

        let right =
            IsotropicLayerAdmittance::evaluate_first(stack.right_exterior(), planar, variable);

        let (expected_r, expected_t) =
            matrix_jet.amplitude_jets(&left, &right, input.incident_side());

        let response = backend
            .solve_plane_wave_first_derivative(&stack, &input_re, variable)
            .unwrap();

        assert_array_close(response.reflection(), expected_r.value(), 1e-12);

        assert_array_close(response.transmission(), expected_t.value(), 1e-12);

        let actual = response.derivatives().unwrap().first();

        assert_array_close(actual.reflection(), expected_r.first(), 1e-12);

        assert_array_close(actual.transmission(), expected_t.first(), 1e-12);
    }

    #[test]
    fn raw_matrix_and_plane_wave_second_derivative_interfaces_agree() {
        let backend = Transfer2::new();
        let stack = stack();
        let input = plane_wave_input(IncidentSide::Left);
        let planar = input.planar();

        let input_re = plane_wave_input_re(IncidentSide::Left);

        let variable = DerivativeVariable::ParallelWavenumberSquared;

        let raw = backend
            .solve_matrix_second_derivative(&stack, planar, variable)
            .unwrap();

        let (matrix, derivatives) = raw.into_parts();
        let (stored_variable, first, second) = derivatives.unwrap().into_parts();

        assert_eq!(stored_variable, variable);

        let matrix_jet = Jet::from_parts(matrix, first, second.unwrap());

        let left =
            IsotropicLayerAdmittance::evaluate_second(stack.left_exterior(), planar, variable);

        let right =
            IsotropicLayerAdmittance::evaluate_second(stack.right_exterior(), planar, variable);

        let (expected_r, expected_t) =
            matrix_jet.amplitude_jets(&left, &right, input.incident_side());

        let response = backend
            .solve_plane_wave_second_derivative(&stack, &input_re, variable)
            .unwrap();

        assert_array_close(response.reflection(), expected_r.value(), 1e-12);

        assert_array_close(response.transmission(), expected_t.value(), 1e-12);

        let derivatives = response.derivatives().unwrap();

        assert_array_close(derivatives.first().reflection(), expected_r.first(), 1e-12);

        assert_array_close(
            derivatives.first().transmission(),
            expected_t.first(),
            1e-12,
        );

        let second = derivatives.second().unwrap();

        assert_array_close(second.reflection(), expected_r.second(), 1e-12);

        assert_array_close(second.transmission(), expected_t.second(), 1e-12);
    }

    #[test]
    fn mode_residual_equals_plane_wave_denominator() {
        let backend = Transfer2::new();
        let stack = stack();
        let planar = planar_input();

        let matrix = backend.solve_matrix(&stack, &planar).unwrap().into_matrix();

        let (left, right) = exterior_admittances(&stack, &planar);

        let expected = outgoing_residual(matrix.into_entries(), &left, &right);

        let actual = backend.outgoing_mode_residual(&stack, &planar).unwrap();

        assert_array_close(actual.value(), &expected, 1e-12);
    }

    #[test]
    fn mode_residual_first_derivative_matches_shared_response_algebra() {
        let backend = Transfer2::new();
        let stack = stack();
        let planar = planar_input();

        let variable = DerivativeVariable::ParallelWavenumber;

        let matrix = backend.evaluate_first(&stack, &planar, variable).unwrap();

        let left =
            IsotropicLayerAdmittance::evaluate_first(stack.left_exterior(), &planar, variable);

        let right =
            IsotropicLayerAdmittance::evaluate_first(stack.right_exterior(), &planar, variable);

        let expected = outgoing_residual(matrix.into_entries(), &left, &right);

        let actual = backend
            .outgoing_mode_residual_first_derivative(&stack, &planar, variable)
            .unwrap();

        assert_array_close(actual.value(), expected.value(), 1e-12);

        assert_array_close(
            actual.derivatives().unwrap().first(),
            expected.first(),
            1e-12,
        );
    }

    #[test]
    fn mode_residual_second_derivative_matches_shared_response_algebra() {
        let backend = Transfer2::new();
        let stack = stack();
        let planar = planar_input();

        let variable = DerivativeVariable::VacuumWavenumber;

        let matrix = backend.evaluate_second(&stack, &planar, variable).unwrap();

        let left =
            IsotropicLayerAdmittance::evaluate_second(stack.left_exterior(), &planar, variable);

        let right =
            IsotropicLayerAdmittance::evaluate_second(stack.right_exterior(), &planar, variable);

        let expected = outgoing_residual(matrix.into_entries(), &left, &right);

        let actual = backend
            .outgoing_mode_residual_second_derivative(&stack, &planar, variable)
            .unwrap();

        let actual_derivatives = actual.derivatives().unwrap();

        assert_array_close(actual.value(), expected.value(), 1e-12);

        assert_array_close(actual_derivatives.first(), expected.first(), 1e-12);

        assert_array_close(
            actual_derivatives.second().unwrap(),
            expected.second(),
            1e-12,
        );
    }

    #[test]
    fn empty_stack_matches_oblique_fresnel_amplitudes_for_both_polarisations() {
        let stack = empty_stack(1.0, 2.25);
        let backend = Transfer2::new();

        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            for side in [IncidentSide::Left, IncidentSide::Right] {
                let planar = PlanarInput::new(arr0(3.0), arr0(0.8), polarisation);
                let c_planar = PlanarInput::new(arr0(c(3.0)), arr0(c(0.8)), polarisation);

                let input = PlaneWaveInput::new(planar, side);

                let response: PlaneWaveResponse<C, ndarray::Ix0> =
                    backend.solve_plane_wave(&stack, &input).unwrap();

                let left = IsotropicLayerAdmittance::evaluate(stack.left_exterior(), &c_planar)
                    .into_inner();

                let right = IsotropicLayerAdmittance::evaluate(stack.right_exterior(), &c_planar)
                    .into_inner();

                let yl = left[()];
                let yr = right[()];

                let (expected_r, expected_t) = match side {
                    IncidentSide::Left => ((yl - yr) / (yl + yr), c(2.0) * yl / (yl + yr)),

                    IncidentSide::Right => ((yr - yl) / (yl + yr), c(2.0) * yr / (yl + yr)),
                };

                assert_complex_close(response.reflection()[()], expected_r, 1e-12);

                assert_complex_close(response.transmission()[()], expected_t, 1e-12);
            }
        }
    }

    #[test]
    fn zero_thickness_layer_does_not_change_matrix_or_plane_wave_response() {
        let backend = Transfer2::new();

        let without_layer = Stack::builder(Constant::new(1.0, 1.0), Constant::new(1.44, 1.0))
            .validation(ValidationConfig::permissive())
            .build()
            .unwrap();

        let with_zero_layer = Stack::builder(Constant::new(1.0, 1.0), Constant::new(1.44, 1.0))
            .with_layer(Constant::new(3.24, 1.0), Thickness::from_cm(0.0).unwrap())
            .validation(ValidationConfig::permissive())
            .build()
            .unwrap();

        let planar = PlanarInput::new(arr0(c(3.0)), arr0(c(0.4)), Polarisation::TransverseMagnetic);
        let re_planar = PlanarInput::new(arr0(3.0), arr0(0.4), Polarisation::TransverseMagnetic);

        let matrix_without = backend.solve_matrix(&without_layer, &planar).unwrap();

        let matrix_with = backend.solve_matrix(&with_zero_layer, &planar).unwrap();

        assert_matrix_close(matrix_with.matrix(), matrix_without.matrix(), 1e-12);

        for side in [IncidentSide::Left, IncidentSide::Right] {
            let input = PlaneWaveInput::new(re_planar.clone(), side);

            let response_without = backend.solve_plane_wave(&without_layer, &input).unwrap();

            let response_with = backend.solve_plane_wave(&with_zero_layer, &input).unwrap();

            assert_array_close(
                response_with.reflection(),
                response_without.reflection(),
                1e-12,
            );

            assert_array_close(
                response_with.transmission(),
                response_without.transmission(),
                1e-12,
            );
        }
    }

    #[test]
    fn uniform_medium_has_zero_reflection() {
        let medium = Constant::new(2.25, 1.0);

        let stack = Stack::builder(medium.clone(), medium)
            .with_layer(medium.clone(), Thickness::from_cm(0.37).unwrap())
            .build()
            .unwrap();

        let backend = Transfer2::new();

        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            for side in [IncidentSide::Left, IncidentSide::Right] {
                dbg!(&polarisation, &side);
                let input =
                    PlaneWaveInput::new(PlanarInput::new(arr0(3.0), arr0(0.6), polarisation), side);

                let response = backend.solve_plane_wave(&stack, &input).unwrap();

                dbg!(&response);

                assert_complex_close(response.reflection()[()], c(0.0), 1e-12);

                assert_relative_eq!(
                    response.transmission()[()].modulus(),
                    1.0,
                    epsilon = 1e-12,
                    max_relative = 1e-12,
                );
            }
        }
    }

    #[test]
    fn lossless_stack_conserves_flux() {
        let stack = Stack::builder(Constant::new(1.0, 1.0), Constant::new(1.44, 1.0))
            .with_layer(Constant::new(2.25, 1.0), Thickness::from_cm(0.15).unwrap())
            .with_layer(Constant::new(3.24, 1.0), Thickness::from_cm(0.27).unwrap())
            .build()
            .unwrap();

        let backend = Transfer2::new();

        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            let planar = PlanarInput::new(arr0(3.0), arr0(0.4), polarisation);
            let c_planar = PlanarInput::new(arr0(c(3.0)), arr0(c(0.4)), polarisation);

            let input = PlaneWaveInput::new(planar, IncidentSide::Left);

            let response: PlaneWaveResponse<C, ndarray::Ix0> =
                backend.solve_plane_wave(&stack, &input).unwrap();

            let left = IsotropicLayerAdmittance::evaluate(stack.left_exterior(), &c_planar);

            let right = IsotropicLayerAdmittance::evaluate(stack.right_exterior(), &c_planar);

            let reflection = response.reflection()[()].modulus_squared();

            let transmission = right.value()[()].real() / left.value()[()].real()
                * response.transmission()[()].modulus_squared();

            assert_relative_eq!(
                reflection + transmission,
                1.0,
                epsilon = 1e-11,
                max_relative = 1e-11,
            );
        }
    }

    #[test]
    fn reciprocal_stack_has_equal_flux_normalised_transmission() {
        let stack = stack();
        let backend = Transfer2::new();

        let planar = PlanarInput::new(arr0(3.0), arr0(0.4), Polarisation::TransverseElectric);
        let c_planar =
            PlanarInput::new(arr0(c(3.0)), arr0(c(0.4)), Polarisation::TransverseElectric);

        let left_response: PlaneWaveResponse<C, ndarray::Ix0> = backend
            .solve_plane_wave(
                &stack,
                &PlaneWaveInput::new(planar.clone(), IncidentSide::Left),
            )
            .unwrap();

        let right_response: PlaneWaveResponse<C, ndarray::Ix0> = backend
            .solve_plane_wave(
                &stack,
                &PlaneWaveInput::new(planar.clone(), IncidentSide::Right),
            )
            .unwrap();

        let left_admittance = IsotropicLayerAdmittance::evaluate(stack.left_exterior(), &c_planar);

        let right_admittance =
            IsotropicLayerAdmittance::evaluate(stack.right_exterior(), &c_planar);

        let left_to_right = right_admittance.value()[()].real()
            / left_admittance.value()[()].real()
            * left_response.transmission()[()].modulus_squared();

        let right_to_left = left_admittance.value()[()].real()
            / right_admittance.value()[()].real()
            * right_response.transmission()[()].modulus_squared();

        assert_relative_eq!(
            left_to_right,
            right_to_left,
            epsilon = 1e-11,
            max_relative = 1e-11,
        );
    }
}
