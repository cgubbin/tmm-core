use ndarray::{Array0, ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::{
        AnalyticResidual, OutgoingModeResidualBackend, PlanarInput,
        derivative::{SpectralDerivativeVariable, StructuralDerivativeVariable},
        evaluator::ComplexPlane,
        field::{
            BoundaryWaveSolution, BoundaryWaves, ExteriorBoundaryWaves, InternalFieldRequest,
            ModeFieldResponse, OutgoingModeFieldBackend, value_fields_from_generic,
        },
        isotropic::IsotropicLayerQuantities,
        mode::{
            DifferentiableOutgoingModeResidualBackend, OutgoingMode, OutgoingModeResponse,
            OutgoingModeStateBackend, ResidualDerivatives,
        },
        transfer2::workspace::{TransferWorkspace, right_outgoing_transfer_state},
    },
    material::{EvaluateDifferentiableMeromorphicMaterial, EvaluateMeromorphicMaterial},
    stack::Stack,
};

use super::{Transfer2, Transfer2Error, response::outgoing_residual};

impl<C, M> OutgoingModeStateBackend<C, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    C::RealField: Copy,
    M: EvaluateMeromorphicMaterial<C, Real = C::RealField>,
{
    fn outgoing_mode_state(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<Array0<C>>,
    ) -> Result<OutgoingModeResponse<C>, Self::Error> {
        let workspace = self.accumulate_with::<ComplexPlane, _, _, _>(
            stack,
            input,
            InternalFieldRequest::None,
        )?;

        let entries = workspace.into_total();

        let left_admittance = IsotropicLayerQuantities::complex_plane(stack.left_exterior(), input)
            .into_admittance()
            .into_inner();

        let right_admittance =
            IsotropicLayerQuantities::complex_plane(stack.right_exterior(), input)
                .into_admittance()
                .into_inner();

        let amplitudes = entries
            .outgoing_mode_extraction(&right_admittance)
            .amplitudes();

        let residual =
            outgoing_residual::<C, ndarray::Ix0, _>(&entries, &left_admittance, &right_admittance);

        Ok(OutgoingModeResponse::new(
            OutgoingMode::new(input.clone()),
            residual[()],
            amplitudes,
        ))
    }
}

impl<C, M> OutgoingModeFieldBackend<C, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    C::RealField: Copy,
    M: EvaluateMeromorphicMaterial<C, Real = C::RealField>,
{
    fn outgoing_mode_internal_fields(
        &self,
        stack: &Stack<M, C::RealField>,
        mode: &OutgoingMode<C>,
    ) -> Result<ModeFieldResponse<C>, Self::Error> {
        let workspace: TransferWorkspace<Array0<C>> = self
            .accumulate_with::<ComplexPlane, _, _, _>(
                stack,
                mode.input(),
                InternalFieldRequest::LayerBoundaries,
            )?;

        let total = workspace.total();

        let left_admittance =
            IsotropicLayerQuantities::complex_plane(stack.left_exterior(), mode.input())
                .into_admittance()
                .into_inner();

        let right_admittance =
            IsotropicLayerQuantities::complex_plane(stack.right_exterior(), mode.input())
                .into_admittance()
                .into_inner();

        let residual =
            outgoing_residual::<C, ndarray::Ix0, _>(&total, &left_admittance, &right_admittance);

        let extraction = total.outgoing_mode_extraction(&right_admittance);

        let amplitudes = extraction.amplitudes();

        let right_state = right_outgoing_transfer_state::<C, ndarray::Ix0, ndarray::Array0<C>>(
            &amplitudes.right_outgoing,
            &right_admittance,
        );

        let layer_waves =
            workspace.reconstruct_layer_boundary_waves::<C, ndarray::Ix0>(right_state);

        let layers = value_fields_from_generic(layer_waves);

        let exterior = ExteriorBoundaryWaves::from_outgoing_values(
            amplitudes.left().clone(),
            amplitudes.right().clone(),
        );

        let response = OutgoingModeResponse::new(mode.clone(), residual[()], amplitudes);

        let boundary_waves = BoundaryWaves::new(exterior, layers);

        Ok(ModeFieldResponse::new(
            response,
            BoundaryWaveSolution::Values(boundary_waves),
        ))
    }
}

impl<C, D, M> OutgoingModeResidualBackend<C, D, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
    M: EvaluateMeromorphicMaterial<C, Real = C::RealField>,
{
    type Error = Transfer2Error;

    fn outgoing_mode_residual(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Result<AnalyticResidual<C, D>, Self::Error> {
        let matrix = self.evaluate_with::<ComplexPlane, _, _, _>(stack, input)?;

        let entries = matrix.into_entries();

        let left_admittance = IsotropicLayerQuantities::complex_plane(stack.left_exterior(), input)
            .into_admittance()
            .into_inner();

        let right_admittance =
            IsotropicLayerQuantities::complex_plane(stack.right_exterior(), input)
                .into_admittance()
                .into_inner();

        let residual = outgoing_residual::<C, D, _>(&entries, &left_admittance, &right_admittance);

        Ok(AnalyticResidual::new(residual))
    }

    fn outgoing_mode_residual_first_structural_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<AnalyticResidual<C, D>, Self::Error> {
        /*
         * evaluate_structural_first_with already:
         *
         * - evaluates the primitive derivative;
         * - applies the requested coordinate chain rule.
         *
         * Both exterior admittances must undergo exactly the
         * same transformation.
         */
        let entries =
            self.evaluate_structural_first_with::<ComplexPlane, _, _, _>(stack, input, variable)?;

        let primitive = variable.primitive();

        let mut left_admittance =
            IsotropicLayerQuantities::evaluate_first_structural_complex_plane(
                stack.left_exterior(),
                input,
                primitive,
            )
            .into_admittance()
            .into_inner();

        let mut right_admittance =
            IsotropicLayerQuantities::evaluate_first_structural_complex_plane(
                stack.right_exterior(),
                input,
                primitive,
            )
            .into_admittance()
            .into_inner();

        if let Some(rule) = variable.chain_rule(input) {
            left_admittance = left_admittance.chain_rule(&rule);

            right_admittance = right_admittance.chain_rule(&rule);
        }

        let residual = outgoing_residual::<C, D, _>(&entries, &left_admittance, &right_admittance);

        let (value, first) = residual.into_parts();

        Ok(AnalyticResidual::with_derivatives(
            value,
            ResidualDerivatives::new(variable.into(), first),
        ))
    }

    fn outgoing_mode_residual_second_structural_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<AnalyticResidual<C, D>, Self::Error> {
        let entries =
            self.evaluate_structural_second_with::<ComplexPlane, _, _, _>(stack, input, variable)?;

        let primitive = variable.primitive();

        let mut left_admittance =
            IsotropicLayerQuantities::evaluate_second_structural_complex_plane(
                stack.left_exterior(),
                input,
                primitive,
            )
            .into_admittance()
            .into_inner();

        let mut right_admittance =
            IsotropicLayerQuantities::evaluate_second_structural_complex_plane(
                stack.right_exterior(),
                input,
                primitive,
            )
            .into_admittance()
            .into_inner();

        if let Some(rule) = variable.chain_rule(input) {
            left_admittance = left_admittance.chain_rule(&rule);

            right_admittance = right_admittance.chain_rule(&rule);
        }

        let residual = outgoing_residual::<C, D, _>(&entries, &left_admittance, &right_admittance);

        let (value, first, second) = residual.into_parts();

        Ok(AnalyticResidual::with_derivatives(
            value,
            ResidualDerivatives::new(variable.into(), first).with_second(second),
        ))
    }
}

impl<C, D, M> DifferentiableOutgoingModeResidualBackend<C, D, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
    M: EvaluateDifferentiableMeromorphicMaterial<C, Real = C::RealField>,
{
    fn outgoing_mode_residual_first_spectral_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Result<AnalyticResidual<C, D>, Self::Error> {
        /*
         * evaluate_spectral_first_with already:
         *
         * - evaluates the primitive derivative;
         * - applies the requested coordinate chain rule.
         *
         * Both exterior admittances must undergo exactly the
         * same transformation.
         */
        let entries =
            self.evaluate_spectral_first_with::<ComplexPlane, _, _, _>(stack, input, variable)?;

        let primitive = variable.primitive();

        let mut left_admittance = IsotropicLayerQuantities::evaluate_first_spectral_complex_plane(
            stack.left_exterior(),
            input,
            primitive,
        )
        .into_admittance()
        .into_inner();

        let mut right_admittance = IsotropicLayerQuantities::evaluate_first_spectral_complex_plane(
            stack.right_exterior(),
            input,
            primitive,
        )
        .into_admittance()
        .into_inner();

        if let Some(rule) = variable.chain_rule(input) {
            left_admittance = left_admittance.chain_rule(&rule);

            right_admittance = right_admittance.chain_rule(&rule);
        }

        let residual = outgoing_residual::<C, D, _>(&entries, &left_admittance, &right_admittance);

        let (value, first) = residual.into_parts();

        Ok(AnalyticResidual::with_derivatives(
            value,
            ResidualDerivatives::new(variable.into(), first),
        ))
    }

    fn outgoing_mode_residual_second_spectral_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Result<AnalyticResidual<C, D>, Self::Error> {
        let entries =
            self.evaluate_spectral_second_with::<ComplexPlane, _, _, _>(stack, input, variable)?;

        let primitive = variable.primitive();

        let mut left_admittance = IsotropicLayerQuantities::evaluate_second_spectral_complex_plane(
            stack.left_exterior(),
            input,
            primitive,
        )
        .into_admittance()
        .into_inner();

        let mut right_admittance =
            IsotropicLayerQuantities::evaluate_second_spectral_complex_plane(
                stack.right_exterior(),
                input,
                primitive,
            )
            .into_admittance()
            .into_inner();

        if let Some(rule) = variable.chain_rule(input) {
            left_admittance = left_admittance.chain_rule(&rule);

            right_admittance = right_admittance.chain_rule(&rule);
        }

        let residual = outgoing_residual::<C, D, _>(&entries, &left_admittance, &right_admittance);

        let (value, first, second) = residual.into_parts();

        Ok(AnalyticResidual::with_derivatives(
            value,
            ResidualDerivatives::new(variable.into(), first).with_second(second),
        ))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        backend::{
            Polarisation,
            derivative::{
                DerivativeVariable, SpectralDerivativeVariable, StructuralDerivativeVariable,
            },
            evaluator::RealAxis,
        },
        material::Constant,
        stack::{Thickness, ValidationConfig},
    };

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
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

    fn make_input(vacuum_wavenumber: f64, parallel_wavenumber: f64) -> PlanarInput<Array0<C>> {
        PlanarInput::new(
            arr0(c(vacuum_wavenumber)),
            arr0(c(parallel_wavenumber)),
            Polarisation::TransverseElectric,
        )
    }

    // Adapt these constructors to the actual Stack API.
    fn empty_stack(left_epsilon: f64, right_epsilon: f64) -> Stack<Constant<f64>, f64> {
        Stack::builder(
            Constant::new(left_epsilon, 1.0),
            Constant::new(right_epsilon, 1.0),
        )
        .validation(ValidationConfig::permissive())
        .build()
        .unwrap()
    }

    fn one_layer_stack(thickness: f64) -> Stack<Constant<f64>, f64> {
        Stack::builder(Constant::new(1.0, 1.0), Constant::new(1.44, 1.0))
            .layer(
                Constant::new(2.25, 1.0),
                Thickness::from_cm(thickness).unwrap(),
            )
            .build()
            .unwrap()
    }

    #[test]
    fn empty_stack_residual_is_sum_of_exterior_admittances() {
        let stack = empty_stack(1.0, 2.25);
        let input = make_input(3.0, 0.0);

        let residual = Transfer2::new()
            .outgoing_mode_residual(&stack, &input)
            .unwrap();

        let expected_left_admittance = 3.0;
        let expected_right_admittance = 4.5;
        let expected = -C::i() * (expected_right_admittance + expected_left_admittance);

        assert_close(residual.value()[()], expected, 1e-12);
    }

    #[test]
    fn residual_is_plane_wave_denominator() {
        let stack = one_layer_stack(0.2);
        let planar = make_input(3.0, 0.4);

        let residual = Transfer2::new()
            .outgoing_mode_residual(&stack, &planar)
            .unwrap();

        let matrix = Transfer2::new()
            .evaluate_with::<RealAxis, _, _, _>(&stack, &planar)
            .unwrap();

        let left = -IsotropicLayerQuantities::complex_plane(stack.left_exterior(), &planar)
            .into_admittance()
            .into_inner()
            * C::i();

        let right = -IsotropicLayerQuantities::complex_plane(stack.right_exterior(), &planar)
            .into_admittance()
            .into_inner()
            * C::i();

        let (a, b, c_, d) = matrix.into_parts();

        let expected = left.clone() * (a - b * right.view()) - (c_ - d * right.view());

        assert_close(residual.value()[()], expected[()], 1e-12);
    }

    #[test]
    fn first_thickness_derivative_matches_finite_difference() {
        let thickness = 0.2;
        let h = 1e-6;
        let input = make_input(3.0, 0.4);

        let analytic = Transfer2::new()
            .outgoing_mode_residual_first_structural_derivative(
                &one_layer_stack(thickness),
                &input,
                StructuralDerivativeVariable::Thickness(0),
            )
            .unwrap();

        let plus = Transfer2::new()
            .outgoing_mode_residual(&one_layer_stack(thickness + h), &input)
            .unwrap();

        let minus = Transfer2::new()
            .outgoing_mode_residual(&one_layer_stack(thickness - h), &input)
            .unwrap();

        let expected = (plus.value()[()] - minus.value()[()]) / (2.0 * h);

        assert_close(analytic.derivatives().unwrap().first()[()], expected, 2e-7);
    }

    #[test]
    fn second_thickness_derivative_matches_finite_difference() {
        let thickness = 0.2;
        let h = 1e-4;
        let input = make_input(3.0, 0.4);

        let analytic = Transfer2::new()
            .outgoing_mode_residual_second_structural_derivative(
                &one_layer_stack(thickness),
                &input,
                StructuralDerivativeVariable::Thickness(0),
            )
            .unwrap();

        let plus = Transfer2::new()
            .outgoing_mode_residual(&one_layer_stack(thickness + h), &input)
            .unwrap();

        let zero = Transfer2::new()
            .outgoing_mode_residual(&one_layer_stack(thickness), &input)
            .unwrap();

        let minus = Transfer2::new()
            .outgoing_mode_residual(&one_layer_stack(thickness - h), &input)
            .unwrap();

        let expected = (plus.value()[()] - c(2.0) * zero.value()[()] + minus.value()[()]) / (h * h);

        assert_close(
            analytic.derivatives().unwrap().second().unwrap()[()],
            expected,
            3e-6,
        );
    }

    #[test]
    fn linear_vacuum_wavenumber_derivative_matches_finite_difference() {
        let stack = one_layer_stack(0.2);

        let vacuum_wavenumber = 3.0;
        let h = 1e-6;

        let input = make_input(vacuum_wavenumber, 0.4);

        let analytic = Transfer2::new()
            .outgoing_mode_residual_first_spectral_derivative(
                &stack,
                &input,
                SpectralDerivativeVariable::VacuumWavenumber,
            )
            .unwrap();

        let plus = Transfer2::new()
            .outgoing_mode_residual(&stack, &make_input(vacuum_wavenumber + h, 0.4))
            .unwrap();

        let minus = Transfer2::new()
            .outgoing_mode_residual(&stack, &make_input(vacuum_wavenumber - h, 0.4))
            .unwrap();

        let expected = (plus.value()[()] - minus.value()[()]) / (2.0 * h);

        assert_close(analytic.derivatives().unwrap().first()[()], expected, 3e-7);
    }

    #[test]
    fn first_order_response_records_requested_variable() {
        let stack = one_layer_stack(0.2);
        let input = make_input(3.0, 0.4);

        let residual = Transfer2::new()
            .outgoing_mode_residual_first_structural_derivative(
                &stack,
                &input,
                StructuralDerivativeVariable::ParallelWavenumberSquared,
            )
            .unwrap();

        let derivatives = residual.derivatives().unwrap();

        assert_eq!(
            derivatives.variable(),
            DerivativeVariable::ParallelWavenumberSquared,
        );
        assert!(derivatives.second().is_none());
    }

    #[test]
    fn second_order_response_contains_second_derivative() {
        let stack = one_layer_stack(0.2);
        let input = make_input(3.0, 0.4);

        let residual = Transfer2::new()
            .outgoing_mode_residual_second_structural_derivative(
                &stack,
                &input,
                StructuralDerivativeVariable::ParallelWavenumberSquared,
            )
            .unwrap();

        assert!(residual.derivatives().unwrap().second().is_some());
    }
}
