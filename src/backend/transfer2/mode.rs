//! Outgoing-mode residual for the 2×2 transfer backend.
//!
//! For
//!
//! ```text
//! M = [A B]
//!     [C D]
//! ```
//!
//! and exterior characteristic admittances `Y_L` and `Y_R`, a right-outgoing
//! state is proportional to:
//!
//! ```text
//! [1, -Y_R]ᵀ.
//! ```
//!
//! Propagation through the finite stack gives:
//!
//! ```text
//! u = A - B Y_R
//! v = C - D Y_R.
//! ```
//!
//! The left-outgoing condition is `v = Y_L u`, giving the characteristic
//! residual:
//!
//! ```text
//! f = Y_L u - v.
//! ```
//!
//! This is also the common denominator of the transfer-backend plane-wave
//! reflection and transmission amplitudes.

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::{
        AnalyticResidual, DerivativeVariable, OutgoingModeBackend, PlanarInput,
        isotropic::IsotropicLayerAdmittance,
    },
    material::Material,
    stack::Stack,
};

use super::{Transfer2, TransferError, response::outgoing_residual};

impl<C, D, M> OutgoingModeBackend<C, D, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
    M: Material<Real = C::RealField>,
{
    type Error = TransferError;

    fn outgoing_mode_residual(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Result<AnalyticResidual<C, D>, Self::Error> {
        let matrix = self.evaluate(stack, input)?;

        let left_admittance =
            IsotropicLayerAdmittance::evaluate(stack.left_exterior(), input).into_inner();

        let right_admittance =
            IsotropicLayerAdmittance::evaluate(stack.right_exterior(), input).into_inner();

        let residual =
            outgoing_residual(matrix.into_entries(), &left_admittance, &right_admittance);

        Ok(AnalyticResidual::new(residual))
    }

    fn outgoing_mode_residual_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<AnalyticResidual<C, D>, Self::Error> {
        let matrix = self.evaluate_first(stack, input, variable)?;

        let left_admittance =
            IsotropicLayerAdmittance::evaluate_first(stack.left_exterior(), input, variable);

        let right_admittance =
            IsotropicLayerAdmittance::evaluate_first(stack.right_exterior(), input, variable);

        let residual =
            outgoing_residual(matrix.into_entries(), &left_admittance, &right_admittance);

        Ok(AnalyticResidual::from_first_jet(residual, variable))
    }

    fn outgoing_mode_residual_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<AnalyticResidual<C, D>, Self::Error> {
        let matrix = self.evaluate_second(stack, input, variable)?;

        let left_admittance =
            IsotropicLayerAdmittance::evaluate_second(stack.left_exterior(), input, variable);

        let right_admittance =
            IsotropicLayerAdmittance::evaluate_second(stack.right_exterior(), input, variable);

        let residual =
            outgoing_residual(matrix.into_entries(), &left_admittance, &right_admittance);

        Ok(AnalyticResidual::from_second_jet(residual, variable))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        backend::{IncidentSide, PlaneWaveBackend, PlaneWaveInput, Polarisation},
        material::Constant,
        stack::{Layer, Thickness, ValidationConfig},
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
            .with_layer(
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

        assert_close(
            residual.value()[()],
            c(expected_left_admittance + expected_right_admittance),
            1e-12,
        );
    }

    #[test]
    fn residual_is_plane_wave_denominator() {
        let stack = one_layer_stack(0.2);
        let planar = make_input(3.0, 0.4);

        let residual = Transfer2::new()
            .outgoing_mode_residual(&stack, &planar)
            .unwrap();

        let matrix = Transfer2::new().evaluate(&stack, &planar).unwrap();

        let left = IsotropicLayerAdmittance::evaluate(stack.left_exterior(), &planar).into_inner();

        let right =
            IsotropicLayerAdmittance::evaluate(stack.right_exterior(), &planar).into_inner();

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
            .outgoing_mode_residual_first_derivative(
                &one_layer_stack(thickness),
                &input,
                DerivativeVariable::Thickness(0),
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
            .outgoing_mode_residual_second_derivative(
                &one_layer_stack(thickness),
                &input,
                DerivativeVariable::Thickness(0),
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
            .outgoing_mode_residual_first_derivative(
                &stack,
                &input,
                DerivativeVariable::VacuumWavenumber,
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
            .outgoing_mode_residual_first_derivative(
                &stack,
                &input,
                DerivativeVariable::ParallelWavenumberSquared,
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
            .outgoing_mode_residual_second_derivative(
                &stack,
                &input,
                DerivativeVariable::ParallelWavenumberSquared,
            )
            .unwrap();

        assert!(residual.derivatives().unwrap().second().is_some());
    }
}
