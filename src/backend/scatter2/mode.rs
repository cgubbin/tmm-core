//! Outgoing-mode residual for the scalar 2×2 scattering backend.
//!
//! An outgoing planar mode is a pole of the scattering response. Rather than
//! returning a scattering entry directly, this backend constructs the
//! characteristic residual
//!
//! ```text
//! Δ = 2 ξ_L / s21,
//! ```
//!
//! where:
//!
//! - `ξ_L` is the transfer-state slope of the left exterior medium;
//! - `s21` is the left-to-right transmission amplitude.
//!
//! For the conventions shared by [`Transfer2`](crate::backend::transfer2::Transfer2)
//! and [`Scatter2`](super::Scatter2), this is exactly the transfer-backend
//! plane-wave denominator:
//!
//! ```text
//! t_left = 2 ξ_L / Δ.
//! ```
//!
//! Consequently, the two backends return the same residual normalisation, not
//! merely residuals with the same zeros.

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::{
        AnalyticResidual, DerivativeVariable, OutgoingModeBackend, PlanarInput,
        algebra::ScalarAlgebra,
        field::InternalFieldRequest,
        isotropic::{
            IsotropicLayerAdmittance, IsotropicLayerFirstDerivatives, IsotropicLayerQuantities,
            IsotropicLayerSecondDerivatives,
        },
        jet::{ArrayJet, ArrayJetFirst},
        mode::{DifferentiableOutgoingModeBackend, ResidualDerivatives},
        scatter2::{Scatter2, Scatter2Error, entries::ScatterEntries},
    },
    material::{DifferentiableMeromorphicMaterial, Material, MeromorphicMaterial},
    stack::Stack,
};

impl<C, D, M> OutgoingModeBackend<C, D, Stack<M, C::RealField>> for Scatter2
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
    M: MeromorphicMaterial<Real = C::RealField>,
{
    type Error = Scatter2Error;

    fn outgoing_mode_residual(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Result<AnalyticResidual<C, D>, Self::Error> {
        let matrix = self.evaluate_meromorphic(stack, input)?;

        let entries = matrix.into_entries();

        let admittance = left_admittance(stack, input);

        let residual = outgoing_residual::<C, D, _>(entries, &admittance);

        Ok(AnalyticResidual::new(residual))
    }
}

impl<C, D, M> DifferentiableOutgoingModeBackend<C, D, Stack<M, C::RealField>> for Scatter2
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
    M: DifferentiableMeromorphicMaterial<Real = C::RealField>,
{
    fn outgoing_mode_residual_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<AnalyticResidual<C, D>, Self::Error> {
        /*
         * evaluate_first already:
         *
         * - evaluates the primitive derivative;
         * - applies the requested linear-coordinate chain rule.
         *
         * The exterior admittance must undergo the same transformation.
         */
        let entries = self.evaluate_first_meromorphic(stack, input, variable)?;

        let primitive = variable.primitive();

        let mut admittance = left_admittance_first(stack, input, primitive);

        if let Some(rule) = variable.chain_rule(input) {
            admittance = admittance.chain_rule(&rule);
        }

        let residual = outgoing_residual::<C, D, _>(entries, &admittance);

        let (value, first) = residual.into_parts();

        Ok(AnalyticResidual::with_derivatives(
            value,
            ResidualDerivatives::new(variable, first),
        ))
    }

    fn outgoing_mode_residual_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<AnalyticResidual<C, D>, Self::Error> {
        let entries = self.evaluate_second_meromorphic(stack, input, variable)?;

        let primitive = variable.primitive();

        let mut admittance = left_admittance_second(stack, input, primitive);

        if let Some(rule) = variable.chain_rule(input) {
            admittance = admittance.chain_rule(&rule);
        }

        let residual = outgoing_residual::<C, D, _>(entries, &admittance);

        let (value, first, second) = residual.into_parts();

        Ok(AnalyticResidual::with_derivatives(
            value,
            ResidualDerivatives::new(variable, first).with_second(second),
        ))
    }
}

pub(crate) fn transfer_state_slope<C, D, A>(admittance: &A) -> A
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D>,
{
    admittance.scale(-C::i())
}

/// Construct the outgoing-mode residual from scattering entries.
///
/// The entry type may be a sampled array, first-order jet, or second-order jet.
fn outgoing_residual<C, D, A>(entries: ScatterEntries<A>, left_admittance: &A) -> A
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D>,
{
    let slope = transfer_state_slope(left_admittance);

    let two = A::constant_like(slope.value(), C::one() + C::one());

    let numerator = two.multiply(&slope);

    /*
     * s21 is transmission from left to right.
     */
    numerator.divide(&entries.s21)
}

fn left_admittance<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
) -> ArrayBase<OwnedRepr<C>, D>
where
    M: MeromorphicMaterial<Real = C::RealField>,
    C: ComplexScalar,
    D: Dimension,
{
    IsotropicLayerQuantities::new_meromorphic(stack.left_exterior(), input)
        .admittance()
        .into_inner()
}

fn left_admittance_first<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    variable: DerivativeVariable,
) -> ArrayJetFirst<C, D>
where
    M: DifferentiableMeromorphicMaterial<Real = C::RealField>,
    C: ComplexScalar,
    D: Dimension,
{
    let quantities = IsotropicLayerQuantities::new_meromorphic(stack.left_exterior(), input);

    match variable {
        DerivativeVariable::VacuumWavenumberSquared => {
            let derivatives = IsotropicLayerFirstDerivatives::complex_vacuum_wavenumber_squared(
                stack.left_exterior(),
                &quantities,
                input.vacuum_wavenumber(),
                input.polarisation(),
            );

            IsotropicLayerAdmittance::first_jet_from_quantities(&quantities, &derivatives)
        }

        DerivativeVariable::ParallelWavenumberSquared => {
            let derivatives =
                IsotropicLayerFirstDerivatives::parallel_wavenumber_squared(&quantities);

            IsotropicLayerAdmittance::first_jet_from_quantities(&quantities, &derivatives)
        }

        DerivativeVariable::Thickness(_) => {
            ArrayJetFirst::constant(quantities.admittance().into_inner())
        }

        DerivativeVariable::VacuumWavenumber | DerivativeVariable::ParallelWavenumber => {
            unreachable!("left_admittance_first requires a primitive variable")
        }
    }
}

fn left_admittance_second<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    variable: DerivativeVariable,
) -> ArrayJet<C, D>
where
    M: DifferentiableMeromorphicMaterial<Real = C::RealField>,
    C: ComplexScalar,
    D: Dimension,
{
    let quantities = IsotropicLayerQuantities::new_meromorphic(stack.left_exterior(), input);

    match variable {
        DerivativeVariable::VacuumWavenumberSquared => {
            let derivatives = IsotropicLayerSecondDerivatives::complex_vacuum_wavenumber_squared(
                stack.left_exterior(),
                &quantities,
                input.vacuum_wavenumber(),
                input.polarisation(),
            );

            IsotropicLayerAdmittance::second_jet_from_quantities(&quantities, &derivatives)
        }

        DerivativeVariable::ParallelWavenumberSquared => {
            let derivatives =
                IsotropicLayerSecondDerivatives::parallel_wavenumber_squared(&quantities);

            IsotropicLayerAdmittance::second_jet_from_quantities(&quantities, &derivatives)
        }

        DerivativeVariable::Thickness(_) => {
            ArrayJet::constant(quantities.admittance().into_inner())
        }

        DerivativeVariable::VacuumWavenumber | DerivativeVariable::ParallelWavenumber => {
            unreachable!("left_admittance_second requires a primitive variable")
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        Thickness, ValidationConfig,
        backend::{Polarisation, transfer2::Transfer2},
        material::{Constant, enums::IsotropicMaterial},
    };

    fn assert_complex_close(actual: Complex64, expected: Complex64, tolerance: f64) {
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

    #[test]
    fn residual_matches_transfer_backend() {
        let stack = two_layer_stack(0.17, 0.29);

        let input = planar(3.0, 0.4, Polarisation::TransverseElectric);

        let scatter = Scatter2::new()
            .outgoing_mode_residual(&stack, &input)
            .unwrap();

        let transfer = Transfer2::new()
            .outgoing_mode_residual(&stack, &input)
            .unwrap();

        assert_complex_close(scatter.value()[()], transfer.value()[()], 1e-11);
    }

    #[test]
    fn residual_first_derivative_matches_transfer_backend() {
        let stack = two_layer_stack(0.17, 0.29);

        let input = planar(3.0, 0.4, Polarisation::TransverseElectric);

        for variable in [
            DerivativeVariable::VacuumWavenumber,
            DerivativeVariable::VacuumWavenumberSquared,
            DerivativeVariable::ParallelWavenumber,
            DerivativeVariable::ParallelWavenumberSquared,
            DerivativeVariable::Thickness(0),
            DerivativeVariable::Thickness(1),
        ] {
            let scatter = Scatter2::new()
                .outgoing_mode_residual_first_derivative(&stack, &input, variable)
                .unwrap();

            let transfer = Transfer2::new()
                .outgoing_mode_residual_first_derivative(&stack, &input, variable)
                .unwrap();

            assert_complex_close(scatter.value()[()], transfer.value()[()], 1e-10);

            assert_complex_close(
                scatter.derivatives().unwrap().first()[()],
                transfer.derivatives().unwrap().first()[()],
                1e-9,
            );
        }
    }

    #[test]
    fn residual_second_derivative_matches_transfer_backend() {
        let stack = two_layer_stack(0.17, 0.29);

        let input = planar(3.0, 0.4, Polarisation::TransverseMagnetic);

        for variable in [
            DerivativeVariable::VacuumWavenumber,
            DerivativeVariable::VacuumWavenumberSquared,
            DerivativeVariable::ParallelWavenumber,
            DerivativeVariable::ParallelWavenumberSquared,
            DerivativeVariable::Thickness(0),
            DerivativeVariable::Thickness(1),
        ] {
            let scatter = Scatter2::new()
                .outgoing_mode_residual_second_derivative(&stack, &input, variable)
                .unwrap();

            let transfer = Transfer2::new()
                .outgoing_mode_residual_second_derivative(&stack, &input, variable)
                .unwrap();

            let scatter_derivatives = scatter.derivatives().unwrap();

            let transfer_derivatives = transfer.derivatives().unwrap();

            assert_complex_close(scatter.value()[()], transfer.value()[()], 1e-10);

            assert_complex_close(
                scatter_derivatives.first()[()],
                transfer_derivatives.first()[()],
                1e-8,
            );

            assert_complex_close(
                scatter_derivatives.second().unwrap()[()],
                transfer_derivatives.second().unwrap()[()],
                1e-7,
            );
        }
    }

    #[test]
    fn empty_stack_residual_has_transfer_normalisation() {
        let stack = empty_stack(1.0, 2.25);

        let input = planar(3.0, 0.0, Polarisation::TransverseElectric);

        let residual = Scatter2::new()
            .outgoing_mode_residual(&stack, &input)
            .unwrap();

        /*
         * Adapt the sign to the transfer-state slope convention currently
         * retained by the codebase.
         */
        let y_left = c(3.0);
        let y_right = c(4.5);

        let expected = -C::i() * (y_left + y_right);

        assert_complex_close(residual.value()[()], expected, 1e-12);
    }
}
