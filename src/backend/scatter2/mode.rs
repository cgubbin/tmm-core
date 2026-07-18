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

use ndarray::{Array0, ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar, IncidentSide,
    backend::{
        AnalyticResidual, OutgoingModeResidualBackend, PlanarInput,
        algebra::ScalarAlgebra,
        derivative::{SpectralDerivativeVariable, StructuralDerivativeVariable},
        evaluator::ComplexPlane,
        field::{
            BoundaryWaveSolution, BoundaryWaves, ExteriorBoundaryWaves, InternalFieldRequest,
            ModeFieldResponse, OutgoingModeFieldBackend, value_fields_from_generic,
        },
        isotropic::{
            IsotropicLayerAdmittance, IsotropicLayerFirstDerivatives, IsotropicLayerQuantities,
            IsotropicLayerSecondDerivatives,
        },
        jet::{ArrayJet, ArrayJetFirst},
        mode::{
            DifferentiableOutgoingModeResidualBackend, OutgoingMode, OutgoingModeResponse,
            OutgoingModeStateBackend, ResidualDerivatives,
        },
        scatter2::{
            Scatter2, Scatter2Error, entries::ScatterEntries, fields::retained_boundary_waves,
            workspace::ScatterWorkspace,
        },
    },
    material::{EvaluateDifferentiableMeromorphicMaterial, EvaluateMeromorphicMaterial},
    stack::Stack,
};

impl<C, D, M> OutgoingModeStateBackend<C, D, Stack<M, C::RealField>> for Scatter2
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
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

        let amplitudes = entries.outgoing_mode_amplitudes();
        let admittance = left_admittance(stack, input);

        let residual = outgoing_residual::<C, ndarray::Ix0, _>(entries, &admittance);

        Ok(OutgoingModeResponse::new(
            OutgoingMode::new(input.clone()),
            residual[()],
            amplitudes,
        ))
    }
}

impl<C, D, M> OutgoingModeFieldBackend<C, D, Stack<M, C::RealField>> for Scatter2
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
    M: EvaluateMeromorphicMaterial<C, Real = C::RealField>,
{
    fn outgoing_mode_internal_fields(
        &self,
        stack: &Stack<M, C::RealField>,
        mode: &OutgoingMode<C>,
    ) -> Result<ModeFieldResponse<C>, Self::Error> {
        let workspace: ScatterWorkspace<Array0<C>> = self
            .accumulate_with::<ComplexPlane, _, _, _>(
                stack,
                mode.input(),
                InternalFieldRequest::LayerBoundaries,
            )?;

        let total = workspace.total();

        let admittance = left_admittance(stack, mode.input());

        let residual = outgoing_residual::<C, ndarray::Ix0, _>(total.clone(), &admittance);

        let amplitudes = total.outgoing_mode_amplitudes();

        let extraction = total.outgoing_mode_extraction();
        let generic_fields =
            retained_outgoing_boundary_waves(&workspace, &extraction, mode.input());

        let layers = value_fields_from_generic(generic_fields);

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

impl<C, D, M> OutgoingModeResidualBackend<C, D, Stack<M, C::RealField>> for Scatter2
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
    M: EvaluateMeromorphicMaterial<C, Real = C::RealField>,
{
    type Error = Scatter2Error;

    fn outgoing_mode_residual(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Result<AnalyticResidual<C, D>, Self::Error> {
        let matrix = self.evaluate_with::<ComplexPlane, _, _, _>(stack, input)?;

        let entries = matrix.into_entries();

        let admittance = left_admittance(stack, input);

        let residual = outgoing_residual::<C, D, _>(entries, &admittance);

        Ok(AnalyticResidual::new(residual))
    }

    fn outgoing_mode_residual_first_structural_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<AnalyticResidual<C, D>, Self::Error> {
        /*
         * evaluate_first already:
         *
         * - evaluates the primitive derivative;
         * - applies the requested linear-coordinate chain rule.
         *
         * The exterior admittance must undergo the same transformation.
         */
        let entries =
            self.evaluate_structural_first_with::<ComplexPlane, _, _, _>(stack, input, variable)?;

        let primitive = variable.primitive();

        let mut admittance = left_admittance_first_structural(stack, input, primitive);

        if let Some(rule) = variable.chain_rule(input) {
            admittance = admittance.chain_rule(&rule);
        }

        let residual = outgoing_residual::<C, D, _>(entries, &admittance);

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

        let mut admittance = left_admittance_second_structural(stack, input, primitive);

        if let Some(rule) = variable.chain_rule(input) {
            admittance = admittance.chain_rule(&rule);
        }

        let residual = outgoing_residual::<C, D, _>(entries, &admittance);

        let (value, first, second) = residual.into_parts();

        Ok(AnalyticResidual::with_derivatives(
            value,
            ResidualDerivatives::new(variable.into(), first).with_second(second),
        ))
    }
}

impl<C, D, M> DifferentiableOutgoingModeResidualBackend<C, D, Stack<M, C::RealField>> for Scatter2
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
         * evaluate_first already:
         *
         * - evaluates the primitive derivative;
         * - applies the requested linear-coordinate chain rule.
         *
         * The exterior admittance must undergo the same transformation.
         */
        let entries =
            self.evaluate_spectral_first_with::<ComplexPlane, _, _, _>(stack, input, variable)?;

        let primitive = variable.primitive();

        let mut admittance = left_admittance_first_spectral(stack, input, primitive);

        if let Some(rule) = variable.chain_rule(input) {
            admittance = admittance.chain_rule(&rule);
        }

        let residual = outgoing_residual::<C, D, _>(entries, &admittance);

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

        let mut admittance = left_admittance_second_spectral(stack, input, primitive);

        if let Some(rule) = variable.chain_rule(input) {
            admittance = admittance.chain_rule(&rule);
        }

        let residual = outgoing_residual::<C, D, _>(entries, &admittance);

        let (value, first, second) = residual.into_parts();

        Ok(AnalyticResidual::with_derivatives(
            value,
            ResidualDerivatives::new(variable.into(), first).with_second(second),
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
    M: EvaluateMeromorphicMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    IsotropicLayerQuantities::complex_plane(stack.left_exterior(), input)
        .admittance()
        .into_inner()
}

fn left_admittance_first_structural<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    variable: StructuralDerivativeVariable,
) -> ArrayJetFirst<C, D>
where
    M: EvaluateMeromorphicMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let quantities = IsotropicLayerQuantities::complex_plane(stack.left_exterior(), input);

    match variable {
        StructuralDerivativeVariable::ParallelWavenumberSquared => {
            let derivatives =
                IsotropicLayerFirstDerivatives::parallel_wavenumber_squared(&quantities);

            IsotropicLayerAdmittance::first_jet_from_quantities(&quantities, &derivatives)
        }

        StructuralDerivativeVariable::Thickness(_) => {
            ArrayJetFirst::constant(quantities.admittance().into_inner())
        }

        StructuralDerivativeVariable::ParallelWavenumber => {
            unreachable!("left_admittance_first requires a primitive variable")
        }
    }
}

fn left_admittance_first_spectral<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    variable: SpectralDerivativeVariable,
) -> ArrayJetFirst<C, D>
where
    M: EvaluateDifferentiableMeromorphicMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let quantities = IsotropicLayerQuantities::complex_plane(stack.left_exterior(), input);

    match variable {
        SpectralDerivativeVariable::VacuumWavenumberSquared => {
            let derivatives =
                IsotropicLayerFirstDerivatives::vacuum_wavenumber_squared_complex_plane(
                    stack.left_exterior(),
                    &quantities,
                    input.vacuum_wavenumber(),
                    input.polarisation(),
                );

            IsotropicLayerAdmittance::first_jet_from_quantities(&quantities, &derivatives)
        }

        SpectralDerivativeVariable::VacuumWavenumber => {
            unreachable!("left_admittance_first requires a primitive variable")
        }
    }
}

fn left_admittance_second_structural<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    variable: StructuralDerivativeVariable,
) -> ArrayJet<C, D>
where
    M: EvaluateMeromorphicMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let quantities = IsotropicLayerQuantities::complex_plane(stack.left_exterior(), input);

    match variable {
        StructuralDerivativeVariable::ParallelWavenumberSquared => {
            let derivatives =
                IsotropicLayerSecondDerivatives::parallel_wavenumber_squared(&quantities);

            IsotropicLayerAdmittance::second_jet_from_quantities(&quantities, &derivatives)
        }

        StructuralDerivativeVariable::Thickness(_) => {
            ArrayJet::constant(quantities.admittance().into_inner())
        }

        StructuralDerivativeVariable::ParallelWavenumber => {
            unreachable!("left_admittance_second requires a primitive variable")
        }
    }
}

fn left_admittance_second_spectral<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    variable: SpectralDerivativeVariable,
) -> ArrayJet<C, D>
where
    M: EvaluateDifferentiableMeromorphicMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let quantities = IsotropicLayerQuantities::complex_plane(stack.left_exterior(), input);

    match variable {
        SpectralDerivativeVariable::VacuumWavenumberSquared => {
            let derivatives =
                IsotropicLayerSecondDerivatives::vacuum_wavenumber_squared_complex_plane(
                    stack.left_exterior(),
                    &quantities,
                    input.vacuum_wavenumber(),
                    input.polarisation(),
                );

            IsotropicLayerAdmittance::second_jet_from_quantities(&quantities, &derivatives)
        }

        SpectralDerivativeVariable::VacuumWavenumber => {
            unreachable!("left_admittance_second requires a primitive variable")
        }
    }
}

pub(crate) struct OutgoingModeExtraction<A> {
    pub(crate) incident_side: IncidentSide,
    pub(crate) scale: A,
    pub(crate) left_outgoing: A,
    pub(crate) right_outgoing: A,
}

impl<C> ScatterEntries<Array0<C>>
where
    C: ComplexScalar,
{
    pub(crate) fn outgoing_mode_extraction(&self) -> OutgoingModeExtraction<Array0<C>> {
        let left_column_norm_squared =
            self.s11[()].modulus_squared() + self.s21[()].modulus_squared();

        let right_column_norm_squared =
            self.s12[()].modulus_squared() + self.s22[()].modulus_squared();

        let (incident_side, left_outgoing, right_outgoing, norm_squared) =
            if left_column_norm_squared >= right_column_norm_squared {
                (
                    IncidentSide::Left,
                    self.s11.clone(),
                    self.s21.clone(),
                    left_column_norm_squared,
                )
            } else {
                (
                    IncidentSide::Right,
                    self.s12.clone(),
                    self.s22.clone(),
                    right_column_norm_squared,
                )
            };

        let norm = C::from_real(norm_squared).sqrt();
        let scale = ndarray::arr0(norm.recip());

        OutgoingModeExtraction {
            incident_side,
            left_outgoing: left_outgoing * scale.clone(),
            right_outgoing: right_outgoing * scale.clone(),
            scale,
        }
    }
}

fn retained_outgoing_boundary_waves<C>(
    workspace: &ScatterWorkspace<Array0<C>>,
    extraction: &OutgoingModeExtraction<Array0<C>>,
    planar: &PlanarInput<Array0<C>>,
) -> Vec<crate::backend::field::LayerBoundaryWavesGeneric<Array0<C>>>
where
    C: ComplexScalar,
{
    retained_boundary_waves(workspace, extraction.incident_side, planar)
        .into_iter()
        .map(|each| each.scale(extraction.scale.clone()))
        .collect()
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
        material::Constant,
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
            .layer(
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
            .layer(
                Constant::new(2.25, 1.0),
                Thickness::from_cm(first_thickness_cm).unwrap(),
            )
            .layer(
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
    fn residual_first_spectral_derivative_matches_transfer_backend() {
        let stack = two_layer_stack(0.17, 0.29);

        let input = planar(3.0, 0.4, Polarisation::TransverseElectric);

        for variable in [
            SpectralDerivativeVariable::VacuumWavenumber,
            SpectralDerivativeVariable::VacuumWavenumberSquared,
        ] {
            let scatter = Scatter2::new()
                .outgoing_mode_residual_first_spectral_derivative(&stack, &input, variable)
                .unwrap();

            let transfer = Transfer2::new()
                .outgoing_mode_residual_first_spectral_derivative(&stack, &input, variable)
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
    fn residual_first_derivative_matches_transfer_backend() {
        let stack = two_layer_stack(0.17, 0.29);

        let input = planar(3.0, 0.4, Polarisation::TransverseElectric);

        for variable in [
            StructuralDerivativeVariable::ParallelWavenumber,
            StructuralDerivativeVariable::ParallelWavenumberSquared,
            StructuralDerivativeVariable::Thickness(0),
            StructuralDerivativeVariable::Thickness(1),
        ] {
            let scatter = Scatter2::new()
                .outgoing_mode_residual_first_structural_derivative(&stack, &input, variable)
                .unwrap();

            let transfer = Transfer2::new()
                .outgoing_mode_residual_first_structural_derivative(&stack, &input, variable)
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
    fn residual_second_spectral_derivative_matches_transfer_backend() {
        let stack = two_layer_stack(0.17, 0.29);

        let input = planar(3.0, 0.4, Polarisation::TransverseMagnetic);

        for variable in [
            SpectralDerivativeVariable::VacuumWavenumber,
            SpectralDerivativeVariable::VacuumWavenumberSquared,
        ] {
            let scatter = Scatter2::new()
                .outgoing_mode_residual_second_spectral_derivative(&stack, &input, variable)
                .unwrap();

            let transfer = Transfer2::new()
                .outgoing_mode_residual_second_spectral_derivative(&stack, &input, variable)
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
    fn residual_second_structural_derivative_matches_transfer_backend() {
        let stack = two_layer_stack(0.17, 0.29);

        let input = planar(3.0, 0.4, Polarisation::TransverseMagnetic);

        for variable in [
            StructuralDerivativeVariable::ParallelWavenumber,
            StructuralDerivativeVariable::ParallelWavenumberSquared,
            StructuralDerivativeVariable::Thickness(0),
            StructuralDerivativeVariable::Thickness(1),
        ] {
            let scatter = Scatter2::new()
                .outgoing_mode_residual_second_structural_derivative(&stack, &input, variable)
                .unwrap();

            let transfer = Transfer2::new()
                .outgoing_mode_residual_second_structural_derivative(&stack, &input, variable)
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
