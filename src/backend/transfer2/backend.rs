//! Internal evaluation kernel for the isotropic 2×2 transfer backend.
//!
//! This module evaluates the native transfer matrix of a finite planar stack.
//! It is responsible for:
//!
//! - evaluating isotropic quantities once per finite layer;
//! - constructing each finite-layer transfer matrix;
//! - composing layer matrices in propagation order;
//! - propagating first- and second-order derivatives;
//! - transforming primitive squared-coordinate derivatives to requested
//!   linear coordinates.
//!
//! The two semi-infinite exterior media do not contribute propagation
//! matrices and are therefore not used here. They are evaluated by the
//! plane-wave and outgoing-mode adapters, where their admittances define the
//! physical boundary conditions.
//!
//! If the finite layers are encountered as `L₁, L₂, …, Lₙ`, accumulation is:
//!
//! ```text
//! M = Lₙ … L₂ L₁
//! ```
//!
//! Value-only, first-order, and second-order calculations use distinct return
//! types so derivative arrays are allocated only when requested.

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::{
        PlanarInput,
        derivative::{SpectralDerivativeVariable, StructuralDerivativeVariable},
        evaluator::{ConstitutiveDerivativeEvaluator, ConstitutiveEvaluator},
        isotropic::{
            IsotropicLayerFirstDerivatives, IsotropicLayerQuantities,
            IsotropicLayerSecondDerivatives,
        },
        transfer2::{
            Matrix2, Transfer2Error,
            jet::{Transfer2Jet, Transfer2JetFirst},
        },
    },
    stack::{Layer, Stack},
};

/// Isotropic 2×2 transfer-matrix backend.
#[derive(Copy, Clone, Debug, Default)]
pub struct Transfer2;

impl Transfer2 {
    /// Construct a 2×2 transfer-matrix backend.
    pub const fn new() -> Self {
        Self
    }

    /// Evaluate the transfer matrix without derivatives.
    pub(crate) fn evaluate_with<E, M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Result<Matrix2<C, D>, Transfer2Error>
    where
        E: ConstitutiveEvaluator<C, D, M>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        let mut total = Matrix2::identity_like(input.vacuum_wavenumber());

        for layer in stack.iter() {
            let quantities = IsotropicLayerQuantities::new::<E, _>(layer.material(), input);

            let layer_matrix = Matrix2::from_layer(&quantities, layer.thickness());

            total = &total * &layer_matrix;
        }

        Ok(total)
    }

    /// Evaluate the transfer matrix and its first derivative.
    pub(crate) fn evaluate_structural_first_with<E, M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<Transfer2JetFirst<C, D>, Transfer2Error>
    where
        E: ConstitutiveEvaluator<C, D, M>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        validate_derivative_variable(stack, variable)?;

        let primitive = variable.primitive();

        let mut total =
            Transfer2JetFirst::constant(Matrix2::identity_like(input.vacuum_wavenumber()));

        for (index, layer) in stack.iter().enumerate() {
            let quantities = IsotropicLayerQuantities::new::<E, _>(layer.material(), input);

            let layer = first_layer_jet_structural(index, layer, &quantities, primitive);

            total = total.multiply(&layer);
        }

        if let Some(rule) = variable.chain_rule(input) {
            total = total.chain_rule(&rule);
        }

        Ok(total)
    }

    /// Evaluate the transfer matrix and its first derivative.
    pub(crate) fn evaluate_spectral_first_with<E, M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Result<Transfer2JetFirst<C, D>, Transfer2Error>
    where
        E: ConstitutiveDerivativeEvaluator<C, D, M>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        let primitive = variable.primitive();

        let mut total =
            Transfer2JetFirst::constant(Matrix2::identity_like(input.vacuum_wavenumber()));

        for layer in stack.iter() {
            let quantities = IsotropicLayerQuantities::new::<E, _>(layer.material(), input);

            let layer =
                first_layer_jet_spectral::<E, _, _, _>(layer, &quantities, input, primitive);

            total = total.multiply(&layer);
        }

        if let Some(rule) = variable.chain_rule(input) {
            total = total.chain_rule(&rule);
        }

        Ok(total)
    }

    /// Evaluate the transfer matrix and its first two derivatives.
    pub(crate) fn evaluate_structural_second_with<E, M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<Transfer2Jet<C, D>, Transfer2Error>
    where
        E: ConstitutiveEvaluator<C, D, M>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        validate_derivative_variable(stack, variable)?;

        let primitive = variable.primitive();

        let mut total = Transfer2Jet::constant(Matrix2::identity_like(input.vacuum_wavenumber()));

        for (index, layer) in stack.iter().enumerate() {
            let quantities = IsotropicLayerQuantities::new::<E, _>(layer.material(), input);

            let layer = second_layer_jet_structural(index, layer, &quantities, primitive);

            total = total.multiply(&layer);
        }

        if let Some(rule) = variable.chain_rule(input) {
            total = total.chain_rule(&rule);
        }

        Ok(total)
    }

    /// Evaluate the transfer matrix and its first two derivatives.
    pub(crate) fn evaluate_spectral_second_with<E, M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Result<Transfer2Jet<C, D>, Transfer2Error>
    where
        E: ConstitutiveDerivativeEvaluator<C, D, M>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        let primitive = variable.primitive();

        let mut total = Transfer2Jet::constant(Matrix2::identity_like(input.vacuum_wavenumber()));

        for layer in stack.iter() {
            let quantities = IsotropicLayerQuantities::new::<E, _>(layer.material(), input);

            let layer =
                second_layer_jet_spectral::<E, _, _, _>(layer, &quantities, input, primitive);

            total = total.multiply(&layer);
        }

        if let Some(rule) = variable.chain_rule(input) {
            total = total.chain_rule(&rule);
        }

        Ok(total)
    }
}

fn validate_derivative_variable<M, R>(
    stack: &Stack<M, R>,
    variable: StructuralDerivativeVariable,
) -> Result<(), Transfer2Error> {
    if let StructuralDerivativeVariable::Thickness(requested) = variable {
        let layer_count = stack.len();

        if requested >= layer_count {
            return Err(Transfer2Error::ThicknessLayerOutOfBounds {
                requested,
                layer_count,
            });
        }
    }

    Ok(())
}

fn first_layer_jet_structural<M, C, D>(
    layer_index: usize,
    layer: &Layer<M, C::RealField>,
    quantities: &IsotropicLayerQuantities<C, D>,
    variable: StructuralDerivativeVariable,
) -> Transfer2JetFirst<C, D>
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let matrix = Matrix2::from_layer(quantities, layer.thickness());

    match variable {
        StructuralDerivativeVariable::Thickness(requested) if requested == layer_index => {
            let first = Matrix2::thickness_derivative(quantities, layer.thickness());

            Transfer2JetFirst::from_parts(matrix, first)
        }

        StructuralDerivativeVariable::Thickness(_) => Transfer2JetFirst::constant(matrix),

        StructuralDerivativeVariable::ParallelWavenumberSquared => {
            let derivatives =
                IsotropicLayerFirstDerivatives::parallel_wavenumber_squared(quantities);

            let first = Matrix2::spectral_derivative(quantities, layer.thickness(), &derivatives);

            Transfer2JetFirst::from_parts(matrix, first)
        }

        StructuralDerivativeVariable::ParallelWavenumber => {
            unreachable!("primitive() returned a linear derivative variable")
        }
    }
}

fn first_layer_jet_spectral<E, M, C, D>(
    layer: &Layer<M, C::RealField>,
    quantities: &IsotropicLayerQuantities<C, D>,
    input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    variable: SpectralDerivativeVariable,
) -> Transfer2JetFirst<C, D>
where
    E: ConstitutiveDerivativeEvaluator<C, D, M>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let matrix = Matrix2::from_layer(quantities, layer.thickness());

    match variable {
        SpectralDerivativeVariable::VacuumWavenumberSquared => {
            let derivatives = IsotropicLayerFirstDerivatives::vacuum_wavenumber_squared::<E, _>(
                layer.material(),
                quantities,
                input.vacuum_wavenumber(),
                input.polarisation(),
            );

            let first = Matrix2::spectral_derivative(quantities, layer.thickness(), &derivatives);

            Transfer2JetFirst::from_parts(matrix, first)
        }

        SpectralDerivativeVariable::VacuumWavenumber => {
            unreachable!("primitive() returned a linear derivative variable")
        }
    }
}

fn second_layer_jet_structural<M, C, D>(
    layer_index: usize,
    layer: &Layer<M, C::RealField>,
    quantities: &IsotropicLayerQuantities<C, D>,
    variable: StructuralDerivativeVariable,
) -> Transfer2Jet<C, D>
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let matrix = Matrix2::from_layer(quantities, layer.thickness());

    match variable {
        StructuralDerivativeVariable::Thickness(requested) if requested == layer_index => {
            let first = Matrix2::thickness_derivative(quantities, layer.thickness());

            let second = Matrix2::thickness_second_derivative(quantities, layer.thickness());

            Transfer2Jet::from_parts(matrix, first, second)
        }

        StructuralDerivativeVariable::Thickness(_) => Transfer2Jet::constant(matrix),

        StructuralDerivativeVariable::ParallelWavenumberSquared => {
            let derivatives =
                IsotropicLayerSecondDerivatives::parallel_wavenumber_squared(quantities);

            let first =
                Matrix2::spectral_derivative(quantities, layer.thickness(), derivatives.first());

            let second =
                Matrix2::spectral_second_derivative(quantities, layer.thickness(), &derivatives);

            Transfer2Jet::from_parts(matrix, first, second)
        }

        StructuralDerivativeVariable::ParallelWavenumber => {
            unreachable!("primitive() returned a linear derivative variable")
        }
    }
}

fn second_layer_jet_spectral<E, M, C, D>(
    layer: &Layer<M, C::RealField>,
    quantities: &IsotropicLayerQuantities<C, D>,
    input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    variable: SpectralDerivativeVariable,
) -> Transfer2Jet<C, D>
where
    E: ConstitutiveDerivativeEvaluator<C, D, M>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let matrix = Matrix2::from_layer(quantities, layer.thickness());

    match variable {
        SpectralDerivativeVariable::VacuumWavenumberSquared => {
            let derivatives = IsotropicLayerSecondDerivatives::vacuum_wavenumber_squared::<E, _>(
                layer.material(),
                quantities,
                input.vacuum_wavenumber(),
                input.polarisation(),
            );

            let first =
                Matrix2::spectral_derivative(quantities, layer.thickness(), derivatives.first());

            let second =
                Matrix2::spectral_second_derivative(quantities, layer.thickness(), &derivatives);

            Transfer2Jet::from_parts(matrix, first, second)
        }

        SpectralDerivativeVariable::VacuumWavenumber => {
            unreachable!("primitive() returned a linear derivative variable")
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
        backend::{Polarisation, evaluator::RealAxis},
        material::Constant,
        stack::{Thickness, ValidationConfig},
    };

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn make_input(vacuum_wavenumber: f64, parallel_wavenumber: f64) -> PlanarInput<Array0<C>> {
        PlanarInput::new(
            arr0(c(vacuum_wavenumber)),
            arr0(c(parallel_wavenumber)),
            Polarisation::TransverseElectric,
        )
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

    fn finite_difference_first(
        plus: &Matrix2<C, ndarray::Ix0>,
        minus: &Matrix2<C, ndarray::Ix0>,
        h: f64,
    ) -> Matrix2<C, ndarray::Ix0> {
        &(plus - minus) * c(1.0 / (2.0 * h))
    }

    fn finite_difference_second(
        plus: &Matrix2<C, ndarray::Ix0>,
        zero: &Matrix2<C, ndarray::Ix0>,
        minus: &Matrix2<C, ndarray::Ix0>,
        h: f64,
    ) -> Matrix2<C, ndarray::Ix0> {
        let twice_zero = zero * c(2.0);
        let numerator = &(plus - &twice_zero) + minus;

        &numerator * c(1.0 / (h * h))
    }

    fn two_layer_stack(first_thickness: f64, second_thickness: f64) -> Stack<Constant<f64>, f64> {
        Stack::builder(Constant::new(1.0, 1.0), Constant::new(1.5, 1.0))
            .layer(
                Constant::new(2.25, 1.0),
                Thickness::from_cm(first_thickness).unwrap(),
            )
            .layer(
                Constant::new(3.24, 1.0),
                Thickness::from_cm(second_thickness).unwrap(),
            )
            .build()
            .unwrap()
    }

    #[test]
    fn empty_stack_evaluates_to_identity() {
        let stack = Stack::builder(Constant::new(1.0, 1.0), Constant::new(1.5, 1.0))
            .validation(ValidationConfig::permissive())
            .build()
            .unwrap();
        let input = make_input(3.0, 0.4);

        let matrix = Transfer2::new()
            .evaluate_with::<RealAxis, _, _, _>(&stack, &input)
            .unwrap();

        let expected = Matrix2::identity_like(input.vacuum_wavenumber());

        assert_matrix_close(&matrix, &expected, 1e-12);
    }

    #[test]
    fn thickness_first_derivative_matches_finite_difference() {
        let d0 = 0.15;
        let d1 = 0.23;
        let h = 1e-6;

        let stack = two_layer_stack(d0, d1);
        let input = make_input(3.0, 0.4);

        let jet = Transfer2::new()
            .evaluate_structural_first_with::<RealAxis, _, _, _>(
                &stack,
                &input,
                StructuralDerivativeVariable::Thickness(0),
            )
            .unwrap();

        let (_, analytic) = jet.into_parts();

        let plus = Transfer2::new()
            .evaluate_with::<RealAxis, _, _, _>(&two_layer_stack(d0 + h, d1), &input)
            .unwrap();

        let minus = Transfer2::new()
            .evaluate_with::<RealAxis, _, _, _>(&two_layer_stack(d0 - h, d1), &input)
            .unwrap();

        let expected = finite_difference_first(&plus, &minus, h);

        assert_matrix_close(&analytic, &expected, 1e-7);
    }

    #[test]
    fn thickness_second_derivative_matches_finite_difference() {
        let d0 = 0.15;
        let d1 = 0.23;
        let h = 1e-4;

        let stack = two_layer_stack(d0, d1);
        let input = make_input(3.0, 0.4);

        let jet = Transfer2::new()
            .evaluate_structural_second_with::<RealAxis, _, _, _>(
                &stack,
                &input,
                StructuralDerivativeVariable::Thickness(1),
            )
            .unwrap();

        let (_, _, analytic) = jet.into_parts();

        let plus = Transfer2::new()
            .evaluate_with::<RealAxis, _, _, _>(&two_layer_stack(d0, d1 + h), &input)
            .unwrap();

        let zero = Transfer2::new()
            .evaluate_with::<RealAxis, _, _, _>(&two_layer_stack(d0, d1), &input)
            .unwrap();

        let minus = Transfer2::new()
            .evaluate_with::<RealAxis, _, _, _>(&two_layer_stack(d0, d1 - h), &input)
            .unwrap();

        let expected = finite_difference_second(&plus, &zero, &minus, h);

        assert_matrix_close(&analytic, &expected, 3e-6);
    }

    #[test]
    fn linear_vacuum_wavenumber_derivative_applies_chain_rule() {
        let stack = two_layer_stack(0.15, 0.23);
        let input = make_input(3.0, 0.4);

        let squared = Transfer2::new()
            .evaluate_spectral_first_with::<RealAxis, _, _, _>(
                &stack,
                &input,
                SpectralDerivativeVariable::VacuumWavenumberSquared,
            )
            .unwrap();

        let linear = Transfer2::new()
            .evaluate_spectral_first_with::<RealAxis, _, _, _>(
                &stack,
                &input,
                SpectralDerivativeVariable::VacuumWavenumber,
            )
            .unwrap();

        let (_, squared_first) = squared.into_parts();
        let (_, linear_first) = linear.into_parts();

        let expected = &squared_first * c(2.0 * 3.0);

        assert_matrix_close(&linear_first, &expected, 1e-12);
    }

    #[test]
    fn invalid_thickness_index_returns_error() {
        let stack = two_layer_stack(0.15, 0.23);
        let input = make_input(3.0, 0.4);

        let error = Transfer2::new()
            .evaluate_structural_first_with::<RealAxis, _, _, _>(
                &stack,
                &input,
                StructuralDerivativeVariable::Thickness(2),
            )
            .unwrap_err();

        assert_eq!(
            error,
            Transfer2Error::ThicknessLayerOutOfBounds {
                requested: 2,
                layer_count: 2,
            }
        );
    }
}
