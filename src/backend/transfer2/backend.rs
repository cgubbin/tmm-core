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
        evaluator::{
            ComplexPlane, ConstitutiveDerivativeEvaluator, ConstitutiveEvaluator, RealAxis,
        },
        field::InternalFieldRequest,
        isotropic::IsotropicLayerQuantities,
        jet::{ArrayJet, ArrayJetFirst},
        transfer2::{
            Transfer2Error,
            matrix::{Matrix2Entries, Transfer2Jet, Transfer2JetFirst, TransferMatrix2},
            workspace::TransferWorkspace,
        },
    },
    material::{EvaluateMaterial, EvaluateMeromorphicMaterial},
    stack::Stack,
};

/// Isotropic 2×2 transfer-matrix backend.
#[derive(Copy, Clone, Debug, Default)]
pub struct Transfer2;

impl Transfer2 {
    /// Construct a 2×2 transfer-matrix backend.
    pub const fn new() -> Self {
        Self
    }

    pub(crate) fn evaluate_real_axis<M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<TransferMatrix2<C, D>, Transfer2Error>
    where
        M: EvaluateMaterial<C, Real = C::RealField>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        let input = input.clone().to_complex();
        self.evaluate_with::<RealAxis, _, _, _>(stack, &input)
    }

    pub(crate) fn evaluate_complex_plane<M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Result<TransferMatrix2<C, D>, Transfer2Error>
    where
        M: EvaluateMeromorphicMaterial<C, Real = C::RealField>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        self.evaluate_with::<ComplexPlane, _, _, _>(stack, input)
    }

    pub(crate) fn evaluate_with<E, M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Result<TransferMatrix2<C, D>, Transfer2Error>
    where
        E: ConstitutiveEvaluator<C, D, M>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        let workspace =
            self.accumulate_with::<E, _, _, _>(stack, input, InternalFieldRequest::None)?;

        let total = workspace.into_total();

        Ok(total.into())
    }

    pub(crate) fn accumulate_real_axis<M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        request: InternalFieldRequest,
    ) -> Result<TransferWorkspace<ArrayBase<OwnedRepr<C>, D>>, Transfer2Error>
    where
        M: EvaluateMaterial<C, Real = C::RealField>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        let input = input.clone().to_complex();
        self.accumulate_with::<RealAxis, _, _, _>(stack, &input, request)
    }

    pub(crate) fn accumulate_complex_plane<M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        request: InternalFieldRequest,
    ) -> Result<TransferWorkspace<ArrayBase<OwnedRepr<C>, D>>, Transfer2Error>
    where
        M: EvaluateMeromorphicMaterial<C, Real = C::RealField>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        self.accumulate_with::<ComplexPlane, _, _, _>(stack, input, request)
    }

    /// Evaluate the native scattering matrix without derivatives.
    pub(crate) fn accumulate_with<E, M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        request: InternalFieldRequest,
    ) -> Result<TransferWorkspace<ArrayBase<OwnedRepr<C>, D>>, Transfer2Error>
    where
        E: ConstitutiveEvaluator<C, D, M>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        type Samples<C, D> = ArrayBase<OwnedRepr<C>, D>;

        let entries = Matrix2Entries::identity_like(input.vacuum_wavenumber());

        let mut workspace: TransferWorkspace<Samples<C, D>> = match request {
            InternalFieldRequest::None => TransferWorkspace::new(entries),
            InternalFieldRequest::LayerBoundaries => {
                TransferWorkspace::retaining_layers_with_capacity(entries, stack.len())
            }
        };

        for layer in stack.iter() {
            let quantities = IsotropicLayerQuantities::new::<E, _>(layer.material(), input);

            let thickness =
                constant_thickness_like(input.vacuum_wavenumber(), layer.thickness().as_cm());

            let layer_matrix = Matrix2Entries::from_layer::<C, D>(&quantities, thickness);

            workspace.append::<C, D>(layer_matrix, quantities);
        }

        Ok(workspace)
    }

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
        let workspace = self.accumulate_structural_first_with::<E, _, _, _>(
            stack,
            input,
            variable,
            InternalFieldRequest::None,
        )?;

        let total = workspace.into_total();

        Ok(total)
    }

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
        let workspace = self.accumulate_spectral_first_with::<E, _, _, _>(
            stack,
            input,
            variable,
            InternalFieldRequest::None,
        )?;

        let total = workspace.into_total();

        Ok(total)
    }

    pub(crate) fn accumulate_structural_first_with<E, M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
        request: InternalFieldRequest,
    ) -> Result<TransferWorkspace<ArrayJetFirst<C, D>>, Transfer2Error>
    where
        E: ConstitutiveEvaluator<C, D, M>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        validate_derivative_variable(stack, variable)?;

        let primitive = variable.primitive();

        let entries = Matrix2Entries::identity_like(input.vacuum_wavenumber());

        let mut workspace = match request {
            InternalFieldRequest::None => TransferWorkspace::new(entries),
            InternalFieldRequest::LayerBoundaries => {
                TransferWorkspace::retaining_layers_with_capacity(entries, stack.len())
            }
        };

        for (index, layer) in stack.iter().enumerate() {
            let quantities =
                IsotropicLayerQuantities::<ArrayJetFirst<C, D>>::evaluate_first_structural::<E, M>(
                    layer.material(),
                    input,
                    primitive,
                );

            let selected = matches!(
                variable,
                StructuralDerivativeVariable::Thickness(
                    requested
                ) if requested == index
            );

            let thickness = first_thickness(
                input.vacuum_wavenumber(),
                layer.thickness().as_cm(),
                selected,
            );

            let layer_matrix = Matrix2Entries::from_layer::<C, D>(&quantities, thickness);

            workspace.append::<C, D>(layer_matrix, quantities);
        }

        if let Some(rule) = variable.chain_rule(input) {
            workspace = workspace.chain_rule(&rule);
        }

        Ok(workspace)
    }

    pub(crate) fn accumulate_spectral_first_with<E, M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: SpectralDerivativeVariable,
        request: InternalFieldRequest,
    ) -> Result<TransferWorkspace<ArrayJetFirst<C, D>>, Transfer2Error>
    where
        E: ConstitutiveDerivativeEvaluator<C, D, M>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        let primitive = variable.primitive();

        let entries = Matrix2Entries::identity_like(input.vacuum_wavenumber());

        let mut workspace = match request {
            InternalFieldRequest::None => TransferWorkspace::new(entries),
            InternalFieldRequest::LayerBoundaries => {
                TransferWorkspace::retaining_layers_with_capacity(entries, stack.len())
            }
        };

        for layer in stack.iter() {
            let quantities =
                IsotropicLayerQuantities::<ArrayJetFirst<C, D>>::evaluate_first_spectral::<E, M>(
                    layer.material(),
                    input,
                    primitive,
                );

            let thickness = ArrayJetFirst::constant(constant_thickness_like(
                input.vacuum_wavenumber(),
                layer.thickness().as_cm(),
            ));

            let layer_matrix = Matrix2Entries::from_layer::<C, D>(&quantities, thickness);

            workspace.append::<C, D>(layer_matrix, quantities);
        }

        if let Some(rule) = variable.chain_rule(input) {
            workspace = workspace.chain_rule(&rule);
        }

        Ok(workspace)
    }

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
        let workspace = self.accumulate_structural_second_with::<E, _, _, _>(
            stack,
            input,
            variable,
            InternalFieldRequest::None,
        )?;

        let total = workspace.into_total();

        Ok(total)
    }

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
        let workspace = self.accumulate_spectral_second_with::<E, _, _, _>(
            stack,
            input,
            variable,
            InternalFieldRequest::None,
        )?;

        let total = workspace.into_total();

        Ok(total)
    }

    pub(crate) fn accumulate_structural_second_with<E, M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
        request: InternalFieldRequest,
    ) -> Result<TransferWorkspace<ArrayJet<C, D>>, Transfer2Error>
    where
        E: ConstitutiveEvaluator<C, D, M>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        validate_derivative_variable(stack, variable)?;

        let primitive = variable.primitive();

        let primitive = variable.primitive();

        let entries = Matrix2Entries::identity_like(input.vacuum_wavenumber());

        let mut workspace = match request {
            InternalFieldRequest::None => TransferWorkspace::new(entries),
            InternalFieldRequest::LayerBoundaries => {
                TransferWorkspace::retaining_layers_with_capacity(entries, stack.len())
            }
        };

        for (index, layer) in stack.iter().enumerate() {
            let quantities = IsotropicLayerQuantities::<ArrayJet<C, D>>::evaluate_second_structural::<
                E,
                M,
            >(layer.material(), input, primitive);

            let selected = matches!(
                variable,
                StructuralDerivativeVariable::Thickness(
                    requested
                ) if requested == index
            );

            let thickness = second_thickness(
                input.vacuum_wavenumber(),
                layer.thickness().as_cm(),
                selected,
            );

            let layer_matrix = Matrix2Entries::from_layer::<C, D>(&quantities, thickness);

            workspace.append::<C, D>(layer_matrix, quantities);
        }

        if let Some(rule) = variable.chain_rule(input) {
            workspace = workspace.chain_rule(&rule);
        }

        Ok(workspace)
    }

    pub(crate) fn accumulate_spectral_second_with<E, M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: SpectralDerivativeVariable,
        request: InternalFieldRequest,
    ) -> Result<TransferWorkspace<ArrayJet<C, D>>, Transfer2Error>
    where
        E: ConstitutiveDerivativeEvaluator<C, D, M>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        let primitive = variable.primitive();

        let entries = Matrix2Entries::identity_like(input.vacuum_wavenumber());

        let mut workspace = match request {
            InternalFieldRequest::None => TransferWorkspace::new(entries),
            InternalFieldRequest::LayerBoundaries => {
                TransferWorkspace::retaining_layers_with_capacity(entries, stack.len())
            }
        };

        for layer in stack.iter() {
            let quantities = IsotropicLayerQuantities::<ArrayJet<C, D>>::evaluate_second_spectral::<
                E,
                M,
            >(layer.material(), input, primitive);

            let thickness = ArrayJet::constant(constant_thickness_like(
                input.vacuum_wavenumber(),
                layer.thickness().as_cm(),
            ));

            let layer_matrix = Matrix2Entries::from_layer::<C, D>(&quantities, thickness);

            workspace.append::<C, D>(layer_matrix, quantities);
        }

        if let Some(rule) = variable.chain_rule(input) {
            workspace = workspace.chain_rule(&rule);
        }

        Ok(workspace)
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

fn constant_thickness_like<C, D>(
    source: &ArrayBase<OwnedRepr<C>, D>,
    thickness_cm: C::RealField,
) -> ArrayBase<OwnedRepr<C>, D>
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    source.mapv(|_| C::from_real(thickness_cm))
}

fn first_thickness<C, D>(
    like: &ArrayBase<OwnedRepr<C>, D>,
    thickness_cm: C::RealField,
    differentiate: bool,
) -> ArrayJetFirst<C, D>
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let value = like.mapv(|_| C::from_real(thickness_cm));

    let first = if differentiate {
        like.mapv(|_| C::one())
    } else {
        like.mapv(|_| C::zero())
    };

    ArrayJetFirst::from_parts(value, first)
}

fn second_thickness<C, D>(
    like: &ArrayBase<OwnedRepr<C>, D>,
    thickness_cm: C::RealField,
    differentiate: bool,
) -> ArrayJet<C, D>
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let value = like.mapv(|_| C::from_real(thickness_cm));

    let first = if differentiate {
        like.mapv(|_| C::one())
    } else {
        like.mapv(|_| C::zero())
    };

    let second = like.mapv(|_| C::zero());

    ArrayJet::from_parts(value, first, second)
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
        actual: &TransferMatrix2<C, ndarray::Ix0>,
        expected: &TransferMatrix2<C, ndarray::Ix0>,
        tolerance: f64,
    ) {
        assert_close(actual.m11()[()], expected.m11()[()], tolerance);
        assert_close(actual.m12()[()], expected.m12()[()], tolerance);
        assert_close(actual.m21()[()], expected.m21()[()], tolerance);
        assert_close(actual.m22()[()], expected.m22()[()], tolerance);
    }

    fn finite_difference_first(
        plus: &TransferMatrix2<C, ndarray::Ix0>,
        minus: &TransferMatrix2<C, ndarray::Ix0>,
        h: f64,
    ) -> TransferMatrix2<C, ndarray::Ix0> {
        &(plus - minus) * c(1.0 / (2.0 * h))
    }

    fn finite_difference_second(
        plus: &TransferMatrix2<C, ndarray::Ix0>,
        zero: &TransferMatrix2<C, ndarray::Ix0>,
        minus: &TransferMatrix2<C, ndarray::Ix0>,
        h: f64,
    ) -> TransferMatrix2<C, ndarray::Ix0> {
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

        let expected = TransferMatrix2::identity_like(input.vacuum_wavenumber());

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

        let (m11, m12, m21, m22) = jet.into_parts();
        let (_, m11) = m11.into_parts();
        let (_, m12) = m12.into_parts();
        let (_, m21) = m21.into_parts();
        let (_, m22) = m22.into_parts();
        let analytic = TransferMatrix2::new(m11, m12, m21, m22);

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

        let (m11, m12, m21, m22) = jet.into_parts();

        let (_, _, m11) = m11.into_parts();
        let (_, _, m12) = m12.into_parts();
        let (_, _, m21) = m21.into_parts();
        let (_, _, m22) = m22.into_parts();
        let analytic = TransferMatrix2::new(m11, m12, m21, m22);

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

        let (m11, m12, m21, m22) = squared.into_parts();
        let (_, m11_squared_first) = m11.into_parts();
        let (_, m12_squared_first) = m12.into_parts();
        let (_, m21_squared_first) = m21.into_parts();
        let (_, m22_squared_first) = m22.into_parts();

        let (m11, m12, m21, m22) = linear.into_parts();
        let (_, m11_linear_first) = m11.into_parts();
        let (_, m12_linear_first) = m12.into_parts();
        let (_, m21_linear_first) = m21.into_parts();
        let (_, m22_linear_first) = m22.into_parts();

        let m11_expected = m11_squared_first * c(2.0 * 3.0);
        let m12_expected = m12_squared_first * c(2.0 * 3.0);
        let m21_expected = m21_squared_first * c(2.0 * 3.0);
        let m22_expected = m22_squared_first * c(2.0 * 3.0);

        assert_close(m11_linear_first[()], m11_expected[()], 1e-12);
        assert_close(m12_linear_first[()], m12_expected[()], 1e-12);
        assert_close(m21_linear_first[()], m21_expected[()], 1e-12);
        assert_close(m22_linear_first[()], m22_expected[()], 1e-12);
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
