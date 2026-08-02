//! Compilation of public stack descriptions.
//!
//! Stack compilation performs three tasks:
//!
//! - validates the public stack;
//! - converts layer thicknesses into canonical centimetres;
//! - seeds derivative jets according to the active parameter assignment.
//!
//! Material handles are copied directly into the canonical stack.
//! Only finite-layer thicknesses participate in derivative seeding.

use std::fmt::Debug;

use nalgebra::ComplexField;
use ndarray::{Array, Dimension};
use num_traits::{Float, FromPrimitive};
use thiserror::Error;

use crate::{
    algebra::{
        ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, ScalarAlgebra,
    },
    input::{
        canonical::{CanonicalLayer, CanonicalStack},
        compile::seed::{SeedJet, UnsupportedDerivativeSlot},
    },
    parameter::{DerivativeMapping, FiniteLayerIndex, Parameter},
    stack::{Stack, ValidationConfig, ValidationError},
};

use super::StackContext;

/// Result of validating and compiling a public stack.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CompiledStack<M, J, R> {
    canonical: CanonicalStack<M, J>,
    context: StackContext<R>,
}

impl<M, J, R> CompiledStack<M, J, R> {
    pub(crate) fn new(canonical: CanonicalStack<M, J>, context: StackContext<R>) -> Self {
        Self { canonical, context }
    }

    pub(crate) fn canonical(&self) -> &CanonicalStack<M, J> {
        &self.canonical
    }

    pub(crate) fn context(&self) -> &StackContext<R> {
        &self.context
    }

    pub(crate) fn into_parts(self) -> (CanonicalStack<M, J>, StackContext<R>) {
        (self.canonical, self.context)
    }
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum StackCompileError<R> {
    #[error("invalid stack: {0}")]
    Validation(#[from] ValidationError<R>),

    #[error("failed to seed thickness at layer {layer}: {source}")]
    Seed {
        layer: usize,

        #[source]
        source: UnsupportedDerivativeSlot,
    },
}

pub(crate) trait ThicknessSlotMap {
    fn slot_for_layer(&self, layer: usize) -> Option<usize>;
}

pub(crate) fn compile_stack<M, J>(
    stack: &Stack<M, <J::Scalar as ComplexField>::RealField>,
    sampled_shape: J::Dimension,
    validation: &ValidationConfig<<J::Scalar as ComplexField>::RealField>,
    mapping: &DerivativeMapping,
) -> Result<
    CompiledStack<M, J, <J::Scalar as ComplexField>::RealField>,
    StackCompileError<<J::Scalar as ComplexField>::RealField>,
>
where
    J: StackThicknessJet,
    J::Scalar: ComplexField,
    <J::Scalar as ComplexField>::RealField: Float + FromPrimitive + Copy + Debug,
    J::Dimension: Dimension + Clone,
    M: Clone,
{
    let assignment = ThicknessAssignment::new(mapping);

    stack.validate(validation)?;

    let mut canonical_layers = Vec::with_capacity(stack.len());

    let mut caller_thicknesses = Vec::with_capacity(stack.len());

    for (layer_index, layer) in stack.layers_left_to_right().iter().enumerate() {
        let thickness = layer.thickness();

        caller_thicknesses.push(thickness);

        let (value, unit) = thickness.into_parts();

        let sampled_thickness: Array<J::Scalar, J::Dimension> =
            super::complexify(&Array::from_elem(sampled_shape.clone(), value));

        let thickness_jet = if let Some(slot) = assignment.slot_for_layer(layer_index) {
            J::variable(sampled_thickness, slot).map_err(|source| StackCompileError::Seed {
                layer: layer_index,
                source,
            })?
        } else {
            <J as SeedJet>::constant(sampled_thickness)
        };

        let thickness_cm = thickness_jet
            .scale_real(unit.to_centimetres_factor::<<J::Scalar as ComplexField>::RealField>());

        canonical_layers.push(CanonicalLayer::new(layer.material().clone(), thickness_cm));
    }

    let canonical = CanonicalStack::new(
        stack.left_exterior().clone(),
        stack.right_exterior().clone(),
        canonical_layers,
    );

    let context = StackContext::new(caller_thicknesses);

    Ok(CompiledStack::new(canonical, context))
}

/// Layer-thickness-specific view over a parameter assignment.
#[derive(Clone, Copy, Debug)]
pub struct ThicknessAssignment<'a> {
    mapping: &'a DerivativeMapping,
}

impl<'a> ThicknessAssignment<'a> {
    pub(crate) const fn new(mapping: &'a DerivativeMapping) -> Self {
        Self { mapping }
    }

    fn slot_for_layer(&self, layer: usize) -> Option<usize> {
        self.mapping
            .slot_for(Parameter::LayerThickness(FiniteLayerIndex(layer)))
    }
}

#[doc(hidden)]
pub trait StackThicknessJet: SeedJet
where
    Self::Scalar: ComplexField,
{
    fn scale_real(&self, factor: <Self::Scalar as ComplexField>::RealField) -> Self;
}

impl<C, D, P> StackThicknessJet for ArrayJet0<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn scale_real(&self, factor: <C>::RealField) -> Self {
        ScalarAlgebra::scale(self, C::from_real(factor))
    }
}

impl<C, D, P> StackThicknessJet for ArrayJet1<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn scale_real(&self, factor: <C>::RealField) -> Self {
        ScalarAlgebra::scale(self, C::from_real(factor))
    }
}

impl<C, D, P> StackThicknessJet for ArrayJet2<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn scale_real(&self, factor: <C>::RealField) -> Self {
        ScalarAlgebra::scale(self, C::from_real(factor))
    }
}

impl<C, D, P> StackThicknessJet for ArrayJetBivariate1<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn scale_real(&self, factor: <C>::RealField) -> Self {
        ScalarAlgebra::scale(self, C::from_real(factor))
    }
}

impl<C, D, P> StackThicknessJet for ArrayJetBivariate2<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn scale_real(&self, factor: <C>::RealField) -> Self {
        ScalarAlgebra::scale(self, C::from_real(factor))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array, Dimension, Ix0, Ix1};
    use num_complex::Complex64;

    use super::*;
    use crate::{
        input::compile::seed::UnsupportedDerivativeSlot,
        stack::{Layer, Thickness},
    };

    /// Minimal jet used to observe how stack compilation seeds a thickness.
    #[derive(Clone, Debug, PartialEq)]
    struct RecordingJet<V> {
        value: V,
        slot: Option<usize>,
    }

    impl<V> RecordingJet<V> {
        fn constant(value: V) -> Self {
            Self { value, slot: None }
        }

        fn variable(value: V, slot: usize) -> Self {
            Self {
                value,
                slot: Some(slot),
            }
        }
    }

    impl<D> crate::algebra::Jet for RecordingJet<Array<Complex64, D>> {
        type Scalar = Complex64;
        type Dimension = D;
    }

    impl<D> SeedJet for RecordingJet<Array<Complex64, D>> {
        const VARIABLE_SLOTS: usize = 2;

        fn constant(value: Array<Complex64, D>) -> Self {
            Self::constant(value)
        }

        fn variable(
            value: Array<Complex64, D>,
            slot: usize,
        ) -> Result<Self, UnsupportedDerivativeSlot> {
            if slot < Self::VARIABLE_SLOTS {
                Ok(Self::variable(value, slot))
            } else {
                Err(UnsupportedDerivativeSlot {
                    slot,
                    available: Self::VARIABLE_SLOTS,
                })
            }
        }
    }

    impl<D> StackThicknessJet for RecordingJet<Array<Complex64, D>>
    where
        D: Dimension,
    {
        // Use the exact required methods from your ScalarAlgebra trait.
        //
        // The relevant implementation should multiply `self.value` by
        // `factor` while preserving `self.slot`.
        fn scale_real(&self, factor: f64) -> Self {
            Self {
                value: self.value.mapv(|value| value * Complex64::new(factor, 0.0)),
                slot: self.slot,
            }
        }
    }

    fn stack_with_two_layers() -> Stack<&'static str, f64> {
        Stack::new(
            "left exterior",
            vec![
                Layer::new("first material", Thickness::nanometres(500.0)),
                Layer::new("second material", Thickness::micrometres(2.0)),
            ],
            "right exterior",
        )
    }

    fn validation() -> ValidationConfig<f64> {
        ValidationConfig::default()
    }

    fn no_derivatives() -> DerivativeMapping {
        DerivativeMapping::none()
    }

    fn thickness_derivative(layer: usize) -> DerivativeMapping {
        DerivativeMapping::new([Parameter::LayerThickness(FiniteLayerIndex(layer))]).unwrap()
    }

    fn mixed_mapping() -> DerivativeMapping {
        DerivativeMapping::new([
            Parameter::Spectral,
            Parameter::LayerThickness(FiniteLayerIndex(1)),
        ])
        .unwrap()
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64, tolerance: f64) {
        let error = (actual - expected).norm();

        assert!(
            error <= tolerance,
            "expected {expected:?}, got {actual:?}; absolute error {error:e}",
        );
    }

    #[test]
    fn thickness_assignment_finds_matching_layer_slot() {
        let mapping = DerivativeMapping::new([
            Parameter::Spectral,
            Parameter::LayerThickness(FiniteLayerIndex(4)),
            Parameter::LayerThickness(FiniteLayerIndex(1)),
        ])
        .unwrap();

        let assignment = ThicknessAssignment::new(&mapping);

        assert_eq!(assignment.slot_for_layer(4), Some(1));
        assert_eq!(assignment.slot_for_layer(1), Some(2));
        assert_eq!(assignment.slot_for_layer(0), None);
        assert_eq!(assignment.slot_for_layer(2), None);
    }

    #[test]
    fn compile_stack_preserves_exterior_materials_and_layer_order() {
        let stack = stack_with_two_layers();

        let compiled = compile_stack::<_, RecordingJet<Array<Complex64, Ix0>>>(
            &stack,
            Ix0(),
            &validation(),
            &no_derivatives(),
        )
        .unwrap();

        let canonical = compiled.canonical();

        assert_eq!(canonical.left_exterior(), &"left exterior");
        assert_eq!(canonical.right_exterior(), &"right exterior");
        assert_eq!(canonical.layer_count(), 2);

        assert_eq!(canonical.layers()[0].material(), &"first material",);

        assert_eq!(canonical.layers()[1].material(), &"second material",);
    }

    #[test]
    fn compile_stack_converts_thicknesses_to_centimetres() {
        let stack = stack_with_two_layers();

        let compiled = compile_stack::<_, RecordingJet<Array<Complex64, Ix0>>>(
            &stack,
            Ix0(),
            &validation(),
            &no_derivatives(),
        )
        .unwrap();

        let layers = compiled.canonical().layers();

        // 500 nm = 5e-5 cm
        assert_complex_close(
            layers[0].thickness_cm().value[()],
            Complex64::new(5.0e-5, 0.0),
            1.0e-15,
        );

        // 2 µm = 2e-4 cm
        assert_complex_close(
            layers[1].thickness_cm().value[()],
            Complex64::new(2.0e-4, 0.0),
            1.0e-15,
        );
    }

    #[test]
    fn compile_stack_compiles_unmapped_thicknesses_as_constants() {
        let stack = stack_with_two_layers();

        let compiled = compile_stack::<_, RecordingJet<Array<Complex64, Ix0>>>(
            &stack,
            Ix0(),
            &validation(),
            &no_derivatives(),
        )
        .unwrap();

        let layers = compiled.canonical().layers();

        assert_eq!(layers[0].thickness_cm().slot, None);
        assert_eq!(layers[1].thickness_cm().slot, None);
    }

    #[test]
    fn compile_stack_seeds_requested_layer_in_mapped_slot() {
        let stack = stack_with_two_layers();
        let mapping = thickness_derivative(1);

        let compiled = compile_stack::<_, RecordingJet<Array<Complex64, Ix0>>>(
            &stack,
            Ix0(),
            &validation(),
            &mapping,
        )
        .unwrap();

        let layers = compiled.canonical().layers();

        assert_eq!(layers[0].thickness_cm().slot, None);
        assert_eq!(layers[1].thickness_cm().slot, Some(0));
    }

    #[test]
    fn compile_stack_uses_mapping_slot_not_layer_index() {
        let stack = stack_with_two_layers();
        let mapping = mixed_mapping();

        let compiled = compile_stack::<_, RecordingJet<Array<Complex64, Ix0>>>(
            &stack,
            Ix0(),
            &validation(),
            &mapping,
        )
        .unwrap();

        let layers = compiled.canonical().layers();

        // Slot zero belongs to the spectral parameter. Layer one therefore
        // occupies slot one, even though its geometric layer index is also one.
        assert_eq!(layers[0].thickness_cm().slot, None);
        assert_eq!(layers[1].thickness_cm().slot, Some(1));
    }

    #[test]
    fn compile_stack_preserves_sampled_shape() {
        let stack = stack_with_two_layers();

        let compiled = compile_stack::<_, RecordingJet<Array<Complex64, Ix1>>>(
            &stack,
            Ix1(3),
            &validation(),
            &no_derivatives(),
        )
        .unwrap();

        for layer in compiled.canonical().layers() {
            assert_eq!(layer.thickness_cm().value.raw_dim(), Ix1(3));
        }

        let first = &compiled.canonical().layers()[0].thickness_cm().value;

        assert!(
            first
                .iter()
                .all(|&value| { (value - Complex64::new(5.0e-5, 0.0)).norm() <= 1.0e-15 })
        );
    }

    #[test]
    fn compile_stack_preserves_mapping_slot_across_sampled_shape() {
        let stack = stack_with_two_layers();
        let mapping = mixed_mapping();

        let compiled = compile_stack::<_, RecordingJet<Array<Complex64, Ix1>>>(
            &stack,
            Ix1(4),
            &validation(),
            &mapping,
        )
        .unwrap();

        let layers = compiled.canonical().layers();

        assert_eq!(layers[0].thickness_cm().slot, None);
        assert_eq!(layers[1].thickness_cm().slot, Some(1));
        assert_eq!(layers[1].thickness_cm().value.raw_dim(), Ix1(4),);
    }

    #[test]
    fn compile_stack_retains_caller_facing_thicknesses_in_context() {
        let stack = stack_with_two_layers();

        let compiled = compile_stack::<_, RecordingJet<Array<Complex64, Ix0>>>(
            &stack,
            Ix0(),
            &validation(),
            &no_derivatives(),
        )
        .unwrap();

        let context = compiled.context();

        assert_eq!(context.layer_count(), 2);

        assert_eq!(
            context.layer_thickness(0),
            Some(&Thickness::nanometres(500.0)),
        );

        assert_eq!(
            context.layer_thickness(1),
            Some(&Thickness::micrometres(2.0)),
        );

        assert_eq!(context.layer_thickness(2), None);
    }

    #[test]
    fn into_parts_returns_canonical_stack_and_context() {
        let stack = stack_with_two_layers();

        let compiled = compile_stack::<_, RecordingJet<Array<Complex64, Ix0>>>(
            &stack,
            Ix0(),
            &validation(),
            &no_derivatives(),
        )
        .unwrap();

        let (canonical, context) = compiled.into_parts();

        assert_eq!(canonical.layer_count(), 2);
        assert_eq!(context.layer_count(), 2);
        assert_eq!(canonical.left_exterior(), &"left exterior");
        assert_eq!(canonical.right_exterior(), &"right exterior");
    }

    #[test]
    fn compile_stack_reports_unsupported_derivative_slot() {
        let stack = stack_with_two_layers();

        let mapping = DerivativeMapping::new([
            Parameter::Spectral,
            Parameter::InPlane,
            Parameter::LayerThickness(FiniteLayerIndex(1)),
        ])
        .unwrap();

        let error = compile_stack::<_, RecordingJet<Array<Complex64, Ix0>>>(
            &stack,
            Ix0(),
            &validation(),
            &mapping,
        )
        .unwrap_err();

        assert_eq!(
            error,
            StackCompileError::Seed {
                layer: 1,
                source: UnsupportedDerivativeSlot {
                    slot: 2,
                    available: 2,
                },
            },
        );
    }

    #[test]
    fn compile_stack_forwards_stack_validation_errors() {
        let stack = Stack::new(
            "left",
            vec![Layer::new("invalid", Thickness::nanometres(-1.0))],
            "right",
        );

        let error = compile_stack::<_, RecordingJet<Array<Complex64, Ix0>>>(
            &stack,
            Ix0(),
            &ValidationConfig::strict(),
            &no_derivatives(),
        )
        .unwrap_err();

        assert!(matches!(error, StackCompileError::Validation(_)));
    }
}
