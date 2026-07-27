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
use std::ops::Mul;

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
        compile::{
            assignment::ThicknessAssignment,
            seed::{SeedJet, UnsupportedDerivativeSlot},
        },
    },
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

pub(crate) fn compile_stack<M, C, D, J, A>(
    stack: &Stack<M, C::RealField>,
    sampled_shape: D,
    validation: &ValidationConfig<C::RealField>,
    assignment: &A,
) -> Result<CompiledStack<M, J, C::RealField>, StackCompileError<C::RealField>>
where
    M: Clone,
    C: ComplexField,
    C::RealField: Float + FromPrimitive + Copy + Debug,
    D: Dimension + Clone,
    J: StackThicknessJet<C, D>,
    A: ThicknessSlotMap + ?Sized,
{
    stack.validate(validation)?;

    let mut canonical_layers = Vec::with_capacity(stack.len());

    let mut caller_thicknesses = Vec::with_capacity(stack.len());

    for (layer_index, layer) in stack.layers_left_to_right().iter().enumerate() {
        let thickness = layer.thickness();

        caller_thicknesses.push(thickness);

        let (value, unit) = thickness.into_parts();

        let sampled_thickness: Array<C, D> =
            super::complexify(&Array::from_elem(sampled_shape.clone(), value));

        let thickness_jet = if let Some(slot) = assignment.slot_for_layer(layer_index) {
            J::variable(sampled_thickness, slot).map_err(|source| StackCompileError::Seed {
                layer: layer_index,
                source,
            })?
        } else {
            <J as SeedJet<Array<C, D>>>::constant(sampled_thickness)
        };

        let thickness_cm = thickness_jet.scale_real(unit.to_centimetres_factor::<C::RealField>());

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

pub(crate) trait StackThicknessJet<C, D>: SeedJet<Array<C, D>>
where
    C: ComplexField,
{
    fn scale_real(&self, factor: C::RealField) -> Self;
}

impl<C, D, P> StackThicknessJet<C, D> for ArrayJet0<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn scale_real(&self, factor: <C>::RealField) -> Self {
        ScalarAlgebra::scale(self, C::from_real(factor))
    }
}

impl<C, D, P> StackThicknessJet<C, D> for ArrayJet1<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn scale_real(&self, factor: <C>::RealField) -> Self {
        ScalarAlgebra::scale(self, C::from_real(factor))
    }
}

impl<C, D, P> StackThicknessJet<C, D> for ArrayJet2<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn scale_real(&self, factor: <C>::RealField) -> Self {
        ScalarAlgebra::scale(self, C::from_real(factor))
    }
}

impl<C, D, P> StackThicknessJet<C, D> for ArrayJetBivariate1<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn scale_real(&self, factor: <C>::RealField) -> Self {
        ScalarAlgebra::scale(self, C::from_real(factor))
    }
}

impl<C, D, P> StackThicknessJet<C, D> for ArrayJetBivariate2<C, D, P>
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
    use std::marker::PhantomData;

    use ndarray::{Array, Dimension, Ix0, Ix1, arr0};
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

    impl<V> SeedJet<V> for RecordingJet<V> {
        const VARIABLE_SLOTS: usize = 2;

        fn constant(value: V) -> Self {
            Self::constant(value)
        }

        fn variable(value: V, slot: usize) -> Result<Self, UnsupportedDerivativeSlot> {
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

    impl<D> StackThicknessJet<Complex64, D> for RecordingJet<Array<Complex64, D>>
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

    /// Simple mapping independent of ParameterAssignment.
    #[derive(Clone, Copy, Debug, Default)]
    struct TestSlotMap {
        assignments: &'static [(usize, usize)],
    }

    impl ThicknessSlotMap for TestSlotMap {
        fn slot_for_layer(&self, layer: usize) -> Option<usize> {
            self.assignments
                .iter()
                .find_map(|&(assigned_layer, slot)| (assigned_layer == layer).then_some(slot))
        }
    }

    fn no_assignments() -> TestSlotMap {
        TestSlotMap { assignments: &[] }
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

    #[test]
    fn preserves_exteriors_materials_and_layer_order() {
        let stack = stack_with_two_layers();

        let compiled = compile_stack::<_, Complex64, Ix0, RecordingJet<Array<Complex64, Ix0>>, _>(
            &stack,
            Ix0(),
            &validation(),
            &no_assignments(),
        )
        .unwrap();

        let (canonical, context) = compiled.into_parts();

        assert_eq!(canonical.left_exterior(), &"left exterior");
        assert_eq!(canonical.right_exterior(), &"right exterior");
        assert_eq!(canonical.layer_count(), 2);

        assert_eq!(canonical.layers()[0].material(), &"first material",);
        assert_eq!(canonical.layers()[1].material(), &"second material",);

        assert_eq!(context.layer_count(), 2);
    }

    #[test]
    fn converts_thicknesses_to_centimetres() {
        let stack = stack_with_two_layers();

        let compiled = compile_stack::<_, Complex64, Ix0, RecordingJet<Array<Complex64, Ix0>>, _>(
            &stack,
            Ix0(),
            &validation(),
            &no_assignments(),
        )
        .unwrap();

        let (canonical, _) = compiled.into_parts();

        let first = canonical.layers()[0].thickness_cm();
        let second = canonical.layers()[1].thickness_cm();

        assert_eq!(first.slot, None);
        assert_eq!(second.slot, None);

        // 500 nm = 5e-5 cm
        assert!((first.value[()] - Complex64::new(5.0e-5, 0.0)).norm() < 1.0e-14);

        // 2 µm = 2e-4 cm
        assert!((second.value[()] - Complex64::new(2.0e-4, 0.0)).norm() < 1.0e-14);
    }

    #[test]
    fn retains_caller_facing_thicknesses_and_units() {
        let stack = stack_with_two_layers();

        let expected = stack
            .layers_left_to_right()
            .iter()
            .map(|layer| layer.thickness())
            .collect::<Vec<_>>();

        let compiled = compile_stack::<_, Complex64, Ix0, RecordingJet<Array<Complex64, Ix0>>, _>(
            &stack,
            Ix0(),
            &validation(),
            &no_assignments(),
        )
        .unwrap();

        let (_, context) = compiled.into_parts();

        assert_eq!(context.layer_thicknesses(), expected.as_slice(),);
    }

    #[test]
    fn unassigned_layers_are_compiled_as_constants() {
        let stack = stack_with_two_layers();

        let compiled = compile_stack::<_, Complex64, Ix0, RecordingJet<Array<Complex64, Ix0>>, _>(
            &stack,
            Ix0(),
            &validation(),
            &no_assignments(),
        )
        .unwrap();

        let (canonical, _) = compiled.into_parts();

        assert_eq!(canonical.layers()[0].thickness_cm().slot, None,);
        assert_eq!(canonical.layers()[1].thickness_cm().slot, None,);
    }

    #[test]
    fn seeds_only_the_assigned_layer() {
        let stack = stack_with_two_layers();

        let assignment = TestSlotMap {
            assignments: &[(1, 0)],
        };

        let compiled = compile_stack::<_, Complex64, Ix0, RecordingJet<Array<Complex64, Ix0>>, _>(
            &stack,
            Ix0(),
            &validation(),
            &assignment,
        )
        .unwrap();

        let (canonical, _) = compiled.into_parts();

        assert_eq!(canonical.layers()[0].thickness_cm().slot, None,);
        assert_eq!(canonical.layers()[1].thickness_cm().slot, Some(0),);
    }

    #[test]
    fn preserves_assigned_derivative_slot() {
        let stack = stack_with_two_layers();

        let assignment = TestSlotMap {
            assignments: &[(0, 1)],
        };

        let compiled = compile_stack::<_, Complex64, Ix0, RecordingJet<Array<Complex64, Ix0>>, _>(
            &stack,
            Ix0(),
            &validation(),
            &assignment,
        )
        .unwrap();

        let (canonical, _) = compiled.into_parts();

        assert_eq!(canonical.layers()[0].thickness_cm().slot, Some(1),);
        assert_eq!(canonical.layers()[1].thickness_cm().slot, None,);
    }

    #[test]
    fn can_seed_multiple_layer_thicknesses() {
        let stack = stack_with_two_layers();

        let assignment = TestSlotMap {
            assignments: &[(0, 1), (1, 0)],
        };

        let compiled = compile_stack::<_, Complex64, Ix0, RecordingJet<Array<Complex64, Ix0>>, _>(
            &stack,
            Ix0(),
            &validation(),
            &assignment,
        )
        .unwrap();

        let (canonical, _) = compiled.into_parts();

        assert_eq!(canonical.layers()[0].thickness_cm().slot, Some(1),);
        assert_eq!(canonical.layers()[1].thickness_cm().slot, Some(0),);
    }

    #[test]
    fn creates_thickness_arrays_with_the_sampled_shape() {
        let stack = stack_with_two_layers();

        let compiled = compile_stack::<_, Complex64, Ix1, RecordingJet<Array<Complex64, Ix1>>, _>(
            &stack,
            ndarray::Ix1(4),
            &validation(),
            &no_assignments(),
        )
        .unwrap();

        let (canonical, _) = compiled.into_parts();

        assert_eq!(canonical.layers()[0].thickness_cm().value.shape(), &[4],);

        assert!(
            canonical.layers()[0]
                .thickness_cm()
                .value
                .iter()
                .all(|value| { (*value - Complex64::new(5.0e-5, 0.0)).norm() < 1.0e-14 })
        );
    }

    #[test]
    fn reports_layer_when_assigned_slot_is_unsupported() {
        let stack = stack_with_two_layers();

        let assignment = TestSlotMap {
            assignments: &[(1, 2)],
        };

        let error = compile_stack::<_, Complex64, Ix0, RecordingJet<Array<Complex64, Ix0>>, _>(
            &stack,
            Ix0(),
            &validation(),
            &assignment,
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
}
