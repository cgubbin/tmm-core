use std::fmt::Debug;
use std::ops::Mul;

use nalgebra::ComplexField;
use ndarray::{Array, Dimension};
use num_traits::{Float, FromPrimitive};
use thiserror::Error;

use crate::{
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

    pub(crate) fn into_parts(self) -> (CanonicalStack<M, J>, StackContext<R>) {
        (self.canonical, self.context)
    }
}

#[derive(Debug, Error)]
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

pub(crate) fn compile_stack<M, C, D, J>(
    stack: &Stack<M, C::RealField>,
    sampled_shape: D,
    validation: &ValidationConfig<C::RealField>,
    assignment: ThicknessAssignment<'_>,
) -> Result<CompiledStack<M, J, C::RealField>, StackCompileError<C::RealField>>
where
    M: Clone,
    C: ComplexField,
    C::RealField: Float + FromPrimitive + Copy + Debug,
    D: Dimension + Clone,
    J: SeedJet<Array<C, D>> + StackJet<C, D>,
{
    stack.validate(validation)?;

    let mut canonical_layers = Vec::with_capacity(stack.len());

    let mut retained_thicknesses = Vec::with_capacity(stack.len());

    for (layer_index, layer) in stack.layers_left_to_right().iter().enumerate() {
        let thickness = layer.thickness();

        retained_thicknesses.push(thickness);

        let (value, unit) = thickness.into_parts();

        let sampled_value: Array<C, D> =
            super::complexify(&Array::from_elem(sampled_shape.clone(), value));

        let thickness_jet = match assignment.slot_for_layer(layer_index) {
            Some(slot) => {
                J::variable(sampled_value, slot).map_err(|source| StackCompileError::Seed {
                    layer: layer_index,
                    source,
                })?
            }

            None => J::constant(sampled_value),
        };

        let thickness_cm = thickness_jet.scale_real(unit.to_centimetres_factor::<C::RealField>());

        canonical_layers.push(CanonicalLayer::new(layer.material().clone(), thickness_cm));
    }

    let canonical = CanonicalStack::new(
        stack.left_exterior().clone(),
        stack.right_exterior().clone(),
        canonical_layers,
    );

    let context = StackContext::new(retained_thicknesses);

    Ok(CompiledStack::new(canonical, context))
}

use crate::algebra::ScalarAlgebra;

/// Operations required to convert caller-facing coordinates into canonical
/// plane-wave coordinates.
pub trait StackJet<C: ComplexField, D>: Sized + Clone {
    /// Multiply every coefficient by a real scalar.
    fn scale_real(self, factor: C::RealField) -> Self;
}

impl<C, D, M> StackJet<C, D> for M
where
    C: ComplexField,
    D: Dimension,
    M: ScalarAlgebra<C, D> + Sized + Clone,
{
    fn scale_real(self, factor: C::RealField) -> Self {
        ScalarAlgebra::scale(&self, C::from_real(factor))
    }
}
