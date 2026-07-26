use ndarray::{Array, Dimension};
use thiserror::Error;
use tmm_units::LengthUnit;

use crate::stack::{Thickness, ValidationError};

use std::convert::Infallible;
use std::marker::PhantomData;

use crate::algebra::{Jet0, Jet1, Jet2};

use std::fmt::Debug;
use std::ops::Mul;

use num_traits::{Float, FromPrimitive};

use crate::input::canonical::{CanonicalLayer, CanonicalStack};
use crate::stack::{Stack, ValidationConfig};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum ThicknessSeedError {
    #[error(
        "requested thickness derivative for layer {index}, \
         but the stack contains {layer_count} layers"
    )]
    LayerOutOfBounds { index: usize, layer_count: usize },
}

#[derive(Debug, Error)]
pub enum StackCompileError<R, E> {
    #[error("invalid stack: {0}")]
    Validation(#[from] ValidationError<R>),

    #[error("failed to seed layer {index}: {source}")]
    Seed {
        index: usize,
        #[source]
        source: E,
    },
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct SeededThickness<J> {
    value: J,
    unit: LengthUnit,
}

impl<J> SeededThickness<J> {
    pub(crate) fn new(value: J, unit: LengthUnit) -> Self {
        Self { value, unit }
    }

    pub(crate) fn into_parts(self) -> (J, LengthUnit) {
        (self.value, self.unit)
    }
}

pub(crate) trait CompileThickness<R, D>
where
    D: Dimension,
{
    type Jet;
    type Error;

    /// Seed one caller-facing layer thickness.
    ///
    /// The returned jet is still expressed in the unit stored by `thickness`.
    fn seed(
        &self,
        layer_index: usize,
        thickness: Thickness<R>,
        shape_source: &Array<R, D>,
    ) -> Result<SeededThickness<Self::Jet>, Self::Error>;
}

pub(crate) fn compile_stack<M, R, D, P>(
    stack: &Stack<M, R>,
    shape_source: &Array<R, D>,
    validation: &ValidationConfig<R>,
    policy: &P,
) -> Result<CanonicalStack<M, P::Jet>, StackCompileError<R, P::Error>>
where
    M: Clone,
    R: Float + FromPrimitive + Copy + Debug,
    D: Dimension,
    P: CompileThickness<R, D>,
    P::Jet: Mul<R, Output = P::Jet>,
{
    stack.validate(validation)?;

    let layers = stack
        .iter()
        .enumerate()
        .map(|(index, layer)| {
            let seeded = policy
                .seed(index, layer.thickness(), shape_source)
                .map_err(|source| -> StackCompileError<R, P::Error> {
                    StackCompileError::Seed { index, source }
                })?;

            let (value, unit) = seeded.into_parts();

            let scale = unit.to_centimetres_factor::<R>();
            let thickness_cm = value * scale;

            Ok::<
                CanonicalLayer<M, <P as CompileThickness<R, D>>::Jet>,
                StackCompileError<R, P::Error>,
            >(CanonicalLayer::new(layer.material().clone(), thickness_cm))
        })
        .collect::<Result<Vec<_>, _>>()?;

    Ok(CanonicalStack::new(
        stack.left_exterior().clone(),
        stack.right_exterior().clone(),
        layers,
    ))
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct CompileStackValue;

impl<R, D> CompileThickness<R, D> for CompileStackValue
where
    R: Clone,
    D: Dimension,
{
    type Jet = Jet0<Array<R, D>>;
    type Error = Infallible;

    fn seed(
        &self,
        _layer_index: usize,
        thickness: Thickness<R>,
        shape_source: &Array<R, D>,
    ) -> Result<SeededThickness<Self::Jet>, Self::Error> {
        let (value, unit) = thickness.into_parts();

        let values = Array::from_elem(shape_source.raw_dim(), value);

        Ok(SeededThickness::new(Jet0::new(values), unit))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CompileLayerFirst {
    layer: usize,
}

impl CompileLayerFirst {
    pub(crate) fn new(layer: usize) -> Self {
        Self { layer }
    }

    pub(crate) fn validate(self, layer_count: usize) -> Result<Self, ThicknessSeedError> {
        if self.layer >= layer_count {
            return Err(ThicknessSeedError::LayerOutOfBounds {
                index: self.layer,
                layer_count,
            });
        }

        Ok(self)
    }
}

impl<R, D> CompileThickness<R, D> for CompileLayerFirst
where
    R: Clone,
    D: Dimension,
{
    type Jet = Jet1<Array<R, D>>;
    type Error = Infallible;

    fn seed(
        &self,
        layer_index: usize,
        thickness: Thickness<R>,
        shape_source: &Array<R, D>,
    ) -> Result<SeededThickness<Self::Jet>, Self::Error> {
        let (value, unit) = thickness.into_parts();

        let values = Array::from_elem(shape_source.raw_dim(), value);

        let jet = if layer_index == self.layer {
            Jet1::variable(values)
        } else {
            Jet1::constant(values)
        };

        Ok(SeededThickness::new(jet, unit))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CompileLayerSecond {
    layer: usize,
}

impl<R, D> CompileThickness<R, D> for CompileLayerSecond
where
    R: Clone,
    D: Dimension,
{
    type Jet = Jet2<Array<R, D>>;
    type Error = Infallible;

    fn seed(
        &self,
        layer_index: usize,
        thickness: Thickness<R>,
        shape_source: &Array<R, D>,
    ) -> Result<SeededThickness<Self::Jet>, Self::Error> {
        let (value, unit) = thickness.into_parts();

        let values = Array::from_elem(shape_source.raw_dim(), value);

        let jet = if layer_index == self.layer {
            Jet2::variable(values)
        } else {
            Jet2::constant(values)
        };

        Ok(SeededThickness::new(jet, unit))
    }
}
