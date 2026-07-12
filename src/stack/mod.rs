mod builder;
mod layer;
mod thickness;
mod units;
mod validation;

pub(crate) use builder::StackBuilder;
pub(crate) use layer::Layer;
pub(crate) use thickness::Thickness;
pub(crate) use validation::ValidationConfig;

use either::Either;
use num_traits::Float;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PropagationDirection {
    Forward,
    Reverse,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Stack<M, F> {
    incident: M,
    substrate: M,
    layers: Vec<Layer<M, F>>,
    direction: PropagationDirection,
}

impl<M, F> Stack<M, F> {
    pub fn builder(incident: M, substrate: M) -> StackBuilder<M, F>
    where
        F: Float,
    {
        StackBuilder::new(incident, substrate)
    }

    pub fn incident(&self) -> &M {
        &self.incident
    }

    pub fn substrate(&self) -> &M {
        &self.substrate
    }

    pub fn layers(&self) -> &[Layer<M, F>] {
        &self.layers
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn layers_in_propagation_order(&self) -> impl DoubleEndedIterator<Item = &Layer<M, F>> {
        match self.direction {
            PropagationDirection::Forward => Either::Left(self.layers.iter()),
            PropagationDirection::Reverse => Either::Right(self.layers.iter().rev()),
        }
    }
}
