mod builder;
mod layer;
mod thickness;
mod units;
mod validation;

pub use builder::StackBuilder;
pub use layer::Layer;
pub use thickness::Thickness;
pub use validation::ValidationConfig;

use either::Either;
use num_traits::Float;

use crate::IncidentSide;

#[derive(Clone, Debug, PartialEq)]
pub struct Stack<M, F> {
    left_exterior: M,
    right_exterior: M,
    layers_left_to_right: Vec<Layer<M, F>>,
}

enum PropagationDirection {
    LeftToRight,
    RightToLeft,
}

impl IncidentSide {
    pub(crate) fn propagation_direction(self) -> PropagationDirection {
        match self {
            Self::Left => PropagationDirection::LeftToRight,

            Self::Right => PropagationDirection::RightToLeft,
        }
    }
}

impl<M, F> Stack<M, F> {
    pub fn builder(left_exterior: M, right_exterior: M) -> StackBuilder<M, F>
    where
        F: Float,
    {
        StackBuilder::new(left_exterior, right_exterior)
    }

    pub fn left_exterior(&self) -> &M {
        &self.left_exterior
    }

    pub fn right_exterior(&self) -> &M {
        &self.right_exterior
    }

    pub fn layers_left_to_right(&self) -> &[Layer<M, F>] {
        &self.layers_left_to_right
    }

    pub fn len(&self) -> usize {
        self.layers_left_to_right.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &Layer<M, F>> {
        self.layers_left_to_right.iter()
    }

    /// Finite layers in the requested geometric direction.
    pub fn layers_in_direction(
        &self,
        direction: PropagationDirection,
    ) -> impl DoubleEndedIterator<Item = &Layer<M, F>> {
        match direction {
            PropagationDirection::LeftToRight => Either::Left(self.layers_left_to_right.iter()),

            PropagationDirection::RightToLeft => {
                Either::Right(self.layers_left_to_right.iter().rev())
            }
        }
    }

    /// Exterior encountered first in the requested direction.
    pub fn entrance_exterior(&self, direction: PropagationDirection) -> &M {
        match direction {
            PropagationDirection::LeftToRight => self.left_exterior(),

            PropagationDirection::RightToLeft => self.right_exterior(),
        }
    }

    /// Exterior encountered last in the requested direction.
    pub fn exit_exterior(&self, direction: PropagationDirection) -> &M {
        match direction {
            PropagationDirection::LeftToRight => self.right_exterior(),

            PropagationDirection::RightToLeft => self.left_exterior(),
        }
    }
}
