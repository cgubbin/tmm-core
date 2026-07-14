use crate::stack::ValidationConfig;

use super::{
    Layer, Stack, Thickness,
    validation::{StackValidator, ValidationError},
};

use num_traits::Float;

pub struct StackBuilder<M, F> {
    left_exterior: M,
    right_exterior: M,
    layers_left_to_right: Vec<Layer<M, F>>,
    validation: ValidationConfig<F>,
}

impl<M, F: Float> StackBuilder<M, F> {
    pub fn new(left_exterior: M, right_exterior: M) -> Self {
        Self {
            left_exterior,
            right_exterior,
            layers_left_to_right: Vec::new(),
            validation: ValidationConfig::default(),
        }
    }

    pub fn with_layer(mut self, material: M, thickness: Thickness<F>) -> Self {
        self.layers_left_to_right
            .push(Layer::new(material, thickness));
        self
    }

    pub fn validation(mut self, validation: ValidationConfig<F>) -> Self {
        self.validation = validation;
        self
    }
}

impl<M, F> StackBuilder<M, F>
where
    F: Float + std::fmt::Debug + std::fmt::Display,
{
    pub fn build(self) -> Result<Stack<M, F>, ValidationError<F>> {
        let thicknesses: Vec<_> = self
            .layers_left_to_right
            .iter()
            .map(|l| l.thickness())
            .collect();
        let validator = StackValidator::new(self.validation);
        validator.validate_thicknesses(&thicknesses)?;

        Ok(Stack {
            left_exterior: self.left_exterior,
            right_exterior: self.right_exterior,
            layers_left_to_right: self.layers_left_to_right,
        })
    }
}
