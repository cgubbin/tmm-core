use crate::stack::ValidationConfig;

use super::{
    Layer, PropagationDirection, Stack, Thickness,
    validation::{StackValidator, ValidationError},
};

use num_traits::Float;

pub struct StackBuilder<M, F> {
    incident: M,
    substrate: M,
    layers: Vec<Layer<M, F>>,
    direction: PropagationDirection,
    validation: ValidationConfig<F>,
}

impl<M, F: Float> StackBuilder<M, F> {
    pub fn new(incident: M, substrate: M) -> Self {
        Self {
            incident,
            substrate,
            layers: Vec::new(),
            direction: PropagationDirection::Forward,
            validation: ValidationConfig::default(),
        }
    }

    pub fn with_layer(mut self, material: M, thickness: Thickness<F>) -> Self {
        self.layers.push(Layer::new(material, thickness));
        self
    }

    pub fn direction(mut self, direction: PropagationDirection) -> Self {
        self.direction = direction;
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
        let thicknesses: Vec<_> = self.layers.iter().map(|l| l.thickness()).collect();
        let validator = StackValidator::new(self.validation);
        validator.validate_thicknesses(&thicknesses)?;

        Ok(Stack {
            incident: self.incident,
            substrate: self.substrate,
            layers: self.layers,
            direction: self.direction,
        })
    }
}
