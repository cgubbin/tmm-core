use num_traits::Float;
use thiserror::Error;

use super::thickness::Thickness;

#[derive(Debug, Clone, PartialEq, Error)]
pub enum ValidationError<F> {
    #[error("stack must contain at least one internal layer")]
    NoLayers,

    #[error("layer count {count} exceeds maximum allowed {max}")]
    TooManyLayers { count: usize, max: usize },

    #[error("zero thickness at layer {index}")]
    ZeroThickness { index: usize },

    #[error("thickness at layer {index} is below minimum {min:?}: {actual:?}")]
    ThicknessTooSmall {
        index: usize,
        actual: Thickness<F>,
        min: Thickness<F>,
    },

    #[error("thickness at layer {index} exceeds maximum {max:?}: {actual:?}")]
    ThicknessTooLarge {
        index: usize,
        actual: Thickness<F>,
        max: Thickness<F>,
    },

    #[error("total thickness exceeds maximum {max:?}: {actual:?}")]
    TotalThicknessTooLarge {
        actual: Thickness<F>,
        max: Thickness<F>,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationConfig<F> {
    pub allow_empty: bool,
    pub allow_zero_thickness: bool,
    pub min_thickness: Option<Thickness<F>>,
    pub max_thickness: Option<Thickness<F>>,
    pub max_total_thickness: Option<Thickness<F>>,
    pub max_layer_count: Option<usize>,
}

impl<F> ValidationConfig<F> {
    pub fn permissive() -> Self {
        Self {
            allow_empty: true,
            allow_zero_thickness: true,
            min_thickness: None,
            max_thickness: None,
            max_total_thickness: None,
            max_layer_count: None,
        }
    }

    pub fn strict() -> Self
    where
        F: Float,
    {
        Self {
            allow_empty: false,
            allow_zero_thickness: false,
            min_thickness: None,
            max_thickness: None,
            max_total_thickness: None,
            max_layer_count: Some(10_000),
        }
    }
}

impl<F> Default for ValidationConfig<F>
where
    F: Float,
{
    fn default() -> Self {
        Self::strict()
    }
}

#[derive(Debug, Clone)]
pub struct StackValidator<F> {
    config: ValidationConfig<F>,
}

impl<F> StackValidator<F> {
    pub fn new(config: ValidationConfig<F>) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &ValidationConfig<F> {
        &self.config
    }
}

impl<F> Default for StackValidator<F>
where
    F: Float,
{
    fn default() -> Self {
        Self::new(ValidationConfig::default())
    }
}

impl<F> StackValidator<F>
where
    F: Float + Copy + std::fmt::Debug,
{
    pub fn validate_thicknesses(
        &self,
        thicknesses: &[Thickness<F>],
    ) -> Result<(), ValidationError<F>> {
        if !self.config.allow_empty && thicknesses.is_empty() {
            return Err(ValidationError::NoLayers);
        }

        if let Some(max) = self.config.max_layer_count {
            if thicknesses.len() > max {
                return Err(ValidationError::TooManyLayers {
                    count: thicknesses.len(),
                    max,
                });
            }
        }

        let mut total = Thickness::zero();

        for (index, thickness) in thicknesses.iter().copied().enumerate() {
            if !self.config.allow_zero_thickness && thickness.is_zero() {
                return Err(ValidationError::ZeroThickness { index });
            }

            if let Some(min) = self.config.min_thickness {
                if thickness < min {
                    return Err(ValidationError::ThicknessTooSmall {
                        index,
                        actual: thickness,
                        min,
                    });
                }
            }

            if let Some(max) = self.config.max_thickness {
                if thickness > max {
                    return Err(ValidationError::ThicknessTooLarge {
                        index,
                        actual: thickness,
                        max,
                    });
                }
            }

            total = total + thickness;
        }

        if let Some(max) = self.config.max_total_thickness {
            if total > max {
                return Err(ValidationError::TotalThicknessTooLarge { actual: total, max });
            }
        }

        Ok(())
    }
}
