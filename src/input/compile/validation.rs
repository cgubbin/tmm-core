use lamina_units::Length;
use num_traits::{Float, FromPrimitive};
use thiserror::Error;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ValidationError<F> {
    #[error("stack must contain at least one internal layer")]
    NoLayers,

    #[error("layer count {count} exceeds maximum allowed {max}")]
    TooManyLayers { count: usize, max: usize },

    #[error("zero thickness at layer {index}")]
    ZeroThickness { index: usize },

    #[error("negative thickness at layer {index}")]
    NegativeThickness { index: usize },

    #[error("thickness at layer {index} is below minimum {min:?}: {actual:?}")]
    ThicknessTooSmall {
        index: usize,
        actual: Length<F>,
        min: Length<F>,
    },

    #[error("thickness at layer {index} exceeds maximum {max:?}: {actual:?}")]
    ThicknessTooLarge {
        index: usize,
        actual: Length<F>,
        max: Length<F>,
    },

    #[error("total thickness exceeds maximum {max:?}: {actual:?}")]
    TotalThicknessTooLarge { actual: Length<F>, max: Length<F> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct ValidationConfig<F> {
    pub allow_empty: bool,
    pub allow_zero_thickness: bool,
    pub min_thickness: Option<Length<F>>,
    pub max_thickness: Option<Length<F>>,
    pub max_total_thickness: Option<Length<F>>,
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

impl<F> ValidationConfig<F>
where
    F: Float + FromPrimitive + Copy + std::fmt::Debug,
{
    pub fn validate_thicknesses(
        &self,
        thicknesses: &[Length<F>],
    ) -> Result<(), ValidationError<F>> {
        if !self.allow_empty && thicknesses.is_empty() {
            return Err(ValidationError::NoLayers);
        }

        if let Some(max) = self.max_layer_count {
            if thicknesses.len() > max {
                return Err(ValidationError::TooManyLayers {
                    count: thicknesses.len(),
                    max,
                });
            }
        }

        let mut total_cm = F::zero();

        for (index, thickness) in thicknesses.iter().copied().enumerate() {
            if !self.allow_zero_thickness && thickness.is_zero() {
                return Err(ValidationError::ZeroThickness { index });
            }

            let (value, unit) = thickness.into_parts();
            let thickness_cm = value * unit.to_centimetres_factor();

            if value < F::zero() {
                return Err(ValidationError::NegativeThickness { index });
            }

            if let Some(min) = self.min_thickness {
                let (value, unit) = min.into_parts();
                let min_thickness_cm = value * unit.to_centimetres_factor();
                if thickness_cm < min_thickness_cm {
                    return Err(ValidationError::ThicknessTooSmall {
                        index,
                        actual: thickness,
                        min,
                    });
                }
            }

            if let Some(max) = self.max_thickness {
                let (value, unit) = max.into_parts();
                let max_thickness_cm = value * unit.to_centimetres_factor();
                if thickness_cm > max_thickness_cm {
                    return Err(ValidationError::ThicknessTooLarge {
                        index,
                        actual: thickness,
                        max,
                    });
                }
            }

            total_cm = total_cm + thickness_cm;
        }

        if let Some(max) = self.max_total_thickness {
            let (value, unit) = max.into_parts();
            let max_thickness_cm = value * unit.to_centimetres_factor();
            if total_cm > max_thickness_cm {
                return Err(ValidationError::TotalThicknessTooLarge {
                    actual: Length::centimetres(total_cm),
                    max,
                });
            }
        }

        Ok(())
    }
}
