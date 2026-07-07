use num_traits::Float;

pub mod constants {
    pub const NM_TO_CM: f64 = 1e-7;
    pub const UM_TO_CM: f64 = 1e-4;
    pub const MM_TO_CM: f64 = 1e-1;
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum UnitError<F> {
    #[error("failed to convert constant {constant} to target floating-point type")]
    FloatConversion { constant: &'static str },

    #[error("invalid length value: {value:?}")]
    InvalidLength { value: F },
}

fn checked_length<F>(value: F) -> Result<F, UnitError<F>>
where
    F: Float,
{
    if !value.is_finite() || value < F::zero() {
        return Err(UnitError::InvalidLength { value });
    }

    Ok(value)
}

pub fn nm_to_cm<F>(nm: F) -> Result<F, UnitError<F>>
where
    F: Float,
{
    let factor = F::from(constants::NM_TO_CM).ok_or(UnitError::FloatConversion {
        constant: "nm→cm",
    })?;

    Ok(checked_length(nm)? * factor)
}

pub fn um_to_cm<F>(um: F) -> Result<F, UnitError<F>>
where
    F: Float,
{
    let factor = F::from(constants::UM_TO_CM).ok_or(UnitError::FloatConversion {
        constant: "µm→cm",
    })?;

    Ok(checked_length(um)? * factor)
}

pub fn mm_to_cm<F>(mm: F) -> Result<F, UnitError<F>>
where
    F: Float,
{
    let factor = F::from(constants::MM_TO_CM).ok_or(UnitError::FloatConversion {
        constant: "mm→cm",
    })?;

    Ok(checked_length(mm)? * factor)
}
