use num_traits::Float;
use std::fmt;

use super::units::{self, UnitError};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Thickness<F> {
    cm: F,
}

impl<F> Thickness<F>
where
    F: Copy,
{
    pub fn as_cm(&self) -> F {
        self.cm
    }
}

impl<F> Thickness<F>
where
    F: Float + std::fmt::Debug,
{
    pub fn from_cm(cm: F) -> Result<Self, UnitError<F>> {
        if !cm.is_finite() || cm < F::zero() {
            return Err(UnitError::InvalidLength { value: cm });
        }

        Ok(Self { cm })
    }

    pub fn from_nm(nm: F) -> Result<Self, UnitError<F>> {
        Ok(Self {
            cm: units::nm_to_cm(nm)?,
        })
    }

    pub fn from_um(um: F) -> Result<Self, UnitError<F>> {
        Ok(Self {
            cm: units::um_to_cm(um)?,
        })
    }

    pub fn from_mm(mm: F) -> Result<Self, UnitError<F>> {
        Ok(Self {
            cm: units::mm_to_cm(mm)?,
        })
    }

    pub fn zero() -> Self {
        Self { cm: F::zero() }
    }

    pub fn is_zero(&self) -> bool {
        self.cm == F::zero()
    }
}

impl<F> fmt::Display for Thickness<F>
where
    F: Float + std::fmt::Debug + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let nm = self.cm / F::from(units::constants::NM_TO_CM).unwrap();
        write!(f, "{nm} nm")
    }
}

impl<F> std::ops::Add for Thickness<F>
where
    F: Float,
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self {
        Self {
            cm: self.cm + rhs.cm,
        }
    }
}
