use num_traits::{Float, FromPrimitive, Zero};
use std::fmt;
use tmm_units::LengthUnit;

use crate::spatial::Length;

/// A length which is guaranteed to be greater than or equal to zero
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Thickness<F>(Length<F>);

impl<F> Thickness<F> {
    fn new(value: F, length_unit: LengthUnit) -> Self {
        Self(Length::new(value, length_unit))
    }

    pub fn centimetres(value: F) -> Self {
        Self::new(value, LengthUnit::Centimetre)
    }

    pub fn millimetres(value: F) -> Self {
        Self::new(value, LengthUnit::Millimetre)
    }

    pub fn micrometres(value: F) -> Self {
        Self::new(value, LengthUnit::Micrometre)
    }

    pub fn nanometres(value: F) -> Self {
        Self::new(value, LengthUnit::Nanometre)
    }

    pub(crate) fn zero() -> Self
    where
        F: Zero,
    {
        Self::new(F::zero(), LengthUnit::Centimetre)
    }

    pub fn is_zero(&self) -> bool
    where
        F: Float,
    {
        self.0.value() == F::zero()
    }

    pub(crate) fn into_parts(self) -> (F, LengthUnit) {
        self.0.into_parts()
    }

    pub(crate) fn into_inner(self) -> Length<F> {
        self.0
    }

    pub(crate) fn as_cm(&self) -> F
    where
        F: Float + FromPrimitive,
    {
        self.0.as_cm()
    }
}

impl<F> fmt::Display for Thickness<F>
where
    F: Float + std::fmt::Debug + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0,)
    }
}
