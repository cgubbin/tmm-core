use num_traits::{Float, Zero};
use std::fmt;
use tmm_units::{LengthUnit, UnitLabel};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Thickness<F> {
    value: F,
    length_unit: LengthUnit,
}

impl<F> Thickness<F> {
    fn new(value: F, length_unit: LengthUnit) -> Self {
        Self { value, length_unit }
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
        Self {
            value: F::zero(),
            length_unit: LengthUnit::Centimetre,
        }
    }

    pub fn is_zero(&self) -> bool
    where
        F: Float,
    {
        self.value == F::zero()
    }

    pub(crate) fn into_parts(self) -> (F, LengthUnit) {
        (self.value, self.length_unit)
    }
}

impl<F> fmt::Display for Thickness<F>
where
    F: Float + std::fmt::Debug + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.value, self.length_unit.symbol())
    }
}
