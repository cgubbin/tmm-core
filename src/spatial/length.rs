use lamina_units::{LengthUnit, UnitLabel};
use num_traits::{Float, FromPrimitive, Zero};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Length<F> {
    value: F,
    unit: LengthUnit,
}

impl<F> Length<F> {
    pub(crate) fn new(value: F, unit: LengthUnit) -> Self {
        Self { value, unit }
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

    pub fn value(&self) -> F
    where
        F: Copy,
    {
        self.value
    }

    pub fn unit(&self) -> LengthUnit {
        self.unit
    }

    pub(crate) fn zero() -> Self
    where
        F: Zero,
    {
        Self {
            value: F::zero(),
            unit: LengthUnit::Centimetre,
        }
    }

    pub(super) fn half(self) -> Self
    where
        F: Float,
    {
        let (value, unit) = self.into_parts();

        Self::new(value / (F::one() + F::one()), unit)
    }

    pub(super) fn scale_by(self, scalar: F) -> Self
    where
        F: Float,
    {
        let (value, unit) = self.into_parts();

        Self::new(value * scalar, unit)
    }

    pub fn is_zero(&self) -> bool
    where
        F: Float,
    {
        self.value == F::zero()
    }

    pub(crate) fn into_parts(self) -> (F, LengthUnit) {
        (self.value, self.unit)
    }

    pub(crate) fn as_cm(&self) -> F
    where
        F: Float + FromPrimitive,
    {
        self.unit.scale_to_centimetres::<F>() * self.value
    }

    pub(crate) fn into_canonical(self) -> F
    where
        F: Float + FromPrimitive,
    {
        self.unit.scale_to_centimetres::<F>() * self.value
    }
}

impl<F> fmt::Display for Length<F>
where
    F: Float + std::fmt::Debug + fmt::Display,
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", self.value, self.unit.symbol())
    }
}
