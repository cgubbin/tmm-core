//! Input types for the 2×2 transfer-matrix backend.
//!
//! `Transfer2Input` stores the physical sampled inputs for a backend
//! evaluation. Derivative variables are intentionally not stored in the input;
//! they are supplied to derivative solver methods instead.

use crate::ComplexScalar;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Polarisation {
    TransverseElectric,
    TransverseMagnetic,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DerivativeVariable {
    Frequency,
    FrequencySquared,
    PropagationConstant,
    PropagationConstantSquared,
    Thickness(usize),
}

impl DerivativeVariable {
    pub fn primitive(self) -> Self {
        match self {
            Self::Frequency => Self::FrequencySquared,
            Self::PropagationConstant => Self::PropagationConstantSquared,
            x => x,
        }
    }

    pub fn is_primitive(self) -> bool {
        self == self.primitive()
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct ChainRule<C> {
    pub(crate) first: C,
    pub(crate) second: C,
}

impl DerivativeVariable {
    pub(crate) fn chain_rule<C: ComplexScalar>(
        self,
        wavenumber: C,
        propagation_constant: C,
    ) -> Option<ChainRule<C>> {
        let two = C::one() + C::one();

        match self {
            Self::Frequency => Some(ChainRule {
                first: two * wavenumber,
                second: two,
            }),

            Self::PropagationConstant => Some(ChainRule {
                first: two * propagation_constant,
                second: two,
            }),

            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Transfer2Input<I> {
    pub wavenumber: I,
    pub propagation_constant_squared: I,
    pub polarisation: Polarisation,
}

impl<I> Transfer2Input<I> {
    pub fn new(wavenumber: I, propagation_constant_squared: I, polarisation: Polarisation) -> Self {
        Self {
            wavenumber,
            propagation_constant_squared,
            polarisation,
        }
    }
}
