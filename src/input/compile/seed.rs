use thiserror::Error;

use crate::algebra::{Jet0, Jet1, Jet2, JetBivariate1, JetBivariate2, JetOneLike, JetZeroLike};

/// Error returned when a derivative variable is assigned to a slot unsupported
/// by the selected jet algebra.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error(
    "derivative slot {slot} is unsupported by this jet algebra; \
     the algebra provides {available} slot(s)"
)]
pub struct UnsupportedDerivativeSlot {
    pub slot: usize,
    pub available: usize,
}

/// Construction of constant and independent-variable jets.
///
/// Slots are zero-indexed:
///
/// - `Jet0` supports no slots;
/// - `Jet1` and univariate `Jet` support slot `0`;
/// - `JetBivariate` supports slots `0` and `1`.
///
/// A future dynamically parameterised jet can implement the same trait without
/// changing coordinate or stack compilation.
pub trait SeedJet<V>: Sized {
    /// Number of independent derivative slots represented by this jet.
    const VARIABLE_SLOTS: usize;

    /// Construct a jet whose value is `value` and whose derivatives vanish.
    fn constant(value: V) -> Self;

    /// Construct a jet whose value is `value` and which is an independent
    /// variable in `slot`.
    fn variable(value: V, slot: usize) -> Result<Self, UnsupportedDerivativeSlot>;
}

impl<V> SeedJet<V> for Jet0<V>
where
    V: JetOneLike + JetZeroLike,
{
    const VARIABLE_SLOTS: usize = 0;

    fn constant(value: V) -> Self {
        Jet0::constant(value)
    }

    fn variable(_value: V, slot: usize) -> Result<Self, UnsupportedDerivativeSlot> {
        Err(UnsupportedDerivativeSlot {
            slot,
            available: Self::VARIABLE_SLOTS,
        })
    }
}

impl<V, P> SeedJet<V> for Jet1<V, P>
where
    V: JetOneLike + JetZeroLike,
{
    const VARIABLE_SLOTS: usize = 1;

    fn constant(value: V) -> Self {
        Jet1::constant(value)
    }

    fn variable(value: V, slot: usize) -> Result<Self, UnsupportedDerivativeSlot> {
        match slot {
            0 => Ok(Jet1::variable(value)),

            _ => Err(UnsupportedDerivativeSlot {
                slot,
                available: Self::VARIABLE_SLOTS,
            }),
        }
    }
}

impl<V, P> SeedJet<V> for Jet2<V, P>
where
    V: JetOneLike + JetZeroLike,
{
    const VARIABLE_SLOTS: usize = 1;

    fn constant(value: V) -> Self {
        Jet2::constant(value)
    }

    fn variable(value: V, slot: usize) -> Result<Self, UnsupportedDerivativeSlot> {
        match slot {
            0 => Ok(Jet2::variable(value)),

            _ => Err(UnsupportedDerivativeSlot {
                slot,
                available: Self::VARIABLE_SLOTS,
            }),
        }
    }
}

impl<V, P> SeedJet<V> for JetBivariate1<V, P>
where
    V: JetOneLike + JetZeroLike,
{
    const VARIABLE_SLOTS: usize = 2;

    fn constant(value: V) -> Self {
        JetBivariate1::constant(value)
    }

    fn variable(value: V, slot: usize) -> Result<Self, UnsupportedDerivativeSlot> {
        match slot {
            0 => Ok(JetBivariate1::variable_x(value)),
            1 => Ok(JetBivariate1::variable_y(value)),

            _ => Err(UnsupportedDerivativeSlot {
                slot,
                available: Self::VARIABLE_SLOTS,
            }),
        }
    }
}

impl<V, P> SeedJet<V> for JetBivariate2<V, P>
where
    V: JetOneLike + JetZeroLike,
{
    const VARIABLE_SLOTS: usize = 2;

    fn constant(value: V) -> Self {
        JetBivariate2::constant(value)
    }

    fn variable(value: V, slot: usize) -> Result<Self, UnsupportedDerivativeSlot> {
        match slot {
            0 => Ok(JetBivariate2::variable_x(value)),
            1 => Ok(JetBivariate2::variable_y(value)),

            _ => Err(UnsupportedDerivativeSlot {
                slot,
                available: Self::VARIABLE_SLOTS,
            }),
        }
    }
}
