//! Construction of constant and independently seeded jets.
//!
//! Caller-facing parameters are assigned to numbered derivative slots by
//! [`ParameterAssignment`](super::assignment::ParameterAssignment). This
//! module maps those slot numbers onto the constructors supplied by each
//! concrete jet algebra.
//!
//! Slot numbers have no intrinsic physical meaning. For example, slot zero
//! represents the spectral coordinate only when the active parameter
//! assignment places that coordinate in slot zero.

use thiserror::Error;

use crate::algebra::{Jet0, Jet1, Jet2, JetBivariate1, JetBivariate2, JetOneLike, JetZeroLike};

/// A requested derivative slot is not represented by the selected jet algebra.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error(
    "derivative slot {slot} is unsupported by this jet algebra; \
     the algebra provides {available} slot(s)"
)]
pub(crate) struct UnsupportedDerivativeSlot {
    /// Requested zero-based derivative slot.
    pub(crate) slot: usize,

    /// Number of slots represented by the selected algebra.
    pub(crate) available: usize,
}

impl UnsupportedDerivativeSlot {
    const fn new(slot: usize, available: usize) -> Self {
        Self { slot, available }
    }
}

/// Construction of constant and independently seeded jets.
///
/// Implementations connect generic compilation code to the constructors of a
/// concrete jet family.
///
/// Supported derivative slots are zero-based:
///
/// - [`Jet0`] supports no independent variables;
/// - [`Jet1`] and [`Jet2`] support slot `0`;
/// - [`JetBivariate1`] and [`JetBivariate2`] support slots `0` and `1`.
///
/// The physical meaning of each slot is supplied separately by a parameter
/// assignment.
pub(crate) trait SeedJet<V>: Sized {
    /// Number of independent-variable slots represented by this algebra.
    const VARIABLE_SLOTS: usize;

    /// Construct a jet with primal value `value` and all derivatives zero.
    fn constant(value: V) -> Self;

    /// Construct a jet with primal value `value`.
    ///
    /// The first derivative in `slot` is seeded to unity and all other
    /// represented derivatives are initialised consistently for an independent
    /// variable.
    fn variable(value: V, slot: usize) -> Result<Self, UnsupportedDerivativeSlot>;
}

impl<V, P> SeedJet<V> for Jet0<V, P>
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
            0 => Ok(JetBivariate1::variable_axis0(value)),
            1 => Ok(JetBivariate1::variable_axis1(value)),

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
            0 => Ok(JetBivariate2::variable_axis0(value)),
            1 => Ok(JetBivariate2::variable_axis1(value)),

            _ => Err(UnsupportedDerivativeSlot {
                slot,
                available: Self::VARIABLE_SLOTS,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::{Array0, arr0};

    type Value = Array0<f64>;

    // Replace `()` with the actual policy/marker used by the jet definitions,
    // or use the existing crate aliases for these types.
    type First = Jet1<Value, ()>;
    type Second = Jet2<Value, ()>;
    type BivariateFirst = JetBivariate1<Value, ()>;
    type BivariateSecond = JetBivariate2<Value, ()>;

    #[test]
    fn jet_families_report_supported_slot_counts() {
        assert_eq!(<Jet0<Value> as SeedJet<Value>>::VARIABLE_SLOTS, 0);
        assert_eq!(<First as SeedJet<Value>>::VARIABLE_SLOTS, 1);
        assert_eq!(<Second as SeedJet<Value>>::VARIABLE_SLOTS, 1);
        assert_eq!(<BivariateFirst as SeedJet<Value>>::VARIABLE_SLOTS, 2,);
        assert_eq!(<BivariateSecond as SeedJet<Value>>::VARIABLE_SLOTS, 2,);
    }

    #[test]
    fn jet0_constructs_constants() {
        let seeded = <Jet0<Value> as SeedJet<Value>>::constant(arr0(3.0));

        assert_eq!(seeded, Jet0::constant(arr0(3.0)));
    }

    #[test]
    fn jet0_rejects_every_variable_slot() {
        for slot in [0, 1, 4] {
            let error = <Jet0<Value> as SeedJet<Value>>::variable(arr0(3.0), slot).unwrap_err();

            assert_eq!(error, UnsupportedDerivativeSlot { slot, available: 0 },);
        }
    }

    #[test]
    fn univariate_first_seeds_slot_zero() {
        let seeded = <First as SeedJet<Value>>::variable(arr0(3.0), 0).unwrap();

        assert_eq!(seeded, Jet1::variable(arr0(3.0)));
    }

    #[test]
    fn univariate_second_seeds_slot_zero() {
        let seeded = <Second as SeedJet<Value>>::variable(arr0(3.0), 0).unwrap();

        assert_eq!(seeded, Jet2::variable(arr0(3.0)));
    }

    #[test]
    fn univariate_jets_reject_slot_one() {
        assert_eq!(
            <First as SeedJet<Value>>::variable(arr0(3.0), 1).unwrap_err(),
            UnsupportedDerivativeSlot {
                slot: 1,
                available: 1,
            },
        );

        assert_eq!(
            <Second as SeedJet<Value>>::variable(arr0(3.0), 1).unwrap_err(),
            UnsupportedDerivativeSlot {
                slot: 1,
                available: 1,
            },
        );
    }

    #[test]
    fn bivariate_first_maps_slots_to_x_and_y() {
        assert_eq!(
            <BivariateFirst as SeedJet<Value>>::variable(arr0(3.0), 0).unwrap(),
            JetBivariate1::variable_axis0(arr0(3.0)),
        );

        assert_eq!(
            <BivariateFirst as SeedJet<Value>>::variable(arr0(3.0), 1).unwrap(),
            JetBivariate1::variable_axis1(arr0(3.0)),
        );
    }

    #[test]
    fn bivariate_second_maps_slots_to_x_and_y() {
        assert_eq!(
            <BivariateSecond as SeedJet<Value>>::variable(arr0(3.0), 0).unwrap(),
            JetBivariate2::variable_axis0(arr0(3.0)),
        );

        assert_eq!(
            <BivariateSecond as SeedJet<Value>>::variable(arr0(3.0), 1).unwrap(),
            JetBivariate2::variable_axis1(arr0(3.0)),
        );
    }

    #[test]
    fn bivariate_jets_reject_slot_two() {
        assert_eq!(
            <BivariateFirst as SeedJet<Value>>::variable(arr0(3.0), 2).unwrap_err(),
            UnsupportedDerivativeSlot {
                slot: 2,
                available: 2,
            },
        );

        assert_eq!(
            <BivariateSecond as SeedJet<Value>>::variable(arr0(3.0), 2).unwrap_err(),
            UnsupportedDerivativeSlot {
                slot: 2,
                available: 2,
            },
        );
    }
}
