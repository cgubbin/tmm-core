//! Construction of constant and independently seeded jets.
//!
//! Caller-facing parameters are assigned to numbered derivative slots by
//! [`DerivativeMapping`](crate::parameter::DerivativeMapping). This module
//! maps those slot numbers onto the constructors supplied by each concrete
//! jet algebra.
//!
//! Slot numbers have no intrinsic physical meaning. For example, slot zero
//! represents the spectral coordinate only when the active derivative mapping
//! places that coordinate in slot zero.

use nalgebra::ComplexField;
use ndarray::{Array, Dimension};
use thiserror::Error;

use crate::algebra::{
    ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, Jet, Jet0, Jet1, Jet2,
    JetBivariate1, JetBivariate2,
};

/// A requested derivative slot is not represented by the selected jet algebra.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Error)]
#[error(
    "derivative slot {slot} is unsupported by this jet algebra; \
     the algebra provides {available} slot(s)"
)]
pub struct UnsupportedDerivativeSlot {
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
/// Derivative order does not determine the number of independent variables.
/// `Jet1` and `Jet2` are both univariate, while the bivariate jet families
/// expose two independent variable slots.
///
/// The physical meaning of each slot is supplied separately by a derivative
/// mapping.
///
/// This trait is public only because it appears in bounds on public evaluator
/// implementations. It is not intended as a user-facing extension point.
#[doc(hidden)]
pub trait SeedJet: Sized + Jet {
    /// Number of independent-variable slots represented by this algebra.
    const VARIABLE_SLOTS: usize;

    /// Construct a jet with primal value `value` and all derivatives zero.
    fn constant(value: Array<Self::Scalar, Self::Dimension>) -> Self;

    /// Construct a jet with primal value `value`.
    ///
    /// The first derivative in `slot` is seeded to unity and all other
    /// represented derivatives are initialised consistently for an independent
    /// variable.
    fn variable(
        value: Array<Self::Scalar, Self::Dimension>,
        slot: usize,
    ) -> Result<Self, UnsupportedDerivativeSlot>;
}

impl<C, D, P> SeedJet for ArrayJet0<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    const VARIABLE_SLOTS: usize = 0;

    fn constant(value: Array<Self::Scalar, Self::Dimension>) -> Self {
        Jet0::constant(value)
    }

    fn variable(
        _value: Array<Self::Scalar, Self::Dimension>,
        slot: usize,
    ) -> Result<Self, UnsupportedDerivativeSlot> {
        Err(UnsupportedDerivativeSlot {
            slot,
            available: Self::VARIABLE_SLOTS,
        })
    }
}

impl<C, D, P> SeedJet for ArrayJet1<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    const VARIABLE_SLOTS: usize = 1;

    fn constant(value: Array<C, D>) -> Self {
        Jet1::constant(value)
    }

    fn variable(value: Array<C, D>, slot: usize) -> Result<Self, UnsupportedDerivativeSlot> {
        match slot {
            0 => Ok(Jet1::variable(value)),

            _ => Err(UnsupportedDerivativeSlot {
                slot,
                available: Self::VARIABLE_SLOTS,
            }),
        }
    }
}

impl<C, D, P> SeedJet for ArrayJet2<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    const VARIABLE_SLOTS: usize = 1;

    fn constant(value: Array<C, D>) -> Self {
        Jet2::constant(value)
    }

    fn variable(value: Array<C, D>, slot: usize) -> Result<Self, UnsupportedDerivativeSlot> {
        match slot {
            0 => Ok(Jet2::variable(value)),

            _ => Err(UnsupportedDerivativeSlot {
                slot,
                available: Self::VARIABLE_SLOTS,
            }),
        }
    }
}

impl<C, D, P> SeedJet for ArrayJetBivariate1<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    const VARIABLE_SLOTS: usize = 2;

    fn constant(value: Array<C, D>) -> Self {
        JetBivariate1::constant(value)
    }

    fn variable(value: Array<C, D>, slot: usize) -> Result<Self, UnsupportedDerivativeSlot> {
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

impl<C, D, P> SeedJet for ArrayJetBivariate2<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    const VARIABLE_SLOTS: usize = 2;

    fn constant(value: Array<C, D>) -> Self {
        JetBivariate2::constant(value)
    }

    fn variable(value: Array<C, D>, slot: usize) -> Result<Self, UnsupportedDerivativeSlot> {
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

    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use crate::algebra::{
        ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, RealParameter,
    };

    type C = Complex64;

    type Value = ArrayJet0<C, Ix0, RealParameter>;
    type First = ArrayJet1<C, Ix0, RealParameter>;
    type Second = ArrayJet2<C, Ix0, RealParameter>;
    type BivariateFirst = ArrayJetBivariate1<C, Ix0, RealParameter>;
    type BivariateSecond = ArrayJetBivariate2<C, Ix0, RealParameter>;

    fn value() -> ndarray::Array0<C> {
        arr0(C::new(3.0, 0.0))
    }

    #[test]
    fn jet_families_report_supported_slot_counts() {
        assert_eq!(<Value as SeedJet>::VARIABLE_SLOTS, 0,);

        assert_eq!(<First as SeedJet>::VARIABLE_SLOTS, 1,);

        assert_eq!(<Second as SeedJet>::VARIABLE_SLOTS, 1,);

        assert_eq!(<BivariateFirst as SeedJet>::VARIABLE_SLOTS, 2,);

        assert_eq!(<BivariateSecond as SeedJet>::VARIABLE_SLOTS, 2,);
    }

    #[test]
    fn value_jet_constructs_constants() {
        let seeded = <Value as SeedJet>::constant(value());

        assert_eq!(seeded, ArrayJet0::constant(value()),);
    }

    #[test]
    fn value_jet_rejects_variable_slots() {
        for slot in [0, 1, 4] {
            let error = <Value as SeedJet>::variable(value(), slot).unwrap_err();

            assert_eq!(error, UnsupportedDerivativeSlot { slot, available: 0 },);
        }
    }

    #[test]
    fn univariate_first_seeds_slot_zero() {
        let seeded = <First as SeedJet>::variable(value(), 0).unwrap();

        assert_eq!(seeded, ArrayJet1::variable(value()),);
    }

    #[test]
    fn univariate_second_seeds_slot_zero() {
        let seeded = <Second as SeedJet>::variable(value(), 0).unwrap();

        assert_eq!(seeded, ArrayJet2::variable(value()),);
    }

    #[test]
    fn univariate_jets_reject_slot_one() {
        assert_eq!(
            <First as SeedJet>::variable(value(), 1,).unwrap_err(),
            UnsupportedDerivativeSlot {
                slot: 1,
                available: 1,
            },
        );

        assert_eq!(
            <Second as SeedJet>::variable(value(), 1,).unwrap_err(),
            UnsupportedDerivativeSlot {
                slot: 1,
                available: 1,
            },
        );
    }

    #[test]
    fn bivariate_first_maps_slots_to_axes() {
        assert_eq!(
            <BivariateFirst as SeedJet>::variable(value(), 0,).unwrap(),
            ArrayJetBivariate1::variable_axis0(value(),),
        );

        assert_eq!(
            <BivariateFirst as SeedJet>::variable(value(), 1,).unwrap(),
            ArrayJetBivariate1::variable_axis1(value(),),
        );
    }

    #[test]
    fn bivariate_second_maps_slots_to_axes() {
        assert_eq!(
            <BivariateSecond as SeedJet>::variable(value(), 0,).unwrap(),
            ArrayJetBivariate2::variable_axis0(value(),),
        );

        assert_eq!(
            <BivariateSecond as SeedJet>::variable(value(), 1,).unwrap(),
            ArrayJetBivariate2::variable_axis1(value(),),
        );
    }

    #[test]
    fn bivariate_jets_reject_slot_two() {
        assert_eq!(
            <BivariateFirst as SeedJet>::variable(value(), 2,).unwrap_err(),
            UnsupportedDerivativeSlot {
                slot: 2,
                available: 2,
            },
        );

        assert_eq!(
            <BivariateSecond as SeedJet>::variable(value(), 2,).unwrap_err(),
            UnsupportedDerivativeSlot {
                slot: 2,
                available: 2,
            },
        );
    }
}
