//! Seeding of caller-facing coordinate values into jet algebras.
//!
//! Coordinate compilation begins with values in the parameterisation supplied
//! by the caller. Before converting those values into canonical coordinates,
//! the compiler must decide whether each coordinate is:
//!
//! - constant for the current solve; or
//! - an independent variable represented by a particular derivative slot.
//!
//! This module performs only that seeding step. It does not validate coordinate
//! values and does not apply unit conversions or coordinate transformations.
//!
//! Seeding before canonicalisation is important: derivatives are therefore
//! taken with respect to the caller-facing coordinate rather than with respect
//! to the canonical backend coordinate.

use ndarray::{Array, Dimension};

use crate::input::compile::seed::{SeedJet, UnsupportedDerivativeSlot};

/// Seed caller-facing coordinate values into a jet algebra.
///
/// When `slot` is `Some(i)`, the coordinate is constructed as the independent
/// variable represented by derivative slot `i`. When `slot` is `None`, it is
/// constructed as a constant and all represented derivatives vanish.
///
/// The returned jet still contains values in the caller-facing
/// parameterisation. Coordinate canonicalisation is performed separately after
/// seeding so that the jet algebra propagates the appropriate chain rule.
///
/// # Errors
///
/// Returns [`UnsupportedDerivativeSlot`] when `slot` is not represented by
/// the selected jet algebra.
pub(crate) fn seed_coordinate<J>(
    values: Array<J::Scalar, J::Dimension>,
    slot: Option<usize>,
) -> Result<J, UnsupportedDerivativeSlot>
where
    J: SeedJet,
{
    match slot {
        Some(slot) => J::variable(values, slot),

        None => Ok(J::constant(values)),
    }
}
