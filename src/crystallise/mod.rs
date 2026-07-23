//! Conversion from internal algebraic values into public differential results.
//!
//! Backend calculations retain values and derivatives in algebra types such as
//! [`ArrayJet`] and [`ArraySpectralJet`]. This module converts those internal
//! representations into stable, backend-independent response types.
//!
//! Crystallisation is intentionally delayed until all requested physical
//! quantities have been evaluated. Crystallised results are intended for
//! inspection, coordinate transformation, plotting, and use by downstream
//! crates.
//!
//! This module performs no differentiation. It only separates values and
//! derivative components already carried by an algebraic result.

mod mode;
mod parts;
mod quantity;
mod storage;

pub(crate) use mode::{DirectionalFirstMode, DirectionalSecondMode, SpectralSecondMode, ValueMode};
pub(crate) use parts::{DirectionalFirstParts, DirectionalSecondParts, SpectralSecondParts};
pub(crate) use quantity::Crystallise;
pub(crate) use storage::{
    IntoDirectionalFirstParts, IntoDirectionalSecondParts, IntoSpectralSecondParts, IntoValue,
};
