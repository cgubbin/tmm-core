//! High-level planar-wave evaluation.
//!
//! This module exposes two complementary evaluator families:
//!
//! - [`RealAxisEvaluator`] accepts caller-facing real coordinates, compiles
//!   them into the canonical backend representation, and maps requested
//!   derivatives back to caller-facing physical parameters;
//! - [`ComplexPlaneEvaluator`] operates directly on canonical complex
//!   coordinates and a precompiled canonical stack. Its jet algebra is
//!   supplied by the caller, so complex-plane derivatives retain their raw
//!   algebraic meaning.
//!
//! Real-axis evaluation is intended for driven plane-wave calculations and
//! physical parameter sensitivities. Complex-plane evaluation is intended for
//! mode finding, analytic continuation, modal reconstruction, and other
//! advanced calculations that require explicit control of complex coordinate
//! and exterior-wavevector branches.

mod complex_plane;
mod error;
// mod query;
mod real_axis;

// #[cfg(test)]
// mod tests;
//
pub use complex_plane::ComplexPlaneEvaluator;
pub use error::{RealAxisEvaluationError, SolveRequestError};
pub use real_axis::RealAxisEvaluator;
