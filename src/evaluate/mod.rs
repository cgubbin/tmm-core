//! High-level plane-wave evaluation.
//!
//! This module compiles caller-facing plane-wave inputs into canonical,
//! derivative-aware problems and solves them using a statically selected
//! backend.
//!
//! Evaluation returns a retained [`PlaneWaveState`]. Derived quantities are
//! computed and crystallised from that state only when requested.

mod error;
mod evaluator;
mod excitation;
mod mode;
mod pair;
mod query;
mod result;
mod state;

#[cfg(test)]
mod tests;

pub use error::{PlaneWaveEvaluationError, SolveRequestError};
pub use evaluator::PlaneWaveEvaluator;
pub use result::PlaneWaveResult;
pub use state::PlaneWaveState;
