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
mod result;
mod state;

pub use error::{PlaneWaveEvaluationError, RequestError};
pub use evaluator::PlaneWaveEvaluator;
pub use result::PlaneWaveResult;
pub use state::PlaneWaveState;
