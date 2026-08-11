//! Canonical inputs consumed by numerical backends.
//!
//! Caller-facing inputs may use different spectral coordinates, in-plane
//! coordinates, units, incidence directions, and derivative parameterisations.
//! The compilation layer converts those inputs into the canonical structures
//! defined here.
//!
//! Canonical plane-wave coordinates are:
//!
//! - vacuum angular wavenumber `k₀`, in inverse centimetres;
//! - conserved parallel angular wavenumber `k∥`, in inverse centimetres.
//!
//! Layer thicknesses are expressed in centimetres. Coordinates and thicknesses
//! use the same sampled jet algebra, so every derivative component and sampled
//! value has a compatible representation.
//!
//! These types are constructed only after public-input validation and
//! coordinate conversion. They therefore encode invariants expected by the
//! backend and do not repeat validation during numerical evaluation.
//!
//! The canonical solve representation is layered as follows:
//!
//! - [`CanonicalCoordinates`] stores canonical spectral and in-plane values;
//! - [`CanonicalStack`] stores the oriented exterior media and finite layers;
//! - [`CanonicalBackendInput`] combines solver input and stack;

mod coordinate;
mod stack;

pub(crate) use coordinate::CanonicalCoordinates;

pub(crate) use stack::{CanonicalLayer, CanonicalStack};

use crate::Polarisation;

/// Complete canonical problem consumed by an oriented backend.
///
/// The solver input and all finite-layer thicknesses use the same sampled
/// algebraic representation `J`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CanonicalBackendInput<M, J> {
    problem: CanonicalProblem<M, J>,
    polarisation: Polarisation,
}

impl<M, J> CanonicalBackendInput<M, J> {
    /// Construct a canonical oriented problem.
    pub(crate) fn new(problem: CanonicalProblem<M, J>, polarisation: Polarisation) -> Self {
        Self {
            problem,
            polarisation,
        }
    }

    /// Return the canonical problem.
    pub(crate) fn problem(&self) -> &CanonicalProblem<M, J> {
        &self.problem
    }

    /// Return the polarisation.
    pub(crate) fn polarisation(&self) -> Polarisation {
        self.polarisation
    }

    /// Consume the problem and return its components.
    pub(crate) fn into_parts(self) -> (CanonicalProblem<M, J>, Polarisation) {
        (self.problem, self.polarisation)
    }
}

#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalProblem<M, J> {
    coordinates: CanonicalCoordinates<J>,
    stack: CanonicalStack<M, J>,
}

impl<M, J> CanonicalProblem<M, J> {
    /// Construct a canonical oriented problem.
    pub(crate) fn new(coordinates: CanonicalCoordinates<J>, stack: CanonicalStack<M, J>) -> Self {
        Self { coordinates, stack }
    }

    /// Return the canonical coordinates.
    pub(crate) fn coordinates(&self) -> &CanonicalCoordinates<J> {
        &self.coordinates
    }

    /// Return the canonical oriented stack.
    pub(crate) fn stack(&self) -> &CanonicalStack<M, J> {
        &self.stack
    }

    /// Consume the problem and return its components.
    pub(crate) fn into_parts(self) -> (CanonicalCoordinates<J>, CanonicalStack<M, J>) {
        (self.coordinates, self.stack)
    }
}
