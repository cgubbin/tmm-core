//! Internal-wave reconstruction and spatial electromagnetic fields.
//!
//! This module represents fields in two stages:
//!
//! 1. backends reconstruct forward and backward wave amplitudes at exterior
//!    and finite-layer boundaries;
//! 2. backend-neutral post-processing propagates those amplitudes to requested
//!    spatial positions and calculates physical field and power quantities.
//!
//! # Boundary-wave convention
//!
//! Directions are always geometric:
//!
//! - `forward` propagates from left to right;
//! - `backward` propagates from right to left.
//!
//! The convention is unchanged by incidence side and applies equally to
//! driven scattering solutions and source-free outgoing modes.
//!
//! Finite-layer waves are retained at both layer boundaries. This avoids
//! reconstructing a boundary by dividing by a small evanescent propagation
//! factor.
//!
//! # Driven fields
//!
//! [`PlaneWaveFieldBackend`] reconstructs boundary waves for driven
//! plane-wave scattering. Differentiated solves may additionally return
//! derivatives of every boundary-wave amplitude.
//!
//! [`PlaneWaveFieldResponse`] provides:
//!
//! - the usual reflection, transmission, and absorptance response;
//! - reconstructed boundary waves;
//! - spatial field sampling;
//! - per-layer and whole-stack power balance.
//!
//! # Outgoing modes
//!
//! [`OutgoingModeFieldBackend`] reconstructs the source-free waves associated
//! with a located mode. Incoming exterior amplitudes are zero. The returned
//! amplitudes share a consistent backend-selected scale, but are not generally
//! quasinormal-mode normalized.
//!
//! # Sampling and observables
//!
//! [`FieldSampling`] expands high-level profile requests into concrete
//! [`FieldPosition`] values. The isotropic post-processing routines then
//! calculate canonical tangential field states and signed normal power flux.
mod boundary;
mod cartesian;
mod error;
mod isotropic;
mod modal;
mod observables;
mod plane_wave;
mod sampling;

pub use boundary::{
    BidirectionalWaveDifferential, BidirectionalWaves, BoundaryWaveDerivatives,
    BoundaryWaveSolution, BoundaryWaves, ExteriorBoundaryWaveDifferential, ExteriorBoundaryWaves,
    LayerBoundaryWaves,
};

pub(crate) use boundary::{
    BidirectionalWavesGeneric, LayerBoundaryWavesGeneric, first_order_fields_from_generic,
    second_order_fields_from_generic, value_fields_from_generic,
};

pub(crate) use cartesian::CartesianVectorAlgebra;

pub use error::PlaneWaveFieldError;

pub use modal::{ModeFieldResponse, OutgoingModeFieldBackend};

pub use isotropic::IsotropicFieldState;

pub use cartesian::{CartesianElectromagneticField, CartesianVector3};

pub use observables::{PlaneWaveFieldSample, PlaneWaveFields, PlaneWavePowerBalance};

pub use plane_wave::{
    DifferentiablePlaneWaveFieldBackend, PlaneWaveFieldBackend, PlaneWaveFieldResponse,
};

pub use sampling::{
    ExteriorSampling, FieldPosition, FieldSampling, FieldSamplingRegion, LayerSampling,
};

/// Internal field data requested from a backend solve.
///
/// The derivative order is represented by the backend workspace entry type.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum InternalFieldRequest {
    /// Compute only the external response.
    None,

    /// Retain enough data to reconstruct waves at finite-layer boundaries.
    LayerBoundaries,
}

impl InternalFieldRequest {
    pub(crate) const fn is_requested(self) -> bool {
        matches!(self, Self::LayerBoundaries)
    }
}

/// Common storage for a response and its reconstructed boundary waves.
///
/// This is deliberately crate-private. Driven and modal public responses have
/// different physical semantics and should remain distinct public types.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct FieldResponse<R, W> {
    response: R,
    boundary_waves: W,
}

impl<R, W> FieldResponse<R, W> {
    pub(crate) fn new(response: R, boundary_waves: W) -> Self {
        Self {
            response,
            boundary_waves,
        }
    }

    pub(crate) fn response(&self) -> &R {
        &self.response
    }

    pub(crate) fn boundary_waves(&self) -> &W {
        &self.boundary_waves
    }

    pub(crate) fn into_parts(self) -> (R, W) {
        (self.response, self.boundary_waves)
    }
}
