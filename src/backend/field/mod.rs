//! Internal-wave reconstruction and spatial electromagnetic fields.

mod boundary;
mod error;
mod modal;
mod observables;
mod plane_wave;
mod sampling;

pub use boundary::{
    BidirectionalWaveDifferential, BidirectionalWaves, BoundaryWaveDerivatives,
    BoundaryWaveSolution, BoundaryWaves, ExteriorBoundaryWaveDifferential, ExteriorBoundaryWaves,
    LayerBoundaryWaveDifferential, LayerBoundaryWaves,
};

pub(crate) use boundary::{
    BidirectionalWavesGeneric, LayerBoundaryWavesGeneric, first_order_fields_from_generic,
    second_order_fields_from_generic, value_fields_from_generic,
};

pub use error::PlaneWaveFieldError;

pub use modal::{ModeFieldResponse, OutgoingModeFieldBackend};

pub use observables::{
    IsotropicFieldState, PlaneWaveFieldSample, PlaneWaveFields, PlaneWavePowerBalance,
    plane_wave_power_balance, sample_plane_wave_field_profile, sample_plane_wave_fields,
};

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
