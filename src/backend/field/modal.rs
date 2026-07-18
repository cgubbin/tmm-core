use super::FieldResponse;

use crate::{
    ComplexScalar,
    backend::{
        field::BoundaryWaveSolution,
        mode::{OutgoingMode, OutgoingModeResponse, OutgoingModeStateBackend},
    },
};

use ndarray::{Dimension, Ix0};

/// Backend capable of reconstructing the outgoing internal waves associated
/// with a source-free mode.
///
/// The returned waves are not necessarily QNM-normalized. Normalization is a
/// separate post-processing stage because it requires the reconstructed field,
/// constitutive derivatives, and finite-domain boundary contributions.
pub trait OutgoingModeFieldBackend<C, D, S>: OutgoingModeStateBackend<C, D, S>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn outgoing_mode_internal_fields(
        &self,
        stack: &S,
        mode: &OutgoingMode<C>,
    ) -> Result<ModeFieldResponse<C>, Self::Error>;
}

/// Mode response together with its outgoing internal boundary waves.
///
/// The boundary waves retain the backend's raw mode scaling. QNM
/// normalization and phase fixing are performed explicitly by later field
/// post-processing.
#[derive(Clone, Debug, PartialEq)]
pub struct ModeFieldResponse<C>
where
    C: ComplexScalar,
{
    inner: FieldResponse<OutgoingModeResponse<C>, BoundaryWaveSolution<C, Ix0>>,
}

impl<C> ModeFieldResponse<C>
where
    C: ComplexScalar,
{
    pub(crate) fn new(
        response: OutgoingModeResponse<C>,
        boundary_waves: BoundaryWaveSolution<C, Ix0>,
    ) -> Self {
        Self {
            inner: FieldResponse::new(response, boundary_waves),
        }
    }

    pub fn response(&self) -> &OutgoingModeResponse<C> {
        &self.inner.response
    }

    pub fn boundary_waves(&self) -> &BoundaryWaveSolution<C, Ix0> {
        &self.inner.boundary_waves
    }

    pub fn into_parts(self) -> (OutgoingModeResponse<C>, BoundaryWaveSolution<C, Ix0>) {
        (self.inner.response, self.inner.boundary_waves)
    }
}
