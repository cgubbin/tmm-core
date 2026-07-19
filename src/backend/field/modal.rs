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
pub trait OutgoingModeFieldBackend<C, S>: OutgoingModeStateBackend<C, S>
where
    C: ComplexScalar,
{
    fn outgoing_mode_internal_fields(
        &self,
        stack: &S,
        mode: &OutgoingMode<C>,
    ) -> Result<ModeFieldResponse<C>, Self::Error>;
}

/// Modal response together with reconstructed outgoing boundary waves.
///
/// The exterior and finite-layer waves share one consistent backend-selected
/// scale. This scale is suitable for relative field profiles, but is not a
/// quasinormal-mode normalization and carries an arbitrary global phase.
///
/// Physical QNM normalization remains a separate post-processing operation
/// involving the reconstructed field, constitutive derivatives, and any
/// required exterior or surface contributions.
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

    /// Return the reconstructed modal state and reevaluated residual.
    pub fn response(&self) -> &OutgoingModeResponse<C> {
        self.inner.response()
    }

    /// Return the exterior and finite-layer boundary waves.
    pub fn boundary_waves(&self) -> &BoundaryWaveSolution<C, Ix0> {
        self.inner.boundary_waves()
    }

    /// Consume the result and return its response and boundary waves.
    pub fn into_parts(self) -> (OutgoingModeResponse<C>, BoundaryWaveSolution<C, Ix0>) {
        self.inner.into_parts()
    }
}
