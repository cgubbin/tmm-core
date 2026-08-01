use crate::{
    LayerDissipation, LayerPower, Response, SpatialProfile, SpatialProfileError, StoredEnergy,
    differential::DifferentialResponse,
    field::ScalarField,
    input::PlaneWavePoint,
    response::{LayerLocation, LayerMetadata},
};

use ndarray::{ArrayView1, Dimension, IntoDimension};

pub type LayerPowerResponse<R, ED, D> =
    Response<LayerPower<ScalarField<R, <ED as Dimension>::Larger>>, LayerMetadata<R, ED>, D>;

pub type StoredEnergyResponse<R, ED, D> =
    Response<StoredEnergy<ScalarField<R, <ED as Dimension>::Larger>>, LayerMetadata<R, ED>, D>;

pub type LayerDissipationResponse<R, ED, D> =
    Response<LayerDissipation<ScalarField<R, <ED as Dimension>::Larger>>, LayerMetadata<R, ED>, D>;

/// A borrowed layer response at one canonical excitation point.
///
/// The excitation dimensions have been selected, leaving the spatial axis.
/// Both observable values and requested derivatives are retained.
pub struct LayerProfile<'a, F, D, R> {
    response: DifferentialResponse<F, D>,
    excitation: PlaneWavePoint<R>,
    layers: ArrayView1<'a, LayerLocation<R>>,
}

impl<'a, F, D, R> LayerProfile<'a, F, D, R> {
    pub fn response(&self) -> &DifferentialResponse<F, D> {
        &self.response
    }

    pub fn observables(&self) -> &F {
        self.response.value()
    }

    pub fn derivatives(&self) -> &D {
        self.response.derivatives()
    }

    pub fn excitation(&self) -> &PlaneWavePoint<R> {
        &self.excitation
    }

    pub fn layers(&self) -> ArrayView1<'a, LayerLocation<R>> {
        self.layers
    }
}

impl<O, D, R, ED> Response<O, D, LayerMetadata<R, ED>>
where
    R: Copy,
    ED: Dimension,
    O: SpatialProfile<ED>,
    D: SpatialProfile<ED>,
{
    /// Extracts a borrowed profile at one excitation point.
    ///
    /// All excitation axes are selected and the final spatial axis is retained.
    /// The returned profile includes both observable values and derivatives.
    ///
    /// # Errors
    ///
    /// Returns [`SpatialProfileError`] if `excitation_index` is outside the
    /// evaluated excitation domain.
    pub fn profile<I>(
        &self,
        excitation_index: I,
    ) -> Result<LayerProfile<'_, O::Profile<'_>, D::Profile<'_>, R>, SpatialProfileError>
    where
        I: IntoDimension<Dim = ED>,
    {
        let excitation_index = excitation_index.into_dimension();

        let values = self.observables().spatial_profile(&excitation_index)?;

        let derivatives = self.derivatives().spatial_profile(&excitation_index)?;

        let excitation = self.metadata().input().get_point(excitation_index).expect(
            "field response metadata and observables must have \
                 matching excitation dimensions",
        );

        Ok(LayerProfile {
            response: DifferentialResponse::new(values, derivatives),
            excitation,
            layers: self.metadata().layers().view(),
        })
    }
}
