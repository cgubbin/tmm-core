use crate::{
    LayerDissipation, LayerPower, Response, StoredEnergy,
    differential::DifferentialResponse,
    field::ScalarField,
    input::CoordinatePoint,
    response::{LayerLocation, LayerMetadata},
};

use ndarray::{ArrayView1, Dimension};

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
    excitation: CoordinatePoint<R>,
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

    pub fn excitation(&self) -> &CoordinatePoint<R> {
        &self.excitation
    }

    pub fn layers(&self) -> ArrayView1<'a, LayerLocation<R>> {
        self.layers
    }
}
