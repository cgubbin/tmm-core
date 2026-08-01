use crate::{
    ConstitutiveFields, DissipationDensity, ElectromagneticFields, EnergyDensity, FieldIndexError,
    InterfacePower, LayerDissipation, LayerPower, ModeResidual, SpatialProfile,
    SpatialProfileError, StoredEnergy, VectorField,
    differential::{DifferentialResponse, NoDerivatives},
    field::{ScalarField, ScalarFieldView1, VectorFieldView1},
    input::{CoordinatePoint, IncidentSide},
    response::StackRegion,
};

use super::{FieldMetadata, Response};

use nalgebra::ComplexField;
use ndarray::{ArrayView1, Dimension, IntoDimension};

/// A borrowed field response at one canonical excitation point.
///
/// The excitation dimensions have been selected, leaving the spatial axis.
/// Both observable values and requested derivatives are retained.
pub struct FieldProfile<'a, F, D, R> {
    response: DifferentialResponse<F, D>,
    excitation: CoordinatePoint<R>,
    positions_cm: ArrayView1<'a, R>,
    regions: ArrayView1<'a, StackRegion>,
}

impl<'a, F, D, R> FieldProfile<'a, F, D, R> {
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

    pub fn positions_cm(&self) -> ArrayView1<'a, R> {
        self.positions_cm
    }

    pub fn regions(&self) -> ArrayView1<'a, StackRegion> {
        self.regions
    }
}

pub type ElectromagneticFieldResponse<C, ED, D> = Response<
    ElectromagneticFields<VectorField<C, <ED as Dimension>::Larger>>,
    D,
    FieldMetadata<<C as ComplexField>::RealField, ED>,
>;

pub type ConstitutiveFieldResponse<C, ED, D> = Response<
    ConstitutiveFields<VectorField<C, <ED as Dimension>::Larger>>,
    D,
    FieldMetadata<<C as ComplexField>::RealField, ED>,
>;

pub type EnergyDensityResponse<R, ED, D> =
    Response<EnergyDensity<ScalarField<R, <ED as Dimension>::Larger>>, D, FieldMetadata<R, ED>>;

pub type DissipationDensityResponse<R, ED, D> = Response<
    DissipationDensity<ScalarField<R, <ED as Dimension>::Larger>>,
    D,
    FieldMetadata<R, ED>,
>;

impl<O, D, R, ED> Response<O, D, FieldMetadata<R, ED>>
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
    ) -> Result<FieldProfile<'_, O::Profile<'_>, D::Profile<'_>, R>, SpatialProfileError>
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

        Ok(FieldProfile {
            response: DifferentialResponse::new(values, derivatives),
            excitation,
            positions_cm: self.metadata().positions_cm().view(),
            regions: self.metadata().regions().view(),
        })
    }
}
