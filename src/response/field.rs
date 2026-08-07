use crate::{
    ConstitutiveFields, DissipationDensity, ElectromagneticFields, EnergyDensity, VectorField,
    differential::DifferentialResponse, field::ScalarField, input::CoordinatePoint,
    response::StackRegion,
};

use super::{FieldMetadata, Response};

use nalgebra::ComplexField;
use ndarray::{ArrayView1, Dimension};

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
