use super::{
    BoundaryWaveSolution, BoundaryWaves, FieldPosition, FieldResponse, FieldSampling,
    PlaneWaveFieldError, PlaneWaveFields, PlaneWavePowerBalance, plane_wave_power_balance,
    sample_plane_wave_field_profile, sample_plane_wave_fields,
};

use crate::{
    ComplexScalar, PlaneWaveBackend, PlaneWaveInput, PlaneWaveResponse,
    PlaneWaveResponseDerivatives, Stack,
    backend::{
        PlaneWaveAmplitudes,
        derivative::{SpectralDerivativeVariable, StructuralDerivativeVariable},
        plane_wave::PlaneWavePower,
    },
    material::EvaluateMaterial,
};

use ndarray::{ArrayBase, Dimension, OwnedRepr};
use num_traits::Float;

/// Backend capable of reconstructing internal waves for a driven plane-wave
/// scattering problem.
pub trait PlaneWaveFieldBackend<C, D, S>: PlaneWaveBackend<C, D, S>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn solve_plane_wave_internal_fields(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error>;

    fn solve_plane_wave_internal_fields_structural_first_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error>;

    fn solve_plane_wave_internal_fields_structural_second_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error>;
}

/// Plane-wave field backend supporting spectral derivatives.
///
/// This remains separate because some backends may support structural
/// differentiation without supporting derivatives of dispersive materials.
pub trait DifferentiablePlaneWaveFieldBackend<C, D, S>: PlaneWaveFieldBackend<C, D, S>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn solve_plane_wave_internal_fields_spectral_first_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error>;

    fn solve_plane_wave_internal_fields_spectral_second_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error>;
}

/// Driven plane-wave response together with internal boundary waves.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveFieldResponse<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    inner: FieldResponse<PlaneWaveResponse<C, D>, BoundaryWaveSolution<C, D>>,
}

impl<C, D> PlaneWaveFieldResponse<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn new(
        response: PlaneWaveResponse<C, D>,
        boundary_waves: BoundaryWaveSolution<C, D>,
    ) -> Self {
        Self {
            inner: FieldResponse::new(response, boundary_waves),
        }
    }

    pub fn response(&self) -> &PlaneWaveResponse<C, D> {
        self.inner.response()
    }

    pub fn boundary_waves(&self) -> &BoundaryWaveSolution<C, D> {
        self.inner.boundary_waves()
    }

    pub fn into_parts(self) -> (PlaneWaveResponse<C, D>, BoundaryWaveSolution<C, D>) {
        self.inner.into_parts()
    }

    pub fn amplitudes(&self) -> &PlaneWaveAmplitudes<C, D> {
        self.response().amplitudes()
    }

    pub fn power(&self) -> &PlaneWavePower<C::RealField, D> {
        self.response().power()
    }

    pub fn derivatives(&self) -> Option<&PlaneWaveResponseDerivatives<C, D>> {
        self.response().derivatives()
    }
}

impl<C, D> PlaneWaveFieldResponse<C, D>
where
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
{
    pub fn sample_fields<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        sampling: &FieldSampling<C::RealField>,
    ) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateMaterial<C, Real = C::RealField>,
    {
        sample_plane_wave_field_profile(stack, input, self.boundary_waves(), sampling)
    }

    pub fn sample_field_positions<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        positions: impl IntoIterator<Item = FieldPosition<C::RealField>>,
    ) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateMaterial<C, Real = C::RealField>,
    {
        sample_plane_wave_fields(stack, input, self.boundary_waves(), positions)
    }

    pub fn power_balance<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWavePowerBalance<C::RealField, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateMaterial<C, Real = C::RealField>,
    {
        plane_wave_power_balance(stack, input, self)
    }
}
