use super::{
    BoundaryWaveSolution, FieldPosition, FieldResponse, FieldSampling, PlaneWaveFieldError,
    PlaneWaveFields, PlaneWavePowerBalance,
};

use crate::{
    ComplexScalar, EvaluateDifferentiableMaterial, PlaneWaveBackend, PlaneWaveInput,
    PlaneWaveResponse, PlaneWaveResponseDerivatives, Stack,
    backend::{
        PlaneWaveAmplitudes,
        derivative::{SpectralDerivativeVariable, StructuralDerivativeVariable},
        field::observables::{
            plane_wave_power_balance_spectral_first, plane_wave_power_balance_spectral_second,
            plane_wave_power_balance_structural_first, plane_wave_power_balance_structural_second,
            plane_wave_power_balance_values, sample_first_order_fields_k0,
            sample_first_order_fields_kx, sample_first_order_fields_thickness,
            sample_plane_wave_field_profile, sample_second_order_fields_full_spectral_hessian,
            sample_second_order_fields_k0, sample_second_order_fields_kx,
            sample_second_order_fields_thickness, sample_value_fields,
        },
        plane_wave::PlaneWavePower,
    },
    material::EvaluateMaterial,
};

use nalgebra::ComplexField;
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
}

pub trait PlaneWaveFieldThicknessDerivativeBackend<C, D, S>: PlaneWaveBackend<C, D, S>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn solve_plane_wave_internal_fields_thickness_first_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        layer: usize,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error>;

    fn solve_plane_wave_internal_fields_thickness_second_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        layer: usize,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error>;
}

pub trait PlaneWaveFieldKxDerivativeBackend<C, D, S>: PlaneWaveBackend<C, D, S>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn solve_plane_wave_internal_fields_kx_first_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error>;

    fn solve_plane_wave_internal_fields_kx_second_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error>;
}

/// Plane-wave field backend supporting spectral derivatives.
///
/// This remains separate because some backends may support structural
/// differentiation without supporting derivatives of dispersive materials.
pub trait PlaneWaveFieldSpectralDerivativeBackend<C, D, S>: PlaneWaveFieldBackend<C, D, S>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn solve_plane_wave_internal_fields_k0_first_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error>;

    fn solve_plane_wave_internal_fields_k0_second_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error>;

    fn solve_plane_wave_internal_fields_full_spectral_hessian(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error>;
}

/// Driven plane-wave response together with internal boundary waves.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveFieldResponse<C, D>
where
    C: ComplexField,
    D: Dimension,
{
    inner: FieldResponse<PlaneWaveResponse<C, D>, BoundaryWaveSolution<C, D>>,
}

impl<C, D> PlaneWaveFieldResponse<C, D>
where
    C: ComplexField,
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
    C::RealField: Copy + ComplexField,
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
        C::RealField: Float,
    {
        sample_plane_wave_field_profile(stack, input, self.boundary_waves(), sampling)
    }

    pub fn sample_fields_thickness_first_derivative<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        sampling: &FieldSampling<C::RealField>,
    ) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Float,
    {
        let positions = sampling.positions(stack)?;

        sample_first_order_fields_thickness(stack, input, self.boundary_waves(), positions)
    }

    pub fn sample_fields_thickness_second_derivative<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        sampling: &FieldSampling<C::RealField>,
    ) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Float,
    {
        let positions = sampling.positions(stack)?;

        sample_second_order_fields_thickness(stack, input, self.boundary_waves(), positions)
    }

    pub fn sample_fields_kx_first_derivative<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        sampling: &FieldSampling<C::RealField>,
    ) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Float,
    {
        let positions = sampling.positions(stack)?;

        sample_first_order_fields_kx(stack, input, self.boundary_waves(), positions)
    }

    pub fn sample_fields_kx_second_derivative<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        sampling: &FieldSampling<C::RealField>,
    ) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Float,
    {
        let positions = sampling.positions(stack)?;

        sample_second_order_fields_kx(stack, input, self.boundary_waves(), positions)
    }

    pub fn sample_fields_k0_first_derivative<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        sampling: &FieldSampling<C::RealField>,
    ) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Float,
    {
        let positions = sampling.positions(stack)?;

        sample_first_order_fields_k0(stack, input, self.boundary_waves(), positions)
    }

    pub fn sample_fields_k0_second_derivative<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        sampling: &FieldSampling<C::RealField>,
    ) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Float,
    {
        let positions = sampling.positions(stack)?;

        sample_second_order_fields_k0(stack, input, self.boundary_waves(), positions)
    }

    pub fn sample_fields_full_spectral_hessian<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        sampling: &FieldSampling<C::RealField>,
    ) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Float,
    {
        let positions = sampling.positions(stack)?;

        sample_second_order_fields_full_spectral_hessian(
            stack,
            input,
            self.boundary_waves(),
            positions,
        )
    }

    pub(crate) fn sample_field_positions<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        positions: impl IntoIterator<Item = FieldPosition<C::RealField>>,
    ) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateMaterial<C, Real = C::RealField>,
        C::RealField: Float,
    {
        sample_value_fields(
            stack,
            input,
            self.boundary_waves().values(),
            positions.into_iter().collect(),
        )
    }

    pub(crate) fn sample_field_positions_thickness_first_derivative<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        positions: impl IntoIterator<Item = FieldPosition<C::RealField>>,
    ) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Float,
    {
        sample_first_order_fields_thickness(
            stack,
            input,
            self.boundary_waves(),
            positions.into_iter().collect(),
        )
    }

    pub(crate) fn sample_field_positions_thickness_second_derivative<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        positions: impl IntoIterator<Item = FieldPosition<C::RealField>>,
    ) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Float,
    {
        sample_second_order_fields_thickness(
            stack,
            input,
            self.boundary_waves(),
            positions.into_iter().collect(),
        )
    }

    pub(crate) fn sample_field_positions_k0_first_derivative<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        positions: impl IntoIterator<Item = FieldPosition<C::RealField>>,
    ) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Float,
    {
        sample_first_order_fields_k0(
            stack,
            input,
            self.boundary_waves(),
            positions.into_iter().collect(),
        )
    }

    pub(crate) fn sample_field_positions_k0_second_derivative<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        positions: impl IntoIterator<Item = FieldPosition<C::RealField>>,
    ) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Float,
    {
        sample_second_order_fields_k0(
            stack,
            input,
            self.boundary_waves(),
            positions.into_iter().collect(),
        )
    }

    pub(crate) fn sample_field_positions_kx_first_derivative<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        positions: impl IntoIterator<Item = FieldPosition<C::RealField>>,
    ) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Float,
    {
        sample_first_order_fields_kx(
            stack,
            input,
            self.boundary_waves(),
            positions.into_iter().collect(),
        )
    }

    pub(crate) fn sample_field_positions_kx_second_derivative<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        positions: impl IntoIterator<Item = FieldPosition<C::RealField>>,
    ) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Float,
    {
        sample_second_order_fields_kx(
            stack,
            input,
            self.boundary_waves(),
            positions.into_iter().collect(),
        )
    }

    pub fn sample_field_positions_full_spectral_hessian<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        positions: impl IntoIterator<Item = FieldPosition<C::RealField>>,
    ) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Float,
    {
        sample_second_order_fields_full_spectral_hessian(
            stack,
            input,
            self.boundary_waves(),
            positions.into_iter().collect(),
        )
    }

    pub fn power_balance<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWavePowerBalance<C::RealField, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateMaterial<C, Real = C::RealField>,
        C::RealField: ComplexField,
    {
        plane_wave_power_balance_values(stack, input, self)
    }

    pub fn power_balance_structural_first_derivative<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWavePowerBalance<C::RealField, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateMaterial<C, Real = C::RealField>,
        C::RealField: ComplexField,
    {
        plane_wave_power_balance_structural_first(stack, input, self)
    }

    pub fn power_balance_structural_second_derivative<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWavePowerBalance<C::RealField, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateMaterial<C, Real = C::RealField>,
        C::RealField: ComplexField,
    {
        plane_wave_power_balance_structural_second(stack, input, self)
    }

    pub fn power_balance_spectral_first_derivative<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWavePowerBalance<C::RealField, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: ComplexField,
    {
        plane_wave_power_balance_spectral_first(stack, input, self)
    }

    pub fn power_balance_spectral_second_derivative<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWavePowerBalance<C::RealField, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: ComplexField,
    {
        plane_wave_power_balance_spectral_second(stack, input, self)
    }
}
