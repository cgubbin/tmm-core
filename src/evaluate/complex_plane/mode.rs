use nalgebra::ComplexField;
use ndarray::{Dimension, Ix0};
use num_traits::{Float, FromPrimitive, One};
use thiserror::Error;

use crate::{
    ComplexPlane, ComplexScalar, ConstitutiveFields, ElectromagneticFields,
    algebra::{
        CartesianScalarAlgebra, ComplexJet, Jet, JetStack, ScalarAlgebra, ScalarAlgebraExpRelExt,
    },
    backend::{
        ExteriorContextProvider, ModalSolutionSource, ModeReconstructionError, PlaneWaveEntries,
        PlaneWaveModeCandidate, PlaneWaveSolutionSource, ReconstructExteriorModeWaves,
        ReconstructLayerModeWaves, RetainedIsotropicLayers,
    },
    input::JetMapping,
    material::{ConstitutiveDerivativeEvaluator, ConstitutiveSpectralFirstLift},
    observable::{
        AggregateBilinearNormalization, ConstitutiveFieldReconstructionError,
        FieldReconstructionError, FieldSamplingContext, LayerAggregateError, LayerEnergyError,
        LayerIntegrationInput, LayerProjectionError, Layers, assemble_layer_integration_inputs,
        project_layer_mode_waves,
    },
    spatial::{FieldSampling, ResolvedFieldSampling, SpatialResponse},
    waves::WaveSamplingContext,
};

use super::ComplexPlaneState;

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum QnmCreationError {
    #[error(transparent)]
    Normalisation(#[from] QnmNormalisationError),

    #[error(transparent)]
    Reconstruction(#[from] ModeReconstructionError),
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum QnmNormalisationError {
    #[error(transparent)]
    ModeProjection(#[from] ModeLayerProjectionError),

    #[error(transparent)]
    Energy(#[from] LayerEnergyError),

    #[error(transparent)]
    Aggregate(#[from] LayerAggregateError),

    #[error("QNM normalization produced a non-finite scale")]
    InvalidScale,
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ModeLayerProjectionError {
    #[error(transparent)]
    Reconstruction(#[from] ModeReconstructionError),

    #[error(transparent)]
    LayerProjection(#[from] LayerProjectionError),
}

#[derive(Clone)]
pub struct ComplexPlaneMode<'a, J, M, W>
where
    J: Jet,
{
    state: &'a ComplexPlaneState<'a, J, M, W>,
    solution: PlaneWaveModeCandidate<J>,
    raw_normalisation: AggregateBilinearNormalization<J>,
}

impl<'a, J, M, W> ComplexPlaneMode<'a, J, M, W>
where
    J: Jet,
{
    pub(crate) fn new(state: &'a ComplexPlaneState<'a, J, M, W>) -> Result<Self, QnmCreationError>
    where
        J: ComplexJet
            + ScalarAlgebra
            + ScalarAlgebraExpRelExt
            + ConstitutiveSpectralFirstLift<ComplexPlane, M>
            + Clone
            + JetMapping,
        J::RealJet: ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::RealJet as Jet>::Scalar: FromPrimitive + One,
        J::Dimension: Dimension,
        ComplexPlane: ConstitutiveDerivativeEvaluator<J::Scalar, J::Dimension, M>,
        W: PlaneWaveSolutionSource
            + ReconstructLayerModeWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>
            + ModalSolutionSource<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        let seed = state.workspace().modal_boundary_solution()?;

        let raw_normalisation = raw_qnm_normalisation_unchecked(&seed, state)?;

        let scale = normalisation_scale(&raw_normalisation)?;

        let solution = seed.scaled(&scale);

        Ok(Self {
            state,
            solution,
            raw_normalisation,
        })
    }

    pub(crate) fn state(&self) -> &'a ComplexPlaneState<'a, J, M, W> {
        self.state
    }

    pub(crate) fn solution(&self) -> &PlaneWaveModeCandidate<J> {
        &self.solution
    }

    pub(crate) fn seed_normalisation(&self) -> &AggregateBilinearNormalization<J> {
        &self.raw_normalisation
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        &'a ComplexPlaneState<'a, J, M, W>,
        PlaneWaveModeCandidate<J>,
        AggregateBilinearNormalization<J>,
    ) {
        (self.state, self.solution, self.raw_normalisation)
    }
}

impl<J, M, W> std::fmt::Debug for ComplexPlaneMode<'_, J, M, W>
where
    J: Jet,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ComplexPlaneMode").finish_non_exhaustive()
    }
}

impl<'a, J, M, W> ComplexPlaneMode<'a, J, M, W>
where
    J: Jet<Dimension = Ix0>,
{
    fn raw_electromagnetic_fields(
        &self,
        sampling: &ResolvedFieldSampling<<J::Scalar as ComplexField>::RealField>,
    ) -> Result<
        ElectromagneticFields<<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector>,
        FieldReconstructionError<<J::Scalar as ComplexField>::RealField>,
    >
    where
        J: JetStack + ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::Scalar as ComplexField>::RealField: Float,
        J::Stacked: CartesianScalarAlgebra,
        W: PlaneWaveSolutionSource
            + RetainedIsotropicLayers<Algebra = J>
            + ReconstructLayerModeWaves<Algebra = J>
            + ReconstructExteriorModeWaves<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        let wave_context = WaveSamplingContext::new(self.state.workspace());

        let boundary_waves = wave_context.modal_boundary_waves(self.solution())?;

        let context = FieldSamplingContext::new(self.state.workspace());

        let compiled_sampling = sampling.compile();

        context.reconstruct_from_boundary_waves(&boundary_waves, &compiled_sampling)
    }

    pub fn fields(
        &self,
        sampling: &FieldSampling<<J::Scalar as ComplexField>::RealField>,
    ) -> Result<
        ComplexModeFieldResponse<J, <J::Scalar as ComplexField>::RealField>,
        FieldReconstructionError<<J::Scalar as ComplexField>::RealField>,
    >
    where
        J: JetStack + ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::Scalar as ComplexField>::RealField: Float,
        J::Stacked: CartesianScalarAlgebra,
        W: PlaneWaveSolutionSource
            + RetainedIsotropicLayers<Algebra = J>
            + ReconstructLayerModeWaves<Algebra = J>
            + ReconstructExteriorModeWaves<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        let sampling = sampling.resolve_canonical(self.state.stack())?;

        let reconstructed = self.raw_electromagnetic_fields(&sampling)?;

        Ok(SpatialResponse::new(reconstructed, sampling))
    }

    pub fn constitutive_fields(
        &self,
        sampling: &FieldSampling<<J::Scalar as ComplexField>::RealField>,
    ) -> Result<
        ComplexModeConstitutiveFieldResponse<J, <J::Scalar as ComplexField>::RealField>,
        ConstitutiveFieldReconstructionError<<J::Scalar as ComplexField>::RealField>,
    >
    where
        J: JetStack + ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::Scalar as ComplexField>::RealField: Float,
        J::Stacked: CartesianScalarAlgebra,
        W: PlaneWaveSolutionSource
            + RetainedIsotropicLayers<Algebra = J>
            + ReconstructLayerModeWaves<Algebra = J>
            + ReconstructExteriorModeWaves<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        let sampling = sampling.resolve_canonical(self.state.stack())?;

        let electromagnetic_fields = self.raw_electromagnetic_fields(&sampling)?;

        let constitutive = self.state.raw_constitutive_parameters(&sampling)?;

        let reconstructed = electromagnetic_fields.into_constitutive_fields(&constitutive);

        Ok(SpatialResponse::new(reconstructed, sampling))
    }
}

pub(crate) fn raw_layer_integration_inputs_unchecked<J, W>(
    seed: &PlaneWaveModeCandidate<J>,
    workspace: &W,
) -> Result<Layers<LayerIntegrationInput<J>>, ModeLayerProjectionError>
where
    W: ReconstructLayerModeWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
    J: Clone,
{
    let boundary_waves = project_layer_mode_waves(workspace, seed)?;

    Ok(assemble_layer_integration_inputs(
        workspace,
        boundary_waves,
    )?)
}

pub(crate) fn raw_qnm_normalisation_unchecked<'a, J, M, W>(
    seed: &PlaneWaveModeCandidate<J>,
    state: &ComplexPlaneState<'a, J, M, W>,
) -> Result<AggregateBilinearNormalization<J>, QnmNormalisationError>
where
    J: ComplexJet
        + JetMapping
        + ScalarAlgebra
        + ScalarAlgebraExpRelExt
        + ConstitutiveSpectralFirstLift<ComplexPlane, M>
        + Clone,
    J::RealJet: ScalarAlgebra,
    J::Scalar: ComplexScalar,
    <J::RealJet as Jet>::Scalar: FromPrimitive + One,
    J::Dimension: Dimension,
    ComplexPlane: ConstitutiveDerivativeEvaluator<J::Scalar, J::Dimension, M>,
    W: PlaneWaveSolutionSource
        + RetainedIsotropicLayers<Algebra = J>
        + ReconstructLayerModeWaves<Algebra = J>,
    <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
{
    let coordinates = state.coordinates();

    let sequence = raw_layer_integration_inputs_unchecked(seed, state.workspace())?
        .integrate_bilinear()
        .into_brillouin_layers(
            state.stack().layers().iter().map(|layer| layer.material()),
            coordinates.vacuum_angular_wavenumber(),
        )?;

    Ok(sequence
        .into_qnm_normalisation(
            coordinates.vacuum_angular_wavenumber(),
            coordinates.parallel_angular_wavenumber(),
        )
        .aggregate()?)
}

fn normalisation_scale<J>(
    qnm_norm: &AggregateBilinearNormalization<J>,
) -> Result<J, QnmNormalisationError>
where
    J: ScalarAlgebra,
{
    let scale = qnm_norm.total().sqrt().reciprocal();

    if !scale.all_finite() {
        return Err(QnmNormalisationError::InvalidScale);
    }

    Ok(scale)
}

pub type ComplexModeFieldResponse<J, R> = SpatialResponse<
    ElectromagneticFields<<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector>,
    R,
>;

pub type ComplexModeConstitutiveFieldResponse<J, R> = SpatialResponse<
    ConstitutiveFields<<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector>,
    R,
>;
