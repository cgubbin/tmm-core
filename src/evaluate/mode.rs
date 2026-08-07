use nalgebra::ComplexField;
use ndarray::Dimension;
use num_traits::{FromPrimitive, One};

use crate::{
    ComplexPlane, ComplexScalar,
    algebra::{ComplexJet, Jet, ScalarAlgebra, ScalarAlgebraExpRelExt},
    backend::{
        ExteriorContextProvider, ModalSolutionSource, ModeReconstructionError, PlaneWaveEntries,
        PlaneWaveModeCandidate, PlaneWaveSolutionSource, ReconstructLayerModeWaves,
        RetainedIsotropicLayers,
    },
    derivative_parts::DerivativePartsPolicy,
    differential::IntoDifferentialResponse,
    evaluate::{
        PlaneWaveState,
        query::{DifferentialResponseFor, PlaneWaveQuery},
    },
    input::JetMapping,
    material::{ConstitutiveSpectralFirstLift, lifting::ConstitutiveDerivativeEvaluator},
    observable::{
        AggregateBilinearNormalization, BilinearLayerNormalization, LayerAggregateError,
        LayerBoundaries, LayerBoundaryWaves, LayerEnergyError, LayerIntegrationInput,
        LayerProjectionError, Layers, assemble_layer_integration_inputs, project_layer_mode_waves,
    },
};

use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum QnmNormalisationError {
    #[error(transparent)]
    ModeProjection(#[from] ModeLayerProjectionError),

    #[error(transparent)]
    Energy(#[from] LayerEnergyError),

    #[error(transparent)]
    Aggregate(#[from] LayerAggregateError),
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ModeLayerProjectionError {
    #[error(transparent)]
    Reconstruction(#[from] ModeReconstructionError),

    #[error(transparent)]
    LayerProjection(#[from] LayerProjectionError),
}

#[derive(Clone)]
pub struct PlaneWaveMode<'a, J, M, W>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
{
    state: &'a PlaneWaveState<J, J::Scalar, M, W>,
    seed: PlaneWaveModeCandidate<J>,
}

impl<'a, J, M, W> PlaneWaveMode<'a, J, M, W>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
{
    /// Construct an excitation after validating the state's projection constraint.
    pub(crate) fn new(
        state: &'a PlaneWaveState<J, J::Scalar, M, W>,
    ) -> Result<Self, ModeReconstructionError>
    where
        W: ModalSolutionSource<Algebra = J>,
    {
        let seed = state.workspace().modal_boundary_solution()?;

        Ok(Self { state, seed })
    }

    pub(crate) fn state(&self) -> &'a PlaneWaveState<J, J::Scalar, M, W> {
        self.state
    }

    pub(crate) fn seed(&self) -> &PlaneWaveModeCandidate<J> {
        &self.seed
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        &'a PlaneWaveState<J, J::Scalar, M, W>,
        PlaneWaveModeCandidate<J>,
    ) {
        (self.state, self.seed)
    }
}

impl<J, M, W> std::fmt::Debug for PlaneWaveMode<'_, J, M, W>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PlaneWaveMode").finish_non_exhaustive()
    }
}

impl<'a, J, M, W> PlaneWaveMode<'a, J, M, W>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    J::Policy: Default,
    W: ReconstructLayerModeWaves<Algebra = J>,
{
    pub(crate) fn raw_layer_mode_waves_unchecked(
        &self,
    ) -> Result<LayerBoundaries<LayerBoundaryWaves<J>>, ModeReconstructionError> {
        project_layer_mode_waves(self.state.workspace(), self.seed())
    }

    pub(crate) fn raw_layer_integration_inputs_unchecked(
        &self,
    ) -> Result<Layers<LayerIntegrationInput<J>>, ModeLayerProjectionError>
    where
        W: RetainedIsotropicLayers<Algebra = J>,
        J: Clone,
    {
        let boundary_waves = self.raw_layer_mode_waves_unchecked()?;

        Ok(assemble_layer_integration_inputs(
            self.state.workspace(),
            boundary_waves,
        )?)
    }

    pub(crate) fn raw_qnm_normalisation_unchecked(
        &self,
    ) -> Result<Layers<BilinearLayerNormalization<J>>, QnmNormalisationError>
    where
        J: ComplexJet
            + ScalarAlgebra
            + ScalarAlgebraExpRelExt
            + ConstitutiveSpectralFirstLift<ComplexPlane, M>
            + Clone,
        J::RealJet: ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::RealJet as Jet>::Scalar: FromPrimitive + One,
        J::Dimension: Dimension,
        ComplexPlane: ConstitutiveDerivativeEvaluator<J::Scalar, J::Dimension, M>,
        W: PlaneWaveSolutionSource + RetainedIsotropicLayers<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        let coordinates = self.state().problem().coordinates();

        let sequence = self
            .raw_layer_integration_inputs_unchecked()?
            .integrate_bilinear()
            .into_brillouin_layers(
                self.state()
                    .problem()
                    .stack()
                    .layers()
                    .iter()
                    .map(|layer| layer.material()),
                coordinates.vacuum_angular_wavenumber(),
            )?;

        Ok(sequence.into_qnm_normalisation(
            coordinates.vacuum_angular_wavenumber(),
            coordinates.parallel_angular_wavenumber(),
        ))
    }

    pub fn qnm_normalisation_contributions(
        &self,
    ) -> Result<
        DifferentialResponseFor<J, Layers<BilinearLayerNormalization<J>>>,
        QnmNormalisationError,
    >
    where
        J: ComplexJet
            + ScalarAlgebra
            + ScalarAlgebraExpRelExt
            + ConstitutiveSpectralFirstLift<ComplexPlane, M>
            + Clone,
        J::RealJet: ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::RealJet as Jet>::Scalar: FromPrimitive + One,
        J::Dimension: Dimension,
        ComplexPlane: ConstitutiveDerivativeEvaluator<J::Scalar, J::Dimension, M>,
        W: PlaneWaveSolutionSource + RetainedIsotropicLayers<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
        J::Policy: DerivativePartsPolicy<Layers<BilinearLayerNormalization<J>>>,
        Layers<BilinearLayerNormalization<J>>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        Ok(self
            .raw_qnm_normalisation_unchecked()?
            .into_differential_response(&J::Policy::default(), self.state().mapping()))
    }

    pub fn qnm_normalisation(
        &self,
    ) -> Result<DifferentialResponseFor<J, AggregateBilinearNormalization<J>>, QnmNormalisationError>
    where
        J: ComplexJet
            + ScalarAlgebra
            + ScalarAlgebraExpRelExt
            + ConstitutiveSpectralFirstLift<ComplexPlane, M>
            + Clone,
        J::RealJet: ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::RealJet as Jet>::Scalar: FromPrimitive + One,
        J::Dimension: Dimension,
        ComplexPlane: ConstitutiveDerivativeEvaluator<J::Scalar, J::Dimension, M>,
        W: PlaneWaveSolutionSource + RetainedIsotropicLayers<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
        J::Policy: DerivativePartsPolicy<AggregateBilinearNormalization<J>>,
        AggregateBilinearNormalization<J>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        Ok(self
            .raw_qnm_normalisation_unchecked()?
            .aggregate()?
            .into_differential_response(&J::Policy::default(), self.state().mapping()))
    }
}
