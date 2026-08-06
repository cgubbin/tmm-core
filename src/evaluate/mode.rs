use nalgebra::ComplexField;
use ndarray::Dimension;
use num_traits::{FromPrimitive, One};

use crate::{
    ComplexPlane, ComplexScalar,
    algebra::{ComplexJet, Jet, ScalarAlgebra, ScalarAlgebraExpRelExt},
    backend::{
        ExteriorAdmittanceProvider, PlaneWaveEntries, PlaneWaveSolutionSource,
        ReconstructLayerModeWaves, RetainedIsotropicLayers,
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
        LayerBoundaries, LayerBoundaryWaves, LayerIntegrationInput, LayerProjectionError, Layers,
        assemble_layer_integration_inputs,
    },
};

use thiserror::Error;

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ModeReconstructionError {
    #[error("workspace does not retain the data required for modal reconstruction")]
    ModeDataNotRetained,

    #[error("the outgoing boundary system has no usable modal null vector")]
    NoModalNullVector,

    #[error("the outgoing boundary system has a degenerate modal null space")]
    DegenerateNullSpace,

    #[error(transparent)]
    LayerProjection(#[from] LayerProjectionError),

    #[error(transparent)]
    Aggregation(#[from] LayerAggregateError),

    #[error(
        "reconstructed modal boundary-wave count {wave_count} does not match \
         retained finite-layer count {layer_count}"
    )]
    LayerCountMismatch {
        wave_count: usize,
        layer_count: usize,
    },
}

#[derive(Debug, Copy, Clone)]
pub struct PlaneWaveMode<'a, J, M, W>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
{
    state: &'a PlaneWaveState<J, J::Scalar, M, W>,
}

impl<'a, J, M, W> PlaneWaveMode<'a, J, M, W>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
{
    /// Construct an excitation after validating the state's projection constraint.
    pub(crate) fn new(state: &'a PlaneWaveState<J, J::Scalar, M, W>) -> Self {
        Self { state }
    }

    pub(crate) fn state(&self) -> &'a PlaneWaveState<J, J::Scalar, M, W> {
        self.state
    }

    pub(crate) fn into_inner(self) -> &'a PlaneWaveState<J, J::Scalar, M, W> {
        self.state
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
        todo!()
    }

    pub(crate) fn raw_layer_integration_inputs_unchecked(
        &self,
    ) -> Result<Layers<LayerIntegrationInput<J>>, ModeReconstructionError>
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
    ) -> Result<Layers<BilinearLayerNormalization<J>>, ModeReconstructionError>
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
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorAdmittanceProvider<Algebra = J>,
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
            )
            .unwrap(); // TODO: Change this error type and catch

        Ok(sequence.into_qnm_normalisation(
            coordinates.vacuum_angular_wavenumber(),
            coordinates.parallel_angular_wavenumber(),
        ))
    }

    pub fn qnm_normalisation_contributions(
        &self,
    ) -> Result<
        DifferentialResponseFor<J, Layers<BilinearLayerNormalization<J>>>,
        ModeReconstructionError,
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
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorAdmittanceProvider<Algebra = J>,
        J::Policy: DerivativePartsPolicy<Layers<BilinearLayerNormalization<J>>>,
        Layers<BilinearLayerNormalization<J>>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        Ok(self
            .raw_qnm_normalisation_unchecked()?
            .into_differential_response(&J::Policy::default(), self.state().mapping()))
    }

    pub fn qnm_normalisation(
        &self,
    ) -> Result<
        DifferentialResponseFor<J, AggregateBilinearNormalization<J>>,
        ModeReconstructionError,
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
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorAdmittanceProvider<Algebra = J>,
        J::Policy: DerivativePartsPolicy<AggregateBilinearNormalization<J>>,
        AggregateBilinearNormalization<J>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        Ok(self
            .raw_qnm_normalisation_unchecked()?
            .aggregate()?
            .into_differential_response(&J::Policy::default(), self.state().mapping()))
    }
}
