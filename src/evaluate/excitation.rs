use std::ops::Neg;

use nalgebra::ComplexField;
use ndarray::{Dimension, Ix0};
use num_traits::{Float, FromPrimitive, One, Zero};

use crate::{
    ComplexScalar, ElectromagneticFields, FiniteLayerIndex, IncidentSide, LayerDissipation,
    PlaneWaveAmplitudes, RealAxis, VectorField,
    algebra::{
        CartesianScalarAlgebra, ComplexJet, Jet, JetStack, RealScalarAlgebra, ScalarAlgebra,
        ScalarAlgebraExpRelExt,
    },
    backend::{
        ExteriorContextProvider, PlaneWaveEntries, PlaneWaveSolutionSource, RetainedIsotropicLayers,
    },
    derivative_parts::DerivativePartsPolicy,
    differential::IntoDifferentialResponse,
    evaluate::{
        PlaneWaveState,
        pair::{PlaneWaveExcitationPair, PlaneWavePairError},
        query::{
            DifferentialResponseFor, PlaneWaveExternalQueries, PlaneWaveQuery, RawAmplitudes,
            RawPower,
        },
        state::{RawInterfacePower, RawLayerPower},
    },
    input::JetMapping,
    material::{
        ConstitutiveEvaluator, ConstitutiveLift, ConstitutiveSpectralFirstLift,
        lifting::ConstitutiveDerivativeEvaluator,
    },
    observable::{
        Amplitudes, BoundaryProjectionError, EnergyConfinement, FieldReconstructionError,
        FieldSamplingContext, InterfaceProjectionError, InterfaceStates, Interfaces,
        LayerBoundaries, LayerBoundaryStates, LayerBoundaryWaves, LayerConfinementError,
        LayerEnergy, LayerEnergyError, LayerParticipation, LayerParticipationError,
        LayerProjectionError, Layers, ProjectAmplitudes, ProjectPower,
    },
    spatial::FieldSampling,
    waves::{ReconstructLayerBoundaryWaves, WaveSamplingContext},
};

#[derive(Debug, Copy, Clone)]
pub struct PlaneWaveExcitation<'a, J, I, M, W>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    I: ComplexField,
{
    state: &'a PlaneWaveState<J, I, M, W>,
    incident_side: IncidentSide,
}

impl<'a, J, I, M, W> PlaneWaveExcitation<'a, J, I, M, W>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    I: ComplexField,
{
    /// Construct an excitation after validating the state's projection constraint.
    pub(crate) fn new(state: &'a PlaneWaveState<J, I, M, W>, incident_side: IncidentSide) -> Self {
        Self {
            state,
            incident_side,
        }
    }

    pub(crate) fn state(&self) -> &'a PlaneWaveState<J, I, M, W> {
        self.state
    }

    pub fn incident_side(&self) -> IncidentSide {
        self.incident_side
    }

    pub(crate) fn into_parts(self) -> (&'a PlaneWaveState<J, I, M, W>, IncidentSide) {
        (self.state, self.incident_side)
    }

    pub fn boundary_waves(
        &self,
    ) -> Result<
        DifferentialResponseFor<J, LayerBoundaries<LayerBoundaryWaves<J>>>,
        BoundaryProjectionError,
    >
    where
        J::Policy: Default + DerivativePartsPolicy<LayerBoundaries<LayerBoundaryWaves<J>>>,
        W: PlaneWaveSolutionSource + ReconstructLayerBoundaryWaves<Algebra = J>,
        LayerBoundaries<LayerBoundaryWaves<J>>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        Ok(self
            .state
            .raw_layer_boundary_waves_unchecked(self.incident_side)?
            .into_differential_response(&J::Policy::default(), self.state.mapping()))
    }

    pub fn boundary_states(
        &self,
    ) -> Result<
        DifferentialResponseFor<J, LayerBoundaries<LayerBoundaryStates<J>>>,
        BoundaryProjectionError,
    >
    where
        J: ScalarAlgebra,
        J::Scalar: ComplexScalar,
        J::Policy: Default + DerivativePartsPolicy<LayerBoundaries<LayerBoundaryStates<J>>>,
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
        LayerBoundaries<LayerBoundaryStates<J>>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        Ok(self
            .state
            .raw_layer_boundary_states_unchecked(self.incident_side)?
            .into_differential_response(&J::Policy::default(), self.state.mapping()))
    }

    pub fn interface_states(
        &self,
    ) -> Result<DifferentialResponseFor<J, Interfaces<InterfaceStates<J>>>, BoundaryProjectionError>
    where
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
        W::Entries: ProjectAmplitudes,
        <W::Entries as ProjectAmplitudes>::Amplitudes: Into<PlaneWaveAmplitudes<J>>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
        J: ScalarAlgebra + Clone,
        J::Scalar: ComplexScalar + One + Zero,
        J::Dimension: Dimension,
        J::Policy: Default + DerivativePartsPolicy<Interfaces<InterfaceStates<J>>>,
        Interfaces<InterfaceStates<J>>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        Ok(self
            .state
            .raw_interface_states_unchecked(self.incident_side)?
            .into_differential_response(&J::Policy::default(), self.state.mapping()))
    }
}

// Real Input Observables
impl<'a, J, R, M, W> PlaneWaveExcitation<'a, J, R, M, W>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField<RealField = R>,
    J::Dimension: Dimension,
    J::Policy: Default,
    R: ComplexField,
    W: PlaneWaveSolutionSource,
{
    pub fn amplitudes(
        &self,
    ) -> DifferentialResponseFor<J, RawAmplitudes<PlaneWaveState<J, R, M, W>, J>>
    where
        W::Entries: ProjectAmplitudes,
        J::Policy: DerivativePartsPolicy<RawAmplitudes<PlaneWaveState<J, R, M, W>, J>>,
        RawAmplitudes<PlaneWaveState<J, R, M, W>, J>:
            IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        self.state
            .raw_amplitudes(self.incident_side)
            .into_differential_response(&J::Policy::default(), self.state.mapping())
    }

    pub fn power(&self) -> DifferentialResponseFor<J, RawPower<PlaneWaveState<J, R, M, W>, J>>
    where
        W::Entries: ProjectPower,
        J::Policy: DerivativePartsPolicy<RawPower<PlaneWaveState<J, R, M, W>, J>>,
        RawPower<PlaneWaveState<J, R, M, W>, J>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        self.state
            .raw_power(self.incident_side)
            .into_differential_response(&J::Policy::default(), self.state.mapping())
    }

    pub fn interface_power(
        &self,
    ) -> Result<DifferentialResponseFor<J, RawInterfacePower<J>>, InterfaceProjectionError>
    where
        J: RealScalarAlgebra,
        J::Policy: DerivativePartsPolicy<RawInterfacePower<J>>,
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
        W::Entries: ProjectAmplitudes<Amplitudes = PlaneWaveAmplitudes<J>>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
        J::RealJet: ScalarAlgebra,
        J::Scalar: ComplexScalar + One + Zero,
        <J::RealJet as Jet>::Scalar: One + Neg<Output = <J::RealJet as Jet>::Scalar>,
        RawInterfacePower<J>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        Ok(self
            .state
            .raw_interface_power_unchecked(self.incident_side)?
            .into_differential_response(&J::Policy::default(), self.state.mapping()))
    }

    pub fn layer_power(
        &self,
    ) -> Result<DifferentialResponseFor<J, RawLayerPower<J>>, InterfaceProjectionError>
    where
        J: RealScalarAlgebra,
        J::Policy: DerivativePartsPolicy<RawLayerPower<J>>,
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
        W::Entries: ProjectAmplitudes<Amplitudes = PlaneWaveAmplitudes<J>>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
        J::RealJet: ScalarAlgebra,
        J::Scalar: ComplexScalar + One + Zero,
        <J::RealJet as Jet>::Scalar: One + Neg<Output = <J::RealJet as Jet>::Scalar>,
        RawLayerPower<J>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        Ok(self
            .state
            .raw_layer_power_unchecked(self.incident_side)?
            .into_differential_response(&J::Policy::default(), self.state.mapping()))
    }

    pub fn layer_dissipation(
        &self,
    ) -> Result<
        DifferentialResponseFor<J, Layers<LayerDissipation<J::RealJet>>>,
        LayerProjectionError,
    >
    where
        J: RealScalarAlgebra + ScalarAlgebraExpRelExt,
        J::RealJet: ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::RealJet as Jet>::Scalar: One,
        J::Policy: DerivativePartsPolicy<Layers<LayerDissipation<J::RealJet>>>,
        Layers<LayerDissipation<J::RealJet>>: IntoDifferentialResponse<J::Policy, J::Mapping>,
        W: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        Ok(self
            .state
            .raw_layer_dissipation_unchecked(self.incident_side)?
            .into_differential_response(&J::Policy::default(), self.state.mapping()))
    }

    pub fn layer_energy_nondispersive(
        &self,
    ) -> Result<DifferentialResponseFor<J, Layers<LayerEnergy<J::RealJet>>>, LayerEnergyError>
    where
        J: ComplexJet + RealScalarAlgebra + ScalarAlgebraExpRelExt + ConstitutiveLift<RealAxis, M>,
        J::RealJet: ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::RealJet as Jet>::Scalar: One + FromPrimitive,
        J::Policy: DerivativePartsPolicy<Layers<LayerEnergy<J::RealJet>>>,
        Layers<LayerEnergy<J::RealJet>>: IntoDifferentialResponse<J::Policy, J::Mapping>,
        W: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
        RealAxis: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        Ok(self
            .state
            .raw_nondispersive_layer_energy_unchecked(self.incident_side)?
            .into_differential_response(&J::Policy::default(), self.state.mapping()))
    }

    pub fn layer_participation_nondispersive(
        &self,
    ) -> Result<
        DifferentialResponseFor<J, Layers<LayerParticipation<J::RealJet>>>,
        LayerParticipationError,
    >
    where
        J: ComplexJet + RealScalarAlgebra + ScalarAlgebraExpRelExt + ConstitutiveLift<RealAxis, M>,
        J::RealJet: ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::RealJet as Jet>::Scalar: One + FromPrimitive,
        J::Policy: DerivativePartsPolicy<Layers<LayerParticipation<J::RealJet>>>,
        Layers<LayerParticipation<J::RealJet>>: IntoDifferentialResponse<J::Policy, J::Mapping>,
        W: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
        RealAxis: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        Ok(self
            .state
            .raw_nondispersive_layer_energy_unchecked(self.incident_side)?
            .participation()?
            .into_differential_response(&J::Policy::default(), self.state.mapping()))
    }

    pub fn layer_confinement_by_nondispersive(
        &self,
        mut include: impl FnMut(FiniteLayerIndex) -> bool,
    ) -> Result<DifferentialResponseFor<J, EnergyConfinement<J::RealJet>>, LayerConfinementError>
    where
        J: ComplexJet + RealScalarAlgebra + ScalarAlgebraExpRelExt + ConstitutiveLift<RealAxis, M>,
        J::RealJet: ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::RealJet as Jet>::Scalar: One + FromPrimitive,
        J::Policy: DerivativePartsPolicy<EnergyConfinement<J::RealJet>>,
        EnergyConfinement<J::RealJet>: IntoDifferentialResponse<J::Policy, J::Mapping>,
        W: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
        RealAxis: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        Ok(self
            .state
            .raw_nondispersive_layer_energy_unchecked(self.incident_side)?
            .confinement_by(|index, _| include(index))?
            .into_differential_response(&J::Policy::default(), self.state.mapping()))
    }

    pub fn layer_energy_dispersive(
        &self,
    ) -> Result<DifferentialResponseFor<J, Layers<LayerEnergy<J::RealJet>>>, LayerEnergyError>
    where
        J: ComplexJet
            + RealScalarAlgebra
            + ScalarAlgebraExpRelExt
            + ConstitutiveSpectralFirstLift<RealAxis, M>,
        J::RealJet: ScalarAlgebra,
        J::Policy: DerivativePartsPolicy<Layers<LayerEnergy<J::RealJet>>>,
        Layers<LayerEnergy<J::RealJet>>: IntoDifferentialResponse<J::Policy, J::Mapping>,
        J::Scalar: ComplexScalar,
        <J::RealJet as Jet>::Scalar: FromPrimitive + One,
        J::Dimension: Dimension,
        RealAxis: ConstitutiveDerivativeEvaluator<J::Scalar, J::Dimension, M>,
        W: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        Ok(self
            .state
            .raw_dispersive_layer_energy_unchecked(self.incident_side)?
            .into_differential_response(&J::Policy::default(), self.state.mapping()))
    }

    pub fn layer_participation_dispersive(
        &self,
    ) -> Result<
        DifferentialResponseFor<J, Layers<LayerParticipation<J::RealJet>>>,
        LayerParticipationError,
    >
    where
        J: ComplexJet
            + RealScalarAlgebra
            + ScalarAlgebraExpRelExt
            + ConstitutiveSpectralFirstLift<RealAxis, M>,
        J::RealJet: ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::RealJet as Jet>::Scalar: One + FromPrimitive,
        J::Policy: DerivativePartsPolicy<Layers<LayerParticipation<J::RealJet>>>,
        Layers<LayerParticipation<J::RealJet>>: IntoDifferentialResponse<J::Policy, J::Mapping>,
        W: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
        RealAxis: ConstitutiveDerivativeEvaluator<J::Scalar, J::Dimension, M>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        Ok(self
            .state
            .raw_dispersive_layer_energy_unchecked(self.incident_side)?
            .participation()?
            .into_differential_response(&J::Policy::default(), self.state.mapping()))
    }

    pub fn layer_confinement_by_dispersive(
        &self,
        mut include: impl FnMut(FiniteLayerIndex) -> bool,
    ) -> Result<DifferentialResponseFor<J, EnergyConfinement<J::RealJet>>, LayerConfinementError>
    where
        J: ComplexJet
            + RealScalarAlgebra
            + ScalarAlgebraExpRelExt
            + ConstitutiveSpectralFirstLift<RealAxis, M>,
        J::RealJet: ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::RealJet as Jet>::Scalar: One + FromPrimitive,
        J::Policy: DerivativePartsPolicy<EnergyConfinement<J::RealJet>>,
        EnergyConfinement<J::RealJet>: IntoDifferentialResponse<J::Policy, J::Mapping>,
        W: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
        RealAxis: ConstitutiveDerivativeEvaluator<J::Scalar, J::Dimension, M>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        Ok(self
            .state
            .raw_dispersive_layer_energy_unchecked(self.incident_side)?
            .confinement_by(|index, _| include(index))?
            .into_differential_response(&J::Policy::default(), self.state.mapping()))
    }

    pub fn evaluate_fields(
        self,
        sampling: &FieldSampling<R>,
    ) -> Result<
        DifferentialResponseFor<
            J,
            ElectromagneticFields<<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector>,
        >,
        FieldReconstructionError<<J::Scalar as ComplexField>::RealField>,
    >
    where
        J: JetStack + ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::Scalar as ComplexField>::RealField: Float + FromPrimitive,
        J::Stacked: CartesianScalarAlgebra,
        J::Policy: DerivativePartsPolicy<
            ElectromagneticFields<<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector>,
        >,
        ElectromagneticFields<<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector>:
            IntoDifferentialResponse<J::Policy, J::Mapping>,
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
        W::Entries: ProjectAmplitudes,
        <W::Entries as ProjectAmplitudes>::Amplitudes: Amplitudes<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        let wave_context = WaveSamplingContext::new(self.state.workspace());

        let boundary_waves = wave_context.driven_boundary_waves(self.incident_side)?;

        let context = FieldSamplingContext::new(self.state.workspace());

        let sampling = sampling.resolve(self.state.stack())?.compile();

        let reconstructed = context.reconstruct_from_boundary_waves(&boundary_waves, &sampling)?;

        Ok(reconstructed.into_differential_response(&J::Policy::default(), self.state.mapping()))
    }
}

impl<'a, J, I, M, W> PlaneWaveExcitation<'a, J, I, M, W>
where
    J: Jet<Dimension = Ix0> + JetMapping + PartialEq,
    J::Scalar: ComplexField,
    I: ComplexField,
    J::Mapping: PartialEq,
    W: RetainedIsotropicLayers<Algebra = J>,
{
    /// Form a validated pair with another scalar excitation.
    ///
    /// This excitation becomes the reference operand. Hermitian contractions
    /// conjugate it; bilinear contractions do not.
    pub fn pair_with<M2, W2>(
        self,
        comparison: PlaneWaveExcitation<'a, J, I, M2, W2>,
    ) -> Result<PlaneWaveExcitationPair<'a, J, I, M, M2, W, W2>, PlaneWavePairError>
    where
        W2: RetainedIsotropicLayers<Algebra = J>,
    {
        PlaneWaveExcitationPair::new(self, comparison)
    }
}
