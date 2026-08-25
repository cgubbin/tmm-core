use std::ops::Neg;

use nalgebra::ComplexField;
use ndarray::{Dimension, Ix0};
use num_traits::{Float, FromPrimitive, One, Zero};

use crate::{
    ComplexScalar, ConstitutiveFields, ElectromagneticDissipation, ElectromagneticFields, ElectromagneticIntensities, FiniteLayerIndex, IncidentSide, InterfacePower, LayerDissipation, LayerPower, PlaneWaveAmplitudes, RealAxis, algebra::{
        CartesianScalarAlgebra, ComplexJet, Jet, JetStack, RealCartesianVectorAlgebra,
        RealScalarAlgebra, ScalarAlgebra, ScalarAlgebraExpRelExt, ScaleBy,
    }, backend::{
        ExteriorContextProvider, PlaneWaveEntries, PlaneWaveSolutionSource, RetainedIsotropicLayers,
    }, derivative_parts::DerivativePartsPolicy, differential::IntoDifferentialResponse, 
    input::JetMapping, material::{
        ConstitutiveEvaluator, ConstitutiveLift, ConstitutiveSpectralFirstLift,
        ConstitutiveDerivativeEvaluator,
    }, observable::{
        Amplitudes, BoundaryProjectionError, ConstitutiveFieldReconstructionError, ConstitutiveSamplingError, ElectromagneticEnergy, EnergyConfinement, FieldReconstructionError, FieldSamplingContext, InterfaceProjectionError, InterfaceStates, InterfaceWaveData, Interfaces, LayerBoundaries, LayerBoundaryStates, LayerBoundaryWaves, LayerConfinementError, LayerEnergy, LayerEnergyError, LayerIntegrationInput, LayerParticipation, LayerParticipationError, LayerProjectionError, Layers, ProjectAmplitudes, ProjectPower, assemble_layer_integration_inputs, electromagnetic_dissipation_coefficients
    }, spatial::{FieldSampling, ResolvedFieldSampling, SpatialResponse}, waves::{ReconstructLayerBoundaryWaves, WaveSamplingContext}
};

use super::{RealAxisExcitationPair, RealAxisPairError, RealAxisState,
        state::{RawInterfacePower, RawLayerPower},
    query::{PlaneWaveQuery, RawPower,RawAmplitudes, DifferentialResponseFor, RealAxisExternalQueries}
};

#[derive(Debug, Copy, Clone)]
pub struct RealAxisExcitation<'a, J, M, W>
where
    J: Jet + JetMapping + ComplexJet,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    <J::Scalar as ComplexField>::RealField: ComplexField,
{
    state: &'a RealAxisState<J, M, W>,
    incident_side: IncidentSide,
    amplitude_scale: J::RealJet,
}

impl<'a, J, M, W> RealAxisExcitation<'a, J, M, W>
where
    J: Jet + JetMapping + ComplexJet,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    <J::Scalar as ComplexField>::RealField: ComplexField,
{
    /// Construct an excitation after validating the state's projection constraint.
    pub(crate) fn new(state: &'a RealAxisState<J, M, W>, incident_side: IncidentSide) -> Self
    where
        J: ComplexJet + ScalarAlgebra + RealScalarAlgebra,
        J::RealJet: ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::RealJet as Jet>::Scalar: FromPrimitive,
        W: PlaneWaveSolutionSource,
        W::Entries: PlaneWaveEntries,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        let exterior = state.workspace().solution().context();

        let k0 = exterior.vacuum_angular_wavenumber();

        let admittance = match incident_side {
            IncidentSide::Left => exterior.left_admittance(),
            IncidentSide::Right => exterior.right_admittance(),
        };

        let incident_flux = admittance
            .real()
            .divide(&k0.real())
            .scale(<J::RealJet as Jet>::Scalar::from_f64(0.5).unwrap());

        let amplitude_scale = incident_flux.sqrt().reciprocal();

        Self {
            state,
            incident_side,
            amplitude_scale,
        }
    }

    pub(crate) fn state(&self) -> &'a RealAxisState<J, M, W> {
        self.state
    }

    pub fn incident_side(&self) -> IncidentSide {
        self.incident_side
    }

    pub fn amplitude_scale(&self) -> &J::RealJet {
        &self.amplitude_scale
    }

    fn normalised_boundary_waves(
        &self,
    ) -> Result<LayerBoundaries<LayerBoundaryWaves<J>>, BoundaryProjectionError>
    where
        J: ScalarAlgebra,
        J::RealJet: Clone,
        W: ReconstructLayerBoundaryWaves<Algebra = J>,
    {
        Ok(self
            .state
            .raw_layer_boundary_waves_unchecked(self.incident_side)?
            .scale_by(&J::into_complex(self.amplitude_scale.clone())))
    }

    fn normalised_boundary_states(
        &self,
    ) -> Result<LayerBoundaries<LayerBoundaryStates<J>>, BoundaryProjectionError>
    where
        J: ScalarAlgebra,
        J::RealJet: Clone,
        J::Scalar: ComplexScalar,
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
    {
        Ok(self
            .state
            .raw_layer_boundary_states_unchecked(self.incident_side)?
            .scale_by(&J::into_complex(self.amplitude_scale.clone())))
    }


    fn normalised_interface_states(
        &self,
    ) -> Result<Interfaces<InterfaceStates<J>>, BoundaryProjectionError>
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
        J::RealJet: Clone,
    {
        Ok(self
            .state
            .raw_interface_states_unchecked(self.incident_side)?
            .scale_by(&J::into_complex(self.amplitude_scale.clone())))
    }


    fn normalised_interface_wave_data(
        &self,
    ) -> Result<Interfaces<InterfaceWaveData<J>>, InterfaceProjectionError>
    where
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
        W::Entries: ProjectAmplitudes<Amplitudes = PlaneWaveAmplitudes<J>>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
        J: ScalarAlgebra + Clone,
        J::Scalar: ComplexScalar + One + Zero,
        J::Dimension: Dimension,
        J::RealJet: Clone,
    {
        Ok(self.state.raw_interface_wave_data_unchecked(self.incident_side)?
            .scale_by(&J::into_complex(self.amplitude_scale.clone())))
    }

    fn normalised_layer_integration_inputs(
        &self,
    ) -> Result<Layers<LayerIntegrationInput<J>>, LayerProjectionError>
    where
        W: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
        J: ScalarAlgebra,
        J::RealJet: Clone,
    {
        let boundary_waves = self.normalised_boundary_waves()?;

        assemble_layer_integration_inputs(self.state.workspace(), boundary_waves)
    }

    pub fn boundary_waves(
        &self,
    ) -> Result<
        DifferentialResponseFor<J, LayerBoundaries<LayerBoundaryWaves<J>>>,
        BoundaryProjectionError,
    >
    where
        J: ScalarAlgebra,
        J::RealJet: Clone,
        J::Policy: Default + DerivativePartsPolicy<LayerBoundaries<LayerBoundaryWaves<J>>>,
        W: PlaneWaveSolutionSource + ReconstructLayerBoundaryWaves<Algebra = J>,
        LayerBoundaries<LayerBoundaryWaves<J>>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        Ok(self
            .normalised_boundary_waves()?
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
        J::RealJet: Clone,
        J::Policy: Default + DerivativePartsPolicy<LayerBoundaries<LayerBoundaryStates<J>>>,
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
        LayerBoundaries<LayerBoundaryStates<J>>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        Ok(self
            .normalised_boundary_states()?
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
        J::RealJet: Clone,
        J::Policy: Default + DerivativePartsPolicy<Interfaces<InterfaceStates<J>>>,
        Interfaces<InterfaceStates<J>>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        Ok(self
            .normalised_interface_states()?
            .into_differential_response(&J::Policy::default(), self.state.mapping()))
    }
}

// Real Input Observables
impl<'a, J, M, W> RealAxisExcitation<'a, J, M, W>
where
    J: Jet + JetMapping + ComplexJet,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    J::Policy: Default,
    <J::Scalar as ComplexField>::RealField: ComplexField,
    W: PlaneWaveSolutionSource,
{
    fn normalised_interface_power(
        &self,
    ) -> Result<Interfaces<InterfacePower<J::RealJet>>, InterfaceProjectionError>
    where
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
        W::Entries: ProjectAmplitudes<Amplitudes = PlaneWaveAmplitudes<J>>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
        J: RealScalarAlgebra + Clone,
        J::RealJet: ScalarAlgebra,
        J::Scalar: ComplexScalar + One + Zero,
        J::Dimension: Dimension,
        <J::RealJet as Jet>::Scalar: One + Neg<Output = <J::RealJet as Jet>::Scalar> + FromPrimitive,
    {
        let interface_data = self.normalised_interface_wave_data()?;


        Ok(interface_data.into_power(self.state.problem().coordinates().vacuum_angular_wavenumber()))
    }

    fn normalised_layer_power(
        &self,
    ) -> Result<Layers<LayerPower<J::RealJet>>, InterfaceProjectionError>
    where
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
        W::Entries: ProjectAmplitudes<Amplitudes = PlaneWaveAmplitudes<J>>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
        J: RealScalarAlgebra + Clone,
        J::RealJet: ScalarAlgebra + Clone,
        J::Scalar: ComplexScalar + One + Zero,
        J::Dimension: Dimension,
        <J::RealJet as Jet>::Scalar: One + Neg<Output = <J::RealJet as Jet>::Scalar> + FromPrimitive,
    {
        Ok(self.normalised_interface_power()?
            .into_layer_power())
    }


    fn normalised_layer_dissipation(
        &self,
    ) -> Result<Layers<LayerDissipation<J::RealJet>>, LayerProjectionError>
    where
        J: RealScalarAlgebra + ScalarAlgebraExpRelExt + Clone,
        J::RealJet: ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::RealJet as Jet>::Scalar: FromPrimitive+ One,
        W: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        let layers = self
            .normalised_layer_integration_inputs()?
            .integrate_hermitian();


        let coordinates = self.state.problem().coordinates();

        Ok(layers.into_dissipation(
            coordinates.vacuum_angular_wavenumber(),
            coordinates.parallel_angular_wavenumber(),
        ))
    }


    fn normalised_nondispersive_layer_energy(
        &self,
    ) -> Result<Layers<LayerEnergy<J::RealJet>>, LayerEnergyError>
    where
        J: ComplexJet
            + RealScalarAlgebra
            + ScalarAlgebraExpRelExt
            + ConstitutiveLift<RealAxis, M>
            + Clone,
        J::RealJet: ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::RealJet as Jet>::Scalar: One + FromPrimitive,
        W: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
        RealAxis: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        let integrated = self
            .normalised_layer_integration_inputs()?
            .integrate_hermitian();

        let problem = self.state.problem();
        let coordinates = problem.coordinates();


        Ok(integrated.into_nondispersive_energy(
            coordinates.vacuum_angular_wavenumber(),
            coordinates.parallel_angular_wavenumber(),
        ))
    }


    fn normalised_dispersive_layer_energy(
        &self,
    ) -> Result<Layers<LayerEnergy<J::RealJet>>, LayerEnergyError>
    where
        J: ComplexJet
            + RealScalarAlgebra
            + ScalarAlgebraExpRelExt
            + ConstitutiveSpectralFirstLift<RealAxis, M>
            + Clone,
        J::RealJet: ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::RealJet as Jet>::Scalar: FromPrimitive + One,
        J::Dimension: Dimension,
        RealAxis: ConstitutiveDerivativeEvaluator<J::Scalar, J::Dimension, M>,
        W: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        let coordinates = self.state.problem().coordinates();


        let sequence = self
            .normalised_layer_integration_inputs()?
            .integrate_hermitian()
            .into_brillouin_layers(
                self.state.problem()
                    .stack()
                    .layers()
                    .iter()
                    .map(|layer| layer.material()),
                coordinates.vacuum_angular_wavenumber(),
            )?;

        Ok(sequence.into_brillouin_energy(
            coordinates.vacuum_angular_wavenumber(),
            coordinates.parallel_angular_wavenumber(),
        ))
    }

    pub fn amplitudes(
        &self,
    ) -> DifferentialResponseFor<J, RawAmplitudes<RealAxisState<J, M, W>, J>>
    where
        W::Entries: ProjectAmplitudes,
        J::Policy: DerivativePartsPolicy<RawAmplitudes<RealAxisState<J, M, W>, J>>,
        RawAmplitudes<RealAxisState<J, M, W>, J>:
            IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        self.state
            .raw_amplitudes(self.incident_side)
            .into_differential_response(&J::Policy::default(), self.state.mapping())
    }

    pub fn power(&self) -> DifferentialResponseFor<J, RawPower<RealAxisState<J, M, W>, J>>
    where
        W::Entries: ProjectPower,
        J::Policy: DerivativePartsPolicy<RawPower<RealAxisState<J, M, W>, J>>,
        RawPower<RealAxisState<J, M, W>, J>: IntoDifferentialResponse<J::Policy, J::Mapping>,
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
        <J::RealJet as Jet>::Scalar: One + Neg<Output = <J::RealJet as Jet>::Scalar> + FromPrimitive,
        RawInterfacePower<J>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        Ok(self
            .normalised_interface_power()?
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
        <J::RealJet as Jet>::Scalar: One + Neg<Output = <J::RealJet as Jet>::Scalar> + FromPrimitive,
        RawLayerPower<J>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        Ok(self
            .normalised_layer_power()?
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
        <J::RealJet as Jet>::Scalar: FromPrimitive  +One,
        J::Policy: DerivativePartsPolicy<Layers<LayerDissipation<J::RealJet>>>,
        Layers<LayerDissipation<J::RealJet>>: IntoDifferentialResponse<J::Policy, J::Mapping>,
        W: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        Ok(self
            .normalised_layer_dissipation()?
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
            .normalised_nondispersive_layer_energy()?
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
            .normalised_nondispersive_layer_energy()?
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
            .normalised_nondispersive_layer_energy()?
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
            .normalised_dispersive_layer_energy()?
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
            .normalised_dispersive_layer_energy()?
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
            .normalised_dispersive_layer_energy()?
            .confinement_by(|index, _| include(index))?
            .into_differential_response(&J::Policy::default(), self.state.mapping()))
    }

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
        J::RealJet: Clone,
        <J::Scalar as ComplexField>::RealField: Float + FromPrimitive,
        J::Stacked: CartesianScalarAlgebra,
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
        W::Entries: ProjectAmplitudes,
        <W::Entries as ProjectAmplitudes>::Amplitudes: Amplitudes<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        let wave_context = WaveSamplingContext::new(self.state.workspace());

        let scale =
        J::into_complex(self.amplitude_scale.clone());

    let boundary_waves = wave_context
        .driven_boundary_waves(self.incident_side)?
        .scale_by(&scale);

        let context = FieldSamplingContext::new(self.state.workspace());

        let compiled_sampling = sampling.compile();

        context.reconstruct_from_boundary_waves(&boundary_waves, &compiled_sampling)
    }

    pub fn evaluate_fields(
        &self,
        sampling: &FieldSampling<<J::Scalar as ComplexField>::RealField>,
    ) -> Result<
        PlaneWaveFieldResponse<J>,
        FieldReconstructionError<<J::Scalar as ComplexField>::RealField>,
    >
    where
        J: JetStack + ScalarAlgebra,
        J::Scalar: ComplexScalar,
        J::RealJet: Clone,
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
        let sampling = sampling.resolve(self.state.stack())?;
        let reconstructed = self.raw_electromagnetic_fields(&sampling)?;

        let differential_response =
            reconstructed.into_differential_response(&J::Policy::default(), self.state.mapping());

        Ok(SpatialResponse::new(differential_response, sampling))
    }

    pub fn evaluate_field_intensities(
        &self,
        sampling: &FieldSampling<<J::Scalar as ComplexField>::RealField>,
    ) -> Result<
        PlaneWaveIntensityResponse<J>,
        FieldReconstructionError<<J::Scalar as ComplexField>::RealField>,
    >
    where
        J: JetStack + ScalarAlgebra,
        J::Scalar: ComplexScalar,
        J::RealJet: Clone,
        <J::Scalar as ComplexField>::RealField: Float + FromPrimitive,
        J::Stacked: CartesianScalarAlgebra,
        <J::Stacked as CartesianScalarAlgebra>::Vector: RealCartesianVectorAlgebra,
        J::Policy: DerivativePartsPolicy<
                ElectromagneticIntensities<
                    <<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector as RealCartesianVectorAlgebra>::RealScalarAlgebra,
                >
        >,
        ElectromagneticIntensities<
        <<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector as RealCartesianVectorAlgebra>::RealScalarAlgebra,
        >:
            IntoDifferentialResponse<J::Policy, J::Mapping>,
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
        W::Entries: ProjectAmplitudes,
        <W::Entries as ProjectAmplitudes>::Amplitudes: Amplitudes<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        let sampling = sampling.resolve(self.state.stack())?;
        let reconstructed = self
            .raw_electromagnetic_fields(&sampling)?
            .into_magnitude_squared();

        let differential_response =
            reconstructed.into_differential_response(&J::Policy::default(), self.state.mapping());

        Ok(SpatialResponse::new(differential_response, sampling))
    }

    pub fn evaluate_complex_poynting_vector(
        &self,
        sampling: &FieldSampling<<J::Scalar as ComplexField>::RealField>,
    ) -> Result<
        PlaneWaveComplexPoyntingVectorResponse<J>,
        FieldReconstructionError<<J::Scalar as ComplexField>::RealField>,
    >
    where
        J: JetStack + ScalarAlgebra,
        J::Scalar: ComplexScalar,
        J::RealJet: Clone,
        <J::Scalar as ComplexField>::RealField: Float + FromPrimitive,
        J::Stacked: CartesianScalarAlgebra,
        <J::Stacked as CartesianScalarAlgebra>::Vector: RealCartesianVectorAlgebra,
        J::Policy:
            DerivativePartsPolicy<<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector>,
        <<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector:
            IntoDifferentialResponse<J::Policy, J::Mapping>,
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
        W::Entries: ProjectAmplitudes,
        <W::Entries as ProjectAmplitudes>::Amplitudes: Amplitudes<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        let sampling = sampling.resolve(self.state.stack())?;
        let reconstructed = self
            .raw_electromagnetic_fields(&sampling)?
            .complex_poynting_vector();

        let differential_response =
            reconstructed.into_differential_response(&J::Policy::default(), self.state.mapping());

        Ok(SpatialResponse::new(differential_response, sampling))
    }

    pub fn evaluate_time_averaged_poynting_vector(
        &self,
        sampling: &FieldSampling<<J::Scalar as ComplexField>::RealField>,
    ) -> Result<
        PlaneWaveTimeAveragedPoyntingVectorResponse<J>,
        FieldReconstructionError<<J::Scalar as ComplexField>::RealField>,
    >
    where
        J: JetStack + ScalarAlgebra,
        J::Scalar: ComplexScalar,
        J::RealJet: Clone,
        <J::Scalar as ComplexField>::RealField: Float + FromPrimitive,
        J::Stacked: CartesianScalarAlgebra,
        <J::Stacked as CartesianScalarAlgebra>::Vector: RealCartesianVectorAlgebra,
        J::Policy:
            DerivativePartsPolicy<
                <<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector as RealCartesianVectorAlgebra>::RealVector
        >,
        <<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector as RealCartesianVectorAlgebra>::RealVector:
            IntoDifferentialResponse<J::Policy, J::Mapping>,
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
        W::Entries: ProjectAmplitudes,
        <W::Entries as ProjectAmplitudes>::Amplitudes: Amplitudes<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        let sampling = sampling.resolve(self.state.stack())?;
        let reconstructed = self
            .raw_electromagnetic_fields(&sampling)?
            .time_averaged_poynting_vector();

        let differential_response =
            reconstructed.into_differential_response(&J::Policy::default(), self.state.mapping());

        Ok(SpatialResponse::new(differential_response, sampling))
    }

    pub fn evaluate_constitutive_fields(
        &self,
        sampling: &FieldSampling<<J::Scalar as ComplexField>::RealField>,
    ) -> Result<
        ConstitutiveFieldResponse<J>,
        ConstitutiveFieldReconstructionError<<J::Scalar as ComplexField>::RealField>,
    >
    where
        J: JetStack + ScalarAlgebra,
        J::Scalar: ComplexScalar,
        J::RealJet: Clone,
        <J::Scalar as ComplexField>::RealField: Float + FromPrimitive,
        J::Stacked: CartesianScalarAlgebra,
        J::Policy: DerivativePartsPolicy<
            ConstitutiveFields<<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector>,
        >,
        ConstitutiveFields<<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector>:
            IntoDifferentialResponse<J::Policy, J::Mapping>,
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
        W::Entries: ProjectAmplitudes,
        <W::Entries as ProjectAmplitudes>::Amplitudes: Amplitudes<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        let sampling = sampling.resolve(self.state.stack())?;
        let electromagnetic_fields = self.raw_electromagnetic_fields(&sampling)?;

        let constitutive = self.state.raw_constitutive_parameters(&sampling)?;

        let reconstructed = electromagnetic_fields.into_constitutive_fields(&constitutive);

        let differential_response =
            reconstructed.into_differential_response(&J::Policy::default(), self.state.mapping());

        Ok(SpatialResponse::new(differential_response, sampling))
    }


    pub fn evaluate_dissipation_density(
        &self,
        sampling: &FieldSampling<<J::Scalar as ComplexField>::RealField>,
    ) -> Result<
        ElectromagneticDissipationResponse<J>,
        ConstitutiveFieldReconstructionError<<J::Scalar as ComplexField>::RealField>,
    >
    where
        J: JetStack + ScalarAlgebra + ComplexJet,
        J::Scalar: ComplexScalar,
        J::RealJet: Jet + Clone,
        <J::Scalar as ComplexField>::RealField: Float + FromPrimitive,
        J::Stacked: CartesianScalarAlgebra + ComplexJet + RealScalarAlgebra,
        <J::Stacked as CartesianScalarAlgebra>::Vector: RealCartesianVectorAlgebra<RealScalarAlgebra =  <<J as JetStack>::Stacked as ComplexJet>::RealJet>,
        J::Policy: DerivativePartsPolicy<
                ElectromagneticDissipation<
                    <<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector as RealCartesianVectorAlgebra>::RealScalarAlgebra,
                >
        >,
        ElectromagneticDissipation<
        <<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector as RealCartesianVectorAlgebra>::RealScalarAlgebra,
        >:
            IntoDifferentialResponse<J::Policy, J::Mapping>,
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
        W::Entries: ProjectAmplitudes,
        <W::Entries as ProjectAmplitudes>::Amplitudes: Amplitudes<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
        <<<J as JetStack>::Stacked as ComplexJet>::RealJet as Jet>::Scalar: num_traits::Float,
        <<J as JetStack>::Stacked as ComplexJet>::RealJet: ScalarAlgebra,
    {
        let sampling = sampling.resolve(self.state.stack())?;
        let electromagnetic_intensities = self
            .raw_electromagnetic_fields(&sampling)?
            .into_magnitude_squared();

        let constitutive = self.state.raw_constitutive_parameters(&sampling)?;

        let k0 = J::stack(
            std::iter::repeat_n(
                self.state
                    .problem()
                    .coordinates()
                    .vacuum_angular_wavenumber()
                    .clone(),
                sampling.len(),
            )
            .collect(),
        )
        .map_err(ConstitutiveSamplingError::Shape)?
        .ok_or(ConstitutiveSamplingError::EmptySampling)
        .map_err(ConstitutiveFieldReconstructionError::Constitutive)?;

        let coefficients = electromagnetic_dissipation_coefficients(&constitutive, &k0);

        let reconstructed = electromagnetic_intensities.into_dissipation(&coefficients);

        let differential_response =
            reconstructed.into_differential_response(&J::Policy::default(), self.state.mapping());

        Ok(SpatialResponse::new(differential_response, sampling))
    }

    pub fn evaluate_energy_density(
        &self,
        sampling: &FieldSampling<<J::Scalar as ComplexField>::RealField>,
    ) -> Result<
        ElectromagneticEnergyResponse<J>,
        ConstitutiveFieldReconstructionError<<J::Scalar as ComplexField>::RealField>,
    >
    where
        J: JetStack + ScalarAlgebra,
        J::RealJet: Clone,
        J::Scalar: ComplexScalar,
        <J::Scalar as ComplexField>::RealField: Float + FromPrimitive,
        J::Stacked: CartesianScalarAlgebra + RealScalarAlgebra,
        <J::Stacked as CartesianScalarAlgebra>::Vector: RealCartesianVectorAlgebra<RealScalarAlgebra =  <<J as JetStack>::Stacked as ComplexJet>::RealJet>,
        <<J::Stacked as CartesianScalarAlgebra>::Vector as RealCartesianVectorAlgebra>::RealScalarAlgebra: ScalarAlgebra,
        J::Policy: DerivativePartsPolicy<
                ElectromagneticEnergy<
                    <<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector as RealCartesianVectorAlgebra>::RealScalarAlgebra,
                >
        >,
        ElectromagneticEnergy<
        <<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector as RealCartesianVectorAlgebra>::RealScalarAlgebra,
        >:
            IntoDifferentialResponse<J::Policy, J::Mapping>,
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
        W::Entries: ProjectAmplitudes,
        <W::Entries as ProjectAmplitudes>::Amplitudes: Amplitudes<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
        RealAxis: ConstitutiveDerivativeEvaluator<J::Scalar, J::Dimension, M>,
        J: ConstitutiveSpectralFirstLift<RealAxis, M>,
        <<<J as JetStack>::Stacked as ComplexJet>::RealJet as Jet>::Scalar: num_traits::Float,
        <<J as JetStack>::Stacked as ComplexJet>::RealJet: ScalarAlgebra,
    {
        let sampling = sampling.resolve(self.state.stack())?;
        let electromagnetic_intensities = self
            .raw_electromagnetic_fields(&sampling)?
            .into_magnitude_squared();

        let coefficients = self
            .state
            .raw_constitutive_spectral_first_parameters::<RealAxis>(&sampling)?
            .into_brillouin_factors()
            .into_hermitian_energy_coefficients();

        let reconstructed = electromagnetic_intensities.into_energy(&coefficients);

        let differential_response =
            reconstructed.into_differential_response(&J::Policy::default(), self.state.mapping());

        Ok(SpatialResponse::new(differential_response, sampling))
    }
}

impl<'a, J, M, W> RealAxisExcitation<'a, J, M, W>
where
    J: Jet<Dimension = Ix0> + JetMapping + PartialEq + ComplexJet,
    J::Scalar: ComplexField,
    J::RealJet: std::fmt::Debug,
    <J::Scalar as ComplexField>::RealField: ComplexField,
    J::Mapping: PartialEq,
    W: RetainedIsotropicLayers<Algebra = J>,
{
    /// Form a validated pair with another scalar excitation.
    ///
    /// This excitation becomes the reference operand. Hermitian contractions
    /// conjugate it; bilinear contractions do not.
    pub fn pair_with<M2, W2>(
        self,
        comparison: RealAxisExcitation<'a, J, M2, W2>,
    ) -> Result<RealAxisExcitationPair<'a, J, M, M2, W, W2>, RealAxisPairError>
    where
        W2: RetainedIsotropicLayers<Algebra = J>,
    {
        RealAxisExcitationPair::new(self, comparison)
    }
}

pub type PlaneWaveFieldResponse<J> = SpatialResponse<
    DifferentialResponseFor<
        J,
        ElectromagneticFields<<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector>,
    >,
    <<J as Jet>::Scalar as ComplexField>::RealField,
>;

pub type ConstitutiveFieldResponse<J> = SpatialResponse<
    DifferentialResponseFor<
        J,
        ConstitutiveFields<<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector>,
    >,
    <<J as Jet>::Scalar as ComplexField>::RealField,
>;

pub type PlaneWaveIntensityResponse<J> = SpatialResponse<
    DifferentialResponseFor<
        J,
        ElectromagneticIntensities<
            <<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector as RealCartesianVectorAlgebra>::RealScalarAlgebra
        >,
    >,
    <<J as Jet>::Scalar as ComplexField>::RealField,
>;

pub type PlaneWaveComplexPoyntingVectorResponse<J> = SpatialResponse<
    DifferentialResponseFor<J, <<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector>,
    <<J as Jet>::Scalar as ComplexField>::RealField,
>;

pub type PlaneWaveTimeAveragedPoyntingVectorResponse<J> = SpatialResponse<
    DifferentialResponseFor<J, 
        <<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector as RealCartesianVectorAlgebra>::RealVector
    >,
    <<J as Jet>::Scalar as ComplexField>::RealField,
>;

pub type ElectromagneticDissipationResponse<J> = SpatialResponse<
    DifferentialResponseFor<
        J,
        ElectromagneticDissipation<
            <<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector as RealCartesianVectorAlgebra>::RealScalarAlgebra
        >
    >,
    <<J as Jet>::Scalar as ComplexField>::RealField,
>;

pub type ElectromagneticEnergyResponse<J> = SpatialResponse<
    DifferentialResponseFor<
        J,
        ElectromagneticEnergy<
            <<<J as JetStack>::Stacked as CartesianScalarAlgebra>::Vector as RealCartesianVectorAlgebra>::RealScalarAlgebra
        >
    >,
    <<J as Jet>::Scalar as ComplexField>::RealField,
>;
