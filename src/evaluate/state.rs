use std::ops::Neg;

use nalgebra::ComplexField;
use ndarray::{Dimension, Ix0, NdIndex};
use num_traits::{FromPrimitive, One, Zero};

use crate::{
    ComplexPlane, ComplexScalar, IncidentSide, InterfacePower, LayerDissipation, LayerPower,
    PlaneWaveAmplitudes, Polarisation, RealAxis, Stack,
    algebra::{
        ComplexJet, Jet, JetStack, RealScalarAlgebra, ScalarAlgebra, ScalarAlgebraExpRelExt,
    },
    backend::{
        ExteriorContextProvider, ModalSolutionSource, PlaneWaveEntries, PlaneWaveSolutionSource,
        PlaneWaveSolutionView, ReconstructLayerModeWaves, RetainedIsotropicLayers,
    },
    derivative_parts::DerivativePartsPolicy,
    differential::IntoDifferentialResponse,
    evaluate::mode::{PlaneWaveMode, QnmCreationError},
    input::{
        CanonicalProblem, CompilationContext, JetMapping, ProjectionConstraint,
        ProjectionConstraintError,
    },
    material::{
        ConstitutiveEvaluator, ConstitutiveLift, ConstitutiveSpectralFirstLift,
        lifting::ConstitutiveDerivativeEvaluator,
    },
    observable::{
        AggregateBilinearNormalization, BoundaryProjectionError, ConstitutiveSamplingContext,
        ConstitutiveSamplingError, InterfaceProjectionError, InterfaceStates, InterfaceWaveData,
        Interfaces, IsotropicConstitutiveParameters, LayerBoundaries, LayerBoundaryStates,
        LayerBoundaryWaves, LayerEnergy, LayerEnergyError, LayerIntegrationInput,
        LayerProjectionError, Layers, ProjectAmplitudes, ProjectPlaneWaveModeDeterminant,
        assemble_interface_wave_data, assemble_layer_integration_inputs, exterior_boundary_states,
        exterior_boundary_waves, project_layer_admittances, project_layer_boundary_states,
        project_layer_boundary_waves,
    },
    projection::{JetPointProjection, PointProjectionError, ProjectPoint},
    spatial::ResolvedFieldSampling,
    waves::ReconstructLayerBoundaryWaves,
};

use super::{
    excitation::PlaneWaveExcitation,
    query::{
        DifferentialResponseFor, PlaneWaveExternalQueries, PlaneWaveQuery, RawModeDeterminant,
    },
};

/// A completed plane-wave evaluation.
///
/// The state retains:
///
/// - the canonical problem supplied to the backend;
/// - either the external solution or a retained backend workspace;
/// - the compiled caller-facing context, including the derivative mapping
///   associated with `J`.
///
/// Observable quantities remain jet-valued until queried. Query methods first
/// project the backend representation, then extract derivative parts, and
/// finally assemble a caller-facing
/// [`DifferentialResponse`](crate::differential::DifferentialResponse).
#[derive(Clone, Debug)]
pub struct PlaneWaveState<J, I, M, W>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    I: ComplexField,
{
    problem: CanonicalProblem<M, J>,
    workspace: W,
    context: CompilationContext<I, J::Dimension, J::Mapping>,
    stack: Stack<M, <J::Scalar as ComplexField>::RealField>,
    polarisation: Polarisation,
}

impl<J, I, M, W> PlaneWaveState<J, I, M, W>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    I: ComplexField,
{
    pub(crate) fn new(
        problem: CanonicalProblem<M, J>,
        workspace: W,
        context: CompilationContext<I, J::Dimension, J::Mapping>,
        stack: Stack<M, <J::Scalar as ComplexField>::RealField>,
        polarisation: Polarisation,
    ) -> Self {
        Self {
            problem,
            workspace,
            context,
            stack,
            polarisation,
        }
    }

    pub fn excitation(
        &self,
        incident_side: IncidentSide,
    ) -> Result<PlaneWaveExcitation<'_, J, I, M, W>, ProjectionConstraintError> {
        if let ProjectionConstraint::Fixed(side) = self.context().projection_constraint() {
            if side != incident_side {
                return Err(ProjectionConstraintError {
                    constraint: side,
                    requested: incident_side,
                });
            }
        }

        Ok(PlaneWaveExcitation::new(self, incident_side))
    }

    pub(crate) fn project_point<Idx>(
        &self,
        index: &Idx,
    ) -> Result<PlaneWaveState<J::PointJet, I, M, W::Point>, PointProjectionError>
    where
        J: JetPointProjection,
        J::PointJet: JetMapping<Mapping = J::Mapping>,
        CanonicalProblem<M, J>:
            ProjectPoint<Dimension = J::Dimension, Point = CanonicalProblem<M, J::PointJet>>,
        W: ProjectPoint<Dimension = J::Dimension>,
        CompilationContext<I, J::Dimension, J::Mapping>:
            ProjectPoint<Dimension = J::Dimension, Point = CompilationContext<I, Ix0, J::Mapping>>,
        Idx: NdIndex<J::Dimension> + Clone,
        M: Clone,
    {
        Ok(PlaneWaveState::new(
            self.problem.project_point(index)?,
            self.workspace.project_point(index)?,
            self.context.project_point(index)?,
            self.stack.clone(),
            self.polarisation,
        ))
    }

    /// Return the compiled canonical plane-wave problem.
    pub fn problem(&self) -> &CanonicalProblem<M, J> {
        &self.problem
    }

    pub fn stack(&self) -> &Stack<M, <J::Scalar as ComplexField>::RealField> {
        &self.stack
    }

    /// Return the computed workspace
    pub fn workspace(&self) -> &W {
        &self.workspace
    }

    pub fn polarisation(&self) -> Polarisation {
        self.polarisation
    }

    pub fn solution(&self) -> PlaneWaveSolutionView<'_, W::Entries>
    where
        W: PlaneWaveSolutionSource,
    {
        self.workspace.solution()
    }

    /// Return the retained compilation metadata.
    pub fn context(&self) -> &CompilationContext<I, J::Dimension, J::Mapping> {
        &self.context
    }

    /// Consume the state and return its components.
    pub fn into_parts(
        self,
    ) -> (
        CanonicalProblem<M, J>,
        W,
        CompilationContext<I, J::Dimension, J::Mapping>,
    ) {
        (self.problem, self.workspace, self.context)
    }

    /// Transform the retained workspace while preserving the canonical
    /// problem and compilation context.
    pub fn map_result<W2>(self, map: impl FnOnce(W) -> W2) -> PlaneWaveState<J, I, M, W2> {
        PlaneWaveState {
            problem: self.problem,
            workspace: map(self.workspace),
            context: self.context,
            stack: self.stack,
            polarisation: self.polarisation,
        }
    }

    pub(crate) fn raw_layer_boundary_waves_unchecked(
        &self,
        incident_side: IncidentSide,
    ) -> Result<LayerBoundaries<LayerBoundaryWaves<J>>, BoundaryProjectionError>
    where
        W: ReconstructLayerBoundaryWaves<Algebra = J>,
    {
        project_layer_boundary_waves(&self.workspace, incident_side)
    }

    pub(crate) fn raw_layer_boundary_states_unchecked(
        &self,
        incident_side: IncidentSide,
    ) -> Result<LayerBoundaries<LayerBoundaryStates<J>>, BoundaryProjectionError>
    where
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
        J: ScalarAlgebra,
        J::Scalar: ComplexScalar,
    {
        project_layer_boundary_states(&self.workspace, incident_side)
    }

    pub(crate) fn raw_interface_states_unchecked(
        &self,
        incident_side: IncidentSide,
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
    {
        let layers = self.raw_layer_boundary_states_unchecked(incident_side)?;

        let amplitudes = self.raw_amplitudes(incident_side).into();

        let exterior = self.solution().context();

        let exterior_states = exterior_boundary_states(
            &amplitudes,
            incident_side,
            exterior.left_admittance(),
            exterior.right_admittance(),
        );

        Ok(layers.into_interface_states(exterior_states.left, exterior_states.right))
    }

    pub(crate) fn raw_interface_wave_data_unchecked(
        &self,
        incident_side: IncidentSide,
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
    {
        let layer_waves = self.raw_layer_boundary_waves_unchecked(incident_side)?;

        let layer_admittances = project_layer_admittances(&self.workspace)?;

        let solution = self.solution();

        let amplitudes = solution.amplitudes(incident_side);

        let exterior = solution.context();

        let left_admittance = exterior.left_admittance().clone();

        let right_admittance = exterior.right_admittance().clone();

        let exterior_waves =
            exterior_boundary_waves(&amplitudes, incident_side, left_admittance.value());

        assemble_interface_wave_data(
            layer_waves,
            layer_admittances,
            exterior_waves,
            left_admittance,
            right_admittance,
        )
    }

    pub(crate) fn raw_layer_integration_inputs_unchecked(
        &self,
        incident_side: IncidentSide,
    ) -> Result<Layers<LayerIntegrationInput<J>>, LayerProjectionError>
    where
        W: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
        J: Clone,
    {
        let boundary_waves = self.raw_layer_boundary_waves_unchecked(incident_side)?;

        assemble_layer_integration_inputs(&self.workspace, boundary_waves)
    }

    pub(super) fn raw_constitutive_parameters(
        &self,
        sampling: &ResolvedFieldSampling<<J::Scalar as ComplexField>::RealField>,
    ) -> Result<IsotropicConstitutiveParameters<J::Stacked>, ConstitutiveSamplingError>
    where
        W: PlaneWaveSolutionSource + RetainedIsotropicLayers<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
        J: JetStack + Clone,
        J::Dimension: Dimension,
    {
        let context = ConstitutiveSamplingContext::new(self.workspace());

        context.sample(sampling)
    }
}

impl<J, M, W> PlaneWaveState<J, <J::Scalar as ComplexField>::RealField, M, W>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    J::Policy: Default,
    <J::Scalar as ComplexField>::RealField: ComplexField,
    W: PlaneWaveSolutionSource,
{
    pub(crate) fn raw_interface_power_unchecked(
        &self,
        incident_side: IncidentSide,
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
        <J::RealJet as Jet>::Scalar: One + Neg<Output = <J::RealJet as Jet>::Scalar>,
    {
        let interface_data = self.raw_interface_wave_data_unchecked(incident_side)?;

        let solution = self.solution();
        let exterior = solution.context();

        let incident_admittance = match incident_side {
            IncidentSide::Left => exterior.left_admittance(),

            IncidentSide::Right => exterior.right_admittance(),
        };

        let incident_flux_magnitude = RealScalarAlgebra::real(incident_admittance);

        Ok(interface_data.into_power(&incident_flux_magnitude))
    }

    pub(crate) fn raw_layer_power_unchecked(
        &self,
        incident_side: IncidentSide,
    ) -> Result<Layers<LayerPower<J::RealJet>>, InterfaceProjectionError>
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
        <J::RealJet as Jet>::Scalar: One + Neg<Output = <J::RealJet as Jet>::Scalar>,
    {
        Ok(self
            .raw_interface_power_unchecked(incident_side)?
            .into_layer_power())
    }

    pub(crate) fn raw_incident_flux_magnitude_unchecked(
        &self,
        incident_side: IncidentSide,
    ) -> J::RealJet
    where
        J: ComplexJet + RealScalarAlgebra,
        J::RealJet: ScalarAlgebra,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        let context = self.solution().context();

        let admittance = match incident_side {
            IncidentSide::Left => context.left_admittance(),
            IncidentSide::Right => context.right_admittance(),
        };

        admittance.real()
    }

    pub(crate) fn raw_layer_dissipation_unchecked(
        &self,
        incident_side: IncidentSide,
    ) -> Result<Layers<LayerDissipation<J::RealJet>>, LayerProjectionError>
    where
        J: RealScalarAlgebra + ScalarAlgebraExpRelExt + Clone,
        J::RealJet: ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::RealJet as Jet>::Scalar: One,
        W: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        let layers = self
            .raw_layer_integration_inputs_unchecked(incident_side)?
            .integrate();

        let coordinates = self.problem().coordinates();

        let incident_flux = self.raw_incident_flux_magnitude_unchecked(incident_side);

        Ok(layers.into_dissipation(
            coordinates.vacuum_angular_wavenumber(),
            coordinates.parallel_angular_wavenumber(),
            &incident_flux,
        ))
    }

    pub(crate) fn raw_nondispersive_layer_energy_unchecked(
        &self,
        incident_side: IncidentSide,
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
            .raw_layer_integration_inputs_unchecked(incident_side)?
            .integrate();

        let problem = self.problem();
        let coordinates = problem.coordinates();

        let incident_flux = self.raw_incident_flux_magnitude_unchecked(incident_side);

        Ok(integrated.into_nondispersive_energy(
            coordinates.vacuum_angular_wavenumber(),
            coordinates.parallel_angular_wavenumber(),
            &incident_flux,
        ))
    }

    pub(crate) fn raw_dispersive_layer_energy_unchecked<E>(
        &self,
        incident_side: IncidentSide,
    ) -> Result<Layers<LayerEnergy<J::RealJet>>, LayerEnergyError>
    where
        J: ComplexJet
            + RealScalarAlgebra
            + ScalarAlgebraExpRelExt
            + ConstitutiveSpectralFirstLift<E, M>
            + Clone,
        J::RealJet: ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::RealJet as Jet>::Scalar: FromPrimitive + One,
        J::Dimension: Dimension,
        E: ConstitutiveDerivativeEvaluator<J::Scalar, J::Dimension, M>,
        W: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        let coordinates = self.problem().coordinates();

        let incident_flux = self.raw_incident_flux_magnitude_unchecked(incident_side);

        let sequence = self
            .raw_layer_integration_inputs_unchecked(incident_side)?
            .integrate()
            .into_brillouin_layers(
                self.problem()
                    .stack()
                    .layers()
                    .iter()
                    .map(|layer| layer.material()),
                coordinates.vacuum_angular_wavenumber(),
            )?;

        Ok(sequence.into_brillouin_energy(
            coordinates.vacuum_angular_wavenumber(),
            coordinates.parallel_angular_wavenumber(),
            &incident_flux,
        ))
    }
}

impl<J, M, W> PlaneWaveState<J, J::Scalar, M, W>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    J::Policy: Default,
    W: PlaneWaveSolutionSource,
{
    pub fn mode(&self) -> Result<PlaneWaveMode<'_, J, M, W>, QnmCreationError>
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
        W: PlaneWaveSolutionSource
            + ReconstructLayerModeWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>
            + ModalSolutionSource<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
        J::Policy: DerivativePartsPolicy<AggregateBilinearNormalization<J>>,
        AggregateBilinearNormalization<J>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        PlaneWaveMode::new(self)
    }

    pub fn determinant(&self) -> DifferentialResponseFor<J, RawModeDeterminant<Self, J>>
    where
        W::Entries: ProjectPlaneWaveModeDeterminant,
        J::Policy: DerivativePartsPolicy<RawModeDeterminant<Self, J>>,
        RawModeDeterminant<Self, J>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        self.raw_determinant()
            .into_differential_response(&J::Policy::default(), self.mapping())
    }
}

pub(super) type RawInterfacePower<J> = Interfaces<InterfacePower<<J as ComplexJet>::RealJet>>;
pub(super) type RawLayerPower<J> = Layers<LayerPower<<J as ComplexJet>::RealJet>>;
