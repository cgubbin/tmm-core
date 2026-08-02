use nalgebra::ComplexField;
use ndarray::Dimension;
use num_traits::{One, Zero};

use crate::{
    ComplexScalar, IncidentSide, PlaneWaveAmplitudes,
    algebra::{
        ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, ComplexJet, Jet,
        ScalarAlgebra,
    },
    backend::{
        ExteriorAdmittanceProvider, PlaneWaveEntries, PlaneWaveSolutionSource,
        PlaneWaveSolutionView, ReconstructLayerBoundaryWaves, RetainedIsotropicLayers,
    },
    derivative_parts::{DerivativePartsPolicy, IntoDerivativeParts},
    differential::{
        BivariateFirst, BivariateSecond, DirectionalFirst, DirectionalSecond,
        IntoDifferentialResponse, NoDerivatives,
    },
    input::{CanonicalProblem, CompilationContext, JetMapping},
    observable::{
        BoundaryProjectionError, InterfaceStates, Interfaces, LayerBoundaries, LayerBoundaryStates,
        LayerBoundaryWaves, ProjectAmplitudes, ProjectPlaneWaveModeDeterminant, ProjectPower,
        assemble_interface_states, exterior_boundary_states, project_boundary_states,
        project_boundary_waves,
    },
};

use super::query::{
    DifferentialResponseFor, PlaneWaveExternalQueries, PlaneWaveQuery, RawAmplitudes,
    RawModeDeterminant, RawPower,
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
    ) -> Self {
        Self {
            problem,
            workspace,
            context,
        }
    }

    /// Return the compiled canonical plane-wave problem.
    pub fn problem(&self) -> &CanonicalProblem<M, J> {
        &self.problem
    }

    /// Return the computed workspace
    pub fn workspace(&self) -> &W {
        &self.workspace
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
        }
    }

    pub(crate) fn raw_layer_boundary_waves(
        &self,
        incident_side: IncidentSide,
    ) -> Result<LayerBoundaries<LayerBoundaryWaves<J>>, BoundaryProjectionError>
    where
        W: ReconstructLayerBoundaryWaves<Algebra = J>,
    {
        project_boundary_waves(&self.workspace, incident_side)
    }

    pub fn boundary_waves(
        &self,
        incident_side: IncidentSide,
    ) -> Result<
        DifferentialResponseFor<J, LayerBoundaries<LayerBoundaryWaves<J>>>,
        BoundaryProjectionError,
    >
    where
        J: JetMapping,
        J::Policy: Default + DerivativePartsPolicy<LayerBoundaries<LayerBoundaryWaves<J>>>,
        W: PlaneWaveSolutionSource + ReconstructLayerBoundaryWaves<Algebra = J>,
        LayerBoundaries<LayerBoundaryWaves<J>>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        Ok(self
            .raw_layer_boundary_waves(incident_side)?
            .into_differential_response(&J::Policy::default(), self.mapping()))
    }

    pub(crate) fn raw_layer_boundary_states(
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
        project_boundary_states(&self.workspace, incident_side)
    }

    pub fn boundary_states(
        &self,
        incident_side: IncidentSide,
    ) -> Result<
        DifferentialResponseFor<J, LayerBoundaries<LayerBoundaryStates<J>>>,
        BoundaryProjectionError,
    >
    where
        J: JetMapping + ScalarAlgebra,
        J::Scalar: ComplexScalar,
        J::Policy: Default + DerivativePartsPolicy<LayerBoundaries<LayerBoundaryStates<J>>>,
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
        LayerBoundaries<LayerBoundaryStates<J>>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        Ok(self
            .raw_layer_boundary_states(incident_side)?
            .into_differential_response(&J::Policy::default(), self.mapping()))
    }

    pub(crate) fn raw_interface_states(
        &self,
        incident_side: IncidentSide,
    ) -> Result<Interfaces<InterfaceStates<J>>, BoundaryProjectionError>
    where
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
        W::Entries: ProjectAmplitudes,
        <W::Entries as ProjectAmplitudes>::Amplitudes: Into<PlaneWaveAmplitudes<J>>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorAdmittanceProvider<Algebra = J>,
        J: ScalarAlgebra + Clone,
        J::Scalar: ComplexScalar + One + Zero,
        J::Dimension: Dimension,
    {
        let layers = self.raw_layer_boundary_states(incident_side)?;

        let amplitudes = self.raw_amplitudes(incident_side).into();

        let exterior = self.solution().context();

        let exterior_states = exterior_boundary_states(
            &amplitudes,
            incident_side,
            exterior.left_admittance(),
            exterior.right_admittance(),
        );

        Ok(assemble_interface_states(
            layers,
            exterior_states.left,
            exterior_states.right,
        ))
    }

    pub fn interface_states(
        &self,
        incident_side: IncidentSide,
    ) -> Result<DifferentialResponseFor<J, Interfaces<InterfaceStates<J>>>, BoundaryProjectionError>
    where
        W: PlaneWaveSolutionSource
            + ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>,
        W::Entries: ProjectAmplitudes,
        <W::Entries as ProjectAmplitudes>::Amplitudes: Into<PlaneWaveAmplitudes<J>>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorAdmittanceProvider<Algebra = J>,
        J: ScalarAlgebra + Clone,
        J::Scalar: ComplexScalar + One + Zero,
        J::Dimension: Dimension,
        J::Policy: Default + DerivativePartsPolicy<Interfaces<InterfaceStates<J>>>,
        Interfaces<InterfaceStates<J>>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        let raw = self.raw_interface_states(incident_side)?;

        Ok(raw.into_differential_response(&J::Policy::default(), self.mapping()))
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
    pub fn amplitudes(
        &self,
        incident_side: IncidentSide,
    ) -> DifferentialResponseFor<J, RawAmplitudes<Self, J>>
    where
        W::Entries: ProjectAmplitudes,
        J::Policy: DerivativePartsPolicy<RawAmplitudes<Self, J>>,
        RawAmplitudes<Self, J>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        self.raw_amplitudes(incident_side)
            .into_differential_response(&J::Policy::default(), self.mapping())
    }

    pub fn power(
        &self,
        incident_side: IncidentSide,
    ) -> DifferentialResponseFor<J, RawPower<Self, J>>
    where
        W::Entries: ProjectPower,
        J::Policy: DerivativePartsPolicy<RawPower<Self, J>>,
        RawPower<Self, J>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        self.raw_power(incident_side)
            .into_differential_response(&J::Policy::default(), self.mapping())
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
