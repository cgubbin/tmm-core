use nalgebra::ComplexField;
use ndarray::Dimension;

use crate::{
    IncidentSide,
    algebra::{ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, ComplexJet, Jet},
    backend::{PlaneWaveSolutionSource, PlaneWaveSolutionView},
    derivative_parts::{DerivativePartsPolicy, IntoDerivativeParts},
    differential::{
        BivariateFirst, BivariateSecond, DirectionalFirst, DirectionalSecond,
        IntoDifferentialResponse, NoDerivatives,
    },
    input::{CanonicalProblem, CompilationContext, JetMapping},
    observable::{ProjectAmplitudes, ProjectPlaneWaveModeDeterminant, ProjectPower},
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
