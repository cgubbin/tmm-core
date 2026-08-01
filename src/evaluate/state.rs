use nalgebra::ComplexField;
use ndarray::Dimension;

use crate::{
    IncidentSide, PlaneWaveObservables,
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
pub struct PlaneWaveState<M, J, I, R>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    I: ComplexField,
{
    problem: CanonicalProblem<M, J>,
    result: R,
    context: CompilationContext<I, J::Dimension, J::Mapping>,
}

impl<M, J, I, R> PlaneWaveState<M, J, I, R>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    I: ComplexField,
{
    pub(crate) fn new(
        problem: CanonicalProblem<M, J>,
        result: R,
        context: CompilationContext<I, J::Dimension, J::Mapping>,
    ) -> Self {
        Self {
            problem,
            result,
            context,
        }
    }

    /// Return the compiled canonical plane-wave problem.
    pub fn problem(&self) -> &CanonicalProblem<M, J> {
        &self.problem
    }

    /// Return the computed result
    pub fn result(&self) -> &R {
        &self.result
    }

    pub fn solution(&self) -> PlaneWaveSolutionView<'_, R::Entries>
    where
        R: PlaneWaveSolutionSource,
    {
        self.result.solution()
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
        R,
        CompilationContext<I, J::Dimension, J::Mapping>,
    ) {
        (self.problem, self.result, self.context)
    }

    /// Transform the retained workspace while preserving the canonical
    /// problem and compilation context.
    pub fn map_result<R2>(self, map: impl FnOnce(R) -> R2) -> PlaneWaveState<M, J, I, R2> {
        PlaneWaveState {
            problem: self.problem,
            result: map(self.result),
            context: self.context,
        }
    }

    fn assemble<T>(&self, value: T) -> DifferentialResponseFor<J, T>
    where
        T: IntoDifferentialResponse<J::Policy, J::Mapping>,
        J::Policy: DerivativePartsPolicy<T>,
    {
        value.into_differential_response(&J::Policy::default(), self.context.mapping())
    }
}

impl<M, J, R> PlaneWaveState<M, J, <J::Scalar as ComplexField>::RealField, R>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    <J::Scalar as ComplexField>::RealField: ComplexField,
{
    fn raw_amplitudes(
        &self,
        incident_side: IncidentSide,
    ) -> <R::Entries as ProjectAmplitudes>::Amplitudes
    where
        R: PlaneWaveSolutionSource,
        R::Entries: ProjectAmplitudes,
    {
        self.solution().amplitudes(incident_side)
    }

    fn raw_power(&self, incident_side: IncidentSide) -> <R::Entries as ProjectPower>::Power
    where
        R: PlaneWaveSolutionSource,
        R::Entries: ProjectPower,
    {
        self.solution().power(incident_side)
    }

    pub fn amplitudes(
        &self,
        incident_side: IncidentSide,
    ) -> DifferentialResponseFor<J, RawAmplitudes<R>>
    where
        R: PlaneWaveSolutionSource,
        R::Entries: ProjectAmplitudes,
        J::Policy: DerivativePartsPolicy<RawAmplitudes<R>>,
        RawAmplitudes<R>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        self.raw_amplitudes(incident_side)
            .into_differential_response(&J::Policy::default(), self.context.mapping())
    }

    pub fn power(&self, incident_side: IncidentSide) -> DifferentialResponseFor<J, RawPower<R>>
    where
        R: PlaneWaveSolutionSource,
        R::Entries: ProjectPower,
        J::Policy: DerivativePartsPolicy<RawPower<R>>,
        RawPower<R>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        self.raw_power(incident_side)
            .into_differential_response(&J::Policy::default(), self.context.mapping())
    }
}

impl<M, J, R> PlaneWaveState<M, J, J::Scalar, R>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
{
    fn raw_determinant(&self) -> <R::Entries as ProjectPlaneWaveModeDeterminant>::Determinant
    where
        R: PlaneWaveSolutionSource,
        R::Entries: ProjectPlaneWaveModeDeterminant,
    {
        self.solution().determinant()
    }

    pub fn determinant(&self) -> DifferentialResponseFor<J, RawModeDeterminant<R>>
    where
        R: PlaneWaveSolutionSource,
        R::Entries: ProjectPlaneWaveModeDeterminant,
        J::Policy: DerivativePartsPolicy<RawModeDeterminant<R>>,
        RawModeDeterminant<R>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        self.raw_determinant()
            .into_differential_response(&J::Policy::default(), self.context.mapping())
    }
}

type EntriesOf<R> = <R as PlaneWaveSolutionSource>::Entries;

type RawAmplitudes<R> = <EntriesOf<R> as ProjectAmplitudes>::Amplitudes;

type RawPower<R> = <EntriesOf<R> as ProjectPower>::Power;

type RawModeDeterminant<R> = <EntriesOf<R> as ProjectPlaneWaveModeDeterminant>::Determinant;

type DifferentialResponseFor<J, T> =
    <T as IntoDifferentialResponse<<J as JetMapping>::Policy, <J as JetMapping>::Mapping>>::Output;
