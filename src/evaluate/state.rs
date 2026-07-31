use crate::{
    IncidentSide, PlaneWaveObservables,
    algebra::{ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, ComplexJet},
    backend::{PlaneWaveSolutionSource, PlaneWaveSolutionView},
    crystallise::{Crystallise, CrystallisePolicy},
    differential::{
        BivariateFirst, BivariateSecond, DirectionalFirst, DirectionalSecond, NoDerivatives,
    },
    input::{CanonicalProblem, CompilationContext, JetEvaluation},
    observable::{ProjectAmplitudes, ProjectPlaneWaveModeDeterminant, ProjectPower},
};

/// A retained plane-wave solution.
///
/// The state contains the canonical problem, the backend-specific retained
/// workspace, and the metadata required to crystallise derivative-aware
/// quantities into caller-facing coordinates and units.
///
/// No derived quantity is crystallised during evaluation.
#[derive(Clone, Debug)]
pub struct PlaneWaveState<M, J, R, Ctx>
where
    J: JetEvaluation,
{
    problem: CanonicalProblem<M, J>,
    result: R,
    context: Ctx,
}

impl<M, J, R, Ctx> PlaneWaveState<M, J, R, Ctx>
where
    J: JetEvaluation,
{
    pub(crate) fn new(problem: CanonicalProblem<M, J>, result: R, context: Ctx) -> Self {
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
    pub fn context(&self) -> &Ctx {
        &self.context
    }

    /// Consume the state and return its components.
    pub fn into_parts(self) -> (CanonicalProblem<M, J>, R, Ctx) {
        (self.problem, self.result, self.context)
    }

    /// Transform the retained workspace while preserving the canonical
    /// problem and compilation context.
    pub fn map_result<R2>(self, map: impl FnOnce(R) -> R2) -> PlaneWaveState<M, J, R2, Ctx> {
        PlaneWaveState {
            problem: self.problem,
            result: map(self.result),
            context: self.context,
        }
    }

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

    pub fn amplitudes(
        &self,
        incident_side: IncidentSide,
    ) -> <J::Policy as CrystallisePolicy<
        <<R as PlaneWaveSolutionSource>::Entries as ProjectAmplitudes>::Amplitudes,
    >>::Output
    where
        R: PlaneWaveSolutionSource,
        R::Entries: ProjectAmplitudes,
        J::Policy: CrystallisePolicy<
            <<R as PlaneWaveSolutionSource>::Entries as ProjectAmplitudes>::Amplitudes,
        >,
    {
        let raw = self.raw_amplitudes(incident_side);
        raw.crystallise(&J::Policy::default())
    }

    fn raw_power(&self, incident_side: IncidentSide) -> <R::Entries as ProjectPower>::Power
    where
        R: PlaneWaveSolutionSource,
        R::Entries: ProjectPower,
    {
        self.solution().power(incident_side)
    }

    pub fn power(
        &self,
        incident_side: IncidentSide,
    ) -> <J::Policy as CrystallisePolicy<
        <<R as PlaneWaveSolutionSource>::Entries as ProjectPower>::Power,
    >>::Output
    where
        R: PlaneWaveSolutionSource,
        R::Entries: ProjectPower,
        J::Policy:
            CrystallisePolicy<<<R as PlaneWaveSolutionSource>::Entries as ProjectPower>::Power>,
    {
        let raw = self.raw_power(incident_side);
        raw.crystallise(&J::Policy::default())
    }

    fn raw_determinant(&self) -> <R::Entries as ProjectPlaneWaveModeDeterminant>::Determinant
    where
        R: PlaneWaveSolutionSource,
        R::Entries: ProjectPlaneWaveModeDeterminant,
    {
        self.solution().determinant()
    }

    pub fn determinant(
        &self,
    ) -> <J::Policy as CrystallisePolicy<
        <<R as PlaneWaveSolutionSource>::Entries as ProjectPlaneWaveModeDeterminant>::Determinant,
    >>::Output
    where
        R: PlaneWaveSolutionSource,
        R::Entries: ProjectPlaneWaveModeDeterminant,
        J::Policy: CrystallisePolicy<<<R as PlaneWaveSolutionSource>::Entries as ProjectPlaneWaveModeDeterminant>::Determinant>
    {
        let raw = self.raw_determinant();
        raw.crystallise(&J::Policy::default())
    }
}
