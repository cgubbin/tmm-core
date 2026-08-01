use nalgebra::ComplexField;
use ndarray::Dimension;

use crate::{
    IncidentSide,
    algebra::Jet,
    backend::PlaneWaveSolutionSource,
    derivative_parts::DerivativePartsPolicy,
    differential::IntoDifferentialResponse,
    input::{CompilationContext, JetMapping},
    observable::{ProjectAmplitudes, ProjectPlaneWaveModeDeterminant, ProjectPower},
};

use super::query::{
    DifferentialResponseFor, PlaneWaveExternalQueries, PlaneWaveQuery, RawAmplitudes,
    RawModeDeterminant, RawPower,
};

/// A completed non-retained plane-wave evaluation.
///
/// The result stores the backend's external solution and the typed compilation
/// context required to assemble caller-facing differential responses.
///
/// No canonical problem or backend workspace is retained. Consequently, the
/// result supports external projections such as amplitudes, powers, and modal
/// determinants, but not internal-field reconstruction.
#[derive(Clone, Debug)]
pub struct PlaneWaveResult<J, I, S>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    I: ComplexField,
{
    solution: S,
    context: CompilationContext<I, J::Dimension, J::Mapping>,
}

impl<J, I, S> PlaneWaveResult<J, I, S>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    I: ComplexField,
{
    pub(crate) fn new(
        solution: S,
        context: CompilationContext<I, J::Dimension, J::Mapping>,
    ) -> Self {
        Self { solution, context }
    }

    pub fn solution(&self) -> &S {
        &self.solution
    }

    /// Return the retained compilation context.
    pub fn context(&self) -> &CompilationContext<I, J::Dimension, J::Mapping> {
        &self.context
    }

    /// Consume the result and return its components.
    pub fn into_parts(self) -> (S, CompilationContext<I, J::Dimension, J::Mapping>) {
        (self.solution, self.context)
    }
}

impl<J, S> PlaneWaveResult<J, <J::Scalar as ComplexField>::RealField, S>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    J::Policy: Default,
    <J::Scalar as ComplexField>::RealField: ComplexField,
    S: PlaneWaveSolutionSource,
{
    pub fn amplitudes(
        &self,
        incident_side: IncidentSide,
    ) -> DifferentialResponseFor<J, RawAmplitudes<Self, J>>
    where
        S::Entries: ProjectAmplitudes,
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
        S::Entries: ProjectPower,
        J::Policy: DerivativePartsPolicy<RawPower<Self, J>>,
        RawPower<Self, J>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        self.raw_power(incident_side)
            .into_differential_response(&J::Policy::default(), self.mapping())
    }
}

impl<J, S> PlaneWaveResult<J, J::Scalar, S>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    J::Policy: Default,
    S: PlaneWaveSolutionSource,
{
    pub fn determinant(&self) -> DifferentialResponseFor<J, RawModeDeterminant<Self, J>>
    where
        S::Entries: ProjectPlaneWaveModeDeterminant,
        J::Policy: DerivativePartsPolicy<RawModeDeterminant<Self, J>>,
        RawModeDeterminant<Self, J>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        self.raw_determinant()
            .into_differential_response(&J::Policy::default(), self.mapping())
    }
}
