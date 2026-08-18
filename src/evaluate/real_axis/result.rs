use nalgebra::ComplexField;
use ndarray::Dimension;

use crate::{
    IncidentSide,
    algebra::Jet,
    backend::PlaneWaveSolutionSource,
    derivative_parts::DerivativePartsPolicy,
    differential::IntoDifferentialResponse,
    input::{CompilationContext, JetMapping, ProjectionConstraint, ProjectionConstraintError},
    observable::{ProjectAmplitudes, ProjectPower},
};

use super::query::{
    DifferentialResponseFor, PlaneWaveQuery, RawAmplitudes, RawPower, RealAxisExternalQueries,
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
pub struct RealAxisResult<J, S>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    <J::Scalar as ComplexField>::RealField: ComplexField,
{
    solution: S,
    context: CompilationContext<<J::Scalar as ComplexField>::RealField, J::Dimension, J::Mapping>,
}

impl<J, S> RealAxisResult<J, S>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    <J::Scalar as ComplexField>::RealField: ComplexField,
{
    pub(crate) fn new(
        solution: S,
        context: CompilationContext<
            <J::Scalar as ComplexField>::RealField,
            J::Dimension,
            J::Mapping,
        >,
    ) -> Self {
        Self { solution, context }
    }

    pub fn solution(&self) -> &S {
        &self.solution
    }

    /// Return the retained compilation context.
    pub fn context(
        &self,
    ) -> &CompilationContext<<J::Scalar as ComplexField>::RealField, J::Dimension, J::Mapping> {
        &self.context
    }

    /// Consume the result and return its components.
    pub fn into_parts(
        self,
    ) -> (
        S,
        CompilationContext<<J::Scalar as ComplexField>::RealField, J::Dimension, J::Mapping>,
    ) {
        (self.solution, self.context)
    }
}

impl<J, S> RealAxisResult<J, S>
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
    ) -> Result<DifferentialResponseFor<J, RawAmplitudes<Self, J>>, ProjectionConstraintError>
    where
        S::Entries: ProjectAmplitudes,
        J::Policy: DerivativePartsPolicy<RawAmplitudes<Self, J>>,
        RawAmplitudes<Self, J>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        if let ProjectionConstraint::Fixed(side) = self.context().projection_constraint() {
            if side != incident_side {
                return Err(ProjectionConstraintError {
                    constraint: side,
                    requested: incident_side,
                });
            }
        }

        Ok(self
            .raw_amplitudes(incident_side)
            .into_differential_response(&J::Policy::default(), self.mapping()))
    }

    pub fn power(
        &self,
        incident_side: IncidentSide,
    ) -> Result<DifferentialResponseFor<J, RawPower<Self, J>>, ProjectionConstraintError>
    where
        S::Entries: ProjectPower,
        J::Policy: DerivativePartsPolicy<RawPower<Self, J>>,
        RawPower<Self, J>: IntoDifferentialResponse<J::Policy, J::Mapping>,
    {
        if let ProjectionConstraint::Fixed(side) = self.context().projection_constraint() {
            if side != incident_side {
                return Err(ProjectionConstraintError {
                    constraint: side,
                    requested: incident_side,
                });
            }
        }

        Ok(self
            .raw_power(incident_side)
            .into_differential_response(&J::Policy::default(), self.mapping()))
    }
}
