use nalgebra::ComplexField;
use ndarray::Dimension;

use crate::{
    algebra::ComplexJet,
    input::{CompilationContext, JetMapping},
    observable::PlaneWaveObservables,
};

/// A non-retained plane-wave evaluation.
///
/// This stores only the external, jet-valued observables and the compilation
/// context needed to crystallise them. No backend workspace or canonical
/// problem is retained.
#[derive(Clone, Debug)]
pub struct PlaneWaveResult<J>
where
    J: ComplexJet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
{
    observables: PlaneWaveObservables<J, J::RealJet>,
    context: CompilationContext<J::Scalar, J::Dimension, J::Mapping>,
}

impl<J> PlaneWaveResult<J>
where
    J: ComplexJet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
{
    pub(crate) fn new(
        observables: PlaneWaveObservables<J, J::RealJet>,
        context: CompilationContext<J::Scalar, J::Dimension, J::Mapping>,
    ) -> Self {
        Self {
            observables,
            context,
        }
    }

    /// Return the uncrystallised, jet-valued observables.
    pub fn raw_observables(&self) -> &PlaneWaveObservables<J, J::RealJet> {
        &self.observables
    }

    /// Return the retained compilation context.
    pub fn context(&self) -> &CompilationContext<J::Scalar, J::Dimension, J::Mapping> {
        &self.context
    }

    /// Consume the result and return its components.
    pub fn into_parts(
        self,
    ) -> (
        PlaneWaveObservables<J, J::RealJet>,
        CompilationContext<J::Scalar, J::Dimension, J::Mapping>,
    ) {
        (self.observables, self.context)
    }
}
