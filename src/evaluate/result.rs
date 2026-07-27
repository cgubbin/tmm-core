use crate::{algebra::ComplexJet, observable::PlaneWaveObservables};

/// A non-retained plane-wave evaluation.
///
/// This stores only the external, jet-valued observables and the compilation
/// context needed to crystallise them. No backend workspace or canonical
/// problem is retained.
#[derive(Clone, Debug)]
pub struct PlaneWaveResult<J, Ctx>
where
    J: ComplexJet,
{
    observables: PlaneWaveObservables<J, J::RealJet>,
    context: Ctx,
}

impl<J, Ctx> PlaneWaveResult<J, Ctx>
where
    J: ComplexJet,
{
    pub(crate) fn new(observables: PlaneWaveObservables<J, J::RealJet>, context: Ctx) -> Self {
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
    pub fn context(&self) -> &Ctx {
        &self.context
    }

    /// Consume the result and return its components.
    pub fn into_parts(self) -> (PlaneWaveObservables<J, J::RealJet>, Ctx) {
        (self.observables, self.context)
    }
}
