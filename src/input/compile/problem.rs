use nalgebra::ComplexField;
use ndarray::Dimension;

use super::JetMapping;
use crate::input::CanonicalBackendInput;

use super::CompilationContext;

/// A validated, canonicalised optical problem together with the information
/// required to interpret its results.
///
/// The backend borrows `canonical`; it does not consume this object. This
/// permits repeated solves, field reconstruction, and subsequent analyses to
/// share one compiled representation.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledProblem<M, J, C, D>
where
    C: ComplexField,
    D: Dimension,
    J: JetMapping,
{
    canonical: CanonicalBackendInput<M, J>,
    context: CompilationContext<C, D, J::Mapping>,
}

impl<M, J, C, D> CompiledProblem<M, J, C, D>
where
    C: ComplexField,
    D: Dimension,
    J: JetMapping,
{
    pub(crate) fn new(
        canonical: CanonicalBackendInput<M, J>,
        context: CompilationContext<C, D, J::Mapping>,
    ) -> Self {
        Self { canonical, context }
    }

    /// Canonical input passed to the numerical backend.
    pub fn canonical(&self) -> &CanonicalBackendInput<M, J> {
        &self.canonical
    }

    /// Caller-facing information used to interpret results.
    pub fn compilation_context(&self) -> &CompilationContext<C, D, J::Mapping> {
        &self.context
    }

    pub fn into_parts(
        self,
    ) -> (
        CanonicalBackendInput<M, J>,
        CompilationContext<C, D, J::Mapping>,
    ) {
        (self.canonical, self.context)
    }
}
