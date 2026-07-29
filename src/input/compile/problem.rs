use nalgebra::ComplexField;
use ndarray::Dimension;

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
{
    canonical: CanonicalBackendInput<M, J>,
    context: CompilationContext<C, D>,
}

impl<M, J, C, D> CompiledProblem<M, J, C, D>
where
    C: ComplexField,
    D: Dimension,
{
    pub(crate) fn new(
        canonical: CanonicalBackendInput<M, J>,
        context: CompilationContext<C, D>,
    ) -> Self {
        Self { canonical, context }
    }

    /// Canonical input passed to the numerical backend.
    pub fn canonical(&self) -> &CanonicalBackendInput<M, J> {
        &self.canonical
    }

    /// Caller-facing information used to interpret results.
    pub fn compilation_context(&self) -> &CompilationContext<C, D> {
        &self.context
    }

    pub fn into_parts(self) -> (CanonicalBackendInput<M, J>, CompilationContext<C, D>) {
        (self.canonical, self.context)
    }
}
