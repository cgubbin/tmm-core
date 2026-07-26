use ndarray::Dimension;

use crate::input::CanonicalProblem;

use super::CompilationContext;

/// A validated, canonicalised optical problem together with the information
/// required to interpret its results.
///
/// The backend borrows `canonical`; it does not consume this object. This
/// permits repeated solves, field reconstruction, and subsequent analyses to
/// share one compiled representation.
#[derive(Clone, Debug, PartialEq)]
pub struct CompiledProblem<M, J, R, D>
where
    D: Dimension,
{
    canonical: CanonicalProblem<M, J>,
    context: CompilationContext<R, D>,
}

impl<M, J, R, D> CompiledProblem<M, J, R, D>
where
    D: Dimension,
{
    pub(crate) fn new(
        canonical: CanonicalProblem<M, J>,
        context: CompilationContext<R, D>,
    ) -> Self {
        Self { canonical, context }
    }

    /// Canonical input passed to the numerical backend.
    pub fn canonical(&self) -> &CanonicalProblem<M, J> {
        &self.canonical
    }

    /// Caller-facing information used to interpret results.
    pub fn context(&self) -> &CompilationContext<R, D> {
        &self.context
    }

    pub fn into_parts(self) -> (CanonicalProblem<M, J>, CompilationContext<R, D>) {
        (self.canonical, self.context)
    }
}
