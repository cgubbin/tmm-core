use crate::{
    ComplexScalar,
    backend::{DerivativeVariable, PlanarInput},
};

use ndarray::{ArrayBase, Dimension, OwnedRepr};

/// Backend capable of constructing an outgoing planar-mode residual.
///
/// The stack's exterior media define the left and right asymptotic regions.
/// Implementations enforce the backend-appropriate outgoing or decaying
/// condition in those regions and return a scalar analytic residual.
///
/// This trait constructs the residual only. It does not count, locate, or
/// refine its zeros.
pub trait OutgoingModeBackend<C, D, S>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Error produced while constructing the mode residual.
    type Error;

    /// Evaluate the outgoing-mode residual.
    fn outgoing_mode_residual(
        &self,
        stack: &S,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Result<AnalyticResidual<C, D>, Self::Error>;

    /// Evaluate the residual and its first derivative.
    fn outgoing_mode_residual_first_derivative(
        &self,
        stack: &S,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<AnalyticResidual<C, D>, Self::Error>;

    /// Evaluate the residual and its first and second derivatives.
    fn outgoing_mode_residual_second_derivative(
        &self,
        stack: &S,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<AnalyticResidual<C, D>, Self::Error>;
}

/// Backend-neutral analytic residual for an outgoing planar mode problem.
///
/// A mode occurs at a zero of `value`.
///
/// The precise algebra used to construct the residual is backend-specific, but
/// an implementation must provide a locally analytic scalar function suitable
/// for complex root finding.
///
/// Downstream mode-solving crates may use:
///
/// ```text
/// f
/// f′ / f
/// f″ / f - (f′ / f)²
/// ```
///
/// without depending on the backend matrix convention.
#[derive(Clone, Debug, PartialEq)]
pub struct AnalyticResidual<C, D>
where
    D: Dimension,
{
    value: ArrayBase<OwnedRepr<C>, D>,
    derivatives: Option<ResidualDerivatives<C, D>>,
}

impl<C, D> AnalyticResidual<C, D>
where
    D: Dimension,
{
    /// Construct a value-only analytic residual.
    pub fn new(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        Self {
            value,
            derivatives: None,
        }
    }

    /// Construct an analytic residual containing derivatives.
    pub fn with_derivatives(
        value: ArrayBase<OwnedRepr<C>, D>,
        derivatives: ResidualDerivatives<C, D>,
    ) -> Self {
        Self {
            value,
            derivatives: Some(derivatives),
        }
    }

    /// Return the residual value.
    pub fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.value
    }

    /// Return residual derivatives, when requested.
    pub fn derivatives(&self) -> Option<&ResidualDerivatives<C, D>> {
        self.derivatives.as_ref()
    }

    /// Consume the residual and return its value and optional derivatives.
    pub fn into_parts(
        self,
    ) -> (
        ArrayBase<OwnedRepr<C>, D>,
        Option<ResidualDerivatives<C, D>>,
    ) {
        (self.value, self.derivatives)
    }
}

/// First and optional second derivatives of an analytic mode residual.
///
/// The residual is differentiated with respect to one [`DerivativeVariable`].
#[derive(Clone, Debug, PartialEq)]
pub struct ResidualDerivatives<C, D>
where
    D: Dimension,
{
    variable: DerivativeVariable,
    first: ArrayBase<OwnedRepr<C>, D>,
    second: Option<ArrayBase<OwnedRepr<C>, D>>,
}

impl<C, D> ResidualDerivatives<C, D>
where
    D: Dimension,
{
    /// Construct first-order residual derivatives.
    pub fn new(variable: DerivativeVariable, first: ArrayBase<OwnedRepr<C>, D>) -> Self {
        Self {
            variable,
            first,
            second: None,
        }
    }

    /// Attach a second residual derivative.
    pub fn with_second(mut self, second: ArrayBase<OwnedRepr<C>, D>) -> Self {
        self.second = Some(second);
        self
    }

    /// Return the independent derivative variable.
    pub fn variable(&self) -> DerivativeVariable {
        self.variable
    }

    /// Return the first residual derivative.
    pub fn first(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.first
    }

    /// Return the second residual derivative, when available.
    pub fn second(&self) -> Option<&ArrayBase<OwnedRepr<C>, D>> {
        self.second.as_ref()
    }
}
