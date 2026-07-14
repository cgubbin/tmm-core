//! Backend-neutral outgoing-mode residuals.
//!
//! This module defines the interface between planar electromagnetic backends
//! and downstream mode-solving algorithms.
//!
//! A backend-specific implementation constructs a scalar characteristic
//! residual for the homogeneous outgoing-wave problem. A planar mode occurs at
//! a zero of that residual.
//!
//! The backend may internally use a transfer matrix, scattering matrix,
//! Berreman matrix, or modal matching system. Downstream solvers interact only
//! with [`AnalyticResidual`] and therefore do not need to understand the
//! backend representation.
//!
//! This module does not locate or refine modes. Argument-principle integration,
//! contour subdivision, continuation, and root refinement belong in downstream
//! crates.

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::{
        DerivativeVariable, PlanarInput,
        jet::{ArrayJet, ArrayJetFirst},
    },
};

/// Backend capable of constructing an outgoing planar-mode residual.
///
/// The stack's exterior media define the two asymptotic regions.
/// Implementations must impose the appropriate outgoing or decaying condition
/// in each exterior region and return a scalar characteristic residual.
///
/// The returned residual must satisfy the following contract:
///
/// - its zeros correspond to outgoing modes;
/// - it is locally analytic in the requested spectral coordinate away from
///   physical branch points and material singularities;
/// - any backend-dependent normalisation factor is nonzero and analytic over
///   the supported search domain;
/// - returned derivatives refer to the same residual normalisation as the
///   returned value.
///
/// These conditions allow downstream mode solvers to use quantities such as
///
/// ```text
/// f′ / f
/// ```
///
/// without depending on the backend's native matrix representation.
///
/// This trait constructs the residual only. It does not count, locate, or
/// refine its zeros.
pub trait OutgoingModeBackend<C, D, S>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Error produced while constructing the outgoing-mode residual.
    type Error;

    /// Evaluate the outgoing-mode residual.
    fn outgoing_mode_residual(
        &self,
        stack: &S,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Result<AnalyticResidual<C, D>, Self::Error>;

    /// Evaluate the residual and its first derivative with respect to
    /// `variable`.
    fn outgoing_mode_residual_first_derivative(
        &self,
        stack: &S,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<AnalyticResidual<C, D>, Self::Error>;

    /// Evaluate the residual and its first and second derivatives with respect
    /// to `variable`.
    fn outgoing_mode_residual_second_derivative(
        &self,
        stack: &S,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<AnalyticResidual<C, D>, Self::Error>;
}

/// Backend-neutral analytic residual for an outgoing planar-mode problem.
///
/// A mode occurs at a zero of [`value`](Self::value).
///
/// The precise algebra used to construct the residual is backend-specific. A
/// transfer-matrix implementation may evaluate a boundary-conditioned field
/// residual, while another backend may use a characteristic denominator or
/// determinant.
///
/// Downstream mode-solving crates may use:
///
/// ```text
/// f
/// f′ / f
/// f″ / f - (f′ / f)²
/// ```
///
/// without inspecting the backend matrix.
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
    /// Construct a residual without derivatives.
    pub fn new(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        Self {
            value,
            derivatives: None,
        }
    }

    /// Construct a residual containing derivatives.
    pub fn with_derivatives(
        value: ArrayBase<OwnedRepr<C>, D>,
        derivatives: ResidualDerivatives<C, D>,
    ) -> Self {
        Self {
            value,
            derivatives: Some(derivatives),
        }
    }

    /// Construct a residual by consuming a first-order array jet.
    pub(crate) fn from_first_jet(jet: ArrayJetFirst<C, D>, variable: DerivativeVariable) -> Self {
        let (value, first) = jet.into_parts();

        Self::with_derivatives(value, ResidualDerivatives::new(variable, first))
    }

    /// Construct a residual by consuming a second-order array jet.
    pub(crate) fn from_second_jet(jet: ArrayJet<C, D>, variable: DerivativeVariable) -> Self {
        let (value, first, second) = jet.into_parts();

        Self::with_derivatives(
            value,
            ResidualDerivatives::new(variable, first).from_parts(second),
        )
    }

    /// Return the residual value.
    pub fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.value
    }

    /// Return residual derivatives, when available.
    pub fn derivatives(&self) -> Option<&ResidualDerivatives<C, D>> {
        self.derivatives.as_ref()
    }

    /// Consume the residual and return its value.
    ///
    /// Any stored derivatives are discarded.
    pub fn into_value(self) -> ArrayBase<OwnedRepr<C>, D> {
        self.value
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

/// Derivatives of an analytic mode residual.
///
/// A first derivative is always present. A second derivative is present only
/// when the residual was produced by a second-derivative backend method.
///
/// Both derivatives refer to the same [`DerivativeVariable`], evaluation
/// point, and residual normalisation.
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

    /// Attach the corresponding second residual derivative.
    pub fn from_parts(mut self, second: ArrayBase<OwnedRepr<C>, D>) -> Self {
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

    /// Consume the derivative result and return the first derivative.
    ///
    /// Any stored second derivative is discarded.
    pub fn into_first(self) -> ArrayBase<OwnedRepr<C>, D> {
        self.first
    }

    /// Consume the derivative result and return all components.
    pub fn into_parts(
        self,
    ) -> (
        DerivativeVariable,
        ArrayBase<OwnedRepr<C>, D>,
        Option<ArrayBase<OwnedRepr<C>, D>>,
    ) {
        (self.variable, self.first, self.second)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr0;
    use num_complex::Complex64;

    use super::*;

    fn c(value: f64) -> Complex64 {
        Complex64::new(value, 0.0)
    }

    #[test]
    fn value_only_residual_has_no_derivatives() {
        let residual = AnalyticResidual::new(arr0(c(3.0)));

        assert_eq!(residual.value()[()], c(3.0));
        assert!(residual.derivatives().is_none());
    }

    #[test]
    fn first_jet_conversion_preserves_components() {
        let jet = ArrayJetFirst::from_parts(arr0(c(3.0)), arr0(c(5.0)));

        let residual = AnalyticResidual::from_first_jet(jet, DerivativeVariable::VacuumWavenumber);

        assert_eq!(residual.value()[()], c(3.0));

        let derivatives = residual.derivatives().unwrap();

        assert_eq!(derivatives.variable(), DerivativeVariable::VacuumWavenumber);
        assert_eq!(derivatives.first()[()], c(5.0));
        assert!(derivatives.second().is_none());
    }

    #[test]
    fn second_jet_conversion_preserves_components() {
        let jet = ArrayJet::from_parts(arr0(c(3.0)), arr0(c(5.0)), arr0(c(7.0)));

        let residual =
            AnalyticResidual::from_second_jet(jet, DerivativeVariable::ParallelWavenumberSquared);

        assert_eq!(residual.value()[()], c(3.0));

        let derivatives = residual.derivatives().unwrap();

        assert_eq!(
            derivatives.variable(),
            DerivativeVariable::ParallelWavenumberSquared
        );
        assert_eq!(derivatives.first()[()], c(5.0));
        assert_eq!(derivatives.second().unwrap()[()], c(7.0));
    }

    #[test]
    fn residual_derivatives_into_parts_preserves_values() {
        let derivatives = ResidualDerivatives::new(DerivativeVariable::Thickness(2), arr0(c(5.0)))
            .from_parts(arr0(c(7.0)));

        let (variable, first, second) = derivatives.into_parts();

        assert_eq!(variable, DerivativeVariable::Thickness(2));
        assert_eq!(first[()], c(5.0));
        assert_eq!(second.unwrap()[()], c(7.0));
    }

    #[test]
    fn analytic_residual_into_parts_preserves_values() {
        let residual = AnalyticResidual::with_derivatives(
            arr0(c(3.0)),
            ResidualDerivatives::new(DerivativeVariable::VacuumWavenumberSquared, arr0(c(5.0))),
        );

        let (value, derivatives) = residual.into_parts();

        assert_eq!(value[()], c(3.0));
        assert_eq!(derivatives.unwrap().first()[()], c(5.0));
    }
}
