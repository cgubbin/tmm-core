//! Raw matrix backend interfaces and results.
//!
//! This module defines the lowest-level public interface implemented by planar
//! matrix backends.
//!
//! A raw backend exposes its native matrix representation without imposing a
//! backend-independent physical interpretation. Depending on the backend, the
//! returned matrix may be:
//!
//! - a 2×2 transfer matrix,
//! - a 2×2 scattering matrix,
//! - a 4×4 Berreman matrix,
//! - another backend-specific representation.
//!
//! Raw matrix evaluations are useful for debugging, custom matrix analyses,
//! and backend development. Reflection, transmission, and outgoing-mode
//! calculations should normally use the corresponding higher-level capability
//! traits instead.
//!
//! Derivative order is selected by the method called:
//!
//! - [`RawMatrixBackend::solve_matrix`] returns the matrix value;
//! - [`RawMatrixBackend::solve_matrix_first_derivative`] also returns its first
//!   derivative;
//! - [`RawMatrixBackend::solve_matrix_second_derivative`] returns its first and
//!   second derivatives.

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::{
        DerivativeVariable, PlanarInput,
        jet::{Jet, JetFirst},
    },
};

/// Backend capable of exposing its native planar matrix representation.
///
/// This is the lowest-level backend interface. The returned matrix is
/// representation-specific, so callers are responsible for understanding its
/// ordering, basis, and composition convention.
///
/// The input arrays must have matching dimensions. No implicit broadcasting is
/// performed.
pub trait RawMatrixBackend<C, D, S>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Native matrix representation returned by this backend.
    type Matrix;

    /// Error produced while evaluating the backend.
    type Error;

    /// Evaluate the backend's native matrix without derivatives.
    fn solve_matrix(
        &self,
        stack: &S,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error>;

    /// Evaluate the native matrix and its first derivative with respect to
    /// `variable`.
    ///
    /// The returned evaluation contains both the matrix value and its first
    /// derivative.
    fn solve_matrix_first_derivative(
        &self,
        stack: &S,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error>;

    /// Evaluate the native matrix and its first two derivatives with respect
    /// to `variable`.
    ///
    /// The returned evaluation contains the matrix value, first derivative,
    /// and second derivative.
    fn solve_matrix_second_derivative(
        &self,
        stack: &S,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error>;
}

/// Raw matrix evaluation produced by a planar backend.
///
/// This type stores the backend's native matrix representation and any
/// derivatives requested during the same evaluation.
///
/// `M` remains backend-specific. A caller using this type must understand the
/// relevant matrix convention. Higher-level physical workflows should instead
/// use backend-neutral plane-wave or mode-residual interfaces.
#[derive(Clone, Debug, PartialEq)]
pub struct MatrixEvaluation<M> {
    matrix: M,
    derivatives: Option<MatrixDerivatives<M>>,
}

impl<M> MatrixEvaluation<M> {
    /// Construct a value-only matrix evaluation.
    pub fn new(matrix: M) -> Self {
        Self {
            matrix,
            derivatives: None,
        }
    }

    /// Construct a matrix evaluation containing derivatives.
    fn with_derivatives(matrix: M, derivatives: MatrixDerivatives<M>) -> Self {
        Self {
            matrix,
            derivatives: Some(derivatives),
        }
    }

    /// Construct an evaluation by consuming a first-order matrix jet.
    pub(crate) fn from_first_jet(jet: JetFirst<M>, variable: DerivativeVariable) -> Self {
        let (matrix, first) = jet.into_parts();

        Self::with_derivatives(matrix, MatrixDerivatives::new(variable, first))
    }

    /// Construct an evaluation by consuming a second-order matrix jet.
    pub(crate) fn from_second_jet(jet: Jet<M>, variable: DerivativeVariable) -> Self {
        let (matrix, first, second) = jet.into_parts();

        Self::with_derivatives(
            matrix,
            MatrixDerivatives::new(variable, first).from_parts(second),
        )
    }

    /// Return the backend-specific matrix.
    pub fn matrix(&self) -> &M {
        &self.matrix
    }

    /// Return the matrix derivatives, when available.
    pub fn derivatives(&self) -> Option<&MatrixDerivatives<M>> {
        self.derivatives.as_ref()
    }

    /// Consume the evaluation and return its matrix.
    ///
    /// Any stored derivatives are discarded.
    pub fn into_matrix(self) -> M {
        self.matrix
    }

    /// Consume the evaluation and return the matrix and optional derivatives.
    pub fn into_parts(self) -> (M, Option<MatrixDerivatives<M>>) {
        (self.matrix, self.derivatives)
    }
}

/// Derivatives of a backend-specific matrix.
///
/// A first derivative is always present. A second derivative is present only
/// when the evaluation was produced by a second-derivative backend method.
///
/// Both derivatives refer to the same [`DerivativeVariable`] and evaluation
/// point.
#[derive(Clone, Debug, PartialEq)]
pub struct MatrixDerivatives<M> {
    variable: DerivativeVariable,
    first: M,
    second: Option<M>,
}

impl<M> MatrixDerivatives<M> {
    /// Construct a first-order matrix derivative result.
    pub fn new(variable: DerivativeVariable, first: M) -> Self {
        Self {
            variable,
            first,
            second: None,
        }
    }

    /// Attach the corresponding second derivative.
    pub fn from_parts(mut self, second: M) -> Self {
        self.second = Some(second);
        self
    }

    /// Return the independent derivative variable.
    pub fn variable(&self) -> DerivativeVariable {
        self.variable
    }

    /// Return the first matrix derivative.
    pub fn first(&self) -> &M {
        &self.first
    }

    /// Return the second matrix derivative, when available.
    pub fn second(&self) -> Option<&M> {
        self.second.as_ref()
    }

    /// Consume the result and return the first derivative.
    ///
    /// Any stored second derivative is discarded.
    pub fn into_first(self) -> M {
        self.first
    }

    /// Consume the result and return all components.
    pub fn into_parts(self) -> (DerivativeVariable, M, Option<M>) {
        (self.variable, self.first, self.second)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_only_evaluation_contains_no_derivatives() {
        let evaluation = MatrixEvaluation::new(3_u32);

        assert_eq!(*evaluation.matrix(), 3);
        assert!(evaluation.derivatives().is_none());
    }

    #[test]
    fn first_jet_conversion_preserves_value_and_derivative() {
        let jet = JetFirst::from_parts(3_u32, 5_u32);

        let evaluation =
            MatrixEvaluation::from_first_jet(jet, DerivativeVariable::VacuumWavenumber);

        assert_eq!(*evaluation.matrix(), 3);

        let derivatives = evaluation.derivatives().unwrap();
        assert_eq!(derivatives.variable(), DerivativeVariable::VacuumWavenumber);
        assert_eq!(*derivatives.first(), 5);
        assert!(derivatives.second().is_none());
    }

    #[test]
    fn second_jet_conversion_preserves_all_components() {
        let jet = Jet::from_parts(3_u32, 5_u32, 7_u32);

        let evaluation =
            MatrixEvaluation::from_second_jet(jet, DerivativeVariable::ParallelWavenumberSquared);

        assert_eq!(*evaluation.matrix(), 3);

        let derivatives = evaluation.derivatives().unwrap();
        assert_eq!(
            derivatives.variable(),
            DerivativeVariable::ParallelWavenumberSquared
        );
        assert_eq!(*derivatives.first(), 5);
        assert_eq!(derivatives.second(), Some(&7));
    }

    #[test]
    fn into_parts_preserves_matrix_and_derivatives() {
        let evaluation = MatrixEvaluation::with_derivatives(
            3_u32,
            MatrixDerivatives::new(DerivativeVariable::Thickness(2), 5_u32).from_parts(7_u32),
        );

        let (matrix, derivatives) = evaluation.into_parts();

        assert_eq!(matrix, 3);

        let (variable, first, second) = derivatives.unwrap().into_parts();

        assert_eq!(variable, DerivativeVariable::Thickness(2));
        assert_eq!(first, 5);
        assert_eq!(second, Some(7));
    }

    #[test]
    fn into_matrix_discards_derivatives() {
        let evaluation = MatrixEvaluation::with_derivatives(
            3_u32,
            MatrixDerivatives::new(DerivativeVariable::VacuumWavenumberSquared, 5_u32),
        );

        assert_eq!(evaluation.into_matrix(), 3);
    }
}
