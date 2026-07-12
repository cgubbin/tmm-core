use crate::backend::{DerivativeVariable, PlanarInput};

use ndarray::{ArrayBase, Dimension, OwnedRepr};

/// Backend capable of exposing its native planar matrix representation.
///
/// This is the lowest-level backend interface. It is appropriate for:
///
/// - debugging,
/// - custom matrix analyses,
/// - backend development,
/// - consumers that deliberately depend on a particular representation.
///
/// It is not the preferred interface for backend-independent reflection,
/// transmission, or mode-solving code.
pub trait RawMatrixBackend<C, D, S>
where
    D: Dimension,
{
    /// Native matrix representation returned by this backend.
    type Matrix;

    /// Error produced while evaluating the backend.
    type Error;

    /// Evaluate the native matrix without derivatives.
    fn solve_matrix(
        &self,
        stack: &S,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error>;

    /// Evaluate the native matrix and its first derivative.
    ///
    /// The returned evaluation contains both the value and first derivative.
    fn solve_matrix_first_derivative(
        &self,
        stack: &S,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error>;

    /// Evaluate the native matrix and its first and second derivatives.
    ///
    /// The returned evaluation contains the value, first derivative, and
    /// second derivative with respect to `variable`.
    fn solve_matrix_second_derivative(
        &self,
        stack: &S,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error>;
}

/// Raw matrix evaluation produced by a planar backend.
///
/// This type contains the backend's native matrix representation together with
/// optional derivatives of that representation.
///
/// It is intentionally representation-specific. A transfer-matrix backend may
/// return a 2×2 or 4×4 transfer matrix, while a scattering backend may return a
/// block scattering matrix.
///
/// Callers using this type are responsible for understanding the backend's
/// matrix convention. Backend-independent physical workflows should use
/// [`PlaneWaveBackend`] or [`OutgoingModeBackend`] instead.
#[derive(Clone, Debug, PartialEq)]
pub struct MatrixEvaluation<M> {
    matrix: M,
    pub(crate) derivatives: Option<MatrixDerivatives<M>>,
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
    pub fn with_derivatives(matrix: M, derivatives: MatrixDerivatives<M>) -> Self {
        Self {
            matrix,
            derivatives: Some(derivatives),
        }
    }

    /// Return the backend-specific matrix.
    pub fn matrix(&self) -> &M {
        &self.matrix
    }

    /// Return the matrix derivatives, when requested.
    pub fn derivatives(&self) -> Option<&MatrixDerivatives<M>> {
        self.derivatives.as_ref()
    }

    /// Consume the evaluation and return the raw matrix.
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

/// First and optional second derivatives of a backend-specific matrix.
///
/// The first derivative is always present. The second derivative is present
/// only when the evaluation was produced by a second-derivative solve.
///
/// `M` is the backend's raw matrix representation.
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

    /// Attach a second matrix derivative.
    ///
    /// The first and second matrices must refer to the same independent
    /// variable and evaluation point.
    pub fn with_second(mut self, second: M) -> Self {
        self.second = Some(second);
        self
    }

    /// Return the independent derivative variable.
    pub fn variable(&self) -> DerivativeVariable {
        self.variable
    }

    /// Return the first derivative of the raw matrix.
    pub fn first(&self) -> &M {
        &self.first
    }

    /// Return the second derivative of the raw matrix, when available.
    pub fn second(&self) -> Option<&M> {
        self.second.as_ref()
    }

    /// Consume the derivative result and return its components.
    pub fn into_parts(self) -> (DerivativeVariable, M, Option<M>) {
        (self.variable, self.first, self.second)
    }
}
