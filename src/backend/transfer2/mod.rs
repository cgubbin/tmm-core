mod backend;
mod error;
mod jet;
mod matrix;
mod mode;
mod plane_wave;
mod response;

pub use error::TransferError;
pub use matrix::Matrix2;

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::{
        DerivativeVariable, MatrixEvaluation, PlanarInput, RawMatrixBackend,
        transfer2::backend::Transfer2,
    },
    material::Material,
    stack::Stack,
};

impl<C, D, M> RawMatrixBackend<C, D, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    D: Dimension,
    M: Material<Real = C::RealField>,
    C::RealField: Copy,
{
    type Matrix = Matrix2<C, D>;
    type Error = TransferError;

    fn solve_matrix(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        self.evaluate(stack, input).map(MatrixEvaluation::new)
    }

    fn solve_matrix_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        self.evaluate_first(stack, input, variable)
            .map(|j| MatrixEvaluation::from_first_jet(j, variable))
    }

    fn solve_matrix_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        self.evaluate_second(stack, input, variable)
            .map(|j| MatrixEvaluation::from_second_jet(j, variable))
    }
}
