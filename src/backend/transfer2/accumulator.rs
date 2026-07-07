use ndarray::Dimension;

use crate::{
    ComplexScalar,
    backend::transfer2::{
        Matrix2, TransferResult, multiply, multiply_first_derivative, multiply_second_derivative,
    },
};

pub(crate) struct MatrixAccumulator<C, D>
where
    D: Dimension,
{
    matrix: Matrix2<C, D>,
    first: Option<Matrix2<C, D>>,
    second: Option<Matrix2<C, D>>,
}

impl<C, D> MatrixAccumulator<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub fn update(&mut self, layer: &Matrix2<C, D>) {
        self.matrix = multiply(layer, &self.matrix);
    }

    pub fn update_first(&mut self, layer: &Matrix2<C, D>, dlayer: &Matrix2<C, D>) {
        let current = self
            .first
            .take()
            .unwrap_or_else(|| Matrix2::zeros_like(layer.m11()));

        self.first = Some(multiply_first_derivative(
            layer,
            dlayer,
            &self.matrix,
            &current,
        ));

        self.matrix = multiply(layer, &self.matrix);
    }

    pub fn update_second(
        &mut self,
        layer: &Matrix2<C, D>,
        dlayer: &Matrix2<C, D>,
        ddlayer: &Matrix2<C, D>,
    ) {
        let second = self
            .second
            .take()
            .unwrap_or_else(|| Matrix2::zeros_like(layer.m11()));
        let first = self
            .first
            .take()
            .unwrap_or_else(|| Matrix2::zeros_like(layer.m11()));

        self.second = Some(multiply_second_derivative(
            layer,
            dlayer,
            ddlayer,
            &self.matrix,
            &first,
            &second,
        ));

        self.first = Some(multiply_first_derivative(
            layer,
            dlayer,
            &self.matrix,
            &first,
        ));

        self.matrix = multiply(layer, &self.matrix);
    }

    fn finish(self) -> TransferResult<C, D> {
        todo!()
    }
}
