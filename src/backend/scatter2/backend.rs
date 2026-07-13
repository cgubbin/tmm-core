use crate::{
    ComplexScalar,
    backend::{
        MatrixEvaluation, PlanarInput,
        isotropic::IsotropicLayerQuantities,
        scatter2::{accumulator::ScatterAccumulator, matrix::ScatterMatrix2},
    },
    material::Material,
    stack::Stack,
};

use ndarray::{ArrayBase, Dimension, OwnedRepr};

#[derive(Copy, Clone, Debug, Default)]
pub struct Scatter2;

impl Scatter2 {
    pub fn new() -> Self {
        Self
    }

    pub fn solve<M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> MatrixEvaluation<ScatterMatrix2<C, D>>
    where
        M: Material<Real = C::RealField>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        let mut accumulator = ScatterAccumulator::new(&input.vacuum_wavenumber);

        for layer in stack.layers_in_propagation_order() {
            //     let q = IsotropicLayerQuantities::new(layer.material(), &input);

            //     let layer_matrix = ScatterMatrix2::from_layer(&q, layer.thickness());

            //     accumulator.update(&layer_matrix);
            todo!()
        }

        accumulator.finish()
    }
}
