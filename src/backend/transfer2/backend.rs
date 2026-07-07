use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{ComplexScalar, material::Material, stack::Stack};

use super::{Transfer2Input, TransferResult, identity_matrix, isotropic_layer_matrix, multiply};

#[derive(Copy, Clone, Debug, Default)]
pub struct Transfer2;

impl Transfer2 {
    pub fn new() -> Self {
        Self
    }

    pub fn solve<M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: Transfer2Input<ArrayBase<OwnedRepr<C>, D>>,
    ) -> TransferResult<C, D>
    where
        M: Material<Real = C::RealField>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        let mut matrix = identity_matrix(&input.wavenumber);

        for layer in stack.layers_in_propagation_order() {
            let layer_matrix = isotropic_layer_matrix(
                layer.material(),
                layer.thickness(),
                &input.wavenumber,
                &input.propagation_constant_squared,
                input.polarisation,
            );

            matrix = multiply(&layer_matrix, &matrix);
        }

        TransferResult::new(matrix)
    }
}
