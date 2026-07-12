mod accumulator;
mod backend;
mod derivatives;
mod error;
mod matrix;
mod result;
mod state;

pub use error::TransferError;
pub use matrix::{Matrix2, multiply_first_derivative, multiply_second_derivative};
use state::{BoundaryMode, BoundaryModeDerivatives, FieldState};

use crate::{
    ComplexScalar,
    backend::{
        DerivativeVariable, MatrixEvaluation, PlanarInput, RawMatrixBackend,
        transfer2::backend::Transfer2,
    },
    material::Material,
    stack::Stack,
};

use ndarray::{ArrayBase, Dimension, OwnedRepr};

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
        Ok(self.solve(stack, input.clone()))
    }

    fn solve_matrix_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        self.solve_first_derivative(stack, input.clone(), variable)
    }

    fn solve_matrix_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<MatrixEvaluation<Self::Matrix>, Self::Error> {
        self.solve_second_derivative(stack, input.clone(), variable)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{arr0, arr1};
    use num_complex::Complex64;

    use crate::{
        backend::transfer2::Transfer2,
        backend::{PlanarInput, Polarisation},
        material::{Constant, IsotropicMaterial},
        stack::{Stack, Thickness, ValidationConfig},
    };

    type C = Complex64;

    fn c(x: f64) -> C {
        C::new(x, 0.0)
    }

    #[test]
    fn empty_stack_has_identity_matrix() {
        let air = IsotropicMaterial::from(Constant::new(1.0));

        let stack = Stack::builder(air.clone(), air)
            .validation(ValidationConfig::permissive())
            .build()
            .unwrap();

        let input = PlanarInput::new(
            arr0(c(1000.0)),
            arr0(c(0.0)),
            Polarisation::TransverseElectric,
        );

        let result = Transfer2::new().solve(&stack, input);
        let m = result.matrix();

        assert_relative_eq!(m.m11()[()], c(1.0));
        assert_relative_eq!(m.m12()[()], c(0.0));
        assert_relative_eq!(m.m21()[()], c(0.0));
        assert_relative_eq!(m.m22()[()], c(1.0));

        assert_relative_eq!(result.determinant()[()], c(1.0));
    }

    #[test]
    fn zero_thickness_layer_is_identity() {
        let air = IsotropicMaterial::from(Constant::new(1.0));
        let layer = IsotropicMaterial::from(Constant::new(2.25));

        let stack = Stack::builder(air.clone(), air)
            .with_layer(layer, Thickness::zero())
            .validation(crate::stack::ValidationConfig::permissive())
            .build()
            .unwrap();

        let input = PlanarInput::new(
            arr0(c(1000.0)),
            arr0(c(0.0)),
            Polarisation::TransverseElectric,
        );

        let result = Transfer2::new().solve(&stack, input);
        let m = result.matrix();

        assert_relative_eq!(m.m11()[()], c(1.0), max_relative = 1e-12);
        assert_relative_eq!(m.m12()[()], c(0.0), max_relative = 1e-12);
        assert_relative_eq!(m.m21()[()], c(0.0), max_relative = 1e-12);
        assert_relative_eq!(m.m22()[()], c(1.0), max_relative = 1e-12);
    }

    #[test]
    fn ndarray_input_shape_is_preserved() {
        let air = IsotropicMaterial::from(Constant::new(1.0));
        let layer = IsotropicMaterial::from(Constant::new(2.25));

        let stack = Stack::builder(air.clone(), air)
            .with_layer(layer, Thickness::from_nm(100.0).unwrap())
            .build()
            .unwrap();

        let input = PlanarInput::new(
            arr1(&[c(1000.0), c(1200.0), c(1400.0)]),
            arr1(&[c(0.0), c(0.0), c(0.0)]),
            Polarisation::TransverseElectric,
        );

        let result = Transfer2::new().solve(&stack, input);

        assert_eq!(result.matrix().m11().shape(), &[3]);
        assert_eq!(result.matrix().m12().shape(), &[3]);
        assert_eq!(result.matrix().m21().shape(), &[3]);
        assert_eq!(result.matrix().m22().shape(), &[3]);
        assert_eq!(result.determinant().shape(), &[3]);
    }

    #[test]
    fn normal_incidence_te_and_tm_have_same_determinant_for_nonmagnetic_isotropic_layer() {
        let air = IsotropicMaterial::from(Constant::new(1.0));
        let layer = IsotropicMaterial::from(Constant::new(2.25));

        let stack = Stack::builder(air.clone(), air)
            .with_layer(layer, Thickness::from_nm(100.0).unwrap())
            .build()
            .unwrap();

        let te = Transfer2::new().solve(
            &stack,
            PlanarInput::new(
                arr0(c(1000.0)),
                arr0(c(0.0)),
                Polarisation::TransverseElectric,
            ),
        );

        let tm = Transfer2::new().solve(
            &stack,
            PlanarInput::new(
                arr0(c(1000.0)),
                arr0(c(0.0)),
                Polarisation::TransverseMagnetic,
            ),
        );

        assert_relative_eq!(
            te.determinant()[()],
            tm.determinant()[()],
            max_relative = 1e-12
        );
    }
}
