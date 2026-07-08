mod accumulator;
mod backend;
mod input;
mod matrix;
mod quantities;
mod result;
mod spectral;
mod state;
mod thickness;

pub use backend::Transfer2;
pub use input::{DerivativeVariable, Polarisation, Transfer2Input};
pub use matrix::{Matrix2, multiply_first_derivative, multiply_second_derivative};
use quantities::{IsotropicLayerQuantities, isotropic_layer_quantities};
pub use result::{TransferDerivatives, TransferResult};
pub use spectral::{
    frequency_squared_derivative, frequency_squared_second_derivative,
    propagation_constant_squared_derivative, propagation_constant_squared_second_derivative,
};
use state::{BoundaryMode, BoundaryModeDerivatives, FieldState};
use thickness::{
    isotropic_layer_matrix, isotropic_layer_thickness_derivative,
    isotropic_layer_thickness_second_derivative,
};

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{arr0, arr1};
    use num_complex::Complex64;

    use crate::{
        backend::transfer2::{Polarisation, Transfer2, Transfer2Input},
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

        let input = Transfer2Input::new(
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

        let input = Transfer2Input::new(
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

        let input = Transfer2Input::new(
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
            Transfer2Input::new(
                arr0(c(1000.0)),
                arr0(c(0.0)),
                Polarisation::TransverseElectric,
            ),
        );

        let tm = Transfer2::new().solve(
            &stack,
            Transfer2Input::new(
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
