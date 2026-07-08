//! Stack-level 2×2 transfer-matrix backend.
//!
//! This module connects the layer-level isotropic transfer-matrix kernel to a
//! [`Stack`](crate::stack::Stack). It is responsible for:
//!
//! - iterating over finite-thickness layers,
//! - multiplying layer matrices in propagation order,
//! - accumulating requested thickness derivatives,
//! - returning a [`TransferResult`].
//!
//! It deliberately does **not** apply external boundary conditions. The backend
//! computes the total stack matrix only. Outgoing-wave residuals, reflection,
//! transmission, and mode-finding functions are post-processing operations on
//! [`TransferResult`].
//!
//! Matrix accumulation follows:
//!
//! ```text
//! M_total = L_N ... L_2 L_1
//! ```
//!
//! where the layer order is supplied by the stack.
//!
//! For a derivative with respect to layer thickness `d_j`, only layer `j` has
//! non-zero local derivatives. Other layers contribute zero local derivative
//! matrices, but still participate in the product-rule accumulation.

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::transfer2::{
        DerivativeVariable, Matrix2, accumulator::MatrixAccumulator,
        isotropic_layer_frequency_squared_derivative,
        isotropic_layer_frequency_squared_second_derivative,
        isotropic_layer_propagation_constant_squared_derivative,
        isotropic_layer_propagation_constant_squared_second_derivative,
        isotropic_layer_thickness_derivative, isotropic_layer_thickness_second_derivative,
        layer::isotropic_layer_quantities,
    },
    material::Material,
    stack::Stack,
};

use super::{Transfer2Input, TransferResult, isotropic_layer_matrix};

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
        let mut accumulator = MatrixAccumulator::new(&input.wavenumber);

        for layer in stack.layers_in_propagation_order() {
            let layer_matrix = isotropic_layer_matrix(
                layer.material(),
                layer.thickness(),
                &input.wavenumber,
                &input.propagation_constant_squared,
                input.polarisation,
            );

            accumulator.update(&layer_matrix);
        }

        accumulator.finish()
    }

    pub fn solve_first_derivative<M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: Transfer2Input<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> TransferResult<C, D>
    where
        M: Material<Real = C::RealField>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        let mut accumulator = MatrixAccumulator::new(&input.wavenumber);

        let requested_variable = variable;
        let primitive_variable = variable.primitive();

        for (index, layer) in stack.layers_in_propagation_order().enumerate() {
            let layer_matrix = isotropic_layer_matrix(
                layer.material(),
                layer.thickness(),
                &input.wavenumber,
                &input.propagation_constant_squared,
                input.polarisation,
            );

            match primitive_variable {
                DerivativeVariable::Thickness(layer_index) if layer_index == index => {
                    let dlayer = isotropic_layer_thickness_derivative(
                        layer.material(),
                        layer.thickness(),
                        &input.wavenumber,
                        &input.propagation_constant_squared,
                        input.polarisation,
                    );
                    accumulator.update_first(primitive_variable, &layer_matrix, &dlayer);
                }
                DerivativeVariable::Thickness(_) => {
                    let zero = Matrix2::zeros_like(layer_matrix.m11());
                    accumulator.update_first(primitive_variable, &layer_matrix, &zero);
                }
                DerivativeVariable::FrequencySquared => {
                    let dlayer = isotropic_layer_frequency_squared_derivative(
                        layer.material(),
                        layer.thickness(),
                        &input.wavenumber,
                        &input.propagation_constant_squared,
                        input.polarisation,
                    );

                    accumulator.update_first(primitive_variable, &layer_matrix, &dlayer);
                }

                DerivativeVariable::PropagationConstantSquared => {
                    let dlayer = isotropic_layer_propagation_constant_squared_derivative(
                        layer.material(),
                        layer.thickness(),
                        &input.wavenumber,
                        &input.propagation_constant_squared,
                        input.polarisation,
                    );

                    accumulator.update_first(primitive_variable, &layer_matrix, &dlayer);
                }

                DerivativeVariable::Frequency | DerivativeVariable::PropagationConstant => {
                    unreachable!("linear spectral derivatives are not primitives")
                }
            }
        }

        let result = accumulator.finish();

        if let Some((first, second)) = chain_rule_coefficients(requested_variable, &input) {
            result.chain_rule(requested_variable, first, second)
        } else {
            result
        }
    }

    pub fn solve_second_derivative<M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: Transfer2Input<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> TransferResult<C, D>
    where
        M: Material<Real = C::RealField>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        let mut accumulator = MatrixAccumulator::new(&input.wavenumber);

        let requested_variable = variable;
        let primitive_variable = variable.primitive();

        for (index, layer) in stack.layers_in_propagation_order().enumerate() {
            let layer_matrix = isotropic_layer_matrix(
                layer.material(),
                layer.thickness(),
                &input.wavenumber,
                &input.propagation_constant_squared,
                input.polarisation,
            );

            match primitive_variable {
                DerivativeVariable::Thickness(layer_index) if layer_index == index => {
                    let dlayer = isotropic_layer_thickness_derivative(
                        layer.material(),
                        layer.thickness(),
                        &input.wavenumber,
                        &input.propagation_constant_squared,
                        input.polarisation,
                    );

                    let ddlayer = isotropic_layer_thickness_second_derivative(
                        layer.material(),
                        layer.thickness(),
                        &input.wavenumber,
                        &input.propagation_constant_squared,
                        input.polarisation,
                    );
                    accumulator.update_second(primitive_variable, &layer_matrix, &dlayer, &ddlayer);
                }
                DerivativeVariable::Thickness(_) => {
                    let zero = Matrix2::zeros_like(layer_matrix.m11());
                    let zero = Matrix2::zeros_like(layer_matrix.m11());
                    accumulator.update_second(variable, &layer_matrix, &zero, &zero);
                }
                DerivativeVariable::FrequencySquared => {
                    let dlayer = isotropic_layer_frequency_squared_derivative(
                        layer.material(),
                        layer.thickness(),
                        &input.wavenumber,
                        &input.propagation_constant_squared,
                        input.polarisation,
                    );

                    let ddlayer = isotropic_layer_frequency_squared_second_derivative(
                        layer.material(),
                        layer.thickness(),
                        &input.wavenumber,
                        &input.propagation_constant_squared,
                        input.polarisation,
                    );
                    accumulator.update_second(primitive_variable, &layer_matrix, &dlayer, &ddlayer);
                }
                DerivativeVariable::PropagationConstantSquared => {
                    let dlayer = isotropic_layer_propagation_constant_squared_derivative(
                        layer.material(),
                        layer.thickness(),
                        &input.wavenumber,
                        &input.propagation_constant_squared,
                        input.polarisation,
                    );

                    let ddlayer = isotropic_layer_propagation_constant_squared_second_derivative(
                        layer.material(),
                        layer.thickness(),
                        &input.wavenumber,
                        &input.propagation_constant_squared,
                        input.polarisation,
                    );
                    accumulator.update_second(primitive_variable, &layer_matrix, &dlayer, &ddlayer);
                }
                DerivativeVariable::Frequency | DerivativeVariable::PropagationConstant => {
                    unreachable!("linear spectral derivatives are not primitives")
                }
            }
        }

        let result = accumulator.finish();

        if let Some((first, second)) = chain_rule_coefficients(requested_variable, &input) {
            result.chain_rule(requested_variable, first, second)
        } else {
            result
        }
    }
}

fn chain_rule_coefficients<C, D>(
    variable: DerivativeVariable,
    input: &Transfer2Input<ArrayBase<OwnedRepr<C>, D>>,
) -> Option<(ArrayBase<OwnedRepr<C>, D>, ArrayBase<OwnedRepr<C>, D>)>
where
    C: ComplexScalar,
    D: Dimension,
{
    let two = C::one() + C::one();

    match variable {
        DerivativeVariable::Frequency => {
            let first = input.wavenumber.mapv(|w| two * w);
            let second = input.wavenumber.mapv(|_| two);
            Some((first, second))
        }

        DerivativeVariable::PropagationConstant => {
            let beta = input.propagation_constant_squared.mapv(|b2| b2.sqrt());
            let first = beta.mapv(|b| two * b);
            let second = beta.mapv(|_| two);
            Some((first, second))
        }

        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{ArrayBase, Dimension, OwnedRepr, arr0, arr1};
    use num_complex::Complex64;

    use std::ops::{Add, Mul};

    use super::*;
    use crate::{
        backend::transfer2::{
            Matrix2, Polarisation, isotropic_layer_matrix, isotropic_layer_thickness_derivative,
            isotropic_layer_thickness_second_derivative,
        },
        material::{Constant, IsotropicMaterial},
        stack::{Stack, Thickness, ValidationConfig},
    };

    type C = Complex64;

    fn c(x: f64) -> C {
        C::new(x, 0.0)
    }

    fn assert_array_close<D>(
        actual: &ArrayBase<OwnedRepr<C>, D>,
        expected: &ArrayBase<OwnedRepr<C>, D>,
        tolerance: f64,
    ) where
        D: Dimension,
    {
        assert_eq!(actual.shape(), expected.shape());

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(
                actual.re,
                expected.re,
                max_relative = tolerance,
                epsilon = tolerance
            );
            assert_relative_eq!(
                actual.im,
                expected.im,
                max_relative = tolerance,
                epsilon = tolerance
            );
        }
    }

    fn assert_matrix_close<D>(actual: &Matrix2<C, D>, expected: &Matrix2<C, D>, tolerance: f64)
    where
        D: Dimension,
    {
        assert_array_close(actual.m11(), expected.m11(), tolerance);
        assert_array_close(actual.m12(), expected.m12(), tolerance);
        assert_array_close(actual.m21(), expected.m21(), tolerance);
        assert_array_close(actual.m22(), expected.m22(), tolerance);
    }

    fn air() -> IsotropicMaterial<f64> {
        Constant::new(1.0).into()
    }

    fn mat(epsilon: f64) -> IsotropicMaterial<f64> {
        Constant::new(epsilon).into()
    }

    fn input0() -> Transfer2Input<ndarray::Array0<C>> {
        Transfer2Input::new(
            arr0(c(1000.0)),
            arr0(c(10.0)),
            Polarisation::TransverseElectric,
        )
    }

    #[test]
    fn empty_stack_has_identity_matrix_when_validation_allows_empty() {
        let stack = Stack::builder(air(), air())
            .validation(ValidationConfig::permissive())
            .build()
            .unwrap();

        let input = input0();

        let result = Transfer2::new().solve(&stack, input);

        let expected = Matrix2::identity_like(result.matrix().m11());

        assert_matrix_close(result.matrix(), &expected, 1e-12);
        assert_relative_eq!(result.determinant()[()], c(1.0), max_relative = 1e-12);
    }

    #[test]
    fn one_layer_stack_matches_layer_matrix() {
        let layer = mat(2.25);
        let thickness = Thickness::from_nm(100.0).unwrap();

        let stack = Stack::builder(air(), air())
            .with_layer(layer.clone(), thickness)
            .build()
            .unwrap();

        let input = input0();

        let result = Transfer2::new().solve(&stack, input.clone());

        let expected = isotropic_layer_matrix(
            &layer,
            thickness,
            &input.wavenumber,
            &input.propagation_constant_squared,
            input.polarisation,
        );

        assert_matrix_close(result.matrix(), &expected, 1e-12);
    }

    #[test]
    fn two_layer_stack_matches_manual_product() {
        let layer0 = mat(2.25);
        let layer1 = mat(3.24);

        let d0 = Thickness::from_nm(100.0).unwrap();
        let d1 = Thickness::from_nm(50.0).unwrap();

        let stack = Stack::builder(air(), air())
            .with_layer(layer0.clone(), d0)
            .with_layer(layer1.clone(), d1)
            .build()
            .unwrap();

        let input = input0();

        let result = Transfer2::new().solve(&stack, input.clone());

        let m0 = isotropic_layer_matrix(
            &layer0,
            d0,
            &input.wavenumber,
            &input.propagation_constant_squared,
            input.polarisation,
        );

        let m1 = isotropic_layer_matrix(
            &layer1,
            d1,
            &input.wavenumber,
            &input.propagation_constant_squared,
            input.polarisation,
        );

        let expected = &m1 * &m0;

        assert_matrix_close(result.matrix(), &expected, 1e-12);
    }

    #[test]
    fn ndarray_input_shape_is_preserved() {
        let stack = Stack::builder(air(), air())
            .with_layer(mat(2.25), Thickness::from_nm(100.0).unwrap())
            .with_layer(mat(3.24), Thickness::from_nm(50.0).unwrap())
            .build()
            .unwrap();

        let input = Transfer2Input::new(
            arr1(&[c(900.0), c(1000.0), c(1100.0)]),
            arr1(&[c(0.0), c(10.0), c(20.0)]),
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
    fn thickness_first_derivative_for_layer_zero_matches_manual_product_rule() {
        let layer0 = mat(2.25);
        let layer1 = mat(3.24);

        let d0 = Thickness::from_nm(100.0).unwrap();
        let d1 = Thickness::from_nm(50.0).unwrap();

        let stack = Stack::builder(air(), air())
            .with_layer(layer0.clone(), d0)
            .with_layer(layer1.clone(), d1)
            .build()
            .unwrap();

        let input = input0();
        let variable = DerivativeVariable::Thickness(0);

        let result = Transfer2::new().solve_first_derivative(&stack, input.clone(), variable);

        let m0 = isotropic_layer_matrix(
            &layer0,
            d0,
            &input.wavenumber,
            &input.propagation_constant_squared,
            input.polarisation,
        );

        let dm0 = isotropic_layer_thickness_derivative(
            &layer0,
            d0,
            &input.wavenumber,
            &input.propagation_constant_squared,
            input.polarisation,
        );

        let m1 = isotropic_layer_matrix(
            &layer1,
            d1,
            &input.wavenumber,
            &input.propagation_constant_squared,
            input.polarisation,
        );

        let expected = &m1 * &dm0;

        assert_matrix_close(result.derivatives().unwrap().first(), &expected, 1e-12);
    }

    #[test]
    fn thickness_first_derivative_for_layer_one_matches_manual_product_rule() {
        let layer0 = mat(2.25);
        let layer1 = mat(3.24);

        let d0 = Thickness::from_nm(100.0).unwrap();
        let d1 = Thickness::from_nm(50.0).unwrap();

        let stack = Stack::builder(air(), air())
            .with_layer(layer0.clone(), d0)
            .with_layer(layer1.clone(), d1)
            .build()
            .unwrap();

        let input = input0();
        let variable = DerivativeVariable::Thickness(1);

        let result = Transfer2::new().solve_first_derivative(&stack, input.clone(), variable);

        let m0 = isotropic_layer_matrix(
            &layer0,
            d0,
            &input.wavenumber,
            &input.propagation_constant_squared,
            input.polarisation,
        );

        let m1 = isotropic_layer_matrix(
            &layer1,
            d1,
            &input.wavenumber,
            &input.propagation_constant_squared,
            input.polarisation,
        );

        let dm1 = isotropic_layer_thickness_derivative(
            &layer1,
            d1,
            &input.wavenumber,
            &input.propagation_constant_squared,
            input.polarisation,
        );

        let expected = &dm1 * &m0;

        assert_matrix_close(result.derivatives().unwrap().first(), &expected, 1e-12);
    }

    #[test]
    fn thickness_first_derivative_matches_finite_difference() {
        let layer0 = mat(2.25);
        let layer1 = mat(3.24);

        let d0_nm = 100.0;
        let d1_nm = 50.0;
        let h_nm = 1e-3;

        let variable = DerivativeVariable::Thickness(0);
        let input = input0();

        let stack = Stack::builder(air(), air())
            .with_layer(layer0.clone(), Thickness::from_nm(d0_nm).unwrap())
            .with_layer(layer1.clone(), Thickness::from_nm(d1_nm).unwrap())
            .build()
            .unwrap();

        let analytical = Transfer2::new()
            .solve_first_derivative(&stack, input.clone(), variable)
            .derivatives()
            .unwrap()
            .first()
            .clone();

        let plus_stack = Stack::builder(air(), air())
            .with_layer(layer0.clone(), Thickness::from_nm(d0_nm + h_nm).unwrap())
            .with_layer(layer1.clone(), Thickness::from_nm(d1_nm).unwrap())
            .build()
            .unwrap();

        let minus_stack = Stack::builder(air(), air())
            .with_layer(layer0.clone(), Thickness::from_nm(d0_nm - h_nm).unwrap())
            .with_layer(layer1.clone(), Thickness::from_nm(d1_nm).unwrap())
            .build()
            .unwrap();

        let plus = Transfer2::new().solve(&plus_stack, input.clone());
        let minus = Transfer2::new().solve(&minus_stack, input);

        let h_cm = Thickness::from_nm(h_nm).unwrap().as_cm();

        let expected =
            (&plus.matrix().add(&minus.matrix().scale(c(-1.0)))).scale(c(1.0 / (2.0 * h_cm)));

        assert_matrix_close(&analytical, &expected, 1e-6);
    }

    #[test]
    fn thickness_second_derivative_for_layer_zero_matches_manual_product_rule() {
        let layer0 = mat(2.25);
        let layer1 = mat(3.24);

        let d0 = Thickness::from_nm(100.0).unwrap();
        let d1 = Thickness::from_nm(50.0).unwrap();

        let stack = Stack::builder(air(), air())
            .with_layer(layer0.clone(), d0)
            .with_layer(layer1.clone(), d1)
            .build()
            .unwrap();

        let input = input0();
        let variable = DerivativeVariable::Thickness(0);

        let result = Transfer2::new().solve_second_derivative(&stack, input.clone(), variable);

        let ddm0 = isotropic_layer_thickness_second_derivative(
            &layer0,
            d0,
            &input.wavenumber,
            &input.propagation_constant_squared,
            input.polarisation,
        );

        let m1 = isotropic_layer_matrix(
            &layer1,
            d1,
            &input.wavenumber,
            &input.propagation_constant_squared,
            input.polarisation,
        );

        let expected = &m1 * &ddm0;

        assert_matrix_close(
            result.derivatives().unwrap().second().unwrap(),
            &expected,
            1e-12,
        );
    }

    #[test]
    fn thickness_second_derivative_matches_finite_difference() {
        let layer0 = mat(2.25);
        let layer1 = mat(3.24);

        let d0_nm = 100.0;
        let d1_nm = 50.0;
        let h_nm = 1e-2;

        let variable = DerivativeVariable::Thickness(0);
        let input = input0();

        let stack = Stack::builder(air(), air())
            .with_layer(layer0.clone(), Thickness::from_nm(d0_nm).unwrap())
            .with_layer(layer1.clone(), Thickness::from_nm(d1_nm).unwrap())
            .build()
            .unwrap();

        let analytical = Transfer2::new()
            .solve_second_derivative(&stack, input.clone(), variable)
            .derivatives()
            .unwrap()
            .second()
            .unwrap()
            .clone();

        let plus_stack = Stack::builder(air(), air())
            .with_layer(layer0.clone(), Thickness::from_nm(d0_nm + h_nm).unwrap())
            .with_layer(layer1.clone(), Thickness::from_nm(d1_nm).unwrap())
            .build()
            .unwrap();

        let zero_stack = Stack::builder(air(), air())
            .with_layer(layer0.clone(), Thickness::from_nm(d0_nm).unwrap())
            .with_layer(layer1.clone(), Thickness::from_nm(d1_nm).unwrap())
            .build()
            .unwrap();

        let minus_stack = Stack::builder(air(), air())
            .with_layer(layer0.clone(), Thickness::from_nm(d0_nm - h_nm).unwrap())
            .with_layer(layer1.clone(), Thickness::from_nm(d1_nm).unwrap())
            .build()
            .unwrap();

        let plus = Transfer2::new().solve(&plus_stack, input.clone());
        let zero = Transfer2::new().solve(&zero_stack, input.clone());
        let minus = Transfer2::new().solve(&minus_stack, input);

        let h_cm = Thickness::from_nm(h_nm).unwrap().as_cm();

        let expected = (&plus
            .matrix()
            .add(&zero.matrix().scale(c(-2.0)))
            .add(minus.matrix()))
            .scale(c(1.0 / (h_cm * h_cm)));

        assert_matrix_close(&analytical, &expected, 1e-4);
    }

    #[test]
    fn frequency_squared_derivative_matches_finite_difference_for_stack() {
        let layer0 = mat(2.25);
        let layer1 = mat(3.24);

        let stack = Stack::builder(air(), air())
            .with_layer(layer0.clone(), Thickness::from_nm(100.0).unwrap())
            .with_layer(layer1.clone(), Thickness::from_nm(50.0).unwrap())
            .build()
            .unwrap();

        let omega2 = 1000.0_f64.powi(2);
        let h = 1e-2 * omega2;
        let beta2 = 100.0;

        let variable = DerivativeVariable::FrequencySquared;

        let input = Transfer2Input::new(
            arr0(c(omega2.sqrt())),
            arr0(c(beta2)),
            Polarisation::TransverseElectric,
        );

        let analytical = Transfer2::new()
            .solve_first_derivative(&stack, input, variable)
            .derivatives()
            .unwrap()
            .first()
            .clone();

        let plus_input = Transfer2Input::new(
            arr0(c((omega2 + h).sqrt())),
            arr0(c(beta2)),
            Polarisation::TransverseElectric,
        );

        let minus_input = Transfer2Input::new(
            arr0(c((omega2 - h).sqrt())),
            arr0(c(beta2)),
            Polarisation::TransverseElectric,
        );

        let plus = Transfer2::new().solve(&stack, plus_input);
        let minus = Transfer2::new().solve(&stack, minus_input);

        let expected =
            (&plus.matrix().add(&minus.matrix().scale(c(-1.0)))).scale(c(1.0 / (2.0 * h)));

        assert_matrix_close(&analytical, &expected, 1e-8);
    }

    #[test]
    fn propagation_constant_squared_derivative_matches_finite_difference_for_stack() {
        let layer0 = mat(2.25);
        let layer1 = mat(3.24);

        let stack = Stack::builder(air(), air())
            .with_layer(layer0.clone(), Thickness::from_nm(100.0).unwrap())
            .with_layer(layer1.clone(), Thickness::from_nm(50.0).unwrap())
            .build()
            .unwrap();

        let omega = 1000.0;
        let beta2 = 100.0;
        let h = 1e-3;

        let variable = DerivativeVariable::PropagationConstantSquared;

        let input = Transfer2Input::new(
            arr0(c(omega)),
            arr0(c(beta2)),
            Polarisation::TransverseElectric,
        );

        let analytical = Transfer2::new()
            .solve_first_derivative(&stack, input, variable)
            .derivatives()
            .unwrap()
            .first()
            .clone();

        let plus_input = Transfer2Input::new(
            arr0(c(omega)),
            arr0(c(beta2 + h)),
            Polarisation::TransverseElectric,
        );

        let minus_input = Transfer2Input::new(
            arr0(c(omega)),
            arr0(c(beta2 - h)),
            Polarisation::TransverseElectric,
        );

        let plus = Transfer2::new().solve(&stack, plus_input);
        let minus = Transfer2::new().solve(&stack, minus_input);

        let expected =
            (&plus.matrix().add(&minus.matrix().scale(c(-1.0)))).scale(c(1.0 / (2.0 * h)));

        assert_matrix_close(&analytical, &expected, 1e-8);
    }

    #[test]
    fn frequency_squared_second_derivative_matches_finite_difference_for_stack() {
        let layer0 = mat(2.25);
        let layer1 = mat(3.24);

        let stack = Stack::builder(air(), air())
            .with_layer(layer0.clone(), Thickness::from_nm(100.0).unwrap())
            .with_layer(layer1.clone(), Thickness::from_nm(50.0).unwrap())
            .build()
            .unwrap();

        let omega2 = 1000.0_f64.powi(2);
        let h = 1e-2 * omega2;
        let beta2 = 100.0;

        let variable = DerivativeVariable::FrequencySquared;

        let input = Transfer2Input::new(
            arr0(c(omega2.sqrt())),
            arr0(c(beta2)),
            Polarisation::TransverseElectric,
        );

        let analytical = Transfer2::new()
            .solve_second_derivative(&stack, input, variable)
            .derivatives()
            .unwrap()
            .second()
            .unwrap()
            .clone();

        let plus_input = Transfer2Input::new(
            arr0(c((omega2 + h).sqrt())),
            arr0(c(beta2)),
            Polarisation::TransverseElectric,
        );

        let zero_input = Transfer2Input::new(
            arr0(c(omega2.sqrt())),
            arr0(c(beta2)),
            Polarisation::TransverseElectric,
        );

        let minus_input = Transfer2Input::new(
            arr0(c((omega2 - h).sqrt())),
            arr0(c(beta2)),
            Polarisation::TransverseElectric,
        );

        let plus = Transfer2::new().solve(&stack, plus_input);
        let zero = Transfer2::new().solve(&stack, zero_input);
        let minus = Transfer2::new().solve(&stack, minus_input);

        let expected = (&plus
            .matrix()
            .add(&zero.matrix().scale(c(-2.0)))
            .add(minus.matrix()))
            .scale(c(1.0 / (h * h)));

        assert_matrix_close(&analytical, &expected, 1e-4);
    }

    #[test]
    fn propagation_constant_squared_second_derivative_matches_finite_difference_for_stack() {
        let layer0 = mat(2.25);
        let layer1 = mat(3.24);

        let stack = Stack::builder(air(), air())
            .with_layer(layer0.clone(), Thickness::from_nm(100.0).unwrap())
            .with_layer(layer1.clone(), Thickness::from_nm(50.0).unwrap())
            .build()
            .unwrap();

        let omega = 1000.0;
        let beta2 = 100.0;
        let h = 1e-2;

        let variable = DerivativeVariable::PropagationConstantSquared;

        let input = Transfer2Input::new(
            arr0(c(omega)),
            arr0(c(beta2)),
            Polarisation::TransverseElectric,
        );

        let analytical = Transfer2::new()
            .solve_second_derivative(&stack, input, variable)
            .derivatives()
            .unwrap()
            .second()
            .unwrap()
            .clone();

        let plus_input = Transfer2Input::new(
            arr0(c(omega)),
            arr0(c(beta2 + h)),
            Polarisation::TransverseElectric,
        );

        let zero_input = Transfer2Input::new(
            arr0(c(omega)),
            arr0(c(beta2)),
            Polarisation::TransverseElectric,
        );

        let minus_input = Transfer2Input::new(
            arr0(c(omega)),
            arr0(c(beta2 - h)),
            Polarisation::TransverseElectric,
        );

        let plus = Transfer2::new().solve(&stack, plus_input);
        let zero = Transfer2::new().solve(&stack, zero_input);
        let minus = Transfer2::new().solve(&stack, minus_input);

        let expected = (&plus
            .matrix()
            .add(&zero.matrix().scale(c(-2.0)))
            .add(minus.matrix()))
            .scale(c(1.0 / (h * h)));

        assert_matrix_close(&analytical, &expected, 1e-5);
    }

    #[test]
    fn frequency_first_derivative_matches_finite_difference_for_stack() {
        let stack = Stack::builder(air(), air())
            .with_layer(mat(2.25), Thickness::from_nm(100.0).unwrap())
            .with_layer(mat(3.24), Thickness::from_nm(50.0).unwrap())
            .build()
            .unwrap();

        let omega = 1000.0;
        let beta2 = 100.0;
        let h = 1e-3;

        let variable = DerivativeVariable::Frequency;

        let input = Transfer2Input::new(
            arr0(c(omega)),
            arr0(c(beta2)),
            Polarisation::TransverseElectric,
        );

        let analytical = Transfer2::new()
            .solve_first_derivative(&stack, input, variable)
            .derivatives()
            .unwrap()
            .first()
            .clone();

        let plus = Transfer2::new().solve(
            &stack,
            Transfer2Input::new(
                arr0(c(omega + h)),
                arr0(c(beta2)),
                Polarisation::TransverseElectric,
            ),
        );

        let minus = Transfer2::new().solve(
            &stack,
            Transfer2Input::new(
                arr0(c(omega - h)),
                arr0(c(beta2)),
                Polarisation::TransverseElectric,
            ),
        );

        let expected =
            (&plus.matrix().add(&minus.matrix().scale(c(-1.0)))).scale(c(1.0 / (2.0 * h)));

        assert_matrix_close(&analytical, &expected, 1e-7);
    }

    #[test]
    fn propagation_constant_first_derivative_matches_finite_difference_for_stack() {
        let stack = Stack::builder(air(), air())
            .with_layer(mat(2.25), Thickness::from_nm(100.0).unwrap())
            .with_layer(mat(3.24), Thickness::from_nm(50.0).unwrap())
            .build()
            .unwrap();

        let omega = 1000.0;
        let beta = 10.0;
        let h = 1e-4;

        let variable = DerivativeVariable::PropagationConstant;

        let input = Transfer2Input::new(
            arr0(c(omega)),
            arr0(c(beta * beta)),
            Polarisation::TransverseElectric,
        );

        let analytical = Transfer2::new()
            .solve_first_derivative(&stack, input, variable)
            .derivatives()
            .unwrap()
            .first()
            .clone();

        let plus = Transfer2::new().solve(
            &stack,
            Transfer2Input::new(
                arr0(c(omega)),
                arr0(c((beta + h) * (beta + h))),
                Polarisation::TransverseElectric,
            ),
        );

        let minus = Transfer2::new().solve(
            &stack,
            Transfer2Input::new(
                arr0(c(omega)),
                arr0(c((beta - h) * (beta - h))),
                Polarisation::TransverseElectric,
            ),
        );

        let expected =
            (&plus.matrix().add(&minus.matrix().scale(c(-1.0)))).scale(c(1.0 / (2.0 * h)));

        assert_matrix_close(&analytical, &expected, 1e-7);
    }

    #[test]
    fn frequency_second_derivative_matches_finite_difference_for_stack() {
        let stack = Stack::builder(air(), air())
            .with_layer(mat(2.25), Thickness::from_nm(100.0).unwrap())
            .with_layer(mat(3.24), Thickness::from_nm(50.0).unwrap())
            .build()
            .unwrap();

        let omega = 1000.0;
        let beta2 = 100.0;
        let h = 1e-2;

        let variable = DerivativeVariable::Frequency;

        let input = Transfer2Input::new(
            arr0(c(omega)),
            arr0(c(beta2)),
            Polarisation::TransverseElectric,
        );

        let analytical = Transfer2::new()
            .solve_second_derivative(&stack, input, variable)
            .derivatives()
            .unwrap()
            .second()
            .unwrap()
            .clone();

        let plus = Transfer2::new().solve(
            &stack,
            Transfer2Input::new(
                arr0(c(omega + h)),
                arr0(c(beta2)),
                Polarisation::TransverseElectric,
            ),
        );

        let zero = Transfer2::new().solve(
            &stack,
            Transfer2Input::new(
                arr0(c(omega)),
                arr0(c(beta2)),
                Polarisation::TransverseElectric,
            ),
        );

        let minus = Transfer2::new().solve(
            &stack,
            Transfer2Input::new(
                arr0(c(omega - h)),
                arr0(c(beta2)),
                Polarisation::TransverseElectric,
            ),
        );

        let expected = (&plus
            .matrix()
            .add(&zero.matrix().scale(c(-2.0)))
            .add(minus.matrix()))
            .scale(c(1.0 / (h * h)));

        assert_matrix_close(&analytical, &expected, 1e-5);
    }

    #[test]
    fn propagation_constant_second_derivative_matches_finite_difference_for_stack() {
        let stack = Stack::builder(air(), air())
            .with_layer(mat(2.25), Thickness::from_nm(100.0).unwrap())
            .with_layer(mat(3.24), Thickness::from_nm(50.0).unwrap())
            .build()
            .unwrap();

        let omega = 1000.0;
        let beta = 10.0;
        let h = 1e-3;

        let variable = DerivativeVariable::PropagationConstant;

        let input = Transfer2Input::new(
            arr0(c(omega)),
            arr0(c(beta * beta)),
            Polarisation::TransverseElectric,
        );

        let analytical = Transfer2::new()
            .solve_second_derivative(&stack, input, variable)
            .derivatives()
            .unwrap()
            .second()
            .unwrap()
            .clone();

        let plus = Transfer2::new().solve(
            &stack,
            Transfer2Input::new(
                arr0(c(omega)),
                arr0(c((beta + h) * (beta + h))),
                Polarisation::TransverseElectric,
            ),
        );

        let zero = Transfer2::new().solve(
            &stack,
            Transfer2Input::new(
                arr0(c(omega)),
                arr0(c(beta * beta)),
                Polarisation::TransverseElectric,
            ),
        );

        let minus = Transfer2::new().solve(
            &stack,
            Transfer2Input::new(
                arr0(c(omega)),
                arr0(c((beta - h) * (beta - h))),
                Polarisation::TransverseElectric,
            ),
        );

        let expected = (&plus
            .matrix()
            .add(&(&zero.matrix()).scale(c(-2.0)))
            .add(minus.matrix()))
            .scale(c(1.0 / (h * h)));

        assert_matrix_close(&analytical, &expected, 1e-5);
    }

    #[test]
    fn chain_rule_result_reports_requested_variable() {
        let stack = Stack::builder(air(), air())
            .with_layer(mat(2.25), Thickness::from_nm(100.0).unwrap())
            .build()
            .unwrap();

        let input = Transfer2Input::new(
            arr0(c(1000.0)),
            arr0(c(100.0)),
            Polarisation::TransverseElectric,
        );

        let result =
            Transfer2::new().solve_first_derivative(&stack, input, DerivativeVariable::Frequency);

        assert_eq!(
            result.derivatives().unwrap().variable(),
            DerivativeVariable::Frequency
        );
    }
}
