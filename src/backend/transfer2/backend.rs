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
    backend::{
        DerivativeVariable, PlanarInput,
        isotropic::{
            IsotropicLayerFirstDerivatives, IsotropicLayerQuantities,
            IsotropicLayerSecondDerivatives,
        },
        transfer2::{Matrix2, TransferError, jet::Transfer2Jet},
    },
    material::Material,
    stack::{Layer, Stack},
};

#[derive(Copy, Clone, Debug, Default)]
pub struct Transfer2;

impl Transfer2 {
    pub fn new() -> Self {
        Self
    }

    pub fn solve<M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Result<Matrix2<C, D>, TransferError>
    where
        M: Material<Real = C::RealField>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        let mut matrix = Matrix2::identity_like(&input.vacuum_wavenumber);

        for layer in stack.layers_in_propagation_order() {
            let q = IsotropicLayerQuantities::new(layer.material(), &input);

            let layer_matrix = Matrix2::from_layer(&q, layer.thickness());

            matrix = layer_matrix.multiply(&matrix);
        }

        Ok(matrix)
    }

    pub fn solve_first_derivative<M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        request: DerivativeVariable,
    ) -> Result<Transfer2Jet<C, D>, TransferError>
    where
        M: Material<Real = C::RealField>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        if let DerivativeVariable::Thickness(requested) = request {
            if requested >= stack.len() {
                return Err(TransferError::ThicknessLayerOutOfBounds {
                    requested,
                    layer_count: stack.len(),
                });
            }
        }

        let mut jet = Transfer2Jet::value_only(Matrix2::identity_like(&input.vacuum_wavenumber));

        let requested_variable = request;
        let primitive_variable = request.primitive();

        for (index, layer) in stack.layers_in_propagation_order().enumerate() {
            let q = IsotropicLayerQuantities::new(layer.material(), &input);

            let matrix = Matrix2::from_layer(&q, layer.thickness());

            let layer_jet = if let Some(first) =
                first_derivative(index, layer, &q, &input, primitive_variable)
            {
                Transfer2Jet::with_first(matrix, first)
            } else {
                Transfer2Jet::value_only(matrix)
            };

            jet = layer_jet.multiply(&jet);
        }

        if let Some(rule) = requested_variable.chain_rule(&input) {
            jet = jet.chain_rule(&rule);
        }

        Ok(jet)
    }

    pub fn solve_second_derivative<M, C, D>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        request: DerivativeVariable,
    ) -> Result<Transfer2Jet<C, D>, TransferError>
    where
        M: Material<Real = C::RealField>,
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
    {
        if let DerivativeVariable::Thickness(requested) = request {
            if requested >= stack.len() {
                return Err(TransferError::ThicknessLayerOutOfBounds {
                    requested,
                    layer_count: stack.len(),
                });
            }
        }

        let mut jet = Transfer2Jet::value_only(Matrix2::identity_like(&input.vacuum_wavenumber));

        let requested_variable = request;
        let primitive_variable = request.primitive();

        for (index, layer) in stack.layers_in_propagation_order().enumerate() {
            let q = IsotropicLayerQuantities::new(layer.material(), &input);
            let matrix = Matrix2::from_layer(&q, layer.thickness());

            let layer_jet = if let Some(dmatrix) =
                second_derivative(index, layer, &q, &input, primitive_variable)
            {
                Transfer2Jet::with_second(matrix, dmatrix.first, dmatrix.second)
            } else {
                Transfer2Jet::value_only(matrix)
            };

            jet = layer_jet.multiply(&jet);
        }

        if let Some(rule) = requested_variable.chain_rule(&input) {
            jet = jet.chain_rule(&rule);
        }

        Ok(jet)
    }
}

fn first_derivative<M, C, D>(
    layer_index: usize,
    layer: &Layer<M, C::RealField>,
    q: &IsotropicLayerQuantities<C, D>,
    input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    request: DerivativeVariable,
) -> Option<Matrix2<C, D>>
where
    M: Material<Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    match request {
        DerivativeVariable::Thickness(index) if index == layer_index => {
            Some(Matrix2::thickness_derivative(q, layer.thickness()))
        }

        DerivativeVariable::Thickness(_) => None,

        DerivativeVariable::VacuumWavenumberSquared => {
            let derivatives =
                IsotropicLayerFirstDerivatives::with_respect_to_vacuum_wavenumber_squared(
                    layer.material(),
                    q,
                    &input.vacuum_wavenumber,
                    input.polarisation,
                );

            Some(Matrix2::spectral_derivative(
                q,
                layer.thickness(),
                &derivatives,
            ))
        }

        DerivativeVariable::ParallelWavenumberSquared => {
            let derivatives =
                IsotropicLayerFirstDerivatives::with_respect_to_parallel_wavenumber_squared(q);

            Some(Matrix2::spectral_derivative(
                q,
                layer.thickness(),
                &derivatives,
            ))
        }

        DerivativeVariable::VacuumWavenumber | DerivativeVariable::ParallelWavenumber => {
            unreachable!("linear variables must be converted to primitive variables")
        }
    }
}

struct LayerDerivativeMatrices<C, D>
where
    D: Dimension,
{
    first: Matrix2<C, D>,
    second: Matrix2<C, D>,
}

fn second_derivative<M, C, D>(
    index: usize,
    layer: &Layer<M, C::RealField>,
    q: &IsotropicLayerQuantities<C, D>,
    input: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    request: DerivativeVariable,
) -> Option<LayerDerivativeMatrices<C, D>>
where
    M: Material<Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    match request {
        DerivativeVariable::Thickness(layer_index) if layer_index == index => {
            let first = Matrix2::thickness_derivative(q, layer.thickness());
            let second = Matrix2::thickness_second_derivative(q, layer.thickness());

            Some(LayerDerivativeMatrices { first, second })
        }
        DerivativeVariable::Thickness(_) => None,
        DerivativeVariable::VacuumWavenumberSquared => {
            let derivatives =
                IsotropicLayerSecondDerivatives::with_respect_to_vacuum_wavenumber_squared(
                    layer.material(),
                    q,
                    &input.vacuum_wavenumber,
                    input.polarisation,
                );

            let first = Matrix2::spectral_derivative(q, layer.thickness(), &derivatives.first);

            let second = Matrix2::spectral_second_derivative(q, layer.thickness(), &derivatives);

            Some(LayerDerivativeMatrices { first, second })
        }
        DerivativeVariable::ParallelWavenumberSquared => {
            let derivatives =
                IsotropicLayerSecondDerivatives::with_respect_to_parallel_wavenumber_squared(q);

            let first = Matrix2::spectral_derivative(q, layer.thickness(), &derivatives.first);
            let second = Matrix2::spectral_second_derivative(q, layer.thickness(), &derivatives);

            Some(LayerDerivativeMatrices { first, second })
        }
        DerivativeVariable::VacuumWavenumber | DerivativeVariable::ParallelWavenumber => {
            unreachable!("linear variables must be converted to primitive variables")
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{ArrayBase, Dimension, OwnedRepr, arr0, arr1};
    use num_complex::Complex64;

    use super::*;
    use crate::{
        backend::transfer2::Matrix2,
        backend::{PlanarInput, Polarisation},
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

    fn input0() -> PlanarInput<ndarray::Array0<C>> {
        PlanarInput::new(
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

        let expected =
            Matrix2::from_layer(&IsotropicLayerQuantities::new(&layer, &input), thickness);

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

        let m0 = Matrix2::from_layer(&IsotropicLayerQuantities::new(&layer0, &input), d0);

        let m1 = Matrix2::from_layer(&IsotropicLayerQuantities::new(&layer1, &input), d1);

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

        let input = PlanarInput::new(
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

        let result = Transfer2::new()
            .solve_first_derivative(&stack, input.clone(), variable)
            .unwrap();

        let q = IsotropicLayerQuantities::new(&layer0, &input);

        let dm0 = Matrix2::thickness_derivative(&q, d0);

        let q1 = IsotropicLayerQuantities::new(&layer1, &input);

        let m1 = Matrix2::from_layer(&q1, d1);

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

        let result = Transfer2::new()
            .solve_first_derivative(&stack, input.clone(), variable)
            .unwrap();

        let q0 = IsotropicLayerQuantities::new(&layer0, &input);

        let m0 = Matrix2::from_layer(&q0, d0);

        let q1 = IsotropicLayerQuantities::new(&layer1, &input);

        let dm1 = Matrix2::thickness_derivative(&q1, d1);

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
            .unwrap()
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

        let result = Transfer2::new()
            .solve_second_derivative(&stack, input.clone(), variable)
            .unwrap();

        let q0 = IsotropicLayerQuantities::new(&layer0, &input);

        let ddm0 = Matrix2::thickness_second_derivative(&q0, d0);

        let q1 = IsotropicLayerQuantities::new(&layer1, &input);

        let m1 = Matrix2::from_layer(&q1, d1);

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
        let h_nm = 1e-1;

        let variable = DerivativeVariable::Thickness(0);
        let input = input0();

        let stack = Stack::builder(air(), air())
            .with_layer(layer0.clone(), Thickness::from_nm(d0_nm).unwrap())
            .with_layer(layer1.clone(), Thickness::from_nm(d1_nm).unwrap())
            .build()
            .unwrap();

        let analytical = Transfer2::new()
            .solve_second_derivative(&stack, input.clone(), variable)
            .unwrap()
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
    fn vacuum_wavenumber_squared_derivative_matches_finite_difference_for_stack() {
        let layer0 = mat(2.25);
        let layer1 = mat(3.24);

        let stack = Stack::builder(air(), air())
            .with_layer(layer0.clone(), Thickness::from_nm(100.0).unwrap())
            .with_layer(layer1.clone(), Thickness::from_nm(50.0).unwrap())
            .build()
            .unwrap();

        let vacuum_wavenumber2 = 1000.0_f64.powi(2);
        let h = 1e-2 * vacuum_wavenumber2;
        let parallel_wavenumber = 100.0;

        let variable = DerivativeVariable::VacuumWavenumberSquared;

        let input = PlanarInput::new(
            arr0(c(vacuum_wavenumber2.sqrt())),
            arr0(c(parallel_wavenumber)),
            Polarisation::TransverseElectric,
        );

        let analytical = Transfer2::new()
            .solve_first_derivative(&stack, input, variable)
            .unwrap()
            .derivatives()
            .unwrap()
            .first()
            .clone();

        let plus_input = PlanarInput::new(
            arr0(c((vacuum_wavenumber2 + h).sqrt())),
            arr0(c(parallel_wavenumber)),
            Polarisation::TransverseElectric,
        );

        let minus_input = PlanarInput::new(
            arr0(c((vacuum_wavenumber2 - h).sqrt())),
            arr0(c(parallel_wavenumber)),
            Polarisation::TransverseElectric,
        );

        let plus = Transfer2::new().solve(&stack, plus_input);
        let minus = Transfer2::new().solve(&stack, minus_input);

        let expected =
            (&plus.matrix().add(&minus.matrix().scale(c(-1.0)))).scale(c(1.0 / (2.0 * h)));

        assert_matrix_close(&analytical, &expected, 1e-8);
    }

    #[test]
    fn parallel_wavenumber_squared_derivative_matches_finite_difference_for_stack() {
        let stack = Stack::builder(air(), air())
            .with_layer(mat(2.25), Thickness::from_nm(100.0).unwrap())
            .with_layer(mat(3.24), Thickness::from_nm(50.0).unwrap())
            .build()
            .unwrap();

        let vacuum_wavenumber = 1000.0;
        let parallel_wavenumber2: f64 = 100.0;
        let h = 1e-2;

        let variable = DerivativeVariable::ParallelWavenumberSquared;

        let analytical = Transfer2::new()
            .solve_first_derivative(
                &stack,
                PlanarInput::new(
                    arr0(c(vacuum_wavenumber)),
                    arr0(c(parallel_wavenumber2.sqrt())),
                    Polarisation::TransverseElectric,
                ),
                variable,
            )
            .unwrap()
            .derivatives()
            .unwrap()
            .first()
            .clone();

        let plus = Transfer2::new().solve(
            &stack,
            PlanarInput::new(
                arr0(c(vacuum_wavenumber)),
                arr0(c((parallel_wavenumber2 + h).sqrt())),
                Polarisation::TransverseElectric,
            ),
        );

        let minus = Transfer2::new().solve(
            &stack,
            PlanarInput::new(
                arr0(c(vacuum_wavenumber)),
                arr0(c((parallel_wavenumber2 - h).sqrt())),
                Polarisation::TransverseElectric,
            ),
        );

        let expected =
            (&plus.matrix().add(&minus.matrix().scale(c(-1.0)))).scale(c(1.0 / (2.0 * h)));

        assert_matrix_close(&analytical, &expected, 1e-7);
    }

    #[test]
    fn vacuum_wavenumber_squared_second_derivative_matches_finite_difference_for_stack() {
        let layer0 = mat(2.25);
        let layer1 = mat(3.24);

        let stack = Stack::builder(air(), air())
            .with_layer(layer0.clone(), Thickness::from_nm(100.0).unwrap())
            .with_layer(layer1.clone(), Thickness::from_nm(50.0).unwrap())
            .build()
            .unwrap();

        let vacuum_wavenumber2 = 1000.0_f64.powi(2);
        let h = 1e-2 * vacuum_wavenumber2;
        let parallel_wavenumber = 100.0;

        let variable = DerivativeVariable::VacuumWavenumberSquared;

        let input = PlanarInput::new(
            arr0(c(vacuum_wavenumber2.sqrt())),
            arr0(c(parallel_wavenumber)),
            Polarisation::TransverseElectric,
        );

        let analytical = Transfer2::new()
            .solve_second_derivative(&stack, input, variable)
            .unwrap()
            .derivatives()
            .unwrap()
            .second()
            .unwrap()
            .clone();

        let plus_input = PlanarInput::new(
            arr0(c((vacuum_wavenumber2 + h).sqrt())),
            arr0(c(parallel_wavenumber)),
            Polarisation::TransverseElectric,
        );

        let zero_input = PlanarInput::new(
            arr0(c(vacuum_wavenumber2.sqrt())),
            arr0(c(parallel_wavenumber)),
            Polarisation::TransverseElectric,
        );

        let minus_input = PlanarInput::new(
            arr0(c((vacuum_wavenumber2 - h).sqrt())),
            arr0(c(parallel_wavenumber)),
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
    fn parallel_wavenumber_squared_second_derivative_matches_finite_difference_for_stack() {
        let layer0 = mat(2.25);
        let layer1 = mat(3.24);

        let stack = Stack::builder(air(), air())
            .with_layer(layer0.clone(), Thickness::from_nm(100.0).unwrap())
            .with_layer(layer1.clone(), Thickness::from_nm(50.0).unwrap())
            .build()
            .unwrap();

        let vacuum_wavenumber = 1000.0;
        let parallel_wavenumber2 = 100.0_f64;
        let h = 1e-2;

        let variable = DerivativeVariable::ParallelWavenumberSquared;

        let input = PlanarInput::new(
            arr0(c(vacuum_wavenumber)),
            arr0(c(parallel_wavenumber2.sqrt())),
            Polarisation::TransverseElectric,
        );

        let analytical = Transfer2::new()
            .solve_second_derivative(&stack, input, variable)
            .unwrap()
            .derivatives()
            .unwrap()
            .second()
            .unwrap()
            .clone();

        let plus_input = PlanarInput::new(
            arr0(c(vacuum_wavenumber)),
            arr0(c((parallel_wavenumber2 + h).sqrt())),
            Polarisation::TransverseElectric,
        );

        let zero_input = PlanarInput::new(
            arr0(c(vacuum_wavenumber)),
            arr0(c(parallel_wavenumber2.sqrt())),
            Polarisation::TransverseElectric,
        );

        let minus_input = PlanarInput::new(
            arr0(c(vacuum_wavenumber)),
            arr0(c((parallel_wavenumber2 - h).sqrt())),
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
    fn vacuum_wavenumber_first_derivative_matches_finite_difference_for_stack() {
        let stack = Stack::builder(air(), air())
            .with_layer(mat(2.25), Thickness::from_nm(100.0).unwrap())
            .with_layer(mat(3.24), Thickness::from_nm(50.0).unwrap())
            .build()
            .unwrap();

        let vacuum_wavenumber = 1000.0;
        let parallel_wavenumber = 100.0;
        let h = 1e-3;

        let variable = DerivativeVariable::VacuumWavenumber;

        let input = PlanarInput::new(
            arr0(c(vacuum_wavenumber)),
            arr0(c(parallel_wavenumber)),
            Polarisation::TransverseElectric,
        );

        let analytical = Transfer2::new()
            .solve_first_derivative(&stack, input, variable)
            .unwrap()
            .derivatives()
            .unwrap()
            .first()
            .clone();

        let plus = Transfer2::new().solve(
            &stack,
            PlanarInput::new(
                arr0(c(vacuum_wavenumber + h)),
                arr0(c(parallel_wavenumber)),
                Polarisation::TransverseElectric,
            ),
        );

        let minus = Transfer2::new().solve(
            &stack,
            PlanarInput::new(
                arr0(c(vacuum_wavenumber - h)),
                arr0(c(parallel_wavenumber)),
                Polarisation::TransverseElectric,
            ),
        );

        let expected =
            (&plus.matrix().add(&minus.matrix().scale(c(-1.0)))).scale(c(1.0 / (2.0 * h)));

        assert_matrix_close(&analytical, &expected, 1e-7);
    }

    #[test]
    fn parallel_wavenumber_first_derivative_matches_finite_difference_for_stack() {
        let stack = Stack::builder(air(), air())
            .with_layer(mat(2.25), Thickness::from_nm(100.0).unwrap())
            .with_layer(mat(3.24), Thickness::from_nm(50.0).unwrap())
            .build()
            .unwrap();

        let vacuum_wavenumber = 1000.0;
        let parallel_wavenumber = 10.0;
        let h = 1e-4;

        let variable = DerivativeVariable::ParallelWavenumber;

        let input = PlanarInput::new(
            arr0(c(vacuum_wavenumber)),
            arr0(c(parallel_wavenumber)),
            Polarisation::TransverseElectric,
        );

        let analytical = Transfer2::new()
            .solve_first_derivative(&stack, input, variable)
            .unwrap()
            .derivatives()
            .unwrap()
            .first()
            .clone();

        let plus = Transfer2::new().solve(
            &stack,
            PlanarInput::new(
                arr0(c(vacuum_wavenumber)),
                arr0(c(parallel_wavenumber + h)),
                Polarisation::TransverseElectric,
            ),
        );

        let minus = Transfer2::new().solve(
            &stack,
            PlanarInput::new(
                arr0(c(vacuum_wavenumber)),
                arr0(c((parallel_wavenumber - h))),
                Polarisation::TransverseElectric,
            ),
        );

        let expected =
            (&plus.matrix().add(&minus.matrix().scale(c(-1.0)))).scale(c(1.0 / (2.0 * h)));

        assert_matrix_close(&analytical, &expected, 1e-7);
    }

    #[test]
    fn vacuum_wavenumber_second_derivative_matches_finite_difference_for_stack() {
        let stack = Stack::builder(air(), air())
            .with_layer(mat(2.25), Thickness::from_nm(100.0).unwrap())
            .with_layer(mat(3.24), Thickness::from_nm(50.0).unwrap())
            .build()
            .unwrap();

        let vacuum_wavenumber = 1000.0;
        let parallel_wavenumber = 100.0;
        let h = 1e-2;

        let variable = DerivativeVariable::VacuumWavenumber;

        let input = PlanarInput::new(
            arr0(c(vacuum_wavenumber)),
            arr0(c(parallel_wavenumber)),
            Polarisation::TransverseElectric,
        );

        let analytical = Transfer2::new()
            .solve_second_derivative(&stack, input, variable)
            .unwrap()
            .derivatives()
            .unwrap()
            .second()
            .unwrap()
            .clone();

        let plus = Transfer2::new().solve(
            &stack,
            PlanarInput::new(
                arr0(c(vacuum_wavenumber + h)),
                arr0(c(parallel_wavenumber)),
                Polarisation::TransverseElectric,
            ),
        );

        let zero = Transfer2::new().solve(
            &stack,
            PlanarInput::new(
                arr0(c(vacuum_wavenumber)),
                arr0(c(parallel_wavenumber)),
                Polarisation::TransverseElectric,
            ),
        );

        let minus = Transfer2::new().solve(
            &stack,
            PlanarInput::new(
                arr0(c(vacuum_wavenumber - h)),
                arr0(c(parallel_wavenumber)),
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
    fn parallel_wavenumber_second_derivative_matches_finite_difference_for_stack() {
        let stack = Stack::builder(air(), air())
            .with_layer(mat(2.25), Thickness::from_nm(100.0).unwrap())
            .with_layer(mat(3.24), Thickness::from_nm(50.0).unwrap())
            .build()
            .unwrap();

        let vacuum_wavenumber = 1000.0;
        let parallel_wavenumber = 10.0;
        let h = 1e-3;

        let variable = DerivativeVariable::ParallelWavenumber;

        let input = PlanarInput::new(
            arr0(c(vacuum_wavenumber)),
            arr0(c(parallel_wavenumber)),
            Polarisation::TransverseElectric,
        );

        let analytical = Transfer2::new()
            .solve_second_derivative(&stack, input, variable)
            .unwrap()
            .derivatives()
            .unwrap()
            .second()
            .unwrap()
            .clone();

        let plus = Transfer2::new().solve(
            &stack,
            PlanarInput::new(
                arr0(c(vacuum_wavenumber)),
                arr0(c(parallel_wavenumber + h)),
                Polarisation::TransverseElectric,
            ),
        );

        let zero = Transfer2::new().solve(
            &stack,
            PlanarInput::new(
                arr0(c(vacuum_wavenumber)),
                arr0(c(parallel_wavenumber)),
                Polarisation::TransverseElectric,
            ),
        );

        let minus = Transfer2::new().solve(
            &stack,
            PlanarInput::new(
                arr0(c(vacuum_wavenumber)),
                arr0(c(parallel_wavenumber - h)),
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

        let input = PlanarInput::new(
            arr0(c(1000.0)),
            arr0(c(100.0)),
            Polarisation::TransverseElectric,
        );

        let result = Transfer2::new()
            .solve_first_derivative(&stack, input, DerivativeVariable::VacuumWavenumber)
            .unwrap();

        assert_eq!(
            result.derivatives().unwrap().variable(),
            DerivativeVariable::VacuumWavenumber
        );
    }
}
