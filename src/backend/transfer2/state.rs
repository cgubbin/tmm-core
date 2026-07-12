//! Boundary and field-state helpers for the 2×2 transfer-matrix backend.
//!
//! This module contains the small state vectors used to apply outgoing-wave
//! boundary conditions and construct mode-finding residuals.
//!
//! The propagated field state is:
//!
//! ```text
//! s = [φ, φ′]ᵀ
//! ```
//!
//! An outgoing transmission-side state is initialized as:
//!
//! ```text
//! s₀ = [1, -γ / m]ᵀ
//! ```
//!
//! where `γ` is the out-of-plane wavevector and `m` is the polarization-dependent
//! factor.

use ndarray::{ArrayBase, Dimension, OwnedRepr, ScalarOperand};

use crate::ComplexScalar;

use super::Matrix2;

/// Boundary data used to construct an outgoing field state.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryMode<C, D>
where
    D: Dimension,
{
    pub gamma: ArrayBase<OwnedRepr<C>, D>,
    pub factor: ArrayBase<OwnedRepr<C>, D>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryModeDerivatives<C, D>
where
    D: Dimension,
{
    pub gamma_first: ArrayBase<OwnedRepr<C>, D>,
    pub factor_first: ArrayBase<OwnedRepr<C>, D>,
    pub gamma_second: Option<ArrayBase<OwnedRepr<C>, D>>,
    pub factor_second: Option<ArrayBase<OwnedRepr<C>, D>>,
}

pub struct BoundaryModeSecondDerivatives<'a, C, D>
where
    D: Dimension,
{
    pub gamma: &'a ArrayBase<OwnedRepr<C>, D>,
    pub factor: &'a ArrayBase<OwnedRepr<C>, D>,
}

impl<C, D> BoundaryModeDerivatives<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub fn zeros_like(mode: &BoundaryMode<C, D>) -> Self {
        let zero = mode.gamma.mapv(|_| C::zero());

        Self {
            gamma_first: zero.clone(),
            factor_first: zero.clone(),
            gamma_second: Some(zero.clone()),
            factor_second: Some(zero),
        }
    }

    pub fn second_gamma(&self) -> Option<&ArrayBase<OwnedRepr<C>, D>> {
        self.gamma_second.as_ref()
    }

    pub fn second_factor(&self) -> Option<&ArrayBase<OwnedRepr<C>, D>> {
        self.factor_second.as_ref()
    }

    pub fn second(&self) -> Option<BoundaryModeSecondDerivatives<'_, C, D>> {
        Some(BoundaryModeSecondDerivatives {
            gamma: self.gamma_second.as_ref()?,
            factor: self.factor_second.as_ref()?,
        })
    }
}

/// Field state `[φ, φ′]`.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldState<C, D>
where
    D: Dimension,
{
    pub value: ArrayBase<OwnedRepr<C>, D>,
    pub derivative: ArrayBase<OwnedRepr<C>, D>,
}

impl<C, D> FieldState<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub fn add(&self, rhs: &Self) -> Self {
        Self {
            value: self.value.clone() + rhs.value.view(),
            derivative: self.derivative.clone() + rhs.derivative.view(),
        }
    }

    pub fn scale(&self, value: C) -> Self {
        Self {
            value: self.value.clone() * value,
            derivative: self.derivative.clone() * value,
        }
    }
}

impl<C, D> BoundaryMode<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub fn outgoing_state(&self) -> FieldState<C, D> {
        FieldState {
            value: self.gamma.mapv(|_| C::one()),
            derivative: -self.gamma.clone() / self.factor.view(),
        }
    }

    pub fn outgoing_state_derivative(
        &self,
        derivatives: &BoundaryModeDerivatives<C, D>,
    ) -> FieldState<C, D> {
        let factor_squared = self.factor.mapv(|x| x * x);

        FieldState {
            value: self.gamma.mapv(|_| C::zero()),
            derivative: -derivatives.gamma_first.clone() / self.factor.view()
                + self.gamma.clone() * derivatives.factor_first.view() / factor_squared.view(),
        }
    }

    pub fn outgoing_state_second_derivative(
        &self,
        derivatives: &BoundaryModeDerivatives<C, D>,
    ) -> Option<FieldState<C, D>>
    where
        C: ScalarOperand,
    {
        let second_derivatives = derivatives.second()?;

        let dg = &derivatives.gamma_first;
        let df = &derivatives.factor_first;
        let ddg = second_derivatives.gamma;
        let ddf = second_derivatives.factor;

        let two = C::one() + C::one();

        let factor_squared = self.factor.mapv(|x| x * x);
        let factor_cubed = self.factor.mapv(|x| x * x * x);

        let derivative = -ddg.clone() / self.factor.view()
            + dg.clone() * df.view() * two / factor_squared.view()
            + self.gamma.clone() * ddf.view() / factor_squared.view()
            - self.gamma.clone() * df.mapv(|x| x * x) * two / factor_cubed.view();

        Some(FieldState {
            value: self.gamma.mapv(|_| C::zero()),
            derivative,
        })
    }
}

impl<C, D> Matrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub fn apply_state(&self, state: &FieldState<C, D>) -> FieldState<C, D> {
        FieldState {
            value: self.m11().clone() * state.value.view()
                + self.m12().clone() * state.derivative.view(),
            derivative: self.m21().clone() * state.value.view()
                + self.m22().clone() * state.derivative.view(),
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{ArrayBase, Dimension, OwnedRepr, arr0};
    use num_complex::Complex64;

    use super::*;
    use crate::{
        backend::{
            DerivativeVariable, MatrixEvaluation, PlanarInput, Polarisation, transfer2::Transfer2,
        },
        material::{Constant, IsotropicMaterial},
        stack::{Stack, Thickness},
    };

    type C = Complex64;

    fn c(x: f64) -> C {
        C::new(x, 0.0)
    }

    fn mat(epsilon: f64) -> IsotropicMaterial<f64> {
        Constant::new(epsilon).into()
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

    fn scalar_mode(gamma: f64, factor: f64) -> BoundaryMode<C, ndarray::Ix0> {
        BoundaryMode {
            gamma: arr0(c(gamma)),
            factor: arr0(c(factor)),
        }
    }

    fn scalar_mode_derivative(dgamma: f64) -> BoundaryModeDerivatives<C, ndarray::Ix0> {
        BoundaryModeDerivatives {
            gamma_first: arr0(c(dgamma)),
            factor_first: arr0(c(0.0)),
            gamma_second: None,
            factor_second: None,
        }
    }

    fn scalar_mode_second_derivative(
        dgamma: f64,
        ddgamma: f64,
    ) -> BoundaryModeDerivatives<C, ndarray::Ix0> {
        BoundaryModeDerivatives {
            gamma_first: arr0(c(dgamma)),
            factor_first: arr0(c(0.0)),
            gamma_second: Some(arr0(c(ddgamma))),
            factor_second: Some(arr0(c(0.0))),
        }
    }

    fn input0() -> PlanarInput<ndarray::Array0<C>> {
        PlanarInput::new(
            arr0(c(1000.0)),
            arr0(c(100.0)),
            Polarisation::TransverseElectric,
        )
    }

    #[test]
    fn outgoing_state_is_unit_value_and_minus_gamma_over_factor() {
        let boundary = scalar_mode(3.0, 2.0);
        let state = boundary.outgoing_state();

        assert_relative_eq!(state.value[()], c(1.0), max_relative = 1e-12);
        assert_relative_eq!(state.derivative[()], c(-1.5), max_relative = 1e-12);
    }

    #[test]
    fn outgoing_state_derivative_matches_formula() {
        let boundary = scalar_mode(3.0, 2.0);

        let derivatives = BoundaryModeDerivatives {
            gamma_first: arr0(c(0.4)),
            factor_first: arr0(c(0.2)),
            gamma_second: None,
            factor_second: None,
        };

        let state = boundary.outgoing_state_derivative(&derivatives);

        let expected = -c(0.4) / c(2.0) + c(0.2) * c(3.0) / c(4.0);

        assert_relative_eq!(state.value[()], c(0.0), max_relative = 1e-12);
        assert_relative_eq!(state.derivative[()], expected, max_relative = 1e-12);
    }

    #[test]
    fn outgoing_state_second_derivative_matches_formula() {
        let boundary = scalar_mode(3.0, 2.0);

        let derivatives = BoundaryModeDerivatives {
            gamma_first: arr0(c(0.4)),
            factor_first: arr0(c(0.2)),
            gamma_second: Some(arr0(c(0.7))),
            factor_second: Some(arr0(c(0.3))),
        };

        let state = boundary
            .outgoing_state_second_derivative(&derivatives)
            .unwrap();

        let expected =
            -c(0.7) / c(2.0) + c(2.0) * c(0.4) * c(0.2) / c(4.0) + c(3.0) * c(0.3) / c(4.0)
                - c(2.0) * c(3.0) * c(0.2 * 0.2) / c(8.0);

        assert_relative_eq!(state.value[()], c(0.0), max_relative = 1e-12);
        assert_relative_eq!(state.derivative[()], expected, max_relative = 1e-12);
    }

    #[test]
    fn outgoing_residual_matches_manual_formula() {
        let matrix = Matrix2::new(arr0(c(1.0)), arr0(c(2.0)), arr0(c(3.0)), arr0(c(4.0)));
        let result = MatrixEvaluation::new(matrix);

        let incident = scalar_mode(5.0, 2.0);
        let transmission = scalar_mode(3.0, 1.5);

        let residual = result.outgoing_residual(&incident, &transmission);

        let s0_value = c(1.0);
        let s0_derivative = -c(3.0) / c(1.5);

        let value = c(1.0) * s0_value + c(2.0) * s0_derivative;
        let derivative = c(3.0) * s0_value + c(4.0) * s0_derivative;

        let expected = derivative - value * c(5.0) / c(2.0);

        assert_relative_eq!(residual[()], expected, max_relative = 1e-12);
    }

    #[test]
    fn outgoing_residual_first_derivative_matches_finite_difference_for_layer_zero() {
        let layer0 = mat(2.25);
        let layer1 = mat(3.24);

        let d0_nm = 100.0;
        let d1_nm = 50.0;
        let h_nm = 1e-3;

        let input = input0();
        let variable = DerivativeVariable::Thickness(0);

        let stack = Stack::builder(mat(1.0), mat(1.0))
            .with_layer(layer0.clone(), Thickness::from_nm(d0_nm).unwrap())
            .with_layer(layer1.clone(), Thickness::from_nm(d1_nm).unwrap())
            .build()
            .unwrap();

        let incident = scalar_mode(1.7, 1.0);
        let transmission = scalar_mode(1.3, 1.0);

        let zero_first = scalar_mode_derivative(0.0);

        let analytical = Transfer2::new()
            .solve_first_derivative(&stack, input.clone(), variable)
            .unwrap()
            .outgoing_residual_derivative(&incident, &zero_first, &transmission, &zero_first)
            .unwrap();

        let plus_stack = Stack::builder(mat(1.0), mat(1.0))
            .with_layer(layer0.clone(), Thickness::from_nm(d0_nm + h_nm).unwrap())
            .with_layer(layer1.clone(), Thickness::from_nm(d1_nm).unwrap())
            .build()
            .unwrap();

        let minus_stack = Stack::builder(mat(1.0), mat(1.0))
            .with_layer(layer0, Thickness::from_nm(d0_nm - h_nm).unwrap())
            .with_layer(layer1, Thickness::from_nm(d1_nm).unwrap())
            .build()
            .unwrap();

        let plus = Transfer2::new()
            .solve(&plus_stack, input.clone())
            .outgoing_residual(&incident, &transmission);

        let minus = Transfer2::new()
            .solve(&minus_stack, input)
            .outgoing_residual(&incident, &transmission);

        let h_cm = Thickness::from_nm(h_nm).unwrap().as_cm();
        let expected = (plus - minus) / c(2.0 * h_cm);

        assert_array_close(&analytical, &expected, 1e-6);
    }

    #[test]
    fn outgoing_residual_first_derivative_matches_finite_difference_for_vacuum_wavenumber_squared()
    {
        let stack = Stack::builder(mat(1.0), mat(1.0))
            .with_layer(mat(2.25), Thickness::from_nm(100.0).unwrap())
            .with_layer(mat(3.24), Thickness::from_nm(50.0).unwrap())
            .build()
            .unwrap();

        let vacuum_wavenumber2 = 1000.0_f64.powi(2);
        let parallel_wavenumber2 = 100.0;
        let h = 1e-2 * vacuum_wavenumber2;

        let variable = DerivativeVariable::VacuumWavenumberSquared;

        let input = PlanarInput::new(
            arr0(c(vacuum_wavenumber2.sqrt())),
            arr0(c(parallel_wavenumber2)),
            Polarisation::TransverseElectric,
        );

        let incident = scalar_mode(1.7, 1.0);
        let transmission = scalar_mode(1.3, 1.0);
        let zero_first = scalar_mode_derivative(0.0);

        let analytical = Transfer2::new()
            .solve_first_derivative(&stack, input, variable)
            .unwrap()
            .outgoing_residual_derivative(&incident, &zero_first, &transmission, &zero_first)
            .unwrap();

        let plus = Transfer2::new()
            .solve(
                &stack,
                PlanarInput::new(
                    arr0(c((vacuum_wavenumber2 + h).sqrt())),
                    arr0(c(parallel_wavenumber2)),
                    Polarisation::TransverseElectric,
                ),
            )
            .outgoing_residual(&incident, &transmission);

        let minus = Transfer2::new()
            .solve(
                &stack,
                PlanarInput::new(
                    arr0(c((vacuum_wavenumber2 - h).sqrt())),
                    arr0(c(parallel_wavenumber2)),
                    Polarisation::TransverseElectric,
                ),
            )
            .outgoing_residual(&incident, &transmission);

        let expected = (plus - minus) / c(2.0 * h);

        assert_array_close(&analytical, &expected, 1e-6);
    }

    #[test]
    fn outgoing_residual_first_derivative_matches_finite_difference_for_parallel_wavenumber_squared()
     {
        let stack = Stack::builder(mat(1.0), mat(1.0))
            .with_layer(mat(2.25), Thickness::from_nm(100.0).unwrap())
            .with_layer(mat(3.24), Thickness::from_nm(50.0).unwrap())
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

        let incident = scalar_mode(1.7, 1.0);
        let transmission = scalar_mode(1.3, 1.0);
        let zero_first = scalar_mode_derivative(0.0);

        let analytical = Transfer2::new()
            .solve_first_derivative(&stack, input, variable)
            .unwrap()
            .outgoing_residual_derivative(&incident, &zero_first, &transmission, &zero_first)
            .unwrap();

        let plus = Transfer2::new()
            .solve(
                &stack,
                PlanarInput::new(
                    arr0(c(vacuum_wavenumber)),
                    arr0(c((parallel_wavenumber2 + h).sqrt())),
                    Polarisation::TransverseElectric,
                ),
            )
            .outgoing_residual(&incident, &transmission);

        let minus = Transfer2::new()
            .solve(
                &stack,
                PlanarInput::new(
                    arr0(c(vacuum_wavenumber)),
                    arr0(c((parallel_wavenumber2 - h).sqrt())),
                    Polarisation::TransverseElectric,
                ),
            )
            .outgoing_residual(&incident, &transmission);

        let expected = (plus - minus) / c(2.0 * h);

        assert_array_close(&analytical, &expected, 1e-6);
    }

    #[test]
    fn outgoing_residual_second_derivative_matches_finite_difference_for_layer_zero() {
        let layer0 = mat(2.25);
        let layer1 = mat(3.24);

        let d0_nm = 100.0;
        let d1_nm = 50.0;
        let h_nm = 1e-1;

        let input = input0();
        let variable = DerivativeVariable::Thickness(0);

        let stack = Stack::builder(mat(1.0), mat(1.0))
            .with_layer(layer0.clone(), Thickness::from_nm(d0_nm).unwrap())
            .with_layer(layer1.clone(), Thickness::from_nm(d1_nm).unwrap())
            .build()
            .unwrap();

        let incident = scalar_mode(1.7, 1.0);
        let transmission = scalar_mode(1.3, 1.0);

        let zero_derivatives = scalar_mode_second_derivative(0.0, 0.0);

        let analytical = Transfer2::new()
            .solve_second_derivative(&stack, input.clone(), variable)
            .unwrap()
            .outgoing_residual_second_derivative(
                &incident,
                &zero_derivatives,
                &transmission,
                &zero_derivatives,
            )
            .unwrap();

        let plus_stack = Stack::builder(mat(1.0), mat(1.0))
            .with_layer(layer0.clone(), Thickness::from_nm(d0_nm + h_nm).unwrap())
            .with_layer(layer1.clone(), Thickness::from_nm(d1_nm).unwrap())
            .build()
            .unwrap();

        let zero_stack = Stack::builder(mat(1.0), mat(1.0))
            .with_layer(layer0.clone(), Thickness::from_nm(d0_nm).unwrap())
            .with_layer(layer1.clone(), Thickness::from_nm(d1_nm).unwrap())
            .build()
            .unwrap();

        let minus_stack = Stack::builder(mat(1.0), mat(1.0))
            .with_layer(layer0, Thickness::from_nm(d0_nm - h_nm).unwrap())
            .with_layer(layer1, Thickness::from_nm(d1_nm).unwrap())
            .build()
            .unwrap();

        let plus = Transfer2::new()
            .solve(&plus_stack, input.clone())
            .outgoing_residual(&incident, &transmission);

        let zero = Transfer2::new()
            .solve(&zero_stack, input.clone())
            .outgoing_residual(&incident, &transmission);

        let minus = Transfer2::new()
            .solve(&minus_stack, input)
            .outgoing_residual(&incident, &transmission);

        let h_cm = Thickness::from_nm(h_nm).unwrap().as_cm();
        let expected = (plus - zero.mapv(|x| x * c(2.0)) + minus) / c(h_cm * h_cm);

        assert_array_close(&analytical, &expected, 1e-4);
    }

    #[test]
    fn outgoing_residual_second_derivative_matches_finite_difference_for_vacuum_wavenumber_squared()
    {
        let stack = Stack::builder(mat(1.0), mat(1.0))
            .with_layer(mat(2.25), Thickness::from_nm(100.0).unwrap())
            .with_layer(mat(3.24), Thickness::from_nm(50.0).unwrap())
            .build()
            .unwrap();

        let vacuum_wavenumber2 = 1000.0_f64.powi(2);
        let parallel_wavenumber = 100.0;
        let h = 1e-2 * vacuum_wavenumber2;

        let variable = DerivativeVariable::VacuumWavenumberSquared;

        let input = PlanarInput::new(
            arr0(c(vacuum_wavenumber2.sqrt())),
            arr0(c(parallel_wavenumber)),
            Polarisation::TransverseElectric,
        );

        let incident = scalar_mode(1.7, 1.0);
        let transmission = scalar_mode(1.3, 1.0);

        let zero_derivatives = scalar_mode_second_derivative(0.0, 0.0);

        let analytical = Transfer2::new()
            .solve_second_derivative(&stack, input, variable)
            .unwrap()
            .outgoing_residual_second_derivative(
                &incident,
                &zero_derivatives,
                &transmission,
                &zero_derivatives,
            )
            .unwrap();

        let plus = Transfer2::new()
            .solve(
                &stack,
                PlanarInput::new(
                    arr0(c((vacuum_wavenumber2 + h).sqrt())),
                    arr0(c(parallel_wavenumber)),
                    Polarisation::TransverseElectric,
                ),
            )
            .outgoing_residual(&incident, &transmission);

        let zero = Transfer2::new()
            .solve(
                &stack,
                PlanarInput::new(
                    arr0(c(vacuum_wavenumber2.sqrt())),
                    arr0(c(parallel_wavenumber)),
                    Polarisation::TransverseElectric,
                ),
            )
            .outgoing_residual(&incident, &transmission);

        let minus = Transfer2::new()
            .solve(
                &stack,
                PlanarInput::new(
                    arr0(c((vacuum_wavenumber2 - h).sqrt())),
                    arr0(c(parallel_wavenumber)),
                    Polarisation::TransverseElectric,
                ),
            )
            .outgoing_residual(&incident, &transmission);

        let expected = (plus - zero.mapv(|x| x * c(2.0)) + minus) / c(h * h);

        assert_array_close(&analytical, &expected, 1e-4);
    }

    #[test]
    fn outgoing_residual_second_derivative_matches_finite_difference_for_parallel_wavenumber_squared()
     {
        let stack = Stack::builder(mat(1.0), mat(1.0))
            .with_layer(mat(2.25), Thickness::from_nm(100.0).unwrap())
            .with_layer(mat(3.24), Thickness::from_nm(50.0).unwrap())
            .build()
            .unwrap();

        let vacuum_wavenumber = 1000.0;
        let parallel_wavenumber2 = 100.0_f64;
        let h = 1e-2;

        let variable = DerivativeVariable::ParallelWavenumberSquared;

        let input = PlanarInput::new(
            arr0(c(vacuum_wavenumber)),
            arr0(c(parallel_wavenumber2)),
            Polarisation::TransverseElectric,
        );

        let incident = scalar_mode(1.7, 1.0);
        let transmission = scalar_mode(1.3, 1.0);

        let zero_derivatives = scalar_mode_second_derivative(0.0, 0.0);

        let analytical = Transfer2::new()
            .solve_second_derivative(&stack, input, variable)
            .unwrap()
            .outgoing_residual_second_derivative(
                &incident,
                &zero_derivatives,
                &transmission,
                &zero_derivatives,
            )
            .unwrap();

        let plus = Transfer2::new()
            .solve(
                &stack,
                PlanarInput::new(
                    arr0(c(vacuum_wavenumber)),
                    arr0(c((parallel_wavenumber2 + h).sqrt())),
                    Polarisation::TransverseElectric,
                ),
            )
            .outgoing_residual(&incident, &transmission);

        let zero = Transfer2::new()
            .solve(
                &stack,
                PlanarInput::new(
                    arr0(c(vacuum_wavenumber)),
                    arr0(c((parallel_wavenumber2).sqrt())),
                    Polarisation::TransverseElectric,
                ),
            )
            .outgoing_residual(&incident, &transmission);

        let minus = Transfer2::new()
            .solve(
                &stack,
                PlanarInput::new(
                    arr0(c(vacuum_wavenumber)),
                    arr0(c((parallel_wavenumber2 - h).sqrt())),
                    Polarisation::TransverseElectric,
                ),
            )
            .outgoing_residual(&incident, &transmission);

        let expected = (plus - zero.mapv(|x| x * c(2.0)) + minus) / c(h * h);

        assert_array_close(&analytical, &expected, 1e-5);
    }
}
