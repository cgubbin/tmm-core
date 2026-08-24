//! Directional contraction of coordinate-free derivative parts.
//!
//! Multivariate derivative parts store derivatives with respect to independent
//! parameter axes. This module contracts those components along a supplied
//! direction in parameter space.
//!
//! For a bivariate direction `v = (v₀, v₁)`, first-order contraction computes
//!
//! ```text
//! Dᵥ f = v₀ ∂₀f + v₁ ∂₁f,
//! ```
//!
//! while second-order contraction computes
//!
//! ```text
//! Dᵥ² f = v₀² ∂₀₀f + 2 v₀v₁ ∂₀₁f + v₁² ∂₁₁f.
//! ```
//!
//! Directions are not normalised. Their components may therefore represent
//! arbitrary scaling of a straight direction in parameter space.
//!
//! Second-order contraction computes the Hessian quadratic form along that
//! direction. It does not include additional chain-rule terms associated with
//! curvature of a non-linear parameterised path.

use super::{
    BivariateFirstParts, BivariateSecondParts, DirectionalFirstParts, DirectionalSecondParts,
};

use ndarray::{ArrayBase, Dimension, OwnedRepr};
use num_traits::One;
use std::ops::{Add, Mul};

/// A value supporting component-wise linear combinations with coefficients of
/// type `R`.
///
/// This capability is intentionally weaker than the backend scalar algebra. It
/// is used after derivative storage has been decomposed, when values may be
/// ordinary arrays, matrices, observable containers, or other composite types.
///
/// Implementations should preserve the structure and shape of `Self`.
pub(crate) trait LinearCombination<R>: Sized {
    /// Multiply every component by `coefficient`.
    fn scaled(self, coefficient: R) -> Self;

    /// Add `coefficient * rhs` component-wise to `self`.
    fn add_scaled(self, rhs: Self, coefficient: R) -> Self;

    /// Construct `first_coefficient * first + second_coefficient * second`.
    fn linear_combination(
        first: Self,
        first_coefficient: R,
        second: Self,
        second_coefficient: R,
    ) -> Self {
        first
            .scaled(first_coefficient)
            .add_scaled(second, second_coefficient)
    }
}

/// Component-wise linear combinations of owned arrays.
///
/// Both arrays passed to [`LinearCombination::add_scaled`] must have matching
/// dimensions and shapes. A shape mismatch follows `ndarray`'s usual arithmetic
/// behaviour and will panic.
impl<T, D, R> LinearCombination<R> for ArrayBase<OwnedRepr<T>, D>
where
    T: Clone + Add<Output = T> + Mul<R, Output = T>,
    D: Dimension,
    R: Copy,
{
    fn scaled(self, coefficient: R) -> Self {
        self.mapv(|value| value * coefficient)
    }

    fn add_scaled(self, rhs: Self, coefficient: R) -> Self {
        self + rhs.scaled(coefficient)
    }
}

/// Contracts a multivariate derivative representation along a supplied
/// parameter-space direction.
///
/// Contraction changes only the derivative representation. The underlying value
/// is preserved unchanged.
pub(crate) trait ContractDirection<D> {
    /// Directional derivative representation produced by the contraction.
    type Output;

    /// Contract `self` along `direction`.
    fn contract_direction(self, direction: D) -> Self::Output;
}

impl<T, R> ContractDirection<BivariateDirection<R>> for BivariateFirstParts<T>
where
    T: LinearCombination<R>,
{
    type Output = DirectionalFirstParts<T>;

    fn contract_direction(self, direction: BivariateDirection<R>) -> Self::Output {
        let (value, axis0, axis1) = self.into_parts();
        let (weight0, weight1) = direction.into_parts();

        let first = T::linear_combination(axis0, weight0, axis1, weight1);

        DirectionalFirstParts::new(value, first)
    }
}

impl<T, R> ContractDirection<BivariateDirection<R>> for BivariateSecondParts<T>
where
    T: LinearCombination<R>,
    R: Copy + One + Add<Output = R> + Mul<Output = R>,
{
    type Output = DirectionalSecondParts<T>;

    fn contract_direction(self, direction: BivariateDirection<R>) -> Self::Output {
        let (value, axis0, axis1, axis0_axis0, axis0_axis1, axis1_axis1) = self.into_parts();

        let (weight0, weight1) = direction.into_parts();

        let first = T::linear_combination(axis0, weight0, axis1, weight1);

        let weight00 = weight0 * weight0;
        let weight01 = (R::one() + R::one()) * weight0 * weight1;
        let weight11 = weight1 * weight1;

        let second = axis0_axis0
            .scaled(weight00)
            .add_scaled(axis0_axis1, weight01)
            .add_scaled(axis1_axis1, weight11);

        DirectionalSecondParts::new(value, first, second)
    }
}

/// A direction in a two-dimensional parameter space.
///
/// The components are coefficients relative to derivative axes zero and one.
/// They are not automatically normalised.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BivariateDirection<R> {
    axis0: R,
    axis1: R,
}

impl<R> BivariateDirection<R> {
    /// Construct a direction with the supplied axis coefficients.
    pub(crate) const fn new(axis0: R, axis1: R) -> Self {
        Self { axis0, axis1 }
    }

    /// Return the coefficient along derivative axis zero.
    pub(crate) fn axis0(&self) -> &R {
        &self.axis0
    }

    /// Return the coefficient along derivative axis one.
    pub(crate) fn axis1(&self) -> &R {
        &self.axis1
    }

    /// Consume the direction and return `(axis0, axis1)`.
    pub(crate) fn into_parts(self) -> (R, R) {
        (self.axis0, self.axis1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::{arr0, array};

    #[test]
    fn direction_preserves_axis_order() {
        let direction = BivariateDirection::new(2.0, -3.0);

        assert_eq!(direction.axis0(), &2.0);
        assert_eq!(direction.axis1(), &-3.0);
        assert_eq!(direction.into_parts(), (2.0, -3.0));
    }

    #[test]
    fn array_linear_combination_applies_coefficients_pointwise() {
        let first = array![1.0, 2.0, 3.0];
        let second = array![4.0, 5.0, 6.0];

        let result = LinearCombination::linear_combination(first, 2.0, second, -0.5);

        assert_eq!(result, array![0.0, 1.5, 3.0]);
    }

    #[test]
    fn array_add_scaled_reuses_left_value_semantically() {
        let left = array![1.0, 2.0];
        let right = array![3.0, 4.0];

        let result = left.add_scaled(right, 2.0);

        assert_eq!(result, array![7.0, 10.0]);
    }

    #[test]
    fn first_order_contraction_preserves_value() {
        let parts = BivariateFirstParts::new(arr0(10.0), arr0(2.0), arr0(3.0));

        let contracted = parts.contract_direction(BivariateDirection::new(4.0, 5.0));

        let (value, first) = contracted.into_parts();

        assert_eq!(value, arr0(10.0));
        assert_eq!(first, arr0(23.0));
    }

    #[test]
    fn first_order_contraction_selects_axis_zero() {
        let parts = BivariateFirstParts::new(arr0(10.0), arr0(2.0), arr0(3.0));

        let contracted = parts.contract_direction(BivariateDirection::new(1.0, 0.0));

        assert_eq!(contracted.into_parts(), (arr0(10.0), arr0(2.0)),);
    }

    #[test]
    fn first_order_contraction_selects_axis_one() {
        let parts = BivariateFirstParts::new(arr0(10.0), arr0(2.0), arr0(3.0));

        let contracted = parts.contract_direction(BivariateDirection::new(0.0, 1.0));

        assert_eq!(contracted.into_parts(), (arr0(10.0), arr0(3.0)),);
    }

    #[test]
    fn first_order_contraction_does_not_normalise_direction() {
        let parts = BivariateFirstParts::new(arr0(10.0), arr0(2.0), arr0(3.0));

        let unit_scaled = parts
            .clone()
            .contract_direction(BivariateDirection::new(1.0, 2.0));

        let doubled = parts.contract_direction(BivariateDirection::new(2.0, 4.0));

        let (_, unit_scaled_first) = unit_scaled.into_parts();
        let (_, doubled_first) = doubled.into_parts();

        assert_eq!(unit_scaled_first, &arr0(8.0));
        assert_eq!(doubled_first, &arr0(16.0));
    }

    #[test]
    fn second_order_contraction_matches_gradient_and_hessian_formula() {
        let parts = BivariateSecondParts::new(
            arr0(10.0),
            arr0(2.0),  // ∂₀f
            arr0(3.0),  // ∂₁f
            arr0(5.0),  // ∂₀₀f
            arr0(7.0),  // ∂₀₁f
            arr0(11.0), // ∂₁₁f
        );

        let contracted = parts.contract_direction(BivariateDirection::new(4.0, 6.0));

        let (value, first, second) = contracted.into_parts();

        let expected_first = 4.0 * 2.0 + 6.0 * 3.0;

        let expected_second =
            4.0_f64.powi(2) * 5.0 + 2.0 * 4.0 * 6.0 * 7.0 + 6.0_f64.powi(2) * 11.0;

        assert_eq!(value, arr0(10.0));
        assert_eq!(first, arr0(expected_first));
        assert_eq!(second, arr0(expected_second));
    }

    #[test]
    fn second_order_axis_zero_contraction_selects_axis_zero_components() {
        let parts = BivariateSecondParts::new(
            arr0(10.0),
            arr0(2.0),
            arr0(3.0),
            arr0(5.0),
            arr0(7.0),
            arr0(11.0),
        );

        let contracted = parts.contract_direction(BivariateDirection::new(1.0, 0.0));

        assert_eq!(contracted.into_parts(), (arr0(10.0), arr0(2.0), arr0(5.0)),);
    }

    #[test]
    fn second_order_axis_one_contraction_selects_axis_one_components() {
        let parts = BivariateSecondParts::new(
            arr0(10.0),
            arr0(2.0),
            arr0(3.0),
            arr0(5.0),
            arr0(7.0),
            arr0(11.0),
        );

        let contracted = parts.contract_direction(BivariateDirection::new(0.0, 1.0));

        assert_eq!(contracted.into_parts(), (arr0(10.0), arr0(3.0), arr0(11.0)),);
    }

    #[test]
    fn second_order_mixed_term_has_factor_two() {
        let parts = BivariateSecondParts::new(
            arr0(0.0),
            arr0(0.0),
            arr0(0.0),
            arr0(0.0),
            arr0(3.0),
            arr0(0.0),
        );

        let contracted = parts.contract_direction(BivariateDirection::new(2.0, 5.0));

        let (_, _, second) = contracted.into_parts();

        assert_eq!(second, arr0(2.0 * 2.0 * 5.0 * 3.0));
    }

    #[test]
    fn contraction_operates_pointwise_on_arrays() {
        let parts = BivariateSecondParts::new(
            array![10.0, 20.0],
            array![1.0, 2.0],
            array![3.0, 4.0],
            array![5.0, 6.0],
            array![7.0, 8.0],
            array![9.0, 10.0],
        );

        let contracted = parts.contract_direction(BivariateDirection::new(2.0, -1.0));

        let (value, first, second) = contracted.into_parts();

        assert_eq!(value, array![10.0, 20.0]);

        assert_eq!(first, array![2.0 * 1.0 - 3.0, 2.0 * 2.0 - 4.0,],);

        assert_eq!(
            second,
            array![4.0 * 5.0 - 4.0 * 7.0 + 9.0, 4.0 * 6.0 - 4.0 * 8.0 + 10.0,],
        );
    }

    #[test]
    fn zero_direction_produces_zero_derivatives() {
        let parts = BivariateSecondParts::new(
            arr0(10.0),
            arr0(2.0),
            arr0(3.0),
            arr0(5.0),
            arr0(7.0),
            arr0(11.0),
        );

        let contracted = parts.contract_direction(BivariateDirection::new(0.0, 0.0));

        assert_eq!(contracted.into_parts(), (arr0(10.0), arr0(0.0), arr0(0.0)),);
    }

    #[test]
    fn second_order_contraction_is_hessian_quadratic_form_only() {
        /*
         * f₀ = 7, f₁ = 11 are deliberately non-zero.
         *
         * Contraction with v = (2, 3) should depend only on the Hessian
         * when computing the second directional derivative. No path-curvature
         * term involving the gradient is introduced.
         */
        let parts = BivariateSecondParts::new(
            arr0(0.0),
            arr0(7.0),
            arr0(11.0),
            arr0(5.0),
            arr0(13.0),
            arr0(17.0),
        );

        let contracted = parts.contract_direction(BivariateDirection::new(2.0, 3.0));

        let (_, _, second) = contracted.into_parts();

        let expected = 4.0 * 5.0 + 2.0 * 2.0 * 3.0 * 13.0 + 9.0 * 17.0;

        assert_eq!(second, arr0(expected));
    }
}
