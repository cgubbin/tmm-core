//! Policies for extracting coordinate-free derivative parts.
//!
//! Internal calculations retain values and derivatives in algebraic types such
//! as univariate and bivariate jets. This module defines policies that select
//! the derivative information required by a caller and convert an internal
//! quantity into a small, flattened parts container.
//!
//! The available policies are:
//!
//! - [`ValueOnly`], which retains only the value;
//! - [`FirstDirectional`], which retains a value and one first derivative;
//! - [`SecondDirectional`], which retains directional derivatives through
//!   second order;
//! - [`FirstBivariate`], which retains a value and two first derivatives;
//! - [`SecondBivariate`], which retains a value, a two-component gradient, and
//!   a symmetric two-dimensional Hessian.
//!
//! The extracted parts do not identify the physical parameters represented by
//! their derivative axes. Parameter metadata is attached later when the parts
//! are assembled into a public differential response.
//!
//! Policies are independent of concrete jet types. Any internal type
//! implementing the corresponding decomposition capability can be processed,
//! including composite quantities such as plane-wave amplitudes and powers.

use super::{
    BivariateFirstParts, BivariateSecondParts, DirectionalFirstParts, DirectionalSecondParts,
    IntoBivariateFirst, IntoBivariateSecond, IntoFirst, IntoSecond, IntoValue, ValuePart,
};

/// Retain values while discarding all derivative components.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ValueOnly;

/// Retain values and one first directional derivative.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FirstDirectional;

/// Retain values and directional derivatives through second order.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SecondDirectional;

/// Retain values and first derivatives with respect to two coordinates.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FirstBivariate;

/// Retain values, a bivariate gradient, and a symmetric bivariate Hessian.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SecondBivariate;

/// Extension trait for extracting coordinate-free derivative parts from an
/// internal algebraic quantity.
#[doc(hidden)]
pub trait IntoDerivativeParts: Sized {
    /// Consume `self` and extract the components selected by `policy`.
    fn into_derivative_parts<P>(self, policy: &P) -> P::Output
    where
        P: DerivativePartsPolicy<Self>,
    {
        policy.derivative_parts(self)
    }
}

impl<T> IntoDerivativeParts for T {}

/// Converts an internal algebraic quantity into a public result.
///
/// Implementations select the required decomposition capability and determine
/// the derivative representation stored in the resulting
/// [`DifferentialResponse`].
#[doc(hidden)]
pub trait DerivativePartsPolicy<T> {
    /// Public response produced by this policy.
    type Output;

    /// Convert `input` into its public value and derivative representation.
    fn derivative_parts(&self, input: T) -> Self::Output;
}

impl<T> DerivativePartsPolicy<T> for ValueOnly
where
    T: IntoValue,
{
    type Output = ValuePart<T::Value>;

    fn derivative_parts(&self, input: T) -> Self::Output {
        input.into_value()
    }
}

impl<T> DerivativePartsPolicy<T> for FirstDirectional
where
    T: IntoFirst,
{
    type Output = DirectionalFirstParts<T::Value>;

    fn derivative_parts(&self, input: T) -> Self::Output {
        let (value, first) = input.into_first().into_parts();
        DirectionalFirstParts::new(value, first)
    }
}

impl<T> DerivativePartsPolicy<T> for SecondDirectional
where
    T: IntoSecond,
{
    type Output = DirectionalSecondParts<T::Value>;

    fn derivative_parts(&self, input: T) -> Self::Output {
        let (value, first, second) = input.into_second().into_parts();
        DirectionalSecondParts::new(value, first, second)
    }
}

impl<T> DerivativePartsPolicy<T> for FirstBivariate
where
    T: IntoBivariateFirst,
{
    type Output = BivariateFirstParts<T::Value>;

    fn derivative_parts(&self, input: T) -> Self::Output {
        let (value, x, y) = input.into_bivariate_first().into_parts();
        BivariateFirstParts::new(value, x, y)
    }
}

impl<T> DerivativePartsPolicy<T> for SecondBivariate
where
    T: IntoBivariateSecond,
{
    type Output = BivariateSecondParts<T::Value>;

    fn derivative_parts(&self, input: T) -> Self::Output {
        let (value, x, y, x_x, x_y, y_y) = input.into_bivariate_second().into_parts();

        BivariateSecondParts::new(value, x, y, x_x, x_y, y_y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct ValueInput {
        value: i32,
    }

    impl IntoValue for ValueInput {
        type Value = i32;

        fn into_value(self) -> ValuePart<Self::Value> {
            ValuePart::new(self.value)
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct FirstInput {
        value: i32,
        first: i32,
    }

    impl IntoValue for FirstInput {
        type Value = i32;

        fn into_value(self) -> ValuePart<Self::Value> {
            ValuePart::new(self.value)
        }
    }

    impl IntoFirst for FirstInput {
        fn into_first(self) -> DirectionalFirstParts<Self::Value> {
            DirectionalFirstParts::new(self.value, self.first)
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct SecondInput {
        value: i32,
        first: i32,
        second: i32,
    }

    impl IntoValue for SecondInput {
        type Value = i32;

        fn into_value(self) -> ValuePart<Self::Value> {
            ValuePart::new(self.value)
        }
    }

    impl IntoFirst for SecondInput {
        fn into_first(self) -> DirectionalFirstParts<Self::Value> {
            DirectionalFirstParts::new(self.value, self.first)
        }
    }

    impl IntoSecond for SecondInput {
        fn into_second(self) -> DirectionalSecondParts<Self::Value> {
            DirectionalSecondParts::new(self.value, self.first, self.second)
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct FirstBivariateInput {
        value: i32,
        axis0: i32,
        axis1: i32,
    }

    impl IntoValue for FirstBivariateInput {
        type Value = i32;

        fn into_value(self) -> ValuePart<Self::Value> {
            ValuePart::new(self.value)
        }
    }

    impl IntoBivariateFirst for FirstBivariateInput {
        fn into_bivariate_first(self) -> BivariateFirstParts<Self::Value> {
            BivariateFirstParts::new(self.value, self.axis0, self.axis1)
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct SecondBivariateInput {
        value: i32,
        axis0: i32,
        axis1: i32,
        axis0_axis0: i32,
        axis0_axis1: i32,
        axis1_axis1: i32,
    }

    impl IntoValue for SecondBivariateInput {
        type Value = i32;

        fn into_value(self) -> ValuePart<Self::Value> {
            ValuePart::new(self.value)
        }
    }

    impl IntoBivariateFirst for SecondBivariateInput {
        fn into_bivariate_first(self) -> BivariateFirstParts<Self::Value> {
            BivariateFirstParts::new(self.value, self.axis0, self.axis1)
        }
    }

    impl IntoBivariateSecond for SecondBivariateInput {
        fn into_bivariate_second(self) -> BivariateSecondParts<Self::Value> {
            BivariateSecondParts::new(
                self.value,
                self.axis0,
                self.axis1,
                self.axis0_axis0,
                self.axis0_axis1,
                self.axis1_axis1,
            )
        }
    }

    #[test]
    fn value_only_extracts_only_the_value() {
        let input = ValueInput { value: 10 };

        let parts = input.into_derivative_parts(&ValueOnly);

        assert_eq!(parts.into_inner(), 10);
    }

    #[test]
    fn first_directional_extracts_value_and_first_derivative() {
        let input = FirstInput {
            value: 10,
            first: 20,
        };

        let parts = input.into_derivative_parts(&FirstDirectional);

        assert_eq!(parts.into_parts(), (10, 20));
    }

    #[test]
    fn second_directional_extracts_all_directional_components() {
        let input = SecondInput {
            value: 10,
            first: 20,
            second: 30,
        };

        let parts = input.into_derivative_parts(&SecondDirectional);

        assert_eq!(parts.into_parts(), (10, 20, 30));
    }

    #[test]
    fn first_bivariate_extracts_value_and_both_first_derivatives() {
        let input = FirstBivariateInput {
            value: 10,
            axis0: 20,
            axis1: 30,
        };

        let parts = input.into_derivative_parts(&FirstBivariate);

        assert_eq!(parts.into_parts(), (10, 20, 30));
    }

    #[test]
    fn second_bivariate_extracts_bivariate_gradient_and_symmetric_bivariate_hessian() {
        let input = SecondBivariateInput {
            value: 10,
            axis0: 20,
            axis1: 30,
            axis0_axis0: 40,
            axis0_axis1: 50,
            axis1_axis1: 60,
        };

        let parts = input.into_derivative_parts(&SecondBivariate);

        assert_eq!(parts.into_parts(), (10, 20, 30, 40, 50, 60),);
    }

    #[test]
    fn policies_support_non_copy_component_types() {
        #[derive(Debug, PartialEq, Eq)]
        struct Input {
            value: String,
            first: String,
        }

        impl IntoValue for Input {
            type Value = String;

            fn into_value(self) -> ValuePart<Self::Value> {
                ValuePart::new(self.value)
            }
        }

        impl IntoFirst for Input {
            fn into_first(self) -> DirectionalFirstParts<Self::Value> {
                DirectionalFirstParts::new(self.value, self.first)
            }
        }

        let input = Input {
            value: String::from("value"),
            first: String::from("first"),
        };

        let parts = input.into_derivative_parts(&FirstDirectional);

        assert_eq!(
            parts.into_parts(),
            (String::from("value"), String::from("first"),),
        );
    }
}
