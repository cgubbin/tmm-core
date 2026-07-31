//! Crystallisation policies for public differential responses.
//!
//! Backend calculations operate on internal jet-valued structures. This
//! module defines policies that separate those structures into values and
//! derivatives and package the result as a [`DifferentialResponse`].
//!
//! The available policies determine the derivative representation exposed to
//! the caller:
//!
//! - [`ValueOnly`] discards all derivative components;
//! - [`FirstDirectional`] retains one directional first derivative;
//! - [`SecondDirectional`] retains directional derivatives through second
//!   order;
//! - [`FirstBivariate`] retains first derivatives with respect to two
//!   coordinates;
//! - [`SecondBivariate`] retains the corresponding gradient and symmetric
//!   Hessian.
//!
//! The policies are independent of the concrete jet types. Any internal type
//! implementing the appropriate decomposition trait can be crystallised,
//! including composite structures such as plane-wave observables.

use crate::{
    algebra::{
        ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, Jet0, Jet1, Jet2,
        JetBivariate1, JetBivariate2,
    },
    crystallise::{
        BivariateFirstParts, BivariateSecondParts, DirectionalFirstParts, DirectionalSecondParts,
        IntoFirst, IntoGradient, IntoHessian, IntoSecond, IntoValue, ValueParts,
    },
    differential::{
        BivariateFirst, BivariateGradient, BivariateHessian, BivariateSecond, DifferentialResponse,
        DirectionalFirst, DirectionalSecond, NoDerivatives,
    },
};

use ndarray::{Array, Dimension};

/// Retain values while discarding all derivative components.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct ValueOnly;

/// Retain values and one first directional derivative.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FirstDirectional;

/// Retain values and directional derivatives through second order.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SecondDirectional;

/// Retain values and first derivatives with respect to two coordinates.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct FirstBivariate;

/// Retain values, a bivariate gradient, and a symmetric bivariate Hessian.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct SecondBivariate;

/// Extension trait for converting an internal algebraic structure into a
/// public differential response.
pub(crate) trait Crystallise: Sized {
    /// Apply `policy` to separate values and derivatives from `self`.
    fn crystallise<P>(self, policy: &P) -> P::Output
    where
        P: CrystallisePolicy<Self>,
    {
        policy.crystallise(self)
    }
}

impl<T> Crystallise for T {}

/// Converts an internal algebraic quantity into a public result.
///
/// Implementations select the required decomposition capability and determine
/// the derivative representation stored in the resulting
/// [`DifferentialResponse`].
pub(crate) trait CrystallisePolicy<T> {
    /// Public response produced by this policy.
    type Output;

    /// Convert `input` into its public value and derivative representation.
    fn crystallise(&self, input: T) -> Self::Output;
}

impl<T> CrystallisePolicy<T> for ValueOnly
where
    T: IntoValue,
{
    type Output = ValueParts<T::Value>;

    fn crystallise(&self, input: T) -> Self::Output {
        ValueParts::new(input.into_value())
    }
}

impl<T> CrystallisePolicy<T> for FirstDirectional
where
    T: IntoFirst,
{
    type Output = DirectionalFirstParts<T::Value>;

    fn crystallise(&self, input: T) -> Self::Output {
        let (value, first) = input.into_first().into_parts();
        DirectionalFirstParts::new(value, first)
    }
}

impl<T> CrystallisePolicy<T> for SecondDirectional
where
    T: IntoSecond,
{
    type Output = DirectionalSecondParts<T::Value>;

    fn crystallise(&self, input: T) -> Self::Output {
        let (value, first, second) = input.into_second().into_parts();
        DirectionalSecondParts::new(value, first, second)
    }
}

impl<T> CrystallisePolicy<T> for FirstBivariate
where
    T: IntoGradient,
{
    type Output = BivariateFirstParts<T::Value>;

    fn crystallise(&self, input: T) -> Self::Output {
        let (value, x, y) = input.into_gradient().into_parts();
        BivariateFirstParts::new(value, x, y)
    }
}

impl<T> CrystallisePolicy<T> for SecondBivariate
where
    T: IntoHessian,
{
    type Output = BivariateSecondParts<T::Value>;

    fn crystallise(&self, input: T) -> Self::Output {
        let (value, x, y, x_x, x_y, y_y) = input.into_hessian().into_parts();

        BivariateSecondParts::new(value, x, y, x_x, x_y, y_y)
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     use crate::crystallise::{
//         BivariateFirstParts, BivariateSecondParts, DirectionalFirstParts,
//         DirectionalSecondParts,
//     };

//     #[derive(Clone, Copy, Debug, PartialEq, Eq)]
//     struct TestInput {
//         value: i32,
//         first: i32,
//         second: i32,
//         x: i32,
//         y: i32,
//         x_x: i32,
//         x_y: i32,
//         y_y: i32,
//     }

//     impl TestInput {
//         fn sample() -> Self {
//             Self {
//                 value: 1,
//                 first: 2,
//                 second: 3,
//                 x: 4,
//                 y: 5,
//                 x_x: 6,
//                 x_y: 7,
//                 y_y: 8,
//             }
//         }
//     }

//     impl IntoValue for TestInput {
//         type Value = i32;

//         fn into_value(self) -> Self::Value {
//             self.value
//         }
//     }

//     impl IntoFirst for TestInput {
//         fn into_first(self) -> DirectionalFirstParts<Self::Value> {
//             DirectionalFirstParts::new(self.value, self.first)
//         }
//     }

//     impl IntoSecond for TestInput {
//         fn into_second(self) -> DirectionalSecondParts<Self::Value> {
//             DirectionalSecondParts::new(self.value, self.first, self.second)
//         }
//     }

//     impl IntoGradient for TestInput {
//         fn into_gradient(self) -> BivariateFirstParts<Self::Value> {
//             BivariateFirstParts::new(self.value, self.x, self.y)
//         }
//     }

//     impl IntoHessian for TestInput {
//         fn into_hessian(self) -> BivariateSecondParts<Self::Value> {
//             BivariateSecondParts::new(self.value, self.x, self.y, self.x_x, self.x_y, self.y_y)
//         }
//     }

//     #[test]
//     fn value_only_retains_value() {
//         let response = TestInput::sample().crystallise(&ValueOnly);

//         assert_eq!(*response.values(), 1);
//     }

//     #[test]
//     fn first_directional_retains_value_coordinate_and_first_derivative() {
//         let coordinate = DirectionalCoordinate::VacuumWavenumber;

//         let response = TestInput::sample().crystallise(&FirstDirectional::new(coordinate));

//         assert_eq!(*response.values(), 1);
//         assert_eq!(response.derivatives().coordinate(), coordinate);
//         assert_eq!(*response.derivatives().first(), 2);
//     }

//     #[test]
//     fn second_directional_retains_value_coordinate_and_both_derivatives() {
//         let coordinate = DirectionalCoordinate::ParallelWavenumber;

//         let response = TestInput::sample().crystallise(&SecondDirectional::new(coordinate));

//         assert_eq!(*response.values(), 1);
//         assert_eq!(response.derivatives().coordinate(), coordinate);
//         assert_eq!(*response.derivatives().first(), 2);
//         assert_eq!(*response.derivatives().second(), 3);
//     }

//     #[test]
//     fn first_bivariate_retains_value_and_both_first_derivatives() {
//         let coordinate0 = DirectionalCoordinate::ParallelWavenumber;
//         let coordinate1 = DirectionalCoordinate::VacuumWavenumber;
//         let response =
//             TestInput::sample().crystallise(&FirstBivariate::new(coordinate0, coordinate1));

//         assert_eq!(*response.values(), 1);
//         assert_eq!(*response.derivatives().axis0(), 4);
//         assert_eq!(*response.derivatives().axis1(), 5);
//     }

//     #[test]
//     fn second_bivariate_retains_value_gradient_and_hessian() {
//         let coordinate0 = DirectionalCoordinate::ParallelWavenumber;
//         let coordinate1 = DirectionalCoordinate::VacuumWavenumber;

//         let response =
//             TestInput::sample().crystallise(&SecondBivariate::new(coordinate0, coordinate1));

//         let first = response.derivatives().first();
//         let second = response.derivatives().second();

//         assert_eq!(*response.values(), 1);

//         assert_eq!(*first.axis0(), 4);
//         assert_eq!(*first.axis1(), 5);

//         assert_eq!(*second.axis0_axis0(), 6);
//         assert_eq!(*second.axis0_axis1(), 7);
//         assert_eq!(*second.axis1_axis1(), 8);
//     }

//     #[test]
//     fn second_bivariate_has_expected_response_type() {
//         let coordinate0 = DirectionalCoordinate::ParallelWavenumber;
//         let coordinate1 = DirectionalCoordinate::VacuumWavenumber;

//         let response =
//             TestInput::sample().crystallise(&SecondBivariate::new(coordinate0, coordinate1));

//         fn assert_type(_: &DifferentialResponse<i32, BivariateSecond<i32>>) {}

//         assert_type(&response);
//     }
// }
