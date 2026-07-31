//! Decomposition of internal jet-valued quantities.
//!
//! This module separates jet algebra from public differential storage.
//! Each [`IntoValue`]-family trait describes a derivative capability and
//! decomposes an internal value into a small parts container:
//!
//! - [`IntoValue`] extracts the sampled value;
//! - [`IntoFirst`] extracts a value and one directional first derivative;
//! - [`IntoSecond`] extracts directional derivatives through second order;
//! - [`IntoGradient`] extracts first derivatives with respect to two
//!   coordinates;
//! - [`IntoHessian`] extracts first and second derivatives with respect to two
//!   coordinates.
//!
//! Composite internal types can implement these traits recursively. This
//! allows crystallisation policies to operate on complete structures, rather
//! than crystallising each scalar observable independently.

use crate::algebra::{Jet0, Jet1, Jet2, JetBivariate1, JetBivariate2};

/// A value
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ValueParts<T> {
    pub(crate) value: T,
}

impl<T> ValueParts<T> {
    /// Construct a value
    pub(crate) fn new(value: T) -> Self {
        Self { value }
    }

    pub(crate) fn value(&self) -> &T {
        &self.value
    }

    /// Consume the container and return `value`.
    pub(crate) fn into_inner(self) -> T {
        self.value
    }
}

/// A value and one first directional derivative.
///
/// Both fields have the same structural type. For a composite input, `value`
/// contains the values of every component and `first` contains the
/// corresponding first derivatives.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DirectionalFirstParts<T> {
    pub(crate) value: T,
    pub(crate) first: T,
}

impl<T> DirectionalFirstParts<T> {
    /// Construct directional first-order parts.
    pub(crate) fn new(value: T, first: T) -> Self {
        Self { value, first }
    }

    pub(crate) fn value(&self) -> &T {
        &self.value
    }

    pub(crate) fn first(&self) -> &T {
        &self.first
    }

    /// Consume the container and return `(value, first)`.
    pub(crate) fn into_parts(self) -> (T, T) {
        (self.value, self.first)
    }
}

/// A value and directional derivatives through second order.
///
/// `second` is the repeated derivative along the same direction as `first`,
/// rather than a mixed derivative.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DirectionalSecondParts<T> {
    pub(crate) value: T,
    pub(crate) first: T,
    pub(crate) second: T,
}

impl<T> DirectionalSecondParts<T> {
    /// Construct directional second-order parts.
    pub(crate) fn new(value: T, first: T, second: T) -> Self {
        Self {
            value,
            first,
            second,
        }
    }

    pub(crate) fn value(&self) -> &T {
        &self.value
    }

    pub(crate) fn first(&self) -> &T {
        &self.first
    }

    pub(crate) fn second(&self) -> &T {
        &self.second
    }

    /// Consume the container and return `(value, first, second)`.
    pub(crate) fn into_parts(self) -> (T, T, T) {
        (self.value, self.first, self.second)
    }
}

/// A value and first derivatives with respect to two coordinates.
///
/// The coordinate interpretation is supplied by the caller. The `axis0` and `y`
/// fields denote the first and second coordinates of the underlying
/// bivariate jet; they are not necessarily spatial coordinates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BivariateFirstParts<T> {
    pub(crate) value: T,
    pub(crate) axis0: T,
    pub(crate) axis1: T,
}

impl<T> BivariateFirstParts<T> {
    /// Construct bivariate first-order parts.
    pub(crate) fn new(value: T, axis0: T, axis1: T) -> Self {
        Self {
            value,
            axis0,
            axis1,
        }
    }

    pub(crate) fn axis0(&self) -> &T {
        &self.axis0
    }

    pub(crate) fn axis1(&self) -> &T {
        &self.axis1
    }

    /// Consume the container and return `(value, axis0, axis1)`.
    pub(crate) fn into_parts(self) -> (T, T, T) {
        (self.value, self.axis0, self.axis1)
    }
}

/// A value, gradient, and symmetric Hessian over two coordinates.
///
/// The stored second derivatives are `axis0_axis0`, `axis0_axis1`, and `axis1_axis1`. Only one mixed
/// component is stored because the underlying Hessian representation assumes
/// symmetry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct BivariateSecondParts<T> {
    pub(crate) value: T,

    pub(crate) axis0: T,
    pub(crate) axis1: T,

    pub(crate) axis0_axis0: T,
    pub(crate) axis0_axis1: T,
    pub(crate) axis1_axis1: T,
}

impl<T> BivariateSecondParts<T> {
    /// Construct bivariate second-order parts.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        value: T,
        axis0: T,
        axis1: T,
        axis0_axis0: T,
        axis0_axis1: T,
        axis1_axis1: T,
    ) -> Self {
        Self {
            value,
            axis0,
            axis1,
            axis0_axis0,
            axis0_axis1,
            axis1_axis1,
        }
    }

    pub(crate) fn axis0(&self) -> &T {
        &self.axis0
    }

    pub(crate) fn axis1(&self) -> &T {
        &self.axis1
    }

    pub(crate) fn axis0_axis0(&self) -> &T {
        &self.axis0_axis0
    }

    pub(crate) fn axis0_axis1(&self) -> &T {
        &self.axis0_axis1
    }

    pub(crate) fn axis1_axis1(&self) -> &T {
        &self.axis1_axis1
    }

    /// Consume the container.
    ///
    /// Components are returned in the order
    /// `(value, axis0, axis1, axis0_axis0, axis0_axis1, axis1_axis1)`.
    pub(crate) fn into_parts(self) -> (T, T, T, T, T, T) {
        (
            self.value,
            self.axis0,
            self.axis1,
            self.axis0_axis0,
            self.axis0_axis1,
            self.axis1_axis1,
        )
    }
}

/// Extract the value component from an internal algebraic quantity.
pub(crate) trait IntoValue {
    /// Structure obtained after removing all derivative storage.
    type Value;

    /// Consume the input and return its value component.
    fn into_value(self) -> Self::Value;
}

/// Extract a value and one directional first derivative.
pub(crate) trait IntoFirst: IntoValue {
    /// Consume the input and return its directional first-order parts.
    fn into_first(self) -> DirectionalFirstParts<Self::Value>;
}

/// Extract a value and directional derivatives through second order.
pub(crate) trait IntoSecond: IntoFirst {
    /// Consume the input and return its directional second-order parts.
    fn into_second(self) -> DirectionalSecondParts<Self::Value>;
}

/// Extract a value and first derivatives over two coordinates.
pub(crate) trait IntoGradient: IntoValue {
    /// Consume the input and return its bivariate first-order parts.
    fn into_gradient(self) -> BivariateFirstParts<Self::Value>;
}

/// Extract a value, gradient, and Hessian over two coordinates.
pub(crate) trait IntoHessian: IntoGradient {
    /// Consume the input and return its bivariate second-order parts.
    fn into_hessian(self) -> BivariateSecondParts<Self::Value>;
}

impl<I, P> IntoValue for Jet0<I, P> {
    type Value = I;

    fn into_value(self) -> Self::Value {
        self.into_inner()
    }
}

impl<I, P> IntoValue for Jet1<I, P> {
    type Value = I;

    fn into_value(self) -> Self::Value {
        let (value, ..) = self.into_parts();
        value
    }
}

impl<I, P> IntoValue for Jet2<I, P> {
    type Value = I;

    fn into_value(self) -> Self::Value {
        let (value, ..) = self.into_parts();
        value
    }
}

impl<I, P> IntoValue for JetBivariate1<I, P> {
    type Value = I;

    fn into_value(self) -> Self::Value {
        let (value, ..) = self.into_parts();
        value
    }
}

impl<I, P> IntoValue for JetBivariate2<I, P> {
    type Value = I;

    fn into_value(self) -> Self::Value {
        let (value, ..) = self.into_parts();
        value
    }
}

impl<I, P> IntoFirst for Jet1<I, P> {
    fn into_first(self) -> DirectionalFirstParts<Self::Value> {
        let (value, first) = self.into_parts();
        DirectionalFirstParts::new(value, first)
    }
}

impl<I, P> IntoFirst for Jet2<I, P> {
    fn into_first(self) -> DirectionalFirstParts<Self::Value> {
        let (value, first, ..) = self.into_parts();
        DirectionalFirstParts::new(value, first)
    }
}

impl<I, P> IntoSecond for Jet2<I, P> {
    fn into_second(self) -> DirectionalSecondParts<Self::Value> {
        let (value, first, second) = self.into_parts();
        DirectionalSecondParts::new(value, first, second)
    }
}

impl<I, P> IntoGradient for JetBivariate1<I, P> {
    fn into_gradient(self) -> BivariateFirstParts<Self::Value> {
        let (value, gradient) = self.into_parts();
        let (axis0, axis1) = gradient.into_parts();

        BivariateFirstParts::new(value, axis0, axis1)
    }
}

impl<I, P> IntoGradient for JetBivariate2<I, P> {
    fn into_gradient(self) -> BivariateFirstParts<Self::Value> {
        let (value, gradient, ..) = self.into_parts();
        let (axis0, axis1) = gradient.into_parts();

        BivariateFirstParts::new(value, axis0, axis1)
    }
}

impl<I, P> IntoHessian for JetBivariate2<I, P> {
    fn into_hessian(self) -> BivariateSecondParts<Self::Value> {
        let (value, gradient, hessian) = self.into_parts();
        let (axis0, axis1) = gradient.into_parts();
        let (axis0_axis0, axis0_axis1, axis1_axis1) = hessian.into_parts();

        BivariateSecondParts::new(value, axis0, axis1, axis0_axis0, axis0_axis1, axis1_axis1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directional_first_parts_preserve_component_order() {
        let parts = DirectionalFirstParts::new(1, 2);

        assert_eq!(parts.into_parts(), (1, 2));
    }

    #[test]
    fn directional_second_parts_preserve_component_order() {
        let parts = DirectionalSecondParts::new(1, 2, 3);

        assert_eq!(parts.into_parts(), (1, 2, 3));
    }

    #[test]
    fn gradient_parts_preserve_component_order() {
        let parts = BivariateFirstParts::new(1, 2, 3);

        assert_eq!(parts.into_parts(), (1, 2, 3));
    }

    #[test]
    fn hessian_parts_preserve_component_order() {
        let parts = BivariateSecondParts::new(1, 2, 3, 4, 5, 6);

        assert_eq!(parts.into_parts(), (1, 2, 3, 4, 5, 6),);
    }

    #[test]
    fn parts_support_non_copy_values() {
        let parts = DirectionalFirstParts::new(String::from("value"), String::from("first"));

        let (value, first) = parts.into_parts();

        assert_eq!(value, "value");
        assert_eq!(first, "first");
    }
}

#[cfg(test)]
mod jet_tests {
    use super::*;

    // Replace these two imports with the actual internal derivative storage
    // types used by the bivariate jets.
    use crate::differential::{BivariateGradient as JetGradient, BivariateHessian as JetHessian};

    #[test]
    fn jet0_into_value_extracts_value() {
        let jet = Jet0::<_, ()>::new(10);

        assert_eq!(jet.into_value(), 10);
    }

    #[test]
    fn jet1_into_value_discards_first_derivative() {
        let jet = Jet1::<_, ()>::from_parts(10, 20);

        assert_eq!(jet.into_value(), 10);
    }

    #[test]
    fn jet1_into_first_extracts_value_and_derivative() {
        let jet = Jet1::<_, ()>::from_parts(10, 20);

        assert_eq!(jet.into_first().into_parts(), (10, 20),);
    }

    #[test]
    fn jet2_into_value_discards_both_derivatives() {
        let jet = Jet2::<_, ()>::from_parts(10, 20, 30);

        assert_eq!(jet.into_value(), 10);
    }

    #[test]
    fn jet2_into_first_discards_only_second_derivative() {
        let jet = Jet2::<_, ()>::from_parts(10, 20, 30);

        assert_eq!(jet.into_first().into_parts(), (10, 20),);
    }

    #[test]
    fn jet2_into_second_extracts_all_components() {
        let jet = Jet2::<_, ()>::from_parts(10, 20, 30);

        assert_eq!(jet.into_second().into_parts(), (10, 20, 30),);
    }

    #[test]
    fn bivariate_first_into_value_discards_gradient() {
        let jet = JetBivariate1::<_, ()>::from_parts(10, JetGradient::new(20, 30));

        assert_eq!(jet.into_value(), 10);
    }

    #[test]
    fn bivariate_first_into_gradient_extracts_all_components() {
        let jet = JetBivariate1::<_, ()>::from_parts(10, JetGradient::new(20, 30));

        assert_eq!(jet.into_gradient().into_parts(), (10, 20, 30),);
    }

    #[test]
    fn bivariate_second_into_value_discards_all_derivatives() {
        let jet = JetBivariate2::<_, ()>::from_parts(
            10,
            JetGradient::new(20, 30),
            JetHessian::new(40, 50, 60),
        );

        assert_eq!(jet.into_value(), 10);
    }

    #[test]
    fn bivariate_second_into_gradient_discards_hessian() {
        let jet = JetBivariate2::<_, ()>::from_parts(
            10,
            JetGradient::new(20, 30),
            JetHessian::new(40, 50, 60),
        );

        assert_eq!(jet.into_gradient().into_parts(), (10, 20, 30),);
    }

    #[test]
    fn bivariate_second_into_hessian_extracts_all_components() {
        let jet = JetBivariate2::<_, ()>::from_parts(
            10,
            JetGradient::new(20, 30),
            JetHessian::new(40, 50, 60),
        );

        assert_eq!(jet.into_hessian().into_parts(), (10, 20, 30, 40, 50, 60),);
    }
}

#[cfg(test)]
mod capability_tests {
    use super::*;

    fn assert_into_value<T: IntoValue>() {}
    fn assert_into_first<T: IntoFirst>() {}
    fn assert_into_second<T: IntoSecond>() {}
    fn assert_into_gradient<T: IntoGradient>() {}
    fn assert_into_hessian<T: IntoHessian>() {}

    #[test]
    fn directional_jets_expose_expected_capabilities() {
        assert_into_value::<Jet0<f64, ()>>();

        assert_into_value::<Jet1<f64, ()>>();
        assert_into_first::<Jet1<f64, ()>>();

        assert_into_value::<Jet2<f64, ()>>();
        assert_into_first::<Jet2<f64, ()>>();
        assert_into_second::<Jet2<f64, ()>>();
    }

    #[test]
    fn bivariate_jets_expose_expected_capabilities() {
        assert_into_value::<JetBivariate1<f64, ()>>();
        assert_into_gradient::<JetBivariate1<f64, ()>>();

        assert_into_value::<JetBivariate2<f64, ()>>();
        assert_into_gradient::<JetBivariate2<f64, ()>>();
        assert_into_hessian::<JetBivariate2<f64, ()>>();
    }
}
