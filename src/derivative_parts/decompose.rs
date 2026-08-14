//! Decomposition capabilities for internal algebraic quantities.
//!
//! The traits in this module describe which derivative components can be
//! extracted from an internal value:
//!
//! - [`IntoValue`] extracts only the value;
//! - [`IntoFirst`] extracts a value and one first directional derivative;
//! - [`IntoSecond`] additionally extracts the repeated second derivative;
//! - [`IntoBivariateFirst`] extracts first derivatives along two axes;
//! - [`IntoBivariateSecond`] additionally extracts the symmetric bivariate Hessian.
//!
//! Implementations are provided for the core jet types. Composite quantities
//! may implement the same traits recursively, allowing derivative-parts
//! policies to operate on complete observable structures.

use crate::algebra::{Jet0, Jet1, Jet2, JetBivariate1, JetBivariate2};

use super::{
    BivariateFirstParts, BivariateSecondParts, DirectionalFirstParts, DirectionalSecondParts,
    ValuePart,
};

/// Extract the value component from an internal algebraic quantity.
#[doc(hidden)]
pub trait IntoValue {
    /// Structure obtained after removing all derivative storage.
    type Value;

    /// Consume the input and return its value component.
    fn into_value(self) -> ValuePart<Self::Value>;
}

/// Extract a value and one directional first derivative.
#[doc(hidden)]
pub trait IntoFirst: IntoValue {
    /// Consume the input and return its directional first-order parts.
    fn into_first(self) -> DirectionalFirstParts<Self::Value>;
}

/// Extract a value and directional derivatives through second order.
#[doc(hidden)]
pub trait IntoSecond: IntoFirst {
    /// Consume the input and return its directional second-order parts.
    fn into_second(self) -> DirectionalSecondParts<Self::Value>;
}

/// Extract a value and first derivatives over two abstract coordinates.
///
/// This operation preserves both algebraic axes; it does not attach physical
/// parameter meanings to them.
#[doc(hidden)]
pub trait IntoBivariateFirst: IntoValue {
    /// Consume the input and return its bivariate first-order parts.
    fn into_bivariate_first(self) -> BivariateFirstParts<Self::Value>;
}

/// Extract a value, first and second, derivatives over two abstract coordinates.
///
/// This operation preserves both algebraic axes; it does not attach physical
/// parameter meanings to them.
#[doc(hidden)]
pub trait IntoBivariateSecond: IntoBivariateFirst {
    /// Consume the input and return its bivariate second-order parts.
    fn into_bivariate_second(self) -> BivariateSecondParts<Self::Value>;
}

impl<I, P> IntoValue for Jet0<I, P> {
    type Value = I;

    fn into_value(self) -> ValuePart<Self::Value> {
        ValuePart::new(self.into_inner())
    }
}

impl<I, P> IntoValue for Jet1<I, P> {
    type Value = I;

    fn into_value(self) -> ValuePart<Self::Value> {
        let (value, ..) = self.into_parts();
        ValuePart::new(value)
    }
}

impl<I, P> IntoValue for Jet2<I, P> {
    type Value = I;

    fn into_value(self) -> ValuePart<Self::Value> {
        let (value, ..) = self.into_parts();
        ValuePart::new(value)
    }
}

impl<I, P> IntoValue for JetBivariate1<I, P> {
    type Value = I;

    fn into_value(self) -> ValuePart<Self::Value> {
        let (value, ..) = self.into_parts();
        ValuePart::new(value)
    }
}

impl<I, P> IntoValue for JetBivariate2<I, P> {
    type Value = I;

    fn into_value(self) -> ValuePart<Self::Value> {
        let (value, ..) = self.into_parts();
        ValuePart::new(value)
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

impl<I, P> IntoBivariateFirst for JetBivariate1<I, P> {
    fn into_bivariate_first(self) -> BivariateFirstParts<Self::Value> {
        let (value, gradient) = self.into_parts();
        let (axis0, axis1) = gradient.into_parts();

        BivariateFirstParts::new(value, axis0, axis1)
    }
}

impl<I, P> IntoBivariateFirst for JetBivariate2<I, P> {
    fn into_bivariate_first(self) -> BivariateFirstParts<Self::Value> {
        let (value, gradient, ..) = self.into_parts();
        let (axis0, axis1) = gradient.into_parts();

        BivariateFirstParts::new(value, axis0, axis1)
    }
}

impl<I, P> IntoBivariateSecond for JetBivariate2<I, P> {
    fn into_bivariate_second(self) -> BivariateSecondParts<Self::Value> {
        let (value, gradient, hessian) = self.into_parts();
        let (axis0, axis1) = gradient.into_parts();
        let (axis0_axis0, axis0_axis1, axis1_axis1) = hessian.into_parts();

        BivariateSecondParts::new(value, axis0, axis1, axis0_axis0, axis0_axis1, axis1_axis1)
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

        assert_eq!(jet.into_value().into_inner(), 10);
    }

    #[test]
    fn jet1_into_value_discards_first_derivative() {
        let jet = Jet1::<_, ()>::from_parts(10, 20);

        assert_eq!(jet.into_value().into_inner(), 10);
    }

    #[test]
    fn jet1_into_first_extracts_value_and_derivative() {
        let jet = Jet1::<_, ()>::from_parts(10, 20);

        assert_eq!(jet.into_first().into_parts(), (10, 20),);
    }

    #[test]
    fn jet2_into_value_discards_both_derivatives() {
        let jet = Jet2::<_, ()>::from_parts(10, 20, 30);

        assert_eq!(jet.into_value().into_inner(), 10);
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
    fn bivariate_first_into_value_discards_bivariate_first() {
        let jet = JetBivariate1::<_, ()>::from_parts(10, JetGradient::new(20, 30));

        assert_eq!(jet.into_value().into_inner(), 10);
    }

    #[test]
    fn bivariate_first_into_bivariate_first_extracts_all_components() {
        let jet = JetBivariate1::<_, ()>::from_parts(10, JetGradient::new(20, 30));

        assert_eq!(jet.into_bivariate_first().into_parts(), (10, 20, 30),);
    }

    #[test]
    fn bivariate_second_into_value_discards_all_derivatives() {
        let jet = JetBivariate2::<_, ()>::from_parts(
            10,
            JetGradient::new(20, 30),
            JetHessian::new(40, 50, 60),
        );

        assert_eq!(jet.into_value().into_inner(), 10);
    }

    #[test]
    fn bivariate_second_into_bivariate_first_discards_bivariate_second() {
        let jet = JetBivariate2::<_, ()>::from_parts(
            10,
            JetGradient::new(20, 30),
            JetHessian::new(40, 50, 60),
        );

        assert_eq!(jet.into_bivariate_first().into_parts(), (10, 20, 30),);
    }

    #[test]
    fn bivariate_second_into_bivariate_second_extracts_all_components() {
        let jet = JetBivariate2::<_, ()>::from_parts(
            10,
            JetGradient::new(20, 30),
            JetHessian::new(40, 50, 60),
        );

        assert_eq!(
            jet.into_bivariate_second().into_parts(),
            (10, 20, 30, 40, 50, 60),
        );
    }
}

#[cfg(test)]
mod capability_tests {
    use super::*;

    fn assert_into_value<T: IntoValue>() {}
    fn assert_into_first<T: IntoFirst>() {}
    fn assert_into_second<T: IntoSecond>() {}
    fn assert_into_bivariate_first<T: IntoBivariateFirst>() {}
    fn assert_into_bivariate_second<T: IntoBivariateSecond>() {}

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
        assert_into_bivariate_first::<JetBivariate1<f64, ()>>();

        assert_into_value::<JetBivariate2<f64, ()>>();
        assert_into_bivariate_first::<JetBivariate2<f64, ()>>();
        assert_into_bivariate_second::<JetBivariate2<f64, ()>>();
    }
}
