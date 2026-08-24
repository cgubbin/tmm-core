//! Coordinate-free derivative parts.
//!
//! These containers hold the primal value and selected derivative components
//! after an internal algebraic quantity has been decomposed.
//!
//! They deliberately carry no physical parameter metadata. Bivariate
//! coordinates are therefore named `axis0` and `axis1`; their meanings are
//! attached only when the parts are crystallised into public differential
//! responses.
//!
//! Parts are assembled into public differential responses in a later stage,
//! where the derivative axes are associated with their physical parameters.

/// A value with all derivative storage removed.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ValuePart<T> {
    value: T,
}

impl<T> ValuePart<T> {
    /// Construct value-only part
    pub(crate) const fn new(value: T) -> Self {
        Self { value }
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
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectionalFirstParts<T> {
    value: T,
    first: T,
}

impl<T> DirectionalFirstParts<T> {
    /// Construct directional first-order parts.
    pub(crate) const fn new(value: T, first: T) -> Self {
        Self { value, first }
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
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectionalSecondParts<T> {
    value: T,
    first: T,
    second: T,
}

impl<T> DirectionalSecondParts<T> {
    /// Construct directional second-order parts.
    pub(crate) const fn new(value: T, first: T, second: T) -> Self {
        Self {
            value,
            first,
            second,
        }
    }

    /// Consume the container and return `(value, first, second)`.
    pub(crate) fn into_parts(self) -> (T, T, T) {
        (self.value, self.first, self.second)
    }
}

/// A value and first derivatives with respect to two coordinates.
///
/// The coordinate interpretation is supplied by the caller. The `axis0` and `axis1`
/// fields denote the first and second coordinates of the underlying
/// bivariate jet; they are not necessarily spatial coordinates.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BivariateFirstParts<T> {
    value: T,
    axis0: T,
    axis1: T,
}

impl<T> BivariateFirstParts<T> {
    /// Construct bivariate first-order parts.
    pub(crate) const fn new(value: T, axis0: T, axis1: T) -> Self {
        Self {
            value,
            axis0,
            axis1,
        }
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
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BivariateSecondParts<T> {
    value: T,

    axis0: T,
    axis1: T,

    axis0_axis0: T,
    axis0_axis1: T,
    axis1_axis1: T,
}

impl<T> BivariateSecondParts<T> {
    /// Construct bivariate second-order parts.
    #[allow(clippy::too_many_arguments)]
    pub(crate) const fn new(
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
