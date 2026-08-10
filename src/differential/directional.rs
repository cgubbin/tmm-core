//! Directional differential representations.
//!
//! Directional derivatives are taken with respect to one caller-facing
//! [`Parameter`].
//!
//! [`DirectionalFirst`] stores one first derivative, while
//! [`DirectionalSecond`] stores both the first derivative and the repeated
//! second derivative with respect to the same parameter.
//!
//! These types contain caller-facing parameter metadata and are therefore
//! assembled only after internal jet-valued quantities have been decomposed
//! into coordinate-free derivative parts.
//!
//! Both representations support:
//!
//! - access to the differentiated parameter;
//! - transformation of the stored derivative values through `map`;
//! - extraction of spatial profiles when the stored value implements
//!   [`SpatialProfile`].

use crate::parameter::Parameter;

/// A first derivative with respect to one caller-facing parameter.
///
/// `first` is the derivative of the associated response value with respect to
/// [`Self::parameter`].
#[derive(Clone, Debug, PartialEq)]
pub struct DirectionalFirst<T> {
    parameter: Parameter,
    first: T,
}

impl<T> DirectionalFirst<T> {
    /// Construct a directional first-derivative representation.
    pub(crate) fn new(parameter: Parameter, first: T) -> Self {
        Self { parameter, first }
    }

    /// Return the caller-facing parameter with respect to which the derivative
    /// was taken.
    pub fn parameter(&self) -> Parameter {
        self.parameter
    }

    /// Return the first derivative.
    pub fn first(&self) -> &T {
        &self.first
    }

    /// Consume the representation and return the first derivative.
    pub fn into_first(self) -> T {
        self.first
    }

    /// Transform the stored derivative while preserving its parameter.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> DirectionalFirst<U> {
        DirectionalFirst {
            parameter: self.parameter,
            first: f(self.first),
        }
    }
}

/// First and repeated second derivatives with respect to one caller-facing
/// parameter.
///
/// Both derivatives refer to the same [`Self::parameter`].
#[derive(Clone, Debug, PartialEq)]
pub struct DirectionalSecond<T> {
    parameter: Parameter,
    first: T,
    second: T,
}

impl<T> DirectionalSecond<T> {
    /// Construct a directional second-order representation.
    pub(crate) fn new(parameter: Parameter, first: T, second: T) -> Self {
        Self {
            parameter,
            first,
            second,
        }
    }

    /// Return the caller-facing parameter with respect to which both
    /// derivatives were taken.
    pub fn parameter(&self) -> Parameter {
        self.parameter
    }

    /// Return the first derivative.
    pub fn first(&self) -> &T {
        &self.first
    }

    /// Return the repeated second derivative.
    pub fn second(&self) -> &T {
        &self.second
    }

    /// Consume the representation while retaining only its first derivative.
    pub fn into_first(self) -> DirectionalFirst<T> {
        DirectionalFirst {
            parameter: self.parameter,
            first: self.first,
        }
    }

    /// Transform both derivative components while preserving their parameter.
    ///
    /// `f` is called first for the first derivative and then for the second
    /// derivative.
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> DirectionalSecond<U> {
        DirectionalSecond {
            parameter: self.parameter,
            first: f(self.first),
            second: f(self.second),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::differential::DirectionalSecond;

    use super::{DirectionalFirst, Parameter};

    #[test]
    fn directional_first_preserves_parameter_and_derivative() {
        let derivatives = DirectionalFirst::new(Parameter::Spectral, 12);

        assert_eq!(derivatives.parameter(), Parameter::Spectral);
        assert_eq!(derivatives.first(), &12);
        assert_eq!(derivatives.into_first(), 12);
    }

    #[test]
    fn directional_second_into_first_discards_only_second_derivative() {
        let derivatives = DirectionalSecond::new(Parameter::InPlane, 12, 34);

        let first = derivatives.into_first();

        assert_eq!(first.parameter(), Parameter::InPlane);
        assert_eq!(first.first(), &12);
    }

    #[test]
    fn directional_second_map_transforms_both_derivatives() {
        let derivatives = DirectionalSecond::new(Parameter::Spectral, 2, 3);

        let mapped = derivatives.map(|value| value * 10);

        assert_eq!(mapped.parameter(), Parameter::Spectral);
        assert_eq!(mapped.first(), &20);
        assert_eq!(mapped.second(), &30);
    }
}
