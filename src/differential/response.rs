//! Differential responses.
//!
//! [`DifferentialResponse`] pairs a sampled or structured value with a
//! derivative representation.
//!
//! The derivative type determines which information is available:
//!
//! - [`NoDerivatives`] represents a value-only response;
//! - [`DirectionalFirst`] stores one first derivative;
//! - [`DirectionalSecond`] stores first and repeated second derivatives;
//! - [`BivariateFirst`] stores first derivatives with respect to two ordered
//!   parameters;
//! - [`BivariateSecond`] stores the corresponding gradient and symmetric
//!   Hessian.
//!
//! Responses are constructed internally after derivative parts have been
//! extracted from backend quantities and assembled with typed parameter
//! mappings.

use crate::{
    SpatialProfile, SpatialProfileError,
    differential::{
        BivariateFirst, BivariateGradient, BivariateHessian, BivariateSecond, DirectionalFirst,
        DirectionalSecond,
    },
    parameter::Parameter,
};

use ndarray::Dimension;

/// A value paired with caller-facing differential information.
///
/// `V` is the value representation and `D` determines the available derivative
/// data. The default derivative type, [`NoDerivatives`], represents a
/// value-only response.
#[derive(Clone, Debug, PartialEq)]
pub struct DifferentialResponse<V, D = NoDerivatives> {
    value: V,
    derivatives: D,
}

impl<V, D> DifferentialResponse<V, D> {
    /// Construct a differential response.
    pub(crate) fn new(value: V, derivatives: D) -> Self {
        Self { value, derivatives }
    }

    /// Return the response value.
    pub fn value(&self) -> &V {
        &self.value
    }

    /// Return the derivative representation.
    pub fn derivatives(&self) -> &D {
        &self.derivatives
    }

    /// Consume the response and return `(value, derivatives)`.
    pub fn into_parts(self) -> (V, D) {
        (self.value, self.derivatives)
    }

    /// Transform the response value while preserving its derivatives.
    pub fn map_value<U>(self, f: impl FnOnce(V) -> U) -> DifferentialResponse<U, D> {
        DifferentialResponse {
            value: f(self.value),
            derivatives: self.derivatives,
        }
    }

    /// Transform the derivative representation while preserving the value.
    pub fn map_derivatives<E>(self, f: impl FnOnce(D) -> E) -> DifferentialResponse<V, E> {
        DifferentialResponse {
            value: self.value,
            derivatives: f(self.derivatives),
        }
    }
}

impl<V, X> DifferentialResponse<V, DirectionalFirst<X>> {
    /// Return the differentiated parameter.
    pub fn parameter(&self) -> Parameter {
        self.derivatives.parameter()
    }

    /// Return the first derivative.
    pub fn first(&self) -> &X {
        self.derivatives.first()
    }
}

impl<V, X> DifferentialResponse<V, DirectionalSecond<X>> {
    /// Return the differentiated parameter.
    pub fn parameter(&self) -> Parameter {
        self.derivatives.parameter()
    }

    /// Return the first derivative.
    pub fn first(&self) -> &X {
        self.derivatives.first()
    }

    /// Return the repeated second derivative.
    pub fn second(&self) -> &X {
        self.derivatives.second()
    }
}

impl<V, X> DifferentialResponse<V, BivariateFirst<X>> {
    /// Return the ordered derivative parameters.
    pub fn parameters(&self) -> [Parameter; 2] {
        self.derivatives.parameters()
    }

    /// Return the derivative along axis zero.
    pub fn axis0(&self) -> &X {
        self.derivatives.axis0()
    }

    /// Return the derivative along axis one.
    pub fn axis1(&self) -> &X {
        self.derivatives.axis1()
    }
}

impl<V, X> DifferentialResponse<V, BivariateSecond<X>> {
    /// Return the ordered derivative parameters.
    pub fn parameters(&self) -> [Parameter; 2] {
        self.derivatives.parameters()
    }

    /// Return the gradient.
    pub fn gradient(&self) -> &BivariateGradient<X> {
        self.derivatives.gradient()
    }

    /// Return the symmetric Hessian.
    pub fn hessian(&self) -> &BivariateHessian<X> {
        self.derivatives.hessian()
    }
}

/// Marker indicating that a response contains no derivative information.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct NoDerivatives;

impl<ED> SpatialProfile<ED> for NoDerivatives
where
    ED: Dimension,
{
    type Profile<'a>
        = NoDerivatives
    where
        Self: 'a;

    fn spatial_profile(
        &self,
        _excitation_index: &ED,
    ) -> Result<Self::Profile<'_>, SpatialProfileError> {
        Ok(NoDerivatives)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::Ix2;

    #[test]
    fn no_derivatives_profiles_to_no_derivatives() {
        let derivatives = NoDerivatives;

        let profile = derivatives
            .spatial_profile(&Ix2(1, 2))
            .expect("NoDerivatives profiling should be infallible");

        assert_eq!(profile, NoDerivatives);
    }

    #[test]
    fn response_preserves_value_and_derivatives() {
        let response =
            DifferentialResponse::new(10, DirectionalFirst::new(Parameter::Spectral, 20));

        assert_eq!(response.value(), &10);
        assert_eq!(response.parameter(), Parameter::Spectral);
        assert_eq!(response.first(), &20);
    }

    #[test]
    fn into_parts_returns_value_and_derivatives() {
        let response = DifferentialResponse::new(10, NoDerivatives);

        assert_eq!(response.into_parts(), (10, NoDerivatives),);
    }

    #[test]
    fn map_value_preserves_derivatives() {
        let response =
            DifferentialResponse::new(10, DirectionalFirst::new(Parameter::Spectral, 20));

        let mapped = response.map_value(|value| value.to_string());

        assert_eq!(mapped.value(), "10");
        assert_eq!(mapped.parameter(), Parameter::Spectral);
        assert_eq!(mapped.first(), &20);
    }

    #[test]
    fn map_derivatives_preserves_value() {
        let response =
            DifferentialResponse::new(10, DirectionalFirst::new(Parameter::Spectral, 20));

        let mapped = response.map_derivatives(DirectionalFirst::into_first);

        assert_eq!(mapped.value(), &10);
        assert_eq!(mapped.derivatives(), &20);
    }
}
