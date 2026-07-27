//! Public response types.
//!
//! A response contains a set of observable values together with an optional
//! differential representation. Domain-specific response types are aliases of
//! [`Response`] with an appropriate observable type.

use crate::differential::{
    BivariateFirst, BivariateHessian, BivariateSecond, DifferentialResponse, DirectionalCoordinate,
    DirectionalFirst, DirectionalSecond, NoDerivatives,
};

/// Observable values together with optional differential information and metadata required for
/// interpretation
///
/// `V` is the complete observable value type. The derivative representation
/// `D` determines which derivative accessors are available.
///
/// Domain-specific response aliases should generally be preferred in public
/// APIs. For example, an evaluator may return a `PlaneWaveResponse` rather
/// than spelling out `Response<PlaneWaveObservables<...>, ...>`.
#[derive(Clone, Debug, PartialEq)]
pub struct Response<V, D, M = NoMetadata> {
    inner: DifferentialResponse<V, D>,
    metadata: M,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoMetadata;

impl<V, D, M> Response<V, D, M> {
    /// Construct a response from observable values and derivative data.
    pub(crate) fn new(inner: DifferentialResponse<V, D>, metadata: M) -> Self {
        Self { inner, metadata }
    }

    /// Return the observable values.
    pub fn observables(&self) -> &V {
        self.inner.values()
    }

    /// Return the derivative values.
    pub fn derivatives(&self) -> &D {
        self.inner.derivatives()
    }

    /// Return the metadata
    pub fn metadata(&self) -> &M {
        &self.metadata
    }

    /// Consume the response and return its observable values, derivative
    /// data and metadata
    pub fn into_parts(self) -> (V, D, M) {
        let (values, derivatives) = self.inner.into_parts();
        (values, derivatives, self.metadata)
    }
}

impl<V, M> Response<V, NoDerivatives, M> {
    /// Consume the response and return its observable values.
    pub fn into_observables_and_metadata(self) -> (V, M) {
        let (observables, NoDerivatives) = self.inner.into_parts();
        (observables, self.metadata)
    }
}

impl<V, M> Response<V, DirectionalFirst<V>, M> {
    /// Return the coordinate with respect to which the derivative was taken.
    pub fn coordinate(&self) -> DirectionalCoordinate {
        self.inner.coordinate()
    }

    /// Return the first directional derivative of the observables.
    pub fn first(&self) -> &V {
        self.inner.first()
    }
}

impl<V, M> Response<V, DirectionalSecond<V>, M> {
    /// Return the coordinate with respect to which the derivatives were taken.
    pub fn coordinate(&self) -> DirectionalCoordinate {
        self.inner.coordinate()
    }

    /// Return the first directional derivative of the observables.
    pub fn first(&self) -> &V {
        self.inner.first()
    }

    /// Return the second directional derivative of the observables.
    pub fn second(&self) -> &V {
        self.inner.second()
    }
}

impl<V, M> Response<V, BivariateFirst<V>, M> {
    /// Return the spectral gradient of the observables.
    pub fn axis0(&self) -> &V {
        self.inner.axis0()
    }

    /// Return the spectral Hessian of the observables.
    pub fn axis1(&self) -> &V {
        self.inner.axis1()
    }
}

impl<V, M> Response<V, BivariateSecond<V>, M> {
    /// Return the spectral gradient of the observables.
    pub fn gradient(&self) -> &BivariateFirst<V> {
        self.inner.gradient()
    }

    /// Return the spectral Hessian of the observables.
    pub fn hessian(&self) -> &BivariateHessian<V> {
        self.inner.hessian()
    }
}

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[derive(Clone, Debug, PartialEq)]
//     struct TestObservables {
//         value: f64,
//     }

//     fn observable(value: f64) -> TestObservables {
//         TestObservables { value }
//     }

//     #[test]
//     fn exposes_observables() {
//         let response = Response::new(DifferentialResponse::new(observable(1.0), NoDerivatives));

//         assert_eq!(response.observables(), &observable(1.0));
//     }

//     #[test]
//     fn value_only_response_consumes_into_observables() {
//         let response = Response::new(DifferentialResponse::new(observable(1.0), NoDerivatives));

//         assert_eq!(response.into_observables(), observable(1.0));
//     }

//     #[test]
//     fn consumes_into_parts() {
//         let response = Response::new(DifferentialResponse::new(observable(1.0), NoDerivatives));

//         let (values, derivatives) = response.into_parts();

//         assert_eq!(values, observable(1.0));
//         assert_eq!(derivatives, NoDerivatives);
//     }

//     #[test]
//     fn directional_first_exposes_coordinate_and_derivative() {
//         let response = Response::new(DifferentialResponse::new(
//             observable(1.0),
//             DirectionalFirst::new(DirectionalCoordinate::VacuumWavenumber, observable(2.0)),
//         ));

//         assert_eq!(response.observables(), &observable(1.0));
//         assert_eq!(
//             response.coordinate(),
//             DirectionalCoordinate::VacuumWavenumber,
//         );
//         assert_eq!(response.first(), &observable(2.0));
//     }

//     #[test]
//     fn directional_second_exposes_both_derivatives() {
//         let response = Response::new(DifferentialResponse::new(
//             observable(1.0),
//             DirectionalSecond::new(
//                 DirectionalCoordinate::ParallelWavenumber,
//                 observable(2.0),
//                 observable(3.0),
//             ),
//         ));

//         assert_eq!(response.observables(), &observable(1.0));
//         assert_eq!(
//             response.coordinate(),
//             DirectionalCoordinate::ParallelWavenumber,
//         );
//         assert_eq!(response.first(), &observable(2.0));
//         assert_eq!(response.second(), &observable(3.0));
//     }

//     #[test]
//     fn response_supports_non_clone_observables() {
//         #[derive(Debug, PartialEq)]
//         struct NonClone(String);

//         let response = Response::new(DifferentialResponse::new(
//             NonClone("values".into()),
//             NoDerivatives,
//         ));

//         assert_eq!(response.into_observables(), NonClone("values".into()),);
//     }
// }
