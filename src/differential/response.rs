use crate::{
    SpatialProfile, SpatialProfileError,
    differential::{
        BivariateFirst, BivariateHessian, BivariateSecond, DirectionalCoordinate, DirectionalFirst,
        DirectionalSecond,
    },
};

use ndarray::Dimension;

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

/// Sampled plane-wave observables with optional differential information.
///
/// The derivative representation `D` determines which derivative accessors
/// are available:
///
/// - [`NoDerivatives`] stores values only;
/// - [`DirectionalFirst`] stores a first derivative along one coordinate;
/// - [`DirectionalSecond`] stores first and second derivatives along one
///   coordinate;
/// - [`SpectralSecond`] stores a gradient and Hessian over the supported
///   spectral coordinates.
#[derive(Clone, Debug, PartialEq)]
pub struct DifferentialResponse<V, D = NoDerivatives> {
    values: V,
    derivatives: D,
}

impl<V, D> DifferentialResponse<V, D> {
    pub(crate) fn new(values: V, derivatives: D) -> Self {
        Self {
            values,
            derivatives,
        }
    }

    pub fn values(&self) -> &V {
        &self.values
    }

    pub fn derivatives(&self) -> &D {
        &self.derivatives
    }

    pub fn into_parts(self) -> (V, D) {
        (self.values, self.derivatives)
    }

    pub fn map_values<U>(self, f: impl FnOnce(V) -> U) -> DifferentialResponse<U, D> {
        DifferentialResponse {
            values: f(self.values),
            derivatives: self.derivatives,
        }
    }

    pub fn map_derivatives<E>(self, f: impl FnOnce(D) -> E) -> DifferentialResponse<V, E> {
        DifferentialResponse {
            values: self.values,
            derivatives: f(self.derivatives),
        }
    }
}

impl<V, X> DifferentialResponse<V, DirectionalFirst<X>> {
    pub(crate) fn coordinate(&self) -> DirectionalCoordinate {
        self.derivatives.coordinate()
    }

    pub(crate) fn first(&self) -> &X {
        self.derivatives.first()
    }
}

impl<V, X> DifferentialResponse<V, DirectionalSecond<X>> {
    pub(crate) fn coordinate(&self) -> DirectionalCoordinate {
        self.derivatives.coordinate()
    }

    pub(crate) fn first(&self) -> &X {
        self.derivatives.first()
    }

    pub(crate) fn second(&self) -> &X {
        self.derivatives.second()
    }
}

impl<V, X> DifferentialResponse<V, BivariateFirst<X>> {
    pub(crate) fn axis0(&self) -> &X {
        self.derivatives.axis0()
    }

    pub(crate) fn axis1(&self) -> &X {
        self.derivatives.axis1()
    }
}

impl<V, X> DifferentialResponse<V, BivariateSecond<X>> {
    pub(crate) fn gradient(&self) -> &BivariateFirst<X> {
        self.derivatives.first()
    }

    pub(crate) fn hessian(&self) -> &BivariateHessian<X> {
        self.derivatives.second()
    }
}

#[cfg(test)]
mod tests {
    use super::{NoDerivatives, SpatialProfile};
    use ndarray::Ix2;

    #[test]
    fn no_derivatives_profiles_to_no_derivatives() {
        let derivatives = NoDerivatives;

        let profile = derivatives
            .spatial_profile(&Ix2(1, 2))
            .expect("NoDerivatives profiling should be infallible");

        assert_eq!(profile, NoDerivatives);
    }
}
