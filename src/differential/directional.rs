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

use crate::{SpatialProfile, SpatialProfileError, parameter::Parameter};

use ndarray::Dimension;

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

impl<V, ED> SpatialProfile<ED> for DirectionalFirst<V>
where
    V: SpatialProfile<ED>,
    ED: Dimension,
{
    type Profile<'a>
        = DirectionalFirst<V::Profile<'a>>
    where
        Self: 'a;

    fn spatial_profile(
        &self,
        excitation_index: &ED,
    ) -> Result<Self::Profile<'_>, SpatialProfileError> {
        Ok(DirectionalFirst {
            parameter: self.parameter,
            first: self.first.spatial_profile(excitation_index)?,
        })
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

impl<V, ED> SpatialProfile<ED> for DirectionalSecond<V>
where
    V: SpatialProfile<ED>,
    ED: Dimension,
{
    type Profile<'a>
        = DirectionalSecond<V::Profile<'a>>
    where
        Self: 'a;

    fn spatial_profile(
        &self,
        excitation_index: &ED,
    ) -> Result<Self::Profile<'_>, SpatialProfileError> {
        Ok(DirectionalSecond {
            parameter: self.parameter,
            first: self.first.spatial_profile(excitation_index)?,
            second: self.second.spatial_profile(excitation_index)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{EnergyDensity, differential::DirectionalSecond, field::ScalarField};

    use super::{DirectionalFirst, Parameter, SpatialProfile};

    use ndarray::{Array2, Ix1, Ix2, arr1};

    #[test]
    fn first_derivatives_profile_every_derivative_branch() {
        let electric = ScalarField::new(Array2::from_shape_fn((2, 3), |(i, k)| {
            10.0 * i as f64 + k as f64
        }));

        let magnetic = ScalarField::new(Array2::from_shape_fn((2, 3), |(i, k)| {
            100.0 + 10.0 * i as f64 + k as f64
        }));

        let coupling = ScalarField::new(Array2::from_shape_fn((2, 3), |(i, k)| {
            100.0 + 10.0 * i as f64 + k as f64
        }));

        let total = ScalarField::new(Array2::from_shape_fn((2, 3), |(i, k)| {
            200.0 + 10.0 * i as f64 + k as f64
        }));

        let density = EnergyDensity::new(electric, magnetic, coupling, total);

        let first = DirectionalFirst::new(Parameter::Spectral, density);

        let profile = first
            .spatial_profile(&Ix1(1))
            .expect("profile should succeed");

        assert_eq!(
            profile.first().electric().values(),
            arr1(&[10.0, 11.0, 12.0]).view(),
        );
        assert_eq!(
            profile.first().magnetic().values(),
            arr1(&[110.0, 111.0, 112.0]).view(),
        );
        assert_eq!(
            profile.first().total().values(),
            arr1(&[210.0, 211.0, 212.0]).view(),
        );
    }

    #[test]
    fn directional_second_profiles_first_and_second_derivatives() {
        fn density(offset: f64) -> EnergyDensity<ScalarField<f64, Ix2>> {
            let electric = ScalarField::new(Array2::from_shape_fn((2, 3), |(i, k)| {
                offset + 10.0 * i as f64 + k as f64
            }));

            let magnetic = ScalarField::new(Array2::from_shape_fn((2, 3), |(i, k)| {
                offset + 100.0 + 10.0 * i as f64 + k as f64
            }));

            let coupling = ScalarField::new(Array2::from_shape_fn((2, 3), |(i, k)| {
                offset + 200.0 + 10.0 * i as f64 + k as f64
            }));

            let total = ScalarField::new(Array2::from_shape_fn((2, 3), |(i, k)| {
                offset + 300.0 + 10.0 * i as f64 + k as f64
            }));

            EnergyDensity::new(electric, magnetic, coupling, total)
        }

        let first = density(1_000.0);
        let second = density(2_000.0);

        let derivatives = DirectionalSecond::new(Parameter::Spectral, first, second);

        let profile = derivatives
            .spatial_profile(&Ix1(1))
            .expect("profile should succeed");

        assert_eq!(
            profile.first().electric().values(),
            arr1(&[1010.0, 1011.0, 1012.0]).view(),
        );
        assert_eq!(
            profile.first().magnetic().values(),
            arr1(&[1110.0, 1111.0, 1112.0]).view(),
        );
        assert_eq!(
            profile.first().coupling().values(),
            arr1(&[1210.0, 1211.0, 1212.0]).view(),
        );
        assert_eq!(
            profile.first().total().values(),
            arr1(&[1310.0, 1311.0, 1312.0]).view(),
        );

        assert_eq!(
            profile.second().electric().values(),
            arr1(&[2010.0, 2011.0, 2012.0]).view(),
        );
        assert_eq!(
            profile.second().magnetic().values(),
            arr1(&[2110.0, 2111.0, 2112.0]).view(),
        );
        assert_eq!(
            profile.second().coupling().values(),
            arr1(&[2210.0, 2211.0, 2212.0]).view(),
        );
        assert_eq!(
            profile.second().total().values(),
            arr1(&[2310.0, 2311.0, 2312.0]).view(),
        );

        assert_eq!(profile.parameter(), Parameter::Spectral,);
    }

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
