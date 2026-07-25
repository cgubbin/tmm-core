use crate::{SpatialProfile, SpatialProfileError};

use super::DirectionalCoordinate;

use ndarray::Dimension;

#[derive(Clone, Debug, PartialEq)]
pub struct DirectionalFirst<T> {
    coordinate: DirectionalCoordinate,
    first: T,
}

impl<T> DirectionalFirst<T> {
    pub(crate) fn new(coordinate: DirectionalCoordinate, first: T) -> Self {
        Self { coordinate, first }
    }

    pub fn coordinate(&self) -> DirectionalCoordinate {
        self.coordinate
    }

    pub fn first(&self) -> &T {
        &self.first
    }

    pub fn into_first(self) -> T {
        self.first
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> DirectionalFirst<U> {
        DirectionalFirst {
            coordinate: self.coordinate,
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
            coordinate: self.coordinate,
            first: self.first.spatial_profile(excitation_index)?,
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DirectionalSecond<T> {
    coordinate: DirectionalCoordinate,
    first: T,
    second: T,
}

impl<T> DirectionalSecond<T> {
    pub(crate) fn new(coordinate: DirectionalCoordinate, first: T, second: T) -> Self {
        Self {
            coordinate,
            first,
            second,
        }
    }

    pub fn coordinate(&self) -> DirectionalCoordinate {
        self.coordinate
    }
    pub fn first(&self) -> &T {
        &self.first
    }
    pub fn second(&self) -> &T {
        &self.second
    }

    pub fn into_first(self) -> DirectionalFirst<T> {
        DirectionalFirst {
            coordinate: self.coordinate,
            first: self.first,
        }
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> DirectionalSecond<U> {
        DirectionalSecond {
            coordinate: self.coordinate,
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
            coordinate: self.coordinate,
            first: self.first.spatial_profile(excitation_index)?,
            second: self.second.spatial_profile(excitation_index)?,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{EnergyDensity, differential::DirectionalSecond, field::ScalarField};

    use super::{DirectionalCoordinate, DirectionalFirst, SpatialProfile};

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

        let first = DirectionalFirst::new(DirectionalCoordinate::VacuumWavenumber, density);

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

        let derivatives =
            DirectionalSecond::new(DirectionalCoordinate::VacuumWavenumber, first, second);

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

        assert_eq!(
            profile.coordinate(),
            DirectionalCoordinate::VacuumWavenumber,
        );
    }
}
