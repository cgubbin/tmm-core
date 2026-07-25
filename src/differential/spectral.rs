use crate::{SpatialProfile, SpatialProfileError};

use ndarray::Dimension;

#[derive(Clone, Debug, PartialEq)]
pub struct SpectralSecond<T> {
    gradient: SpectralGradient<T>,
    hessian: SpectralHessian<T>,
}

impl<V, ED> SpatialProfile<ED> for SpectralSecond<V>
where
    V: SpatialProfile<ED>,
    ED: Dimension,
{
    type Profile<'a>
        = SpectralSecond<V::Profile<'a>>
    where
        Self: 'a;

    fn spatial_profile(
        &self,
        excitation_index: &ED,
    ) -> Result<Self::Profile<'_>, SpatialProfileError> {
        Ok(SpectralSecond {
            gradient: self.gradient.spatial_profile(excitation_index)?,
            hessian: self.hessian.spatial_profile(excitation_index)?,
        })
    }
}

impl<T> SpectralSecond<T> {
    pub(crate) fn new(gradient: SpectralGradient<T>, hessian: SpectralHessian<T>) -> Self {
        Self { gradient, hessian }
    }

    pub fn gradient(&self) -> &SpectralGradient<T> {
        &self.gradient
    }

    pub fn hessian(&self) -> &SpectralHessian<T> {
        &self.hessian
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> SpectralSecond<U> {
        SpectralSecond {
            gradient: self.gradient.map(&mut f),
            hessian: self.hessian.map(f),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpectralGradient<T> {
    vacuum_wavenumber: T,
    parallel_wavenumber: T,
}

impl<V, ED> SpatialProfile<ED> for SpectralGradient<V>
where
    V: SpatialProfile<ED>,
    ED: Dimension,
{
    type Profile<'a>
        = SpectralGradient<V::Profile<'a>>
    where
        Self: 'a;

    fn spatial_profile(
        &self,
        excitation_index: &ED,
    ) -> Result<Self::Profile<'_>, SpatialProfileError> {
        Ok(SpectralGradient {
            vacuum_wavenumber: self.vacuum_wavenumber.spatial_profile(excitation_index)?,
            parallel_wavenumber: self.parallel_wavenumber.spatial_profile(excitation_index)?,
        })
    }
}

impl<T> SpectralGradient<T> {
    pub(crate) fn new(vacuum_wavenumber: T, parallel_wavenumber: T) -> Self {
        Self {
            vacuum_wavenumber,
            parallel_wavenumber,
        }
    }

    pub fn vacuum_wavenumber(&self) -> &T {
        &self.vacuum_wavenumber
    }

    pub fn parallel_wavenumber(&self) -> &T {
        &self.parallel_wavenumber
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> SpectralGradient<U> {
        SpectralGradient {
            vacuum_wavenumber: f(self.vacuum_wavenumber),
            parallel_wavenumber: f(self.parallel_wavenumber),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpectralHessian<T> {
    vacuum_wavenumber_vacuum_wavenumber: T,
    vacuum_wavenumber_parallel_wavenumber: T,
    parallel_wavenumber_parallel_wavenumber: T,
}

impl<V, ED> SpatialProfile<ED> for SpectralHessian<V>
where
    V: SpatialProfile<ED>,
    ED: Dimension,
{
    type Profile<'a>
        = SpectralHessian<V::Profile<'a>>
    where
        Self: 'a;

    fn spatial_profile(
        &self,
        excitation_index: &ED,
    ) -> Result<Self::Profile<'_>, SpatialProfileError> {
        Ok(SpectralHessian {
            vacuum_wavenumber_vacuum_wavenumber: self
                .vacuum_wavenumber_vacuum_wavenumber
                .spatial_profile(excitation_index)?,
            vacuum_wavenumber_parallel_wavenumber: self
                .vacuum_wavenumber_parallel_wavenumber
                .spatial_profile(excitation_index)?,
            parallel_wavenumber_parallel_wavenumber: self
                .parallel_wavenumber_parallel_wavenumber
                .spatial_profile(excitation_index)?,
        })
    }
}

impl<T> SpectralHessian<T> {
    pub(crate) fn new(
        vacuum_wavenumber_vacuum_wavenumber: T,
        vacuum_wavenumber_parallel_wavenumber: T,
        parallel_wavenumber_parallel_wavenumber: T,
    ) -> Self {
        Self {
            vacuum_wavenumber_vacuum_wavenumber,
            vacuum_wavenumber_parallel_wavenumber,
            parallel_wavenumber_parallel_wavenumber,
        }
    }

    pub fn vacuum_wavenumber_vacuum_wavenumber(&self) -> &T {
        &self.vacuum_wavenumber_vacuum_wavenumber
    }

    pub fn vacuum_wavenumber_parallel_wavenumber(&self) -> &T {
        &self.vacuum_wavenumber_parallel_wavenumber
    }

    pub fn parallel_wavenumber_parallel_wavenumber(&self) -> &T {
        &self.parallel_wavenumber_parallel_wavenumber
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> SpectralHessian<U> {
        SpectralHessian {
            vacuum_wavenumber_vacuum_wavenumber: f(self.vacuum_wavenumber_vacuum_wavenumber),
            vacuum_wavenumber_parallel_wavenumber: f(self.vacuum_wavenumber_parallel_wavenumber),
            parallel_wavenumber_parallel_wavenumber: f(self.parallel_wavenumber_parallel_wavenumber),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{EnergyDensity, field::ScalarField};

    use super::*;

    use ndarray::{Array2, Ix1, Ix2, arr1};

    #[test]
    fn spectral_second_profiles_gradient_and_hessian_branches() {
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

        let gradient = SpectralGradient::new(
            density(1_000.0), // dx
            density(2_000.0), // dy
        );

        let hessian = SpectralHessian::new(
            density(3_000.0), // dxdx
            density(4_000.0), // dxdy
            density(5_000.0), // dydy
        );

        let derivatives = SpectralSecond::new(gradient, hessian);

        let profile = derivatives
            .spatial_profile(&Ix1(1))
            .expect("profile should succeed");

        // Gradient: dx.
        assert_eq!(
            profile.gradient().vacuum_wavenumber().electric().values(),
            arr1(&[1010.0, 1011.0, 1012.0]).view(),
        );
        assert_eq!(
            profile.gradient().vacuum_wavenumber().magnetic().values(),
            arr1(&[1110.0, 1111.0, 1112.0]).view(),
        );
        assert_eq!(
            profile.gradient().vacuum_wavenumber().coupling().values(),
            arr1(&[1210.0, 1211.0, 1212.0]).view(),
        );
        assert_eq!(
            profile.gradient().vacuum_wavenumber().total().values(),
            arr1(&[1310.0, 1311.0, 1312.0]).view(),
        );

        // Gradient: dy.
        assert_eq!(
            profile.gradient().parallel_wavenumber().electric().values(),
            arr1(&[2010.0, 2011.0, 2012.0]).view(),
        );
        assert_eq!(
            profile.gradient().parallel_wavenumber().magnetic().values(),
            arr1(&[2110.0, 2111.0, 2112.0]).view(),
        );
        assert_eq!(
            profile.gradient().parallel_wavenumber().coupling().values(),
            arr1(&[2210.0, 2211.0, 2212.0]).view(),
        );
        assert_eq!(
            profile.gradient().parallel_wavenumber().total().values(),
            arr1(&[2310.0, 2311.0, 2312.0]).view(),
        );

        // Hessian: d²/dx².
        assert_eq!(
            profile
                .hessian()
                .vacuum_wavenumber_vacuum_wavenumber()
                .electric()
                .values(),
            arr1(&[3010.0, 3011.0, 3012.0]).view(),
        );
        assert_eq!(
            profile
                .hessian()
                .vacuum_wavenumber_vacuum_wavenumber()
                .magnetic()
                .values(),
            arr1(&[3110.0, 3111.0, 3112.0]).view(),
        );
        assert_eq!(
            profile
                .hessian()
                .vacuum_wavenumber_vacuum_wavenumber()
                .coupling()
                .values(),
            arr1(&[3210.0, 3211.0, 3212.0]).view(),
        );
        assert_eq!(
            profile
                .hessian()
                .vacuum_wavenumber_vacuum_wavenumber()
                .total()
                .values(),
            arr1(&[3310.0, 3311.0, 3312.0]).view(),
        );

        // Hessian: d²/dxdy.
        assert_eq!(
            profile
                .hessian()
                .vacuum_wavenumber_parallel_wavenumber()
                .electric()
                .values(),
            arr1(&[4010.0, 4011.0, 4012.0]).view(),
        );
        assert_eq!(
            profile
                .hessian()
                .vacuum_wavenumber_parallel_wavenumber()
                .magnetic()
                .values(),
            arr1(&[4110.0, 4111.0, 4112.0]).view(),
        );
        assert_eq!(
            profile
                .hessian()
                .vacuum_wavenumber_parallel_wavenumber()
                .coupling()
                .values(),
            arr1(&[4210.0, 4211.0, 4212.0]).view(),
        );
        assert_eq!(
            profile
                .hessian()
                .vacuum_wavenumber_parallel_wavenumber()
                .total()
                .values(),
            arr1(&[4310.0, 4311.0, 4312.0]).view(),
        );

        // Hessian: d²/dy².
        assert_eq!(
            profile
                .hessian()
                .parallel_wavenumber_parallel_wavenumber()
                .electric()
                .values(),
            arr1(&[5010.0, 5011.0, 5012.0]).view(),
        );
        assert_eq!(
            profile
                .hessian()
                .parallel_wavenumber_parallel_wavenumber()
                .magnetic()
                .values(),
            arr1(&[5110.0, 5111.0, 5112.0]).view(),
        );
        assert_eq!(
            profile
                .hessian()
                .parallel_wavenumber_parallel_wavenumber()
                .coupling()
                .values(),
            arr1(&[5210.0, 5211.0, 5212.0]).view(),
        );
        assert_eq!(
            profile
                .hessian()
                .parallel_wavenumber_parallel_wavenumber()
                .total()
                .values(),
            arr1(&[5310.0, 5311.0, 5312.0]).view(),
        );
    }
}
