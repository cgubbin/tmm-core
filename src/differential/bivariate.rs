//! Bivariate differential representations.
//!
//! Bivariate derivatives are taken with respect to two ordered caller-facing
//! parameters.
//!
//! [`BivariateFirst`] stores the two first derivatives directly.
//! [`BivariateSecond`] stores a [`BivariateGradient`] and a symmetric
//! [`BivariateHessian`].
//!
//! Axis zero corresponds to `parameters()[0]`, and axis one corresponds to
//! `parameters()[1]`. This ordering is significant and is preserved throughout
//! response assembly, mapping, and spatial-profile extraction.
//!
//! The axis names do not imply spatial coordinates. Either axis may represent
//! any supported [`Parameter`], including a spectral coordinate, an in-plane
//! coordinate, or a finite-layer thickness.

use crate::{SpatialProfile, SpatialProfileError, parameter::Parameter};

use ndarray::Dimension;

/// First and second derivatives with respect to two ordered caller-facing
/// parameters.
///
/// The gradient and Hessian share the parameter ordering returned by
/// [`Self::parameters`].
#[derive(Clone, Debug, PartialEq)]
pub struct BivariateSecond<T> {
    parameters: [Parameter; 2],
    gradient: BivariateGradient<T>,
    hessian: BivariateHessian<T>,
}

impl<T> BivariateSecond<T> {
    /// Construct a bivariate second-order representation.
    pub(crate) fn new(
        parameters: [Parameter; 2],
        gradient: BivariateGradient<T>,
        hessian: BivariateHessian<T>,
    ) -> Self {
        Self {
            parameters,
            gradient,
            hessian,
        }
    }

    /// Return the ordered derivative parameters.
    pub fn parameters(&self) -> [Parameter; 2] {
        self.parameters
    }

    /// Return the gradient.
    pub fn gradient(&self) -> &BivariateGradient<T> {
        &self.gradient
    }

    /// Return the symmetric Hessian.
    pub fn hessian(&self) -> &BivariateHessian<T> {
        &self.hessian
    }

    /// Return the gradient.
    ///
    /// This is an alias for [`Self::gradient`].
    pub fn first(&self) -> &BivariateGradient<T> {
        self.gradient()
    }

    /// Return the Hessian.
    ///
    /// This is an alias for [`Self::hessian`].
    pub fn second(&self) -> &BivariateHessian<T> {
        self.hessian()
    }

    /// Transform every gradient and Hessian component while preserving the
    /// parameter ordering.
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> BivariateSecond<U> {
        BivariateSecond {
            parameters: self.parameters,
            gradient: self.gradient.map(&mut f),
            hessian: self.hessian.map(f),
        }
    }

    pub fn into_parts(self) -> ([Parameter; 2], BivariateGradient<T>, BivariateHessian<T>) {
        (self.parameters, self.gradient, self.hessian)
    }
}

impl<V, ED> SpatialProfile<ED> for BivariateSecond<V>
where
    V: SpatialProfile<ED>,
    ED: Dimension,
{
    type Profile<'a>
        = BivariateSecond<V::Profile<'a>>
    where
        Self: 'a;

    fn spatial_profile(
        &self,
        excitation_index: &ED,
    ) -> Result<Self::Profile<'_>, SpatialProfileError> {
        Ok(BivariateSecond {
            parameters: self.parameters,
            gradient: self.gradient.spatial_profile(excitation_index)?,
            hessian: self.hessian.spatial_profile(excitation_index)?,
        })
    }
}

/// First derivatives with respect to two ordered caller-facing parameters.
///
/// `axis0` is the derivative with respect to `parameters()[0]`, while `axis1`
/// is the derivative with respect to `parameters()[1]`.
#[derive(Clone, Debug, PartialEq)]
pub struct BivariateFirst<T> {
    parameters: [Parameter; 2],
    axis0: T,
    axis1: T,
}

impl<T> BivariateFirst<T> {
    /// Construct a bivariate first-order representation.
    pub(crate) fn new(parameters: [Parameter; 2], axis0: T, axis1: T) -> Self {
        Self {
            parameters,
            axis0,
            axis1,
        }
    }

    /// Return the ordered derivative parameters.
    pub fn parameters(&self) -> [Parameter; 2] {
        self.parameters
    }

    /// Return the derivative with respect to `parameters()[0]`.
    pub fn axis0(&self) -> &T {
        &self.axis0
    }

    /// Return the derivative with respect to `parameters()[1]`.
    pub fn axis1(&self) -> &T {
        &self.axis1
    }

    /// Transform both derivative components while preserving parameter order.
    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> BivariateFirst<U> {
        BivariateFirst {
            parameters: self.parameters,
            axis0: f(self.axis0),
            axis1: f(self.axis1),
        }
    }

    /// Consume the representation and return
    /// `(parameters, axis0, axis1)`.
    pub fn into_parts(self) -> ([Parameter; 2], T, T) {
        (self.parameters, self.axis0, self.axis1)
    }
}

impl<V, ED> SpatialProfile<ED> for BivariateFirst<V>
where
    V: SpatialProfile<ED>,
    ED: Dimension,
{
    type Profile<'a>
        = BivariateFirst<V::Profile<'a>>
    where
        Self: 'a;

    fn spatial_profile(
        &self,
        excitation_index: &ED,
    ) -> Result<Self::Profile<'_>, SpatialProfileError> {
        Ok(BivariateFirst {
            parameters: self.parameters,
            axis0: self.axis0.spatial_profile(excitation_index)?,
            axis1: self.axis1.spatial_profile(excitation_index)?,
        })
    }
}

/// A coordinate-free two-component gradient.
///
/// Parameter metadata is stored by the enclosing [`BivariateSecond`].
#[derive(Clone, Debug, PartialEq)]
pub struct BivariateGradient<T> {
    axis0: T,
    axis1: T,
}

impl<T> BivariateGradient<T> {
    pub(crate) fn new(axis0: T, axis1: T) -> Self {
        Self { axis0, axis1 }
    }

    pub fn axis0(&self) -> &T {
        &self.axis0
    }

    pub fn axis1(&self) -> &T {
        &self.axis1
    }

    pub fn into_parts(self) -> (T, T) {
        (self.axis0, self.axis1)
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> BivariateGradient<U> {
        BivariateGradient {
            axis0: f(self.axis0),
            axis1: f(self.axis1),
        }
    }
}

impl<V, ED> SpatialProfile<ED> for BivariateGradient<V>
where
    V: SpatialProfile<ED>,
    ED: Dimension,
{
    type Profile<'a>
        = BivariateGradient<V::Profile<'a>>
    where
        Self: 'a;

    fn spatial_profile(
        &self,
        excitation_index: &ED,
    ) -> Result<Self::Profile<'_>, SpatialProfileError> {
        Ok(BivariateGradient {
            axis0: self.axis0.spatial_profile(excitation_index)?,
            axis1: self.axis1.spatial_profile(excitation_index)?,
        })
    }
}

/// A coordinate-free symmetric Hessian over two derivative axes.
///
/// Only one mixed derivative is stored. The representation assumes symmetry:
///
/// ```text
/// [ axis0_axis0  axis0_axis1 ]
/// [ axis0_axis1  axis1_axis1 ]
/// ```
///
/// Parameter metadata is stored by the enclosing [`BivariateSecond`].
#[derive(Clone, Debug, PartialEq)]
pub struct BivariateHessian<T> {
    axis0_axis0: T,
    axis0_axis1: T,
    axis1_axis1: T,
}

impl<T> BivariateHessian<T> {
    pub(crate) fn new(axis0_axis0: T, axis0_axis1: T, axis1_axis1: T) -> Self {
        Self {
            axis0_axis0,
            axis0_axis1,
            axis1_axis1,
        }
    }

    pub fn axis0_axis0(&self) -> &T {
        &self.axis0_axis0
    }

    pub fn axis0_axis1(&self) -> &T {
        &self.axis0_axis1
    }

    pub fn axis1_axis1(&self) -> &T {
        &self.axis1_axis1
    }

    pub fn into_parts(self) -> (T, T, T) {
        (self.axis0_axis0, self.axis0_axis1, self.axis1_axis1)
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> BivariateHessian<U> {
        BivariateHessian {
            axis0_axis0: f(self.axis0_axis0),
            axis0_axis1: f(self.axis0_axis1),
            axis1_axis1: f(self.axis1_axis1),
        }
    }
}

impl<V, ED> SpatialProfile<ED> for BivariateHessian<V>
where
    V: SpatialProfile<ED>,
    ED: Dimension,
{
    type Profile<'a>
        = BivariateHessian<V::Profile<'a>>
    where
        Self: 'a;

    fn spatial_profile(
        &self,
        excitation_index: &ED,
    ) -> Result<Self::Profile<'_>, SpatialProfileError> {
        Ok(BivariateHessian {
            axis0_axis0: self.axis0_axis0.spatial_profile(excitation_index)?,
            axis0_axis1: self.axis0_axis1.spatial_profile(excitation_index)?,
            axis1_axis1: self.axis1_axis1.spatial_profile(excitation_index)?,
        })
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

        let gradient = BivariateGradient::new(
            density(1_000.0), // dx
            density(2_000.0), // dy
        );

        let hessian = BivariateHessian::new(
            density(3_000.0), // dxdx
            density(4_000.0), // dxdy
            density(5_000.0), // dydy
        );

        let derivatives =
            BivariateSecond::new([Parameter::Spectral, Parameter::InPlane], gradient, hessian);

        let profile = derivatives
            .spatial_profile(&Ix1(1))
            .expect("profile should succeed");

        // Gradient: daxis0.
        assert_eq!(
            profile.gradient().axis0().electric().values(),
            arr1(&[1010.0, 1011.0, 1012.0]).view(),
        );
        assert_eq!(
            profile.gradient().axis0().magnetic().values(),
            arr1(&[1110.0, 1111.0, 1112.0]).view(),
        );
        assert_eq!(
            profile.gradient().axis0().coupling().values(),
            arr1(&[1210.0, 1211.0, 1212.0]).view(),
        );
        assert_eq!(
            profile.gradient().axis0().total().values(),
            arr1(&[1310.0, 1311.0, 1312.0]).view(),
        );

        // Gradient: dy.
        assert_eq!(
            profile.gradient().axis1().electric().values(),
            arr1(&[2010.0, 2011.0, 2012.0]).view(),
        );
        assert_eq!(
            profile.gradient().axis1().magnetic().values(),
            arr1(&[2110.0, 2111.0, 2112.0]).view(),
        );
        assert_eq!(
            profile.gradient().axis1().coupling().values(),
            arr1(&[2210.0, 2211.0, 2212.0]).view(),
        );
        assert_eq!(
            profile.gradient().axis1().total().values(),
            arr1(&[2310.0, 2311.0, 2312.0]).view(),
        );

        // Hessian: d²/daxis0².
        assert_eq!(
            profile.hessian().axis0_axis0().electric().values(),
            arr1(&[3010.0, 3011.0, 3012.0]).view(),
        );
        assert_eq!(
            profile.hessian().axis0_axis0().magnetic().values(),
            arr1(&[3110.0, 3111.0, 3112.0]).view(),
        );
        assert_eq!(
            profile.hessian().axis0_axis0().coupling().values(),
            arr1(&[3210.0, 3211.0, 3212.0]).view(),
        );
        assert_eq!(
            profile.hessian().axis0_axis0().total().values(),
            arr1(&[3310.0, 3311.0, 3312.0]).view(),
        );

        // Hessian: d²/daxis0daxis1.
        assert_eq!(
            profile.hessian().axis0_axis1().electric().values(),
            arr1(&[4010.0, 4011.0, 4012.0]).view(),
        );
        assert_eq!(
            profile.hessian().axis0_axis1().magnetic().values(),
            arr1(&[4110.0, 4111.0, 4112.0]).view(),
        );
        assert_eq!(
            profile.hessian().axis0_axis1().coupling().values(),
            arr1(&[4210.0, 4211.0, 4212.0]).view(),
        );
        assert_eq!(
            profile.hessian().axis0_axis1().total().values(),
            arr1(&[4310.0, 4311.0, 4312.0]).view(),
        );

        // Hessian: d²/daxis1².
        assert_eq!(
            profile.hessian().axis1_axis1().electric().values(),
            arr1(&[5010.0, 5011.0, 5012.0]).view(),
        );
        assert_eq!(
            profile.hessian().axis1_axis1().magnetic().values(),
            arr1(&[5110.0, 5111.0, 5112.0]).view(),
        );
        assert_eq!(
            profile.hessian().axis1_axis1().coupling().values(),
            arr1(&[5210.0, 5211.0, 5212.0]).view(),
        );
        assert_eq!(
            profile.hessian().axis1_axis1().total().values(),
            arr1(&[5310.0, 5311.0, 5312.0]).view(),
        );
    }

    #[test]
    fn bivariate_first_preserves_parameter_and_axis_order() {
        let derivatives = BivariateFirst::new([Parameter::InPlane, Parameter::Spectral], 10, 20);

        assert_eq!(
            derivatives.parameters(),
            [Parameter::InPlane, Parameter::Spectral],
        );
        assert_eq!(derivatives.axis0(), &10);
        assert_eq!(derivatives.axis1(), &20);

        assert_eq!(
            derivatives.into_parts(),
            ([Parameter::InPlane, Parameter::Spectral], 10, 20,),
        );
    }

    #[test]
    fn bivariate_second_preserves_gradient_and_hessian() {
        let gradient = BivariateGradient::new(10, 20);
        let hessian = BivariateHessian::new(30, 40, 50);

        let derivatives =
            BivariateSecond::new([Parameter::Spectral, Parameter::InPlane], gradient, hessian);

        assert_eq!(derivatives.gradient().axis0(), &10);
        assert_eq!(derivatives.gradient().axis1(), &20);
        assert_eq!(derivatives.hessian().axis0_axis0(), &30);
        assert_eq!(derivatives.hessian().axis0_axis1(), &40);
        assert_eq!(derivatives.hessian().axis1_axis1(), &50);
    }

    #[test]
    fn bivariate_second_map_preserves_component_order() {
        let derivatives = BivariateSecond::new(
            [Parameter::Spectral, Parameter::InPlane],
            BivariateGradient::new(1, 2),
            BivariateHessian::new(3, 4, 5),
        );

        let mapped = derivatives.map(|value| value * 10);

        assert_eq!(mapped.gradient().axis0(), &10);
        assert_eq!(mapped.gradient().axis1(), &20);
        assert_eq!(mapped.hessian().axis0_axis0(), &30);
        assert_eq!(mapped.hessian().axis0_axis1(), &40);
        assert_eq!(mapped.hessian().axis1_axis1(), &50);
    }
}
