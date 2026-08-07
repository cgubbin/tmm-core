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

use crate::parameter::Parameter;


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

#[cfg(test)]
mod tests {
    use super::*;

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
