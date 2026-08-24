use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, Ix0, OwnedRepr};
use num_traits::{FromPrimitive, float::FloatCore};
use std::fmt::Debug;

use crate::differential::{BivariateGradient, BivariateHessian};

use super::{
    ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, FirstOrderExpansion,
    RealParameter, SecondOrderExpansion,
};

/// Common representation metadata for a differential jet.
///
/// A jet combines a sampled primal value with zero or more derivative
/// components. All components share the same scalar type and sampled
/// dimension.
pub trait Jet {
    /// Sample dimension carried by each jet component.
    type Dimension: Dimension;

    /// Scalar stored by each sampled component.
    type Scalar: ComplexField;

    /// Equivalent jet family for a point-valued (`Ix0`) sample.
    type PointJet: Jet<Dimension = Ix0, Scalar = Self::Scalar>;
}

/// A complex-valued jet with a corresponding real-valued jet representation.
///
/// `RealJet` preserves the derivative family, sampled dimension, and parameter
/// policy while replacing the complex scalar by its real field.
pub trait ComplexJet: Jet {
    type RealJet;

    /// Promote a real-valued jet to the corresponding complex representation.
    fn into_complex(real: Self::RealJet) -> Self;
}

#[doc(hidden)]
pub trait ScalarAlgebra: Clone + Sized + std::fmt::Debug + Jet {
    /// Return the primal sampled value.
    fn value(&self) -> &ArrayBase<OwnedRepr<Self::Scalar>, Self::Dimension>;

    /// Lift a sampled value as a derivative-free constant.
    fn lift_constant(value: ArrayBase<OwnedRepr<Self::Scalar>, Self::Dimension>) -> Self;

    /// Construct a derivative-free constant with the sampled shape of `source`.
    fn filled_constant_like(
        source: &ArrayBase<OwnedRepr<Self::Scalar>, Self::Dimension>,
        value: Self::Scalar,
    ) -> Self;

    /// Construct zero with the same sampled shape.
    fn zero_like(&self) -> Self;

    /// Construct unity with the same sampled shape.
    fn one_like(&self) -> Self;

    /// Construct a constant with the same sampled shape as this value.
    fn constant(&self, value: Self::Scalar) -> Self {
        Self::filled_constant_like(self.value(), value)
    }

    /// Add another scalar-algebra value.
    fn add(&self, rhs: &Self) -> Self;

    /// Subtract another scalar-algebra value.
    fn subtract(&self, rhs: &Self) -> Self;

    /// Negate this value.
    fn negate(&self) -> Self;

    /// Multiply pointwise.
    fn multiply(&self, rhs: &Self) -> Self;

    /// Compute the pointwise reciprocal.
    fn reciprocal(&self) -> Self;

    /// Scale by a scalar constant.
    fn scale(&self, coefficient: Self::Scalar) -> Self;

    /// Apply the complex exponential pointwise.
    fn exp(&self) -> Self;

    /// Apply sine pointwise.
    fn sin(&self) -> Self;

    /// Apply cosine pointwise.
    fn cos(&self) -> Self;

    /// Apply the principal square root pointwise.
    fn sqrt(&self) -> Self;

    /// Return whether every primal and derivative component is finite.
    fn all_finite(&self) -> bool;

    /// Apply the square pointwise
    fn square(&self) -> Self {
        self.multiply(self)
    }

    /// Divide pointwise
    fn divide(&self, rhs: &Self) -> Self {
        self.multiply(&rhs.reciprocal())
    }
}

/// Numerically stable relative-exponential evaluation for scalar jets.
///
/// This is separated from [`ScalarAlgebra`] because its implementation
/// requires additional floating-point capabilities from the real scalar type.
#[doc(hidden)]
pub trait ScalarAlgebraExpRelExt: ScalarAlgebra {
    fn exprel(&self) -> Self;
}

/// Operations that are valid when the active differentiation coordinates are
/// real.
///
/// This trait is deliberately not implemented for jets parameterised by
/// [`super::HolomorphicParameter`].
pub trait RealScalarAlgebra: ScalarAlgebra + ComplexJet {
    fn conjugated(&self) -> Self;

    fn real(&self) -> Self::RealJet;

    fn imaginary(&self) -> Self::RealJet;

    /// Return `|self|²`, differentiating with respect to the active real
    /// parameter.
    ///
    /// This operation involves complex conjugation and is therefore not
    /// holomorphic.
    fn magnitude_squared(&self) -> Self::RealJet {
        self.multiply(&self.conjugated()).real()
    }

    fn hermitian_product(&self, other: &Self) -> Self {
        self.conjugated().multiply(other)
    }
}

// -------------------------------------------------------------------------
// Sampled unary-function composition
// -------------------------------------------------------------------------

/// Compose a sampled scalar function with a first-order argument jet.
///
/// `expansion` contains the sampled function value and derivative with
/// respect to its direct argument. The jet algebra applies the chain rule
/// using the derivatives already carried by `argument`.
pub(crate) trait FirstOrderFunctionAlgebra: ScalarAlgebra {
    fn compose_sampled_function(
        argument: &Self,
        expansion: FirstOrderExpansion<ArrayBase<OwnedRepr<Self::Scalar>, Self::Dimension>>,
    ) -> Self;
}

/// Compose a sampled scalar function with a second-order argument jet.
///
/// `expansion` contains the sampled function value and first, and second derivatives with
/// respect to its direct argument. The jet algebra applies the chain rule
/// using the derivatives already carried by `argument`.
pub(crate) trait SecondOrderFunctionAlgebra: ScalarAlgebra {
    fn compose_sampled_function(
        argument: &Self,
        expansion: SecondOrderExpansion<ArrayBase<OwnedRepr<Self::Scalar>, Self::Dimension>>,
    ) -> Self;
}

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

fn complex_is_finite<C>(value: &C) -> bool
where
    C: ComplexField + Copy,
{
    value.real().is_finite() && value.imaginary().is_finite()
}

fn array_is_finite<C, D>(value: &ArrayBase<OwnedRepr<C>, D>) -> bool
where
    C: ComplexField + Copy,
    D: Dimension,
{
    value.iter().all(complex_is_finite)
}

// -------------------------------------------------------------------------
// Plain sampled arrays
// -------------------------------------------------------------------------
//
impl<C, D, P> Jet for ArrayJet0<C, D, P>
where
    C: ComplexField,
    D: Dimension,
{
    type Scalar = C;
    type Dimension = D;
    type PointJet = ArrayJet0<C, Ix0, P>;
}

impl<C, D, P> ScalarAlgebra for ArrayJet0<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn value(&self) -> &ArrayBase<OwnedRepr<C>, Self::Dimension> {
        ArrayJet0::value(self)
    }

    fn lift_constant(value: ArrayBase<OwnedRepr<C>, Self::Dimension>) -> Self {
        ArrayJet0::constant(value)
    }

    fn filled_constant_like(source: &ArrayBase<OwnedRepr<C>, Self::Dimension>, value: C) -> Self {
        ArrayJet0::constant_like(source, value)
    }

    fn zero_like(&self) -> Self {
        ArrayJet0::constant_like(self.value(), C::zero())
    }

    fn one_like(&self) -> Self {
        ArrayJet0::constant_like(self.value(), C::one())
    }

    fn add(&self, rhs: &Self) -> Self {
        ArrayJet0::add(self, rhs)
    }

    fn subtract(&self, rhs: &Self) -> Self {
        ArrayJet0::subtract(self, rhs)
    }

    fn negate(&self) -> Self {
        ArrayJet0::negate(self)
    }

    fn multiply(&self, rhs: &Self) -> Self {
        ArrayJet0::multiply(self, rhs)
    }

    fn reciprocal(&self) -> Self {
        ArrayJet0::reciprocal(self)
    }

    fn scale(&self, coefficient: C) -> Self {
        ArrayJet0::scale_by(self, coefficient)
    }

    fn exp(&self) -> Self {
        self.clone().exp()
    }

    fn sin(&self) -> Self {
        self.clone().sin()
    }

    fn cos(&self) -> Self {
        self.clone().cos()
    }

    fn sqrt(&self) -> Self {
        self.clone().sqrt()
    }

    fn all_finite(&self) -> bool {
        array_is_finite(self.value())
    }
}

impl<C, D, P> ScalarAlgebraExpRelExt for ArrayJet0<C, D, P>
where
    C: ComplexField + Copy,
    C::RealField: FloatCore + FromPrimitive,
    D: Dimension,
    P: Clone + Debug,
{
    fn exprel(&self) -> Self {
        self.clone().exprel()
    }
}

impl<C, D, P> ComplexJet for ArrayJet0<C, D, P>
where
    C: ComplexField,
    D: Dimension,
{
    type RealJet = ArrayJet0<C::RealField, D, P>;

    fn into_complex(real: Self::RealJet) -> Self {
        let value = real.into_inner();

        ArrayJet0::new(value.mapv(ComplexField::from_real))
    }
}

impl<C, D> RealScalarAlgebra for ArrayJet0<C, D, RealParameter>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn conjugated(&self) -> Self {
        ArrayJet0::conjugated(self)
    }

    fn real(&self) -> Self::RealJet {
        ArrayJet0::real(self)
    }

    fn imaginary(&self) -> Self::RealJet {
        ArrayJet0::imaginary(self)
    }
}

// -------------------------------------------------------------------------
// First-order univariate jets
// -------------------------------------------------------------------------

impl<C, D, P> Jet for ArrayJet1<C, D, P>
where
    C: ComplexField,
    D: Dimension,
{
    type Scalar = C;
    type Dimension = D;
    type PointJet = ArrayJet1<C, Ix0, P>;
}

impl<C, D, P> ScalarAlgebra for ArrayJet1<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArrayJet1::value(self)
    }

    fn lift_constant(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJet1::constant(value)
    }

    fn filled_constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        ArrayJet1::constant_like(source, value)
    }

    fn zero_like(&self) -> Self {
        ArrayJet1::constant_like(self.value(), C::zero())
    }

    fn one_like(&self) -> Self {
        ArrayJet1::constant_like(self.value(), C::one())
    }

    fn add(&self, rhs: &Self) -> Self {
        ArrayJet1::add(self, rhs)
    }

    fn subtract(&self, rhs: &Self) -> Self {
        ArrayJet1::subtract(self, rhs)
    }

    fn negate(&self) -> Self {
        ArrayJet1::negate(self)
    }

    fn multiply(&self, rhs: &Self) -> Self {
        ArrayJet1::multiply(self, rhs)
    }

    fn reciprocal(&self) -> Self {
        ArrayJet1::reciprocal(self)
    }

    fn scale(&self, coefficient: C) -> Self {
        ArrayJet1::scale_by(self, coefficient)
    }

    fn exp(&self) -> Self {
        self.clone().exp()
    }

    fn sin(&self) -> Self {
        self.clone().sin()
    }

    fn cos(&self) -> Self {
        self.clone().cos()
    }

    fn sqrt(&self) -> Self {
        self.clone().sqrt()
    }

    fn all_finite(&self) -> bool {
        array_is_finite(self.value()) && array_is_finite(self.first())
    }
}

impl<C, D, P> ScalarAlgebraExpRelExt for ArrayJet1<C, D, P>
where
    C: ComplexField + Copy,
    C::RealField: FloatCore + FromPrimitive,
    D: Dimension,
    P: Clone + Debug,
{
    fn exprel(&self) -> Self {
        self.clone().exprel()
    }
}

impl<C, D, P> ComplexJet for ArrayJet1<C, D, P>
where
    C: ComplexField,
    D: Dimension,
{
    type RealJet = ArrayJet1<C::RealField, D, P>;

    fn into_complex(real: Self::RealJet) -> Self {
        let (value, first) = real.into_parts();

        ArrayJet1::from_parts(
            value.mapv(ComplexField::from_real),
            first.mapv(ComplexField::from_real),
        )
    }
}

impl<C, D> RealScalarAlgebra for ArrayJet1<C, D, RealParameter>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn conjugated(&self) -> Self {
        ArrayJet1::conjugated(self)
    }

    fn real(&self) -> Self::RealJet {
        ArrayJet1::real(self)
    }

    fn imaginary(&self) -> Self::RealJet {
        ArrayJet1::imaginary(self)
    }
}

impl<C, D, P> FirstOrderFunctionAlgebra for ArrayJet1<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn compose_sampled_function(
        argument: &Self,
        expansion: FirstOrderExpansion<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self {
        ArrayJet1::compose_sampled_function(argument, expansion)
    }
}

// -------------------------------------------------------------------------
// Second-order univariate jets
// -------------------------------------------------------------------------

impl<C, D, P> Jet for ArrayJet2<C, D, P>
where
    C: ComplexField,
    D: Dimension,
{
    type Scalar = C;
    type Dimension = D;
    type PointJet = ArrayJet2<C, Ix0, P>;
}

impl<C, D, P> ScalarAlgebra for ArrayJet2<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArrayJet2::value(self)
    }

    fn lift_constant(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJet2::constant(value)
    }

    fn filled_constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        ArrayJet2::constant_like(source, value)
    }

    fn zero_like(&self) -> Self {
        ArrayJet2::constant_like(self.value(), C::zero())
    }

    fn one_like(&self) -> Self {
        ArrayJet2::constant_like(self.value(), C::one())
    }

    fn add(&self, rhs: &Self) -> Self {
        ArrayJet2::add(self, rhs)
    }

    fn subtract(&self, rhs: &Self) -> Self {
        ArrayJet2::subtract(self, rhs)
    }

    fn negate(&self) -> Self {
        ArrayJet2::negate(self)
    }

    fn multiply(&self, rhs: &Self) -> Self {
        ArrayJet2::multiply(self, rhs)
    }

    fn reciprocal(&self) -> Self {
        ArrayJet2::reciprocal(self)
    }

    fn scale(&self, coefficient: C) -> Self {
        ArrayJet2::scale_by(self, coefficient)
    }

    fn exp(&self) -> Self {
        self.clone().exp()
    }

    fn sin(&self) -> Self {
        self.clone().sin()
    }

    fn cos(&self) -> Self {
        self.clone().cos()
    }

    fn sqrt(&self) -> Self {
        self.clone().sqrt()
    }

    fn all_finite(&self) -> bool {
        array_is_finite(self.value())
            && array_is_finite(self.first())
            && array_is_finite(self.second())
    }
}

impl<C, D, P> ScalarAlgebraExpRelExt for ArrayJet2<C, D, P>
where
    C: ComplexField + Copy,
    C::RealField: FloatCore + FromPrimitive,
    D: Dimension,
    P: Clone + Debug,
{
    fn exprel(&self) -> Self {
        self.clone().exprel()
    }
}

impl<C, D, P> ComplexJet for ArrayJet2<C, D, P>
where
    C: ComplexField,
    D: Dimension,
{
    type RealJet = ArrayJet2<C::RealField, D, P>;

    fn into_complex(real: Self::RealJet) -> Self {
        let (value, first, second) = real.into_parts();

        ArrayJet2::from_parts(
            value.mapv(ComplexField::from_real),
            first.mapv(ComplexField::from_real),
            second.mapv(ComplexField::from_real),
        )
    }
}

impl<C, D> RealScalarAlgebra for ArrayJet2<C, D, RealParameter>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn conjugated(&self) -> Self {
        ArrayJet2::conjugated(self)
    }

    fn real(&self) -> Self::RealJet {
        ArrayJet2::real(self)
    }

    fn imaginary(&self) -> Self::RealJet {
        ArrayJet2::imaginary(self)
    }
}

impl<C, D, P> SecondOrderFunctionAlgebra for ArrayJet2<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn compose_sampled_function(
        argument: &Self,
        expansion: SecondOrderExpansion<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self {
        ArrayJet2::compose_sampled_function(argument, expansion)
    }
}

// -------------------------------------------------------------------------
// Second-order bivariate jets
// -------------------------------------------------------------------------

impl<C, D, P> Jet for ArrayJetBivariate1<C, D, P>
where
    C: ComplexField,
    D: Dimension,
{
    type Scalar = C;
    type Dimension = D;
    type PointJet = ArrayJetBivariate1<C, Ix0, P>;
}

impl<C, D, P> ScalarAlgebra for ArrayJetBivariate1<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArrayJetBivariate1::value(self)
    }

    fn lift_constant(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJetBivariate1::constant(value)
    }

    fn filled_constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        ArrayJetBivariate1::constant_like(source, value)
    }

    fn zero_like(&self) -> Self {
        ArrayJetBivariate1::constant_like(self.value(), C::zero())
    }

    fn one_like(&self) -> Self {
        ArrayJetBivariate1::constant_like(self.value(), C::one())
    }

    fn add(&self, rhs: &Self) -> Self {
        ArrayJetBivariate1::add(self, rhs)
    }

    fn subtract(&self, rhs: &Self) -> Self {
        ArrayJetBivariate1::subtract(self, rhs)
    }

    fn negate(&self) -> Self {
        ArrayJetBivariate1::negate(self)
    }

    fn multiply(&self, rhs: &Self) -> Self {
        ArrayJetBivariate1::multiply(self, rhs)
    }

    fn reciprocal(&self) -> Self {
        ArrayJetBivariate1::reciprocal(self)
    }

    fn scale(&self, coefficient: C) -> Self {
        ArrayJetBivariate1::scale_by(self, coefficient)
    }

    fn exp(&self) -> Self {
        self.clone().exp()
    }

    fn sin(&self) -> Self {
        self.clone().sin()
    }

    fn cos(&self) -> Self {
        self.clone().cos()
    }

    fn sqrt(&self) -> Self {
        self.clone().sqrt()
    }

    fn all_finite(&self) -> bool {
        array_is_finite(self.value())
            && array_is_finite(self.axis0())
            && array_is_finite(self.axis1())
    }
}

impl<C, D, P> ScalarAlgebraExpRelExt for ArrayJetBivariate1<C, D, P>
where
    C: ComplexField + Copy,
    C::RealField: FloatCore + FromPrimitive,
    D: Dimension,
    P: Clone + Debug,
{
    fn exprel(&self) -> Self {
        self.clone().exprel()
    }
}

impl<C, D, P> ComplexJet for ArrayJetBivariate1<C, D, P>
where
    C: ComplexField,
    D: Dimension,
{
    type RealJet = ArrayJetBivariate1<C::RealField, D, P>;

    fn into_complex(real: Self::RealJet) -> Self {
        let (value, gradient) = real.into_parts();

        let (axis0, axis1) = gradient.into_parts();

        ArrayJetBivariate1::from_parts(
            value.mapv(ComplexField::from_real),
            BivariateGradient::new(
                axis0.mapv(ComplexField::from_real),
                axis1.mapv(ComplexField::from_real),
            ),
        )
    }
}

impl<C, D> RealScalarAlgebra for ArrayJetBivariate1<C, D, RealParameter>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn conjugated(&self) -> Self {
        ArrayJetBivariate1::conjugated(self)
    }

    fn real(&self) -> Self::RealJet {
        ArrayJetBivariate1::real(self)
    }

    fn imaginary(&self) -> Self::RealJet {
        ArrayJetBivariate1::imaginary(self)
    }
}

impl<C, D, P> FirstOrderFunctionAlgebra for ArrayJetBivariate1<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn compose_sampled_function(
        argument: &Self,
        expansion: FirstOrderExpansion<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self {
        ArrayJetBivariate1::compose_sampled_function(argument, expansion)
    }
}

impl<C, D, P> Jet for ArrayJetBivariate2<C, D, P>
where
    C: ComplexField,
    D: Dimension,
{
    type Scalar = C;
    type Dimension = D;
    type PointJet = ArrayJetBivariate2<C, Ix0, P>;
}

impl<C, D, P> ScalarAlgebra for ArrayJetBivariate2<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArrayJetBivariate2::value(self)
    }

    fn lift_constant(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJetBivariate2::constant(value)
    }

    fn filled_constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        ArrayJetBivariate2::constant_like(source, value)
    }

    fn zero_like(&self) -> Self {
        ArrayJetBivariate2::constant_like(self.value(), C::zero())
    }

    fn one_like(&self) -> Self {
        ArrayJetBivariate2::constant_like(self.value(), C::one())
    }

    fn add(&self, rhs: &Self) -> Self {
        ArrayJetBivariate2::add(self, rhs)
    }

    fn subtract(&self, rhs: &Self) -> Self {
        ArrayJetBivariate2::subtract(self, rhs)
    }

    fn negate(&self) -> Self {
        ArrayJetBivariate2::negate(self)
    }

    fn multiply(&self, rhs: &Self) -> Self {
        ArrayJetBivariate2::multiply(self, rhs)
    }

    fn reciprocal(&self) -> Self {
        ArrayJetBivariate2::reciprocal(self)
    }

    fn scale(&self, coefficient: C) -> Self {
        ArrayJetBivariate2::scale_by(self, coefficient)
    }

    fn exp(&self) -> Self {
        self.clone().exp()
    }

    fn sin(&self) -> Self {
        self.clone().sin()
    }

    fn cos(&self) -> Self {
        self.clone().cos()
    }

    fn sqrt(&self) -> Self {
        self.clone().sqrt()
    }

    fn all_finite(&self) -> bool {
        array_is_finite(self.value())
            && array_is_finite(self.axis0())
            && array_is_finite(self.axis1())
            && array_is_finite(self.axis0_axis0())
            && array_is_finite(self.axis0_axis1())
            && array_is_finite(self.axis1_axis1())
    }
}

impl<C, D, P> ScalarAlgebraExpRelExt for ArrayJetBivariate2<C, D, P>
where
    C: ComplexField + Copy,
    C::RealField: FloatCore + FromPrimitive,
    D: Dimension,
    P: Clone + Debug,
{
    fn exprel(&self) -> Self {
        self.clone().exprel()
    }
}

impl<C, D, P> ComplexJet for ArrayJetBivariate2<C, D, P>
where
    C: ComplexField,
    D: Dimension,
{
    type RealJet = ArrayJetBivariate2<C::RealField, D, P>;

    fn into_complex(real: Self::RealJet) -> Self {
        let (value, gradient, hessian) = real.into_parts();

        let (axis0, axis1) = gradient.into_parts();
        let (axis0_axis0, axis0_axis1, axis1_axis1) = hessian.into_parts();

        ArrayJetBivariate2::from_parts(
            value.mapv(ComplexField::from_real),
            BivariateGradient::new(
                axis0.mapv(ComplexField::from_real),
                axis1.mapv(ComplexField::from_real),
            ),
            BivariateHessian::new(
                axis0_axis0.mapv(ComplexField::from_real),
                axis0_axis1.mapv(ComplexField::from_real),
                axis1_axis1.mapv(ComplexField::from_real),
            ),
        )
    }
}

impl<C, D> RealScalarAlgebra for ArrayJetBivariate2<C, D, RealParameter>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn conjugated(&self) -> Self {
        ArrayJetBivariate2::conjugated(self)
    }

    fn real(&self) -> Self::RealJet {
        ArrayJetBivariate2::real(self)
    }

    fn imaginary(&self) -> Self::RealJet {
        ArrayJetBivariate2::imaginary(self)
    }
}

impl<C, D, P> SecondOrderFunctionAlgebra for ArrayJetBivariate2<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn compose_sampled_function(
        argument: &Self,
        expansion: SecondOrderExpansion<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self {
        ArrayJetBivariate2::compose_sampled_function(argument, expansion)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;
    use num_traits::Zero;

    use crate::{
        algebra::{
            ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2,
            HolomorphicParameter, RealParameter,
        },
        differential::{BivariateGradient, BivariateHessian},
    };

    type C = Complex64;

    type J0 = ArrayJet0<C, Ix0, RealParameter>;
    type J1 = ArrayJet1<C, Ix0, RealParameter>;
    type J2 = ArrayJet2<C, Ix0, RealParameter>;
    type JB1 = ArrayJetBivariate1<C, Ix0, RealParameter>;
    type JB2 = ArrayJetBivariate2<C, Ix0, RealParameter>;

    type HolomorphicJ1 = ArrayJet1<C, Ix0, HolomorphicParameter>;
    type HolomorphicJ2 = ArrayJet2<C, Ix0, HolomorphicParameter>;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    // ---------------------------------------------------------------------
    // Jet metadata
    // ---------------------------------------------------------------------

    #[test]
    fn point_jet_preserves_family_scalar_and_policy() {
        fn assert_point_jet<J>()
        where
            J: Jet<Dimension = Ix0, Scalar = C>,
            J::PointJet: Jet<Dimension = Ix0, Scalar = C>,
        {
        }

        assert_point_jet::<J0>();
        assert_point_jet::<J1>();
        assert_point_jet::<J2>();
        assert_point_jet::<JB1>();
        assert_point_jet::<JB2>();
    }

    // ---------------------------------------------------------------------
    // Constant lifting
    // ---------------------------------------------------------------------

    #[test]
    fn value_only_lift_constant_preserves_value() {
        let value = arr0(c(2.0, -1.0));

        let jet = <J0 as ScalarAlgebra>::lift_constant(value.clone());

        assert_eq!(jet.value(), &value);
    }

    #[test]
    fn first_order_lift_constant_zeros_derivative() {
        let value = arr0(c(2.0, -1.0));

        let jet = <J1 as ScalarAlgebra>::lift_constant(value.clone());

        assert_eq!(jet.value(), &value);
        assert_eq!(jet.first(), &arr0(C::new(0.0, 0.0)));
    }

    #[test]
    fn second_order_lift_constant_zeros_all_derivatives() {
        let value = arr0(c(2.0, -1.0));

        let jet = <J2 as ScalarAlgebra>::lift_constant(value.clone());

        assert_eq!(jet.value(), &value);
        assert_eq!(jet.first(), &arr0(C::new(0.0, 0.0)));
        assert_eq!(jet.second(), &arr0(C::new(0.0, 0.0)));
    }

    #[test]
    fn bivariate_lift_constant_zeros_gradient_and_hessian() {
        let value = arr0(c(2.0, -1.0));

        let jet = <JB2 as ScalarAlgebra>::lift_constant(value.clone());

        assert_eq!(jet.value(), &value);

        assert_eq!(jet.axis0(), &arr0(C::zero()));
        assert_eq!(jet.axis1(), &arr0(C::zero()));

        assert_eq!(jet.axis0_axis0(), &arr0(C::zero()));
        assert_eq!(jet.axis0_axis1(), &arr0(C::zero()));
        assert_eq!(jet.axis1_axis1(), &arr0(C::zero()));
    }

    #[test]
    fn filled_constant_like_preserves_sample_shape() {
        use ndarray::{Ix1, array};

        type J = ArrayJet1<C, Ix1, RealParameter>;

        let source = array![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0)];

        let jet = <J as ScalarAlgebra>::filled_constant_like(&source, c(7.0, -2.0));

        assert_eq!(
            jet.value(),
            &array![c(7.0, -2.0), c(7.0, -2.0), c(7.0, -2.0)],
        );

        assert_eq!(jet.first(), &array![C::zero(), C::zero(), C::zero()],);
    }

    // ---------------------------------------------------------------------
    // Sampled function composition
    // ---------------------------------------------------------------------

    #[test]
    fn first_order_composition_respects_preseeded_holomorphic_direction() {
        /*
         * x(z) = 3 + 2 z locally
         *
         *     x      = 3
         *     dx/dz  = 2
         *
         * f(x) = x²
         *
         *     f(3)   = 9
         *     f'(3)  = 6
         *
         * Therefore
         *
         *     df/dz = f'(x) dx/dz = 12.
         */

        let argument = HolomorphicJ1::from_parts(arr0(c(3.0, 0.0)), arr0(c(2.0, 0.0)));

        let expansion = FirstOrderExpansion::new(arr0(c(9.0, 0.0)), arr0(c(6.0, 0.0)));

        let result = <HolomorphicJ1 as FirstOrderFunctionAlgebra>::compose_sampled_function(
            &argument, expansion,
        );

        assert_eq!(result.value(), &arr0(c(9.0, 0.0)));
        assert_eq!(result.first(), &arr0(c(12.0, 0.0)));
    }

    #[test]
    fn second_order_composition_applies_full_chain_rule() {
        /*
         * At the evaluation point:
         *
         *     x       = 3
         *     x'      = 2
         *     x''     = 5
         *
         * f(x) = x²:
         *
         *     f       = 9
         *     f'      = 6
         *     f''     = 2
         *
         * Therefore
         *
         *     (f ∘ x)'  = f' x'
         *                = 12
         *
         *     (f ∘ x)'' = f'' (x')² + f' x''
         *                = 2 * 4 + 6 * 5
         *                = 38.
         */

        let argument =
            HolomorphicJ2::from_parts(arr0(c(3.0, 0.0)), arr0(c(2.0, 0.0)), arr0(c(5.0, 0.0)));

        let expansion =
            SecondOrderExpansion::new(arr0(c(9.0, 0.0)), arr0(c(6.0, 0.0)), arr0(c(2.0, 0.0)));

        let result = <HolomorphicJ2 as SecondOrderFunctionAlgebra>::compose_sampled_function(
            &argument, expansion,
        );

        assert_eq!(result.value(), &arr0(c(9.0, 0.0)));
        assert_eq!(result.first(), &arr0(c(12.0, 0.0)));
        assert_eq!(result.second(), &arr0(c(38.0, 0.0)));
    }

    #[test]
    fn bivariate_first_order_composition_applies_chain_rule_to_both_axes() {
        /*
         * x_u = 2
         * x_v = -3
         * f'(x) = 6
         *
         * Therefore
         *
         *     f_u = 12
         *     f_v = -18.
         */

        let argument = JB1::from_parts(
            arr0(c(3.0, 0.0)),
            BivariateGradient::new(arr0(c(2.0, 0.0)), arr0(c(-3.0, 0.0))),
        );

        let expansion = FirstOrderExpansion::new(arr0(c(9.0, 0.0)), arr0(c(6.0, 0.0)));

        let result =
            <JB1 as FirstOrderFunctionAlgebra>::compose_sampled_function(&argument, expansion);

        assert_eq!(result.value(), &arr0(c(9.0, 0.0)));
        assert_eq!(result.axis0(), &arr0(c(12.0, 0.0)));
        assert_eq!(result.axis1(), &arr0(c(-18.0, 0.0)));
    }

    // ---------------------------------------------------------------------
    // Complex / real representation conversion
    // ---------------------------------------------------------------------

    #[test]
    fn into_complex_preserves_all_univariate_components() {
        type Complex = ArrayJet2<C, Ix0, RealParameter>;
        type Real = ArrayJet2<f64, Ix0, RealParameter>;

        let real = Real::from_parts(arr0(2.0), arr0(-3.0), arr0(5.0));

        let complex = <Complex as ComplexJet>::into_complex(real);

        assert_eq!(complex.value(), &arr0(c(2.0, 0.0)));
        assert_eq!(complex.first(), &arr0(c(-3.0, 0.0)));
        assert_eq!(complex.second(), &arr0(c(5.0, 0.0)));
    }

    #[test]
    fn into_complex_preserves_bivariate_gradient_and_hessian() {
        type Complex = ArrayJetBivariate2<C, Ix0, RealParameter>;

        type Real = ArrayJetBivariate2<f64, Ix0, RealParameter>;

        let real = Real::from_parts(
            arr0(1.0),
            BivariateGradient::new(arr0(2.0), arr0(3.0)),
            BivariateHessian::new(arr0(4.0), arr0(5.0), arr0(6.0)),
        );

        let complex = <Complex as ComplexJet>::into_complex(real);

        assert_eq!(complex.value(), &arr0(c(1.0, 0.0)));
        assert_eq!(complex.axis0(), &arr0(c(2.0, 0.0)));
        assert_eq!(complex.axis1(), &arr0(c(3.0, 0.0)));
        assert_eq!(complex.axis0_axis0(), &arr0(c(4.0, 0.0)),);
        assert_eq!(complex.axis0_axis1(), &arr0(c(5.0, 0.0)),);
        assert_eq!(complex.axis1_axis1(), &arr0(c(6.0, 0.0)),);
    }

    // ---------------------------------------------------------------------
    // Real-parameter-only operations
    // ---------------------------------------------------------------------

    #[test]
    fn real_scalar_algebra_propagates_real_and_imaginary_parts() {
        let source = J1::from_parts(arr0(c(2.0, -3.0)), arr0(c(5.0, 7.0)));

        let real = source.real();
        let imaginary = source.imaginary();

        assert_eq!(real.value(), &arr0(2.0));
        assert_eq!(real.first(), &arr0(5.0));

        assert_eq!(imaginary.value(), &arr0(-3.0));
        assert_eq!(imaginary.first(), &arr0(7.0));
    }

    #[test]
    fn magnitude_squared_propagates_derivative_for_real_parameter() {
        /*
         * z(t) = (2 + 3i) + (5 - 7i) t
         *
         * d|z|²/dt = 2 Re(conj(z) z')
         *           = 2 (2*5 + 3*(-7))
         *           = -22.
         */

        let source = J1::from_parts(arr0(c(2.0, 3.0)), arr0(c(5.0, -7.0)));

        let magnitude_squared = RealScalarAlgebra::magnitude_squared(&source);

        assert_eq!(magnitude_squared.value(), &arr0(13.0),);

        assert_eq!(magnitude_squared.first(), &arr0(-22.0),);
    }

    // ---------------------------------------------------------------------
    // Finiteness
    // ---------------------------------------------------------------------

    #[test]
    fn all_finite_accepts_finite_values_and_derivatives() {
        let zero = J0::new(arr0(c(1.0, 2.0)));

        let first = J1::from_parts(arr0(c(1.0, 2.0)), arr0(c(3.0, 4.0)));

        let second = J2::from_parts(arr0(c(1.0, 2.0)), arr0(c(3.0, 4.0)), arr0(c(5.0, 6.0)));

        let bivariate = JB2::from_parts(
            arr0(c(1.0, 2.0)),
            BivariateGradient::new(arr0(c(3.0, 4.0)), arr0(c(5.0, 6.0))),
            BivariateHessian::new(arr0(c(7.0, 8.0)), arr0(c(9.0, 10.0)), arr0(c(11.0, 12.0))),
        );

        assert!(zero.all_finite());
        assert!(first.all_finite());
        assert!(second.all_finite());
        assert!(bivariate.all_finite());
    }

    #[test]
    fn all_finite_checks_first_derivative() {
        let source = J1::from_parts(arr0(c(1.0, 0.0)), arr0(c(f64::INFINITY, 0.0)));

        assert!(!source.all_finite());
    }

    #[test]
    fn all_finite_checks_second_derivative() {
        let source = J2::from_parts(arr0(c(1.0, 0.0)), arr0(c(2.0, 0.0)), arr0(c(0.0, f64::NAN)));

        assert!(!source.all_finite());
    }

    #[test]
    fn all_finite_checks_bivariate_mixed_derivative() {
        let source = JB2::from_parts(
            arr0(c(1.0, 0.0)),
            BivariateGradient::new(arr0(c(2.0, 0.0)), arr0(c(3.0, 0.0))),
            BivariateHessian::new(arr0(c(4.0, 0.0)), arr0(c(f64::NAN, 0.0)), arr0(c(6.0, 0.0))),
        );

        assert!(!source.all_finite());
    }

    // ---------------------------------------------------------------------
    // ScalarAlgebra default operations
    // ---------------------------------------------------------------------

    #[test]
    fn square_uses_scalar_algebra_multiplication() {
        let source = J1::from_parts(arr0(c(3.0, 0.0)), arr0(c(2.0, 0.0)));

        let result = ScalarAlgebra::square(&source);

        assert_eq!(result.value(), &arr0(c(9.0, 0.0)));
        assert_eq!(result.first(), &arr0(c(12.0, 0.0)));
    }

    #[test]
    fn divide_uses_reciprocal_with_derivatives() {
        /*
         * f = 6, f' = 4
         * g = 3, g' = 2
         *
         * f/g = 2
         *
         * (f/g)' = (f'g - fg') / g²
         *          = (12 - 12) / 9
         *          = 0.
         */

        let numerator = J1::from_parts(arr0(c(6.0, 0.0)), arr0(c(4.0, 0.0)));

        let denominator = J1::from_parts(arr0(c(3.0, 0.0)), arr0(c(2.0, 0.0)));

        let result = ScalarAlgebra::divide(&numerator, &denominator);

        assert_eq!(result.value(), &arr0(c(2.0, 0.0)));
        assert_eq!(result.first(), &arr0(c(0.0, 0.0)));
    }
}
