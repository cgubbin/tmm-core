//! Zero-order differential jets.
//!
//! [`Jet0`] wraps a primal value without carrying derivative components:
//!
//! ```text
//! (f)
//! ```
//!
//! It provides the value-only member of the jet families used throughout the
//! differential algebra. Keeping the same parameter marker as higher-order
//! jets allows generic code to preserve whether a calculation belongs to a
//! real-parameter or holomorphic family even when no derivatives are
//! requested.
//!
//! The underlying value type determines which algebraic operations are
//! available through capability traits. Operations on [`Jet0`] simply apply
//! the corresponding operation to its wrapped value; no product or chain
//! rules are required.
//!
//! For complex-valued payloads, non-holomorphic operations such as
//! conjugation, real-part extraction, and Hermitian products are exposed only
//! for [`RealParameter`] jets. This preserves the same capability boundary as
//! the higher-order jet families.

use crate::algebra::{JetMultiplyByScalar, exprel};

use super::{
    JetAdditive, JetBilinear, JetConjugate, JetConstant, JetCrossProduct, JetHermitianProduct,
    JetRealPart, JetReciprocal, JetScaleBy, RealParameter,
};

use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, OwnedRepr};
use num_traits::{FromPrimitive, float::FloatCore};
use std::marker::PhantomData;

pub(crate) type ArrayJet0<C, D, P> = Jet0<ArrayBase<OwnedRepr<C>, D>, P>;

/// A zero-order jet containing only a primal value.
///
/// `P` identifies the parameter policy of the corresponding differential
/// family but has no effect on the stored value.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct Jet0<I, P = RealParameter> {
    value: I,
    parameter: PhantomData<P>,
}

impl<I, P> Jet0<I, P> {
    /// Construct a zero-order jet.
    pub(crate) fn new(value: I) -> Self {
        Self {
            value,
            parameter: PhantomData,
        }
    }

    /// Return the value.
    pub(crate) fn value(&self) -> &I {
        &self.value
    }

    /// Consume the jet and return its components.
    pub(crate) fn into_inner(self) -> I {
        self.value
    }
}

impl<I, P> Jet0<I, P>
where
    I: JetAdditive,
{
    pub(crate) fn add(&self, rhs: &Self) -> Self {
        Self::new(self.value.jet_add(&rhs.value))
    }

    pub(crate) fn subtract(&self, rhs: &Self) -> Self {
        Self::new(self.value.jet_subtract(&rhs.value))
    }

    pub(crate) fn negate(&self) -> Self {
        Self::new(self.value.jet_negate())
    }
}

impl<I, P> Jet0<I, P>
where
    I: JetBilinear,
{
    /// Multiply the wrapped values using their bilinear product.
    pub(crate) fn multiply(&self, rhs: &Self) -> Self {
        Self::new(self.value.jet_multiply(&rhs.value))
    }
}

impl<I, P> Jet0<I, P>
where
    I: JetConstant,
{
    /// Construct a zero-order constant with the representation of `source`.
    pub(crate) fn constant_like(source: &I, value: I::Scalar) -> Self {
        Self::new(source.jet_constant_like(value))
    }
}

impl<I, P> Jet0<I, P> {
    /// Lift a value into the zero-order jet representation.
    pub(crate) fn constant(value: I) -> Self {
        Self::new(value)
    }
}

impl<I, P> Jet0<I, P>
where
    I: JetScaleBy,
{
    /// Scale the wrapped value by a constant scalar.
    pub(crate) fn scale_by(&self, value: I::Scalar) -> Self {
        Self::new(self.value.jet_scale_by(value))
    }
}

impl<I, P> Jet0<I, P>
where
    I: JetReciprocal,
{
    /// Compute the elementwise reciprocal.
    pub(crate) fn reciprocal(&self) -> Self {
        let inverse = self.value.jet_elementwise_reciprocal();

        Self::new(inverse)
    }
}

impl<V, P> Jet0<V, P> {
    pub(crate) fn multiply_by_scalar<S>(&self, scalar: &Jet0<S, P>) -> Self
    where
        V: JetMultiplyByScalar<S>,
    {
        Self::new(self.value().jet_multiply_by_scalar(scalar.value()))
    }
}

impl<I, P> Jet0<I, P>
where
    I: JetCrossProduct,
{
    /// Compute the cross product of two wrapped values
    pub(crate) fn cross(&self, rhs: &Self) -> Self {
        let value = self.value.jet_cross(&rhs.value);

        Self::new(value)
    }
}

impl<I> Jet0<I, RealParameter>
where
    I: JetHermitianProduct,
    I::Output: JetAdditive,
{
    /// Compute the Hermitian product of two real-parameter zero-order jets.
    ///
    /// This operation is intentionally unavailable for holomorphic-parameter
    /// jets because the Hermitian product conjugates its first operand.
    pub(crate) fn hermitian_dot_product(&self, rhs: &Self) -> Jet0<I::Output, RealParameter> {
        let value = self.value().jet_hermitian_product(rhs.value());

        Jet0::new(value)
    }
}

impl<I> Jet0<I, RealParameter>
where
    I: JetConjugate,
{
    /// Conjugate a jet differentiated with respect to a real parameter.
    ///
    /// For real `x`,
    ///
    /// ```text
    /// d(conj(f))/dx  = conj(df/dx)
    /// ```
    ///
    /// This operation is intentionally unavailable for holomorphic-parameter
    /// jets because complex conjugation does not preserve holomorphicity.
    pub(crate) fn conjugated(&self) -> Self {
        Self::new(self.value.jet_conjugate())
    }
}

impl<I> Jet0<I, RealParameter>
where
    I: JetRealPart,
{
    /// Extract the real parts of a jet differentiated with respect to a real
    /// parameter.
    ///
    /// This operation is intentionally unavailable for holomorphic-parameter
    /// jets.
    pub(crate) fn real(&self) -> Jet0<I::RealOutput, RealParameter> {
        Jet0::new(self.value.jet_real())
    }

    /// Extract the imaginary parts of a jet differentiated with respect to a real
    /// parameter.
    ///
    /// This operation is intentionally unavailable for holomorphic-parameter
    /// jets.
    pub(crate) fn imaginary(&self) -> Jet0<I::RealOutput, RealParameter> {
        Jet0::new(self.value.jet_imaginary())
    }
}

impl<C, D, P> ArrayJet0<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    pub(crate) fn exprel(self) -> Self
    where
        C::RealField: FloatCore + FromPrimitive,
    {
        let value = self.into_inner();

        let result = value.mapv(|x| exprel(x));

        Self::new(result)
    }

    pub(crate) fn exp(self) -> Self {
        let value = self.into_inner();

        let result = value.mapv(|x| x.exp());

        Self::new(result)
    }

    pub(crate) fn sin(self) -> Self {
        let value = self.into_inner();

        let result = value.mapv(|x| x.sin());

        Self::new(result)
    }

    pub(crate) fn cos(self) -> Self {
        let value = self.into_inner();

        let result = value.mapv(|x| x.cos());

        Self::new(result)
    }

    /// Compute the elementwise principal square root and its derivative.
    ///
    /// The derivative is `f' / (2 sqrt(f))`. It is singular where the value is
    /// zero.
    pub(crate) fn sqrt(self) -> Self {
        let value = self.into_inner();

        let result = value.mapv(|x| x.sqrt());

        Self::new(result)
    }
}

#[cfg(test)]
mod jet0_deref {
    impl<I, P> std::ops::Deref for super::Jet0<I, P> {
        type Target = I;

        fn deref(&self) -> &Self::Target {
            &self.value
        }
    }
}
