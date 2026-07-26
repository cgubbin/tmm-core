//! zero-order differential jets.
//!
//! This module provides generic containers for propagating first
//! derivatives through algebraic expressions.
//!
//! [`Jet0`] stores a value
//!
//! ```text
//! (f)
//! ```
//!
//! The parameter semantics are represented by a marker type:
//!
//! - [`RealParameter`] permits differentiation of complex-valued expressions
//!   with respect to a real parameter, including conjugation, real-part
//!   extraction, and Hermitian products;
//! - [`HolomorphicParameter`] represents complex differentiation and exposes
//!   only operations that preserve holomorphicity.
//!
//! Holomorphic jets are suitable for analytic or meromorphic characteristic
//! functions used in argument-principle mode finding. Real-parameter jets are
//! suitable for derivatives of complex fields and physical observables with
//! respect to real frequency, wavenumber, angle, or thickness.
//!
//! The underlying value type determines which algebraic operations are
//! available through capability traits:
//!
//! - [`JetAdditive`] supports addition, subtraction, and negation;
//! - [`JetBilinear`] supports a bilinear product and its product rules;
//! - [`JetField`] supports elementwise reciprocals and division.
//!
//! Arrays implement these capabilities elementwise. Transfer matrices may
//! implement additive and bilinear operations because ordinary matrix
//! multiplication is bilinear. Scattering matrices must not implement
//! [`JetBilinear`] because the Redheffer star product is rational rather than
//! bilinear.
//!
//! Value-only calculations should use the underlying value type directly.
//! [`Jet`] is reserved for calculations that request derivatives.
use crate::algebra::JetMultiplyByScalar;

use super::{
    HolomorphicParameter, JetAdditive, JetBilinear, JetConjugate, JetConstant, JetCrossProduct,
    JetField, JetHermitianProduct, JetOneLike, JetRealPart, JetScaleBy, JetZeroLike, RealParameter,
};

use nalgebra::ComplexField;
use ndarray::{Array, ArrayBase, Dimension, OwnedRepr};
use std::marker::PhantomData;

pub(crate) type ArrayJet0<C, D, P> = Jet0<ArrayBase<OwnedRepr<C>, D>, P>;

pub(crate) type PhysicalJet0<C, D> = ArrayJet0<C, D, RealParameter>;
pub(crate) type ModeJet0<C, D> = ArrayJet0<C, D, HolomorphicParameter>;

/// A value and its first derivative.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Jet0<I, P = RealParameter> {
    value: I,
    parameter: PhantomData<P>,
}

impl<I, P> std::ops::Deref for Jet0<I, P> {
    type Target = I;

    fn deref(&self) -> &Self::Target {
        &self.value
    }
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
    /// Multiply two zero-order jets using the product rule.
    pub(crate) fn multiply(&self, rhs: &Self) -> Self {
        Self::new(self.value.jet_multiply(&rhs.value))
    }
}

impl<I, P> Jet0<I, P>
where
    I: JetConstant + JetZeroLike,
{
    /// Construct a constant zero-order jet with zero derivative.
    pub(crate) fn constant_like(source: &I, value: I::Scalar) -> Self {
        Self::new(source.jet_constant_like(value))
    }
}

impl<I, P> Jet0<I, P>
where
    I: JetZeroLike,
{
    /// Construct a zero-order jet whose derivative is zero.
    pub(crate) fn constant(value: I) -> Self {
        Self::new(value)
    }
}

impl<I, P> Jet0<I, P>
where
    I: JetScaleBy,
{
    /// Scale the value and derivative by a constant scalar.
    pub(crate) fn scale_by(&self, value: I::Scalar) -> Self {
        Self::new(self.value.jet_scale_by(value))
    }
}

impl<I, P> Jet0<I, P>
where
    I: JetField,
{
    /// Compute the elementwise reciprocal and its first derivative.
    pub(crate) fn reciprocal(&self) -> Self {
        let inverse = self.value.jet_elementwise_reciprocal();

        Self::new(inverse)
    }

    /// Divide two zero-order jets elementwise.
    pub(crate) fn divide(&self, rhs: &Self) -> Self {
        self.multiply(&rhs.reciprocal())
    }
}

impl<V, P> Jet0<V, P> {
    pub fn multiply_by_scalar<S>(&self, scalar: &Jet0<S, P>) -> Self
    where
        V: JetAdditive + JetMultiplyByScalar<S>,
        P: Clone,
    {
        Self::new(self.value().jet_multiply_by_scalar(scalar.value()))
    }
}

impl<I, P> Jet0<I, P>
where
    I: JetCrossProduct + JetAdditive,
{
    /// Compute the cross product of two zero-order jets.
    ///
    /// The derivative is evaluated using the bilinear product rule.
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
    /// This operation does not preserve holomorphicity and is therefore not
    /// available for holomorphic-parameter jets.
    /// Compute the Hermitian product of two jets differentiated with respect
    /// to a real scalar parameter.
    ///
    /// The Hermitian product is assumed to be conjugate-linear in its first
    /// operand and linear in its second operand:
    ///
    /// ```text
    /// h      = ⟨f, g⟩
    /// h′     = ⟨f′, g⟩ + ⟨f, g′⟩
    /// ```
    ///
    /// This operation is intentionally unavailable for holomorphic-parameter
    /// jets because conjugating the first operand does not preserve
    /// holomorphicity.
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
}

impl<C, D, P> ArrayJet0<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
{
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
    /// zero. For complex values, the branch convention is that of
    /// [`ComplexField::sqrt`].
    pub(crate) fn sqrt(self) -> Self {
        let value = self.into_inner();

        let result = value.mapv(|x| x.sqrt());

        Self::new(result)
    }
}

impl<I, P> Jet0<I, P> {
    /// Apply the same representation transformation independently to the value
    /// and derivative.
    ///
    /// This does not apply the differential chain rule and must not be used as a
    /// general function map.
    pub(crate) fn map_components<O, F>(self, mut f: F) -> Jet0<O, P>
    where
        F: FnMut(I) -> O,
    {
        Jet0::new(f(self.value))
    }

    pub(crate) fn variable(value: I) -> Self
    where
        I: JetOneLike,
    {
        Self::new(value)
    }
}
