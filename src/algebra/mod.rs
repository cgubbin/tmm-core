//! Differential algebra used by Lamina's numerical backends.
//!
//! This module defines the operations required to propagate values and
//! derivatives through transfer-, scattering-, field-, and observable
//! calculations.
//!
//! Concrete jet families represent value-only, first-order, and second-order
//! directional or bivariate derivatives. Their payloads need not be scalar:
//! the same product rules are used for sampled arrays, Cartesian fields,
//! matrices, and other algebraic values.
//!
//! Two derivative policies are distinguished:
//!
//! - [`RealParameter`] represents differentiation with respect to a real
//!   parameter while allowing complex-valued intermediate quantities;
//! - [`HolomorphicParameter`] represents differentiation with respect to a
//!   complex parameter and permits only operations compatible with
//!   holomorphic differentiation.
//!
//! Most users should interact with the higher-level evaluator API rather than
//! these algebraic implementation details.

mod cartesian;
mod exprel;
mod jet_bivariate_one;
mod jet_bivariate_two;
mod jet_one;
mod jet_two;
mod jet_zero;
mod scalar;
mod scale;
mod seed;
mod stack;

#[cfg(test)]
mod tests;

pub use jet_one::ModeJet1;
pub use scalar::ScalarAlgebra;
pub use seed::SeedJet;

pub(crate) use jet_bivariate_one::{ArrayJetBivariate1, JetBivariate1};
pub(crate) use jet_bivariate_two::{ArrayJetBivariate2, JetBivariate2};
pub(crate) use jet_one::{ArrayJet1, FirstOrderExpansion, Jet1};
pub(crate) use jet_two::{ArrayJet2, Jet2, SecondOrderExpansion};
pub(crate) use jet_zero::{ArrayJet0, Jet0};

pub(crate) use exprel::{exprel, exprel_first, exprel_second};
pub(crate) use scalar::{ComplexJet, Jet, RealScalarAlgebra, ScalarAlgebraExpRelExt};
pub(crate) use seed::UnsupportedDerivativeSlot;
pub(crate) use stack::JetStack;

pub(crate) use cartesian::{
    CartesianScalarAlgebra, CartesianVectorAlgebra, RealCartesianVectorAlgebra,
};
pub(crate) use scale::ScaleBy;

use nalgebra::ComplexField;
use ndarray::{Array, Dimension};

/// Differentiation with respect to a real scalar parameter.
///
/// Values and derivatives may be complex. Non-holomorphic operations such as
/// complex conjugation and real-part extraction are permitted because the
/// independent parameter is real.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RealParameter;

/// Holomorphic differentiation with respect to a complex scalar parameter.
///
/// Operations that depend on the complex conjugate of the independent
/// variable must not be available for jets carrying this marker.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HolomorphicParameter;

/// Construct a zero value with the same representation and sampled shape.
pub trait JetZeroLike: Clone {
    fn jet_zeros_like(shape_source: &Self) -> Self;
}

/// Construct a unity value with the same representation and sampled shape.
pub trait JetOneLike: Clone {
    fn jet_ones_like(shape_source: &Self) -> Self;
}

/// Additive operations supported componentwise by jets.
pub trait JetAdditive: Sized {
    /// Add two values.
    fn jet_add(&self, rhs: &Self) -> Self;

    /// Subtract `rhs` from this value.
    fn jet_subtract(&self, rhs: &Self) -> Self;

    /// Negate this value.
    fn jet_negate(&self) -> Self;

    /// Return twice this value.
    fn jet_double(&self) -> Self {
        self.jet_add(self)
    }
}

/// A bilinear product for which the ordinary product rule applies.
pub trait JetBilinear: JetAdditive {
    /// Multiply or compose two values using a bilinear operation.
    fn jet_multiply(&self, rhs: &Self) -> Self;

    /// Multiply or compose two values using a bilinear operation.
    fn jet_square(&self) -> Self {
        self.jet_multiply(self)
    }
}

/// Scale a value by a scalar.
pub trait JetScaleBy: Clone {
    type Scalar: Copy;

    fn jet_scale_by(&self, value: Self::Scalar) -> Self;
}

/// Construct a scalar constant with the same sampled shape.
pub trait JetConstant: Clone {
    type Scalar: Copy;

    fn jet_constant_like(&self, value: Self::Scalar) -> Self;
}

/// Elementwise field operations used by reciprocal and division.
///
/// This trait is appropriate for sampled scalar arrays, not matrices.
pub trait JetReciprocal: JetBilinear {
    fn jet_elementwise_reciprocal(&self) -> Self;
}

/// A bilinear cross product between values of the same type.
///
/// Implementations must be bilinear in both operands so that the ordinary
/// first- and second-order product rules are valid.
pub trait JetCrossProduct: Clone {
    fn jet_cross(&self, rhs: &Self) -> Self;
}

/// A conjugating inner product between jet payload values.
///
/// This operation is not holomorphic and must only be exposed by jet
/// families whose differentiation policy permits conjugation.
pub trait JetHermitianProduct: Clone {
    type Output;

    fn jet_hermitian_product(&self, rhs: &Self) -> Self::Output;
}

/// Pointwise or componentwise complex conjugation.
///
/// Conjugation is not holomorphic and must not be exposed through
/// holomorphic jet algebras.
pub trait JetConjugate {
    fn jet_conjugate(&self) -> Self;
}

/// Extraction of real and imaginary components.
///
/// These operations are not holomorphic and are intended for jets
/// differentiated with respect to real parameters.
pub trait JetRealPart {
    type RealOutput;

    fn jet_real(&self) -> Self::RealOutput;
    fn jet_imaginary(&self) -> Self::RealOutput;
}

/// Multiplication by a separate scalar-valued payload.
///
/// This is used when a structured jet payload, such as a Cartesian vector
/// field, is multiplied pointwise by a scalar field.
pub trait JetMultiplyByScalar<S> {
    fn jet_multiply_by_scalar(&self, scalar: &S) -> Self;
}

impl<C, D> JetZeroLike for Array<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn jet_zeros_like(shape_source: &Self) -> Self {
        Array::zeros(shape_source.raw_dim())
    }
}

impl<C, D> JetOneLike for Array<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn jet_ones_like(shape_source: &Self) -> Self {
        Array::ones(shape_source.raw_dim())
    }
}

impl<C, D> JetAdditive for Array<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn jet_add(&self, rhs: &Self) -> Self {
        self.clone() + rhs.view()
    }

    fn jet_subtract(&self, rhs: &Self) -> Self {
        self.clone() - rhs.view()
    }

    fn jet_negate(&self) -> Self {
        -self.clone()
    }
}

impl<C, D> JetBilinear for Array<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn jet_multiply(&self, rhs: &Self) -> Self {
        self.clone() * rhs.view()
    }
}

impl<C, D> JetScaleBy for Array<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Scalar = C;

    fn jet_scale_by(&self, value: Self::Scalar) -> Self {
        self.mapv(|x| x * value)
    }
}

impl<C, D> JetConstant for Array<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Scalar = C;

    fn jet_constant_like(&self, value: C) -> Self {
        Array::from_elem(self.raw_dim(), value)
    }
}

impl<C, D> JetRealPart for Array<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type RealOutput = Array<C::RealField, D>;

    fn jet_real(&self) -> Self::RealOutput {
        self.mapv(|z| z.real())
    }

    fn jet_imaginary(&self) -> Self::RealOutput {
        self.mapv(|z| z.imaginary())
    }
}

impl<C, D> JetConjugate for Array<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn jet_conjugate(&self) -> Self {
        self.mapv(|z| z.conjugate())
    }
}

impl<C, D> JetReciprocal for Array<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn jet_elementwise_reciprocal(&self) -> Self {
        self.mapv(|x| C::one() / x)
    }
}
