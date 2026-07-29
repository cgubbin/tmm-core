//! Minimal jet operations required for coordinate canonicalisation.
//!
//! Coordinate compilation converts caller-facing spectral and in-plane
//! coordinates into the canonical variables consumed by the numerical
//! backend:
//!
//! - vacuum angular wavenumber in inverse centimetres;
//! - parallel angular wavenumber in inverse centimetres.
//!
//! These conversions require only a small subset of the complete jet algebra:
//! scaling by a real coefficient, multiplication, reciprocation, and sine.
//! [`CanonicalCoordinateJet`] exposes that minimal interface so coordinate
//! compilation does not depend directly on the much larger
//! [`ScalarAlgebra`](crate::algebra::ScalarAlgebra) contract.
//!
//! Implementations are provided explicitly for each supported jet family.
//! This makes the jet representations accepted by coordinate compilation
//! deliberate and permits lightweight test implementations without conflicting
//! with a blanket implementation.

use nalgebra::ComplexField;
use ndarray::Dimension;

use std::fmt::Debug;

use crate::algebra::{
    ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, Jet, ScalarAlgebra,
};

/// Jet operations required to convert caller-facing coordinates into canonical
/// backend coordinates.
///
/// This trait is an internal adapter over the complete scalar jet algebra. It
/// contains only the operations used by spectral and in-plane coordinate
/// transformations.
///
/// Methods consume `self` because canonicalisation forms a sequence of
/// transformations and does not need to retain intermediate jets.
///
/// The scalar type `C` is the complex coefficient type used by the backend,
/// while `D` is the sampled ndarray dimension carried by each jet coefficient.
pub(crate) trait CanonicalCoordinateJet: Sized + Clone + Jet
where
    Self::Scalar: ComplexField,
{
    /// Multiply every jet coefficient by a real scalar.
    ///
    /// This is used for unit conversions and fixed physical constants. Since
    /// `factor` is independent of the caller-facing coordinates, it does not
    /// introduce any additional derivative terms.
    fn scale_real(self, factor: <Self::Scalar as ComplexField>::RealField) -> Self;

    /// Return the multiplicative reciprocal of this jet.
    ///
    /// This operation is required when converting vacuum wavelength into
    /// vacuum angular wavenumber:
    ///
    /// ```text
    /// k₀ = 2π / λ
    /// ```
    ///
    /// The jet algebra propagates derivatives through the reciprocal.
    fn reciprocal(self) -> Self;

    /// Apply sine through the jet algebra.
    ///
    /// This operation is required when converting an incident angle into
    /// parallel angular wavenumber:
    ///
    /// ```text
    /// k∥ = nᵢ k₀ sin(θ)
    /// ```
    ///
    /// The operation applies to the complete jet, not merely to its primal
    /// value, so the corresponding derivatives are propagated automatically.
    fn sin(self) -> Self;

    /// Multiply two jets coefficient-wise through the jet algebra.
    ///
    /// This is used for coordinate transformations whose result depends on
    /// multiple potentially differentiated quantities, such as:
    ///
    /// ```text
    /// k∥ = n_eff k₀
    /// ```
    ///
    /// and:
    ///
    /// ```text
    /// k∥ = nᵢ k₀ sin(θ)
    /// ```
    fn multiply(self, rhs: Self) -> Self;
}

impl<C, D, P> CanonicalCoordinateJet for ArrayJet0<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn scale_real(self, factor: C::RealField) -> Self {
        ScalarAlgebra::scale(&self, C::from_real(factor))
    }

    fn reciprocal(self) -> Self {
        ScalarAlgebra::reciprocal(&self)
    }

    fn sin(self) -> Self {
        ScalarAlgebra::sin(&self)
    }

    fn multiply(self, rhs: Self) -> Self {
        ScalarAlgebra::multiply(&self, &rhs)
    }
}

impl<C, D, P> CanonicalCoordinateJet for ArrayJet1<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn scale_real(self, factor: C::RealField) -> Self {
        ScalarAlgebra::scale(&self, C::from_real(factor))
    }

    fn reciprocal(self) -> Self {
        ScalarAlgebra::reciprocal(&self)
    }

    fn sin(self) -> Self {
        ScalarAlgebra::sin(&self)
    }

    fn multiply(self, rhs: Self) -> Self {
        ScalarAlgebra::multiply(&self, &rhs)
    }
}

impl<C, D, P> CanonicalCoordinateJet for ArrayJet2<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn scale_real(self, factor: C::RealField) -> Self {
        ScalarAlgebra::scale(&self, C::from_real(factor))
    }

    fn reciprocal(self) -> Self {
        ScalarAlgebra::reciprocal(&self)
    }

    fn sin(self) -> Self {
        ScalarAlgebra::sin(&self)
    }

    fn multiply(self, rhs: Self) -> Self {
        ScalarAlgebra::multiply(&self, &rhs)
    }
}

impl<C, D, P> CanonicalCoordinateJet for ArrayJetBivariate1<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn scale_real(self, factor: C::RealField) -> Self {
        ScalarAlgebra::scale(&self, C::from_real(factor))
    }

    fn reciprocal(self) -> Self {
        ScalarAlgebra::reciprocal(&self)
    }

    fn sin(self) -> Self {
        ScalarAlgebra::sin(&self)
    }

    fn multiply(self, rhs: Self) -> Self {
        ScalarAlgebra::multiply(&self, &rhs)
    }
}

impl<C, D, P> CanonicalCoordinateJet for ArrayJetBivariate2<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn scale_real(self, factor: C::RealField) -> Self {
        ScalarAlgebra::scale(&self, C::from_real(factor))
    }

    fn reciprocal(self) -> Self {
        ScalarAlgebra::reciprocal(&self)
    }

    fn sin(self) -> Self {
        ScalarAlgebra::sin(&self)
    }

    fn multiply(self, rhs: Self) -> Self {
        ScalarAlgebra::multiply(&self, &rhs)
    }
}
