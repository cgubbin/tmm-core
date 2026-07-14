//! Shared boundary-response algebra for the 2×2 transfer backend.
//!
//! This module interprets transfer-matrix entries together with the
//! characteristic admittances of the two exterior media.
//!
//! For
//!
//! ```text
//! M = [A B]
//!     [C D]
//! ```
//!
//! define:
//!
//! ```text
//! u = A - B Y_R
//! v = C - D Y_R
//! Δ = Y_L u - v
//! ```
//!
//! `Δ` is both:
//!
//! - the common denominator of the plane-wave reflection and transmission
//!   amplitudes;
//! - an outgoing-mode characteristic residual.
//!
//! The algebra is generic over sampled arrays and first- or second-order jets.

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::{
        jet::{ArrayJet, ArrayJetFirst},
        transfer2::Matrix2,
    },
};

pub(super) type SampleArray<C, D> = ArrayBase<OwnedRepr<C>, D>;

/// Algebra required by the transfer-matrix amplitude expressions.
///
/// This is implemented by:
///
/// - sampled arrays for value-only evaluation;
/// - first-order sampled jets;
/// - second-order sampled jets.
///
/// Consequently, the physical reflection and transmission formula is written
/// only once.
pub(super) trait ResponseAlgebra<C, D>: Sized
where
    D: Dimension,
{
    /// Return the underlying sampled value.
    fn value(&self) -> &SampleArray<C, D>;

    /// Construct a constant with the same sampled shape as `source`.
    fn constant_like(source: &SampleArray<C, D>, value: C) -> Self;

    /// Add two values.
    fn add(&self, rhs: &Self) -> Self;

    /// Subtract `rhs` from this value.
    fn subtract(&self, rhs: &Self) -> Self;

    /// Multiply two values elementwise.
    fn multiply(&self, rhs: &Self) -> Self;

    /// Divide two values elementwise.
    fn divide(&self, rhs: &Self) -> Self;
}

impl<C, D> ResponseAlgebra<C, D> for ArrayBase<OwnedRepr<C>, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn value(&self) -> &SampleArray<C, D> {
        self
    }

    fn constant_like(source: &SampleArray<C, D>, value: C) -> Self {
        source.mapv(|_| value)
    }

    fn add(&self, rhs: &Self) -> Self {
        self.clone() + rhs.view()
    }

    fn subtract(&self, rhs: &Self) -> Self {
        self.clone() - rhs.view()
    }

    fn multiply(&self, rhs: &Self) -> Self {
        self.clone() * rhs.view()
    }

    fn divide(&self, rhs: &Self) -> Self {
        self.clone() / rhs.view()
    }
}

impl<C, D> ResponseAlgebra<C, D> for ArrayJetFirst<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArrayJetFirst::value(self)
    }

    fn constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        ArrayJetFirst::constant_like(source, value)
    }

    fn add(&self, rhs: &Self) -> Self {
        ArrayJetFirst::add(self, rhs)
    }

    fn subtract(&self, rhs: &Self) -> Self {
        ArrayJetFirst::subtract(self, rhs)
    }

    fn multiply(&self, rhs: &Self) -> Self {
        ArrayJetFirst::multiply(self, rhs)
    }

    fn divide(&self, rhs: &Self) -> Self {
        ArrayJetFirst::divide(self, rhs)
    }
}

impl<C, D> ResponseAlgebra<C, D> for ArrayJet<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArrayJet::value(self)
    }

    fn constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        ArrayJet::constant_like(source, value)
    }

    fn add(&self, rhs: &Self) -> Self {
        ArrayJet::add(self, rhs)
    }

    fn subtract(&self, rhs: &Self) -> Self {
        ArrayJet::subtract(self, rhs)
    }

    fn multiply(&self, rhs: &Self) -> Self {
        ArrayJet::multiply(self, rhs)
    }

    fn divide(&self, rhs: &Self) -> Self {
        ArrayJet::divide(self, rhs)
    }
}

pub(super) struct Matrix2Entries<A> {
    pub(super) m11: A,
    pub(super) m12: A,
    pub(super) m21: A,
    pub(super) m22: A,
}

pub(super) struct TransferBoundaryTerms<A> {
    pub(super) u: A,
    pub(super) v: A,
    pub(super) b_yr: A,
    pub(super) d_yr: A,
    pub(super) denominator: A,
}

impl<A> Matrix2Entries<A> {
    pub(super) fn boundary_terms<C, D>(
        &self,
        left_admittance: &A,
        right_admittance: &A,
    ) -> TransferBoundaryTerms<A>
    where
        C: ComplexScalar,
        D: Dimension,
        A: ResponseAlgebra<C, D>,
    {
        let Matrix2Entries {
            m11: a,
            m12: b,
            m21: c,
            m22: d,
        } = self;

        let b_yr = b.multiply(right_admittance);
        let d_yr = d.multiply(right_admittance);

        let u = a.subtract(&b_yr);
        let v = c.subtract(&d_yr);

        let denominator = left_admittance.multiply(&u).subtract(&v);

        TransferBoundaryTerms {
            u,
            v,
            b_yr,
            d_yr,
            denominator,
        }
    }
}

pub(super) fn outgoing_residual<C, D, A>(
    matrix: Matrix2Entries<A>,
    left_admittance: &A,
    right_admittance: &A,
) -> A
where
    C: ComplexScalar,
    D: Dimension,
    A: ResponseAlgebra<C, D>,
{
    matrix
        .boundary_terms::<C, D>(left_admittance, right_admittance)
        .denominator
}

impl<C, D> Matrix2<C, D>
where
    D: Dimension,
{
    /// Consume the matrix and move its entries into the response algebra.
    pub(super) fn into_entries(self) -> Matrix2Entries<SampleArray<C, D>> {
        let (m11, m12, m21, m22) = self.into_parts();

        Matrix2Entries { m11, m12, m21, m22 }
    }
}
