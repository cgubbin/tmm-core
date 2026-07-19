//! Differential algebra for the 2×2 transfer-matrix backend.
//!
//! This module connects [`TransferMatrix2`] to the generic jet infrastructure.
//! Ordinary transfer-matrix composition is bilinear, so first- and
//! second-order matrix jets can use the standard product rules implemented by
//! [`JetFirst`] and [`Jet`].
//!
//! It also converts matrix-valued jets into entry-wise array jets for the
//! backend-specific plane-wave amplitude calculation. This conversion consumes
//! the matrix jet and moves its arrays into scalar jets without cloning them.
//!
//! The plane-wave conversion includes the admittances of the two exterior
//! media. For a transfer matrix
//!
//! ```text
//! M = [A B]
//!     [C D]
//! ```
//!
//! and exterior admittances `Y_L` and `Y_R`, define
//!
//! ```text
//! u = A - B Y_R
//! v = C - D Y_R
//! Δ = Y_L u - v
//! ```
//!
//! For incidence from the left,
//!
//! ```text
//! r_L = (Y_L u + v) / Δ
//! t_L = 2 Y_L / Δ
//! ```
//!
//! For incidence from the right,
//!
//! ```text
//! p = A + B Y_R
//! q = C + D Y_R
//! r_R = (q - Y_L p) / Δ
//! t_R = 2 Y_R det(M) / Δ
//! ```
//!
//! All operations are performed on jets, so the same expressions generate
//! value, first-derivative, and second-derivative results.

use crate::{
    ComplexScalar,
    backend::{
        jet::{ChainRuleScale, Jet, JetAdditive, JetBilinear, JetFirst, JetZeroLike},
        transfer2::TransferMatrix2,
    },
};

use ndarray::{ArrayBase, Dimension, OwnedRepr};

/// Second order [`Jet`] of 2x2 transfer matrix
pub(crate) type Transfer2Jet<C, D> = Jet<TransferMatrix2<C, D>>;

/// First order [`JetFirst`] of 2x2 transfer matrix
pub(crate) type Transfer2JetFirst<C, D> = JetFirst<TransferMatrix2<C, D>>;

impl<C, D> JetZeroLike for TransferMatrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn jet_zeros_like(shape_source: &Self) -> Self {
        Self::zeros_like(shape_source.m11())
    }
}

impl<C, D> JetAdditive for TransferMatrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn jet_add(&self, rhs: &Self) -> Self {
        self + rhs
    }

    fn jet_negate(&self) -> Self {
        Self::new(
            -self.m11().clone(),
            -self.m12().clone(),
            -self.m21().clone(),
            -self.m22().clone(),
        )
    }

    fn jet_subtract(&self, rhs: &Self) -> Self {
        self - rhs
    }
}

impl<C, D> JetBilinear for TransferMatrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn jet_multiply(&self, rhs: &Self) -> Self {
        self * rhs
    }

    fn jet_double(&self) -> Self {
        self * (C::one() + C::one())
    }
}

impl<C, D> ChainRuleScale<ArrayBase<OwnedRepr<C>, D>> for TransferMatrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn scale_by(&self, coefficient: &ArrayBase<OwnedRepr<C>, D>) -> Self {
        self.scale_by_array(coefficient)
    }
}
