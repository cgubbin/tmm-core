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
        algebra::ScalarAlgebra,
        transfer2::{Matrix2, plane_wave::boundary_slope},
    },
};

pub(super) struct Matrix2Entries<A> {
    pub(super) m11: A,
    pub(super) m12: A,
    pub(super) m21: A,
    pub(super) m22: A,
}

pub(super) struct TransferBoundaryTerms<A> {
    pub(super) u: A,
    pub(super) v: A,
    pub(super) b_right: A,
    pub(super) d_right: A,
    pub(super) denominator: A,
}

impl<A> Matrix2Entries<A> {
    pub(super) fn boundary_terms<C, D>(
        &self,
        left_slope: &A,
        right_slope: &A,
    ) -> TransferBoundaryTerms<A>
    where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D>,
    {
        let Matrix2Entries { m11, m12, m21, m22 } = self;

        let b_right = m12.multiply(right_slope);

        let d_right = m22.multiply(right_slope);

        let u = m11.subtract(&b_right);

        let v = m21.subtract(&d_right);

        let denominator = left_slope.multiply(&u).subtract(&v);

        TransferBoundaryTerms {
            b_right,
            d_right,
            u,
            v,
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
    A: ScalarAlgebra<C, D>,
{
    let left_slope = boundary_slope::<C, D, A>(left_admittance);

    let right_slope = boundary_slope::<C, D, A>(right_admittance);

    matrix.boundary_terms(&left_slope, &right_slope).denominator
}

impl<C, D> Matrix2<C, D>
where
    D: Dimension,
{
    /// Consume the matrix and move its entries into the response algebra.
    pub(super) fn into_entries(self) -> Matrix2Entries<ArrayBase<OwnedRepr<C>, D>> {
        let (m11, m12, m21, m22) = self.into_parts();

        Matrix2Entries { m11, m12, m21, m22 }
    }
}
