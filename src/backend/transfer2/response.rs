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

use ndarray::{Array0, Dimension, Ix0};

use crate::{
    ComplexScalar, IncidentSide,
    backend::{
        algebra::ScalarAlgebra,
        mode::OutgoingModeAmplitudes,
        transfer2::{matrix::Matrix2Entries, plane_wave::boundary_slope},
    },
};

pub(super) struct TransferBoundaryTerms<A> {
    pub(super) u: A,
    pub(super) v: A,
    pub(super) b_right: A,
    pub(super) d_right: A,
    pub(super) denominator: A,
}

pub(super) fn outgoing_residual<C, D, A>(
    matrix: &Matrix2Entries<A>,
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

pub(crate) struct RightOutgoingState<A> {
    pub(crate) field: A,
    pub(crate) slope: A,
}

pub(crate) struct OutgoingModeExtraction<A> {
    pub(crate) incident_side: IncidentSide,
    pub(crate) scale: A,
    pub(crate) left_outgoing: A,
    pub(crate) right_outgoing: A,
}

impl<C> OutgoingModeExtraction<Array0<C>>
where
    C: ComplexScalar,
{
    pub(crate) fn amplitudes(&self) -> OutgoingModeAmplitudes<C> {
        OutgoingModeAmplitudes::normalised(self.left_outgoing[()], self.right_outgoing[()])
    }
}

impl<A> Matrix2Entries<A> {
    pub(crate) fn boundary_terms<C, D>(
        &self,
        left_slope: &A,
        right_slope: &A,
    ) -> TransferBoundaryTerms<A>
    where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D>,
    {
        let Matrix2Entries { m12, m22, .. } = self;

        let b_right = m12.multiply(right_slope);
        let d_right = m22.multiply(right_slope);

        let right_state = self.apply_right_outgoing::<C, D>(right_slope);

        let denominator = left_slope
            .multiply(&right_state.field)
            .subtract(&right_state.slope);

        TransferBoundaryTerms {
            b_right,
            d_right,
            u: right_state.field,
            v: right_state.slope,
            denominator,
        }
    }

    /// Apply the matrix to the unit right-outgoing boundary state.
    ///
    /// For right exterior slope `ξR`, this computes:
    ///
    /// ```text
    /// u = A - B ξR
    /// v = C - D ξR.
    /// ```
    pub(crate) fn apply_right_outgoing<C, D>(&self, right_slope: &A) -> RightOutgoingState<A>
    where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D>,
    {
        let field = self.m11.subtract(&self.m12.multiply(right_slope));

        let slope = self.m21.subtract(&self.m22.multiply(right_slope));

        RightOutgoingState { field, slope }
    }
}

impl<C> Matrix2Entries<Array0<C>>
where
    C: ComplexScalar,
{
    pub(crate) fn outgoing_mode_extraction(
        &self,
        right_admittance: &Array0<C>,
    ) -> OutgoingModeExtraction<Array0<C>> {
        let right_slope = boundary_slope::<C, ndarray::Ix0, _>(right_admittance);

        let state = self.apply_right_outgoing::<C, ndarray::Ix0>(&right_slope);

        /*
         * The unnormalised outgoing amplitudes are:
         *
         * left  = u,
         * right = 1.
         */
        let norm_squared = state.field[()].modulus_squared() + C::one().modulus_squared();

        let scale_value = C::from_real(norm_squared).sqrt().recip();

        let scale = ndarray::arr0(scale_value);

        OutgoingModeExtraction {
            incident_side: IncidentSide::Right,

            left_outgoing: state.field * scale.clone(),

            right_outgoing: scale.clone(),

            scale,
        }
    }
}
