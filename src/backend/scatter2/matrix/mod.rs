//! Scalar-channel 2×2 scattering matrices.
//!
//! [`ScatterMatrix2`] relates incoming channel amplitudes to outgoing channel
//! amplitudes:
//!
//! ```text
//! [a_L^-]   [s11 s12] [a_L^+]
//! [a_R^+] = [s21 s22] [a_R^-]
//! ```
//!
//! Each matrix element is an `ndarray` array over the sampled input grid.
//! Composition uses the Redheffer star product rather than ordinary matrix
//! multiplication.

mod algebra;
mod interface;
mod propagation;

pub(crate) use algebra::star_product;

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::ComplexScalar;

#[derive(Clone, Debug, PartialEq)]
pub struct ScatterMatrix2<C, D>
where
    D: Dimension,
{
    s11: ArrayBase<OwnedRepr<C>, D>,
    s12: ArrayBase<OwnedRepr<C>, D>,
    s21: ArrayBase<OwnedRepr<C>, D>,
    s22: ArrayBase<OwnedRepr<C>, D>,
}

impl<C, D> ScatterMatrix2<C, D>
where
    D: Dimension,
{
    pub fn new(
        s11: ArrayBase<OwnedRepr<C>, D>,
        s12: ArrayBase<OwnedRepr<C>, D>,
        s21: ArrayBase<OwnedRepr<C>, D>,
        s22: ArrayBase<OwnedRepr<C>, D>,
    ) -> Self {
        Self { s11, s12, s21, s22 }
    }

    pub fn s11(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.s11
    }

    pub fn s12(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.s12
    }

    pub fn s21(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.s21
    }

    pub fn s22(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.s22
    }

    /// Construct the transparent two-port identity.
    ///
    /// This matrix transmits both directions without reflection:
    ///
    /// ```text
    /// S_identity = [0 1]
    ///              [1 0]
    /// ```
    pub fn identity_like(shape_source: &ArrayBase<OwnedRepr<C>, D>) -> Self
    where
        C: ComplexScalar,
    {
        let zero = shape_source.mapv(|_| C::zero());
        let one = shape_source.mapv(|_| C::one());

        Self::new(zero.clone(), one.clone(), one, zero)
    }

    pub fn zeros_like(shape_source: &ArrayBase<OwnedRepr<C>, D>) -> Self
    where
        C: ComplexScalar,
    {
        let zero = shape_source.mapv(|_| C::zero());

        Self::new(zero.clone(), zero.clone(), zero.clone(), zero)
    }

    pub fn scale_by_array(&self, values: &ArrayBase<OwnedRepr<C>, D>) -> Self
    where
        C: ComplexScalar,
    {
        Self::new(
            self.s11.clone() * values.view(),
            self.s12.clone() * values.view(),
            self.s21.clone() * values.view(),
            self.s22.clone() * values.view(),
        )
    }
}
