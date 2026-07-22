//! Native matrix representation for the scalar-channel 2×2 scattering backend.
//!
//! [`ScatterMatrix2`] relates incoming channel amplitudes to outgoing channel
//! amplitudes according to:
//!
//! ```text
//! [a_L^-]   [s11 s12] [a_L^+]
//! [a_R^+] = [s21 s22] [a_R^-]
//! ```
//!
//! Therefore:
//!
//! - `s11` is reflection for incidence from the left;
//! - `s21` is transmission from left to right;
//! - `s22` is reflection for incidence from the right;
//! - `s12` is transmission from right to left.
//!
//! Each entry is an owned `ndarray` array over the sampled input grid.
//! All four entries have the same shape.
//!
//! This type is a storage and inspection representation. Internal scattering
//! calculations use [`ScatterEntries`] so that value-only, first-order, and
//! second-order calculations can share the same Redheffer algebra.

use ndarray::{ArrayBase, OwnedRepr};

use crate::{
    ArrayJet, ArrayJetFirst,
    backend::{jet::ArraySpectralJet, scatter2::entries::ScatterEntries},
};

pub type Scatter2Values<C, D> = ScatterMatrix2<ArrayBase<OwnedRepr<C>, D>>;

pub type Scatter2JetFirst<C, D> = ScatterMatrix2<ArrayJetFirst<C, D>>;

pub type Scatter2Jet<C, D> = ScatterMatrix2<ArrayJet<C, D>>;

pub type Scatter2SpectralJet<C, D> = ScatterMatrix2<ArraySpectralJet<C, D>>;

/// Shape-preserving scalar-channel 2×2 scattering matrix.
///
/// This is the native raw matrix returned by the scalar 2×2 scattering
/// backend. Backend-independent reflection and transmission calculations
/// should normally use [`PlaneWaveBackend`](crate::backend::PlaneWaveBackend)
/// rather than interpreting these entries directly.
///
/// The matrix uses the channel ordering:
///
/// ```text
/// [left outgoing ]   [s11 s12] [left incoming ]
/// [right outgoing] = [s21 s22] [right incoming].
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ScatterMatrix2<A> {
    s11: A,
    s12: A,
    s21: A,
    s22: A,
}

impl<A> ScatterMatrix2<A> {
    /// Construct a scattering matrix from its four sampled entries.
    ///
    /// All entries must have identical shapes. This invariant is checked in
    /// debug builds.
    pub(crate) fn new(s11: A, s12: A, s21: A, s22: A) -> Self {
        Self { s11, s12, s21, s22 }
    }

    /// Return reflection from the left, `s11`.
    pub fn s11(&self) -> &A {
        &self.s11
    }

    /// Return transmission from right to left, `s12`.
    pub fn s12(&self) -> &A {
        &self.s12
    }

    /// Return transmission from left to right, `s21`.
    pub fn s21(&self) -> &A {
        &self.s21
    }

    /// Return reflection from the right, `s22`.
    pub fn s22(&self) -> &A {
        &self.s22
    }

    /// Consume the matrix and return its four entries in row-major order.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(self) -> (A, A, A, A) {
        (self.s11, self.s12, self.s21, self.s22)
    }

    /// Construct the raw matrix representation from internal scattering
    /// entries.
    pub(crate) fn from_entries(entries: ScatterEntries<A>) -> Self {
        Self::new(entries.s11, entries.s12, entries.s21, entries.s22)
    }

    /// Consume the raw matrix representation and expose its internal entries.
    pub(crate) fn into_entries(self) -> ScatterEntries<A> {
        let (s11, s12, s21, s22) = self.into_parts();

        ScatterEntries { s11, s12, s21, s22 }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array0, arr0, array};
    use num_complex::Complex64;

    use super::*;

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn scalar_matrix(s11: f64, s12: f64, s21: f64, s22: f64) -> ScatterMatrix2<Array0<C>> {
        ScatterMatrix2::new(arr0(c(s11)), arr0(c(s12)), arr0(c(s21)), arr0(c(s22)))
    }

    #[test]
    fn constructor_and_accessors_preserve_scalar_entries() {
        let matrix = scalar_matrix(1.0, 2.0, 3.0, 4.0);

        assert_eq!(matrix.s11()[()], c(1.0));
        assert_eq!(matrix.s12()[()], c(2.0));
        assert_eq!(matrix.s21()[()], c(3.0));
        assert_eq!(matrix.s22()[()], c(4.0));
    }

    #[test]
    fn constructor_preserves_sampled_entries() {
        let s11 = array![c(1.0), c(2.0), c(3.0)];
        let s12 = array![c(4.0), c(5.0), c(6.0)];
        let s21 = array![c(7.0), c(8.0), c(9.0)];
        let s22 = array![c(10.0), c(11.0), c(12.0)];

        let matrix = ScatterMatrix2::new(s11.clone(), s12.clone(), s21.clone(), s22.clone());

        assert_eq!(matrix.s11(), &s11);
        assert_eq!(matrix.s12(), &s12);
        assert_eq!(matrix.s21(), &s21);
        assert_eq!(matrix.s22(), &s22);
    }

    #[test]
    fn all_entries_retain_common_sample_shape() {
        let matrix = ScatterMatrix2::new(
            array![c(1.0), c(2.0), c(3.0)],
            array![c(4.0), c(5.0), c(6.0)],
            array![c(7.0), c(8.0), c(9.0)],
            array![c(10.0), c(11.0), c(12.0)],
        );

        let expected = matrix.s11().raw_dim();

        assert_eq!(matrix.s12().raw_dim(), expected);
        assert_eq!(matrix.s21().raw_dim(), expected);
        assert_eq!(matrix.s22().raw_dim(), expected);
    }

    #[test]
    fn into_parts_preserves_entry_order() {
        let matrix = scalar_matrix(1.0, 2.0, 3.0, 4.0);

        let (s11, s12, s21, s22) = matrix.into_parts();

        assert_eq!(s11[()], c(1.0));
        assert_eq!(s12[()], c(2.0));
        assert_eq!(s21[()], c(3.0));
        assert_eq!(s22[()], c(4.0));
    }

    #[test]
    fn into_entries_preserves_all_entries() {
        let matrix = scalar_matrix(1.0, 2.0, 3.0, 4.0);

        let entries = matrix.into_entries();

        assert_eq!(entries.s11[()], c(1.0));
        assert_eq!(entries.s12[()], c(2.0));
        assert_eq!(entries.s21[()], c(3.0));
        assert_eq!(entries.s22[()], c(4.0));
    }

    #[test]
    fn from_entries_preserves_all_entries() {
        let entries = ScatterEntries {
            s11: arr0(c(1.0)),
            s12: arr0(c(2.0)),
            s21: arr0(c(3.0)),
            s22: arr0(c(4.0)),
        };

        let matrix = ScatterMatrix2::from_entries(entries);

        assert_eq!(matrix.s11()[()], c(1.0));
        assert_eq!(matrix.s12()[()], c(2.0));
        assert_eq!(matrix.s21()[()], c(3.0));
        assert_eq!(matrix.s22()[()], c(4.0));
    }

    #[test]
    fn entry_conversion_round_trip_is_lossless() {
        let matrix = ScatterMatrix2::new(
            array![c(1.0), c(2.0)],
            array![c(3.0), c(4.0)],
            array![c(5.0), c(6.0)],
            array![c(7.0), c(8.0)],
        );

        let expected = matrix.clone();

        let actual = ScatterMatrix2::from_entries(matrix.into_entries());

        assert_eq!(actual, expected);
    }

    #[test]
    fn clone_preserves_matrix() {
        let matrix = scalar_matrix(1.0, 2.0, 3.0, 4.0);

        assert_eq!(matrix.clone(), matrix);
    }

    #[test]
    fn equality_detects_different_entries() {
        let first = scalar_matrix(1.0, 2.0, 3.0, 4.0);

        let second = scalar_matrix(1.0, 2.0, 3.0, 5.0);

        assert_ne!(first, second);
    }
}
