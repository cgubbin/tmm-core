//! Shape-preserving 2×2 transfer matrices.
//!
//! This module provides an entry representation:
//!
//! [`Transfer2Entries`] is an internal algebraic representation used while
//!   evaluating value-only, first-order, and second-order transfer matrices.
//!
//! [`Transfer2Entries`] is generic over its scalar-entry representation. Its
//! entries may therefore be sampled arrays, first-order jets, or second-order
//! jets.
//!
//! Transfer matrices compose by ordinary matrix multiplication. Each layer
//! matrix maps the state at its right boundary to the state at its left
//! boundary:
//!
//! ```text
//! ψ_left = L ψ_right.
//! ```
//!
//! Layers are appended in physical left-to-right order. If the accumulated
//! matrix contains the layers already encountered and `L` is the next layer to
//! the right, accumulation is:
//!
//! ```text
//! M_total <- M_total L.
//! ```
//!
//! Consequently, for layers `L₀, L₁, …, Lₙ` stored from left to right,
//!
//! ```text
//! ψ_left = L₀ L₁ … Lₙ ψ_right.
//! ```
//!
//! Layer-specific construction is implemented in the private [`layer`] module.

use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, IntoDimension, OwnedRepr};
use num_traits::{One, Zero};

use crate::{
    ComplexScalar, Polarisation,
    algebra::{
        ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, ScalarAlgebra,
    },
    backend::{
        PlaneWaveEntries, isotropic::IsotropicLayerQuantities, transfer2::error::Transfer2Entry,
    },
    input::CanonicalCoordinates,
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};

/// Zero-order entry-wise transfer representation.
pub(crate) type Transfer2Jet0<C, D, P> = Transfer2Entries<ArrayJet0<C, D, P>>;

/// First-order entry-wise transfer representation.
pub(crate) type Transfer2Jet1<C, D, P> = Transfer2Entries<ArrayJet1<C, D, P>>;

/// Second-order entry-wise transfer representation.
pub(crate) type Transfer2Jet2<C, D, P> = Transfer2Entries<ArrayJet2<C, D, P>>;

/// Bivariate entrywise transfer representation
pub(crate) type Transfer2JetBivariate1<C, D, P> = Transfer2Entries<ArrayJetBivariate1<C, D, P>>;

/// Second-order bivariate entrywise transfer representation
pub(crate) type Transfer2JetBivariate2<C, D, P> = Transfer2Entries<ArrayJetBivariate2<C, D, P>>;

/// Internal algebraic representation of a 2×2 transfer matrix.
///
/// `A` may be:
///
/// - an owned sampled array;
/// - a first-order scalar jet;
/// - a second-order scalar jet.
///
/// All four entries must share the same sampled shape and derivative
/// representation.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct Transfer2Entries<A> {
    pub(super) m11: A,
    pub(super) m12: A,
    pub(super) m21: A,
    pub(super) m22: A,
}

impl<A> Transfer2Entries<A> {
    /// Construct an internal matrix from its four entries.
    pub(crate) fn new(m11: A, m12: A, m21: A, m22: A) -> Self {
        Self { m11, m12, m21, m22 }
    }

    /// Return entry `(1, 1)`.
    pub(crate) fn m11(&self) -> &A {
        &self.m11
    }

    /// Return entry `(1, 2)`.
    pub(crate) fn m12(&self) -> &A {
        &self.m12
    }

    /// Return entry `(2, 1)`.
    pub(crate) fn m21(&self) -> &A {
        &self.m21
    }

    /// Return entry `(2, 2)`.
    pub(crate) fn m22(&self) -> &A {
        &self.m22
    }

    /// Consume the matrix and return its entries in row-major order.
    pub(crate) fn into_parts(self) -> (A, A, A, A) {
        (self.m11, self.m12, self.m21, self.m22)
    }

    pub(crate) fn first_non_finite(&self) -> Option<(Transfer2Entry, Vec<usize>)>
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexField,
        A::Dimension: Dimension,
    {
        [
            (Transfer2Entry::M11, self.m11.value()),
            (Transfer2Entry::M12, self.m12.value()),
            (Transfer2Entry::M21, self.m21.value()),
            (Transfer2Entry::M22, self.m22.value()),
        ]
        .into_iter()
        .find_map(|(entry, values)| {
            values.indexed_iter().find_map(|(index, value)| {
                (!value.is_finite()).then(|| {
                    (
                        entry,
                        index.into_dimension().as_array_view().to_owned().to_vec(),
                    )
                })
            })
        })
    }
}

impl<A> Transfer2Entries<A> {
    pub(crate) fn identity_like(source: &ArrayBase<OwnedRepr<A::Scalar>, A::Dimension>) -> Self
    where
        A: ScalarAlgebra + Clone,
        A::Scalar: ComplexField + One + Zero,
    {
        let zero = A::filled_constant_like(source, <A::Scalar as Zero>::zero());
        let one = A::filled_constant_like(source, <A::Scalar as One>::one());

        Self::new(one.clone(), zero.clone(), zero, one)
    }

    /// Multiply two matrices using the supplied scalar algebra.
    ///
    /// This computes `self * rhs`:
    ///
    /// ```text
    /// [a b] [e f]   [ae + bg  af + bh]
    /// [c d] [g h] = [ce + dg  cf + dh]
    /// ```
    ///
    /// When `A` is a jet representation, all matrix-product derivatives are
    /// generated by the scalar product and sum rules.
    pub(crate) fn multiply(&self, rhs: &Self) -> Self
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexField,
        A::Dimension: Dimension,
    {
        let m11 = self
            .m11
            .multiply(&rhs.m11)
            .add(&self.m12.multiply(&rhs.m21));

        let m12 = self
            .m11
            .multiply(&rhs.m12)
            .add(&self.m12.multiply(&rhs.m22));

        let m21 = self
            .m21
            .multiply(&rhs.m11)
            .add(&self.m22.multiply(&rhs.m21));

        let m22 = self
            .m21
            .multiply(&rhs.m12)
            .add(&self.m22.multiply(&rhs.m22));

        Self::new(m11, m12, m21, m22)
    }

    /// Compute the determinant using the supplied scalar algebra.
    ///
    /// The determinant is:
    ///
    /// ```text
    /// det(M) = m11 m22 - m12 m21.
    /// ```
    pub(crate) fn determinant(&self) -> A
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexField,
        A::Dimension: Dimension,
    {
        self.m11
            .multiply(&self.m22)
            .subtract(&self.m12.multiply(&self.m21))
    }

    pub(crate) fn from_layer(quantities: &IsotropicLayerQuantities<A>, thickness: &A) -> Self
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexField + One,
        A::Dimension: Dimension,
    {
        let theta = quantities.kappa().multiply(thickness);

        let sin = theta.sin();
        let cos = theta.cos();

        let factor_over_kappa = quantities.factor().divide(quantities.kappa());

        let kappa_over_factor = quantities.kappa().divide(quantities.factor());

        Self::new(
            cos.clone(),
            sin.multiply(&factor_over_kappa)
                .scale(-<A::Scalar as One>::one()),
            sin.multiply(&kappa_over_factor),
            cos,
        )
    }
}

#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct Transfer2ExteriorContext<A> {
    left_quantities: IsotropicLayerQuantities<A>,
    right_quantities: IsotropicLayerQuantities<A>,
}

impl<J> Transfer2ExteriorContext<J> {
    pub(super) fn new<E, M>(
        coordinates: &CanonicalCoordinates<J>,
        left_exterior: &M,
        right_exterior: &M,
        polarisation: Polarisation,
    ) -> Self
    where
        J: ScalarAlgebra + ConstitutiveLift<E, M> + Clone,
        J::Scalar: ComplexScalar,
        J::Dimension: Dimension,
        E: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
    {
        let left_quantities =
            IsotropicLayerQuantities::evaluate::<E, M>(left_exterior, coordinates, polarisation);

        let right_quantities =
            IsotropicLayerQuantities::evaluate::<E, M>(right_exterior, coordinates, polarisation);

        Self {
            left_quantities,
            right_quantities,
        }
    }

    pub(crate) fn left_quantities(&self) -> &IsotropicLayerQuantities<J> {
        &self.left_quantities
    }

    pub(crate) fn right_quantities(&self) -> &IsotropicLayerQuantities<J> {
        &self.right_quantities
    }
}

impl<A> PlaneWaveEntries for Transfer2Entries<A> {
    type ExteriorContext = Transfer2ExteriorContext<A>;
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array0, Ix0, arr0};
    use num_complex::Complex64;

    use super::*;
    use crate::{
        algebra::{ArrayJet0, Jet0, RealParameter},
        input::{CanonicalCoordinates, Polarisation},
        test_support::{
            C, TOLERANCE,
            assertions::assert_complex_close,
            c,
            jet::zero_jet_from_value,
            materials::{constant, linear},
        },
    };

    type A = ArrayJet0<C, Ix0, RealParameter>;

    fn entries(m11: C, m12: C, m21: C, m22: C) -> Transfer2Entries<A> {
        Transfer2Entries::new(
            zero_jet_from_value(m11),
            zero_jet_from_value(m12),
            zero_jet_from_value(m21),
            zero_jet_from_value(m22),
        )
    }

    fn coordinates(k0: f64, k_parallel: f64) -> CanonicalCoordinates<A> {
        CanonicalCoordinates::new(
            zero_jet_from_value(c(k0)),
            zero_jet_from_value(c(k_parallel)),
        )
    }

    #[test]
    fn identity_has_unit_diagonal_and_zero_off_diagonal() {
        let identity = Transfer2Entries::<A>::identity_like(&arr0(c(0.0)));

        assert_complex_close(identity.m11()[()], c(1.0), TOLERANCE);
        assert_complex_close(identity.m12()[()], c(0.0), TOLERANCE);
        assert_complex_close(identity.m21()[()], c(0.0), TOLERANCE);
        assert_complex_close(identity.m22()[()], c(1.0), TOLERANCE);
    }

    #[test]
    fn identity_is_neutral_on_both_sides() {
        let matrix = entries(c(1.0), c(2.0), c(3.0), c(4.0));

        let identity = Transfer2Entries::<A>::identity_like(&arr0(c(0.0)));

        assert_eq!(identity.multiply(&matrix), matrix,);

        assert_eq!(matrix.multiply(&identity), matrix,);
    }

    #[test]
    fn multiplication_uses_standard_matrix_order() {
        let left = entries(c(1.0), c(2.0), c(3.0), c(4.0));

        let right = entries(c(5.0), c(6.0), c(7.0), c(8.0));

        let product = left.multiply(&right);

        assert_complex_close(product.m11()[()], c(1.0 * 5.0 + 2.0 * 7.0), TOLERANCE);
        assert_complex_close(product.m12()[()], c(1.0 * 6.0 + 2.0 * 8.0), TOLERANCE);
        assert_complex_close(product.m21()[()], c(3.0 * 5.0 + 4.0 * 7.0), TOLERANCE);
        assert_complex_close(product.m22()[()], c(3.0 * 6.0 + 4.0 * 8.0), TOLERANCE);
    }

    #[test]
    fn multiplication_is_not_accidentally_reversed() {
        let left = entries(c(1.0), c(2.0), c(0.0), c(1.0));

        let right = entries(c(1.0), c(0.0), c(3.0), c(1.0));

        assert_ne!(left.multiply(&right), right.multiply(&left),);
    }

    #[test]
    fn determinant_uses_standard_formula() {
        let matrix = entries(c(2.0), c(3.0), c(5.0), c(7.0));

        assert_complex_close(
            matrix.determinant()[()],
            c(2.0 * 7.0 - 3.0 * 5.0),
            TOLERANCE,
        );
    }

    #[test]
    fn zero_thickness_layer_is_identity() {
        let material = constant(4.0, 1.0);
        let coordinates = coordinates(2.0, 0.3);

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
        );

        let matrix = Transfer2Entries::from_layer(&quantities, &zero_jet_from_value(c(0.0)));

        let identity = Transfer2Entries::<A>::identity_like(&arr0(c(0.0)));

        assert_eq!(matrix, identity);
    }

    #[test]
    fn homogeneous_layer_matches_analytic_entries() {
        let material = constant(4.0, 1.0);
        let coordinates = coordinates(2.0, 0.0);

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
        );

        let thickness = 0.2;

        let matrix = Transfer2Entries::from_layer(&quantities, &zero_jet_from_value(c(thickness)));

        let kappa = 4.0;
        let admittance = 4.0;
        let theta = kappa * thickness;

        assert_complex_close(matrix.m11()[()], c(theta.cos()), TOLERANCE);

        assert_complex_close(matrix.m12()[()], c(-theta.sin() / admittance), TOLERANCE);

        assert_complex_close(matrix.m21()[()], c(admittance * theta.sin()), TOLERANCE);

        assert_complex_close(matrix.m22()[()], c(theta.cos()), TOLERANCE);
    }

    #[test]
    fn homogeneous_layer_has_unit_determinant() {
        let material = constant(3.0, 1.5);
        let coordinates = coordinates(2.0, 0.4);

        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            let quantities =
                IsotropicLayerQuantities::real_axis(&material, &coordinates, polarisation);

            let matrix = Transfer2Entries::from_layer(&quantities, &zero_jet_from_value(c(0.37)));

            assert_complex_close(matrix.determinant()[()], c(1.0), TOLERANCE);
        }
    }

    #[test]
    fn te_and_tm_use_their_respective_admittances() {
        let material = constant(4.0, 2.0);
        let coordinates = coordinates(3.0, 1.0);
        let thickness = zero_jet_from_value(c(0.1));

        let te = Transfer2Entries::from_layer(
            &IsotropicLayerQuantities::real_axis(
                &material,
                &coordinates,
                Polarisation::TransverseElectric,
            ),
            &thickness,
        );

        let tm = Transfer2Entries::from_layer(
            &IsotropicLayerQuantities::real_axis(
                &material,
                &coordinates,
                Polarisation::TransverseMagnetic,
            ),
            &thickness,
        );

        assert_ne!(te.m12(), tm.m12());
        assert_ne!(te.m21(), tm.m21());
    }

    #[test]
    fn dispersive_material_changes_layer_matrix() {
        let material = linear(2.0, 0.4, 1.0, 0.1);

        let first = Transfer2Entries::from_layer(
            &IsotropicLayerQuantities::real_axis(
                &material,
                &coordinates(2.0, 0.2),
                Polarisation::TransverseElectric,
            ),
            &zero_jet_from_value(c(0.1)),
        );

        let second = Transfer2Entries::from_layer(
            &IsotropicLayerQuantities::real_axis(
                &material,
                &coordinates(3.0, 0.2),
                Polarisation::TransverseElectric,
            ),
            &zero_jet_from_value(c(0.1)),
        );

        assert_ne!(first, second);
    }

    #[test]
    fn detects_non_finite_entry_and_index() {
        let matrix = Transfer2Entries::new(
            zero_jet_from_value(c(1.0)),
            zero_jet_from_value(C::new(f64::INFINITY, 0.0)),
            zero_jet_from_value(c(0.0)),
            zero_jet_from_value(c(1.0)),
        );

        assert_eq!(
            matrix.first_non_finite(),
            Some((Transfer2Entry::M12, Vec::new(),)),
        );
    }
}
