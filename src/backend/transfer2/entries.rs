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
    ComplexScalar, IncidentSide, PlaneWaveAmplitudes, PlaneWaveDeterminant, PlaneWavePower,
    Polarisation,
    algebra::{
        ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, Jet,
        RealScalarAlgebra, ScalarAlgebra,
    },
    backend::{
        ExteriorAdmittanceProvider, PlaneWaveEntries, PlaneWaveModeCandidate,
        isotropic::IsotropicLayerQuantities, transfer2::error::Transfer2Entry,
    },
    input::CanonicalCoordinates,
    material::{ConstitutiveEvaluator, ConstitutiveLift},
    observable::{BoundaryState, ProjectAmplitudes, ProjectPlaneWaveModeDeterminant, ProjectPower},
};

use super::{
    projection::{right_incoming_column, right_outgoing_column},
    state::transfer_state_slope,
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

    pub(super) fn sample_source(&self) -> &ArrayBase<OwnedRepr<A::Scalar>, A::Dimension>
    where
        A: ScalarAlgebra,
    {
        self.m11().value()
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
    left_admittance: A,
    right_admittance: A,
}

impl<A> ExteriorAdmittanceProvider for Transfer2ExteriorContext<A> {
    type Algebra = A;

    fn left_admittance(&self) -> &A {
        &self.left_admittance
    }

    fn right_admittance(&self) -> &A {
        &self.right_admittance
    }
}

impl<J> Transfer2ExteriorContext<J> {
    pub(crate) fn from_parts(left_admittance: J, right_admittance: J) -> Self {
        Self {
            left_admittance,
            right_admittance,
        }
    }

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
            left_admittance: left_quantities.into_admittance().into_inner(),
            right_admittance: right_quantities.into_admittance().into_inner(),
        }
    }
}

impl<A> PlaneWaveEntries for Transfer2Entries<A> {
    type ExteriorContext = Transfer2ExteriorContext<A>;
    type Algebra = A;
}

impl<J> ProjectAmplitudes for Transfer2Entries<J>
where
    J: ScalarAlgebra,
    J::Scalar: ComplexScalar,
    J::Dimension: Dimension,
{
    type Amplitudes = PlaneWaveAmplitudes<J>;

    fn project_amplitudes(
        &self,
        exterior: &Self::ExteriorContext,
        incident_side: IncidentSide,
    ) -> Self::Amplitudes {
        let left_slope = transfer_state_slope(exterior.left_admittance());

        let right_slope = transfer_state_slope(exterior.right_admittance());

        let (right_outgoing_field, right_outgoing_slope) =
            right_outgoing_column(self, &right_slope);

        let denominator = left_slope
            .multiply(&right_outgoing_field)
            .subtract(&right_outgoing_slope);

        let two = J::filled_constant_like(
            denominator.value(),
            <J::Scalar as One>::one() + <J::Scalar as One>::one(),
        );

        match incident_side {
            IncidentSide::Left => {
                /*
                 * At the left boundary:
                 *
                 *   field = 1 + r
                 *   slope = ξL (r - 1)
                 *
                 * At the right boundary only the transmitted right-going wave
                 * remains. This gives:
                 *
                 *   t = 2 ξL / D
                 *   r = t p - 1.
                 */
                let transmission = two.multiply(&left_slope).divide(&denominator);

                let reflection = transmission
                    .multiply(&right_outgoing_field)
                    .subtract(&transmission.constant(<J::Scalar as One>::one()));

                PlaneWaveAmplitudes::new(reflection, transmission)
            }

            IncidentSide::Right => {
                /*
                 * At the right boundary the incident left-going basis state is
                 * [1, +ξR], while the reflected right-going basis state is
                 * [1, -ξR].
                 */
                let (right_incoming_field, right_incoming_slope) =
                    right_incoming_column(self, &right_slope);

                let reflection = right_incoming_slope
                    .subtract(&left_slope.multiply(&right_incoming_field))
                    .divide(&denominator);

                /*
                 * Compute transmission from the propagated field rather than
                 * assuming det(M) = 1:
                 *
                 *   t = a + r p.
                 *
                 * This remains correct for any transfer representation using
                 * the documented state convention.
                 */
                let transmission =
                    right_incoming_field.add(&reflection.multiply(&right_outgoing_field));

                PlaneWaveAmplitudes::new(reflection, transmission)
            }
        }
    }
}

impl<J> ProjectPower for Transfer2Entries<J>
where
    J: Clone + RealScalarAlgebra,
    J::Scalar: ComplexScalar,
    J::Dimension: Dimension,
    J::RealJet: ScalarAlgebra,
    <J::RealJet as Jet>::Scalar: One,
{
    type Power = PlaneWavePower<J::RealJet>;

    fn project_power(
        &self,
        exterior: &Self::ExteriorContext,
        incident_side: IncidentSide,
    ) -> Self::Power {
        let (reflection, transmission) = self
            .project_amplitudes(exterior, incident_side)
            .into_parts();

        let (incident_admittance, transmitted_admittance) = match incident_side {
            IncidentSide::Left => (exterior.left_admittance(), exterior.right_admittance()),

            IncidentSide::Right => (exterior.right_admittance(), exterior.left_admittance()),
        };

        PlaneWavePower::from_amplitudes_and_admittance(
            &reflection,
            &transmission,
            incident_admittance,
            transmitted_admittance,
        )
    }
}

impl<J> ProjectPlaneWaveModeDeterminant for Transfer2Entries<J>
where
    J: ScalarAlgebra,
    J::Scalar: ComplexScalar + One,
    J::Dimension: Dimension,
{
    type Determinant = PlaneWaveDeterminant<J>;

    fn project_determinant(&self, exterior: &Self::ExteriorContext) -> Self::Determinant {
        let left_slope = transfer_state_slope(exterior.left_admittance());

        let right_slope = transfer_state_slope(exterior.right_admittance());

        let candidate = right_gauged_mode_candidate(self, &left_slope, &right_slope);

        PlaneWaveDeterminant::new(candidate.into_projective_residual())
    }
}

pub(crate) fn right_gauged_mode_candidate<A>(
    entries: &Transfer2Entries<A>,
    left_slope: &A,
    right_slope: &A,
) -> PlaneWaveModeCandidate<A>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    let (field, secondary) = right_outgoing_column(entries, right_slope);

    let residual = left_slope.multiply(&field).subtract(&secondary);

    PlaneWaveModeCandidate::new(BoundaryState::new(field, secondary), residual)
}

#[cfg(test)]
mod tests {
    use ndarray::{Ix0, arr0};

    use super::*;
    use crate::{
        algebra::{ArrayJet0, RealParameter},
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

#[cfg(test)]
mod projection_tests {
    use approx::assert_relative_eq;
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        algebra::{ArrayJet0, Jet0, RealParameter},
        backend::transfer2::projection::outgoing_residual,
        input::{CanonicalCoordinates, IncidentSide},
        test_support::materials::constant,
    };

    type C = Complex64;
    type J = ArrayJet0<C, Ix0, RealParameter>;

    const TOLERANCE: f64 = 1.0e-12;

    fn c(real: f64) -> C {
        C::new(real, 0.0)
    }

    fn jet(value: C) -> J {
        Jet0::new(arr0(value))
    }

    fn assert_close(actual: C, expected: C) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    fn coordinates() -> CanonicalCoordinates<J> {
        CanonicalCoordinates::new(jet(c(2.0)), jet(c(0.0)))
    }

    fn exterior_context(
        left_epsilon: f64,
        right_epsilon: f64,
        polarisation: Polarisation,
    ) -> Transfer2ExteriorContext<J> {
        Transfer2ExteriorContext::new::<crate::domain::RealAxis, _>(
            &coordinates(),
            &constant(left_epsilon, 1.0),
            &constant(right_epsilon, 1.0),
            polarisation,
        )
    }

    fn identity() -> Transfer2Entries<J> {
        Transfer2Entries::identity_like(coordinates().vacuum_angular_wavenumber().value())
    }

    #[test]
    fn equal_exteriors_and_identity_give_unit_transmission() {
        let entries = identity();

        let exterior = exterior_context(1.0, 1.0, Polarisation::TransverseElectric);

        for side in [IncidentSide::Left, IncidentSide::Right] {
            let amplitudes = entries.project_amplitudes(&exterior, side);

            assert_close(amplitudes.reflection()[()], c(0.0));

            assert_close(amplitudes.transmission()[()], c(1.0));
        }
    }

    #[test]
    fn identity_with_different_exteriors_matches_interface_fresnel_amplitudes() {
        let entries = identity();

        let exterior = exterior_context(1.0, 4.0, Polarisation::TransverseElectric);

        let left = entries.project_amplitudes(&exterior, IncidentSide::Left);

        assert_close(left.reflection()[()], c(-1.0 / 3.0));

        assert_close(left.transmission()[()], c(2.0 / 3.0));

        let right = entries.project_amplitudes(&exterior, IncidentSide::Right);

        assert_close(right.reflection()[()], c(1.0 / 3.0));

        assert_close(right.transmission()[()], c(4.0 / 3.0));
    }

    #[test]
    fn identity_interface_conserves_power() {
        let entries = identity();

        let exterior = exterior_context(1.0, 4.0, Polarisation::TransverseElectric);

        for side in [IncidentSide::Left, IncidentSide::Right] {
            let power = entries.project_power(&exterior, side);

            let total =
                power.reflectance()[()] + power.transmittance()[()] + power.absorptance()[()];

            assert_relative_eq!(total, 1.0, epsilon = TOLERANCE, max_relative = TOLERANCE,);

            assert_relative_eq!(
                power.absorptance()[()],
                0.0,
                epsilon = TOLERANCE,
                max_relative = TOLERANCE,
            );
        }
    }

    #[test]
    fn outgoing_residual_matches_direct_boundary_condition() {
        let entries = Transfer2Entries::new(jet(c(2.0)), jet(c(3.0)), jet(c(5.0)), jet(c(7.0)));

        let left_slope = jet(C::new(0.0, -2.0));
        let right_slope = jet(C::new(0.0, -3.0));

        let residual = outgoing_residual(&entries, &left_slope, &right_slope);

        let p = c(2.0) - c(3.0) * C::new(0.0, -3.0);

        let q = c(5.0) - c(7.0) * C::new(0.0, -3.0);

        let expected = C::new(0.0, -2.0) * p - q;

        assert_close(residual[()], expected);
    }

    #[test]
    fn determinant_projection_uses_both_exterior_admittances() {
        let entries = Transfer2Entries::new(jet(c(1.0)), jet(c(0.2)), jet(c(0.3)), jet(c(1.1)));

        let first = entries.project_determinant(&exterior_context(
            1.0,
            2.25,
            Polarisation::TransverseElectric,
        ));

        let second = entries.project_determinant(&exterior_context(
            1.0,
            4.0,
            Polarisation::TransverseElectric,
        ));

        assert_ne!(first.value(), second.value(),);
    }
}
