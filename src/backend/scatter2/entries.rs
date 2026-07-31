//! Entry-wise scattering algebra.
//!
//! [`Scatter2Entries`] is the internal computational representation used by the
//! scalar-channel scattering backend. Its entry type determines the derivative
//! order carried by the representation:
//!
//! - [`SampleArray`] for value-only evaluation;
//! - [`ArrayJetFirst`] for first-order evaluation;
//! - [`ArrayJet`] for first- and second-order evaluation.
//!
//! Redheffer composition is evaluated directly over these scalar entries. This
//! allows the scalar jet algebra to differentiate the rational star product
//! without incorrectly treating the scattering matrix itself as a bilinear
//! algebra.

use ndarray::{ArrayBase, Dimension, OwnedRepr};
use num_traits::{One, Zero};

use crate::{
    ComplexScalar, PlaneWaveAmplitudes, PlaneWavePower, Polarisation,
    algebra::{
        ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, Jet,
        RealScalarAlgebra, ScalarAlgebra,
    },
    backend::{PlaneWaveEntries, isotropic::IsotropicLayerQuantities},
    input::{CanonicalCoordinates, CanonicalProblem, IncidentSide},
    material::{ConstitutiveEvaluator, ConstitutiveLift},
    observable::{
        PlaneWaveDeterminant, ProjectAmplitudes, ProjectPlaneWaveModeDeterminant, ProjectPower,
    },
};

/// Owned sampled scalar array used by the scattering backend.
pub(crate) type SampleArray<C, D> = ArrayBase<OwnedRepr<C>, D>;

/// Zero-order entry-wise scattering representation.
pub(crate) type Scatter2Jet0<C, D, P> = Scatter2Entries<ArrayJet0<C, D, P>>;

/// First-order entry-wise scattering representation.
pub(crate) type Scatter2Jet1<C, D, P> = Scatter2Entries<ArrayJet1<C, D, P>>;

/// Second-order entry-wise scattering representation.
pub(crate) type Scatter2Jet2<C, D, P> = Scatter2Entries<ArrayJet2<C, D, P>>;

/// Bivariate entry-wise scattering representation.
pub(crate) type Scatter2JetBivariate1<C, D, P> = Scatter2Entries<ArrayJetBivariate1<C, D, P>>;

/// Bivariate entry-wise scattering representation.
pub(crate) type Scatter2JetBivariate2<C, D, P> = Scatter2Entries<ArrayJetBivariate2<C, D, P>>;

/// Four scalar entries of a 2×2 scattering matrix.
///
/// The entry type `A` determines whether this value represents:
///
/// - a value-only scattering matrix;
/// - a scattering matrix and its first derivative;
/// - a scattering matrix and its first two derivatives.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Scatter2Entries<A> {
    pub(crate) s11: A,
    pub(crate) s12: A,
    pub(crate) s21: A,
    pub(crate) s22: A,
}

impl<A> Scatter2Entries<A> {
    /// Construct the transparent identity under Redheffer composition.
    ///
    /// The identity reflects neither incident channel and transmits both
    /// channels unchanged:
    ///
    /// ```text
    /// S_identity = [0 1]
    ///              [1 0].
    /// ```
    pub(crate) fn identity_like(source: &SampleArray<A::Scalar, A::Dimension>) -> Self
    where
        A: ScalarAlgebra,
        A::Scalar: One + Zero,
        A::Dimension: Dimension,
    {
        Self {
            s11: A::filled_constant_like(source, <A::Scalar as Zero>::zero()),
            s12: A::filled_constant_like(source, <A::Scalar as One>::one()),
            s21: A::filled_constant_like(source, <A::Scalar as One>::one()),
            s22: A::filled_constant_like(source, <A::Scalar as Zero>::zero()),
        }
    }

    /// Consume the entries and return reflection and transmission channels for
    /// the selected incident side.
    ///
    /// With the scattering convention
    ///
    /// ```text
    /// [a_L^-]   [s11 s12] [a_L^+]
    /// [a_R^+] = [s21 s22] [a_R^-],
    /// ```
    ///
    /// the returned tuple is `(reflection, transmission)`.
    pub(crate) fn amplitudes(&self, side: IncidentSide) -> (A, A)
    where
        A: Clone,
    {
        let (reflection, transmission) = self.amplitudes_ref(side);

        (reflection.clone(), transmission.clone())
    }

    pub(crate) fn amplitudes_ref(&self, side: IncidentSide) -> (&'_ A, &'_ A) {
        match side {
            IncidentSide::Left => (&self.s11, &self.s21),

            IncidentSide::Right => (&self.s22, &self.s12),
        }
    }

    pub(crate) fn into_amplitudes(self, side: IncidentSide) -> (A, A) {
        match side {
            IncidentSide::Left => (self.s11, self.s21),

            IncidentSide::Right => (self.s22, self.s12),
        }
    }
}

pub(crate) struct Scatter2ExteriorContext<A> {
    left_admittance: A,
    right_admittance: A,
}

impl<J> Scatter2ExteriorContext<J> {
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

    pub(super) fn incident_and_transmitted_admittances(&self, side: IncidentSide) -> (&J, &J) {
        match side {
            IncidentSide::Left => (&self.left_admittance, &self.right_admittance),
            IncidentSide::Right => (&self.right_admittance, &self.left_admittance),
        }
    }
}

impl<A> PlaneWaveEntries for Scatter2Entries<A> {
    type ExteriorContext = Scatter2ExteriorContext<A>;
}

/// Compose two scalar-channel scattering networks.
///
/// `left` is encountered first in physical propagation order and `right` is
/// encountered second. The result therefore represents:
///
/// ```text
/// left followed by right.
/// ```
///
/// For scalar channels, the Redheffer denominator is:
///
/// ```text
/// δ = 1 - left.s22 * right.s11.
/// ```
///
/// The operation is evaluated over [`ScalarAlgebra`], so the same
/// implementation supports value-only, first-order, and second-order entries.
pub(crate) fn cascade<A>(
    left: &Scatter2Entries<A>,
    right: &Scatter2Entries<A>,
) -> Scatter2Entries<A>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar + One,
    A::Dimension: Dimension,
{
    let one = A::filled_constant_like(left.s11.value(), <A::Scalar as One>::one());

    let denominator = one.subtract(&left.s22.multiply(&right.s11));

    let s11 = left.s11.add(
        &left
            .s12
            .multiply(&right.s11)
            .multiply(&left.s21)
            .divide(&denominator),
    );

    let s12 = left.s12.multiply(&right.s12).divide(&denominator);

    let s21 = right.s21.multiply(&left.s21).divide(&denominator);

    let s22 = right.s22.add(
        &right
            .s21
            .multiply(&left.s22)
            .multiply(&right.s12)
            .divide(&denominator),
    );

    Scatter2Entries { s11, s12, s21, s22 }
}

impl<J> ProjectAmplitudes for Scatter2Entries<J>
where
    J: Clone,
{
    type Amplitudes = PlaneWaveAmplitudes<J>;

    fn project_amplitudes(
        &self,
        _exterior: &Self::ExteriorContext,
        incident_side: IncidentSide,
    ) -> Self::Amplitudes {
        let (reflection, transmission) = self.amplitudes(incident_side);

        PlaneWaveAmplitudes::new(reflection, transmission)
    }
}

impl<J> ProjectPower for Scatter2Entries<J>
where
    J: Clone + RealScalarAlgebra,
    J::RealJet: ScalarAlgebra,
    <J::RealJet as Jet>::Scalar: One,
{
    type Power = PlaneWavePower<J::RealJet>;

    fn project_power(
        &self,
        exterior: &Self::ExteriorContext,
        incident_side: IncidentSide,
    ) -> Self::Power {
        let (reflection, transmission) = self.amplitudes_ref(incident_side);

        let (incident_admittance, transmitted_admittance) =
            exterior.incident_and_transmitted_admittances(incident_side);

        PlaneWavePower::from_amplitudes_and_admittance(
            reflection,
            transmission,
            incident_admittance,
            transmitted_admittance,
        )
    }
}

impl<J> ProjectPlaneWaveModeDeterminant for Scatter2Entries<J>
where
    J: ScalarAlgebra,
    J::Scalar: ComplexScalar + One,
    J::Dimension: Dimension,
{
    type Determinant = PlaneWaveDeterminant<J>;

    fn project_determinant(&self, exterior: &Self::ExteriorContext) -> Self::Determinant {
        let value = characteristic_function(self, &exterior.left_admittance);

        PlaneWaveDeterminant::new(value)
    }
}

pub(crate) fn transfer_state_slope<A>(admittance: &A) -> A
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    admittance.scale(-<A::Scalar as ComplexScalar>::i())
}

/// Construct the outgoing-mode residual from scattering entries.
///
/// The entry type may be a sampled array, first-order jet, or second-order jet.
fn characteristic_function<A>(entries: &Scatter2Entries<A>, left_admittance: &A) -> A
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar + One,
    A::Dimension: Dimension,
{
    let slope = transfer_state_slope(left_admittance);

    let two = A::filled_constant_like(
        slope.value(),
        <A::Scalar as One>::one() + <A::Scalar as One>::one(),
    );

    let numerator = two.multiply(&slope);

    /*
     * s21 is transmission from left to right.
     */
    numerator.divide(&entries.s21)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array0, Ix0, arr0, array};
    use num_complex::Complex64;

    use super::{Scatter2Entries, cascade};

    use crate::{
        algebra::{ArrayJet0, ArrayJet1, ArrayJet2, RealParameter, ScalarAlgebra},
        input::IncidentSide,
        test_support::{
            C,
            assertions::{assert_array_close, assert_complex_close},
            c,
            jet::{J0, J1, J2, P, zero_jet_from_value},
        },
    };

    type ScalarEntries0 = Scatter2Entries<J0>;
    type ScalarEntries1 = Scatter2Entries<J1>;
    type ScalarEntries2 = Scatter2Entries<J2>;

    const TOLERANCE: f64 = 1e-12;

    fn scalar_entries(s11: C, s12: C, s21: C, s22: C) -> ScalarEntries0 {
        Scatter2Entries {
            s11: zero_jet_from_value(s11),
            s12: zero_jet_from_value(s12),
            s21: zero_jet_from_value(s21),
            s22: zero_jet_from_value(s22),
        }
    }

    fn assert_value_entries_close(
        actual: &ScalarEntries0,
        expected: &ScalarEntries0,
        tolerance: f64,
    ) {
        assert_array_close(actual.s11.value(), expected.s11.value(), tolerance);

        assert_array_close(actual.s12.value(), expected.s12.value(), tolerance);

        assert_array_close(actual.s21.value(), expected.s21.value(), tolerance);

        assert_array_close(actual.s22.value(), expected.s22.value(), tolerance);
    }

    fn assert_first_entries_close(
        actual: &ScalarEntries1,
        expected: &Scatter2Entries<Array0<C>>,
        tolerance: f64,
    ) {
        assert_array_close(actual.s11.first(), &expected.s11, tolerance);
        assert_array_close(actual.s12.first(), &expected.s12, tolerance);
        assert_array_close(actual.s21.first(), &expected.s21, tolerance);
        assert_array_close(actual.s22.first(), &expected.s22, tolerance);
    }

    fn assert_second_entries_close(
        actual: &ScalarEntries2,
        expected: &Scatter2Entries<Array0<C>>,
        tolerance: f64,
    ) {
        assert_array_close(actual.s11.second(), &expected.s11, tolerance);
        assert_array_close(actual.s12.second(), &expected.s12, tolerance);
        assert_array_close(actual.s21.second(), &expected.s21, tolerance);
        assert_array_close(actual.s22.second(), &expected.s22, tolerance);
    }

    fn scalar_array_entries(s11: C, s12: C, s21: C, s22: C) -> Scatter2Entries<Array0<C>> {
        Scatter2Entries {
            s11: arr0(s11),
            s12: arr0(s12),
            s21: arr0(s21),
            s22: arr0(s22),
        }
    }

    #[test]
    fn identity_has_transparent_entries() {
        let source = arr0(c(7.0));

        let identity: ScalarEntries0 = Scatter2Entries::identity_like(&source);

        assert_complex_close(identity.s11.value()[()], c(0.0), TOLERANCE);
        assert_complex_close(identity.s12.value()[()], c(1.0), TOLERANCE);
        assert_complex_close(identity.s21.value()[()], c(1.0), TOLERANCE);
        assert_complex_close(identity.s22.value()[()], c(0.0), TOLERANCE);
    }

    #[test]
    fn identity_has_zero_derivatives() {
        let source = arr0(c(7.0));

        let identity: ScalarEntries2 = Scatter2Entries::identity_like(&source);

        for entry in [&identity.s11, &identity.s12, &identity.s21, &identity.s22] {
            assert_complex_close(entry.first()[()], c(0.0), TOLERANCE);
            assert_complex_close(entry.second()[()], c(0.0), TOLERANCE);
        }
    }

    #[test]
    fn identity_preserves_sample_shape() {
        let source = array![c(1.0), c(2.0), c(3.0)];

        let identity: Scatter2Entries<ArrayJet0<C, ndarray::Ix1, P>> =
            Scatter2Entries::identity_like(&source);

        let expected = source.raw_dim();

        assert_eq!(identity.s11.value().raw_dim(), expected);
        assert_eq!(identity.s12.value().raw_dim(), expected);
        assert_eq!(identity.s21.value().raw_dim(), expected);
        assert_eq!(identity.s22.value().raw_dim(), expected);
    }

    #[test]
    fn amplitudes_extract_left_incident_channels() {
        let entries = scalar_entries(c(1.0), c(2.0), c(3.0), c(4.0));

        let (reflection, transmission) = entries.amplitudes(IncidentSide::Left);

        assert_complex_close(reflection.value()[()], c(1.0), TOLERANCE);
        assert_complex_close(transmission.value()[()], c(3.0), TOLERANCE);
    }

    #[test]
    fn amplitudes_extract_right_incident_channels() {
        let entries = scalar_entries(c(1.0), c(2.0), c(3.0), c(4.0));

        let (reflection, transmission) = entries.amplitudes(IncidentSide::Right);

        assert_complex_close(reflection.value()[()], c(4.0), TOLERANCE);
        assert_complex_close(transmission.value()[()], c(2.0), TOLERANCE);
    }

    #[test]
    fn cascade_matches_scalar_redheffer_formula() {
        let left = scalar_entries(c(0.1), c(0.8), c(0.7), c(-0.2));

        let right = scalar_entries(c(0.3), c(0.6), c(0.5), c(-0.1));

        let actual = cascade(&left, &right);

        let l11 = left.s11.value()[()];
        let l12 = left.s12.value()[()];
        let l21 = left.s21.value()[()];
        let l22 = left.s22.value()[()];

        let r11 = right.s11.value()[()];
        let r12 = right.s12.value()[()];
        let r21 = right.s21.value()[()];
        let r22 = right.s22.value()[()];

        let denominator = c(1.0) - l22 * r11;

        let expected = scalar_entries(
            l11 + l12 * r11 * l21 / denominator,
            l12 * r12 / denominator,
            r21 * l21 / denominator,
            r22 + r21 * l22 * r12 / denominator,
        );

        assert_value_entries_close(&actual, &expected, TOLERANCE);
    }

    #[test]
    fn cascade_preserves_complex_values() {
        let left = scalar_entries(
            C::new(0.1, 0.04),
            C::new(0.8, -0.07),
            C::new(0.7, 0.02),
            C::new(-0.2, 0.03),
        );

        let right = scalar_entries(
            C::new(0.3, -0.05),
            C::new(0.6, 0.08),
            C::new(0.5, -0.04),
            C::new(-0.1, 0.06),
        );

        let actual = cascade(&left, &right);

        let l11 = left.s11.value()[()];
        let l12 = left.s12.value()[()];
        let l21 = left.s21.value()[()];
        let l22 = left.s22.value()[()];

        let r11 = right.s11.value()[()];
        let r12 = right.s12.value()[()];
        let r21 = right.s21.value()[()];
        let r22 = right.s22.value()[()];

        let denominator = c(1.0) - l22 * r11;

        let expected = scalar_entries(
            l11 + l12 * r11 * l21 / denominator,
            l12 * r12 / denominator,
            r21 * l21 / denominator,
            r22 + r21 * l22 * r12 / denominator,
        );

        assert_value_entries_close(&actual, &expected, TOLERANCE);
    }

    #[test]
    fn identity_is_left_identity() {
        let source = arr0(c(0.0));

        let identity: ScalarEntries0 = Scatter2Entries::identity_like(&source);

        let network = scalar_entries(c(0.1), c(0.8), c(0.7), c(-0.2));

        let actual = cascade(&identity, &network);

        assert_value_entries_close(&actual, &network, TOLERANCE);
    }

    #[test]
    fn identity_is_right_identity() {
        let source = arr0(c(0.0));

        let identity: ScalarEntries0 = Scatter2Entries::identity_like(&source);

        let network = scalar_entries(c(0.1), c(0.8), c(0.7), c(-0.2));

        let actual = cascade(&network, &identity);

        assert_value_entries_close(&actual, &network, TOLERANCE);
    }

    #[test]
    fn cascade_is_associative_for_scalar_channels() {
        let first = scalar_entries(c(0.05), c(0.91), c(0.87), c(-0.03));

        let second = scalar_entries(c(0.14), c(0.76), c(0.72), c(-0.11));

        let third = scalar_entries(c(-0.08), c(0.83), c(0.79), c(0.06));

        let first_second = cascade(&first, &second);
        let left_associated = cascade(&first_second, &third);

        let second_third = cascade(&second, &third);
        let right_associated = cascade(&first, &second_third);

        assert_value_entries_close(&left_associated, &right_associated, TOLERANCE);
    }

    #[test]
    fn cascade_preserves_sample_shape_and_operates_pointwise() {
        type J = ArrayJet0<C, ndarray::Ix1, P>;

        let left = Scatter2Entries {
            s11: J::new(array![c(0.1), c(0.2)]),
            s12: J::new(array![c(0.8), c(0.7)]),
            s21: J::new(array![c(0.7), c(0.6)]),
            s22: J::new(array![c(-0.2), c(-0.1)]),
        };

        let right = Scatter2Entries {
            s11: J::new(array![c(0.3), c(0.4)]),
            s12: J::new(array![c(0.6), c(0.5)]),
            s21: J::new(array![c(0.5), c(0.4)]),
            s22: J::new(array![c(-0.1), c(-0.2)]),
        };

        let result = cascade(&left, &right);

        let expected_dimension = left.s11.value().raw_dim();

        assert_eq!(result.s11.value().raw_dim(), expected_dimension);
        assert_eq!(result.s12.value().raw_dim(), expected_dimension);
        assert_eq!(result.s21.value().raw_dim(), expected_dimension);
        assert_eq!(result.s22.value().raw_dim(), expected_dimension);

        for index in 0..2 {
            let l11 = left.s11.value()[index];
            let l12 = left.s12.value()[index];
            let l21 = left.s21.value()[index];
            let l22 = left.s22.value()[index];

            let r11 = right.s11.value()[index];
            let r12 = right.s12.value()[index];
            let r21 = right.s21.value()[index];
            let r22 = right.s22.value()[index];

            let denominator = c(1.0) - l22 * r11;

            assert_complex_close(
                result.s11.value()[index],
                l11 + l12 * r11 * l21 / denominator,
                TOLERANCE,
            );

            assert_complex_close(
                result.s12.value()[index],
                l12 * r12 / denominator,
                TOLERANCE,
            );

            assert_complex_close(
                result.s21.value()[index],
                r21 * l21 / denominator,
                TOLERANCE,
            );

            assert_complex_close(
                result.s22.value()[index],
                r22 + r21 * l22 * r12 / denominator,
                TOLERANCE,
            );
        }
    }

    fn parameterised_entries(x: f64) -> (ScalarEntries0, ScalarEntries0) {
        let left = scalar_entries(
            c(0.1 + 0.03 * x + 0.01 * x * x),
            c(0.8 - 0.04 * x + 0.02 * x * x),
            c(0.7 + 0.02 * x - 0.015 * x * x),
            c(-0.2 + 0.01 * x + 0.005 * x * x),
        );

        let right = scalar_entries(
            c(0.3 - 0.02 * x + 0.012 * x * x),
            c(0.6 + 0.05 * x - 0.01 * x * x),
            c(0.5 - 0.03 * x + 0.02 * x * x),
            c(-0.1 + 0.04 * x + 0.015 * x * x),
        );

        (left, right)
    }

    fn first_order_entry_jets() -> (ScalarEntries1, ScalarEntries1) {
        let left = Scatter2Entries {
            s11: J1::from_parts(arr0(c(0.1)), arr0(c(0.03))),
            s12: J1::from_parts(arr0(c(0.8)), arr0(c(-0.04))),
            s21: J1::from_parts(arr0(c(0.7)), arr0(c(0.02))),
            s22: J1::from_parts(arr0(c(-0.2)), arr0(c(0.01))),
        };

        let right = Scatter2Entries {
            s11: J1::from_parts(arr0(c(0.3)), arr0(c(-0.02))),
            s12: J1::from_parts(arr0(c(0.6)), arr0(c(0.05))),
            s21: J1::from_parts(arr0(c(0.5)), arr0(c(-0.03))),
            s22: J1::from_parts(arr0(c(-0.1)), arr0(c(0.04))),
        };

        (left, right)
    }

    #[test]
    fn first_derivative_matches_finite_difference() {
        let (left, right) = first_order_entry_jets();

        let analytic = cascade(&left, &right);

        let h = 1e-6;

        let (left_plus, right_plus) = parameterised_entries(h);
        let plus = cascade(&left_plus, &right_plus);

        let (left_minus, right_minus) = parameterised_entries(-h);
        let minus = cascade(&left_minus, &right_minus);

        let expected = scalar_array_entries(
            (plus.s11.value()[()] - minus.s11.value()[()]) / (2.0 * h),
            (plus.s12.value()[()] - minus.s12.value()[()]) / (2.0 * h),
            (plus.s21.value()[()] - minus.s21.value()[()]) / (2.0 * h),
            (plus.s22.value()[()] - minus.s22.value()[()]) / (2.0 * h),
        );

        assert_first_entries_close(&analytic, &expected, 1e-8);
    }

    fn second_order_entry_jets() -> (ScalarEntries2, ScalarEntries2) {
        let left = Scatter2Entries {
            s11: J2::from_parts(arr0(c(0.1)), arr0(c(0.03)), arr0(c(0.02))),
            s12: J2::from_parts(arr0(c(0.8)), arr0(c(-0.04)), arr0(c(0.04))),
            s21: J2::from_parts(arr0(c(0.7)), arr0(c(0.02)), arr0(c(-0.03))),
            s22: J2::from_parts(arr0(c(-0.2)), arr0(c(0.01)), arr0(c(0.01))),
        };

        let right = Scatter2Entries {
            s11: J2::from_parts(arr0(c(0.3)), arr0(c(-0.02)), arr0(c(0.024))),
            s12: J2::from_parts(arr0(c(0.6)), arr0(c(0.05)), arr0(c(-0.02))),
            s21: J2::from_parts(arr0(c(0.5)), arr0(c(-0.03)), arr0(c(0.04))),
            s22: J2::from_parts(arr0(c(-0.1)), arr0(c(0.04)), arr0(c(0.03))),
        };

        (left, right)
    }

    #[test]
    fn second_derivative_matches_finite_difference() {
        let (left, right) = second_order_entry_jets();

        let analytic = cascade(&left, &right);

        let h = 1e-4;
        let h_squared = h * h;

        let (left_plus, right_plus) = parameterised_entries(h);
        let plus = cascade(&left_plus, &right_plus);

        let (left_zero, right_zero) = parameterised_entries(0.0);
        let zero = cascade(&left_zero, &right_zero);

        let (left_minus, right_minus) = parameterised_entries(-h);
        let minus = cascade(&left_minus, &right_minus);

        let expected = scalar_array_entries(
            (plus.s11.value()[()] - c(2.0) * zero.s11.value()[()] + minus.s11.value()[()])
                / h_squared,
            (plus.s12.value()[()] - c(2.0) * zero.s12.value()[()] + minus.s12.value()[()])
                / h_squared,
            (plus.s21.value()[()] - c(2.0) * zero.s21.value()[()] + minus.s21.value()[()])
                / h_squared,
            (plus.s22.value()[()] - c(2.0) * zero.s22.value()[()] + minus.s22.value()[()])
                / h_squared,
        );

        assert_second_entries_close(&analytic, &expected, 2e-7);
    }
}
