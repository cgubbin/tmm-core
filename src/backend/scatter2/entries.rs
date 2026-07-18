//! Entry-wise scattering algebra.
//!
//! [`ScatterEntries`] is the internal computational representation used by the
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

use crate::{
    ComplexScalar,
    backend::{
        IncidentSide,
        algebra::ScalarAlgebra,
        derivative::ChainRule,
        jet::{ArrayJet, ArrayJetFirst},
        mode::OutgoingModeAmplitudes,
    },
};

/// Owned sampled scalar array used by the scattering backend.
pub(crate) type SampleArray<C, D> = ArrayBase<OwnedRepr<C>, D>;

/// First-order entry-wise scattering representation.
pub(crate) type Scatter2JetFirst<C, D> = ScatterEntries<ArrayJetFirst<C, D>>;

/// Second-order entry-wise scattering representation.
pub(crate) type Scatter2Jet<C, D> = ScatterEntries<ArrayJet<C, D>>;

/// Four scalar entries of a 2×2 scattering matrix.
///
/// The entry type `A` determines whether this value represents:
///
/// - a value-only scattering matrix;
/// - a scattering matrix and its first derivative;
/// - a scattering matrix and its first two derivatives.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ScatterEntries<A> {
    pub(crate) s11: A,
    pub(crate) s12: A,
    pub(crate) s21: A,
    pub(crate) s22: A,
}

impl<A> ScatterEntries<A> {
    /// Construct the transparent identity under Redheffer composition.
    ///
    /// The identity reflects neither incident channel and transmits both
    /// channels unchanged:
    ///
    /// ```text
    /// S_identity = [0 1]
    ///              [1 0].
    /// ```
    pub(crate) fn identity_like<C, D>(source: &SampleArray<C, D>) -> Self
    where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D>,
    {
        Self {
            s11: A::constant_like(source, C::zero()),
            s12: A::constant_like(source, C::one()),
            s21: A::constant_like(source, C::one()),
            s22: A::constant_like(source, C::zero()),
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
    pub(crate) fn amplitudes(self, side: IncidentSide) -> (A, A) {
        match side {
            IncidentSide::Left => (self.s11, self.s21),

            IncidentSide::Right => (self.s22, self.s12),
        }
    }
}

impl<C> ScatterEntries<ndarray::Array0<C>>
where
    C: ComplexScalar,
{
    pub(crate) fn outgoing_mode_amplitudes(&self) -> OutgoingModeAmplitudes<C> {
        let left_norm = self.s11[()].modulus_squared() + self.s21[()].modulus_squared();

        let right_norm = self.s12[()].modulus_squared() + self.s22[()].modulus_squared();

        let (left_outgoing, right_outgoing) = if left_norm >= right_norm {
            (self.s11[()].clone(), self.s21[()].clone())
        } else {
            (self.s12[()].clone(), self.s22[()].clone())
        };

        OutgoingModeAmplitudes::normalised(left_outgoing, right_outgoing)
    }
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
pub(crate) fn cascade<C, D, A>(
    left: &ScatterEntries<A>,
    right: &ScatterEntries<A>,
) -> ScatterEntries<A>
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D>,
{
    let one = A::constant_like(left.s11.value(), C::one());

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

    ScatterEntries { s11, s12, s21, s22 }
}

/// Apply a sampled chain-rule transformation to one scattering entry.
pub(crate) trait ApplyChainRule<R>: Sized {
    /// Transform derivatives from the primitive coordinate to the requested
    /// coordinate.
    fn apply_chain_rule(self, rule: &ChainRule<R>) -> Self;
}

impl<C, D> ApplyChainRule<SampleArray<C, D>> for ArrayJetFirst<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn apply_chain_rule(self, rule: &ChainRule<SampleArray<C, D>>) -> Self {
        self.chain_rule(rule)
    }
}

impl<C, D> ApplyChainRule<SampleArray<C, D>> for ArrayJet<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn apply_chain_rule(self, rule: &ChainRule<SampleArray<C, D>>) -> Self {
        self.chain_rule(rule)
    }
}

impl<A> ScatterEntries<A> {
    /// Apply an entrywise chain-rule transformation.
    ///
    /// This is used after the complete scattering cascade has been evaluated
    /// in the backend's primitive squared spectral coordinate.
    pub(crate) fn chain_rule<R>(self, rule: &ChainRule<R>) -> Self
    where
        A: ApplyChainRule<R>,
    {
        Self {
            s11: self.s11.apply_chain_rule(rule),
            s12: self.s12.apply_chain_rule(rule),
            s21: self.s21.apply_chain_rule(rule),
            s22: self.s22.apply_chain_rule(rule),
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array0, ArrayBase, Dimension, Ix0, OwnedRepr, arr0, array};
    use num_complex::Complex64;

    use super::*;

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn scalar_entries(s11: C, s12: C, s21: C, s22: C) -> ScatterEntries<Array0<C>> {
        ScatterEntries {
            s11: arr0(s11),
            s12: arr0(s12),
            s21: arr0(s21),
            s22: arr0(s22),
        }
    }

    fn assert_complex_close(actual: C, expected: C, tolerance: f64) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = tolerance,
            max_relative = tolerance,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = tolerance,
            max_relative = tolerance,
        );
    }

    fn assert_array_close<D>(
        actual: &ArrayBase<OwnedRepr<C>, D>,
        expected: &ArrayBase<OwnedRepr<C>, D>,
        tolerance: f64,
    ) where
        D: Dimension,
    {
        assert_eq!(actual.raw_dim(), expected.raw_dim());

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_complex_close(actual, expected, tolerance);
        }
    }

    fn assert_entries_close(
        actual: &ScatterEntries<Array0<C>>,
        expected: &ScatterEntries<Array0<C>>,
        tolerance: f64,
    ) {
        assert_array_close(&actual.s11, &expected.s11, tolerance);

        assert_array_close(&actual.s12, &expected.s12, tolerance);

        assert_array_close(&actual.s21, &expected.s21, tolerance);

        assert_array_close(&actual.s22, &expected.s22, tolerance);
    }

    #[test]
    fn identity_has_transparent_entries() {
        let source = arr0(c(7.0));

        let identity: ScatterEntries<Array0<C>> = ScatterEntries::identity_like(&source);

        assert_eq!(identity.s11[()], c(0.0));
        assert_eq!(identity.s12[()], c(1.0));
        assert_eq!(identity.s21[()], c(1.0));
        assert_eq!(identity.s22[()], c(0.0));
    }

    #[test]
    fn identity_preserves_sample_shape() {
        let source = array![c(1.0), c(2.0), c(3.0)];

        let identity: ScatterEntries<ArrayBase<OwnedRepr<C>, ndarray::Ix1>> =
            ScatterEntries::identity_like(&source);

        let expected = source.raw_dim();

        assert_eq!(identity.s11.raw_dim(), expected);
        assert_eq!(identity.s12.raw_dim(), expected);
        assert_eq!(identity.s21.raw_dim(), expected);
        assert_eq!(identity.s22.raw_dim(), expected);
    }

    #[test]
    fn amplitudes_extract_left_incident_channels() {
        let entries = scalar_entries(c(1.0), c(2.0), c(3.0), c(4.0));

        let (reflection, transmission) = entries.amplitudes(IncidentSide::Left);

        assert_eq!(reflection[()], c(1.0));
        assert_eq!(transmission[()], c(3.0));
    }

    #[test]
    fn amplitudes_extract_right_incident_channels() {
        let entries = scalar_entries(c(1.0), c(2.0), c(3.0), c(4.0));

        let (reflection, transmission) = entries.amplitudes(IncidentSide::Right);

        assert_eq!(reflection[()], c(4.0));
        assert_eq!(transmission[()], c(2.0));
    }

    #[test]
    fn cascade_matches_scalar_redheffer_formula() {
        let left = scalar_entries(c(0.1), c(0.8), c(0.7), c(-0.2));

        let right = scalar_entries(c(0.3), c(0.6), c(0.5), c(-0.1));

        let actual = cascade::<C, Ix0, _>(&left, &right);

        let l11 = left.s11[()];
        let l12 = left.s12[()];
        let l21 = left.s21[()];
        let l22 = left.s22[()];

        let r11 = right.s11[()];
        let r12 = right.s12[()];
        let r21 = right.s21[()];
        let r22 = right.s22[()];

        let denominator = c(1.0) - l22 * r11;

        let expected = scalar_entries(
            l11 + l12 * r11 * l21 / denominator,
            l12 * r12 / denominator,
            r21 * l21 / denominator,
            r22 + r21 * l22 * r12 / denominator,
        );

        assert_entries_close(&actual, &expected, 1e-12);
    }

    #[test]
    fn identity_is_left_identity() {
        let source = arr0(c(0.0));

        let identity: ScatterEntries<Array0<C>> = ScatterEntries::identity_like(&source);

        let network = scalar_entries(c(0.1), c(0.8), c(0.7), c(-0.2));

        let actual = cascade::<C, Ix0, _>(&identity, &network);

        assert_entries_close(&actual, &network, 1e-12);
    }

    #[test]
    fn identity_is_right_identity() {
        let source = arr0(c(0.0));

        let identity: ScatterEntries<Array0<C>> = ScatterEntries::identity_like(&source);

        let network = scalar_entries(c(0.1), c(0.8), c(0.7), c(-0.2));

        let actual = cascade::<C, Ix0, _>(&network, &identity);

        assert_entries_close(&actual, &network, 1e-12);
    }

    #[test]
    fn cascade_is_associative_for_scalar_channels() {
        let first = scalar_entries(c(0.05), c(0.91), c(0.87), c(-0.03));

        let second = scalar_entries(c(0.14), c(0.76), c(0.72), c(-0.11));

        let third = scalar_entries(c(-0.08), c(0.83), c(0.79), c(0.06));

        let first_second = cascade::<C, Ix0, _>(&first, &second);

        let left_associated = cascade::<C, Ix0, _>(&first_second, &third);

        let second_third = cascade::<C, Ix0, _>(&second, &third);

        let right_associated = cascade::<C, Ix0, _>(&first, &second_third);

        assert_entries_close(&left_associated, &right_associated, 1e-12);
    }

    #[test]
    fn cascade_preserves_sample_shape() {
        let left = ScatterEntries {
            s11: array![c(0.1), c(0.2)],
            s12: array![c(0.8), c(0.7)],
            s21: array![c(0.7), c(0.6)],
            s22: array![c(-0.2), c(-0.1)],
        };

        let right = ScatterEntries {
            s11: array![c(0.3), c(0.4)],
            s12: array![c(0.6), c(0.5)],
            s21: array![c(0.5), c(0.4)],
            s22: array![c(-0.1), c(-0.2)],
        };

        let result = cascade::<C, ndarray::Ix1, _>(&left, &right);

        let expected = left.s11.raw_dim();

        assert_eq!(result.s11.raw_dim(), expected);
        assert_eq!(result.s12.raw_dim(), expected);
        assert_eq!(result.s21.raw_dim(), expected);
        assert_eq!(result.s22.raw_dim(), expected);
    }

    fn parameterised_entries(x: f64) -> (ScatterEntries<Array0<C>>, ScatterEntries<Array0<C>>) {
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

    fn first_order_entry_jets() -> (
        ScatterEntries<ArrayJetFirst<C, Ix0>>,
        ScatterEntries<ArrayJetFirst<C, Ix0>>,
    ) {
        let (left, right) = parameterised_entries(0.0);

        let left = ScatterEntries {
            s11: ArrayJetFirst::from_parts(left.s11, arr0(c(0.03))),
            s12: ArrayJetFirst::from_parts(left.s12, arr0(c(-0.04))),
            s21: ArrayJetFirst::from_parts(left.s21, arr0(c(0.02))),
            s22: ArrayJetFirst::from_parts(left.s22, arr0(c(0.01))),
        };

        let right = ScatterEntries {
            s11: ArrayJetFirst::from_parts(right.s11, arr0(c(-0.02))),
            s12: ArrayJetFirst::from_parts(right.s12, arr0(c(0.05))),
            s21: ArrayJetFirst::from_parts(right.s21, arr0(c(-0.03))),
            s22: ArrayJetFirst::from_parts(right.s22, arr0(c(0.04))),
        };

        (left, right)
    }

    #[test]
    fn first_derivative_matches_finite_difference() {
        let (left, right) = first_order_entry_jets();

        let analytic = cascade::<C, Ix0, _>(&left, &right);

        let h = 1e-6;

        let (left_plus, right_plus) = parameterised_entries(h);

        let plus = cascade::<C, Ix0, _>(&left_plus, &right_plus);

        let (left_minus, right_minus) = parameterised_entries(-h);

        let minus = cascade::<C, Ix0, _>(&left_minus, &right_minus);

        let expected_s11 = (plus.s11[()] - minus.s11[()]) / (2.0 * h);

        let expected_s12 = (plus.s12[()] - minus.s12[()]) / (2.0 * h);

        let expected_s21 = (plus.s21[()] - minus.s21[()]) / (2.0 * h);

        let expected_s22 = (plus.s22[()] - minus.s22[()]) / (2.0 * h);

        assert_complex_close(analytic.s11.first()[()], expected_s11, 1e-8);

        assert_complex_close(analytic.s12.first()[()], expected_s12, 1e-8);

        assert_complex_close(analytic.s21.first()[()], expected_s21, 1e-8);

        assert_complex_close(analytic.s22.first()[()], expected_s22, 1e-8);
    }

    fn second_order_entry_jets() -> (
        ScatterEntries<ArrayJet<C, Ix0>>,
        ScatterEntries<ArrayJet<C, Ix0>>,
    ) {
        let (left, right) = parameterised_entries(0.0);

        let left = ScatterEntries {
            s11: ArrayJet::from_parts(left.s11, arr0(c(0.03)), arr0(c(0.02))),
            s12: ArrayJet::from_parts(left.s12, arr0(c(-0.04)), arr0(c(0.04))),
            s21: ArrayJet::from_parts(left.s21, arr0(c(0.02)), arr0(c(-0.03))),
            s22: ArrayJet::from_parts(left.s22, arr0(c(0.01)), arr0(c(0.01))),
        };

        let right = ScatterEntries {
            s11: ArrayJet::from_parts(right.s11, arr0(c(-0.02)), arr0(c(0.024))),
            s12: ArrayJet::from_parts(right.s12, arr0(c(0.05)), arr0(c(-0.02))),
            s21: ArrayJet::from_parts(right.s21, arr0(c(-0.03)), arr0(c(0.04))),
            s22: ArrayJet::from_parts(right.s22, arr0(c(0.04)), arr0(c(0.03))),
        };

        (left, right)
    }

    #[test]
    fn second_derivative_matches_finite_difference() {
        let (left, right) = second_order_entry_jets();

        let analytic = cascade::<C, Ix0, _>(&left, &right);

        let h = 1e-4;

        let (left_plus, right_plus) = parameterised_entries(h);

        let plus = cascade::<C, Ix0, _>(&left_plus, &right_plus);

        let (left_zero, right_zero) = parameterised_entries(0.0);

        let zero = cascade::<C, Ix0, _>(&left_zero, &right_zero);

        let (left_minus, right_minus) = parameterised_entries(-h);

        let minus = cascade::<C, Ix0, _>(&left_minus, &right_minus);

        let h2 = h * h;

        let expected_s11 = (plus.s11[()] - c(2.0) * zero.s11[()] + minus.s11[()]) / h2;

        let expected_s12 = (plus.s12[()] - c(2.0) * zero.s12[()] + minus.s12[()]) / h2;

        let expected_s21 = (plus.s21[()] - c(2.0) * zero.s21[()] + minus.s21[()]) / h2;

        let expected_s22 = (plus.s22[()] - c(2.0) * zero.s22[()] + minus.s22[()]) / h2;

        assert_complex_close(analytic.s11.second()[()], expected_s11, 2e-7);

        assert_complex_close(analytic.s12.second()[()], expected_s12, 2e-7);

        assert_complex_close(analytic.s21.second()[()], expected_s21, 2e-7);

        assert_complex_close(analytic.s22.second()[()], expected_s22, 2e-7);
    }

    #[test]
    fn first_order_chain_rule_is_applied_to_every_entry() {
        let entries = ScatterEntries {
            s11: ArrayJetFirst::from_parts(arr0(c(1.0)), arr0(c(2.0))),
            s12: ArrayJetFirst::from_parts(arr0(c(3.0)), arr0(c(4.0))),
            s21: ArrayJetFirst::from_parts(arr0(c(5.0)), arr0(c(6.0))),
            s22: ArrayJetFirst::from_parts(arr0(c(7.0)), arr0(c(8.0))),
        };

        let rule = ChainRule {
            first: arr0(c(10.0)),
            second: arr0(c(20.0)),
        };

        let transformed = entries.chain_rule(&rule);

        assert_eq!(transformed.s11.first()[()], c(20.0),);

        assert_eq!(transformed.s12.first()[()], c(40.0),);

        assert_eq!(transformed.s21.first()[()], c(60.0),);

        assert_eq!(transformed.s22.first()[()], c(80.0),);
    }

    #[test]
    fn second_order_chain_rule_is_applied_to_every_entry() {
        let entries = ScatterEntries {
            s11: ArrayJet::from_parts(arr0(c(1.0)), arr0(c(2.0)), arr0(c(3.0))),
            s12: ArrayJet::from_parts(arr0(c(4.0)), arr0(c(5.0)), arr0(c(6.0))),
            s21: ArrayJet::from_parts(arr0(c(7.0)), arr0(c(8.0)), arr0(c(9.0))),
            s22: ArrayJet::from_parts(arr0(c(10.0)), arr0(c(11.0)), arr0(c(12.0))),
        };

        let rule = ChainRule {
            first: arr0(c(2.0)),
            second: arr0(c(3.0)),
        };

        let transformed = entries.chain_rule(&rule);

        // y' = primitive_first * 2
        assert_eq!(transformed.s11.first()[()], c(4.0),);

        // y'' = primitive_second * 2²
        //     + primitive_first * 3
        assert_eq!(transformed.s11.second()[()], c(3.0 * 4.0 + 2.0 * 3.0),);

        assert_eq!(transformed.s12.second()[()], c(6.0 * 4.0 + 5.0 * 3.0),);

        assert_eq!(transformed.s21.second()[()], c(9.0 * 4.0 + 8.0 * 3.0),);

        assert_eq!(transformed.s22.second()[()], c(12.0 * 4.0 + 11.0 * 3.0),);
    }
}
