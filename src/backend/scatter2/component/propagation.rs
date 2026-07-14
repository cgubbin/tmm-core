//! Scattering entries for propagation through a homogeneous layer.
//!
//! A homogeneous finite layer contributes no reflection by itself. Its
//! scattering representation transmits both propagation directions with the
//! same phase factor:
//!
//! ```text
//! p = exp(i κ d)
//!
//! S_propagation = [0 p]
//!                 [p 0].
//! ```
//!
//! The implementation is generic over sampled arrays and scalar array jets, so
//! the same construction is used for value-only, first-order, and second-order
//! evaluation.

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::{
        algebra::ScalarAlgebra,
        jet::{ArrayJet, ArrayJetFirst},
    },
};

use super::super::entries::ScatterEntries;

/// Scalar-like value supporting an elementwise complex exponential.
///
/// This trait is private to homogeneous propagation. It allows the propagation
/// constructor to operate uniformly on sampled values and derivative jets.
pub(crate) trait Exponential: Sized {
    /// Apply the elementwise complex exponential.
    fn exponential(self) -> Self;
}

impl<C, D> Exponential for ArrayBase<OwnedRepr<C>, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn exponential(self) -> Self {
        self.mapv(|value| value.exp())
    }
}

impl<C, D> Exponential for ArrayJetFirst<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn exponential(self) -> Self {
        self.exp()
    }
}

impl<C, D> Exponential for ArrayJet<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn exponential(self) -> Self {
        self.exp()
    }
}

/// Construct homogeneous propagation entries from an exponent.
///
/// `exponent` represents:
///
/// ```text
/// i κ d
/// ```
///
/// and may be a sampled value, first-order jet, or second-order jet. This
/// lower-level constructor is used when the exponent itself must carry
/// derivative information, such as for a layer-thickness derivative.
pub(crate) fn propagation_from_exponent<C, D, A>(exponent: A) -> ScatterEntries<A>
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Exponential + Clone,
{
    let phase = exponent.exponential();

    let zero = A::constant_like(phase.value(), C::zero());

    ScatterEntries {
        s11: zero.clone(),
        s12: phase.clone(),
        s21: phase,
        s22: zero,
    }
}

/// Construct homogeneous propagation entries from a normal wavenumber.
///
/// The propagation phase is:
///
/// ```text
/// p = exp(i κ d),
/// ```
///
/// where `thickness` is expressed in the inverse unit corresponding to
/// `kappa`. For the current backend, `kappa` is in inverse centimetres and
/// thickness is converted to centimetres before calling this function.
///
/// This constructor treats `thickness` as constant. For derivatives with
/// respect to thickness, construct a jet for the complete exponent and call
/// [`propagation_from_exponent`].
pub(crate) fn propagation<C, D, A>(kappa: &A, thickness: C) -> ScatterEntries<A>
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Exponential + Clone,
{
    let exponent = kappa.scale(C::i() * thickness);

    propagation_from_exponent::<C, D, A>(exponent)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{ArrayBase, Dimension, Ix0, OwnedRepr, arr0, array};
    use num_complex::Complex64;

    use super::*;

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
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

    #[test]
    fn zero_exponent_produces_transparent_identity() {
        let exponent = arr0(c(0.0));

        let entries = propagation_from_exponent::<C, Ix0, _>(exponent);

        assert_complex_close(entries.s11[()], c(0.0), 1e-12);

        assert_complex_close(entries.s12[()], c(1.0), 1e-12);

        assert_complex_close(entries.s21[()], c(1.0), 1e-12);

        assert_complex_close(entries.s22[()], c(0.0), 1e-12);
    }

    #[test]
    fn propagation_matches_expected_phase() {
        let kappa = arr0(c(2.3));
        let thickness = c(0.4);

        let entries = propagation::<C, Ix0, _>(&kappa, thickness);

        let expected = (C::i() * c(2.3) * thickness).exp();

        assert_complex_close(entries.s11[()], c(0.0), 1e-12);

        assert_complex_close(entries.s12[()], expected, 1e-12);

        assert_complex_close(entries.s21[()], expected, 1e-12);

        assert_complex_close(entries.s22[()], c(0.0), 1e-12);
    }

    #[test]
    fn propagation_is_symmetric_between_directions() {
        let kappa = arr0(C::new(2.3, 0.4));

        let entries = propagation::<C, Ix0, _>(&kappa, c(0.4));

        assert_complex_close(entries.s12[()], entries.s21[()], 1e-12);
    }

    #[test]
    fn evanescent_propagation_decays() {
        let kappa = arr0(C::new(0.0, 2.0));

        let entries = propagation::<C, Ix0, _>(&kappa, c(0.5));

        let expected = (-1.0_f64).exp();

        assert_relative_eq!(
            entries.s21[()].norm(),
            expected,
            epsilon = 1e-12,
            max_relative = 1e-12,
        );
    }

    #[test]
    fn sampled_propagation_preserves_shape() {
        let kappa = array![c(1.0), c(2.0), c(3.0),];

        let entries = propagation::<C, ndarray::Ix1, _>(&kappa, c(0.2));

        let expected = kappa.raw_dim();

        assert_eq!(entries.s11.raw_dim(), expected);
        assert_eq!(entries.s12.raw_dim(), expected);
        assert_eq!(entries.s21.raw_dim(), expected);
        assert_eq!(entries.s22.raw_dim(), expected);
    }

    #[test]
    fn first_order_exponential_jet_matches_analytic_derivative() {
        let exponent_value = C::new(-0.2, 0.7);

        let exponent_first = C::new(0.3, -0.1);

        let exponent = ArrayJetFirst::from_parts(arr0(exponent_value), arr0(exponent_first));

        let entries = propagation_from_exponent::<C, Ix0, _>(exponent);

        let phase = exponent_value.exp();

        let expected_first = phase * exponent_first;

        assert_complex_close(entries.s12.value()[()], phase, 1e-12);

        assert_complex_close(entries.s12.first()[()], expected_first, 1e-12);

        assert_complex_close(entries.s21.first()[()], expected_first, 1e-12);

        assert_complex_close(entries.s11.first()[()], c(0.0), 1e-12);

        assert_complex_close(entries.s22.first()[()], c(0.0), 1e-12);
    }

    #[test]
    fn second_order_exponential_jet_matches_analytic_derivative() {
        let exponent_value = C::new(-0.2, 0.7);

        let exponent_first = C::new(0.3, -0.1);

        let exponent_second = C::new(-0.05, 0.08);

        let exponent = ArrayJet::from_parts(
            arr0(exponent_value),
            arr0(exponent_first),
            arr0(exponent_second),
        );

        let entries = propagation_from_exponent::<C, Ix0, _>(exponent);

        let phase = exponent_value.exp();

        let expected_first = phase * exponent_first;

        let expected_second = phase * (exponent_first * exponent_first + exponent_second);

        assert_complex_close(entries.s12.value()[()], phase, 1e-12);

        assert_complex_close(entries.s12.first()[()], expected_first, 1e-12);

        assert_complex_close(entries.s12.second()[()], expected_second, 1e-12);

        assert_complex_close(entries.s21.second()[()], expected_second, 1e-12);

        assert_complex_close(entries.s11.second()[()], c(0.0), 1e-12);

        assert_complex_close(entries.s22.second()[()], c(0.0), 1e-12);
    }

    fn spectral_kappa(x: f64) -> C {
        c(2.0 + 0.3 * x + 0.05 * x * x)
    }

    #[test]
    fn spectral_first_derivative_matches_finite_difference() {
        let thickness = c(0.4);

        let kappa = ArrayJetFirst::from_parts(arr0(spectral_kappa(0.0)), arr0(c(0.3)));

        let analytic = propagation::<C, Ix0, _>(&kappa, thickness);

        let h = 1e-6;

        let plus = propagation::<C, Ix0, _>(&arr0(spectral_kappa(h)), thickness);

        let minus = propagation::<C, Ix0, _>(&arr0(spectral_kappa(-h)), thickness);

        let expected = (plus.s21[()] - minus.s21[()]) / (2.0 * h);

        assert_complex_close(analytic.s21.first()[()], expected, 1e-8);
    }

    #[test]
    fn spectral_second_derivative_matches_finite_difference() {
        let thickness = c(0.4);

        let kappa = ArrayJet::from_parts(arr0(spectral_kappa(0.0)), arr0(c(0.3)), arr0(c(0.1)));

        let analytic = propagation::<C, Ix0, _>(&kappa, thickness);

        let h = 1e-4;

        let plus = propagation::<C, Ix0, _>(&arr0(spectral_kappa(h)), thickness);

        let zero = propagation::<C, Ix0, _>(&arr0(spectral_kappa(0.0)), thickness);

        let minus = propagation::<C, Ix0, _>(&arr0(spectral_kappa(-h)), thickness);

        let expected = (plus.s21[()] - c(2.0) * zero.s21[()] + minus.s21[()]) / (h * h);

        assert_complex_close(analytic.s21.second()[()], expected, 2e-7);
    }

    #[test]
    fn thickness_first_derivative_matches_finite_difference() {
        let kappa = c(2.3);
        let thickness = 0.4;

        /*
         * exponent = i κ d
         *
         * d exponent / d d = i κ
         */
        let exponent =
            ArrayJetFirst::from_parts(arr0(C::i() * kappa * c(thickness)), arr0(C::i() * kappa));

        let analytic = propagation_from_exponent::<C, Ix0, _>(exponent);

        let h = 1e-6;

        let plus = propagation::<C, Ix0, _>(&arr0(kappa), c(thickness + h));

        let minus = propagation::<C, Ix0, _>(&arr0(kappa), c(thickness - h));

        let expected = (plus.s21[()] - minus.s21[()]) / (2.0 * h);

        assert_complex_close(analytic.s21.first()[()], expected, 1e-8);
    }

    #[test]
    fn thickness_second_derivative_matches_finite_difference() {
        let kappa = c(2.3);
        let thickness = 0.4;

        /*
         * exponent = i κ d
         *
         * exponent'  = i κ
         * exponent'' = 0
         */
        let exponent = ArrayJet::from_parts(
            arr0(C::i() * kappa * c(thickness)),
            arr0(C::i() * kappa),
            arr0(c(0.0)),
        );

        let analytic = propagation_from_exponent::<C, Ix0, _>(exponent);

        let h = 1e-4;

        let plus = propagation::<C, Ix0, _>(&arr0(kappa), c(thickness + h));

        let zero = propagation::<C, Ix0, _>(&arr0(kappa), c(thickness));

        let minus = propagation::<C, Ix0, _>(&arr0(kappa), c(thickness - h));

        let expected = (plus.s21[()] - c(2.0) * zero.s21[()] + minus.s21[()]) / (h * h);

        assert_complex_close(analytic.s21.second()[()], expected, 2e-7);
    }

    #[test]
    fn zero_derivative_jet_reproduces_value_path() {
        let kappa_value = arr0(C::new(2.3, 0.2));

        let expected = propagation::<C, Ix0, _>(&kappa_value, c(0.4));

        let kappa = ArrayJetFirst::from_parts(kappa_value, arr0(c(0.0)));

        let actual = propagation::<C, Ix0, _>(&kappa, c(0.4));

        assert_array_close(actual.s12.value(), &expected.s12, 1e-12);

        assert_array_close(actual.s21.value(), &expected.s21, 1e-12);

        assert_complex_close(actual.s12.first()[()], c(0.0), 1e-12);

        assert_complex_close(actual.s21.first()[()], c(0.0), 1e-12);
    }
}
