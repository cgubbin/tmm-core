use crate::{
    algebra::{
        ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, RealParameter,
        ScalarAlgebra,
    },
    backend::{BidirectionalWaves, LayerBoundaryWaves},
};

use super::{C, TOLERANCE, c};

use approx::assert_relative_eq;
use ndarray::{ArrayBase, Data, Dimension, Ix0};

pub fn assert_real_close(actual: f64, expected: f64, tolerance: f64) {
    let error = (actual - expected).abs();

    assert!(
        error <= tolerance,
        "expected {expected:e}, got {actual:e}; \
             absolute error = {error:e}",
    );
}

pub fn assert_complex_close(actual: C, expected: C, tolerance: f64) {
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

pub fn assert_array_close<D>(
    actual: &ArrayBase<impl Data<Elem = C>, D>,
    expected: &ArrayBase<impl Data<Elem = C>, D>,
    tolerance: f64,
) where
    D: Dimension,
{
    assert_eq!(actual.raw_dim(), expected.raw_dim());

    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        assert_complex_close(actual, expected, tolerance);
    }
}

pub fn assert_dispersion_relation(
    epsilon: C,
    mu: C,
    kappa: C,
    k0: f64,
    k_parallel: f64,
    tolerance: f64,
) {
    assert_complex_close(
        kappa * kappa,
        epsilon * mu * c(k0 * k0) - c(k_parallel * k_parallel),
        tolerance,
    );
}

pub(crate) fn assert_bidirectional_waves_close<A>(
    actual: &BidirectionalWaves<A>,
    expected: &BidirectionalWaves<A>,
    tolerance: f64,
) where
    A: ScalarAlgebra<Scalar = C>,
    A::Dimension: ndarray::Dimension,
{
    assert_array_close(
        actual.forward().value(),
        expected.forward().value(),
        tolerance,
    );

    assert_array_close(
        actual.backward().value(),
        expected.backward().value(),
        tolerance,
    );
}

pub(crate) fn assert_layer_boundary_waves_close<A>(
    actual: &LayerBoundaryWaves<A>,
    expected: &LayerBoundaryWaves<A>,
    tolerance: f64,
) where
    A: ScalarAlgebra<Scalar = C>,
    A::Dimension: ndarray::Dimension,
{
    assert_bidirectional_waves_close(actual.left(), expected.left(), tolerance);

    assert_bidirectional_waves_close(actual.right(), expected.right(), tolerance);
}

type D = Ix0;

type ZeroJet = ArrayJet0<C, D, RealParameter>;

type FirstJet = ArrayJet1<C, D, RealParameter>;

type SecondJet = ArrayJet2<C, D, RealParameter>;

type BivariateFirstJet = ArrayJetBivariate1<C, D, RealParameter>;

type BivariateSecondJet = ArrayJetBivariate2<C, D, RealParameter>;

pub(crate) const VALUE_TOLERANCE: f64 = 1.0e-11;
pub(crate) const FIRST_TOLERANCE: f64 = 1.0e-10;
pub(crate) const SECOND_TOLERANCE: f64 = 1.0e-9;

pub(crate) fn assert_zero_jet_close(actual: &ZeroJet, expected: &ZeroJet) {
    assert_complex_close(actual.value()[()], expected.value()[()], VALUE_TOLERANCE);
}

pub(crate) fn assert_first_jet_close(actual: &FirstJet, expected: &FirstJet) {
    assert_complex_close(actual.value()[()], expected.value()[()], VALUE_TOLERANCE);

    assert_complex_close(actual.first()[()], expected.first()[()], FIRST_TOLERANCE);
}

pub(crate) fn assert_second_jet_close(actual: &SecondJet, expected: &SecondJet) {
    assert_complex_close(actual.value()[()], expected.value()[()], VALUE_TOLERANCE);

    assert_complex_close(actual.first()[()], expected.first()[()], FIRST_TOLERANCE);

    assert_complex_close(actual.second()[()], expected.second()[()], SECOND_TOLERANCE);
}

pub(crate) fn assert_bivariate_first_jet_close(
    actual: &BivariateFirstJet,
    expected: &BivariateFirstJet,
) {
    assert_complex_close(actual.value()[()], expected.value()[()], VALUE_TOLERANCE);

    assert_complex_close(actual.axis0()[()], expected.axis0()[()], FIRST_TOLERANCE);

    assert_complex_close(actual.axis1()[()], expected.axis1()[()], FIRST_TOLERANCE);
}

pub(crate) fn assert_bivariate_second_jet_close(
    actual: &BivariateSecondJet,
    expected: &BivariateSecondJet,
) {
    assert_complex_close(actual.value()[()], expected.value()[()], VALUE_TOLERANCE);

    assert_complex_close(actual.axis0()[()], expected.axis0()[()], FIRST_TOLERANCE);

    assert_complex_close(actual.axis1()[()], expected.axis1()[()], FIRST_TOLERANCE);

    assert_complex_close(
        actual.axis0_axis0()[()],
        expected.axis0_axis0()[()],
        SECOND_TOLERANCE,
    );

    assert_complex_close(
        actual.axis0_axis1()[()],
        expected.axis0_axis1()[()],
        SECOND_TOLERANCE,
    );

    assert_complex_close(
        actual.axis1_axis1()[()],
        expected.axis1_axis1()[()],
        SECOND_TOLERANCE,
    );
}

pub(crate) fn assert_zero_waves_close(
    actual: &BidirectionalWaves<ZeroJet>,
    expected: &BidirectionalWaves<ZeroJet>,
    tolerance: f64,
) {
    assert_complex_close(actual.forward()[()], expected.forward()[()], tolerance);

    assert_complex_close(actual.backward()[()], expected.backward()[()], tolerance);
}

pub(crate) fn assert_zero_layer_close(
    actual: &LayerBoundaryWaves<ZeroJet>,
    expected: &LayerBoundaryWaves<ZeroJet>,
    tolerance: f64,
) {
    assert_zero_waves_close(actual.left(), expected.left(), tolerance);

    assert_zero_waves_close(actual.right(), expected.right(), tolerance);
}

pub(crate) fn assert_zero_layers_close(
    actual: &[LayerBoundaryWaves<ZeroJet>],
    expected: &[LayerBoundaryWaves<ZeroJet>],
    tolerance: f64,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "backends returned different layer counts",
    );

    for (layer_index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_zero_layer_close(actual, expected, tolerance);
    }
}

pub(crate) fn assert_first_waves_close(
    actual: &BidirectionalWaves<FirstJet>,
    expected: &BidirectionalWaves<FirstJet>,
) {
    assert_first_jet_close(actual.forward(), expected.forward());

    assert_first_jet_close(actual.backward(), expected.backward());
}

pub(crate) fn assert_second_waves_close(
    actual: &BidirectionalWaves<SecondJet>,
    expected: &BidirectionalWaves<SecondJet>,
) {
    assert_second_jet_close(actual.forward(), expected.forward());

    assert_second_jet_close(actual.backward(), expected.backward());
}

pub(crate) fn assert_bivariate_first_waves_close(
    actual: &BidirectionalWaves<BivariateFirstJet>,
    expected: &BidirectionalWaves<BivariateFirstJet>,
) {
    assert_bivariate_first_jet_close(actual.forward(), expected.forward());

    assert_bivariate_first_jet_close(actual.backward(), expected.backward());
}

pub(crate) fn assert_bivariate_second_waves_close(
    actual: &BidirectionalWaves<BivariateSecondJet>,
    expected: &BidirectionalWaves<BivariateSecondJet>,
) {
    assert_bivariate_second_jet_close(actual.forward(), expected.forward());

    assert_bivariate_second_jet_close(actual.backward(), expected.backward());
}

pub(crate) fn assert_first_layers_close(
    actual: &[LayerBoundaryWaves<FirstJet>],
    expected: &[LayerBoundaryWaves<FirstJet>],
) {
    assert_eq!(actual.len(), expected.len());

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_first_waves_close(actual.left(), expected.left());

        assert_first_waves_close(actual.right(), expected.right());
    }
}

pub(crate) fn assert_second_layers_close(
    actual: &[LayerBoundaryWaves<SecondJet>],
    expected: &[LayerBoundaryWaves<SecondJet>],
) {
    assert_eq!(actual.len(), expected.len());

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_second_waves_close(actual.left(), expected.left());

        assert_second_waves_close(actual.right(), expected.right());
    }
}

pub(crate) fn assert_bivariate_first_layers_close(
    actual: &[LayerBoundaryWaves<BivariateFirstJet>],
    expected: &[LayerBoundaryWaves<BivariateFirstJet>],
) {
    assert_eq!(actual.len(), expected.len());

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_bivariate_first_waves_close(actual.left(), expected.left());

        assert_bivariate_first_waves_close(actual.right(), expected.right());
    }
}

pub(crate) fn assert_bivariate_second_layers_close(
    actual: &[LayerBoundaryWaves<BivariateSecondJet>],
    expected: &[LayerBoundaryWaves<BivariateSecondJet>],
) {
    assert_eq!(actual.len(), expected.len());

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_bivariate_second_waves_close(actual.left(), expected.left());

        assert_bivariate_second_waves_close(actual.right(), expected.right());
    }
}
