use crate::{
    algebra::ScalarAlgebra,
    observable::{
        BoundaryState, BoundaryWaves, InterfaceStates, Interfaces, LayerBoundaries,
        LayerBoundaryStates, LayerBoundaryWaves,
    },
    test_support::jet::{HoloJ0, RealJ0, RealJ1, RealJ2, RealJB1, RealJB2},
    waves::BidirectionalWaves,
};

use super::{C, c};

use approx::assert_relative_eq;
use ndarray::{ArrayBase, Data, Dimension, Ix0, OwnedRepr};

pub fn assert_real_zero(actual: f64, tolerance: f64) {
    assert!(
        actual.abs() <= tolerance,
        "expected 0.0, got {actual:e}; \
             absolute error = {actual:e}",
    );
}

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

pub fn assert_real_array_close<D>(
    actual: &ArrayBase<impl Data<Elem = f64>, D>,
    expected: &ArrayBase<impl Data<Elem = f64>, D>,
    tolerance: f64,
) where
    D: Dimension,
{
    assert_eq!(actual.raw_dim(), expected.raw_dim());

    for (&actual, &expected) in actual.iter().zip(expected.iter()) {
        assert_real_close(actual, expected, tolerance);
    }
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

pub(crate) fn assert_boundary_waves_close<A>(
    actual: &BoundaryWaves<A>,
    expected: &BoundaryWaves<A>,
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
    assert_boundary_waves_close(actual.left(), expected.left(), tolerance);

    assert_boundary_waves_close(actual.right(), expected.right(), tolerance);
}

pub(crate) const VALUE_TOLERANCE: f64 = 1.0e-11;
pub(crate) const FIRST_TOLERANCE: f64 = 1.0e-10;
pub(crate) const SECOND_TOLERANCE: f64 = 1.0e-9;

pub(crate) fn assert_zero_jet_close(actual: &RealJ0, expected: &RealJ0) {
    assert_complex_close(actual.value()[()], expected.value()[()], VALUE_TOLERANCE);
}

pub(crate) fn assert_holo_zero_jet_close(actual: &HoloJ0, expected: &HoloJ0) {
    assert_complex_close(actual.value()[()], expected.value()[()], VALUE_TOLERANCE);
}

pub(crate) fn assert_zero_jet_zero(actual: &RealJ0) {
    assert_complex_close(actual.value()[()], c(0.0), VALUE_TOLERANCE);
}

pub(crate) fn assert_first_jet_close(actual: &RealJ1, expected: &RealJ1) {
    assert_complex_close(actual.value()[()], expected.value()[()], VALUE_TOLERANCE);

    assert_complex_close(actual.first()[()], expected.first()[()], FIRST_TOLERANCE);
}

pub(crate) fn assert_second_jet_close(actual: &RealJ2, expected: &RealJ2) {
    assert_complex_close(actual.value()[()], expected.value()[()], VALUE_TOLERANCE);

    assert_complex_close(actual.first()[()], expected.first()[()], FIRST_TOLERANCE);

    assert_complex_close(actual.second()[()], expected.second()[()], SECOND_TOLERANCE);
}

pub(crate) fn assert_bivariate_first_jet_close(actual: &RealJB1, expected: &RealJB1) {
    assert_complex_close(actual.value()[()], expected.value()[()], VALUE_TOLERANCE);

    assert_complex_close(actual.axis0()[()], expected.axis0()[()], FIRST_TOLERANCE);

    assert_complex_close(actual.axis1()[()], expected.axis1()[()], FIRST_TOLERANCE);
}

pub(crate) fn assert_bivariate_second_jet_close(actual: &RealJB2, expected: &RealJB2) {
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
    actual: &BoundaryWaves<RealJ0>,
    expected: &BoundaryWaves<RealJ0>,
    tolerance: f64,
) {
    assert_complex_close(actual.forward()[()], expected.forward()[()], tolerance);

    assert_complex_close(actual.backward()[()], expected.backward()[()], tolerance);
}

pub(crate) fn assert_zero_layer_close(
    actual: &LayerBoundaryWaves<RealJ0>,
    expected: &LayerBoundaryWaves<RealJ0>,
    tolerance: f64,
) {
    assert_zero_waves_close(actual.left(), expected.left(), tolerance);

    assert_zero_waves_close(actual.right(), expected.right(), tolerance);
}

pub(crate) fn assert_zero_layers_close(
    actual: &[LayerBoundaryWaves<RealJ0>],
    expected: &[LayerBoundaryWaves<RealJ0>],
    tolerance: f64,
) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "backends returned different layer counts",
    );

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_zero_layer_close(actual, expected, tolerance);
    }
}

pub(crate) fn assert_first_waves_close(
    actual: &BoundaryWaves<RealJ1>,
    expected: &BoundaryWaves<RealJ1>,
) {
    assert_first_jet_close(actual.forward(), expected.forward());

    assert_first_jet_close(actual.backward(), expected.backward());
}

pub(crate) fn assert_second_waves_close(
    actual: &BoundaryWaves<RealJ2>,
    expected: &BoundaryWaves<RealJ2>,
) {
    assert_second_jet_close(actual.forward(), expected.forward());

    assert_second_jet_close(actual.backward(), expected.backward());
}

pub(crate) fn assert_bivariate_first_waves_close(
    actual: &BoundaryWaves<RealJB1>,
    expected: &BoundaryWaves<RealJB1>,
) {
    assert_bivariate_first_jet_close(actual.forward(), expected.forward());

    assert_bivariate_first_jet_close(actual.backward(), expected.backward());
}

pub(crate) fn assert_bivariate_second_waves_close(
    actual: &BoundaryWaves<RealJB2>,
    expected: &BoundaryWaves<RealJB2>,
) {
    assert_bivariate_second_jet_close(actual.forward(), expected.forward());

    assert_bivariate_second_jet_close(actual.backward(), expected.backward());
}

pub(crate) fn assert_first_layers_close(
    actual: &[LayerBoundaryWaves<RealJ1>],
    expected: &[LayerBoundaryWaves<RealJ1>],
) {
    assert_eq!(actual.len(), expected.len());

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_first_waves_close(actual.left(), expected.left());

        assert_first_waves_close(actual.right(), expected.right());
    }
}

pub(crate) fn assert_second_layers_close(
    actual: &[LayerBoundaryWaves<RealJ2>],
    expected: &[LayerBoundaryWaves<RealJ2>],
) {
    assert_eq!(actual.len(), expected.len());

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_second_waves_close(actual.left(), expected.left());

        assert_second_waves_close(actual.right(), expected.right());
    }
}

pub(crate) fn assert_bivariate_first_layers_close(
    actual: &[LayerBoundaryWaves<RealJB1>],
    expected: &[LayerBoundaryWaves<RealJB1>],
) {
    assert_eq!(actual.len(), expected.len());

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_bivariate_first_waves_close(actual.left(), expected.left());

        assert_bivariate_first_waves_close(actual.right(), expected.right());
    }
}

pub(crate) fn assert_bivariate_second_layers_close(
    actual: &[LayerBoundaryWaves<RealJB2>],
    expected: &[LayerBoundaryWaves<RealJB2>],
) {
    assert_eq!(actual.len(), expected.len());

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_bivariate_second_waves_close(actual.left(), expected.left());

        assert_bivariate_second_waves_close(actual.right(), expected.right());
    }
}

pub(crate) type ValueArray = ArrayBase<OwnedRepr<C>, Ix0>;

pub(crate) fn assert_interface_continuity(
    interfaces: &Interfaces<InterfaceStates<ValueArray>>,
    tolerance: f64,
) {
    assert!(
        !interfaces.is_empty(),
        "a valid stack must contain at least one interface",
    );

    for interface in interfaces.iter() {
        assert_boundary_state_close(interface.left(), interface.right(), tolerance);
    }
}

pub(crate) fn assert_boundary_state_jet_close(
    actual: &BoundaryState<RealJ0>,
    expected: &BoundaryState<RealJ0>,
    tolerance: f64,
) {
    assert_array_close(actual.field(), expected.field(), tolerance);
    assert_array_close(actual.secondary(), expected.secondary(), tolerance);
}

pub(crate) fn assert_boundary_state_close(
    actual: &BoundaryState<ValueArray>,
    expected: &BoundaryState<ValueArray>,
    tolerance: f64,
) {
    assert_array_close(actual.field(), expected.field(), tolerance);
    assert_array_close(actual.secondary(), expected.secondary(), tolerance);
}

pub(crate) fn assert_boundary_waves_close_arr(
    actual: &BoundaryWaves<ValueArray>,
    expected: &BoundaryWaves<ValueArray>,
    tolerance: f64,
) {
    assert_array_close(actual.forward(), expected.forward(), tolerance);
    assert_array_close(actual.backward(), expected.backward(), tolerance);
}

pub(crate) fn assert_layer_boundary_waves_close_arr(
    actual: &LayerBoundaryWaves<ValueArray>,
    expected: &LayerBoundaryWaves<ValueArray>,
    tolerance: f64,
) {
    assert_boundary_waves_close_arr(actual.left(), expected.left(), tolerance);
    assert_boundary_waves_close_arr(actual.right(), expected.right(), tolerance);
}

pub(crate) fn assert_layer_boundary_states_close(
    actual: &LayerBoundaryStates<ValueArray>,
    expected: &LayerBoundaryStates<ValueArray>,
    tolerance: f64,
) {
    assert_boundary_state_close(actual.left(), expected.left(), tolerance);

    assert_boundary_state_close(actual.right(), expected.right(), tolerance);
}

pub(crate) fn assert_layer_waves_collection_close(
    actual: &LayerBoundaries<LayerBoundaryWaves<ValueArray>>,
    expected: &LayerBoundaries<LayerBoundaryWaves<ValueArray>>,
    tolerance: f64,
) {
    assert_eq!(actual.len(), expected.len());

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_layer_boundary_waves_close_arr(actual, expected, tolerance);
    }
}

pub(crate) fn assert_layer_states_collection_close(
    actual: &LayerBoundaries<LayerBoundaryStates<ValueArray>>,
    expected: &LayerBoundaries<LayerBoundaryStates<ValueArray>>,
    tolerance: f64,
) {
    assert_eq!(actual.len(), expected.len());

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_layer_boundary_states_close(actual, expected, tolerance);
    }
}
