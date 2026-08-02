use crate::{
    algebra::{ArrayJet1, ArrayJet2},
    test_support::assertions::assert_array_close,
};

use super::{C, c, jet::P};

use ndarray::{Array, ArrayBase, Data, Dimension};

pub const VALUE_TOLERANCE: f64 = 1e-12;
pub const FIRST_DERIVATIVE_TOLERANCE: f64 = 1e-7;
pub const SECOND_DERIVATIVE_TOLERANCE: f64 = 1e-5;

/*
 * A relatively large step is appropriate for the second central
 * difference because its truncation error is O(h²), while round-off is
 * amplified by 1/h².
 */
pub const FIRST_DIFFERENCE_STEP: f64 = 1e-6;
pub const SECOND_DIFFERENCE_STEP: f64 = 1e-4;

/// Evaluate a first derivative using the centred finite-difference formula
///
/// ```text
/// f'(x) ≈ [f(x + h) - f(x - h)] / 2h.
/// ```
pub fn central_first_difference(function: impl Fn(f64) -> C, x: f64, step: f64) -> C {
    assert!(step.is_finite());
    assert!(step > 0.0);

    (function(x + step) - function(x - step)) / (2.0 * step)
}

/// Evaluate a second derivative using the centred finite-difference formula
///
/// ```text
/// f''(x) ≈ [f(x + h) - 2f(x) + f(x - h)] / h².
/// ```
pub fn central_second_difference(function: impl Fn(f64) -> C, x: f64, step: f64) -> C {
    assert!(step.is_finite());
    assert!(step > 0.0);

    let centre = function(x);

    (function(x + step) - centre * 2.0 + function(x - step)) / step.powi(2)
}

/// Apply a centred first difference pointwise to an array-valued function.
pub fn central_first_difference_array<D>(
    function: impl Fn(f64) -> Array<C, D>,
    x: f64,
    step: f64,
) -> Array<C, D>
where
    D: Dimension,
{
    assert!(step.is_finite());
    assert!(step > 0.0);

    let upper = function(x + step);
    let lower = function(x - step);

    (upper - lower) / c(2.0 * step)
}

/// Apply a centred second difference pointwise to an array-valued function.
pub fn central_second_difference_array<D>(
    function: impl Fn(f64) -> Array<C, D>,
    x: f64,
    step: f64,
) -> Array<C, D>
where
    D: Dimension,
{
    assert!(step.is_finite());
    assert!(step > 0.0);

    let upper = function(x + step);
    let centre = function(x);
    let lower = function(x - step);

    (upper - centre * c(2.0) + lower) / c(step.powi(2))
}

/// Assert the value and first derivative carried by a first-order jet.
pub fn assert_first_order_jet_close<D>(
    actual: &ArrayJet1<C, D, P>,
    expected_value: &ArrayBase<impl Data<Elem = C>, D>,
    expected_first: &ArrayBase<impl Data<Elem = C>, D>,
) where
    D: Dimension,
{
    assert_array_close(actual.value(), expected_value, VALUE_TOLERANCE);

    assert_array_close(actual.first(), expected_first, FIRST_DERIVATIVE_TOLERANCE);
}

/// Assert the value, first derivative, and second derivative carried by a
/// second-order jet.
pub fn assert_second_order_jet_close<D>(
    actual: &ArrayJet2<C, D, P>,
    expected_value: &ArrayBase<impl Data<Elem = C>, D>,
    expected_first: &ArrayBase<impl Data<Elem = C>, D>,
    expected_second: &ArrayBase<impl Data<Elem = C>, D>,
) where
    D: Dimension,
{
    assert_array_close(actual.value(), expected_value, VALUE_TOLERANCE);

    assert_array_close(actual.first(), expected_first, FIRST_DERIVATIVE_TOLERANCE);

    assert_array_close(
        actual.second(),
        expected_second,
        SECOND_DERIVATIVE_TOLERANCE,
    );
}

/// Construct a first-order jet representing the independent variable
///
/// ```text
/// x(λ) = value + λ,
/// ```
///
/// so that `dx/dλ = 1`.
pub fn independent_first<D>(value: Array<C, D>) -> ArrayJet1<C, D, P>
where
    D: Dimension,
{
    let first = Array::from_elem(value.raw_dim(), c(1.0));

    ArrayJet1::from_parts(value, first)
}

/// Construct a second-order jet representing the independent variable
///
/// ```text
/// x(λ) = value + λ,
/// ```
///
/// so that `dx/dλ = 1` and `d²x/dλ² = 0`.
pub fn independent_second<D>(value: Array<C, D>) -> ArrayJet2<C, D, P>
where
    D: Dimension,
{
    let first = Array::from_elem(value.raw_dim(), c(1.0));
    let second = Array::from_elem(value.raw_dim(), c(0.0));

    ArrayJet2::from_parts(value, first, second)
}

/// Construct a first-order jet that is constant with respect to the
/// differentiation parameter.
pub fn constant_first<D>(value: Array<C, D>) -> ArrayJet1<C, D, P>
where
    D: Dimension,
{
    let first = Array::from_elem(value.raw_dim(), c(0.0));

    ArrayJet1::from_parts(value, first)
}

/// Construct a second-order jet that is constant with respect to the
/// differentiation parameter.
pub fn constant_second<D>(value: Array<C, D>) -> ArrayJet2<C, D, P>
where
    D: Dimension,
{
    let first = Array::from_elem(value.raw_dim(), c(0.0));
    let second = Array::from_elem(value.raw_dim(), c(0.0));

    ArrayJet2::from_parts(value, first, second)
}
