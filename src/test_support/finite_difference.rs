use super::C;

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
