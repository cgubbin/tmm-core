use super::{C, TOLERANCE, c};

use approx::assert_relative_eq;
use ndarray::{ArrayBase, Data, Dimension};

fn assert_real_close(actual: f64, expected: f64) {
    let error = (actual - expected).abs();

    assert!(
        error <= TOLERANCE,
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
