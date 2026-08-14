//! Numerically stable relative-exponential functions.
//!
//! This module evaluates
//!
//! ```text
//! exprel(z) = (exp(z) - 1) / z
//! ```
//!
//! together with its first two derivatives.
//!
//! Direct evaluation loses precision near `z = 0` through cancellation, so
//! local Taylor series are used below order-dependent thresholds.
//!
//! The functions are used in jet propagation for expressions whose analytic
//! limit at zero must be retained exactly.

use nalgebra::ComplexField;
use num_traits::{FromPrimitive, float::FloatCore};

/// Evaluate `(exp(z) - 1) / z` using a stable series near zero.
pub(crate) fn exprel<C>(z: C) -> C
where
    C: ComplexField + Copy,
    C::RealField: FromPrimitive + FloatCore,
{
    let threshold = <C::RealField as FloatCore>::epsilon().powf(
        <C::RealField as FromPrimitive>::from_f64(1.0 / 4.0).expect("1/4 must be representable"),
    );

    if z.modulus() < threshold {
        exprel_series(z)
    } else {
        (z.exp() - C::one()) / z
    }
}

/// Evaluate the first derivative of [`exprel`].
pub(crate) fn exprel_first<C>(z: C) -> C
where
    C: ComplexField + Copy,
    C::RealField: FloatCore + FromPrimitive,
{
    let threshold = <C::RealField as FloatCore>::epsilon().powf(
        <C::RealField as FromPrimitive>::from_f64(1.0 / 5.0).expect("1/5 must be representable"),
    );

    if z.modulus() < threshold {
        exprel_first_series(z)
    } else {
        let one = C::one();

        (z.exp() * (z - one) + one) / (z * z)
    }
}

/// Evaluate the second derivative of [`exprel`].
pub(crate) fn exprel_second<C>(z: C) -> C
where
    C: ComplexField + Copy,
    C::RealField: FloatCore + FromPrimitive,
{
    let threshold = <C::RealField as FloatCore>::epsilon().powf(
        <C::RealField as FromPrimitive>::from_f64(1.0 / 6.0).expect("1/6 must be representable"),
    );

    if z.modulus() < threshold {
        exprel_second_series(z)
    } else {
        let one = C::one();
        let two = one + one;

        /*
         * f''(z) =
         * [exp(z)(z² - 2z + 2) - 2] / z³
         */
        (z.exp() * (z * z - two * z + two) - two) / (z * z * z)
    }
}

fn from_f64<C>(value: f64) -> C
where
    C: ComplexField,
    C::RealField: FromPrimitive,
{
    C::from_real(C::RealField::from_f64(value).expect("coefficient must be representable"))
}

fn exprel_series<C>(z: C) -> C
where
    C: ComplexField + Copy,
    C::RealField: FromPrimitive,
{
    from_f64::<C>(1.0)
        + z * (from_f64::<C>(1.0 / 2.0)
            + z * (from_f64::<C>(1.0 / 6.0)
                + z * (from_f64::<C>(1.0 / 24.0)
                    + z * (from_f64::<C>(1.0 / 120.0)
                        + z * (from_f64::<C>(1.0 / 720.0)
                            + z * (from_f64::<C>(1.0 / 5040.0)
                                + z * from_f64::<C>(1.0 / 40320.0)))))))
}

fn exprel_first_series<C>(z: C) -> C
where
    C: ComplexField + Copy,
    C::RealField: FromPrimitive,
{
    from_f64::<C>(1.0 / 2.0)
        + z * (from_f64::<C>(1.0 / 3.0)
            + z * (from_f64::<C>(1.0 / 8.0)
                + z * (from_f64::<C>(1.0 / 30.0)
                    + z * (from_f64::<C>(1.0 / 144.0)
                        + z * (from_f64::<C>(1.0 / 840.0) + z * from_f64::<C>(1.0 / 5760.0))))))
}

fn exprel_second_series<C>(z: C) -> C
where
    C: ComplexField + Copy,
    C::RealField: FromPrimitive,
{
    from_f64::<C>(1.0 / 3.0)
        + z * (from_f64::<C>(1.0 / 4.0)
            + z * (from_f64::<C>(1.0 / 10.0)
                + z * (from_f64::<C>(1.0 / 36.0)
                    + z * (from_f64::<C>(1.0 / 168.0) + z * from_f64::<C>(1.0 / 960.0)))))
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    use super::{exprel, exprel_first, exprel_second};

    const TOLERANCE: f64 = 2.0e-13;

    fn assert_complex_close(actual: Complex64, expected: Complex64, tolerance: f64) {
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

    fn defining_exprel(z: Complex64) -> Complex64 {
        (z.exp() - Complex64::new(1.0, 0.0)) / z
    }

    fn defining_exprel_first(z: Complex64) -> Complex64 {
        let one = Complex64::new(1.0, 0.0);

        (z.exp() * (z - one) + one) / (z * z)
    }

    fn defining_exprel_second(z: Complex64) -> Complex64 {
        let two = Complex64::new(2.0, 0.0);

        (z.exp() * (z * z - two * z + two) - two) / (z * z * z)
    }

    #[test]
    fn exprel_has_correct_values_at_zero() {
        assert_eq!(exprel(0.0_f64), 1.0);
        assert_eq!(exprel_first(0.0_f64), 0.5);
        assert_eq!(exprel_second(0.0_f64), 1.0 / 3.0);

        assert_eq!(exprel(Complex64::new(0.0, 0.0)), Complex64::new(1.0, 0.0),);

        assert_eq!(
            exprel_first(Complex64::new(0.0, 0.0)),
            Complex64::new(0.5, 0.0),
        );

        assert_eq!(
            exprel_second(Complex64::new(0.0, 0.0)),
            Complex64::new(1.0 / 3.0, 0.0),
        );
    }

    #[test]
    fn exprel_real_matches_defining_expression_away_from_zero() {
        for z in [-2.0_f64, -0.4, 0.2, 1.7] {
            assert_relative_eq!(
                exprel(z),
                z.exp_m1() / z,
                epsilon = TOLERANCE,
                max_relative = TOLERANCE,
            );
        }
    }

    #[test]
    fn exprel_complex_matches_defining_expression_away_from_zero() {
        for z in [
            Complex64::new(0.7, -0.4),
            Complex64::new(-0.3, 1.1),
            Complex64::new(1.5, 0.2),
        ] {
            assert_complex_close(exprel(z), defining_exprel(z), TOLERANCE);
        }
    }

    #[test]
    fn derivatives_match_closed_forms_away_from_zero() {
        for z in [
            Complex64::new(0.7, -0.4),
            Complex64::new(-0.3, 1.1),
            Complex64::new(1.5, 0.2),
        ] {
            assert_complex_close(exprel_first(z), defining_exprel_first(z), 5.0e-13);

            assert_complex_close(exprel_second(z), defining_exprel_second(z), 2.0e-12);
        }
    }

    #[test]
    fn small_real_arguments_use_stable_limits() {
        let z = 1.0e-14_f64;

        assert_relative_eq!(
            exprel(z),
            1.0 + z / 2.0,
            epsilon = 1.0e-15,
            max_relative = 1.0e-15,
        );

        assert_relative_eq!(
            exprel_first(z),
            0.5 + z / 3.0,
            epsilon = 1.0e-15,
            max_relative = 1.0e-15,
        );

        assert_relative_eq!(
            exprel_second(z),
            1.0 / 3.0 + z / 4.0,
            epsilon = 1.0e-15,
            max_relative = 1.0e-15,
        );
    }

    #[test]
    fn small_complex_arguments_match_local_series() {
        let z = Complex64::new(1.0e-10, -2.0e-10);

        let expected = Complex64::new(1.0, 0.0) + z / 2.0 + z * z / 6.0 + z * z * z / 24.0;

        let expected_first = Complex64::new(0.5, 0.0) + z / 3.0 + z * z / 8.0 + z * z * z / 30.0;

        let expected_second =
            Complex64::new(1.0 / 3.0, 0.0) + z / 4.0 + z * z / 10.0 + z * z * z / 36.0;

        assert_complex_close(exprel(z), expected, 1.0e-15);
        assert_complex_close(exprel_first(z), expected_first, 1.0e-15);
        assert_complex_close(exprel_second(z), expected_second, 1.0e-15);
    }

    #[test]
    fn functions_respect_complex_conjugation() {
        let z = Complex64::new(0.7, -0.4);

        assert_complex_close(exprel(z.conj()), exprel(z).conj(), TOLERANCE);

        assert_complex_close(exprel_first(z.conj()), exprel_first(z).conj(), TOLERANCE);

        assert_complex_close(exprel_second(z.conj()), exprel_second(z).conj(), 2.0e-12);
    }

    #[test]
    fn first_derivative_matches_central_difference() {
        let z = Complex64::new(0.4, -0.3);
        let step = 1.0e-6;

        let finite_difference = (exprel(z + Complex64::new(step, 0.0))
            - exprel(z - Complex64::new(step, 0.0)))
            / (2.0 * step);

        assert_complex_close(exprel_first(z), finite_difference, 2.0e-8);
    }

    #[test]
    fn second_derivative_matches_central_difference() {
        let z = Complex64::new(0.4, -0.3);
        let step = 2.0e-4;

        let finite_difference = (exprel(z + Complex64::new(step, 0.0)) - 2.0 * exprel(z)
            + exprel(z - Complex64::new(step, 0.0)))
            / step.powi(2);

        assert_complex_close(exprel_second(z), finite_difference, 2.0e-8);
    }

    #[test]
    fn values_remain_continuous_across_numerical_branches() {
        let epsilon = f64::EPSILON;

        let value_threshold = epsilon.powf(1.0 / 4.0);
        let first_threshold = epsilon.powf(1.0 / 5.0);
        let second_threshold = epsilon.powf(1.0 / 6.0);

        for threshold in [value_threshold, first_threshold, second_threshold] {
            let below = Complex64::new(threshold * (1.0 - 1.0e-6), threshold * 0.2);

            let above = Complex64::new(threshold * (1.0 + 1.0e-6), threshold * 0.2);

            // The points are different, so compare against the analytic
            // functions rather than directly against each other.
            assert_complex_close(exprel(below), defining_exprel(below), 2.0e-12);

            assert_complex_close(exprel(above), defining_exprel(above), 2.0e-12);
        }
    }
}

#[cfg(test)]
mod jet_exprel_tests {
    use approx::assert_relative_eq;
    use ndarray::{Array1, arr1};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        algebra::{Jet0, Jet1, Jet2, JetBivariate1, JetBivariate2, RealParameter},
        differential::{BivariateGradient, BivariateHessian},
    };

    const TOLERANCE: f64 = 2.0e-12;

    fn assert_complex_close(actual: Complex64, expected: Complex64) {
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

    fn values() -> Array1<Complex64> {
        arr1(&[
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0e-12, -2.0e-12),
            Complex64::new(0.4, -0.2),
            Complex64::new(-0.7, 0.5),
        ])
    }

    fn first_seed() -> Array1<Complex64> {
        arr1(&[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.3, -0.1),
            Complex64::new(-0.2, 0.4),
            Complex64::new(0.7, 0.2),
        ])
    }

    fn second_seed() -> Array1<Complex64> {
        arr1(&[
            Complex64::new(0.2, -0.1),
            Complex64::new(-0.4, 0.2),
            Complex64::new(0.6, 0.3),
            Complex64::new(-0.1, 0.5),
        ])
    }

    #[test]
    fn exprel_maps_every_value_elementwise() {
        let input = values();
        let jet: Jet0<_, RealParameter> = Jet0::new(input.clone());

        let actual = jet.exprel();

        for (actual, input) in actual.value().iter().zip(input.iter()) {
            assert_complex_close(*actual, exprel(*input));
        }
    }

    #[test]
    fn exprel_first_jet_applies_chain_rule_elementwise() {
        let value = values();
        let first = first_seed();

        let jet: Jet1<_, RealParameter> = Jet1::from_parts(value.clone(), first.clone());

        let actual = jet.exprel();

        for index in 0..value.len() {
            assert_complex_close(actual.value()[index], exprel(value[index]));

            assert_complex_close(
                actual.first()[index],
                exprel_first(value[index]) * first[index],
            );
        }
    }

    #[test]
    fn exprel_first_jet_is_correct_at_zero() {
        let jet: Jet1<_, RealParameter> = Jet1::from_parts(
            arr1(&[Complex64::new(0.0, 0.0)]),
            arr1(&[Complex64::new(2.0, -1.0)]),
        );

        let actual = jet.exprel();

        assert_complex_close(actual.value()[0], Complex64::new(1.0, 0.0));

        assert_complex_close(actual.first()[0], Complex64::new(1.0, -0.5));
    }

    #[test]
    fn exprel_second_jet_applies_second_order_chain_rule() {
        let value = values();
        let first = first_seed();
        let second = second_seed();

        let jet: Jet2<_, RealParameter> =
            Jet2::from_parts(value.clone(), first.clone(), second.clone());

        let actual = jet.exprel();

        for index in 0..value.len() {
            let expected_first = exprel_first(value[index]) * first[index];

            let expected_second = exprel_second(value[index]) * first[index] * first[index]
                + exprel_first(value[index]) * second[index];

            assert_complex_close(actual.value()[index], exprel(value[index]));

            assert_complex_close(actual.first()[index], expected_first);

            assert_complex_close(actual.second()[index], expected_second);
        }
    }

    #[test]
    fn exprel_second_jet_has_correct_derivatives_at_zero() {
        let first = Complex64::new(2.0, -1.0);
        let second = Complex64::new(0.3, 0.4);

        let jet: Jet2<_, RealParameter> = Jet2::from_parts(
            arr1(&[Complex64::new(0.0, 0.0)]),
            arr1(&[first]),
            arr1(&[second]),
        );

        let actual = jet.exprel();

        assert_complex_close(actual.first()[0], first / 2.0);

        assert_complex_close(actual.second()[0], first * first / 3.0 + second / 2.0);
    }

    #[test]
    fn exprel_bivariate_first_applies_chain_rule_to_both_axes() {
        let value = values();

        let axis0 = first_seed();

        let axis1 = arr1(&[
            Complex64::new(-0.3, 0.2),
            Complex64::new(0.5, 0.1),
            Complex64::new(0.2, -0.6),
            Complex64::new(-0.4, 0.3),
        ]);

        let jet: JetBivariate1<_, RealParameter> = JetBivariate1::from_parts(
            value.clone(),
            BivariateGradient::new(axis0.clone(), axis1.clone()),
        );

        let actual = jet.exprel();

        for index in 0..value.len() {
            let derivative = exprel_first(value[index]);

            assert_complex_close(actual.value()[index], exprel(value[index]));

            assert_complex_close(actual.axis0()[index], derivative * axis0[index]);

            assert_complex_close(actual.axis1()[index], derivative * axis1[index]);
        }
    }

    #[test]
    fn exprel_bivariate_second_applies_full_hessian_chain_rule() {
        let value = values();

        let axis0 = first_seed();

        let axis1 = arr1(&[
            Complex64::new(-0.3, 0.2),
            Complex64::new(0.5, 0.1),
            Complex64::new(0.2, -0.6),
            Complex64::new(-0.4, 0.3),
        ]);

        let axis0_axis0 = second_seed();

        let axis0_axis1 = arr1(&[
            Complex64::new(0.1, 0.3),
            Complex64::new(-0.2, 0.4),
            Complex64::new(0.5, -0.1),
            Complex64::new(0.2, 0.6),
        ]);

        let axis1_axis1 = arr1(&[
            Complex64::new(-0.2, 0.1),
            Complex64::new(0.3, 0.5),
            Complex64::new(-0.4, -0.2),
            Complex64::new(0.6, 0.1),
        ]);

        let jet: JetBivariate2<_, RealParameter> = JetBivariate2::from_parts(
            value.clone(),
            BivariateGradient::new(axis0.clone(), axis1.clone()),
            BivariateHessian::new(
                axis0_axis0.clone(),
                axis0_axis1.clone(),
                axis1_axis1.clone(),
            ),
        );

        let actual = jet.exprel();

        for index in 0..value.len() {
            let first_factor = exprel_first(value[index]);

            let second_factor = exprel_second(value[index]);

            assert_complex_close(actual.value()[index], exprel(value[index]));

            assert_complex_close(actual.axis0()[index], first_factor * axis0[index]);

            assert_complex_close(actual.axis1()[index], first_factor * axis1[index]);

            assert_complex_close(
                actual.axis0_axis0()[index],
                second_factor * axis0[index] * axis0[index] + first_factor * axis0_axis0[index],
            );

            assert_complex_close(
                actual.axis0_axis1()[index],
                second_factor * axis0[index] * axis1[index] + first_factor * axis0_axis1[index],
            );

            assert_complex_close(
                actual.axis1_axis1()[index],
                second_factor * axis1[index] * axis1[index] + first_factor * axis1_axis1[index],
            );
        }
    }
}
