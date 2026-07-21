use super::support::C;
use crate::backend::{
    PlaneWaveFieldSample,
    tests::support::{
        assert_complex_derivative_close, assert_complex_second_derivative_close,
        central_first_complex, central_second_complex,
    },
};

use ndarray::Ix0;

#[derive(Clone, Copy, Debug)]
pub(crate) enum TestDerivativeOrder {
    First,
    Second,
}

pub(crate) fn check_complex_field_derivative_against_finite_difference<
    ValueSample,
    FirstSample,
    SecondSample,
    SolveValue,
    SolveFirst,
    SolveSecond,
    ExtractValue,
    ExtractFirst,
    ExtractSecond,
>(
    parameter: f64,
    step: f64,
    order: TestDerivativeOrder,
    mut solve_value: SolveValue,
    mut solve_first: SolveFirst,
    mut solve_second: SolveSecond,
    mut extract_value: ExtractValue,
    mut extract_first: ExtractFirst,
    mut extract_second: ExtractSecond,
) where
    SolveValue: FnMut(f64) -> ValueSample,
    SolveFirst: FnMut(f64) -> FirstSample,
    SolveSecond: FnMut(f64) -> SecondSample,
    ExtractValue: FnMut(&ValueSample) -> C,
    ExtractFirst: FnMut(&FirstSample) -> C,
    ExtractSecond: FnMut(&SecondSample) -> C,
{
    match order {
        TestDerivativeOrder::First => {
            let analytic_sample = solve_first(parameter);
            let analytic = extract_first(&analytic_sample);

            let numerical = central_first_complex(
                |value| {
                    let sample = solve_value(value);
                    extract_value(&sample)
                },
                parameter,
                step,
            );

            assert_complex_derivative_close(analytic, numerical);
        }

        TestDerivativeOrder::Second => {
            let analytic_sample = solve_second(parameter);
            let analytic = extract_second(&analytic_sample);

            let numerical_from_values = central_second_complex(
                |value| {
                    let sample = solve_value(value);
                    extract_value(&sample)
                },
                parameter,
                step,
            );

            let numerical_from_first = central_first_complex(
                |value| {
                    let sample = solve_first(value);
                    extract_first(&sample)
                },
                parameter,
                step,
            );

            assert_complex_second_derivative_close(analytic, numerical_from_values);

            assert_complex_second_derivative_close(analytic, numerical_from_first);
        }
    }
}
pub(crate) fn check_complex_field_derivative_against_finite_difference<
    SolveValue,
    SolveFirst,
    SolveSecond,
    ExtractValue,
    ExtractFirst,
    ExtractSecond,
>(
    parameter: f64,
    step: f64,
    order: TestDerivativeOrder,
    mut solve_value: SolveValue,
    mut solve_first: SolveFirst,
    mut solve_second: SolveSecond,
    mut extract_value: ExtractValue,
    mut extract_first: ExtractFirst,
    mut extract_second: ExtractSecond,
) where
    SolveValue: FnMut(f64) -> PlaneWaveFieldSample<C, Ix0>,
    SolveFirst: FnMut(f64) -> PlaneWaveFieldSample<C, Ix0>,
    SolveSecond: FnMut(f64) -> PlaneWaveFieldSample<C, Ix0>,
    ExtractValue: FnMut(&PlaneWaveFieldSample<C, Ix0>) -> C,
    ExtractFirst: FnMut(&PlaneWaveFieldSample<C, Ix0>) -> C,
    ExtractSecond: FnMut(&PlaneWaveFieldSample<C, Ix0>) -> C,
{
    match order {
        TestDerivativeOrder::First => {
            let analytic_sample = solve_first(parameter);
            let analytic = extract_first(&analytic_sample);

            let numerical = central_first_complex(
                |value| {
                    let sample = solve_value(value);
                    extract_value(&sample)
                },
                parameter,
                step,
            );

            assert_complex_derivative_close(analytic, numerical);
        }

        TestDerivativeOrder::Second => {
            let analytic_sample = solve_second(parameter);
            let analytic = extract_second(&analytic_sample);

            let numerical_from_values = central_second_complex(
                |value| {
                    let sample = solve_value(value);
                    extract_value(&sample)
                },
                parameter,
                step,
            );

            let numerical_from_first = central_first_complex(
                |value| {
                    let sample = solve_first(value);
                    extract_first(&sample)
                },
                parameter,
                step,
            );

            assert_complex_second_derivative_close(analytic, numerical_from_values);

            assert_complex_second_derivative_close(analytic, numerical_from_first);
        }
    }
}
