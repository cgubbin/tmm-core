use crate::{
    ComplexScalar, Polarisation,
    algebra::ScalarAlgebra,
    backend::{RunMode, isotropic::IsotropicLayerQuantities, scatter2::Scatter2ExteriorContext},
    input::{CanonicalCoordinates, CanonicalSolverInput, CanonicalStack},
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};

use super::{Scatter2, Scatter2Entries, Scatter2Error, Scatter2Workspace};

use nalgebra::ComplexField;
use ndarray::Dimension;
use num_traits::{One, Zero};

impl Scatter2 {
    /// Construct the complete scattering representation of a planar stack.
    ///
    /// Components are accumulated from left to right:
    ///
    /// ```text
    /// exterior
    ///     │
    /// interface
    ///     │
    /// propagation
    ///     │
    /// interface
    ///     │
    /// propagation
    ///     │
    /// ...
    ///     │
    /// interface
    ///     │
    /// exterior
    /// ```
    ///
    /// For each finite layer:
    ///
    /// 1. evaluate the constitutive quantities;
    /// 2. compute the characteristic admittance;
    /// 3. construct the entrance interface;
    /// 4. construct propagation through the homogeneous layer;
    /// 5. append both to the accumulated Redheffer network.
    ///
    /// If internal fields were requested, each interface and propagation component
    /// is also retained so that forward and backward waves can later be
    /// reconstructed without repeating the scattering calculation.
    ///
    /// The returned workspace therefore contains either
    ///
    /// - only the accumulated scattering response, or
    /// - the accumulated response together with the component decomposition used
    ///   for field reconstruction.
    pub(crate) fn accumulate<J, E, M>(
        &self,
        coordinates: &CanonicalCoordinates<J>,
        stack: &CanonicalStack<M, J>,
        polarisation: Polarisation,
        request: RunMode,
    ) -> Result<Scatter2Workspace<J>, Scatter2Error>
    where
        J: ScalarAlgebra + ConstitutiveLift<E, M> + Clone,
        J::Scalar: ComplexScalar,
        <J::Scalar as ComplexField>::RealField: Copy,
        J::Dimension: Dimension,
        E: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
    {
        let context = Scatter2ExteriorContext::new(
            coordinates,
            stack.left_exterior(),
            stack.right_exterior(),
            polarisation,
        );

        let mut workspace = Scatter2Workspace::new(
            coordinates.vacuum_angular_wavenumber().value(),
            context,
            request,
            stack.layer_count(),
        );

        let left_quantities = IsotropicLayerQuantities::evaluate::<E, M>(
            stack.left_exterior(),
            coordinates,
            polarisation,
        );

        let mut current_admittance = left_quantities.into_admittance().into_inner();

        for layer in stack.layers() {
            let quantities = IsotropicLayerQuantities::evaluate::<E, M>(
                layer.material(),
                coordinates,
                polarisation,
            );

            let imaginary_unit = J::filled_constant_like(
                coordinates.vacuum_angular_wavenumber().value(),
                <J::Scalar as ComplexScalar>::i(),
            );

            let exponent = quantities
                .kappa()
                .multiply(&imaginary_unit)
                .multiply(layer.thickness_cm());

            let layer_admittance = quantities.into_admittance().into_inner();

            let interface = interface(&current_admittance, &layer_admittance);

            let propagation = propagation_from_exponent(exponent);

            workspace.append_layer(interface, propagation);

            current_admittance = layer_admittance;
        }

        let right_quantities = IsotropicLayerQuantities::evaluate::<E, M>(
            stack.right_exterior(),
            coordinates,
            polarisation,
        );

        let right_admittance = right_quantities.into_admittance().into_inner();

        let final_interface = interface(&current_admittance, &right_admittance);

        workspace.append(final_interface);

        Ok(workspace)
    }
}

/// Construct the scattering entries for an interface.
///
/// `left` and `right` are the physical characteristic admittances of the media
/// immediately to the left and right of the interface.
///
/// With the channel convention
///
/// ```text
/// [a_L^-]   [s11 s12] [a_L^+]
/// [a_R^+] = [s21 s22] [a_R^-],
/// ```
///
/// the entries are:
///
/// ```text
/// s11 = (Y_L - Y_R) / (Y_L + Y_R)
/// s12 = 2 Y_R / (Y_L + Y_R)
/// s21 = 2 Y_L / (Y_L + Y_R)
/// s22 = (Y_R - Y_L) / (Y_L + Y_R).
/// ```
///
/// The interface becomes singular when
///
/// YL + YR = 0,
///
/// corresponding to equal and opposite characteristic admittances.
///
/// No explicit check is performed; callers are expected to avoid singular
//  constitutive parameters
pub(crate) fn interface<A>(left: &A, right: &A) -> Scatter2Entries<A>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexField + One,
    A::Dimension: Dimension,
{
    let two = A::filled_constant_like(
        left.value(),
        <A::Scalar as One>::one() + <A::Scalar as One>::one(),
    );

    let denominator = left.add(right);

    Scatter2Entries {
        s11: left.subtract(right).divide(&denominator),

        s12: two.multiply(right).divide(&denominator),

        s21: two.multiply(left).divide(&denominator),

        s22: right.subtract(left).divide(&denominator),
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
///
///Reflection is identically zero, while forward and backward transmission are
// equal to exp(iκd).
pub(crate) fn propagation_from_exponent<A>(exponent: A) -> Scatter2Entries<A>
where
    A: ScalarAlgebra + Clone,
    A::Scalar: ComplexScalar + Zero,
    A::Dimension: Dimension,
{
    let phase = exponent.exp();

    let zero = A::filled_constant_like(phase.value(), <A::Scalar as Zero>::zero());

    Scatter2Entries {
        s11: zero.clone(),
        s12: phase.clone(),
        s21: phase,
        s22: zero,
    }
}

#[cfg(test)]
mod interface_tests {
    use super::interface;

    use crate::{
        algebra::{ArrayJet0, ArrayJet1, ArrayJet2},
        backend::scatter2::Scatter2Entries,
        test_support::{
            TOLERANCE,
            assertions::{assert_array_close, assert_complex_close},
            c,
            finite_difference::{
                FIRST_DERIVATIVE_TOLERANCE, FIRST_DIFFERENCE_STEP, SECOND_DERIVATIVE_TOLERANCE,
                SECOND_DIFFERENCE_STEP, central_first_difference, central_second_difference,
            },
            jet::{
                J0, P, constant_first, constant_second, independent_first, independent_second,
                zero_jet_from_real_value,
            },
        },
    };

    use ndarray::{Ix0, Ix1, arr0, array};
    use num_complex::Complex64;

    fn assert_entries_close(actual: &Scatter2Entries<J0>, expected: &Scatter2Entries<J0>) {
        assert_array_close(actual.s11.value(), expected.s11.value(), TOLERANCE);
        assert_array_close(actual.s12.value(), expected.s12.value(), TOLERANCE);
        assert_array_close(actual.s21.value(), expected.s21.value(), TOLERANCE);
        assert_array_close(actual.s22.value(), expected.s22.value(), TOLERANCE);
    }

    #[test]
    fn identical_media_are_transparent() {
        let interface = interface(
            &zero_jet_from_real_value(2.5),
            &zero_jet_from_real_value(2.5),
        );

        assert_complex_close(interface.s11.value()[()], c(0.0), TOLERANCE);
        assert_complex_close(interface.s12.value()[()], c(1.0), TOLERANCE);
        assert_complex_close(interface.s21.value()[()], c(1.0), TOLERANCE);
        assert_complex_close(interface.s22.value()[()], c(0.0), TOLERANCE);
    }

    #[test]
    fn matches_fresnel_formula() {
        let yl = 2.0;
        let yr = 5.0;

        let result = interface(&zero_jet_from_real_value(yl), &zero_jet_from_real_value(yr));

        let denominator = yl + yr;

        assert_complex_close(
            result.s11.value()[()],
            c((yl - yr) / denominator),
            TOLERANCE,
        );

        assert_complex_close(
            result.s12.value()[()],
            c((2.0 * yr) / denominator),
            TOLERANCE,
        );

        assert_complex_close(
            result.s21.value()[()],
            c((2.0 * yl) / denominator),
            TOLERANCE,
        );

        assert_complex_close(
            result.s22.value()[()],
            c((yr - yl) / denominator),
            TOLERANCE,
        );
    }

    #[test]
    fn reflection_changes_sign_when_media_are_swapped() {
        let left = interface(
            &zero_jet_from_real_value(2.0),
            &zero_jet_from_real_value(5.0),
        );

        let right = interface(
            &zero_jet_from_real_value(5.0),
            &zero_jet_from_real_value(2.0),
        );

        assert_complex_close(right.s11.value()[()], -left.s11.value()[()], TOLERANCE);

        assert_complex_close(right.s22.value()[()], -left.s22.value()[()], TOLERANCE);
    }

    #[test]
    fn reciprocal_interface_satisfies_flux_relation() {
        let yl = 2.0;
        let yr = 5.0;

        let result = interface(&zero_jet_from_real_value(yl), &zero_jet_from_real_value(yr));

        let lhs = result.s12.value()[()] * c(yl);

        let rhs = result.s21.value()[()] * c(yr);

        assert_complex_close(lhs, rhs, TOLERANCE);
    }

    #[test]
    fn operates_pointwise_on_arrays() {
        let left: ArrayJet0<_, _, P> = ArrayJet0::constant(array![c(1.0), c(2.0), c(3.0),]);

        let right = ArrayJet0::constant(array![c(4.0), c(5.0), c(6.0),]);

        let result = interface(&left, &right);

        for i in 0..3 {
            let yl = left.value()[i].re;
            let yr = right.value()[i].re;

            let denominator = yl + yr;

            assert_complex_close(result.s11.value()[i], c((yl - yr) / denominator), TOLERANCE);

            assert_complex_close(result.s12.value()[i], c(2.0 * yr / denominator), TOLERANCE);

            assert_complex_close(result.s21.value()[i], c(2.0 * yl / denominator), TOLERANCE);

            assert_complex_close(result.s22.value()[i], c((yr - yl) / denominator), TOLERANCE);
        }
    }

    #[test]
    fn first_derivative_matches_finite_difference() {
        let left_value = 2.3;
        let right_value = 4.7;

        let left = independent_first(arr0(c(left_value)));
        let right = constant_first(arr0(c(right_value)));

        let actual = interface(&left, &right);

        let expected_s11 = central_first_difference(
            |value| {
                let denominator = value + right_value;

                c((value - right_value) / denominator)
            },
            left_value,
            FIRST_DIFFERENCE_STEP,
        );

        let expected_s12 = central_first_difference(
            |value| {
                let denominator = value + right_value;

                c(2.0 * right_value / denominator)
            },
            left_value,
            FIRST_DIFFERENCE_STEP,
        );

        let expected_s21 = central_first_difference(
            |value| {
                let denominator = value + right_value;

                c(2.0 * value / denominator)
            },
            left_value,
            FIRST_DIFFERENCE_STEP,
        );

        let expected_s22 = central_first_difference(
            |value| {
                let denominator = value + right_value;

                c((right_value - value) / denominator)
            },
            left_value,
            FIRST_DIFFERENCE_STEP,
        );

        assert_complex_close(
            actual.s11.first()[()],
            expected_s11,
            FIRST_DERIVATIVE_TOLERANCE,
        );

        assert_complex_close(
            actual.s12.first()[()],
            expected_s12,
            FIRST_DERIVATIVE_TOLERANCE,
        );

        assert_complex_close(
            actual.s21.first()[()],
            expected_s21,
            FIRST_DERIVATIVE_TOLERANCE,
        );

        assert_complex_close(
            actual.s22.first()[()],
            expected_s22,
            FIRST_DERIVATIVE_TOLERANCE,
        );
    }

    #[test]
    fn first_derivative_with_both_variable_admittances_matches_finite_difference() {
        let parameter = 0.4;

        let left_value = |x: f64| 2.0 + 0.7 * x;
        let right_value = |x: f64| 4.0 - 0.3 * x;

        let left: ArrayJet1<_, _, P> =
            ArrayJet1::from_parts(arr0(c(left_value(parameter))), arr0(c(0.7)));

        let right = ArrayJet1::from_parts(arr0(c(right_value(parameter))), arr0(c(-0.3)));

        let actual = interface(&left, &right);

        let evaluate = |x: f64| {
            let left = left_value(x);
            let right = right_value(x);
            let denominator = left + right;

            [
                c((left - right) / denominator),
                c(2.0 * right / denominator),
                c(2.0 * left / denominator),
                c((right - left) / denominator),
            ]
        };

        let expected = [
            central_first_difference(|x| evaluate(x)[0], parameter, FIRST_DIFFERENCE_STEP),
            central_first_difference(|x| evaluate(x)[1], parameter, FIRST_DIFFERENCE_STEP),
            central_first_difference(|x| evaluate(x)[2], parameter, FIRST_DIFFERENCE_STEP),
            central_first_difference(|x| evaluate(x)[3], parameter, FIRST_DIFFERENCE_STEP),
        ];

        assert_complex_close(
            actual.s11.first()[()],
            expected[0],
            FIRST_DERIVATIVE_TOLERANCE,
        );

        assert_complex_close(
            actual.s12.first()[()],
            expected[1],
            FIRST_DERIVATIVE_TOLERANCE,
        );

        assert_complex_close(
            actual.s21.first()[()],
            expected[2],
            FIRST_DERIVATIVE_TOLERANCE,
        );

        assert_complex_close(
            actual.s22.first()[()],
            expected[3],
            FIRST_DERIVATIVE_TOLERANCE,
        );
    }

    #[test]
    fn second_derivative_matches_finite_difference() {
        let left_value = 2.3;
        let right_value = 4.7;

        let left = independent_second(arr0(c(left_value)));
        let right = constant_second(arr0(c(right_value)));

        let actual = interface(&left, &right);

        let expected_s11 = central_second_difference(
            |value| {
                let denominator = value + right_value;

                c((value - right_value) / denominator)
            },
            left_value,
            SECOND_DIFFERENCE_STEP,
        );

        let expected_s12 = central_second_difference(
            |value| {
                let denominator = value + right_value;

                c(2.0 * right_value / denominator)
            },
            left_value,
            SECOND_DIFFERENCE_STEP,
        );

        let expected_s21 = central_second_difference(
            |value| {
                let denominator = value + right_value;

                c(2.0 * value / denominator)
            },
            left_value,
            SECOND_DIFFERENCE_STEP,
        );

        let expected_s22 = central_second_difference(
            |value| {
                let denominator = value + right_value;

                c((right_value - value) / denominator)
            },
            left_value,
            SECOND_DIFFERENCE_STEP,
        );

        assert_complex_close(
            actual.s11.second()[()],
            expected_s11,
            SECOND_DERIVATIVE_TOLERANCE,
        );

        assert_complex_close(
            actual.s12.second()[()],
            expected_s12,
            SECOND_DERIVATIVE_TOLERANCE,
        );

        assert_complex_close(
            actual.s21.second()[()],
            expected_s21,
            SECOND_DERIVATIVE_TOLERANCE,
        );

        assert_complex_close(
            actual.s22.second()[()],
            expected_s22,
            SECOND_DERIVATIVE_TOLERANCE,
        );
    }

    #[test]
    fn second_derivative_with_nonlinear_admittances_matches_finite_difference() {
        let parameter = 0.4;

        let left_value = |x: f64| 2.0 + 0.7 * x + 0.2 * x.powi(2);
        let right_value = |x: f64| 4.0 - 0.3 * x + 0.1 * x.powi(2);

        let left: ArrayJet2<_, _, P> = ArrayJet2::from_parts(
            arr0(c(left_value(parameter))),
            arr0(c(0.7 + 0.4 * parameter)),
            arr0(c(0.4)),
        );

        let right = ArrayJet2::from_parts(
            arr0(c(right_value(parameter))),
            arr0(c(-0.3 + 0.2 * parameter)),
            arr0(c(0.2)),
        );

        let actual = interface(&left, &right);

        let evaluate = |x: f64| {
            let left = left_value(x);
            let right = right_value(x);
            let denominator = left + right;

            [
                c((left - right) / denominator),
                c(2.0 * right / denominator),
                c(2.0 * left / denominator),
                c((right - left) / denominator),
            ]
        };

        let expected_first = [
            central_first_difference(|x| evaluate(x)[0], parameter, FIRST_DIFFERENCE_STEP),
            central_first_difference(|x| evaluate(x)[1], parameter, FIRST_DIFFERENCE_STEP),
            central_first_difference(|x| evaluate(x)[2], parameter, FIRST_DIFFERENCE_STEP),
            central_first_difference(|x| evaluate(x)[3], parameter, FIRST_DIFFERENCE_STEP),
        ];

        let expected_second = [
            central_second_difference(|x| evaluate(x)[0], parameter, SECOND_DIFFERENCE_STEP),
            central_second_difference(|x| evaluate(x)[1], parameter, SECOND_DIFFERENCE_STEP),
            central_second_difference(|x| evaluate(x)[2], parameter, SECOND_DIFFERENCE_STEP),
            central_second_difference(|x| evaluate(x)[3], parameter, SECOND_DIFFERENCE_STEP),
        ];

        assert_complex_close(
            actual.s11.first()[()],
            expected_first[0],
            FIRST_DERIVATIVE_TOLERANCE,
        );
        assert_complex_close(
            actual.s12.first()[()],
            expected_first[1],
            FIRST_DERIVATIVE_TOLERANCE,
        );
        assert_complex_close(
            actual.s21.first()[()],
            expected_first[2],
            FIRST_DERIVATIVE_TOLERANCE,
        );
        assert_complex_close(
            actual.s22.first()[()],
            expected_first[3],
            FIRST_DERIVATIVE_TOLERANCE,
        );

        assert_complex_close(
            actual.s11.second()[()],
            expected_second[0],
            SECOND_DERIVATIVE_TOLERANCE,
        );
        assert_complex_close(
            actual.s12.second()[()],
            expected_second[1],
            SECOND_DERIVATIVE_TOLERANCE,
        );
        assert_complex_close(
            actual.s21.second()[()],
            expected_second[2],
            SECOND_DERIVATIVE_TOLERANCE,
        );
        assert_complex_close(
            actual.s22.second()[()],
            expected_second[3],
            SECOND_DERIVATIVE_TOLERANCE,
        );
    }

    #[test]
    fn first_derivative_operates_pointwise() {
        let parameter = 0.3;

        let left_values = array![
            1.5 + 0.2 * parameter,
            2.0 - 0.4 * parameter,
            3.0 + 0.7 * parameter,
        ]
        .mapv(c);

        let left_first = array![0.2, -0.4, 0.7].mapv(c);

        let right_values = array![4.0, 5.0, 6.0].mapv(c);

        let left = ArrayJet1::from_parts(left_values, left_first);
        let right = constant_first(right_values);

        let actual = interface(&left, &right);

        for index in 0..3 {
            let base_left = [1.5, 2.0, 3.0][index];
            let slope = [0.2, -0.4, 0.7][index];
            let fixed_right = [4.0, 5.0, 6.0][index];

            let expected = central_first_difference(
                |x| {
                    let left = base_left + slope * x;
                    let denominator = left + fixed_right;

                    c((left - fixed_right) / denominator)
                },
                parameter,
                FIRST_DIFFERENCE_STEP,
            );

            assert_complex_close(
                actual.s11.first()[index],
                expected,
                FIRST_DERIVATIVE_TOLERANCE,
            );
        }
    }
}

#[cfg(test)]
mod propagation_tests {
    use super::propagation_from_exponent;

    use crate::test_support::{
        C, TOLERANCE as VALUE_TOLERANCE,
        assertions::assert_complex_close,
        c,
        finite_difference::{
            FIRST_DERIVATIVE_TOLERANCE, FIRST_DIFFERENCE_STEP, SECOND_DERIVATIVE_TOLERANCE,
            SECOND_DIFFERENCE_STEP, central_first_difference, central_second_difference,
        },
        jet::{
            constant_first, constant_second, independent_first, independent_second,
            quadratic_second, zero_jet_from_array, zero_jet_from_real_value, zero_jet_from_value,
        },
    };

    use ndarray::{arr0, array};
    use num_complex::Complex64;

    #[test]
    fn zero_exponent_is_transparent() {
        let exponent = zero_jet_from_value(c(0.0));

        let result = propagation_from_exponent(exponent);

        assert_complex_close(result.s11[()], C::new(0.0, 0.0), VALUE_TOLERANCE);
        assert_complex_close(result.s12[()], C::new(1.0, 0.0), VALUE_TOLERANCE);
        assert_complex_close(result.s21[()], C::new(1.0, 0.0), VALUE_TOLERANCE);
        assert_complex_close(result.s22[()], C::new(0.0, 0.0), VALUE_TOLERANCE);
    }

    #[test]
    fn transmission_is_exponential_of_exponent() {
        let value = C::new(0.3, -0.2);
        let exponent = zero_jet_from_value(value);

        let result = propagation_from_exponent(exponent.clone());
        let expected = value.exp();

        assert_complex_close(result.s11[()], C::new(0.0, 0.0), VALUE_TOLERANCE);
        assert_complex_close(result.s12[()], expected.clone(), VALUE_TOLERANCE);
        assert_complex_close(result.s21[()], expected, VALUE_TOLERANCE);
        assert_complex_close(result.s22[()], C::new(0.0, 0.0), VALUE_TOLERANCE);
    }

    #[test]
    fn real_negative_exponent_produces_attenuation() {
        let value = C::new(-0.7, 0.0);
        let exponent = zero_jet_from_value(value);

        let result = propagation_from_exponent(exponent);
        let expected = value.exp();

        assert_complex_close(result.s12[()], expected, VALUE_TOLERANCE);
        assert_complex_close(result.s21[()], expected, VALUE_TOLERANCE);

        assert!(result.s12[()].norm() < 1.0);
    }

    #[test]
    fn imaginary_exponent_produces_phase_only_propagation() {
        let value = C::new(0.0, 0.8);
        let exponent = zero_jet_from_value(value);

        let result = propagation_from_exponent(exponent);
        let expected = value.exp();

        assert_complex_close(result.s12[()], expected, VALUE_TOLERANCE);
        assert_complex_close(result.s21[()], expected, VALUE_TOLERANCE);
        assert_complex_close(
            C::new(result.s12[()].norm(), 0.0),
            C::new(1.0, 0.0),
            VALUE_TOLERANCE,
        );
    }

    #[test]
    fn operates_pointwise_on_arrays() {
        let values = array![C::new(0.0, 0.0), C::new(-0.2, 0.4), C::new(0.1, -0.7),];
        let exponents = zero_jet_from_array(values.clone());

        let result = propagation_from_exponent(exponents);

        for index in 0..values.len() {
            let expected = values[index].exp();

            assert_complex_close(result.s11[index], C::new(0.0, 0.0), VALUE_TOLERANCE);
            assert_complex_close(result.s12[index], expected, VALUE_TOLERANCE);
            assert_complex_close(result.s21[index], expected, VALUE_TOLERANCE);
            assert_complex_close(result.s22[index], C::new(0.0, 0.0), VALUE_TOLERANCE);
        }
    }

    #[test]
    fn first_derivative_matches_finite_difference() {
        let parameter = 0.4;

        let exponent = independent_first(arr0(C::new(parameter, 0.0)));
        let actual = propagation_from_exponent(exponent);

        let expected =
            central_first_difference(|x| C::new(x, 0.0).exp(), parameter, FIRST_DIFFERENCE_STEP);

        assert_complex_close(
            actual.s11.first()[()],
            C::new(0.0, 0.0),
            FIRST_DERIVATIVE_TOLERANCE,
        );
        assert_complex_close(actual.s12.first()[()], expected, FIRST_DERIVATIVE_TOLERANCE);
        assert_complex_close(actual.s21.first()[()], expected, FIRST_DERIVATIVE_TOLERANCE);
        assert_complex_close(
            actual.s22.first()[()],
            C::new(0.0, 0.0),
            FIRST_DERIVATIVE_TOLERANCE,
        );
    }

    #[test]
    fn constant_exponent_has_zero_first_derivative() {
        let exponent = constant_first(arr0(C::new(0.3, -0.2)));

        let actual = propagation_from_exponent(exponent);

        assert_complex_close(actual.s11.first()[()], C::new(0.0, 0.0), VALUE_TOLERANCE);
        assert_complex_close(actual.s12.first()[()], C::new(0.0, 0.0), VALUE_TOLERANCE);
        assert_complex_close(actual.s21.first()[()], C::new(0.0, 0.0), VALUE_TOLERANCE);
        assert_complex_close(actual.s22.first()[()], C::new(0.0, 0.0), VALUE_TOLERANCE);
    }

    #[test]
    fn first_derivative_with_affine_complex_exponent_matches_finite_difference() {
        let parameter = 0.4;
        let slope = C::new(0.7, -0.3);

        let exponent_value = |x: f64| C::new(0.2, 0.5) + slope * x;

        let exponent =
            crate::test_support::jet::affine_first(arr0(exponent_value(parameter)), arr0(slope));

        let actual = propagation_from_exponent(exponent);

        let expected = central_first_difference(
            |x| exponent_value(x).exp(),
            parameter,
            FIRST_DIFFERENCE_STEP,
        );

        assert_complex_close(actual.s12.first()[()], expected, FIRST_DERIVATIVE_TOLERANCE);
        assert_complex_close(actual.s21.first()[()], expected, FIRST_DERIVATIVE_TOLERANCE);
    }

    #[test]
    fn second_derivative_matches_finite_difference() {
        let parameter = 0.4;

        let exponent = independent_second(arr0(C::new(parameter, 0.0)));
        let actual = propagation_from_exponent(exponent);

        let expected_first =
            central_first_difference(|x| C::new(x, 0.0).exp(), parameter, FIRST_DIFFERENCE_STEP);

        let expected_second =
            central_second_difference(|x| C::new(x, 0.0).exp(), parameter, SECOND_DIFFERENCE_STEP);

        assert_complex_close(
            actual.s12.first()[()],
            expected_first,
            FIRST_DERIVATIVE_TOLERANCE,
        );
        assert_complex_close(
            actual.s21.first()[()],
            expected_first,
            FIRST_DERIVATIVE_TOLERANCE,
        );

        assert_complex_close(
            actual.s12.second()[()],
            expected_second,
            SECOND_DERIVATIVE_TOLERANCE,
        );
        assert_complex_close(
            actual.s21.second()[()],
            expected_second,
            SECOND_DERIVATIVE_TOLERANCE,
        );

        assert_complex_close(
            actual.s11.second()[()],
            C::new(0.0, 0.0),
            SECOND_DERIVATIVE_TOLERANCE,
        );
        assert_complex_close(
            actual.s22.second()[()],
            C::new(0.0, 0.0),
            SECOND_DERIVATIVE_TOLERANCE,
        );
    }

    #[test]
    fn constant_exponent_has_zero_second_derivative() {
        let exponent = constant_second(arr0(C::new(0.3, -0.2)));

        let actual = propagation_from_exponent(exponent);

        assert_complex_close(actual.s11.second()[()], C::new(0.0, 0.0), VALUE_TOLERANCE);
        assert_complex_close(actual.s12.second()[()], C::new(0.0, 0.0), VALUE_TOLERANCE);
        assert_complex_close(actual.s21.second()[()], C::new(0.0, 0.0), VALUE_TOLERANCE);
        assert_complex_close(actual.s22.second()[()], C::new(0.0, 0.0), VALUE_TOLERANCE);
    }

    #[test]
    fn nonlinear_exponent_derivatives_match_finite_difference() {
        let parameter = 0.4;

        let exponent_value =
            |x: f64| C::new(0.2, 0.5) + C::new(0.7, -0.3) * x + C::new(0.15, 0.1) * x.powi(2);

        let exponent_first = |x: f64| C::new(0.7, -0.3) + C::new(0.3, 0.2) * x;

        let exponent_second = C::new(0.3, 0.2);

        let exponent = quadratic_second(
            arr0(exponent_value(parameter)),
            arr0(exponent_first(parameter)),
            arr0(exponent_second),
        );

        let actual = propagation_from_exponent(exponent);

        let expected_first = central_first_difference(
            |x| exponent_value(x).exp(),
            parameter,
            FIRST_DIFFERENCE_STEP,
        );

        let expected_second = central_second_difference(
            |x| exponent_value(x).exp(),
            parameter,
            SECOND_DIFFERENCE_STEP,
        );

        assert_complex_close(
            actual.s12.first()[()],
            expected_first,
            FIRST_DERIVATIVE_TOLERANCE,
        );
        assert_complex_close(
            actual.s21.first()[()],
            expected_first,
            FIRST_DERIVATIVE_TOLERANCE,
        );

        assert_complex_close(
            actual.s12.second()[()],
            expected_second,
            SECOND_DERIVATIVE_TOLERANCE,
        );
        assert_complex_close(
            actual.s21.second()[()],
            expected_second,
            SECOND_DERIVATIVE_TOLERANCE,
        );
    }

    #[test]
    fn first_derivative_operates_pointwise() {
        let parameter = 0.3;

        let offsets = [C::new(0.1, 0.2), C::new(-0.3, 0.5), C::new(0.4, -0.2)];

        let slopes = [C::new(0.2, -0.1), C::new(-0.4, 0.3), C::new(0.7, 0.2)];

        let values = array![
            offsets[0] + slopes[0] * parameter,
            offsets[1] + slopes[1] * parameter,
            offsets[2] + slopes[2] * parameter,
        ];

        let first = array![slopes[0], slopes[1], slopes[2]];

        let exponent = crate::test_support::jet::affine_first(values, first);

        let actual = propagation_from_exponent(exponent);

        for index in 0..3 {
            let expected = central_first_difference(
                |x| (offsets[index] + slopes[index] * x).exp(),
                parameter,
                FIRST_DIFFERENCE_STEP,
            );

            assert_complex_close(
                actual.s12.first()[index],
                expected,
                FIRST_DERIVATIVE_TOLERANCE,
            );
            assert_complex_close(
                actual.s21.first()[index],
                expected,
                FIRST_DERIVATIVE_TOLERANCE,
            );
        }
    }
}

#[cfg(test)]
mod accumulate_tests {
    use super::*;

    use crate::{
        Polarisation, RealAxis,
        backend::{RunMode, scatter2::Scatter2},
        test_support::{
            coordinates::test_coordinates,
            stack::{empty_stack, single_layer_stack, stack_with_layers, two_layer_stack},
        },
    };

    #[test]
    fn response_only_does_not_retain_components() {
        let backend = Scatter2::new();

        let coordinates = test_coordinates();
        let stack = empty_stack();

        let workspace = backend
            .accumulate::<_, RealAxis, _>(
                &coordinates,
                &stack,
                Polarisation::TransverseElectric,
                RunMode::ResponseOnly,
            )
            .unwrap();

        let (_, retained) = workspace.into_parts();

        assert!(retained.is_none());
    }

    #[test]
    fn internal_fields_retains_empty_stack_interface() {
        let backend = Scatter2::new();

        let coordinates = test_coordinates();
        let stack = empty_stack();

        let workspace = backend
            .accumulate::<_, RealAxis, _>(
                &coordinates,
                &stack,
                Polarisation::TransverseElectric,
                RunMode::InternalFields,
            )
            .unwrap();

        let (_, retained) = workspace.into_parts();

        let retained = retained.expect("retained components requested");

        assert_eq!(retained.components.len(), 1);
        assert!(retained.layer_cuts.is_empty());
    }

    #[test]
    fn single_layer_records_expected_component_sequence() {
        let backend = Scatter2::new();

        let coordinates = test_coordinates();
        let stack = single_layer_stack();

        let workspace = backend
            .accumulate::<_, RealAxis, _>(
                &coordinates,
                &stack,
                Polarisation::TransverseElectric,
                RunMode::InternalFields,
            )
            .unwrap();

        let (_, retained) = workspace.into_parts();

        let retained = retained.unwrap();

        // interface
        // propagation
        // interface
        assert_eq!(retained.components.len(), 3);

        assert_eq!(retained.layer_cuts.len(), 1);

        let cut = &retained.layer_cuts[0];

        assert_eq!(cut.left(), 1);
        assert_eq!(cut.right(), 2);
    }

    #[test]
    fn two_layers_record_expected_component_sequence() {
        let backend = Scatter2::new();

        let coordinates = test_coordinates();
        let stack = two_layer_stack();

        let workspace = backend
            .accumulate::<_, RealAxis, _>(
                &coordinates,
                &stack,
                Polarisation::TransverseElectric,
                RunMode::InternalFields,
            )
            .unwrap();

        let (_, retained) = workspace.into_parts();

        let retained = retained.unwrap();

        // I P I P I
        assert_eq!(retained.components.len(), 5);

        assert_eq!(retained.layer_cuts.len(), 2);

        assert_eq!(retained.layer_cuts[0].left(), 1);
        assert_eq!(retained.layer_cuts[0].right(), 2);

        assert_eq!(retained.layer_cuts[1].left(), 3);
        assert_eq!(retained.layer_cuts[1].right(), 4);
    }

    #[test]
    fn component_count_matches_number_of_layers() {
        let backend = Scatter2::new();

        for layer_count in 0..5 {
            let coordinates = test_coordinates();
            let stack = stack_with_layers(layer_count);

            let workspace = backend
                .accumulate::<_, RealAxis, _>(
                    &coordinates,
                    &stack,
                    Polarisation::TransverseElectric,
                    RunMode::InternalFields,
                )
                .unwrap();

            let (_, retained) = workspace.into_parts();

            let retained = retained.unwrap();

            assert_eq!(retained.components.len(), 2 * layer_count + 1,);

            assert_eq!(retained.layer_cuts.len(), layer_count,);
        }
    }
}
