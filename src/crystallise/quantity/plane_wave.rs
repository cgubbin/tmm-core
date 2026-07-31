//! Recursive crystallisation support for plane-wave observables.
//!
//! Plane-wave results are composite structures:
//!
//! - [`PlaneWaveAmplitudes`] stores complex reflection and transmission
//!   amplitudes;
//! - [`PlaneWavePower`] stores real reflectance, transmittance, and
//!   absorptance;
//! - [`PlaneWaveObservables`] combines both structures.
//!
//! During backend evaluation, each leaf is an internal jet carrying its value
//! and derivative components. The implementations in this module recursively
//! transpose those structures so that values and derivatives are grouped into
//! complete observable sets.
//!
//! For example, a first directional result is transformed conceptually from
//!
//! ```text
//! PlaneWaveObservables<Jet1>
//! ```
//!
//! into
//!
//! ```text
//! DirectionalFirstParts<PlaneWaveObservables<Value>>
//! ```
//!
//! where the `value` field contains all physical values and the `first` field
//! contains the corresponding first derivatives.
//!
//! These implementations contain no plane-wave differentiation logic. They
//! only reorganise derivative components already present in the observable
//! leaves.

use crate::{
    PlaneWaveAmplitudes, PlaneWavePower,
    algebra::ComplexJet,
    crystallise::{
        BivariateFirstParts, BivariateSecondParts, DirectionalFirstParts, DirectionalSecondParts,
        IntoFirst, IntoGradient, IntoHessian, IntoSecond, IntoValue,
    },
    observable::PlaneWaveObservables,
};

/// Extract the value components of a complete plane-wave observable set.
impl<J> IntoValue for PlaneWaveObservables<J, J::RealJet>
where
    J: ComplexJet + IntoValue,
    J::RealJet: IntoValue,
{
    type Value = PlaneWaveObservables<J::Value, <J::RealJet as IntoValue>::Value>;

    fn into_value(self) -> Self::Value {
        let (amplitudes, power) = self.into_parts();

        PlaneWaveObservables::new(amplitudes.into_value(), power.into_value())
    }
}

/// Extract the value components of reflection and transmission amplitudes.
impl<J> IntoValue for PlaneWaveAmplitudes<J>
where
    J: IntoValue,
{
    type Value = PlaneWaveAmplitudes<J::Value>;

    fn into_value(self) -> Self::Value {
        let (reflection, transmission) = self.into_parts();

        PlaneWaveAmplitudes::new(reflection.into_value(), transmission.into_value())
    }
}

/// Extract the value components of the plane-wave power observables.
impl<J> IntoValue for PlaneWavePower<J>
where
    J: IntoValue,
{
    type Value = PlaneWavePower<J::Value>;

    fn into_value(self) -> Self::Value {
        let (reflectance, transmittance, absorptance) = self.into_parts();

        PlaneWavePower::new(
            reflectance.into_value(),
            transmittance.into_value(),
            absorptance.into_value(),
        )
    }
}

/// Separate complete plane-wave observables into values and first directional
/// derivatives.
impl<J> IntoFirst for PlaneWaveObservables<J, J::RealJet>
where
    J: ComplexJet + IntoFirst,
    J::RealJet: IntoFirst,
{
    fn into_first(self) -> DirectionalFirstParts<Self::Value> {
        let (amplitudes, power) = self.into_parts();

        let (amplitudes, amplitudes_first) = amplitudes.into_first().into_parts();

        let (power, power_first) = power.into_first().into_parts();

        DirectionalFirstParts::new(
            PlaneWaveObservables::new(amplitudes, power),
            PlaneWaveObservables::new(amplitudes_first, power_first),
        )
    }
}

/// Separate amplitudes into values and first directional derivatives.
impl<J> IntoFirst for PlaneWaveAmplitudes<J>
where
    J: IntoFirst,
{
    fn into_first(self) -> DirectionalFirstParts<Self::Value> {
        let (reflection, transmission) = self.into_parts();

        let (reflection, reflection_first) = reflection.into_first().into_parts();

        let (transmission, transmission_first) = transmission.into_first().into_parts();

        DirectionalFirstParts::new(
            PlaneWaveAmplitudes::new(reflection, transmission),
            PlaneWaveAmplitudes::new(reflection_first, transmission_first),
        )
    }
}

/// Separate power observables into values and first directional derivatives.
impl<J> IntoFirst for PlaneWavePower<J>
where
    J: IntoFirst,
{
    fn into_first(self) -> DirectionalFirstParts<Self::Value> {
        let (reflectance, transmittance, absorptance) = self.into_parts();

        let (reflectance, reflectance_first) = reflectance.into_first().into_parts();

        let (transmittance, transmittance_first) = transmittance.into_first().into_parts();

        let (absorptance, absorptance_first) = absorptance.into_first().into_parts();

        DirectionalFirstParts::new(
            PlaneWavePower::new(reflectance, transmittance, absorptance),
            PlaneWavePower::new(reflectance_first, transmittance_first, absorptance_first),
        )
    }
}

/// Separate complete plane-wave observables into values and directional
/// derivatives through second order.
impl<J> IntoSecond for PlaneWaveObservables<J, J::RealJet>
where
    J: ComplexJet + IntoSecond,
    J::RealJet: IntoSecond,
{
    fn into_second(self) -> DirectionalSecondParts<Self::Value> {
        let (amplitudes, power) = self.into_parts();

        let (amplitudes, amplitudes_first, amplitudes_second) =
            amplitudes.into_second().into_parts();

        let (power, power_first, power_second) = power.into_second().into_parts();

        DirectionalSecondParts::new(
            PlaneWaveObservables::new(amplitudes, power),
            PlaneWaveObservables::new(amplitudes_first, power_first),
            PlaneWaveObservables::new(amplitudes_second, power_second),
        )
    }
}

/// Separate amplitudes into values and directional derivatives through second
/// order.
impl<J> IntoSecond for PlaneWaveAmplitudes<J>
where
    J: IntoSecond,
{
    fn into_second(self) -> DirectionalSecondParts<Self::Value> {
        let (reflection, transmission) = self.into_parts();

        let (reflection, reflection_first, reflection_second) =
            reflection.into_second().into_parts();

        let (transmission, transmission_first, transmission_second) =
            transmission.into_second().into_parts();

        DirectionalSecondParts::new(
            PlaneWaveAmplitudes::new(reflection, transmission),
            PlaneWaveAmplitudes::new(reflection_first, transmission_first),
            PlaneWaveAmplitudes::new(reflection_second, transmission_second),
        )
    }
}

/// Separate power observables into values and directional derivatives through
/// second order.
impl<J> IntoSecond for PlaneWavePower<J>
where
    J: IntoSecond,
{
    fn into_second(self) -> DirectionalSecondParts<Self::Value> {
        let (reflectance, transmittance, absorptance) = self.into_parts();

        let (reflectance, reflectance_first, reflectance_second) =
            reflectance.into_second().into_parts();

        let (transmittance, transmittance_first, transmittance_second) =
            transmittance.into_second().into_parts();

        let (absorptance, absorptance_first, absorptance_second) =
            absorptance.into_second().into_parts();

        DirectionalSecondParts::new(
            PlaneWavePower::new(reflectance, transmittance, absorptance),
            PlaneWavePower::new(reflectance_first, transmittance_first, absorptance_first),
            PlaneWavePower::new(reflectance_second, transmittance_second, absorptance_second),
        )
    }
}

/// Separate complete plane-wave observables into values and first derivatives
/// with respect to two coordinates.
impl<J> IntoGradient for PlaneWaveObservables<J, J::RealJet>
where
    J: ComplexJet + IntoGradient,
    J::RealJet: IntoGradient,
{
    fn into_gradient(self) -> BivariateFirstParts<Self::Value> {
        let (amplitudes, power) = self.into_parts();

        let (amplitudes, amplitudes_x, amplitudes_y) = amplitudes.into_gradient().into_parts();

        let (power, power_x, power_y) = power.into_gradient().into_parts();

        BivariateFirstParts::new(
            PlaneWaveObservables::new(amplitudes, power),
            PlaneWaveObservables::new(amplitudes_x, power_x),
            PlaneWaveObservables::new(amplitudes_y, power_y),
        )
    }
}

/// Separate amplitudes into values and first derivatives with respect to two
/// coordinates.
impl<J> IntoGradient for PlaneWaveAmplitudes<J>
where
    J: IntoGradient,
{
    fn into_gradient(self) -> BivariateFirstParts<Self::Value> {
        let (reflection, transmission) = self.into_parts();

        let (reflection, reflection_x, reflection_y) = reflection.into_gradient().into_parts();

        let (transmission, transmission_x, transmission_y) =
            transmission.into_gradient().into_parts();

        BivariateFirstParts::new(
            PlaneWaveAmplitudes::new(reflection, transmission),
            PlaneWaveAmplitudes::new(reflection_x, transmission_x),
            PlaneWaveAmplitudes::new(reflection_y, transmission_y),
        )
    }
}

/// Separate power observables into values and first derivatives with respect
/// to two coordinates.
impl<J> IntoGradient for PlaneWavePower<J>
where
    J: IntoGradient,
{
    fn into_gradient(self) -> BivariateFirstParts<Self::Value> {
        let (reflectance, transmittance, absorptance) = self.into_parts();

        let (reflectance, reflectance_x, reflectance_y) = reflectance.into_gradient().into_parts();

        let (transmittance, transmittance_x, transmittance_y) =
            transmittance.into_gradient().into_parts();

        let (absorptance, absorptance_x, absorptance_y) = absorptance.into_gradient().into_parts();

        BivariateFirstParts::new(
            PlaneWavePower::new(reflectance, transmittance, absorptance),
            PlaneWavePower::new(reflectance_x, transmittance_x, absorptance_x),
            PlaneWavePower::new(reflectance_y, transmittance_y, absorptance_y),
        )
    }
}

/// Separate complete plane-wave observables into values, a bivariate gradient,
/// and a symmetric bivariate Hessian.
impl<J> IntoHessian for PlaneWaveObservables<J, J::RealJet>
where
    J: ComplexJet + IntoHessian,
    J::RealJet: IntoHessian,
{
    fn into_hessian(self) -> BivariateSecondParts<Self::Value> {
        let (amplitudes, power) = self.into_parts();

        let (
            amplitudes,
            amplitudes_x,
            amplitudes_y,
            amplitudes_x_x,
            amplitudes_x_y,
            amplitudes_y_y,
        ) = amplitudes.into_hessian().into_parts();

        let (power, power_x, power_y, power_x_x, power_x_y, power_y_y) =
            power.into_hessian().into_parts();

        BivariateSecondParts::new(
            PlaneWaveObservables::new(amplitudes, power),
            PlaneWaveObservables::new(amplitudes_x, power_x),
            PlaneWaveObservables::new(amplitudes_y, power_y),
            PlaneWaveObservables::new(amplitudes_x_x, power_x_x),
            PlaneWaveObservables::new(amplitudes_x_y, power_x_y),
            PlaneWaveObservables::new(amplitudes_y_y, power_y_y),
        )
    }
}

/// Separate amplitudes into values, a bivariate gradient, and a symmetric
/// bivariate Hessian.
impl<J> IntoHessian for PlaneWaveAmplitudes<J>
where
    J: IntoHessian,
{
    fn into_hessian(self) -> BivariateSecondParts<Self::Value> {
        let (reflection, transmission) = self.into_parts();

        let (
            reflection,
            reflection_x,
            reflection_y,
            reflection_x_x,
            reflection_x_y,
            reflection_y_y,
        ) = reflection.into_hessian().into_parts();

        let (
            transmission,
            transmission_x,
            transmission_y,
            transmission_x_x,
            transmission_x_y,
            transmission_y_y,
        ) = transmission.into_hessian().into_parts();

        BivariateSecondParts::new(
            PlaneWaveAmplitudes::new(reflection, transmission),
            PlaneWaveAmplitudes::new(reflection_x, transmission_x),
            PlaneWaveAmplitudes::new(reflection_y, transmission_y),
            PlaneWaveAmplitudes::new(reflection_x_x, transmission_x_x),
            PlaneWaveAmplitudes::new(reflection_x_y, transmission_x_y),
            PlaneWaveAmplitudes::new(reflection_y_y, transmission_y_y),
        )
    }
}

/// Separate power observables into values, a bivariate gradient, and a
/// symmetric bivariate Hessian.
impl<J> IntoHessian for PlaneWavePower<J>
where
    J: IntoHessian,
{
    fn into_hessian(self) -> BivariateSecondParts<Self::Value> {
        let (reflectance, transmittance, absorptance) = self.into_parts();

        let (
            reflectance,
            reflectance_x,
            reflectance_y,
            reflectance_x_x,
            reflectance_x_y,
            reflectance_y_y,
        ) = reflectance.into_hessian().into_parts();

        let (
            transmittance,
            transmittance_x,
            transmittance_y,
            transmittance_x_x,
            transmittance_x_y,
            transmittance_y_y,
        ) = transmittance.into_hessian().into_parts();

        let (
            absorptance,
            absorptance_x,
            absorptance_y,
            absorptance_x_x,
            absorptance_x_y,
            absorptance_y_y,
        ) = absorptance.into_hessian().into_parts();

        BivariateSecondParts::new(
            PlaneWavePower::new(reflectance, transmittance, absorptance),
            PlaneWavePower::new(reflectance_x, transmittance_x, absorptance_x),
            PlaneWavePower::new(reflectance_y, transmittance_y, absorptance_y),
            PlaneWavePower::new(reflectance_x_x, transmittance_x_x, absorptance_x_x),
            PlaneWavePower::new(reflectance_x_y, transmittance_x_y, absorptance_x_y),
            PlaneWavePower::new(reflectance_y_y, transmittance_y_y, absorptance_y_y),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::algebra::{Jet0, Jet2, RealParameter};

    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Leaf {
        value: i32,
        first: i32,
        second: i32,
        x: i32,
        y: i32,
        x_x: i32,
        x_y: i32,
        y_y: i32,
    }

    impl Leaf {
        fn new(offset: i32) -> Self {
            Self {
                value: offset + 1,
                first: offset + 2,
                second: offset + 3,
                x: offset + 4,
                y: offset + 5,
                x_x: offset + 6,
                x_y: offset + 7,
                y_y: offset + 8,
            }
        }
    }

    impl IntoValue for Leaf {
        type Value = i32;

        fn into_value(self) -> Self::Value {
            self.value
        }
    }

    impl IntoFirst for Leaf {
        fn into_first(self) -> DirectionalFirstParts<Self::Value> {
            DirectionalFirstParts::new(self.value, self.first)
        }
    }

    impl IntoSecond for Leaf {
        fn into_second(self) -> DirectionalSecondParts<Self::Value> {
            DirectionalSecondParts::new(self.value, self.first, self.second)
        }
    }

    impl IntoGradient for Leaf {
        fn into_gradient(self) -> BivariateFirstParts<Self::Value> {
            BivariateFirstParts::new(self.value, self.x, self.y)
        }
    }

    impl IntoHessian for Leaf {
        fn into_hessian(self) -> BivariateSecondParts<Self::Value> {
            BivariateSecondParts::new(self.value, self.x, self.y, self.x_x, self.x_y, self.y_y)
        }
    }

    #[test]
    fn amplitudes_into_value_preserves_field_order() {
        let amplitudes = PlaneWaveAmplitudes::new(Leaf::new(0), Leaf::new(10));

        let amplitudes = amplitudes.into_value();
        let (reflection, transmission) = amplitudes.into_parts();

        assert_eq!(reflection, 1);
        assert_eq!(transmission, 11);
    }

    #[test]
    fn power_into_value_preserves_field_order() {
        let power = PlaneWavePower::new(Leaf::new(0), Leaf::new(10), Leaf::new(20));

        let power = power.into_value();

        let (reflectance, transmittance, absorptance) = power.into_parts();

        assert_eq!(reflectance, 1);
        assert_eq!(transmittance, 11);
        assert_eq!(absorptance, 21);
    }

    #[test]
    fn amplitudes_into_first_transposes_values_and_derivatives() {
        let amplitudes = PlaneWaveAmplitudes::new(Leaf::new(0), Leaf::new(10));

        let (values, first) = amplitudes.into_first().into_parts();

        let (reflection, transmission) = values.into_parts();

        let (reflection_first, transmission_first) = first.into_parts();

        assert_eq!(reflection, 1);
        assert_eq!(transmission, 11);
        assert_eq!(reflection_first, 2);
        assert_eq!(transmission_first, 12);
    }

    #[test]
    fn power_into_first_transposes_all_observables() {
        let power = PlaneWavePower::new(Leaf::new(0), Leaf::new(10), Leaf::new(20));

        let (values, first) = power.into_first().into_parts();

        assert_eq!(values.into_parts(), (1, 11, 21),);

        assert_eq!(first.into_parts(), (2, 12, 22),);
    }

    #[test]
    fn amplitudes_into_second_transposes_both_orders() {
        let amplitudes = PlaneWaveAmplitudes::new(Leaf::new(0), Leaf::new(10));

        let (values, first, second) = amplitudes.into_second().into_parts();

        assert_eq!(values.into_parts(), (1, 11));
        assert_eq!(first.into_parts(), (2, 12));
        assert_eq!(second.into_parts(), (3, 13));
    }

    #[test]
    fn power_into_second_transposes_both_orders() {
        let power = PlaneWavePower::new(Leaf::new(0), Leaf::new(10), Leaf::new(20));

        let (values, first, second) = power.into_second().into_parts();

        assert_eq!(values.into_parts(), (1, 11, 21));
        assert_eq!(first.into_parts(), (2, 12, 22));
        assert_eq!(second.into_parts(), (3, 13, 23));
    }

    #[test]
    fn amplitudes_into_gradient_transposes_both_coordinates() {
        let amplitudes = PlaneWaveAmplitudes::new(Leaf::new(0), Leaf::new(10));

        let (values, x, y) = amplitudes.into_gradient().into_parts();

        assert_eq!(values.into_parts(), (1, 11));
        assert_eq!(x.into_parts(), (4, 14));
        assert_eq!(y.into_parts(), (5, 15));
    }

    #[test]
    fn power_into_gradient_transposes_both_coordinates() {
        let power = PlaneWavePower::new(Leaf::new(0), Leaf::new(10), Leaf::new(20));

        let (values, x, y) = power.into_gradient().into_parts();

        assert_eq!(values.into_parts(), (1, 11, 21));
        assert_eq!(x.into_parts(), (4, 14, 24));
        assert_eq!(y.into_parts(), (5, 15, 25));
    }

    #[test]
    fn amplitudes_into_hessian_preserves_component_order() {
        let amplitudes = PlaneWaveAmplitudes::new(Leaf::new(0), Leaf::new(10));

        let (values, x, y, x_x, x_y, y_y) = amplitudes.into_hessian().into_parts();

        assert_eq!(values.into_parts(), (1, 11));
        assert_eq!(x.into_parts(), (4, 14));
        assert_eq!(y.into_parts(), (5, 15));
        assert_eq!(x_x.into_parts(), (6, 16));
        assert_eq!(x_y.into_parts(), (7, 17));
        assert_eq!(y_y.into_parts(), (8, 18));
    }

    #[test]
    fn power_into_hessian_preserves_component_order() {
        let power = PlaneWavePower::new(Leaf::new(0), Leaf::new(10), Leaf::new(20));

        let (values, x, y, x_x, x_y, y_y) = power.into_hessian().into_parts();

        assert_eq!(values.into_parts(), (1, 11, 21));
        assert_eq!(x.into_parts(), (4, 14, 24));
        assert_eq!(y.into_parts(), (5, 15, 25));
        assert_eq!(x_x.into_parts(), (6, 16, 26));
        assert_eq!(x_y.into_parts(), (7, 17, 27));
        assert_eq!(y_y.into_parts(), (8, 18, 28));
    }

    use ndarray::arr0;
    use num_complex::Complex64;

    fn c(real: f64, imag: f64) -> Complex64 {
        Complex64::new(real, imag)
    }

    #[test]
    fn complete_observables_transpose_directional_second_jets() {
        let reflection = Jet2::from_parts(arr0(c(2.0, 0.5)), arr0(c(1.0, 0.1)), arr0(c(5.0, 0.25)));
        let transmission =
            Jet2::from_parts(arr0(c(20.0, 5.0)), arr0(c(1.5, 0.2)), arr0(c(-5.0, -0.25)));

        let reflectance: Jet2<_, RealParameter> = Jet2::from_parts(arr0(2.0), arr0(1.0), arr0(5.0));
        let transmittance = Jet2::from_parts(arr0(3.0), arr0(4.0), arr0(6.0));
        let absorptance = Jet2::from_parts(arr0(7.0), arr0(8.0), arr0(9.0));
        // Construct actual complex and real ArrayJet1 leaves here.
        let observables = PlaneWaveObservables::new(
            PlaneWaveAmplitudes::new(reflection.clone(), transmission.clone()),
            PlaneWavePower::new(
                reflectance.clone(),
                transmittance.clone(),
                absorptance.clone(),
            ),
        );

        let (values, first) = observables.clone().into_first().into_parts();

        assert_eq!(values.reflection()[()], reflection.value()[()]);
        assert_eq!(values.transmission()[()], transmission.value()[()]);
        assert_eq!(values.reflectance()[()], reflectance.value()[()]);
        assert_eq!(values.transmittance()[()], transmittance.value()[()]);
        assert_eq!(values.absorptance()[()], absorptance.value()[()]);

        assert_eq!(first.reflection()[()], reflection.first()[()]);
        assert_eq!(first.transmission()[()], transmission.first()[()]);
        assert_eq!(first.reflectance()[()], reflectance.first()[()]);
        assert_eq!(first.transmittance()[()], transmittance.first()[()]);
        assert_eq!(first.absorptance()[()], absorptance.first()[()]);

        let (values, first, second) = observables.into_second().into_parts();

        assert_eq!(values.reflection()[()], reflection.value()[()]);
        assert_eq!(values.transmission()[()], transmission.value()[()]);
        assert_eq!(values.reflectance()[()], reflectance.value()[()]);
        assert_eq!(values.transmittance()[()], transmittance.value()[()]);
        assert_eq!(values.absorptance()[()], absorptance.value()[()]);

        assert_eq!(first.reflection()[()], reflection.first()[()]);
        assert_eq!(first.transmission()[()], transmission.first()[()]);
        assert_eq!(first.reflectance()[()], reflectance.first()[()]);
        assert_eq!(first.transmittance()[()], transmittance.first()[()]);
        assert_eq!(first.absorptance()[()], absorptance.first()[()]);

        assert_eq!(second.reflection()[()], reflection.second()[()]);
        assert_eq!(second.transmission()[()], transmission.second()[()]);
        assert_eq!(second.reflectance()[()], reflectance.second()[()]);
        assert_eq!(second.transmittance()[()], transmittance.second()[()]);
        assert_eq!(second.absorptance()[()], absorptance.second()[()]);

        // Assert at least one amplitude and every power field, ensuring the
        // outer observable structure was also transposed.
    }
}
