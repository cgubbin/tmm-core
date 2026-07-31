use crate::{
    IncidentSide,
    algebra::{ComplexJet, RealScalarAlgebra, ScalarAlgebra},
    backend::PlaneWaveEntries,
};

use num_traits::One;

pub trait ProjectAmplitudes: PlaneWaveEntries {
    type Amplitudes;

    fn project_amplitudes(
        &self,
        exterior: &Self::ExteriorContext,
        incident_side: IncidentSide,
    ) -> Self::Amplitudes;
}

pub trait ProjectPower: PlaneWaveEntries {
    type Power;

    fn project_power(
        &self,
        exterior: &Self::ExteriorContext,
        incident_side: IncidentSide,
    ) -> Self::Power;
}

/// Backend-neutral physical plane-wave observables.
///
/// This type groups the physically observable quantities associated with the
/// scattering of a single incident plane wave.
///
/// It contains both complex field-amplitude coefficients (`r`, `t`) and the
/// corresponding real power coefficients (`R`, `T`, `A`).
///
/// Power coefficients are stored explicitly rather than derived from the
/// amplitudes because they depend on the normalization convention and the
/// ratio of transmitted and incident power flux.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveObservables<C, R> {
    amplitudes: PlaneWaveAmplitudes<C>,
    power: PlaneWavePower<R>,
}

impl<C, R> PlaneWaveObservables<C, R> {
    /// Construct a value-only physical plane-wave response.
    pub fn new(amplitudes: PlaneWaveAmplitudes<C>, power: PlaneWavePower<R>) -> Self {
        Self { amplitudes, power }
    }

    /// Return the complex reflection and transmission amplitudes.
    pub fn amplitudes(&self) -> &PlaneWaveAmplitudes<C> {
        &self.amplitudes
    }

    /// Return the real power coefficients.
    pub fn power(&self) -> &PlaneWavePower<R> {
        &self.power
    }

    /// Return the complex reflection amplitude coefficient.
    pub fn reflection(&self) -> &C {
        self.amplitudes.reflection()
    }

    /// Return the complex transmission amplitude coefficient.
    pub fn transmission(&self) -> &C {
        self.amplitudes.transmission()
    }

    /// Return the power absorptance.
    pub fn absorptance(&self) -> &R {
        self.power.absorptance()
    }

    /// Return the power reflectance.
    pub fn reflectance(&self) -> &R {
        self.power.reflectance()
    }

    /// Return the power transmittance.
    pub fn transmittance(&self) -> &R {
        self.power.transmittance()
    }

    pub fn map<C2, R2>(
        self,
        complex: impl Fn(C) -> C2,
        real: impl Fn(R) -> R2,
    ) -> PlaneWaveObservables<C2, R2> {
        PlaneWaveObservables {
            amplitudes: self.amplitudes.map(complex),
            power: self.power.map(real),
        }
    }

    pub fn map_amplitudes<U>(
        self,
        f: impl FnOnce(PlaneWaveAmplitudes<C>) -> PlaneWaveAmplitudes<U>,
    ) -> PlaneWaveObservables<U, R> {
        PlaneWaveObservables {
            amplitudes: f(self.amplitudes),
            power: self.power,
        }
    }

    pub fn map_power<U>(
        self,
        f: impl FnOnce(PlaneWavePower<R>) -> PlaneWavePower<U>,
    ) -> PlaneWaveObservables<C, U> {
        PlaneWaveObservables {
            amplitudes: self.amplitudes,
            power: f(self.power),
        }
    }

    /// Consume the response and return all components.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(self) -> (PlaneWaveAmplitudes<C>, PlaneWavePower<R>) {
        (self.amplitudes, self.power)
    }
}

/// Complex reflection and transmission amplitude coefficients.
///
/// For a unit-amplitude incident field:
///
/// ```text
/// reflected field   = r
/// transmitted field = t
/// ```
///
/// These are field-amplitude coefficients rather than power coefficients.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveAmplitudes<C> {
    reflection: C,
    transmission: C,
}

impl<C> PlaneWaveAmplitudes<C> {
    /// Construct complex reflection and transmission amplitudes.
    pub fn new(reflection: C, transmission: C) -> Self {
        Self {
            reflection,
            transmission,
        }
    }

    /// Return the complex reflection amplitude coefficient.
    pub fn reflection(&self) -> &C {
        &self.reflection
    }

    /// Return the complex transmission amplitude coefficient.
    pub fn transmission(&self) -> &C {
        &self.transmission
    }

    pub fn map<U>(self, f: impl Fn(C) -> U) -> PlaneWaveAmplitudes<U> {
        PlaneWaveAmplitudes {
            reflection: f(self.reflection),
            transmission: f(self.transmission),
        }
    }

    /// Consume the pair and return its amplitude arrays.
    pub fn into_parts(self) -> (C, C) {
        (self.reflection, self.transmission)
    }
}

/// Real reflected, transmitted and absorbed power fractions.
///
/// Reflectance and transmittance are defined relative to the incident power
/// flux. The backend is responsible for applying the appropriate transmitted-
/// to-incident port flux ratio when constructing `transmittance`.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWavePower<R> {
    reflectance: R,
    transmittance: R,
    absorptance: R,
}

impl<R> PlaneWavePower<R> {
    /// Construct real power reflectance and transmittance arrays.
    pub fn new(reflectance: R, transmittance: R, absorptance: R) -> Self {
        Self {
            reflectance,
            transmittance,
            absorptance,
        }
    }

    /// Return the power reflectance.
    pub fn reflectance(&self) -> &R {
        &self.reflectance
    }

    /// Return the power transmittance.
    pub fn transmittance(&self) -> &R {
        &self.transmittance
    }

    /// Return the total absorbed power fraction.
    pub fn absorptance(&self) -> &R {
        &self.absorptance
    }

    pub fn map<U>(self, f: impl Fn(R) -> U) -> PlaneWavePower<U> {
        PlaneWavePower {
            reflectance: f(self.reflectance),
            transmittance: f(self.transmittance),
            absorptance: f(self.absorptance),
        }
    }

    pub fn from_amplitudes_and_admittance<C>(
        reflection: &C,
        transmission: &C,
        incident_admittance: &C,
        transmitted_admittance: &C,
    ) -> Self
    where
        C: RealScalarAlgebra<RealJet = R>,
        R: ScalarAlgebra,
        R::Scalar: One,
    {
        let flux_ratio = transmitted_admittance
            .real()
            .divide(&incident_admittance.real());

        let reflectance = reflection.magnitude_squared();

        let transmittance = transmission.magnitude_squared().multiply(&flux_ratio);

        let absorptance = transmittance
            .constant(<R::Scalar as One>::one())
            .subtract(&reflectance)
            .subtract(&transmittance);

        Self {
            reflectance,
            transmittance,
            absorptance,
        }
    }

    /// Consume the value and return its power arrays.
    pub fn into_parts(self) -> (R, R, R) {
        (self.reflectance, self.transmittance, self.absorptance)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Ix0;

    use crate::{
        algebra::ScalarAlgebra,
        test_support::{
            C, TOLERANCE,
            assertions::assert_real_close,
            jet::{J0, zero_jet_from_value},
        },
    };

    use super::{PlaneWaveAmplitudes, PlaneWaveObservables, PlaneWavePower};

    #[test]
    fn amplitudes_store_reflection_and_transmission() {
        let amplitudes = PlaneWaveAmplitudes::new(1, 2);

        assert_eq!(amplitudes.reflection(), &1);
        assert_eq!(amplitudes.transmission(), &2);
    }

    #[test]
    fn amplitudes_into_parts_preserves_component_order() {
        let amplitudes = PlaneWaveAmplitudes::new(1, 2);

        let (reflection, transmission) = amplitudes.into_parts();

        assert_eq!(reflection, 1);
        assert_eq!(transmission, 2);
    }

    #[test]
    fn amplitudes_map_transforms_both_components() {
        let amplitudes = PlaneWaveAmplitudes::new(1, 2);

        let mapped = amplitudes.map(|value| value.to_string());

        assert_eq!(mapped.reflection(), "1");
        assert_eq!(mapped.transmission(), "2");
    }

    #[test]
    fn power_stores_reflectance_transmittance_and_absorptance() {
        let power = PlaneWavePower::new(1, 2, 3);

        assert_eq!(power.reflectance(), &1);
        assert_eq!(power.transmittance(), &2);
        assert_eq!(power.absorptance(), &3);
    }

    #[test]
    fn power_into_parts_preserves_component_order() {
        let power = PlaneWavePower::new(1, 2, 3);

        let (reflectance, transmittance, absorptance) = power.into_parts();

        assert_eq!(reflectance, 1);
        assert_eq!(transmittance, 2);
        assert_eq!(absorptance, 3);
    }

    #[test]
    fn power_map_transforms_all_components() {
        let power = PlaneWavePower::new(1, 2, 3);

        let mapped = power.map(|value| value.to_string());

        assert_eq!(mapped.reflectance(), "1");
        assert_eq!(mapped.transmittance(), "2");
        assert_eq!(mapped.absorptance(), "3");
    }

    #[test]
    fn observables_store_amplitudes_and_power() {
        let amplitudes = PlaneWaveAmplitudes::new(1, 2);
        let power = PlaneWavePower::new(3, 4, 5);

        let observables = PlaneWaveObservables::new(amplitudes.clone(), power.clone());

        assert_eq!(observables.amplitudes(), &amplitudes);
        assert_eq!(observables.power(), &power);
    }

    #[test]
    fn observables_forward_component_accessors() {
        let observables =
            PlaneWaveObservables::new(PlaneWaveAmplitudes::new(1, 2), PlaneWavePower::new(3, 4, 5));

        assert_eq!(observables.reflection(), &1);
        assert_eq!(observables.transmission(), &2);
        assert_eq!(observables.reflectance(), &3);
        assert_eq!(observables.transmittance(), &4);
        assert_eq!(observables.absorptance(), &5);
    }

    #[test]
    fn observables_into_parts_preserves_groups() {
        let observables =
            PlaneWaveObservables::new(PlaneWaveAmplitudes::new(1, 2), PlaneWavePower::new(3, 4, 5));

        let (amplitudes, power) = observables.into_parts();

        assert_eq!(amplitudes, PlaneWaveAmplitudes::new(1, 2));
        assert_eq!(power, PlaneWavePower::new(3, 4, 5));
    }

    #[test]
    fn observables_map_transforms_complex_and_real_storage_independently() {
        let observables =
            PlaneWaveObservables::new(PlaneWaveAmplitudes::new(1, 2), PlaneWavePower::new(3, 4, 5));

        let mapped = observables.map(
            |value| format!("complex-{value}"),
            |value| format!("real-{value}"),
        );

        assert_eq!(mapped.reflection(), "complex-1");
        assert_eq!(mapped.transmission(), "complex-2");
        assert_eq!(mapped.reflectance(), "real-3");
        assert_eq!(mapped.transmittance(), "real-4");
        assert_eq!(mapped.absorptance(), "real-5");
    }

    #[test]
    fn observables_map_amplitudes_leaves_power_unchanged() {
        let observables =
            PlaneWaveObservables::new(PlaneWaveAmplitudes::new(1, 2), PlaneWavePower::new(3, 4, 5));

        let mapped =
            observables.map_amplitudes(|amplitudes| amplitudes.map(|value| value.to_string()));

        assert_eq!(mapped.reflection(), "1");
        assert_eq!(mapped.transmission(), "2");
        assert_eq!(mapped.reflectance(), &3);
        assert_eq!(mapped.transmittance(), &4);
        assert_eq!(mapped.absorptance(), &5);
    }

    #[test]
    fn observables_map_power_leaves_amplitudes_unchanged() {
        let observables =
            PlaneWaveObservables::new(PlaneWaveAmplitudes::new(1, 2), PlaneWavePower::new(3, 4, 5));

        let mapped = observables.map_power(|power| power.map(|value| value.to_string()));

        assert_eq!(mapped.reflection(), &1);
        assert_eq!(mapped.transmission(), &2);
        assert_eq!(mapped.reflectance(), "3");
        assert_eq!(mapped.transmittance(), "4");
        assert_eq!(mapped.absorptance(), "5");
    }

    #[test]
    fn accessors_work_with_non_copy_storage() {
        let observables = PlaneWaveObservables::new(
            PlaneWaveAmplitudes::new(String::from("reflection"), String::from("transmission")),
            PlaneWavePower::new(
                String::from("reflectance"),
                String::from("transmittance"),
                String::from("absorptance"),
            ),
        );

        assert_eq!(observables.reflection(), "reflection");
        assert_eq!(observables.transmission(), "transmission");
        assert_eq!(observables.reflectance(), "reflectance");
        assert_eq!(observables.transmittance(), "transmittance");
        assert_eq!(observables.absorptance(), "absorptance");
    }

    #[test]
    fn map_consumes_non_clone_storage() {
        #[derive(Debug, PartialEq)]
        struct NonClone(i32);

        let observables = PlaneWaveObservables::new(
            PlaneWaveAmplitudes::new(NonClone(1), NonClone(2)),
            PlaneWavePower::new(NonClone(3), NonClone(4), NonClone(5)),
        );

        let mapped = observables.map(|value| value.0 * 10, |value| value.0 * 100);

        assert_eq!(mapped.reflection(), &10);
        assert_eq!(mapped.transmission(), &20);
        assert_eq!(mapped.reflectance(), &300);
        assert_eq!(mapped.transmittance(), &400);
        assert_eq!(mapped.absorptance(), &500);
    }

    type Algebra = J0;

    fn scalar(value: impl Into<C>) -> Algebra {
        zero_jet_from_value(value.into())
    }

    fn value<J>(jet: &J) -> J::Scalar
    where
        J: ScalarAlgebra<Dimension = Ix0>,
        J::Scalar: Copy,
    {
        jet.value()[()]
    }

    #[test]
    fn from_amplitudes_and_admittances_gives_unit_transmission_for_equal_admittances() {
        let amplitudes = PlaneWaveAmplitudes::new(scalar(0.0), scalar(1.0));

        let incident_admittance = scalar(2.0);
        let transmitted_admittance = scalar(2.0);

        let power = PlaneWavePower::from_amplitudes_and_admittance(
            amplitudes.reflection(),
            amplitudes.transmission(),
            &incident_admittance,
            &transmitted_admittance,
        );

        assert_real_close(value(power.reflectance()), 0.0, TOLERANCE);
        assert_real_close(value(power.transmittance()), 1.0, TOLERANCE);
        assert_real_close(value(power.absorptance()), 0.0, TOLERANCE);
    }

    #[test]
    fn from_amplitudes_computes_partial_reflection_and_transmission() {
        let amplitudes = PlaneWaveAmplitudes::new(scalar(0.5), scalar(0.5));

        let incident_admittance = scalar(1.0);
        let transmitted_admittance = scalar(1.0);

        let power = PlaneWavePower::from_amplitudes_and_admittance(
            amplitudes.reflection(),
            amplitudes.transmission(),
            &incident_admittance,
            &transmitted_admittance,
        );

        assert_real_close(value(power.reflectance()), 0.25, TOLERANCE);
        assert_real_close(value(power.transmittance()), 0.25, TOLERANCE);
        assert_real_close(value(power.absorptance()), 0.50, TOLERANCE);
    }

    #[test]
    fn from_amplitudes_applies_transmitted_to_incident_flux_ratio() {
        let amplitudes = PlaneWaveAmplitudes::new(scalar(0.0), scalar(0.5));

        let incident_admittance = scalar(2.0);
        let transmitted_admittance = scalar(6.0);

        let power = PlaneWavePower::from_amplitudes_and_admittance(
            amplitudes.reflection(),
            amplitudes.transmission(),
            &incident_admittance,
            &transmitted_admittance,
        );

        // T = |0.5|² * 6/2 = 0.75
        assert_real_close(value(power.reflectance()), 0.0, TOLERANCE);
        assert_real_close(value(power.transmittance()), 0.75, TOLERANCE);
        assert_real_close(value(power.absorptance()), 0.25, TOLERANCE);
    }

    #[test]
    fn from_amplitudes_uses_complex_magnitude_squared() {
        let amplitudes =
            PlaneWaveAmplitudes::new(scalar(C::new(0.3, 0.4)), scalar(C::new(0.0, 0.5)));

        let incident_admittance = scalar(1.0);
        let transmitted_admittance = scalar(2.0);

        let power = PlaneWavePower::from_amplitudes_and_admittance(
            amplitudes.reflection(),
            amplitudes.transmission(),
            &incident_admittance,
            &transmitted_admittance,
        );

        // |0.3 + 0.4i|² = 0.25
        //
        // |0.5i|² * 2 = 0.5
        //
        // A = 1 - 0.25 - 0.5 = 0.25
        assert_real_close(value(power.reflectance()), 0.25, TOLERANCE);
        assert_real_close(value(power.transmittance()), 0.50, TOLERANCE);
        assert_real_close(value(power.absorptance()), 0.25, TOLERANCE);
    }

    #[test]
    fn from_amplitudes_uses_real_parts_for_flux_ratio() {
        let amplitudes = PlaneWaveAmplitudes::new(scalar(0.0), scalar(0.5));

        let incident_admittance = scalar(C::new(2.0, 20.0));

        let transmitted_admittance = scalar(C::new(6.0, -40.0));

        let power = PlaneWavePower::from_amplitudes_and_admittance(
            amplitudes.reflection(),
            amplitudes.transmission(),
            &incident_admittance,
            &transmitted_admittance,
        );

        assert_real_close(value(power.reflectance()), 0.0, TOLERANCE);
        assert_real_close(value(power.transmittance()), 0.75, TOLERANCE);
        assert_real_close(value(power.absorptance()), 0.25, TOLERANCE);
    }
}

#[cfg(test)]
mod projection_tests {
    use crate::{
        backend::{Scatter2Entries, Scatter2ExteriorContext},
        test_support::{
            C, TOLERANCE,
            assertions::{assert_complex_close, assert_real_close},
            jet::{J0, zero_jet_from_value},
        },
    };

    use super::*;

    use ndarray::{Array0, Ix0, arr0};

    type Algebra = J0;

    fn scalar(value: impl Into<C>) -> Algebra {
        zero_jet_from_value(value.into())
    }

    fn value<J>(jet: &J) -> J::Scalar
    where
        J: ScalarAlgebra<Dimension = Ix0>,
        J::Scalar: Copy,
    {
        jet.value()[()]
    }

    fn entries(
        s11: impl Into<C>,
        s12: impl Into<C>,
        s21: impl Into<C>,
        s22: impl Into<C>,
    ) -> Scatter2Entries<Algebra> {
        Scatter2Entries::from_parts(scalar(s11), scalar(s12), scalar(s21), scalar(s22))
    }

    fn exterior_context(
        left_admittance: impl Into<C>,
        right_admittance: impl Into<C>,
    ) -> Scatter2ExteriorContext<Algebra> {
        Scatter2ExteriorContext::from_parts(scalar(left_admittance), scalar(right_admittance))

        // If no constructor exists, use:
        //
        // Scatter2ExteriorContext {
        //     left_admittance: scalar(left_admittance),
        //     right_admittance: scalar(right_admittance),
        // }
    }

    #[test]
    fn project_amplitudes_from_left_returns_s11_and_s21() {
        let entries = entries(1.0, 2.0, 3.0, 4.0);
        let exterior = exterior_context(5.0, 6.0);

        let amplitudes = entries.project_amplitudes(&exterior, IncidentSide::Left);

        assert_complex_close(value(amplitudes.reflection()), C::new(1.0, 0.0), TOLERANCE);
        assert_complex_close(
            value(amplitudes.transmission()),
            C::new(3.0, 0.0),
            TOLERANCE,
        );
    }

    #[test]
    fn project_amplitudes_from_right_returns_s22_and_s12() {
        let entries = entries(1.0, 2.0, 3.0, 4.0);
        let exterior = exterior_context(5.0, 6.0);

        let amplitudes = entries.project_amplitudes(&exterior, IncidentSide::Right);

        assert_complex_close(value(amplitudes.reflection()), C::new(4.0, 0.0), TOLERANCE);
        assert_complex_close(
            value(amplitudes.transmission()),
            C::new(2.0, 0.0),
            TOLERANCE,
        );
    }

    #[test]
    fn project_amplitudes_does_not_depend_on_exterior_context() {
        let entries = entries(
            C::new(0.25, -0.5),
            C::new(-0.75, 0.25),
            C::new(1.25, 0.5),
            C::new(-0.5, -0.25),
        );

        let first_context = exterior_context(1.0, 2.0);
        let second_context = exterior_context(17.0, 31.0);

        let first = entries.project_amplitudes(&first_context, IncidentSide::Left);
        let second = entries.project_amplitudes(&second_context, IncidentSide::Left);

        assert_complex_close(
            value(first.reflection()),
            value(second.reflection()),
            TOLERANCE,
        );
        assert_complex_close(
            value(first.transmission()),
            value(second.transmission()),
            TOLERANCE,
        );
    }

    #[test]
    fn project_power_from_left_uses_left_incident_admittance() {
        let entries = entries(
            0.25, // reflection from left
            0.75, 0.50, // transmission from left
            0.125,
        );

        let exterior = exterior_context(
            2.0, // incident admittance
            8.0, // transmitted admittance
        );

        let power = entries.project_power(&exterior, IncidentSide::Left);

        // R = |0.25|² = 0.0625
        //
        // T = |0.50|² Re(8) / Re(2)
        //   = 0.25 * 4
        //   = 1
        //
        // A = 1 - R - T = -0.0625
        assert_real_close(value(power.reflectance()), 0.0625, TOLERANCE);
        assert_real_close(value(power.transmittance()), 1.0, TOLERANCE);
        assert_real_close(value(power.absorptance()), -0.0625, TOLERANCE);
    }

    #[test]
    fn project_power_from_right_swaps_exterior_admittances() {
        let entries = entries(
            0.125, 0.50, // transmission from right
            0.75, 0.25, // reflection from right
        );

        let exterior = exterior_context(
            2.0, // transmitted when incident from right
            8.0, // incident when incident from right
        );

        let power = entries.project_power(&exterior, IncidentSide::Right);

        // R = |0.25|² = 0.0625
        //
        // T = |0.50|² Re(2) / Re(8)
        //   = 0.25 * 0.25
        //   = 0.0625
        //
        // A = 1 - R - T = 0.875
        assert_real_close(value(power.reflectance()), 0.0625, TOLERANCE);
        assert_real_close(value(power.transmittance()), 0.0625, TOLERANCE);
        assert_real_close(value(power.absorptance()), 0.875, TOLERANCE);
    }

    #[test]
    fn project_power_uses_real_parts_of_admittances() {
        let entries = entries(0.0, 0.0, 0.5, 0.0);

        let exterior = exterior_context(C::new(2.0, 100.0), C::new(6.0, -100.0));

        let power = entries.project_power(&exterior, IncidentSide::Left);

        // Only the real parts enter the flux ratio:
        //
        // T = |0.5|² * 6/2 = 0.75
        assert_real_close(value(power.reflectance()), 0.0, TOLERANCE);
        assert_real_close(value(power.transmittance()), 0.75, TOLERANCE);
        assert_real_close(value(power.absorptance()), 0.25, TOLERANCE);
    }
}
