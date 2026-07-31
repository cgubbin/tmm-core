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
}
