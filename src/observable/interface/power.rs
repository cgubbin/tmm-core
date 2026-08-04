//! Normalized interface-resolved power flux.
//!
//! Flux uses a global left-to-right sign convention independent of the
//! incident side:
//!
//! - positive values carry power towards the right;
//! - negative values carry power towards the left.
//!
//! Every quantity is normalized by the positive magnitude of the incident
//! unit-amplitude wave flux.

use std::ops::Neg;

use crate::{
    ComplexScalar, SpatialProfile, SpatialProfileError,
    algebra::{Jet, RealScalarAlgebra, ScalarAlgebra},
    field::{ScalarField, ScalarFieldView1},
    observable::BoundaryWaves,
};

use super::{InterfaceWaveData, Interfaces};

use ndarray::Dimension;
use num_traits::One;

/// Normalized signed power-flux quantities associated with a pair of
/// direction-labelled waves.
///
/// All fluxes are normalized by the magnitude of the incident-wave flux.
/// Positive values point in the global left-to-right stack direction and
/// negative values point right-to-left.
///
/// `forward_flux` and `backward_flux` are the contributions associated with
/// the direction-labelled waves. `net_flux` is evaluated from the complete
/// boundary state.
///
/// In lossless propagating media:
///
/// ```text
/// net_flux = forward_flux + backward_flux.
/// ```
///
/// In lossy or evanescent media, interference terms may prevent the
/// direction-labelled quantities from forming a complete additive
/// decomposition.
#[derive(Clone, Debug, PartialEq)]
pub struct DirectedPower<R> {
    forward_flux: R,
    backward_flux: R,
    net_flux: R,
}

impl<R, D> SpatialProfile<D::Smaller> for DirectedPower<ScalarField<R, D>>
where
    D: Dimension,
    D::Smaller: Dimension<Larger = D>,
{
    type Profile<'a>
        = DirectedPower<ScalarFieldView1<'a, R>>
    where
        Self: 'a;

    fn spatial_profile(
        &self,
        excitation_index: &D::Smaller,
    ) -> Result<Self::Profile<'_>, SpatialProfileError> {
        Ok(DirectedPower {
            forward_flux: self.forward_flux.profile_last_axis(excitation_index)?,
            backward_flux: self.backward_flux.profile_last_axis(excitation_index)?,
            net_flux: self.net_flux.profile_last_axis(excitation_index)?,
        })
    }
}

impl<R> DirectedPower<R> {
    pub(crate) fn new(forward_flux: R, backward_flux: R, net_flux: R) -> Self {
        Self {
            forward_flux,
            backward_flux,
            net_flux,
        }
    }

    /// Return the flux associated with the forward-labelled wave.
    pub fn forward_flux(&self) -> &R {
        &self.forward_flux
    }

    /// Return the flux associated with the backward-labelled wave.
    ///
    /// This is a signed quantity and is normally negative for a propagating
    /// wave carrying energy in the reverse stack direction.
    pub fn backward_flux(&self) -> &R {
        &self.backward_flux
    }

    /// Return the physical time-averaged normal Poynting flux.
    pub fn net_flux(&self) -> &R {
        &self.net_flux
    }

    pub fn into_parts(self) -> (R, R, R) {
        (self.forward_flux, self.backward_flux, self.net_flux)
    }

    pub fn map<U>(self, mut f: impl FnMut(R) -> U) -> DirectedPower<U> {
        DirectedPower {
            forward_flux: f(self.forward_flux),
            backward_flux: f(self.backward_flux),
            net_flux: f(self.net_flux),
        }
    }
}

/// Normalized normal power flux immediately on either side of one interface.
///
/// At an ordinary source-free interface, `left_net_flux` and
/// `right_net_flux` should agree up to numerical error. Their difference is a
/// direct interface-level conservation diagnostic.
///
/// The directional decompositions on the two sides need not agree because
/// they are expressed using different local characteristic admittances.
#[derive(Clone, Debug, PartialEq)]
pub struct InterfacePower<R> {
    left: DirectedPower<R>,
    right: DirectedPower<R>,
}

impl<R, D> SpatialProfile<D::Smaller> for InterfacePower<ScalarField<R, D>>
where
    D: Dimension,
    D::Smaller: Dimension<Larger = D>,
{
    type Profile<'a>
        = InterfacePower<ScalarFieldView1<'a, R>>
    where
        Self: 'a;

    fn spatial_profile(
        &self,
        excitation_index: &D::Smaller,
    ) -> Result<Self::Profile<'_>, SpatialProfileError> {
        Ok(InterfacePower {
            left: self.left.spatial_profile(excitation_index)?,
            right: self.right.spatial_profile(excitation_index)?,
        })
    }
}

impl<R> InterfacePower<R> {
    pub(crate) fn new(left: DirectedPower<R>, right: DirectedPower<R>) -> Self {
        Self { left, right }
    }

    pub fn left(&self) -> &DirectedPower<R> {
        &self.left
    }

    pub fn right(&self) -> &DirectedPower<R> {
        &self.right
    }

    /// Return the normalized net flux immediately to the interface's left.
    pub fn left_net_flux(&self) -> &R {
        self.left.net_flux()
    }

    /// Return the normalized net flux immediately to the interface's right.
    pub fn right_net_flux(&self) -> &R {
        self.right.net_flux()
    }

    pub fn into_parts(self) -> (DirectedPower<R>, DirectedPower<R>) {
        (self.left, self.right)
    }

    pub fn map<U>(self, mut f: impl FnMut(R) -> U) -> InterfacePower<U> {
        InterfacePower {
            left: self.left.map(&mut f),
            right: self.right.map(f),
        }
    }
}

impl<A> Interfaces<InterfaceWaveData<A>> {
    pub(crate) fn into_power(
        self,
        incident_flux_magnitude: &A::RealJet,
    ) -> Interfaces<InterfacePower<A::RealJet>>
    where
        A: RealScalarAlgebra + Clone,
        A::RealJet: ScalarAlgebra,
        <A::RealJet as Jet>::Scalar: Neg<Output = <A::RealJet as Jet>::Scalar> + One,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        self.map(|interface| {
            let (left, right) = interface.into_parts();

            let (left_waves, left_admittance) = left.into_parts();

            let (right_waves, right_admittance) = right.into_parts();

            let left_power =
                project_directed_power(left_waves, left_admittance, incident_flux_magnitude);

            let right_power =
                project_directed_power(right_waves, right_admittance, incident_flux_magnitude);

            InterfacePower::new(left_power, right_power)
        })
    }
}

pub(crate) fn project_directed_power<A>(
    waves: BoundaryWaves<A>,
    admittance: A,
    incident_flux_magnitude: &A::RealJet,
) -> DirectedPower<A::RealJet>
where
    A: RealScalarAlgebra + Clone,
    A::RealJet: ScalarAlgebra,
    <A::RealJet as Jet>::Scalar: Neg<Output = <A::RealJet as Jet>::Scalar> + One,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    let state = waves.clone().into_state(&admittance);

    let admittance_real = admittance.real();

    let forward_flux = waves
        .forward()
        .magnitude_squared()
        .multiply(&admittance_real)
        .divide(incident_flux_magnitude);

    let backward_flux = waves
        .backward()
        .magnitude_squared()
        .multiply(&admittance_real)
        .scale(-<A::RealJet as Jet>::Scalar::one())
        .divide(incident_flux_magnitude);

    let net_flux = state
        .field()
        .hermitian_product(state.secondary())
        .imaginary()
        .divide(incident_flux_magnitude);

    DirectedPower::new(forward_flux, backward_flux, net_flux)
}

#[cfg(test)]
mod tests {
    use super::{DirectedPower, InterfacePower};

    #[test]
    fn project_directed_power_stores_all_fluxes() {
        let power = DirectedPower::new(1, 2, 3);

        assert_eq!(power.forward_flux(), &1);
        assert_eq!(power.backward_flux(), &2);
        assert_eq!(power.net_flux(), &3);
    }

    #[test]
    fn project_directed_power_into_parts_preserves_order() {
        let power = DirectedPower::new(1, 2, 3);

        assert_eq!(power.into_parts(), (1, 2, 3));
    }

    #[test]
    fn project_directed_power_map_transforms_all_fluxes() {
        let power = DirectedPower::new(1, 2, 3);

        let mapped = power.map(|value| format!("flux-{value}"));

        assert_eq!(mapped.forward_flux(), "flux-1");
        assert_eq!(mapped.backward_flux(), "flux-2");
        assert_eq!(mapped.net_flux(), "flux-3");
    }

    #[test]
    fn interface_power_stores_both_sides() {
        let left = DirectedPower::new(1, 2, 3);
        let right = DirectedPower::new(4, 5, 6);

        let interface = InterfacePower::new(left.clone(), right.clone());

        assert_eq!(interface.left(), &left);
        assert_eq!(interface.right(), &right);
        assert_eq!(interface.left_net_flux(), &3);
        assert_eq!(interface.right_net_flux(), &6);
    }

    #[test]
    fn interface_power_into_parts_preserves_side_order() {
        let interface =
            InterfacePower::new(DirectedPower::new(1, 2, 3), DirectedPower::new(4, 5, 6));

        let (left, right) = interface.into_parts();

        assert_eq!(left, DirectedPower::new(1, 2, 3));
        assert_eq!(right, DirectedPower::new(4, 5, 6));
    }

    #[test]
    fn interface_power_map_transforms_both_sides() {
        let interface =
            InterfacePower::new(DirectedPower::new(1, 2, 3), DirectedPower::new(4, 5, 6));

        let mapped = interface.map(|value| value.to_string());

        assert_eq!(mapped.left().forward_flux(), "1");
        assert_eq!(mapped.left().backward_flux(), "2");
        assert_eq!(mapped.left().net_flux(), "3");
        assert_eq!(mapped.right().forward_flux(), "4");
        assert_eq!(mapped.right().backward_flux(), "5");
        assert_eq!(mapped.right().net_flux(), "6");
    }

    #[test]
    fn mapping_consumes_non_clone_storage() {
        #[derive(Debug, PartialEq)]
        struct NonClone(i32);

        let interface = InterfacePower::new(
            DirectedPower::new(NonClone(1), NonClone(2), NonClone(3)),
            DirectedPower::new(NonClone(4), NonClone(5), NonClone(6)),
        );

        let mapped = interface.map(|value| value.0 * 10);

        assert_eq!(mapped.left().forward_flux(), &10);
        assert_eq!(mapped.left().backward_flux(), &20);
        assert_eq!(mapped.left().net_flux(), &30);
        assert_eq!(mapped.right().forward_flux(), &40);
        assert_eq!(mapped.right().backward_flux(), &50);
        assert_eq!(mapped.right().net_flux(), &60);
    }
}

#[cfg(test)]
mod projection_tests {
    use approx::assert_relative_eq;
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use super::*;
    use crate::{
        algebra::{ArrayJet0, Jet0, RealParameter},
        observable::BoundaryWaves,
    };

    type C = Complex64;
    type A = ArrayJet0<C, Ix0, RealParameter>;
    type RA = ArrayJet0<f64, Ix0, RealParameter>;

    const TOLERANCE: f64 = 1.0e-12;

    fn jet(value: C) -> A {
        Jet0::new(arr0(value))
    }

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn scalar_real(value: &RA) -> f64 {
        value.value()[()]
    }

    fn assert_real_close(actual: f64, expected: f64) {
        assert_relative_eq!(
            actual,
            expected,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn pure_forward_wave_has_positive_unit_flux() {
        let waves = BoundaryWaves::new(jet(c(1.0, 0.0)), jet(c(0.0, 0.0)));

        let admittance = jet(c(2.5, 0.0));
        let incident_flux = admittance.real();

        let power = project_directed_power(waves, admittance, &incident_flux);

        assert_real_close(scalar_real(power.forward_flux()), 1.0);

        assert_real_close(scalar_real(power.backward_flux()), 0.0);

        assert_real_close(scalar_real(power.net_flux()), 1.0);
    }

    #[test]
    fn pure_backward_wave_has_negative_unit_flux() {
        let waves = BoundaryWaves::new(jet(c(0.0, 0.0)), jet(c(1.0, 0.0)));

        let admittance = jet(c(2.5, 0.0));
        let incident_flux = admittance.real();

        let power = project_directed_power(waves, admittance, &incident_flux);

        assert_real_close(scalar_real(power.forward_flux()), 0.0);

        assert_real_close(scalar_real(power.backward_flux()), -1.0);

        assert_real_close(scalar_real(power.net_flux()), -1.0);
    }

    #[test]
    fn complex_amplitudes_use_magnitude_squared() {
        let forward = c(1.0, 2.0);
        let backward = c(-0.5, 0.25);

        let waves = BoundaryWaves::new(jet(forward), jet(backward));

        let admittance = jet(c(3.0, 0.0));
        let incident_flux = admittance.real();

        let power = project_directed_power(waves, admittance, &incident_flux);

        assert_real_close(scalar_real(power.forward_flux()), forward.norm_sqr());

        assert_real_close(scalar_real(power.backward_flux()), -backward.norm_sqr());
    }

    #[test]
    fn lossless_mixed_wave_net_flux_is_directional_sum() {
        let waves = BoundaryWaves::new(jet(c(0.8, 0.2)), jet(c(-0.3, 0.1)));

        let admittance = jet(c(2.0, 0.0));
        let incident_flux = admittance.real();

        let power = project_directed_power(waves, admittance, &incident_flux);

        let directional_sum =
            scalar_real(power.forward_flux()) + scalar_real(power.backward_flux());

        assert_real_close(scalar_real(power.net_flux()), directional_sum);
    }

    #[test]
    fn normalization_uses_incident_flux_magnitude() {
        let waves = BoundaryWaves::new(jet(c(1.0, 0.0)), jet(c(0.0, 0.0)));

        let local_admittance = jet(c(6.0, 0.0));

        let incident_flux = Jet0::new(arr0(2.0));

        let power = project_directed_power(waves, local_admittance, &incident_flux);

        assert_real_close(scalar_real(power.forward_flux()), 3.0);

        assert_real_close(scalar_real(power.net_flux()), 3.0);
    }

    #[test]
    fn lossy_mixed_wave_net_flux_uses_full_state_expression() {
        let forward = c(0.8, 0.3);
        let backward = c(-0.2, 0.5);
        let admittance_value = c(2.0, 0.7);

        let waves = BoundaryWaves::new(jet(forward), jet(backward));

        let admittance = jet(admittance_value);

        // Use an independent positive normalization.
        let incident_flux = Jet0::new(arr0(1.5));

        let power = project_directed_power(waves, admittance, &incident_flux);

        let xi = -C::i() * admittance_value;
        let field = forward + backward;
        let secondary = xi * (backward - forward);

        let expected = (field.conj() * secondary).im / 1.5;

        assert_real_close(scalar_real(power.net_flux()), expected);
    }
}
