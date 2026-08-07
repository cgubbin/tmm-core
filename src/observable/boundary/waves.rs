//! Directional-wave amplitudes at planar boundaries.
//!
//! Direction labels use the global stack orientation:
//!
//! - `forward` propagates towards increasing stack coordinate;
//! - `backward` propagates towards decreasing stack coordinate.
//!
//! The amplitudes are expressed in the directional basis of the local medium.
//! They are therefore suitable for propagation within a homogeneous layer,
//! but are not themselves continuity variables across unlike media.
//!
//! Conversion to [`BoundaryState`] produces the canonical isotropic state used
//! for interface continuity and physical projections.

use ndarray::Dimension;

use crate::{ComplexScalar, algebra::ScalarAlgebra};

use super::{BoundaryState, LayerBoundaryStates};

/// Forward- and backward-propagating amplitudes at one planar boundary.
///
/// The amplitudes are expressed in the directional basis of the medium
/// containing the boundary. They are therefore not generally continuous
/// across an interface between different media.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryWaves<A> {
    forward: A,
    backward: A,
}

impl<A> BoundaryWaves<A> {
    /// Construct directional boundary amplitudes.
    pub(crate) const fn new(forward: A, backward: A) -> Self {
        Self { forward, backward }
    }

    /// Return the forward-propagating amplitude.
    pub fn forward(&self) -> &A {
        &self.forward
    }

    /// Return the backward-propagating amplitude.
    pub fn backward(&self) -> &A {
        &self.backward
    }

    /// Consume the waves and return `(forward, backward)`.
    pub fn into_parts(self) -> (A, A) {
        (self.forward, self.backward)
    }

    /// Transform both directional amplitudes.
    pub fn map<B>(self, mut map: impl FnMut(A) -> B) -> BoundaryWaves<B> {
        BoundaryWaves {
            forward: map(self.forward),
            backward: map(self.backward),
        }
    }

    pub(crate) fn state(&self, admittance: &A) -> BoundaryState<A>
    where
        A: ScalarAlgebra + Clone,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        self.clone().into_state(admittance)
    }

    /// Convert directional amplitudes into the canonical isotropic boundary
    /// state for `admittance`.
    ///
    /// The conversion uses
    ///
    /// ```text
    /// ξ         = -i Y
    /// field     = forward + backward
    /// secondary = ξ (backward - forward).
    /// ```
    /// Equivalently:
    ///
    /// ```text
    /// secondary = factor⁻¹ ∂z field,
    /// ```
    ///
    /// where `factor = μ` for TE and `factor = ε` for TM.
    pub(crate) fn into_state(self, admittance: &A) -> BoundaryState<A>
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        let (forward, backward) = self.into_parts();

        let characteristic_slope = admittance.scale(-<A::Scalar as ComplexScalar>::i());

        let field = forward.add(&backward);

        let secondary = characteristic_slope.multiply(&backward.subtract(&forward));

        BoundaryState::new(field, secondary)
    }
}

impl<A> From<crate::waves::BidirectionalWaves<A>> for BoundaryWaves<A> {
    fn from(value: crate::waves::BidirectionalWaves<A>) -> Self {
        let (forward, backward) = value.into_parts();
        Self::new(forward, backward)
    }
}

/// Directional amplitudes at both boundaries of one finite layer.
///
/// Both boundaries are represented in the directional basis of the finite
/// layer itself.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerBoundaryWaves<A> {
    left: BoundaryWaves<A>,
    right: BoundaryWaves<A>,
}

impl<A> LayerBoundaryWaves<A> {
    /// Construct the boundary waves of a finite layer.
    pub(crate) const fn new(left: BoundaryWaves<A>, right: BoundaryWaves<A>) -> Self {
        Self { left, right }
    }

    /// Return the amplitudes at the layer's left boundary.
    pub fn left(&self) -> &BoundaryWaves<A> {
        &self.left
    }

    /// Return the amplitudes at the layer's right boundary.
    pub fn right(&self) -> &BoundaryWaves<A> {
        &self.right
    }

    /// Consume the container and return `(left, right)`.
    pub fn into_parts(self) -> (BoundaryWaves<A>, BoundaryWaves<A>) {
        (self.left, self.right)
    }

    /// Transform every directional amplitude at both boundaries.
    pub fn map<B>(self, mut map: impl FnMut(A) -> B) -> LayerBoundaryWaves<B> {
        LayerBoundaryWaves {
            left: self.left.map(&mut map),
            right: self.right.map(map),
        }
    }

    /// Convert both boundaries into canonical isotropic states using the
    /// finite layer's characteristic admittance.
    pub(crate) fn into_states(self, admittance: &A) -> LayerBoundaryStates<A>
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        let (left, right) = self.into_parts();

        LayerBoundaryStates::new(left.into_state(admittance), right.into_state(admittance))
    }
}

impl<A> From<crate::waves::LayerBoundaryWaves<A>> for LayerBoundaryWaves<A> {
    fn from(value: crate::waves::LayerBoundaryWaves<A>) -> Self {
        let (left, right) = value.into_parts();

        Self::new(BoundaryWaves::from(left), BoundaryWaves::from(right))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use super::*;
    use crate::{
        algebra::{ArrayJet0, Jet0, RealParameter},
        waves::{
            BidirectionalWaves as BackendBoundaryWaves,
            LayerBoundaryWaves as BackendLayerBoundaryWaves,
        },
    };

    type C = Complex64;
    type A = ArrayJet0<C, Ix0, RealParameter>;

    const TOLERANCE: f64 = 1.0e-12;

    fn jet(value: C) -> A {
        Jet0::new(arr0(value))
    }

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn assert_close(actual: C, expected: C) {
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

    fn assert_jet_close(actual: &A, expected: C) {
        assert_close(actual.value()[()], expected);
    }

    #[test]
    fn boundary_waves_store_directional_amplitudes() {
        let waves = BoundaryWaves::new(jet(c(1.0, 0.2)), jet(c(-0.3, 0.4)));

        assert_jet_close(waves.forward(), c(1.0, 0.2));
        assert_jet_close(waves.backward(), c(-0.3, 0.4));
    }

    #[test]
    fn boundary_waves_into_parts_preserves_order() {
        let waves = BoundaryWaves::new(jet(c(1.0, 0.0)), jet(c(2.0, 0.0)));

        let (forward, backward) = waves.into_parts();

        assert_jet_close(&forward, c(1.0, 0.0));
        assert_jet_close(&backward, c(2.0, 0.0));
    }

    #[test]
    fn boundary_waves_map_transforms_both_amplitudes() {
        let waves = BoundaryWaves::new(2.0, 3.0);

        let mapped = waves.map(|value| value * 10.0);

        assert_eq!(mapped.forward(), &20.0);
        assert_eq!(mapped.backward(), &30.0);
    }

    #[test]
    fn layer_boundary_waves_preserve_left_right_order() {
        let left = BoundaryWaves::new(1, 2);
        let right = BoundaryWaves::new(3, 4);

        let waves = LayerBoundaryWaves::new(left, right);

        assert_eq!(waves.left().forward(), &1);
        assert_eq!(waves.left().backward(), &2);
        assert_eq!(waves.right().forward(), &3);
        assert_eq!(waves.right().backward(), &4);
    }

    #[test]
    fn layer_boundary_waves_map_transforms_every_amplitude() {
        let waves = LayerBoundaryWaves::new(BoundaryWaves::new(1, 2), BoundaryWaves::new(3, 4));

        let mapped = waves.map(|value| value * 2);

        assert_eq!(mapped.left().forward(), &2);
        assert_eq!(mapped.left().backward(), &4);
        assert_eq!(mapped.right().forward(), &6);
        assert_eq!(mapped.right().backward(), &8);
    }

    #[test]
    fn pure_forward_wave_converts_to_expected_state() {
        let admittance = jet(c(2.5, 0.0));

        let state = BoundaryWaves::new(jet(c(1.2, -0.3)), jet(c(0.0, 0.0))).into_state(&admittance);

        let amplitude = c(1.2, -0.3);

        assert_jet_close(state.field(), amplitude);
        assert_jet_close(state.secondary(), C::i() * c(2.5, 0.0) * amplitude);
    }

    #[test]
    fn pure_backward_wave_converts_to_expected_state() {
        let admittance = jet(c(2.5, 0.0));

        let state = BoundaryWaves::new(jet(c(0.0, 0.0)), jet(c(-0.4, 0.7))).into_state(&admittance);

        let amplitude = c(-0.4, 0.7);

        assert_jet_close(state.field(), amplitude);
        assert_jet_close(state.secondary(), -C::i() * c(2.5, 0.0) * amplitude);
    }

    #[test]
    fn mixed_waves_convert_to_expected_state() {
        let admittance = jet(c(3.0, 0.0));
        let forward = c(1.0, 0.4);
        let backward = c(-0.2, 0.7);

        let state = BoundaryWaves::new(jet(forward), jet(backward)).into_state(&admittance);

        let xi = -C::i() * c(3.0, 0.0);

        assert_jet_close(state.field(), forward + backward);
        assert_jet_close(state.secondary(), xi * (backward - forward));
    }

    #[test]
    fn layer_waves_convert_both_boundaries_with_same_admittance() {
        let admittance = jet(c(2.0, 0.0));

        let waves = LayerBoundaryWaves::new(
            BoundaryWaves::new(jet(c(1.0, 0.0)), jet(c(0.2, 0.0))),
            BoundaryWaves::new(jet(c(0.7, 0.1)), jet(c(-0.1, 0.2))),
        );

        let states = waves.into_states(&admittance);

        assert_jet_close(states.left().field(), c(1.2, 0.0));

        assert_jet_close(states.right().field(), c(0.6, 0.3));
    }

    #[test]
    fn backend_boundary_waves_convert_without_reordering() {
        let backend = BackendBoundaryWaves::new(jet(c(1.0, 0.2)), jet(c(-0.3, 0.4)));

        let observable = BoundaryWaves::from(backend);

        assert_jet_close(observable.forward(), c(1.0, 0.2));

        assert_jet_close(observable.backward(), c(-0.3, 0.4));
    }

    #[test]
    fn backend_layer_boundary_waves_convert_without_reordering() {
        let backend = BackendLayerBoundaryWaves::new(
            BackendBoundaryWaves::new(jet(c(1.0, 0.0)), jet(c(2.0, 0.0))),
            BackendBoundaryWaves::new(jet(c(3.0, 0.0)), jet(c(4.0, 0.0))),
        );

        let observable = LayerBoundaryWaves::from(backend);

        assert_jet_close(observable.left().forward(), c(1.0, 0.0));

        assert_jet_close(observable.left().backward(), c(2.0, 0.0));

        assert_jet_close(observable.right().forward(), c(3.0, 0.0));

        assert_jet_close(observable.right().backward(), c(4.0, 0.0));
    }

    #[test]
    fn complex_admittance_converts_with_full_characteristic_slope() {
        let forward = c(0.8, -0.2);
        let backward = c(-0.3, 0.5);
        let admittance_value = c(2.0, 0.7);

        let state =
            BoundaryWaves::new(jet(forward), jet(backward)).into_state(&jet(admittance_value));

        let xi = -C::i() * admittance_value;

        assert_jet_close(state.field(), forward + backward);

        assert_jet_close(state.secondary(), xi * (backward - forward));
    }
}
