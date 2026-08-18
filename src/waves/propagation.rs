//! Spatial propagation of directional wave amplitudes.
//!
//! This module reconstructs wave amplitudes away from the reference planes at
//! which they were obtained from a retained backend solution.
//!
//! Propagation is evaluated directly in the scalar algebra. For jet-backed
//! calculations, phase accumulation therefore preserves derivatives with
//! respect to spectral, in-plane, thickness, and other active parameters.
//! Wave amplitudes are not crystallised to ordinary arrays during spatial
//! reconstruction.
//!
//! # Coordinate convention
//!
//! The global longitudinal coordinate increases from left to right. Forward
//! waves propagate towards increasing coordinate and backward waves towards
//! decreasing coordinate.
//!
//! With longitudinal wavevector `κ`, propagation through signed displacement
//! `Δz` uses
//!
//! ```text
//! forward:  a⁺(z + Δz) = a⁺(z) exp(+i κ Δz)
//! backward: a⁻(z + Δz) = a⁻(z) exp(-i κ Δz)
//! ```
//!
//! A positive displacement therefore moves the evaluation point towards the
//! right, while a negative displacement moves it towards the left.
//!
//! # Finite layers
//!
//! [`LayerBoundaryWaves`] retains directional amplitudes at both boundaries of
//! a finite layer. Interior reconstruction deliberately uses different
//! reference planes for the two propagation directions:
//!
//! ```text
//! left boundary              sample                 right boundary
//!      |------------------------|-------------------------|
//!      a⁺(0)  ----------------->|
//!                               |<-----------------  a⁻(d)
//!            offset = z             distance = d - z
//! ```
//! Thus
//!
//! ```text
//! a⁺(z) = a⁺(0) exp[+i κ z]
//! a⁻(z) = a⁻(d) exp[+i κ (d - z)].
//! ```
//!
//! This avoids reconstructing an interior amplitude by dividing through a
//! potentially very small propagation factor in strongly absorbing or
//! evanescent layers.
//!
//! Layer positions preserve their geometric relationship to the finite layer:
//!
//! - [`CanonicalLayerPosition::FromLeft`] is a fixed distance from the left
//!   boundary;
//! - [`CanonicalLayerPosition::FromRight`] is a fixed distance from the right
//!   boundary;
//! - [`CanonicalLayerPosition::Fraction`] follows a fixed fraction of the
//!   layer thickness.
//!
//! The corresponding propagation distances are constructed in the scalar
//! algebra, so structural derivatives include motion of relative sampling
//! positions when appropriate.
//!
//! # Exterior media
//!
//! Exterior samples use [`PropagateWaves`] directly. A point a positive
//! distance from the stack is reached using:
//!
//! - `-distance` in the left exterior;
//! - `+distance` in the right exterior.
//!
//! Higher-level dispatch from [`CanonicalFieldPosition`] to these propagation
//! operations is implemented by the wave-sampling layer.

use nalgebra::ComplexField;
use ndarray::Dimension;

use crate::{
    ComplexScalar,
    algebra::ScalarAlgebra,
    spatial::CanonicalLayerPosition,
    waves::{BidirectionalWaves, LayerBoundaryWaves},
};

/// Propagate bidirectional waves away from one common reference plane.
///
/// The displacement is signed in the global longitudinal coordinate:
/// positive values move towards the right and negative values towards the
/// left.
pub(crate) trait PropagateWaves<A> {
    /// Propagate both directional amplitudes through `distance`.
    ///
    /// Exterior sample distances are fixed caller sampling coordinates and therefore are not seeded
    /// as differential variables; finite-layer relative positions may depend on layer thickness
    /// and are consequently represented in the scalar algebra.
    fn propagate(
        &self,
        longitudinal_wavevector: &A,
        distance: <A::Scalar as ComplexField>::RealField,
    ) -> BidirectionalWaves<A>
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexScalar,
        <A::Scalar as ComplexField>::RealField: Copy,
        A::Dimension: Dimension;
}

/// Reconstruct directional waves at a position inside a finite layer.
///
/// Forward and backward amplitudes are propagated from the layer boundary
/// appropriate to their propagation direction.
pub(crate) trait PropagateLayerWaves<A> {
    /// Reconstruct waves at `position` inside the layer.
    ///
    /// The position retains its geometric relationship to the layer so that
    /// derivatives with respect to layer thickness include the corresponding
    /// motion of relative sampling coordinates.
    fn propagate_to_position(
        &self,
        longitudinal_wavevector: &A,
        thickness: &A,
        position: CanonicalLayerPosition<<A::Scalar as ComplexField>::RealField>,
    ) -> BidirectionalWaves<A>
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexScalar,
        <A::Scalar as ComplexField>::RealField: Copy,
        A::Dimension: Dimension;
}

impl<A> PropagateWaves<A> for BidirectionalWaves<A>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    <A::Scalar as ComplexField>::RealField: Copy,
    A::Dimension: Dimension,
{
    fn propagate(
        &self,
        longitudinal_wavevector: &A,
        distance: <A::Scalar as ComplexField>::RealField,
    ) -> BidirectionalWaves<A> {
        let distance = A::Scalar::from_real(distance);

        let forward_phase = longitudinal_wavevector
            .scale(A::Scalar::i() * distance)
            .exp();

        let backward_phase = longitudinal_wavevector
            .scale(-A::Scalar::i() * distance)
            .exp();

        BidirectionalWaves::new(
            self.forward().multiply(&forward_phase),
            self.backward().multiply(&backward_phase),
        )
    }
}

impl<A> PropagateLayerWaves<A> for LayerBoundaryWaves<A>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    <A::Scalar as ComplexField>::RealField: Copy,
    A::Dimension: Dimension,
{
    fn propagate_to_position(
        &self,
        longitudinal_wavevector: &A,
        thickness: &A,
        position: CanonicalLayerPosition<<A::Scalar as ComplexField>::RealField>,
    ) -> BidirectionalWaves<A> {
        let (distance_from_left, distance_from_right) = match position {
            CanonicalLayerPosition::FromLeft(distance) => {
                let distance =
                    A::filled_constant_like(thickness.value(), A::Scalar::from_real(distance));

                let from_right = thickness.subtract(&distance);

                (distance, from_right)
            }

            CanonicalLayerPosition::FromRight(distance) => {
                let distance =
                    A::filled_constant_like(thickness.value(), A::Scalar::from_real(distance));

                let from_left = thickness.subtract(&distance);

                (from_left, distance)
            }

            CanonicalLayerPosition::Fraction(fraction) => {
                let fraction = A::Scalar::from_real(fraction);

                let from_left = thickness.scale(fraction);

                let from_right = thickness.subtract(&from_left);

                (from_left, from_right)
            }
        };

        let forward_phase = longitudinal_wavevector
            .multiply(&distance_from_left)
            .scale(A::Scalar::i())
            .exp();

        let backward_phase = longitudinal_wavevector
            .multiply(&distance_from_right)
            .scale(A::Scalar::i())
            .exp();

        BidirectionalWaves::new(
            self.left().forward().multiply(&forward_phase),
            self.right().backward().multiply(&backward_phase),
        )
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Ix0, arr0};

    use crate::{
        algebra::{ArrayJet0, ArrayJet1, Jet0, Jet1, RealParameter},
        spatial::CanonicalLayerPosition,
        test_support::{C, TOLERANCE, assertions::assert_complex_close},
        waves::{BidirectionalWaves, LayerBoundaryWaves},
    };

    use super::{PropagateLayerWaves, PropagateWaves};

    type J0 = ArrayJet0<C, Ix0, RealParameter>;
    type J1 = ArrayJet1<C, Ix0, RealParameter>;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn jet0(value: C) -> J0 {
        Jet0::new(arr0(value))
    }

    fn jet1(value: C, first: C) -> J1 {
        Jet1::from_parts(arr0(value), arr0(first))
    }

    fn assert_close(actual: C, expected: C) {
        assert_complex_close(actual, expected, TOLERANCE);
    }

    // ---------------------------------------------------------------------
    // Common-reference propagation
    // ---------------------------------------------------------------------

    #[test]
    fn zero_distance_preserves_both_waves() {
        let waves = BidirectionalWaves::new(jet0(c(2.0, 1.0)), jet0(c(3.0, -1.0)));

        let kappa = jet0(c(1.5, 0.0));

        let sampled = waves.propagate(&kappa, 0.0);

        assert_close(sampled.forward().value()[()], c(2.0, 1.0));

        assert_close(sampled.backward().value()[()], c(3.0, -1.0));
    }

    #[test]
    fn positive_displacement_uses_opposite_directional_phases() {
        let waves = BidirectionalWaves::new(jet0(c(2.0, 0.0)), jet0(c(3.0, 0.0)));

        let kappa = jet0(c(1.5, 0.0));
        let distance = 0.4;

        let sampled = waves.propagate(&kappa, distance);

        let forward_phase = (C::i() * c(1.5, 0.0) * distance).exp();

        let backward_phase = (-C::i() * c(1.5, 0.0) * distance).exp();

        assert_close(sampled.forward().value()[()], c(2.0, 0.0) * forward_phase);

        assert_close(sampled.backward().value()[()], c(3.0, 0.0) * backward_phase);
    }

    #[test]
    fn negative_displacement_reverses_geometric_phase() {
        let waves = BidirectionalWaves::new(jet0(c(2.0, 0.0)), jet0(c(3.0, 0.0)));

        let kappa = jet0(c(1.5, 0.0));
        let distance = 0.4;

        let sampled = waves.propagate(&kappa, -distance);

        let forward_phase = (-C::i() * c(1.5, 0.0) * distance).exp();

        let backward_phase = (C::i() * c(1.5, 0.0) * distance).exp();

        assert_close(sampled.forward().value()[()], c(2.0, 0.0) * forward_phase);

        assert_close(sampled.backward().value()[()], c(3.0, 0.0) * backward_phase);
    }

    // ---------------------------------------------------------------------
    // Finite-layer value propagation
    // ---------------------------------------------------------------------

    #[test]
    fn from_left_zero_uses_left_forward_reference() {
        let boundaries = LayerBoundaryWaves::new(
            BidirectionalWaves::new(jet0(c(2.0, 1.0)), jet0(c(99.0, 0.0))),
            BidirectionalWaves::new(jet0(c(98.0, 0.0)), jet0(c(0.5, -0.25))),
        );

        let kappa = jet0(c(1.5, 0.0));
        let thickness = jet0(c(0.4, 0.0));

        let sampled = boundaries.propagate_to_position(
            &kappa,
            &thickness,
            CanonicalLayerPosition::FromLeft(0.0),
        );

        assert_close(sampled.forward().value()[()], c(2.0, 1.0));

        let phase = (C::i() * c(1.5, 0.0) * 0.4).exp();

        assert_close(sampled.backward().value()[()], c(0.5, -0.25) * phase);
    }

    #[test]
    fn from_right_zero_uses_right_backward_reference() {
        let boundaries = LayerBoundaryWaves::new(
            BidirectionalWaves::new(jet0(c(2.0, 1.0)), jet0(c(99.0, 0.0))),
            BidirectionalWaves::new(jet0(c(98.0, 0.0)), jet0(c(0.5, -0.25))),
        );

        let kappa = jet0(c(1.5, 0.0));
        let thickness = jet0(c(0.4, 0.0));

        let sampled = boundaries.propagate_to_position(
            &kappa,
            &thickness,
            CanonicalLayerPosition::FromRight(0.0),
        );

        let phase = (C::i() * c(1.5, 0.0) * 0.4).exp();

        assert_close(sampled.forward().value()[()], c(2.0, 1.0) * phase);

        assert_close(sampled.backward().value()[()], c(0.5, -0.25));
    }

    #[test]
    fn fractional_position_uses_fraction_of_layer_thickness() {
        let boundaries = LayerBoundaryWaves::new(
            BidirectionalWaves::new(jet0(c(2.0, 0.0)), jet0(c(99.0, 0.0))),
            BidirectionalWaves::new(jet0(c(98.0, 0.0)), jet0(c(3.0, 0.0))),
        );

        let kappa = jet0(c(1.5, 0.0));
        let thickness = jet0(c(0.8, 0.0));

        let sampled = boundaries.propagate_to_position(
            &kappa,
            &thickness,
            CanonicalLayerPosition::Fraction(0.25),
        );

        let from_left = 0.2;
        let from_right = 0.6;

        let forward_phase = (C::i() * c(1.5, 0.0) * from_left).exp();

        let backward_phase = (C::i() * c(1.5, 0.0) * from_right).exp();

        assert_close(sampled.forward().value()[()], c(2.0, 0.0) * forward_phase);

        assert_close(sampled.backward().value()[()], c(3.0, 0.0) * backward_phase);
    }

    // ---------------------------------------------------------------------
    // Thickness derivative semantics
    // ---------------------------------------------------------------------

    #[test]
    fn from_left_position_keeps_left_distance_fixed_under_thickness_derivative() {
        let boundaries = LayerBoundaryWaves::new(
            BidirectionalWaves::new(
                jet1(c(2.0, 0.0), c(0.0, 0.0)),
                jet1(c(0.0, 0.0), c(0.0, 0.0)),
            ),
            BidirectionalWaves::new(
                jet1(c(0.0, 0.0), c(0.0, 0.0)),
                jet1(c(3.0, 0.0), c(0.0, 0.0)),
            ),
        );

        let kappa = jet1(c(1.5, 0.0), c(0.0, 0.0));

        // Thickness is the active variable.
        let thickness = jet1(c(0.8, 0.0), c(1.0, 0.0));

        let sampled = boundaries.propagate_to_position(
            &kappa,
            &thickness,
            CanonicalLayerPosition::FromLeft(0.2),
        );

        /*
         * FromLeft(0.2):
         *
         * distance_from_left  = 0.2       derivative 0
         * distance_from_right = L - 0.2   derivative 1
         */
        assert_close(sampled.forward().first()[()], c(0.0, 0.0));

        let distance_from_right = 0.6;

        let backward = c(3.0, 0.0) * (C::i() * c(1.5, 0.0) * distance_from_right).exp();

        let expected_backward_first = C::i() * c(1.5, 0.0) * backward;

        assert_close(sampled.backward().value()[()], backward);

        assert_close(sampled.backward().first()[()], expected_backward_first);
    }

    #[test]
    fn from_right_position_keeps_right_distance_fixed_under_thickness_derivative() {
        let boundaries = LayerBoundaryWaves::new(
            BidirectionalWaves::new(
                jet1(c(2.0, 0.0), c(0.0, 0.0)),
                jet1(c(0.0, 0.0), c(0.0, 0.0)),
            ),
            BidirectionalWaves::new(
                jet1(c(0.0, 0.0), c(0.0, 0.0)),
                jet1(c(3.0, 0.0), c(0.0, 0.0)),
            ),
        );

        let kappa = jet1(c(1.5, 0.0), c(0.0, 0.0));

        let thickness = jet1(c(0.8, 0.0), c(1.0, 0.0));

        let sampled = boundaries.propagate_to_position(
            &kappa,
            &thickness,
            CanonicalLayerPosition::FromRight(0.2),
        );

        /*
         * FromRight(0.2):
         *
         * distance_from_left  = L - 0.2   derivative 1
         * distance_from_right = 0.2       derivative 0
         */
        let distance_from_left = 0.6;

        let forward = c(2.0, 0.0) * (C::i() * c(1.5, 0.0) * distance_from_left).exp();

        let expected_forward_first = C::i() * c(1.5, 0.0) * forward;

        assert_close(sampled.forward().value()[()], forward);

        assert_close(sampled.forward().first()[()], expected_forward_first);

        assert_close(sampled.backward().first()[()], c(0.0, 0.0));
    }

    #[test]
    fn fractional_position_moves_with_layer_thickness() {
        let boundaries = LayerBoundaryWaves::new(
            BidirectionalWaves::new(
                jet1(c(2.0, 0.0), c(0.0, 0.0)),
                jet1(c(0.0, 0.0), c(0.0, 0.0)),
            ),
            BidirectionalWaves::new(
                jet1(c(0.0, 0.0), c(0.0, 0.0)),
                jet1(c(3.0, 0.0), c(0.0, 0.0)),
            ),
        );

        let kappa = jet1(c(1.5, 0.0), c(0.0, 0.0));

        let thickness = jet1(c(0.8, 0.0), c(1.0, 0.0));

        let fraction = 0.25;

        let sampled = boundaries.propagate_to_position(
            &kappa,
            &thickness,
            CanonicalLayerPosition::Fraction(fraction),
        );

        /*
         * z = α L
         *
         * dz/dL = α
         *
         * d(L-z)/dL = 1-α
         */
        let from_left = 0.2;
        let from_right = 0.6;

        let forward = c(2.0, 0.0) * (C::i() * c(1.5, 0.0) * from_left).exp();

        let backward = c(3.0, 0.0) * (C::i() * c(1.5, 0.0) * from_right).exp();

        let expected_forward_first = C::i() * c(1.5, 0.0) * fraction * forward;

        let expected_backward_first = C::i() * c(1.5, 0.0) * (1.0 - fraction) * backward;

        assert_close(sampled.forward().value()[()], forward);

        assert_close(sampled.forward().first()[()], expected_forward_first);

        assert_close(sampled.backward().value()[()], backward);

        assert_close(sampled.backward().first()[()], expected_backward_first);
    }

    #[test]
    fn right_boundary_tracks_full_thickness_derivative() {
        let boundaries = LayerBoundaryWaves::new(
            BidirectionalWaves::new(
                jet1(c(2.0, 0.0), c(0.0, 0.0)),
                jet1(c(0.0, 0.0), c(0.0, 0.0)),
            ),
            BidirectionalWaves::new(
                jet1(c(0.0, 0.0), c(0.0, 0.0)),
                jet1(c(3.0, 0.0), c(0.0, 0.0)),
            ),
        );

        let kappa = jet1(c(1.5, 0.0), c(0.0, 0.0));

        let thickness = jet1(c(0.8, 0.0), c(1.0, 0.0));

        let sampled = boundaries.propagate_to_position(
            &kappa,
            &thickness,
            CanonicalLayerPosition::FromRight(0.0),
        );

        /*
         * At the right boundary:
         *
         * left distance  = L
         * right distance = 0
         */
        let forward = c(2.0, 0.0) * (C::i() * c(1.5, 0.0) * 0.8).exp();

        let expected_forward_first = C::i() * c(1.5, 0.0) * forward;

        assert_close(sampled.forward().value()[()], forward);

        assert_close(sampled.forward().first()[()], expected_forward_first);

        assert_close(sampled.backward().value()[()], c(3.0, 0.0));

        assert_close(sampled.backward().first()[()], c(0.0, 0.0));
    }
}
