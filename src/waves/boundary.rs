//! Backend-independent wave-amplitude containers.
//!
//! These types describe forward and backward wave amplitudes at exterior and
//! finite-layer boundaries. They contain no backend-specific reconstruction
//! logic: transfer- and scattering-matrix backends may construct them using
//! different algorithms.

use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, OwnedRepr};
use num_traits::{One, Zero};

use crate::{ComplexScalar, IncidentSide, algebra::ScalarAlgebra};

#[derive(Clone, Debug)]
pub(crate) struct BoundaryWaveSolution<A> {
    exterior: ExteriorBoundaryWaves<A>,
    layers: Vec<LayerBoundaryWaves<A>>,
}

impl<A> BoundaryWaveSolution<A> {
    pub(crate) fn new(
        exterior: ExteriorBoundaryWaves<A>,
        layers: Vec<LayerBoundaryWaves<A>>,
    ) -> Self {
        Self { exterior, layers }
    }

    pub(crate) fn exterior(&self) -> &ExteriorBoundaryWaves<A> {
        &self.exterior
    }

    pub(crate) fn layers(&self) -> &[LayerBoundaryWaves<A>] {
        &self.layers
    }

    pub(crate) fn into_parts(self) -> (ExteriorBoundaryWaves<A>, Vec<LayerBoundaryWaves<A>>) {
        (self.exterior, self.layers)
    }
}

/// Forward- and backward-propagating wave amplitudes at one longitudinal
/// position.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BidirectionalWaves<A> {
    forward: A,
    backward: A,
}

impl<A> BidirectionalWaves<A> {
    pub(crate) fn new(forward: A, backward: A) -> Self {
        Self { forward, backward }
    }

    pub(crate) fn forward(&self) -> &A {
        &self.forward
    }

    pub(crate) fn backward(&self) -> &A {
        &self.backward
    }

    pub(crate) fn into_parts(self) -> (A, A) {
        (self.forward, self.backward)
    }
}

impl<C, D> BidirectionalWaves<ArrayBase<OwnedRepr<C>, D>>
where
    C: ComplexField,
    D: Dimension,
{
    /// Scale both propagation directions pointwise by the same factor.
    pub(crate) fn scale(self, factor: ArrayBase<OwnedRepr<C>, D>) -> Self {
        Self {
            forward: self.forward * factor.clone(),
            backward: self.backward * factor,
        }
    }
}

/// Bidirectional wave amplitudes at the left and right boundaries of one
/// finite layer.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerBoundaryWaves<A> {
    left: BidirectionalWaves<A>,
    right: BidirectionalWaves<A>,
}

impl<A> LayerBoundaryWaves<A> {
    pub(crate) fn new(left: BidirectionalWaves<A>, right: BidirectionalWaves<A>) -> Self {
        Self { left, right }
    }

    pub(crate) fn left(&self) -> &BidirectionalWaves<A> {
        &self.left
    }

    pub(crate) fn right(&self) -> &BidirectionalWaves<A> {
        &self.right
    }

    pub(crate) fn into_parts(self) -> (BidirectionalWaves<A>, BidirectionalWaves<A>) {
        (self.left, self.right)
    }
}

impl<C, D> LayerBoundaryWaves<ArrayBase<OwnedRepr<C>, D>>
where
    C: ComplexField,
    D: Dimension,
{
    /// Scale all wave amplitudes at both layer boundaries pointwise.
    pub(crate) fn scale(self, factor: ArrayBase<OwnedRepr<C>, D>) -> Self {
        Self {
            left: self.left.scale(factor.clone()),
            right: self.right.scale(factor),
        }
    }
}

/// Bidirectional wave amplitudes at the left and right exterior boundaries of
/// a complete stack.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExteriorBoundaryWaves<A> {
    left: BidirectionalWaves<A>,
    right: BidirectionalWaves<A>,
}

impl<A> ExteriorBoundaryWaves<A> {
    pub(crate) fn new(left: BidirectionalWaves<A>, right: BidirectionalWaves<A>) -> Self {
        Self { left, right }
    }

    pub(crate) fn left(&self) -> &BidirectionalWaves<A> {
        &self.left
    }

    pub(crate) fn right(&self) -> &BidirectionalWaves<A> {
        &self.right
    }

    pub(crate) fn into_parts(self) -> (BidirectionalWaves<A>, BidirectionalWaves<A>) {
        (self.left, self.right)
    }
}

impl<A> ExteriorBoundaryWaves<A>
where
    A: ScalarAlgebra + Clone,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    pub(crate) fn from_amplitudes(
        reflection: &A,
        transmission: &A,
        incident_side: IncidentSide,
    ) -> Self {
        let zero = A::filled_constant_like(reflection.value(), <A::Scalar as Zero>::zero());

        let one = A::filled_constant_like(reflection.value(), <A::Scalar as One>::one());

        match incident_side {
            IncidentSide::Left => Self::new(
                BidirectionalWaves::new(one, reflection.clone()),
                BidirectionalWaves::new(transmission.clone(), zero),
            ),

            IncidentSide::Right => Self::new(
                BidirectionalWaves::new(zero, transmission.clone()),
                BidirectionalWaves::new(reflection.clone(), one),
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr0, array};

    use super::{BidirectionalWaves, ExteriorBoundaryWaves, LayerBoundaryWaves};

    use crate::test_support::{C, TOLERANCE, assertions::assert_array_close, c};

    #[test]
    fn bidirectional_waves_store_both_directions() {
        let waves = BidirectionalWaves::new(arr0(c(2.0)), arr0(c(3.0)));

        assert_eq!(waves.forward()[()], c(2.0));
        assert_eq!(waves.backward()[()], c(3.0));
    }

    #[test]
    fn bidirectional_waves_into_parts_returns_both_directions() {
        let waves = BidirectionalWaves::new(arr0(c(2.0)), arr0(c(3.0)));

        let (forward, backward) = waves.into_parts();

        assert_eq!(forward[()], c(2.0));
        assert_eq!(backward[()], c(3.0));
    }

    #[test]
    fn bidirectional_waves_scale_both_directions() {
        let waves = BidirectionalWaves::new(array![c(1.0), c(2.0)], array![c(3.0), c(4.0)]);

        let factor = array![C::new(2.0, 1.0), C::new(-1.0, 0.5),];

        let scaled = waves.scale(factor.clone());

        assert_array_close(
            scaled.forward(),
            &(array![c(1.0), c(2.0)] * factor.clone()),
            TOLERANCE,
        );

        assert_array_close(
            scaled.backward(),
            &(array![c(3.0), c(4.0)] * factor),
            TOLERANCE,
        );
    }

    #[test]
    fn bidirectional_waves_scale_preserves_shape() {
        let waves = BidirectionalWaves::new(
            array![c(1.0), c(2.0), c(3.0)],
            array![c(4.0), c(5.0), c(6.0)],
        );

        let expected_shape = waves.forward().raw_dim();

        let scaled = waves.scale(array![c(2.0), c(3.0), c(4.0),]);

        assert_eq!(scaled.forward().raw_dim(), expected_shape);
        assert_eq!(scaled.backward().raw_dim(), expected_shape);
    }

    #[test]
    fn layer_boundary_waves_store_both_boundaries() {
        let left = BidirectionalWaves::new(arr0(c(1.0)), arr0(c(2.0)));

        let right = BidirectionalWaves::new(arr0(c(3.0)), arr0(c(4.0)));

        let boundaries = LayerBoundaryWaves::new(left, right);

        assert_eq!(boundaries.left().forward()[()], c(1.0));
        assert_eq!(boundaries.left().backward()[()], c(2.0));
        assert_eq!(boundaries.right().forward()[()], c(3.0));
        assert_eq!(boundaries.right().backward()[()], c(4.0));
    }

    #[test]
    fn layer_boundary_waves_into_parts_returns_both_boundaries() {
        let left = BidirectionalWaves::new(arr0(c(1.0)), arr0(c(2.0)));

        let right = BidirectionalWaves::new(arr0(c(3.0)), arr0(c(4.0)));

        let boundaries = LayerBoundaryWaves::new(left, right);

        let (left, right) = boundaries.into_parts();

        assert_eq!(left.forward()[()], c(1.0));
        assert_eq!(left.backward()[()], c(2.0));
        assert_eq!(right.forward()[()], c(3.0));
        assert_eq!(right.backward()[()], c(4.0));
    }

    #[test]
    fn layer_boundary_waves_scale_all_amplitudes() {
        let left = BidirectionalWaves::new(array![c(1.0), c(2.0)], array![c(3.0), c(4.0)]);

        let right = BidirectionalWaves::new(array![c(5.0), c(6.0)], array![c(7.0), c(8.0)]);

        let factor = array![C::new(2.0, 1.0), C::new(-1.0, 0.5),];

        let boundaries = LayerBoundaryWaves::new(left, right).scale(factor.clone());

        assert_array_close(
            boundaries.left().forward(),
            &(array![c(1.0), c(2.0)] * factor.clone()),
            TOLERANCE,
        );

        assert_array_close(
            boundaries.left().backward(),
            &(array![c(3.0), c(4.0)] * factor.clone()),
            TOLERANCE,
        );

        assert_array_close(
            boundaries.right().forward(),
            &(array![c(5.0), c(6.0)] * factor.clone()),
            TOLERANCE,
        );

        assert_array_close(
            boundaries.right().backward(),
            &(array![c(7.0), c(8.0)] * factor),
            TOLERANCE,
        );
    }

    #[test]
    fn exterior_boundary_waves_store_both_boundaries() {
        let left = BidirectionalWaves::new(arr0(c(1.0)), arr0(c(2.0)));

        let right = BidirectionalWaves::new(arr0(c(3.0)), arr0(c(4.0)));

        let boundaries = ExteriorBoundaryWaves::new(left, right);

        assert_eq!(boundaries.left().forward()[()], c(1.0));
        assert_eq!(boundaries.left().backward()[()], c(2.0));
        assert_eq!(boundaries.right().forward()[()], c(3.0));
        assert_eq!(boundaries.right().backward()[()], c(4.0));
    }

    #[test]
    fn exterior_boundary_waves_into_parts_returns_both_boundaries() {
        let left = BidirectionalWaves::new(arr0(c(1.0)), arr0(c(2.0)));

        let right = BidirectionalWaves::new(arr0(c(3.0)), arr0(c(4.0)));

        let boundaries = ExteriorBoundaryWaves::new(left, right);

        let (left, right) = boundaries.into_parts();

        assert_eq!(left.forward()[()], c(1.0));
        assert_eq!(left.backward()[()], c(2.0));
        assert_eq!(right.forward()[()], c(3.0));
        assert_eq!(right.backward()[()], c(4.0));
    }
}
