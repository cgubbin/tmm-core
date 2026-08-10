//! Canonical isotropic states at planar boundaries.
//!
//! The canonical state contains the two continuity variables used by the
//! isotropic transfer and scattering formulations. Unlike directional-wave
//! amplitudes, the state has direct interface-continuity semantics.
//!
//! The state is algebraic and holomorphic. Hermitian operations required for
//! real-frequency power, dissipation, and energy are applied only in later
//! observable projections.

use crate::algebra::{ScalarAlgebra, ScaleBy};

/// Canonical isotropic state at one planar boundary.
///
/// The state stores two algebraically paired components:
///
/// ```text
/// field     = forward + backward
/// secondary = ξ (backward - forward)
/// ξ         = -i Y
/// ```
///
/// This is equivalent to
///
/// ```text
/// secondary = factor⁻¹ ∂z field,
/// ```
///
/// where `factor = μ` for TE and `factor = ε` for TM.
///
/// `secondary` is not a complex conjugate of `field`. Both components remain
/// holomorphic functions of complex coordinates whenever the underlying
/// material and wave quantities are holomorphic.
///
/// Physical real-frequency quantities such as power flux are obtained later
/// through the appropriate Hermitian projection.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryState<A> {
    field: A,
    secondary: A,
}

impl<A> BoundaryState<A> {
    pub(crate) const fn new(field: A, secondary: A) -> Self {
        Self { field, secondary }
    }

    /// Return canonical field component
    pub fn field(&self) -> &A {
        &self.field
    }

    /// Return canonical secondary component
    pub fn secondary(&self) -> &A {
        &self.secondary
    }

    /// Consume the state and return
    pub fn into_parts(self) -> (A, A) {
        (self.field, self.secondary)
    }

    pub fn scaled(mut self, factor: &A) -> Self
    where
        A: ScalarAlgebra,
    {
        self.field = self.field.multiply(factor);
        self.secondary = self.secondary.multiply(factor);

        self
    }

    /// Transform both canonical components
    pub fn map<B>(self, mut map: impl FnMut(A) -> B) -> BoundaryState<B> {
        BoundaryState {
            field: map(self.field),
            secondary: map(self.secondary),
        }
    }
}

impl<A> ScaleBy<A> for BoundaryState<A>
where
    A: ScalarAlgebra,
{
    fn scale_by(self, scale: &A) -> Self {
        let (field, secondary) = self.into_parts();

        Self::new(field.multiply(scale), secondary.multiply(scale))
    }
}

/// Canonical isotropic states at both boundaries of one finite layer.
///
/// Both states are reconstructed from directional waves expressed in the same
/// finite-layer basis and use the same characteristic admittance.
///
/// The two states may be compared to study propagation across the layer.
/// Interface continuity instead compares the right state of one medium with
/// the left state independently reconstructed in the adjacent medium.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerBoundaryStates<A> {
    left: BoundaryState<A>,
    right: BoundaryState<A>,
}

impl<A> LayerBoundaryStates<A> {
    pub(crate) const fn new(left: BoundaryState<A>, right: BoundaryState<A>) -> Self {
        Self { left, right }
    }

    pub fn left(&self) -> &BoundaryState<A> {
        &self.left
    }

    pub fn right(&self) -> &BoundaryState<A> {
        &self.right
    }

    pub fn into_parts(self) -> (BoundaryState<A>, BoundaryState<A>) {
        (self.left, self.right)
    }

    pub fn map<B>(self, mut map: impl FnMut(A) -> B) -> LayerBoundaryStates<B> {
        LayerBoundaryStates {
            left: self.left.map(&mut map),
            right: self.right.map(map),
        }
    }
}

impl<A> ScaleBy<A> for LayerBoundaryStates<A>
where
    BoundaryState<A>: ScaleBy<A>,
{
    fn scale_by(self, scale: &A) -> Self {
        let (left, right) = self.into_parts();

        Self::new(left.scale_by(scale), right.scale_by(scale))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boundary_state_stores_both_components() {
        let state = BoundaryState::new(1, 2);

        assert_eq!(state.field(), &1);
        assert_eq!(state.secondary(), &2);
    }

    #[test]
    fn boundary_state_into_parts_preserves_component_order() {
        let state = BoundaryState::new("field", "secondary");

        assert_eq!(state.into_parts(), ("field", "secondary"),);
    }

    #[test]
    fn boundary_state_map_transforms_both_components() {
        let state = BoundaryState::new(2, 3);

        let mapped = state.map(|value| value * 10);

        assert_eq!(mapped.field(), &20);
        assert_eq!(mapped.secondary(), &30);
    }

    #[test]
    fn layer_boundary_states_preserve_left_right_order() {
        let states = LayerBoundaryStates::new(BoundaryState::new(1, 2), BoundaryState::new(3, 4));

        assert_eq!(states.left().field(), &1);
        assert_eq!(states.left().secondary(), &2);
        assert_eq!(states.right().field(), &3);
        assert_eq!(states.right().secondary(), &4);
    }

    #[test]
    fn layer_boundary_states_into_parts_preserves_order() {
        let states = LayerBoundaryStates::new(BoundaryState::new(1, 2), BoundaryState::new(3, 4));

        let (left, right) = states.into_parts();

        assert_eq!(left.into_parts(), (1, 2));
        assert_eq!(right.into_parts(), (3, 4));
    }

    #[test]
    fn layer_boundary_states_map_transforms_every_component() {
        let states = LayerBoundaryStates::new(BoundaryState::new(1, 2), BoundaryState::new(3, 4));

        let mapped = states.map(|value| value * 2);

        assert_eq!(mapped.left().field(), &2);
        assert_eq!(mapped.left().secondary(), &4);
        assert_eq!(mapped.right().field(), &6);
        assert_eq!(mapped.right().secondary(), &8);
    }
}
