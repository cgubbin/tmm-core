//! Canonical states immediately on either side of planar interfaces.

use crate::{
    algebra::ScaleBy,
    observable::{BoundaryState, LayerBoundaries, LayerBoundaryStates},
};

use super::{InterfaceWaveData, Interfaces};

/// Boundary States in the two external media
pub(crate) struct ExteriorBoundaryStates<A> {
    pub(crate) left: BoundaryState<A>,
    pub(crate) right: BoundaryState<A>,
}

/// Canonical states immediately on either side of one planar interface.
///
/// `left` is reconstructed using the medium immediately to the interface's
/// left, while `right` is reconstructed using the medium immediately to its
/// right.
///
/// At an ordinary source-free interface, the canonical state is continuous,
/// so the two values should agree up to numerical error. Both reconstructions
/// are retained because they have different provenance and provide a direct
/// interface-continuity diagnostic.
#[derive(Clone, Debug, PartialEq)]
pub struct InterfaceStates<A> {
    left: BoundaryState<A>,
    right: BoundaryState<A>,
}

impl<A> InterfaceStates<A> {
    pub(crate) const fn new(left: BoundaryState<A>, right: BoundaryState<A>) -> Self {
        Self { left, right }
    }

    /// Return the state reconstructed immediately to the interface's left.
    pub fn left(&self) -> &BoundaryState<A> {
        &self.left
    }

    /// Return the state reconstructed immediately to the interface's right.
    pub fn right(&self) -> &BoundaryState<A> {
        &self.right
    }

    /// Consume the interface and return `(left, right)`.
    pub fn into_parts(self) -> (BoundaryState<A>, BoundaryState<A>) {
        (self.left, self.right)
    }

    /// Transform every scalar component on both interface sides.
    pub fn map<B>(self, mut map: impl FnMut(A) -> B) -> InterfaceStates<B> {
        InterfaceStates {
            left: self.left.map(&mut map),
            right: self.right.map(map),
        }
    }
}

impl<A> ScaleBy<A> for InterfaceStates<A>
where
    BoundaryState<A>: ScaleBy<A>,
{
    fn scale_by(self, scale: &A) -> Self {
        let (left, right) = self.into_parts();

        Self::new(left.scale_by(scale), right.scale_by(scale))
    }
}

impl<A> LayerBoundaries<LayerBoundaryStates<A>> {
    /// Assemble canonical interface states from finite-layer boundary states.
    ///
    /// Inputs must be ordered in physical left-to-right layer order. A stack with
    /// `N` finite layers produces `N + 1` interface records.
    ///
    /// For an empty finite stack, the two exterior states form the sole interface.
    pub(crate) fn into_interface_states(
        self,
        left_exterior: BoundaryState<A>,
        right_exterior: BoundaryState<A>,
    ) -> Interfaces<InterfaceStates<A>> {
        let mut layers = self.into_inner().into_iter();

        let Some(first) = layers.next() else {
            return Interfaces::new(vec![InterfaceStates::new(left_exterior, right_exterior)]);
        };

        let (first_left, first_right) = first.into_parts();

        let mut interfaces = Vec::with_capacity(layers.len() + 2);

        interfaces.push(InterfaceStates::new(left_exterior, first_left));

        let mut previous_right = first_right;

        for layer in layers {
            let (current_left, current_right) = layer.into_parts();

            interfaces.push(InterfaceStates::new(previous_right, current_left));

            previous_right = current_right;
        }

        interfaces.push(InterfaceStates::new(previous_right, right_exterior));

        Interfaces::new(interfaces)
    }
}

impl<A> Interfaces<InterfaceWaveData<A>> {
    /// Convert assembled wave data directly into canonical interface states.
    ///
    /// This consumes the internal wave data and avoids cloning the directional
    /// waves solely to construct their states.
    pub(crate) fn into_states(self) -> Interfaces<InterfaceStates<A>>
    where
        A: crate::algebra::ScalarAlgebra,
        A::Scalar: crate::ComplexScalar,
        A::Dimension: ndarray::Dimension,
    {
        self.map(|interface| {
            let (left, right) = interface.into_parts();

            InterfaceStates::new(left.into_state(), right.into_state())
        })
    }
}
