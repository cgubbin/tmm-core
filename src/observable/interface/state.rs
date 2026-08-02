//! Canonical states immediately on either side of planar interfaces.

use crate::observable::BoundaryState;

/// Canonical states immediately on either side of one planar interface.
///
/// `left` is reconstructed in the basis of the medium immediately to the
/// interface's left, while `right` is reconstructed in the basis of the
/// medium immediately to its right.
///
/// For an ordinary source-free interface, the physical state is continuous,
/// so the two values should agree up to numerical error. Both are retained to
/// preserve the provenance of each reconstruction and to support interface
/// diagnostics.
#[derive(Clone, Debug, PartialEq)]
pub struct InterfaceStates<A> {
    left: BoundaryState<A>,
    right: BoundaryState<A>,
}

impl<A> InterfaceStates<A> {
    pub(crate) const fn new(left: BoundaryState<A>, right: BoundaryState<A>) -> Self {
        Self { left, right }
    }

    /// Return the state immediately to the interface's left.
    pub fn left(&self) -> &BoundaryState<A> {
        &self.left
    }

    /// Return the state immediately to the interface's right.
    pub fn right(&self) -> &BoundaryState<A> {
        &self.right
    }

    /// Consume the interface and return `(left, right)`.
    pub fn into_parts(self) -> (BoundaryState<A>, BoundaryState<A>) {
        (self.left, self.right)
    }

    /// Transform both interface-side states component-wise.
    pub fn map<B>(self, mut map: impl FnMut(A) -> B) -> InterfaceStates<B> {
        InterfaceStates {
            left: self.left.map(&mut map),
            right: self.right.map(map),
        }
    }
}
