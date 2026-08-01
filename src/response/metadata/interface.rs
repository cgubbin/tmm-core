use ndarray::{Array1, Dimension};

use crate::input::CoordinateInput;

/// Index of an interface in a planar stack.
///
/// For a stack containing `N` finite layers, there are `N + 1` interfaces:
///
/// - interface zero separates the left exterior from finite layer zero;
/// - interface `N` separates finite layer `N - 1` from the right exterior.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct InterfaceIndex(usize);

impl InterfaceIndex {
    pub(crate) const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Return the zero-based interface index.
    pub const fn get(self) -> usize {
        self.0
    }
}

impl From<InterfaceIndex> for usize {
    fn from(index: InterfaceIndex) -> Self {
        index.get()
    }
}

/// Canonical geometric location of a stack interface.
///
/// The position is expressed in centimetres.
#[derive(Clone, Debug, PartialEq)]
pub struct InterfaceLocation<R> {
    index: InterfaceIndex,
    position_cm: R,
}

impl<R> InterfaceLocation<R> {
    pub(crate) fn new(index: InterfaceIndex, position_cm: R) -> Self {
        Self { index, position_cm }
    }

    /// Return the interface index.
    pub fn index(&self) -> InterfaceIndex {
        self.index
    }

    /// Return the interface position in centimetres.
    pub fn position_cm(&self) -> &R {
        &self.position_cm
    }

    /// Consume the location and return its components.
    pub fn into_parts(self) -> (InterfaceIndex, R) {
        (self.index, self.position_cm)
    }
}

/// Metadata for an observable resolved over stack interfaces.
///
/// The plane-wave input describes the canonical excitation coordinates. The
/// interface locations describe the final result axis associated with stack
/// interfaces.
///
/// Entries in `interfaces` are ordered in the same way as the corresponding
/// interface axis in the observable.
#[derive(Clone, Debug, PartialEq)]
pub struct InterfaceMetadata<R, D>
where
    D: Dimension,
{
    input: CoordinateInput<R, D>,
    interfaces: Array1<InterfaceLocation<R>>,
}

impl<R, D> InterfaceMetadata<R, D>
where
    D: Dimension,
{
    pub(crate) fn new(
        input: CoordinateInput<R, D>,
        interfaces: Array1<InterfaceLocation<R>>,
    ) -> Self {
        Self { input, interfaces }
    }

    /// Return the canonical external-excitation input.
    pub fn input(&self) -> &CoordinateInput<R, D> {
        &self.input
    }

    /// Return the interfaces represented by the observable.
    ///
    /// The order matches the interface axis of the observable.
    pub fn interfaces(&self) -> &Array1<InterfaceLocation<R>> {
        &self.interfaces
    }

    /// Return the number of represented interfaces.
    pub fn len(&self) -> usize {
        self.interfaces.len()
    }

    /// Return whether no interfaces are represented.
    pub fn is_empty(&self) -> bool {
        self.interfaces.is_empty()
    }

    /// Consume the metadata and return its components.
    pub fn into_parts(self) -> (CoordinateInput<R, D>, Array1<InterfaceLocation<R>>) {
        (self.input, self.interfaces)
    }
}
