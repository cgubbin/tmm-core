use ndarray::{Array1, Dimension};

use crate::input::PlaneWaveInput;

/// Index of a finite layer in a planar stack.
///
/// Layer indices exclude the semi-infinite exterior media. Layer zero is the
/// first finite layer after the left exterior medium.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LayerIndex(usize);

impl LayerIndex {
    pub(crate) const fn new(index: usize) -> Self {
        Self(index)
    }

    /// Return the zero-based finite-layer index.
    pub const fn get(self) -> usize {
        self.0
    }
}

impl From<LayerIndex> for usize {
    fn from(index: LayerIndex) -> Self {
        index.get()
    }
}

/// Canonical geometric location of a finite layer.
///
/// Both boundary positions are expressed in centimetres.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerLocation<R> {
    index: LayerIndex,
    left_position_cm: R,
    right_position_cm: R,
}

impl<R> LayerLocation<R> {
    pub(crate) fn new(index: LayerIndex, left_position_cm: R, right_position_cm: R) -> Self {
        Self {
            index,
            left_position_cm,
            right_position_cm,
        }
    }

    /// Return the finite-layer index.
    pub fn index(&self) -> LayerIndex {
        self.index
    }

    /// Return the position of the left layer boundary in centimetres.
    pub fn left_position_cm(&self) -> &R {
        &self.left_position_cm
    }

    /// Return the position of the right layer boundary in centimetres.
    pub fn right_position_cm(&self) -> &R {
        &self.right_position_cm
    }

    /// Consume the location and return its components.
    pub fn into_parts(self) -> (LayerIndex, R, R) {
        (self.index, self.left_position_cm, self.right_position_cm)
    }
}

/// Metadata for an observable resolved over finite stack layers.
///
/// The plane-wave input describes the canonical excitation coordinates. The
/// layer locations describe the final result axis associated with the finite
/// layers.
///
/// Entries in `layers` are ordered in the same way as the corresponding layer
/// axis in the observable.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerMetadata<R, D>
where
    D: Dimension,
{
    input: PlaneWaveInput<R, D>,
    layers: Array1<LayerLocation<R>>,
}

impl<R, D> LayerMetadata<R, D>
where
    D: Dimension,
{
    pub(crate) fn new(input: PlaneWaveInput<R, D>, layers: Array1<LayerLocation<R>>) -> Self {
        Self { input, layers }
    }

    /// Return the canonical external-excitation input.
    pub fn input(&self) -> &PlaneWaveInput<R, D> {
        &self.input
    }

    /// Return the finite layers represented by the observable.
    ///
    /// The order matches the layer axis of the observable.
    pub fn layers(&self) -> &Array1<LayerLocation<R>> {
        &self.layers
    }

    /// Return the number of represented finite layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Return whether no finite layers are represented.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// Consume the metadata and return its components.
    pub fn into_parts(self) -> (PlaneWaveInput<R, D>, Array1<LayerLocation<R>>) {
        (self.input, self.layers)
    }
}
