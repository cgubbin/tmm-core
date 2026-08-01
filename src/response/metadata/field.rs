use ndarray::{Array1, Dimension};

use crate::input::CoordinateInput;

use super::LayerIndex;

/// Region of a planar stack containing a spatial field sample.
///
/// The left and right exterior regions are geometric regions; they do not
/// depend on the incident side.
///
/// Recording the region separately from the position also disambiguates field
/// limits evaluated exactly at an interface.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum StackRegion {
    /// Semi-infinite medium to the left of the finite stack.
    LeftExterior,

    /// A finite layer.
    Layer(LayerIndex),

    /// Semi-infinite medium to the right of the finite stack.
    RightExterior,
}

/// Metadata for fields and local field-derived observables.
///
/// The excitation input describes the pre-broadcast canonical `k₀` and `k∥`
/// samples. Spatial positions are expressed in centimetres.
///
/// `positions_cm` and `regions` have identical shapes. Corresponding entries
/// jointly identify a spatial sample. The region is required because fields can
/// have different limiting values on opposite sides of an interface.
///
/// `ED` is the dimension of the canonical excitation arrays. `Ix1` is the
/// dimension of the spatial sampling array.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldMetadata<R, ED>
where
    ED: Dimension,
{
    input: CoordinateInput<R, ED>,
    positions_cm: Array1<R>,
    regions: Array1<StackRegion>,
}

impl<R, ED> FieldMetadata<R, ED>
where
    ED: Dimension,
{
    pub(crate) fn new(
        input: CoordinateInput<R, ED>,
        positions_cm: Array1<R>,
        regions: Array1<StackRegion>,
    ) -> Self {
        debug_assert_eq!(positions_cm.len(), regions.len());

        Self {
            input,
            positions_cm,
            regions,
        }
    }

    /// Return the canonical external-excitation input.
    pub fn input(&self) -> &CoordinateInput<R, ED> {
        &self.input
    }

    /// Return the sampled spatial positions in centimetres.
    pub fn positions_cm(&self) -> &Array1<R> {
        &self.positions_cm
    }

    /// Return the stack region associated with each spatial sample.
    pub fn regions(&self) -> &Array1<StackRegion> {
        &self.regions
    }

    /// Consume the metadata and return its components.
    pub fn into_parts(self) -> (CoordinateInput<R, ED>, Array1<R>, Array1<StackRegion>) {
        (self.input, self.positions_cm, self.regions)
    }
}
