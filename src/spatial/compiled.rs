use num_traits::{Float, FromPrimitive};

use crate::{FiniteLayerIndex, spatial::sampling::ResolvedLayerPosition};

/// A field-sampling request compiled into backend canonical coordinates.
///
/// Every spatial distance is represented in the backend's canonical spatial
/// coordinate. Region and finite-layer identity are retained so that the field
/// reconstruction backend can apply the appropriate propagation algebra at
/// each position.
///
/// This type is produced from a resolved sampling request and is not exposed as
/// part of the public user-facing sampling API.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CompiledFieldSampling<R> {
    positions: Vec<CanonicalFieldPosition<R>>,
}

impl<R> CompiledFieldSampling<R> {
    pub(crate) fn new(positions: Vec<CanonicalFieldPosition<R>>) -> Self {
        Self { positions }
    }

    /// Returns the canonical sampling positions.
    pub(crate) fn positions(&self) -> &[CanonicalFieldPosition<R>] {
        &self.positions
    }

    /// Returns the number of sampling positions.
    pub(crate) fn len(&self) -> usize {
        self.positions.len()
    }

    /// Returns `true` if no sampling positions are present.
    pub(crate) fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Consumes the compiled request and returns its canonical positions.
    pub(crate) fn into_positions(self) -> Vec<CanonicalFieldPosition<R>> {
        self.positions
    }
}

/// A field-sampling position expressed in the backend's canonical spatial
/// coordinate.
///
/// Distances in the exterior regions are measured away from the adjacent stack
/// boundary according to the convention established by the sampling API.
/// Layer offsets are measured from the corresponding finite layer's left
/// boundary.
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum CanonicalFieldPosition<R> {
    /// A point in the semi-infinite medium to the left of the stack.
    LeftExterior {
        /// Canonical distance from the left stack boundary.
        distance: R,
    },

    /// A point inside a finite layer.
    Layer {
        /// Finite layer containing the point.
        index: FiniteLayerIndex,

        /// Canonical fraction from the layer's left boundary.
        position: CanonicalLayerPosition<R>,
    },

    /// A point in the semi-infinite medium to the right of the stack.
    RightExterior {
        /// Canonical distance from the right stack boundary.
        distance: R,
    },
}

impl<R> From<ResolvedLayerPosition<R>> for CanonicalLayerPosition<R>
where
    R: Float + FromPrimitive,
{
    fn from(value: ResolvedLayerPosition<R>) -> Self {
        match value {
            ResolvedLayerPosition::FromLeft(distance) => {
                let (value, unit) = distance.into_parts();

                CanonicalLayerPosition::FromLeft(value * unit.scale_to_centimetres::<R>())
            }
            ResolvedLayerPosition::FromRight(distance) => {
                let (value, unit) = distance.into_parts();
                CanonicalLayerPosition::FromRight(value * unit.scale_to_centimetres::<R>())
            }
            ResolvedLayerPosition::Fraction(fraction) => CanonicalLayerPosition::Fraction(fraction),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) enum CanonicalLayerPosition<R> {
    /// Fixed distance from the left layer boundary.
    FromLeft(R),

    /// Fixed distance from the right layer boundary.
    FromRight(R),

    /// Fraction of the layer thickness.
    Fraction(R),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exposes_positions() {
        let layer = FiniteLayerIndex::new(1);

        let positions = vec![
            CanonicalFieldPosition::LeftExterior { distance: 1.0 },
            CanonicalFieldPosition::Layer {
                index: layer,
                position: CanonicalLayerPosition::FromLeft(2.0),
            },
            CanonicalFieldPosition::RightExterior { distance: 3.0 },
        ];

        let compiled = CompiledFieldSampling::new(positions.clone());

        assert_eq!(compiled.positions(), positions.as_slice());
    }

    #[test]
    fn reports_length_and_emptiness() {
        let empty = CompiledFieldSampling::<f64>::new(Vec::new());

        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let non_empty = CompiledFieldSampling::new(vec![CanonicalFieldPosition::LeftExterior {
            distance: 1.0,
        }]);

        assert!(!non_empty.is_empty());
        assert_eq!(non_empty.len(), 1);
    }

    #[test]
    fn into_positions_recovers_positions() {
        let positions = vec![
            CanonicalFieldPosition::LeftExterior { distance: 1.0 },
            CanonicalFieldPosition::RightExterior { distance: 2.0 },
        ];

        let compiled = CompiledFieldSampling::new(positions.clone());

        assert_eq!(compiled.into_positions(), positions);
    }
}
