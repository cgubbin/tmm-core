use num_traits::{Float, FromPrimitive};

use crate::spatial::{CanonicalFieldPosition, CompiledFieldSampling};

use super::FieldPosition;

/// A field-sampling request resolved against a particular physical stack.
///
/// Unlike [`CompiledFieldSampling`], this representation retains the
/// caller-facing spatial units used by the original request. Each position has
/// also been resolved to an unambiguous region of the stack:
///
/// - the left exterior,
/// - a particular finite layer, or
/// - the right exterior.
///
/// Resolution is responsible for validating the requested positions against
/// the stack geometry. Compilation subsequently converts their distances into
/// the backend's canonical spatial coordinate.
#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedFieldSampling<R> {
    positions: Vec<FieldPosition<R>>,
}

impl<R> ResolvedFieldSampling<R> {
    pub(crate) fn new(positions: Vec<FieldPosition<R>>) -> Self {
        Self { positions }
    }

    /// Returns the resolved field-sampling positions.
    pub fn positions(&self) -> &[FieldPosition<R>] {
        &self.positions
    }

    /// Returns the number of requested sampling positions.
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    /// Returns `true` if no sampling positions were requested.
    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Consumes the sampling request and returns its resolved positions.
    pub fn into_positions(self) -> Vec<FieldPosition<R>> {
        self.positions
    }

    /// Converts the resolved sampling request to backend canonical coordinates.
    ///
    /// Region and finite-layer identity are preserved. Only the spatial
    /// distance or offset associated with each position is converted.
    pub(crate) fn compile(self) -> CompiledFieldSampling<R>
    where
        R: Float + FromPrimitive,
    {
        let positions = self
            .positions
            .into_iter()
            .map(|position| match position {
                FieldPosition::LeftExterior { distance } => CanonicalFieldPosition::LeftExterior {
                    distance: distance.into_canonical(),
                },
                FieldPosition::Layer { index, offset } => CanonicalFieldPosition::Layer {
                    index,
                    offset: offset.into_canonical(),
                },
                FieldPosition::RightExterior { distance } => {
                    CanonicalFieldPosition::RightExterior {
                        distance: distance.into_canonical(),
                    }
                }
            })
            .collect();

        CompiledFieldSampling::new(positions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Adjust these imports to the precise location/names used by sampling.rs.
    use crate::FiniteLayerIndex;
    use crate::spatial::Length;

    #[test]
    fn exposes_positions() {
        let positions = vec![
            FieldPosition::LeftExterior {
                distance: Length::nanometres(10.0),
            },
            FieldPosition::Layer {
                index: FiniteLayerIndex::new(1),
                offset: Length::nanometres(20.0),
            },
        ];

        let resolved = ResolvedFieldSampling::new(positions.clone());

        assert_eq!(resolved.positions(), positions.as_slice());
    }

    #[test]
    fn reports_length_and_emptiness() {
        let empty = ResolvedFieldSampling::<f64>::new(Vec::new());

        assert!(empty.is_empty());
        assert_eq!(empty.len(), 0);

        let non_empty = ResolvedFieldSampling::new(vec![FieldPosition::RightExterior {
            distance: Length::nanometres(10.0),
        }]);

        assert!(!non_empty.is_empty());
        assert_eq!(non_empty.len(), 1);
    }

    #[test]
    fn into_positions_recovers_positions() {
        let positions = vec![
            FieldPosition::LeftExterior {
                distance: Length::nanometres(10.0),
            },
            FieldPosition::RightExterior {
                distance: Length::micrometres(2.0),
            },
        ];

        let resolved = ResolvedFieldSampling::new(positions.clone());

        assert_eq!(resolved.into_positions(), positions);
    }

    fn assert_canonical_field_position_close(
        actual: &CanonicalFieldPosition<f64>,
        expected: &CanonicalFieldPosition<f64>,
    ) {
        match (actual, expected) {
            (
                CanonicalFieldPosition::LeftExterior { distance: first },
                CanonicalFieldPosition::LeftExterior { distance: second },
            ) => approx::assert_relative_eq!(first, second),
            (
                CanonicalFieldPosition::RightExterior { distance: first },
                CanonicalFieldPosition::RightExterior { distance: second },
            ) => approx::assert_relative_eq!(first, second),
            (
                CanonicalFieldPosition::Layer {
                    offset: first_offset,
                    index: first_index,
                },
                CanonicalFieldPosition::Layer {
                    offset: second_offset,
                    index: second_index,
                },
            ) => {
                assert_eq!(first_index, second_index);
                approx::assert_relative_eq!(first_offset, second_offset);
            }

            _ => panic!("enum variants did not match"),
        }
    }

    #[test]
    fn compile_preserves_regions_and_converts_distances() {
        let layer = FiniteLayerIndex::new(2);

        let resolved = ResolvedFieldSampling::new(vec![
            FieldPosition::LeftExterior {
                distance: Length::nanometres(100.0),
            },
            FieldPosition::Layer {
                index: layer,
                offset: Length::micrometres(2.0),
            },
            FieldPosition::RightExterior {
                distance: Length::centimetres(3.0),
            },
        ]);

        let compiled = resolved.compile();

        for (actual, expected) in compiled.positions().iter().zip(&[
            CanonicalFieldPosition::LeftExterior { distance: 1.0e-5 },
            CanonicalFieldPosition::Layer {
                index: layer,
                offset: 2.0e-4,
            },
            CanonicalFieldPosition::RightExterior { distance: 3.0 },
        ]) {
            assert_canonical_field_position_close(actual, expected);
        }
    }
}
