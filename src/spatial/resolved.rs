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
    pub(crate) fn compile(&self) -> CompiledFieldSampling<R>
    where
        R: Float + FromPrimitive,
    {
        let positions = self
            .positions
            .iter()
            .map(|position| match position {
                FieldPosition::LeftExterior { distance } => CanonicalFieldPosition::LeftExterior {
                    distance: distance.into_canonical(),
                },
                FieldPosition::Layer { index, position } => CanonicalFieldPosition::Layer {
                    index: *index,
                    position: position.clone().into(),
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

    use crate::{
        FiniteLayerIndex,
        spatial::{CanonicalLayerPosition, Length, ResolvedLayerPosition},
    };

    #[test]
    fn exposes_positions() {
        let positions = vec![
            FieldPosition::LeftExterior {
                distance: Length::nanometres(10.0),
            },
            FieldPosition::Layer {
                index: FiniteLayerIndex::new(1),
                position: ResolvedLayerPosition::FromLeft(Length::nanometres(20.0)),
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
            FieldPosition::Layer {
                index: FiniteLayerIndex::new(0),
                position: ResolvedLayerPosition::FromRight(Length::micrometres(2.0)),
            },
            FieldPosition::RightExterior {
                distance: Length::micrometres(3.0),
            },
        ];

        let resolved = ResolvedFieldSampling::new(positions.clone());

        assert_eq!(resolved.into_positions(), positions);
    }

    fn assert_canonical_layer_position_close(
        actual: &CanonicalLayerPosition<f64>,
        expected: &CanonicalLayerPosition<f64>,
    ) {
        match (actual, expected) {
            (
                CanonicalLayerPosition::FromLeft(actual),
                CanonicalLayerPosition::FromLeft(expected),
            )
            | (
                CanonicalLayerPosition::FromRight(actual),
                CanonicalLayerPosition::FromRight(expected),
            )
            | (
                CanonicalLayerPosition::Fraction(actual),
                CanonicalLayerPosition::Fraction(expected),
            ) => {
                approx::assert_relative_eq!(actual, expected);
            }

            _ => panic!("layer-position variants did not match"),
        }
    }

    fn assert_canonical_field_position_close(
        actual: &CanonicalFieldPosition<f64>,
        expected: &CanonicalFieldPosition<f64>,
    ) {
        match (actual, expected) {
            (
                CanonicalFieldPosition::LeftExterior { distance: actual },
                CanonicalFieldPosition::LeftExterior { distance: expected },
            )
            | (
                CanonicalFieldPosition::RightExterior { distance: actual },
                CanonicalFieldPosition::RightExterior { distance: expected },
            ) => {
                approx::assert_relative_eq!(actual, expected);
            }

            (
                CanonicalFieldPosition::Layer {
                    index: actual_index,
                    position: actual_position,
                },
                CanonicalFieldPosition::Layer {
                    index: expected_index,
                    position: expected_position,
                },
            ) => {
                assert_eq!(actual_index, expected_index);

                assert_canonical_layer_position_close(actual_position, expected_position);
            }

            _ => panic!("field-position variants did not match"),
        }
    }

    #[test]
    fn compile_preserves_regions_and_converts_distances() {
        let layer0 = FiniteLayerIndex::new(0);
        let layer1 = FiniteLayerIndex::new(1);
        let layer2 = FiniteLayerIndex::new(2);

        let resolved = ResolvedFieldSampling::new(vec![
            FieldPosition::LeftExterior {
                distance: Length::nanometres(100.0),
            },
            FieldPosition::Layer {
                index: layer0,
                position: ResolvedLayerPosition::FromLeft(Length::micrometres(2.0)),
            },
            FieldPosition::Layer {
                index: layer1,
                position: ResolvedLayerPosition::FromRight(Length::millimetres(3.0)),
            },
            FieldPosition::Layer {
                index: layer2,
                position: ResolvedLayerPosition::Fraction(0.25),
            },
            FieldPosition::RightExterior {
                distance: Length::centimetres(3.0),
            },
        ]);

        let compiled = resolved.compile();

        let expected = [
            CanonicalFieldPosition::LeftExterior { distance: 1.0e-5 },
            CanonicalFieldPosition::Layer {
                index: layer0,
                position: CanonicalLayerPosition::FromLeft(2.0e-4),
            },
            CanonicalFieldPosition::Layer {
                index: layer1,
                position: CanonicalLayerPosition::FromRight(0.3),
            },
            CanonicalFieldPosition::Layer {
                index: layer2,
                position: CanonicalLayerPosition::Fraction(0.25),
            },
            CanonicalFieldPosition::RightExterior { distance: 3.0 },
        ];

        assert_eq!(compiled.positions().len(), expected.len());

        for (actual, expected) in compiled.positions().iter().zip(expected.iter()) {
            assert_canonical_field_position_close(actual, expected);
        }
    }

    #[test]
    fn compile_preserves_layer_position_semantics() {
        let layer = FiniteLayerIndex::new(0);

        let resolved = ResolvedFieldSampling::new(vec![
            FieldPosition::Layer {
                index: layer,
                position: ResolvedLayerPosition::FromLeft(Length::nanometres(100.0)),
            },
            FieldPosition::Layer {
                index: layer,
                position: ResolvedLayerPosition::FromRight(Length::nanometres(100.0)),
            },
            FieldPosition::Layer {
                index: layer,
                position: ResolvedLayerPosition::Fraction(0.5),
            },
        ]);

        let compiled = resolved.compile();

        let expected = [
            CanonicalFieldPosition::Layer {
                index: layer,
                position: CanonicalLayerPosition::FromLeft(1.0e-5),
            },
            CanonicalFieldPosition::Layer {
                index: layer,
                position: CanonicalLayerPosition::FromRight(1.0e-5),
            },
            CanonicalFieldPosition::Layer {
                index: layer,
                position: CanonicalLayerPosition::Fraction(0.5),
            },
        ];

        for (actual, expected) in compiled.positions().iter().zip(expected.iter()) {
            assert_canonical_field_position_close(actual, expected);
        }
    }
}
