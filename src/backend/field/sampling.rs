use num_traits::{Float, Zero};

use crate::Stack;

use super::PlaneWaveFieldError;

/// Position at which a plane-wave field is sampled.
///
/// All distances use centimetres, matching the canonical stack thickness and
/// spectral-coordinate convention.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum FieldPosition<R> {
    /// Point in the left exterior.
    ///
    /// `distance` is non-negative and is measured leftward from the stack's
    /// left boundary.
    LeftExterior { distance: R },

    /// Point inside a finite layer.
    ///
    /// `offset` is measured rightward from the layer's left boundary and must
    /// satisfy:
    ///
    /// ```text
    /// 0 <= offset <= layer thickness.
    /// ```
    Layer { index: usize, offset: R },

    /// Point in the right exterior.
    ///
    /// `distance` is non-negative and is measured rightward from the stack's
    /// right boundary.
    RightExterior { distance: R },
}

impl<R> FieldPosition<R>
where
    R: Zero,
{
    pub fn left_boundary() -> Self {
        Self::LeftExterior {
            distance: R::zero(),
        }
    }

    pub fn right_boundary() -> Self {
        Self::RightExterior {
            distance: R::zero(),
        }
    }

    pub fn layer_left(index: usize) -> Self {
        Self::Layer {
            index,
            offset: R::zero(),
        }
    }

    pub fn layer_right(index: usize, thickness: R) -> Self {
        Self::Layer {
            index,
            offset: thickness,
        }
    }

    pub fn layer_centre(index: usize, thickness: R) -> Self
    where
        R: Float,
    {
        Self::Layer {
            index,
            offset: thickness / (R::one() + R::one()),
        }
    }
}

/// High-level specification for sampling fields across a planar stack.
///
/// Requests are expanded in insertion order. This allows callers to construct
/// profiles containing exterior regions, selected finite layers, interfaces,
/// and layer centres.
#[derive(Clone, Debug, PartialEq)]
pub struct FieldSampling<R> {
    regions: Vec<FieldSamplingRegion<R>>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum FieldSamplingRegion<R> {
    LeftExterior(ExteriorSampling<R>),

    Layer {
        index: usize,
        sampling: LayerSampling<R>,
    },

    RightExterior(ExteriorSampling<R>),

    LayerCentres,

    LayerInterfaces,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LayerSampling<R> {
    /// One point.
    Point(R),

    /// Explicit offsets.
    Offsets(Vec<R>),

    /// Uniform spacing.
    Uniform {
        points: usize,
        include_left: bool,
        include_right: bool,
    },
}

impl<R> LayerSampling<R> {
    pub fn point(point: R) -> Self {
        Self::Point(point)
    }

    pub fn offsets(offsets: Vec<R>) -> Self {
        Self::Offsets(offsets)
    }

    pub fn uniform(points: usize) -> Self {
        Self::Uniform {
            points,
            include_left: true,
            include_right: false,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum ExteriorSampling<R> {
    Point(R),

    Distances(Vec<R>),

    Uniform { points: usize, distance: R },
}

impl<R> Default for FieldSampling<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R> ExteriorSampling<R> {
    pub fn point(point: R) -> Self {
        Self::Point(point)
    }

    pub fn distances(distances: Vec<R>) -> Self {
        Self::Distances(distances)
    }

    pub fn uniform(points: usize, distance: R) -> Self {
        Self::Uniform { points, distance }
    }
}

impl<R> FieldSampling<R> {
    /// Create an empty sampling specification.
    pub fn new() -> Self {
        Self {
            regions: Vec::new(),
        }
    }

    /// Append an explicit left-exterior sampling request.
    pub fn left_exterior(mut self, sampling: ExteriorSampling<R>) -> Self {
        self.regions
            .push(FieldSamplingRegion::LeftExterior(sampling));

        self
    }

    /// Append an explicit finite-layer sampling request.
    pub fn layer(mut self, index: usize, sampling: LayerSampling<R>) -> Self {
        self.regions
            .push(FieldSamplingRegion::Layer { index, sampling });

        self
    }

    /// Append an explicit right-exterior sampling request.
    pub fn right_exterior(mut self, sampling: ExteriorSampling<R>) -> Self {
        self.regions
            .push(FieldSamplingRegion::RightExterior(sampling));

        self
    }

    /// Append the centre of every finite layer.
    pub fn layer_centres(mut self) -> Self {
        self.regions.push(FieldSamplingRegion::LayerCentres);

        self
    }

    /// Append both boundaries of every finite layer.
    ///
    /// Adjacent layers deliberately produce two samples at a shared physical
    /// interface: one referenced to the layer on the left and one referenced
    /// to the layer on the right.
    pub fn layer_interfaces(mut self) -> Self {
        self.regions.push(FieldSamplingRegion::LayerInterfaces);

        self
    }

    /// Return the underlying requests.
    pub fn regions(&self) -> &[FieldSamplingRegion<R>] {
        &self.regions
    }

    /// Consume the specification and return its underlying requests.
    pub fn into_regions(self) -> Vec<FieldSamplingRegion<R>> {
        self.regions
    }

    /// Expand this specification into concrete field positions.
    ///
    /// Positions are returned in request order.
    pub fn positions<M>(
        &self,
        stack: &Stack<M, R>,
    ) -> Result<Vec<FieldPosition<R>>, PlaneWaveFieldError<R>>
    where
        R: Float,
    {
        let mut positions = Vec::new();

        for region in &self.regions {
            match region {
                FieldSamplingRegion::LeftExterior(sampling) => {
                    expand_left_exterior(sampling, &mut positions)?;
                }

                FieldSamplingRegion::Layer { index, sampling } => {
                    let Some(layer) = stack.layers_left_to_right().get(*index) else {
                        return Err(PlaneWaveFieldError::LayerOutOfBounds {
                            requested: *index,
                            layer_count: stack.len(),
                        });
                    };

                    expand_layer(*index, layer.thickness().as_cm(), sampling, &mut positions)?;
                }

                FieldSamplingRegion::RightExterior(sampling) => {
                    expand_right_exterior(sampling, &mut positions)?;
                }

                FieldSamplingRegion::LayerCentres => {
                    let two = R::one() + R::one();

                    for (index, layer) in stack.layers_left_to_right().iter().enumerate() {
                        positions.push(FieldPosition::Layer {
                            index,
                            offset: layer.thickness().as_cm() / two,
                        });
                    }
                }

                FieldSamplingRegion::LayerInterfaces => {
                    for (index, layer) in stack.layers_left_to_right().iter().enumerate() {
                        positions.push(FieldPosition::Layer {
                            index,
                            offset: R::zero(),
                        });

                        positions.push(FieldPosition::Layer {
                            index,
                            offset: layer.thickness().as_cm(),
                        });
                    }
                }
            }
        }

        Ok(positions)
    }
}

fn expand_left_exterior<R>(
    sampling: &ExteriorSampling<R>,
    positions: &mut Vec<FieldPosition<R>>,
) -> Result<(), PlaneWaveFieldError<R>>
where
    R: Copy + Float,
{
    match sampling {
        ExteriorSampling::Point(distance) => {
            validate_exterior_distance(*distance)?;

            positions.push(FieldPosition::LeftExterior {
                distance: *distance,
            });
        }

        ExteriorSampling::Distances(distances) => {
            for &distance in distances {
                validate_exterior_distance(distance)?;

                positions.push(FieldPosition::LeftExterior { distance });
            }
        }

        ExteriorSampling::Uniform { points, distance } => {
            validate_exterior_distance(*distance)?;

            let distances = uniform_closed_interval(R::zero(), *distance, *points)?;

            /*
             * Larger left-exterior distances have more negative global
             * coordinates, so reverse the distance grid to preserve
             * geometric left-to-right ordering.
             */
            for distance in distances.into_iter().rev() {
                positions.push(FieldPosition::LeftExterior { distance });
            }
        }
    }

    Ok(())
}

fn expand_right_exterior<R>(
    sampling: &ExteriorSampling<R>,
    positions: &mut Vec<FieldPosition<R>>,
) -> Result<(), PlaneWaveFieldError<R>>
where
    R: Copy + Float,
{
    match sampling {
        ExteriorSampling::Point(distance) => {
            validate_exterior_distance(*distance)?;

            positions.push(FieldPosition::RightExterior {
                distance: *distance,
            });
        }

        ExteriorSampling::Distances(distances) => {
            for &distance in distances {
                validate_exterior_distance(distance)?;

                positions.push(FieldPosition::RightExterior { distance });
            }
        }

        ExteriorSampling::Uniform { points, distance } => {
            validate_exterior_distance(*distance)?;

            for distance in uniform_closed_interval(R::zero(), *distance, *points)? {
                positions.push(FieldPosition::RightExterior { distance });
            }
        }
    }

    Ok(())
}

fn expand_layer<R>(
    index: usize,
    thickness: R,
    sampling: &LayerSampling<R>,
    positions: &mut Vec<FieldPosition<R>>,
) -> Result<(), PlaneWaveFieldError<R>>
where
    R: Copy + Float,
{
    match sampling {
        LayerSampling::Point(offset) => {
            validate_layer_offset(index, *offset, thickness)?;

            positions.push(FieldPosition::Layer {
                index,
                offset: *offset,
            });
        }

        LayerSampling::Offsets(offsets) => {
            for &offset in offsets {
                validate_layer_offset(index, offset, thickness)?;

                positions.push(FieldPosition::Layer { index, offset });
            }
        }

        LayerSampling::Uniform {
            points,
            include_left,
            include_right,
        } => {
            let offsets = uniform_layer_offsets(thickness, *points, *include_left, *include_right)?;

            for offset in offsets {
                positions.push(FieldPosition::Layer { index, offset });
            }
        }
    }

    Ok(())
}

fn uniform_closed_interval<R>(
    start: R,
    end: R,
    points: usize,
) -> Result<Vec<R>, PlaneWaveFieldError<R>>
where
    R: Copy + Float,
{
    if points == 0 {
        return Err(PlaneWaveFieldError::EmptyUniformSampling);
    }

    if points == 1 {
        return Ok(vec![start]);
    }

    let denominator = R::from(points - 1).expect("usize point count should be representable");

    let step = (end - start) / denominator;

    Ok((0..points)
        .map(|index| {
            let index = R::from(index).expect("usize sample index should be representable");

            start + index * step
        })
        .collect())
}

fn uniform_layer_offsets<R>(
    thickness: R,
    points: usize,
    include_left: bool,
    include_right: bool,
) -> Result<Vec<R>, PlaneWaveFieldError<R>>
where
    R: Copy + Float,
{
    if points == 0 {
        return Err(PlaneWaveFieldError::EmptyUniformSampling);
    }

    if points == 1 {
        return match (include_left, include_right) {
            (true, true) => Err(PlaneWaveFieldError::AmbiguousSinglePointLayerSampling),

            (true, false) => Ok(vec![R::zero()]),

            (false, true) => Ok(vec![thickness]),

            (false, false) => {
                let two = R::one() + R::one();

                Ok(vec![thickness / two])
            }
        };
    }

    let left_exclusion = usize::from(!include_left);

    let right_exclusion = usize::from(!include_right);

    let denominator_count = points - 1 + left_exclusion + right_exclusion;

    let denominator =
        R::from(denominator_count).expect("usize point count should be representable");

    let first_index = left_exclusion;

    Ok((0..points)
        .map(|sample| {
            let index = first_index + sample;

            let fraction =
                R::from(index).expect("usize sample index should be representable") / denominator;

            thickness * fraction
        })
        .collect())
}

pub(crate) fn validate_layer_offset<R>(
    layer: usize,
    offset: R,
    thickness: R,
) -> Result<(), PlaneWaveFieldError<R>>
where
    R: Copy + Float,
{
    if !offset.is_finite() || offset < R::zero() || offset > thickness {
        return Err(PlaneWaveFieldError::InvalidLayerOffset {
            layer,
            offset,
            thickness,
        });
    }

    Ok(())
}

pub(super) fn validate_exterior_distance<R>(distance: R) -> Result<(), PlaneWaveFieldError<R>>
where
    R: Copy + Float,
{
    if !distance.is_finite() || distance < R::zero() {
        return Err(PlaneWaveFieldError::InvalidExteriorDistance { distance });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_layer_sampling_includes_both_boundaries() {
        let offsets = uniform_layer_offsets(2.0_f64, 5, true, true).unwrap();

        assert_eq!(offsets, vec![0.0, 0.5, 1.0, 1.5, 2.0,],);
    }

    #[test]
    fn uniform_interior_layer_sampling_excludes_boundaries() {
        let offsets = uniform_layer_offsets(1.0_f64, 3, false, false).unwrap();

        assert_eq!(offsets, vec![0.25, 0.5, 0.75,],);
    }

    #[test]
    fn uniform_layer_sampling_can_include_only_left_boundary() {
        let offsets = uniform_layer_offsets(1.0_f64, 3, true, false).unwrap();

        assert_eq!(offsets, vec![0.0, 1.0 / 3.0, 2.0 / 3.0,],);
    }

    #[test]
    fn uniform_layer_sampling_can_include_only_right_boundary() {
        let offsets = uniform_layer_offsets(1.0_f64, 3, false, true).unwrap();

        assert_eq!(offsets, vec![1.0 / 3.0, 2.0 / 3.0, 1.0,],);
    }

    #[test]
    fn single_interior_sample_is_layer_centre() {
        let offsets = uniform_layer_offsets(2.0_f64, 1, false, false).unwrap();

        assert_eq!(offsets, vec![1.0]);
    }

    #[test]
    fn single_sample_cannot_include_both_layer_boundaries() {
        let result = uniform_layer_offsets(2.0_f64, 1, true, true);

        assert_eq!(
            result,
            Err(PlaneWaveFieldError::AmbiguousSinglePointLayerSampling,),
        );
    }

    #[test]
    fn zero_uniform_samples_are_rejected() {
        let result = uniform_layer_offsets(2.0_f64, 0, true, true);

        assert_eq!(result, Err(PlaneWaveFieldError::EmptyUniformSampling,),);
    }

    #[test]
    fn left_exterior_uniform_sampling_is_geometrically_ordered() {
        let mut positions = Vec::new();

        expand_left_exterior(&ExteriorSampling::uniform(3, 2.0_f64), &mut positions).unwrap();

        assert_eq!(
            positions,
            vec![
                FieldPosition::LeftExterior { distance: 2.0 },
                FieldPosition::LeftExterior { distance: 1.0 },
                FieldPosition::LeftExterior { distance: 0.0 },
            ],
        );
    }

    #[test]
    fn right_exterior_uniform_sampling_is_geometrically_ordered() {
        let mut positions = Vec::new();

        expand_right_exterior(&ExteriorSampling::uniform(3, 2.0_f64), &mut positions).unwrap();

        assert_eq!(
            positions,
            vec![
                FieldPosition::RightExterior { distance: 0.0 },
                FieldPosition::RightExterior { distance: 1.0 },
                FieldPosition::RightExterior { distance: 2.0 },
            ],
        );
    }

    #[test]
    fn field_sampling_preserves_request_order() {
        let sampling = FieldSampling::new()
            .left_exterior(ExteriorSampling::point(1.0))
            .layer(2, LayerSampling::point(0.25))
            .right_exterior(ExteriorSampling::point(2.0));

        assert_eq!(
            sampling.regions(),
            &[
                FieldSamplingRegion::LeftExterior(ExteriorSampling::Point(1.0,),),
                FieldSamplingRegion::Layer {
                    index: 2,
                    sampling: LayerSampling::Point(0.25,),
                },
                FieldSamplingRegion::RightExterior(ExteriorSampling::Point(2.0,),),
            ],
        );
    }
}
