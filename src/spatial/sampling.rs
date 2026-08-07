//! Unit-aware spatial sampling specifications for planar fields.
//!
//! Field evaluation is split into two independent sampling stages:
//!
//! 1. a plane-wave state selects a point in spectral and in-plane parameter
//!    space;
//! 2. [`FieldSampling`] selects physical positions through the corresponding
//!    planar stack.
//!
//! This module implements the second stage. A [`FieldSampling`] is a
//! declarative request: it may describe points in either exterior medium,
//! explicit or uniformly distributed points in finite layers, every layer
//! centre, or every finite-layer boundary. Calling [`FieldSampling::resolve`]
//! against a [`Stack`] expands those requests into concrete
//! [`FieldPosition`] values.
//!
//! # Units
//!
//! Requested distances are stored as [`Length`] values and retain the units
//! supplied by the caller. They are not converted into the backend's
//! canonical centimetre coordinate while the sampling specification is being
//! constructed. This allows, for example, a nanometre layer offset and a
//! micrometre exterior distance to coexist in the same request.
//!
//! Physical comparisons made during validation are unit-aware. In
//! particular, an explicit layer offset is compared with the actual layer
//! thickness after both have been converted to a common physical length.
//!
//! # Coordinate conventions
//!
//! - left-exterior distances are non-negative and measured leftward from the
//!   left stack boundary;
//! - finite-layer offsets are non-negative and measured rightward from that
//!   layer's left boundary;
//! - right-exterior distances are non-negative and measured rightward from
//!   the right stack boundary.
//!
//! A [`FieldPosition`] therefore identifies both a physical region and a
//! region-local distance. Global spatial coordinates are constructed later by
//! the field evaluator.
//!
//! # Ordering
//!
//! Top-level regions are resolved in insertion order. Explicit offset and
//! distance lists retain caller order. Uniformly generated exterior profiles
//! are ordered geometrically from left to right; consequently a uniform
//! left-exterior request appears from its largest outward distance back to
//! zero.
//!
//! [`FieldSamplingRegion::LayerInterfaces`] deliberately produces two samples
//! at each interface between finite layers: the right boundary of the layer
//! on the left and the left boundary of the layer on the right. These samples
//! occupy the same global coordinate but retain distinct material context,
//! which is required for discontinuous field components.

use num_traits::{Float, FromPrimitive, Zero};
use tmm_units::LengthUnit;

use crate::{FiniteLayerIndex, Stack, Thickness, spatial::ResolvedFieldSampling};

use super::Length;

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum FieldSamplingError<R> {
    #[error("requested finite layer {requested:?} is outside stack with {layer_count} layers")]
    LayerOutOfBounds {
        requested: FiniteLayerIndex,
        layer_count: usize,
    },

    #[error("uniform field sampling requires at least one point")]
    EmptyUniformSampling,

    #[error("one-point layer sampling cannot include both distinct boundaries")]
    AmbiguousSinglePointLayerSampling,

    #[error("layer offset {offset:?} is outside [0, {thickness:?}] for layer {layer:?}")]
    InvalidLayerOffset {
        layer: FiniteLayerIndex,
        offset: Length<R>,
        thickness: Thickness<R>,
    },

    #[error("field distance must be finite and non-negative, got {distance:?}")]
    InvalidExteriorDistance { distance: Length<R> },
}

/// One physical position at which fields are to be evaluated.
///
/// Coordinates are local to the containing region rather than global stack
/// coordinates. This preserves which side of an interface was requested.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum FieldPosition<R> {
    /// Position in the left semi-infinite exterior.
    ///
    /// `distance` is measured outward, to the left, from the stack boundary.
    LeftExterior { distance: Length<R> },

    /// Position in a finite layer.
    ///
    /// `offset` is measured rightward from the layer's left boundary.
    Layer {
        index: FiniteLayerIndex,
        offset: Length<R>,
    },

    /// Position in the right semi-infinite exterior.
    ///
    /// `distance` is measured outward, to the right, from the stack boundary.
    RightExterior { distance: Length<R> },
}

impl<R> FieldPosition<R>
where
    R: Zero,
{
    pub fn left_boundary() -> Self {
        Self::LeftExterior {
            distance: Length::zero(),
        }
    }

    pub fn right_boundary() -> Self {
        Self::RightExterior {
            distance: Length::zero(),
        }
    }

    pub fn layer_left(index: impl Into<FiniteLayerIndex>) -> Self {
        Self::Layer {
            index: index.into(),
            offset: Length::zero(),
        }
    }

    pub fn layer_right(index: impl Into<FiniteLayerIndex>, thickness: Thickness<R>) -> Self {
        Self::Layer {
            index: index.into(),
            offset: thickness.into_inner(),
        }
    }

    pub fn layer_centre(index: impl Into<FiniteLayerIndex>, thickness: Thickness<R>) -> Self
    where
        R: Float,
    {
        Self::Layer {
            index: index.into(),
            offset: thickness.into_inner().half(),
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

/// One high-level spatial region included in a [`FieldSampling`] request.
#[derive(Clone, Debug, PartialEq)]
pub enum FieldSamplingRegion<R> {
    LeftExterior(ExteriorSampling<R>),

    Layer {
        index: FiniteLayerIndex,
        sampling: LayerSampling<R>,
    },

    RightExterior(ExteriorSampling<R>),

    LayerCentres,

    LayerInterfaces,
}

/// Sampling pattern within one finite layer.
///
/// Layer coordinates are offsets measured rightward from the layer's left
/// boundary.
#[derive(Clone, Debug, PartialEq)]
pub enum LayerSampling<R> {
    /// One point.
    Point(Length<R>),

    /// Explicit offsets.
    Offsets(Vec<Length<R>>),

    /// Uniform spacing.
    Uniform {
        points: usize,
        include_left: bool,
        include_right: bool,
    },
}

impl<R> LayerSampling<R> {
    pub fn point(point: Length<R>) -> Self {
        Self::Point(point)
    }

    pub fn offsets(offsets: impl IntoIterator<Item = Length<R>>) -> Self {
        Self::Offsets(offsets.into_iter().collect())
    }

    pub fn offsets_in(values: impl IntoIterator<Item = R>, unit: LengthUnit) -> Self {
        Self::offsets(values.into_iter().map(|each| Length::new(each, unit)))
    }

    /// Uniformly sample a finite layer.
    ///
    /// By default, the left boundary is included and the right boundary is
    /// excluded. This is convenient for profiles spanning consecutive layers
    /// because it avoids duplicate coordinates at ordinary interfaces.
    pub fn uniform(points: usize) -> Self {
        Self::uniform_with_boundaries(points, true, false)
    }

    pub fn uniform_with_boundaries(points: usize, include_left: bool, include_right: bool) -> Self {
        Self::Uniform {
            points,
            include_left,
            include_right,
        }
    }

    pub fn boundaries() -> Self {
        Self::Uniform {
            points: 2,
            include_left: true,
            include_right: true,
        }
    }
}

/// Sampling pattern in one semi-infinite exterior medium.
///
/// Distances are non-negative and measured outward from the adjacent stack
/// boundary.
#[derive(Clone, Debug, PartialEq)]
pub enum ExteriorSampling<R> {
    Point(Length<R>),

    Distances(Vec<Length<R>>),

    Uniform { points: usize, distance: Length<R> },
}

impl<R> Default for FieldSampling<R> {
    fn default() -> Self {
        Self::new()
    }
}

impl<R> ExteriorSampling<R> {
    pub fn point(point: Length<R>) -> Self {
        Self::Point(point)
    }

    pub fn distances(distances: impl IntoIterator<Item = Length<R>>) -> Self {
        Self::Distances(distances.into_iter().collect())
    }

    pub fn distances_in(values: impl IntoIterator<Item = R>, unit: LengthUnit) -> Self {
        Self::distances(values.into_iter().map(|each| Length::new(each, unit)))
    }

    pub fn uniform(points: usize, distance: Length<R>) -> Self {
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

    pub fn len(&self) -> usize {
        self.regions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.regions.is_empty()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &FieldSamplingRegion<R>> {
        self.regions.iter()
    }

    /// Append an explicit left-exterior sampling request.
    pub fn left_exterior(mut self, sampling: ExteriorSampling<R>) -> Self {
        self.regions
            .push(FieldSamplingRegion::LeftExterior(sampling));

        self
    }

    /// Append an explicit finite-layer sampling request.
    pub fn layer(mut self, index: impl Into<FiniteLayerIndex>, sampling: LayerSampling<R>) -> Self {
        self.regions.push(FieldSamplingRegion::Layer {
            index: index.into(),
            sampling,
        });

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

    /// Resolve this declarative sampling request against `stack`.
    ///
    /// Stack-dependent requests such as layer centres, interfaces, and uniform
    /// finite-layer samples are expanded using the actual finite-layer
    /// thicknesses. Explicit layer indices and offsets are validated here.
    ///
    /// The resulting positions retain their physical [`Length`] representation;
    /// conversion to canonical backend coordinates belongs to field evaluation.
    ///
    /// # Ordering
    ///
    /// Regions are resolved in insertion order. Explicit point lists preserve
    /// their supplied order. Uniform exterior sampling is geometrically ordered
    /// from left to right.
    ///
    /// # Errors
    ///
    /// Returns [`FieldSamplingError`] when:
    ///
    /// - a requested finite-layer index does not exist;
    /// - a uniform request contains zero points;
    /// - a one-point finite-layer request asks for both distinct boundaries;
    /// - a layer offset lies outside its layer;
    /// - an exterior distance is negative or non-finite.
    pub fn resolve<M>(
        &self,
        stack: &Stack<M, R>,
    ) -> Result<ResolvedFieldSampling<R>, FieldSamplingError<R>>
    where
        R: Float + FromPrimitive,
    {
        let mut positions = Vec::new();

        for region in &self.regions {
            match region {
                FieldSamplingRegion::LeftExterior(sampling) => {
                    expand_left_exterior(sampling, &mut positions)?;
                }

                FieldSamplingRegion::Layer { index, sampling } => {
                    let Some(layer) = stack.layers_left_to_right().get(index.0) else {
                        return Err(FieldSamplingError::LayerOutOfBounds {
                            requested: *index,
                            layer_count: stack.len(),
                        });
                    };

                    expand_layer(*index, layer.thickness(), sampling, &mut positions)?;
                }

                FieldSamplingRegion::RightExterior(sampling) => {
                    expand_right_exterior(sampling, &mut positions)?;
                }

                FieldSamplingRegion::LayerCentres => {
                    for (index, layer) in stack
                        .layers_left_to_right()
                        .iter()
                        .enumerate()
                        .map(|(index, layer)| (FiniteLayerIndex(index), layer))
                    {
                        positions.push(FieldPosition::Layer {
                            index,
                            offset: layer.thickness().into_inner().half(),
                        });
                    }
                }

                FieldSamplingRegion::LayerInterfaces => {
                    for (index, layer) in stack
                        .layers_left_to_right()
                        .iter()
                        .enumerate()
                        .map(|(index, layer)| (FiniteLayerIndex(index), layer))
                    {
                        positions.push(FieldPosition::Layer {
                            index,
                            offset: Length::zero(),
                        });

                        positions.push(FieldPosition::Layer {
                            index,
                            offset: layer.thickness().into_inner(),
                        });
                    }
                }
            }
        }

        Ok(ResolvedFieldSampling::new(positions))
    }
}

fn expand_left_exterior<R>(
    sampling: &ExteriorSampling<R>,
    positions: &mut Vec<FieldPosition<R>>,
) -> Result<(), FieldSamplingError<R>>
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

            let distances = uniform_closed_interval_from_zero(*distance, *points)?;

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
) -> Result<(), FieldSamplingError<R>>
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

            for distance in uniform_closed_interval_from_zero(*distance, *points)? {
                positions.push(FieldPosition::RightExterior { distance });
            }
        }
    }

    Ok(())
}

fn expand_layer<R>(
    index: FiniteLayerIndex,
    thickness: Thickness<R>,
    sampling: &LayerSampling<R>,
    positions: &mut Vec<FieldPosition<R>>,
) -> Result<(), FieldSamplingError<R>>
where
    R: Copy + Float + FromPrimitive,
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

fn uniform_closed_interval_from_zero<R>(
    end: Length<R>,
    points: usize,
) -> Result<Vec<Length<R>>, FieldSamplingError<R>>
where
    R: Copy + Float,
{
    if points == 0 {
        return Err(FieldSamplingError::EmptyUniformSampling);
    }

    if points == 1 {
        return Ok(vec![Length::zero()]);
    }

    let denominator = R::from(points - 1).expect("usize point count should be representable");

    let step_size = end.scale_by(R::one() / denominator);

    Ok((0..points)
        .map(|index| {
            let index = R::from(index).expect("usize sample index should be representable");

            step_size.scale_by(index)
        })
        .collect())
}

fn uniform_layer_offsets<R>(
    thickness: Thickness<R>,
    points: usize,
    include_left: bool,
    include_right: bool,
) -> Result<Vec<Length<R>>, FieldSamplingError<R>>
where
    R: Copy + Float,
{
    if points == 0 {
        return Err(FieldSamplingError::EmptyUniformSampling);
    }

    let thickness = thickness.into_inner();

    if points == 1 {
        return match (include_left, include_right) {
            (true, true) => Err(FieldSamplingError::AmbiguousSinglePointLayerSampling),

            (true, false) => Ok(vec![Length::zero()]),

            (false, true) => Ok(vec![thickness]),

            (false, false) => Ok(vec![thickness.half()]),
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

            thickness.scale_by(fraction)
        })
        .collect())
}

pub(crate) fn validate_layer_offset<R>(
    layer: FiniteLayerIndex,
    offset: Length<R>,
    thickness: Thickness<R>,
) -> Result<(), FieldSamplingError<R>>
where
    R: Copy + Float + FromPrimitive,
{
    if !offset.value().is_finite()
        || offset.value() < R::zero()
        || offset.as_cm() > thickness.as_cm()
    {
        return Err(FieldSamplingError::InvalidLayerOffset {
            layer,
            offset,
            thickness,
        });
    }

    Ok(())
}

pub(super) fn validate_exterior_distance<R>(
    distance: Length<R>,
) -> Result<(), FieldSamplingError<R>>
where
    R: Copy + Float,
{
    if !distance.value().is_finite() || distance.value() < R::zero() {
        return Err(FieldSamplingError::InvalidExteriorDistance { distance });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{Constant, Layer};

    type R = f64;

    fn nm(value: R) -> Length<R> {
        Thickness::nanometres(value).into_inner()
    }

    fn um(value: R) -> Length<R> {
        Thickness::micrometres(value).into_inner()
    }

    fn thickness_nm(value: R) -> Thickness<R> {
        Thickness::nanometres(value)
    }

    fn thickness_um(value: R) -> Thickness<R> {
        Thickness::micrometres(value)
    }

    fn stack() -> Stack<Constant<R>, R> {
        Stack::new(
            Constant::dielectric(1.0),
            vec![
                Layer::new(Constant::dielectric(2.0), thickness_nm(500.0)),
                Layer::new(Constant::dielectric(3.0), thickness_um(2.0)),
            ],
            Constant::dielectric(1.0),
        )
    }

    fn assert_same_length(actual: Length<R>, expected: Length<R>) {
        let error = (actual.as_cm() - expected.as_cm()).abs();

        assert!(
            error <= 1.0e-14,
            "expected {:?}, got {:?}",
            expected,
            actual,
        );
    }

    fn assert_lengths(actual: &[Length<R>], expected: &[Length<R>]) {
        assert_eq!(actual.len(), expected.len());

        for (&actual, &expected) in actual.iter().zip(expected) {
            assert_same_length(actual, expected);
        }
    }

    // ---------------------------------------------------------------------
    // FieldPosition convenience constructors
    // ---------------------------------------------------------------------

    #[test]
    fn left_boundary_is_zero_distance_in_left_exterior() {
        assert_eq!(
            FieldPosition::<R>::left_boundary(),
            FieldPosition::LeftExterior {
                distance: Length::zero(),
            },
        );
    }

    #[test]
    fn right_boundary_is_zero_distance_in_right_exterior() {
        assert_eq!(
            FieldPosition::<R>::right_boundary(),
            FieldPosition::RightExterior {
                distance: Length::zero(),
            },
        );
    }

    #[test]
    fn layer_left_is_zero_offset() {
        assert_eq!(
            FieldPosition::<R>::layer_left(FiniteLayerIndex(3)),
            FieldPosition::Layer {
                index: FiniteLayerIndex(3),
                offset: Length::zero(),
            },
        );
    }

    #[test]
    fn layer_right_uses_complete_layer_thickness() {
        let thickness = thickness_nm(750.0);

        assert_eq!(
            FieldPosition::layer_right(FiniteLayerIndex(2), thickness,),
            FieldPosition::Layer {
                index: FiniteLayerIndex(2),
                offset: thickness.into_inner(),
            },
        );
    }

    #[test]
    fn layer_centre_uses_half_layer_thickness() {
        let thickness = thickness_um(2.0);

        let FieldPosition::Layer { index, offset } =
            FieldPosition::layer_centre(FiniteLayerIndex(4), thickness)
        else {
            panic!("expected finite-layer position");
        };

        assert_eq!(index, FiniteLayerIndex(4));

        assert_same_length(offset, um(1.0));
    }

    // ---------------------------------------------------------------------
    // Request constructors
    // ---------------------------------------------------------------------

    #[test]
    fn layer_sampling_point_stores_length() {
        let point = nm(125.0);

        assert_eq!(LayerSampling::point(point), LayerSampling::Point(point),);
    }

    #[test]
    fn layer_sampling_offsets_preserves_values_and_units() {
        let values = vec![nm(100.0), um(0.25), nm(400.0)];

        assert_eq!(
            LayerSampling::offsets(values.clone()),
            LayerSampling::Offsets(values),
        );
    }

    #[test]
    fn uniform_layer_sampling_defaults_to_left_inclusive_right_exclusive() {
        assert_eq!(
            LayerSampling::<R>::uniform(5),
            LayerSampling::Uniform {
                points: 5,
                include_left: true,
                include_right: false,
            },
        );
    }

    #[test]
    fn boundaries_requests_two_layer_boundaries() {
        assert_eq!(
            LayerSampling::<R>::boundaries(),
            LayerSampling::Uniform {
                points: 2,
                include_left: true,
                include_right: true,
            },
        );
    }

    #[test]
    fn exterior_sampling_constructors_preserve_lengths() {
        assert_eq!(
            ExteriorSampling::point(um(1.0)),
            ExteriorSampling::Point(um(1.0)),
        );

        assert_eq!(
            ExteriorSampling::distances(vec![nm(100.0), um(1.0)],),
            ExteriorSampling::Distances(vec![nm(100.0), um(1.0)],),
        );

        assert_eq!(
            ExteriorSampling::uniform(5, um(2.0),),
            ExteriorSampling::Uniform {
                points: 5,
                distance: um(2.0),
            },
        );
    }

    #[test]
    fn field_sampling_default_is_empty() {
        let sampling = FieldSampling::<R>::default();

        assert!(sampling.regions().is_empty());
    }

    #[test]
    fn field_sampling_preserves_region_request_order() {
        let sampling = FieldSampling::new()
            .left_exterior(ExteriorSampling::point(um(1.0)))
            .layer(FiniteLayerIndex(1), LayerSampling::point(nm(250.0)))
            .right_exterior(ExteriorSampling::point(um(2.0)))
            .layer_centres()
            .layer_interfaces();

        assert_eq!(
            sampling.regions(),
            &[
                FieldSamplingRegion::LeftExterior(ExteriorSampling::Point(um(1.0),),),
                FieldSamplingRegion::Layer {
                    index: FiniteLayerIndex(1),
                    sampling: LayerSampling::Point(nm(250.0),),
                },
                FieldSamplingRegion::RightExterior(ExteriorSampling::Point(um(2.0),),),
                FieldSamplingRegion::LayerCentres,
                FieldSamplingRegion::LayerInterfaces,
            ],
        );
    }

    #[test]
    fn into_regions_preserves_requests() {
        let sampling: FieldSampling<f64> = FieldSampling::new().layer_centres().layer_interfaces();

        assert_eq!(
            sampling.into_regions(),
            vec![
                FieldSamplingRegion::LayerCentres,
                FieldSamplingRegion::LayerInterfaces,
            ],
        );
    }

    // ---------------------------------------------------------------------
    // Uniform exterior expansion
    // ---------------------------------------------------------------------

    #[test]
    fn uniform_closed_interval_includes_zero_and_end() {
        let values = uniform_closed_interval_from_zero(um(2.0), 5).unwrap();

        assert_lengths(&values, &[um(0.0), um(0.5), um(1.0), um(1.5), um(2.0)]);
    }

    #[test]
    fn one_point_closed_interval_returns_zero() {
        let values = uniform_closed_interval_from_zero(um(5.0), 1).unwrap();

        assert_lengths(&values, &[um(0.0)]);
    }

    #[test]
    fn zero_point_closed_interval_is_rejected() {
        assert_eq!(
            uniform_closed_interval_from_zero(um(2.0), 0,),
            Err(FieldSamplingError::EmptyUniformSampling,),
        );
    }

    #[test]
    fn left_exterior_uniform_sampling_is_geometrically_left_to_right() {
        let mut positions = Vec::new();

        expand_left_exterior(&ExteriorSampling::uniform(3, um(2.0)), &mut positions).unwrap();

        assert_eq!(
            positions,
            vec![
                FieldPosition::LeftExterior { distance: um(2.0) },
                FieldPosition::LeftExterior { distance: um(1.0) },
                FieldPosition::LeftExterior { distance: um(0.0) },
            ],
        );
    }

    #[test]
    fn right_exterior_uniform_sampling_is_geometrically_left_to_right() {
        let mut positions = Vec::new();

        expand_right_exterior(&ExteriorSampling::uniform(3, um(2.0)), &mut positions).unwrap();

        assert_eq!(
            positions,
            vec![
                FieldPosition::RightExterior { distance: um(0.0) },
                FieldPosition::RightExterior { distance: um(1.0) },
                FieldPosition::RightExterior { distance: um(2.0) },
            ],
        );
    }

    #[test]
    fn explicit_left_exterior_distances_preserve_caller_order() {
        let mut positions = Vec::new();

        expand_left_exterior(
            &ExteriorSampling::distances(vec![um(1.0), nm(250.0), um(2.0)]),
            &mut positions,
        )
        .unwrap();

        assert_eq!(
            positions,
            vec![
                FieldPosition::LeftExterior { distance: um(1.0) },
                FieldPosition::LeftExterior {
                    distance: nm(250.0),
                },
                FieldPosition::LeftExterior { distance: um(2.0) },
            ],
        );
    }

    // ---------------------------------------------------------------------
    // Uniform layer expansion
    // ---------------------------------------------------------------------

    #[test]
    fn uniform_layer_sampling_includes_both_boundaries() {
        let offsets = uniform_layer_offsets(thickness_um(2.0), 5, true, true).unwrap();

        assert_lengths(&offsets, &[um(0.0), um(0.5), um(1.0), um(1.5), um(2.0)]);
    }

    #[test]
    fn uniform_layer_sampling_excludes_both_boundaries() {
        let offsets = uniform_layer_offsets(thickness_um(1.0), 3, false, false).unwrap();

        assert_lengths(&offsets, &[um(0.25), um(0.5), um(0.75)]);
    }

    #[test]
    fn uniform_layer_sampling_can_include_only_left_boundary() {
        let offsets = uniform_layer_offsets(thickness_um(1.0), 3, true, false).unwrap();

        assert_lengths(&offsets, &[um(0.0), um(1.0 / 3.0), um(2.0 / 3.0)]);
    }

    #[test]
    fn uniform_layer_sampling_can_include_only_right_boundary() {
        let offsets = uniform_layer_offsets(thickness_um(1.0), 3, false, true).unwrap();

        assert_lengths(&offsets, &[um(1.0 / 3.0), um(2.0 / 3.0), um(1.0)]);
    }

    #[test]
    fn one_interior_layer_sample_is_layer_centre() {
        let offsets = uniform_layer_offsets(thickness_um(2.0), 1, false, false).unwrap();

        assert_lengths(&offsets, &[um(1.0)]);
    }

    #[test]
    fn one_left_inclusive_layer_sample_is_left_boundary() {
        let offsets = uniform_layer_offsets(thickness_um(2.0), 1, true, false).unwrap();

        assert_lengths(&offsets, &[um(0.0)]);
    }

    #[test]
    fn one_right_inclusive_layer_sample_is_right_boundary() {
        let offsets = uniform_layer_offsets(thickness_um(2.0), 1, false, true).unwrap();

        assert_lengths(&offsets, &[um(2.0)]);
    }

    #[test]
    fn one_sample_cannot_represent_both_distinct_layer_boundaries() {
        assert_eq!(
            uniform_layer_offsets(thickness_um(2.0), 1, true, true,),
            Err(FieldSamplingError::AmbiguousSinglePointLayerSampling,),
        );
    }

    #[test]
    fn zero_uniform_layer_samples_are_rejected() {
        assert_eq!(
            uniform_layer_offsets(thickness_um(2.0), 0, true, false,),
            Err(FieldSamplingError::EmptyUniformSampling,),
        );
    }

    // ---------------------------------------------------------------------
    // Validation
    // ---------------------------------------------------------------------

    #[test]
    fn zero_exterior_distance_is_valid() {
        assert_eq!(validate_exterior_distance(um(0.0),), Ok(()),);
    }

    #[test]
    fn negative_exterior_distance_is_rejected() {
        let distance = um(-1.0);

        assert_eq!(
            validate_exterior_distance(distance,),
            Err(FieldSamplingError::InvalidExteriorDistance { distance },),
        );
    }

    #[test]
    fn non_finite_exterior_distance_is_rejected() {
        let distance = um(R::NAN);

        assert!(matches!(
            validate_exterior_distance(distance,),
            Err(FieldSamplingError::InvalidExteriorDistance { .. }),
        ));
    }

    #[test]
    fn layer_offset_at_left_boundary_is_valid() {
        assert_eq!(
            validate_layer_offset(FiniteLayerIndex(2), nm(0.0), thickness_um(1.0),),
            Ok(()),
        );
    }

    #[test]
    fn layer_offset_at_right_boundary_is_valid() {
        assert_eq!(
            validate_layer_offset(FiniteLayerIndex(2), nm(1000.0), thickness_um(1.0),),
            Ok(()),
        );
    }

    #[test]
    fn mixed_unit_layer_offset_below_thickness_is_valid() {
        assert_eq!(
            validate_layer_offset(FiniteLayerIndex(2), nm(999.0), thickness_um(1.0),),
            Ok(()),
        );
    }

    #[test]
    fn mixed_unit_layer_offset_above_thickness_is_rejected() {
        let offset = nm(1001.0);
        let thickness = thickness_um(1.0);

        assert_eq!(
            validate_layer_offset(FiniteLayerIndex(2), offset, thickness,),
            Err(FieldSamplingError::InvalidLayerOffset {
                layer: FiniteLayerIndex(2),
                offset,
                thickness,
            },),
        );
    }

    #[test]
    fn negative_layer_offset_is_rejected() {
        let offset = nm(-1.0);
        let thickness = thickness_um(1.0);

        assert_eq!(
            validate_layer_offset(FiniteLayerIndex(2), offset, thickness,),
            Err(FieldSamplingError::InvalidLayerOffset {
                layer: FiniteLayerIndex(2),
                offset,
                thickness,
            },),
        );
    }

    #[test]
    fn non_finite_layer_offset_is_rejected() {
        let offset = nm(R::NAN);

        assert!(matches!(
            validate_layer_offset(FiniteLayerIndex(2), offset, thickness_um(1.0),),
            Err(FieldSamplingError::InvalidLayerOffset { .. }),
        ));
    }

    // ---------------------------------------------------------------------
    // Stack resolution
    // ---------------------------------------------------------------------

    #[test]
    fn empty_sampling_resolves_to_empty_sampling() {
        let actual = FieldSampling::<R>::new().resolve(&stack()).unwrap();

        assert_eq!(actual, ResolvedFieldSampling::new(Vec::new(),),);
    }

    #[test]
    fn resolve_preserves_top_level_request_order() {
        let actual = FieldSampling::new()
            .left_exterior(ExteriorSampling::point(um(1.0)))
            .layer(FiniteLayerIndex(0), LayerSampling::point(nm(125.0)))
            .right_exterior(ExteriorSampling::point(um(2.0)))
            .resolve(&stack())
            .unwrap();

        assert_eq!(
            actual,
            ResolvedFieldSampling::new(vec![
                FieldPosition::LeftExterior { distance: um(1.0) },
                FieldPosition::Layer {
                    index: FiniteLayerIndex(0),
                    offset: nm(125.0),
                },
                FieldPosition::RightExterior { distance: um(2.0) },
            ]),
        );
    }

    #[test]
    fn resolve_rejects_out_of_range_layer_index() {
        let error = FieldSampling::new()
            .layer(FiniteLayerIndex(2), LayerSampling::point(nm(10.0)))
            .resolve(&stack())
            .unwrap_err();

        assert_eq!(
            error,
            FieldSamplingError::LayerOutOfBounds {
                requested: FiniteLayerIndex(2),
                layer_count: 2,
            },
        );
    }

    #[test]
    fn resolve_validates_explicit_offset_against_actual_layer_thickness() {
        let offset = nm(501.0);
        let thickness = thickness_nm(500.0);

        let error = FieldSampling::new()
            .layer(FiniteLayerIndex(0), LayerSampling::point(offset))
            .resolve(&stack())
            .unwrap_err();

        assert_eq!(
            error,
            FieldSamplingError::InvalidLayerOffset {
                layer: FiniteLayerIndex(0),
                offset,
                thickness,
            },
        );
    }

    #[test]
    fn resolve_accepts_explicit_offset_expressed_in_different_unit() {
        let actual = FieldSampling::new()
            .layer(0, LayerSampling::point(um(0.25)))
            .resolve(&stack())
            .unwrap();

        assert_eq!(
            actual,
            ResolvedFieldSampling::new(vec![FieldPosition::Layer {
                index: FiniteLayerIndex(0),
                offset: um(0.25),
            },]),
        );
    }

    #[test]
    fn resolve_layer_centres_uses_each_layers_own_thickness_unit() {
        let actual = FieldSampling::<R>::new()
            .layer_centres()
            .resolve(&stack())
            .unwrap();

        assert_eq!(
            actual,
            ResolvedFieldSampling::new(vec![
                FieldPosition::Layer {
                    index: FiniteLayerIndex(0),
                    offset: nm(250.0),
                },
                FieldPosition::Layer {
                    index: FiniteLayerIndex(1),
                    offset: um(1.0),
                },
            ]),
        );
    }

    #[test]
    fn resolve_layer_interfaces_retains_both_sides_of_internal_interface() {
        let actual = FieldSampling::<R>::new()
            .layer_interfaces()
            .resolve(&stack())
            .unwrap();

        assert_eq!(
            actual,
            ResolvedFieldSampling::new(vec![
                FieldPosition::Layer {
                    index: FiniteLayerIndex(0),
                    offset: Length::zero(),
                },
                FieldPosition::Layer {
                    index: FiniteLayerIndex(0),
                    offset: nm(500.0),
                },
                FieldPosition::Layer {
                    index: FiniteLayerIndex(1),
                    offset: Length::zero(),
                },
                FieldPosition::Layer {
                    index: FiniteLayerIndex(1),
                    offset: um(2.0),
                },
            ]),
        );
    }

    #[test]
    fn resolve_uniform_layer_sampling_uses_actual_layer_thickness() {
        let actual = FieldSampling::new()
            .layer(
                1,
                LayerSampling::Uniform {
                    points: 3,
                    include_left: true,
                    include_right: true,
                },
            )
            .resolve(&stack())
            .unwrap();

        assert_eq!(
            actual,
            ResolvedFieldSampling::new(vec![
                FieldPosition::Layer {
                    index: FiniteLayerIndex(1),
                    offset: um(0.0),
                },
                FieldPosition::Layer {
                    index: FiniteLayerIndex(1),
                    offset: um(1.0),
                },
                FieldPosition::Layer {
                    index: FiniteLayerIndex(1),
                    offset: um(2.0),
                },
            ]),
        );
    }

    #[test]
    fn resolve_uniform_left_exterior_uses_geometric_order() {
        let actual = FieldSampling::new()
            .left_exterior(ExteriorSampling::uniform(3, um(2.0)))
            .resolve(&stack())
            .unwrap();

        assert_eq!(
            actual,
            ResolvedFieldSampling::new(vec![
                FieldPosition::LeftExterior { distance: um(2.0) },
                FieldPosition::LeftExterior { distance: um(1.0) },
                FieldPosition::LeftExterior { distance: um(0.0) },
            ]),
        );
    }

    #[test]
    fn resolve_uniform_right_exterior_uses_geometric_order() {
        let actual = FieldSampling::new()
            .right_exterior(ExteriorSampling::uniform(3, um(2.0)))
            .resolve(&stack())
            .unwrap();

        assert_eq!(
            actual,
            ResolvedFieldSampling::new(vec![
                FieldPosition::RightExterior { distance: um(0.0) },
                FieldPosition::RightExterior { distance: um(1.0) },
                FieldPosition::RightExterior { distance: um(2.0) },
            ]),
        );
    }
}
