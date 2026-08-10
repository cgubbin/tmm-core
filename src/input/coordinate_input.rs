//! Sampled caller-facing coordinate inputs.
//!
//! This module pairs a caller-facing [`Coordinates`] description with sampled
//! spectral and in-plane values.
//!
//! Coordinate inputs contain only the information needed to compile the
//! canonical plane-wave coordinates. Polarisation is supplied separately when
//! evaluating the backend, while incidence direction is normally selected
//! later when projecting amplitudes and powers.
//!
//! Some coordinate representations are intrinsically defined. Examples include
//! parallel wavenumber, propagation constant, and effective index. These use
//! [`CoordinateInput::intrinsic`].
//!
//! Incident angle is defined relative to one exterior medium and therefore
//! requires an [`IncidentSide`] reference. Such inputs use
//! [`CoordinateInput::incident_referenced`].
//!
//! Spectral and in-plane arrays must have exactly the same shape. Inputs that
//! could be broadcast to a common shape are rejected; callers must perform any
//! desired broadcasting explicitly.

use nalgebra::ComplexField;
use ndarray::{Array, Array1, Array2, Dimension, IntoDimension, Ix0, Ix1, Ix2, arr0};

use super::{Coordinates, IncidentSide, PlaneWaveInputError, ReferenceRequirement};

/// A scalar coordinate input.
pub type CoordinatePoint<S> = CoordinateInput<S, Ix0>;

/// A one-dimensional sequence of coordinate inputs.
pub type CoordinateSamples<S> = CoordinateInput<S, Ix1>;

/// A two-dimensional grid of coordinate inputs.
pub type CoordinateGrid<S> = CoordinateInput<S, Ix2>;

/// Reference required to interpret caller-facing coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinateReference {
    /// The coordinate representation is independent of an incident exterior.
    Intrinsic,

    /// The coordinate representation is defined relative to the exterior on
    /// the supplied side.
    IncidentSide(IncidentSide),
}

/// Sampled spectral and in-plane coordinate values.
///
/// The coordinate description determines how the supplied values are
/// interpreted and converted into the backend's canonical vacuum and parallel
/// angular wavenumbers.
///
/// `S` may be real for scattering calculations or complex for modal
/// continuation. Both arrays have the same dimension and exact sampled shape.
#[derive(Clone, Debug, PartialEq)]
pub struct CoordinateInput<S, D>
where
    D: Dimension,
{
    coordinates: Coordinates,
    values: CoordinateValues<S, D>,
    reference: CoordinateReference,
}

impl<S, D> CoordinateInput<S, D>
where
    D: Dimension,
{
    /// Construct an input whose coordinates are intrinsically defined.
    ///
    /// This constructor is appropriate for in-plane coordinates such as
    /// parallel wavenumber, propagation constant, and effective index.
    ///
    /// # Errors
    ///
    /// Returns an error when:
    ///
    /// - the in-plane coordinate is incident angle;
    /// - the spectral and in-plane arrays have different shapes;
    /// - either array contains a non-finite value.
    pub fn intrinsic(
        coordinates: Coordinates,
        spectral: Array<S, D>,
        in_plane: Array<S, D>,
    ) -> Result<Self, PlaneWaveInputError>
    where
        S: ComplexField,
    {
        match coordinates.reference_requirement() {
            ReferenceRequirement::IncidentSide => {
                Err(PlaneWaveInputError::IncidentReferenceRequired)
            }
            ReferenceRequirement::Intrinsic => Self::new(
                coordinates,
                spectral,
                in_plane,
                CoordinateReference::Intrinsic,
            ),
        }
    }

    /// Construct an input defined relative to an incident exterior medium.
    ///
    /// This constructor is used for incident-angle coordinates. `side`
    /// identifies the exterior refractive index used to convert the angle into
    /// the conserved parallel wavenumber.
    ///
    /// # Errors
    ///
    /// Returns an error when:
    ///
    /// - the in-plane coordinate is not incident angle;
    /// - the spectral and in-plane arrays have different shapes;
    /// - either array contains a non-finite value.
    pub fn incident_referenced(
        coordinates: Coordinates,
        spectral: Array<S, D>,
        in_plane: Array<S, D>,
        side: IncidentSide,
    ) -> Result<Self, PlaneWaveInputError>
    where
        S: ComplexField,
    {
        match coordinates.reference_requirement() {
            ReferenceRequirement::Intrinsic => {
                Err(PlaneWaveInputError::UnexpectedIncidentReference)
            }
            ReferenceRequirement::IncidentSide => Self::new(
                coordinates,
                spectral,
                in_plane,
                CoordinateReference::IncidentSide(side),
            ),
        }
    }

    /// Construct a validated coordinate input with an explicit reference.
    ///
    /// Coordinate/reference compatibility must be established by the calling
    /// constructor.
    pub(crate) fn new(
        coordinates: Coordinates,
        spectral: Array<S, D>,
        in_plane: Array<S, D>,
        reference: CoordinateReference,
    ) -> Result<Self, PlaneWaveInputError>
    where
        S: ComplexField,
    {
        validate_matching_shapes(&spectral, &in_plane)?;

        if let Some(index) = first_non_finite_index(&spectral) {
            return Err(PlaneWaveInputError::NonFiniteSpectralValue { index });
        }

        if let Some(index) = first_non_finite_index(&in_plane) {
            return Err(PlaneWaveInputError::NonFiniteInPlaneValue { index });
        }

        Ok(Self {
            coordinates,
            values: CoordinateValues::new(spectral, in_plane),
            reference,
        })
    }

    /// Return the caller-facing coordinate description.
    pub fn coordinates(&self) -> Coordinates {
        self.coordinates
    }

    /// Return the sampled spectral values.
    pub fn spectral(&self) -> &Array<S, D> {
        self.values.spectral()
    }

    /// Return the sampled in-plane values.
    pub fn in_plane(&self) -> &Array<S, D> {
        self.values.in_plane()
    }

    /// Return the reference used to interpret the coordinate values.
    pub fn reference(&self) -> CoordinateReference {
        self.reference
    }

    /// Return the sampled shape.
    pub fn raw_dim(&self) -> D {
        self.values.raw_dim()
    }

    /// Extract one sampled state as a scalar coordinate input.
    ///
    /// The coordinate description and reference are preserved. Returns `None`
    /// when `index` lies outside the sampled arrays.
    pub fn get_point<I>(&self, index: I) -> Option<CoordinatePoint<S>>
    where
        S: Copy,
        I: IntoDimension<Dim = D>,
    {
        Some(CoordinateInput {
            coordinates: self.coordinates,
            values: self.values.get_point(index)?,
            reference: self.reference,
        })
    }

    /// Consume the input and return its compilation components.
    pub(crate) fn into_parts(self) -> (Coordinates, CoordinateValues<S, D>, CoordinateReference) {
        (self.coordinates, self.values, self.reference)
    }
}

impl<S> CoordinateInput<S, Ix0> {
    /// Construct a scalar input using intrinsically defined coordinates.
    pub fn point(
        coordinates: Coordinates,
        spectral: S,
        in_plane: S,
    ) -> Result<Self, PlaneWaveInputError>
    where
        S: ComplexField,
    {
        Self::intrinsic(coordinates, arr0(spectral), arr0(in_plane))
    }

    /// Construct a scalar incident-angle input.
    pub fn incident_angle_point(
        coordinates: Coordinates,
        spectral: S,
        angle: S,
        side: IncidentSide,
    ) -> Result<Self, PlaneWaveInputError>
    where
        S: ComplexField,
    {
        Self::incident_referenced(coordinates, arr0(spectral), arr0(angle), side)
    }
}

impl<S> CoordinateInput<S, Ix1> {
    /// Construct a sampled sequence using intrinsically defined coordinates.
    pub fn samples(
        coordinates: Coordinates,
        spectral: Array1<S>,
        in_plane: Array1<S>,
    ) -> Result<Self, PlaneWaveInputError>
    where
        S: ComplexField,
    {
        Self::intrinsic(coordinates, spectral, in_plane)
    }

    /// Construct a sampled sequence of incident-angle inputs.
    pub fn incident_angle_samples(
        coordinates: Coordinates,
        spectral: Array1<S>,
        angles: Array1<S>,
        side: IncidentSide,
    ) -> Result<Self, PlaneWaveInputError>
    where
        S: ComplexField,
    {
        Self::incident_referenced(coordinates, spectral, angles, side)
    }
}

impl<S> CoordinateInput<S, Ix2> {
    /// Construct a grid using intrinsically defined coordinates.
    ///
    /// Both arrays must already have the complete grid shape. Column and row
    /// vectors are not implicitly broadcast.
    pub fn grid(
        coordinates: Coordinates,
        spectral: Array2<S>,
        in_plane: Array2<S>,
    ) -> Result<Self, PlaneWaveInputError>
    where
        S: ComplexField,
    {
        Self::intrinsic(coordinates, spectral, in_plane)
    }

    /// Construct a grid of incident-angle inputs.
    ///
    /// Both arrays must already have the complete grid shape.
    pub fn incident_angle_grid(
        coordinates: Coordinates,
        spectral: Array2<S>,
        angles: Array2<S>,
        side: IncidentSide,
    ) -> Result<Self, PlaneWaveInputError>
    where
        S: ComplexField,
    {
        Self::incident_referenced(coordinates, spectral, angles, side)
    }
}

/// Paired spectral and in-plane sampled values.
///
/// This is an internal storage type. Shape, finiteness, and coordinate-reference
/// invariants are established by [`CoordinateInput`] before construction.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CoordinateValues<S, D>
where
    D: Dimension,
{
    spectral: Array<S, D>,
    in_plane: Array<S, D>,
}

impl<S, D> CoordinateValues<S, D>
where
    D: Dimension,
{
    pub(crate) fn new(spectral: Array<S, D>, in_plane: Array<S, D>) -> Self {
        Self { spectral, in_plane }
    }

    pub(crate) fn spectral(&self) -> &Array<S, D> {
        &self.spectral
    }

    pub(crate) fn in_plane(&self) -> &Array<S, D> {
        &self.in_plane
    }

    fn get_point<I>(&self, index: I) -> Option<CoordinateValues<S, Ix0>>
    where
        S: Copy,
        I: IntoDimension<Dim = D>,
    {
        let index = index.into_dimension();

        Some(CoordinateValues {
            spectral: arr0(*self.spectral.get(index.clone())?),
            in_plane: arr0(*self.in_plane.get(index)?),
        })
    }

    pub(crate) fn raw_dim(&self) -> D {
        self.spectral.raw_dim()
    }

    pub(crate) fn into_parts(self) -> (Array<S, D>, Array<S, D>) {
        (self.spectral, self.in_plane)
    }
}

fn validate_matching_shapes<S, D>(
    spectral: &Array<S, D>,
    in_plane: &Array<S, D>,
) -> Result<(), PlaneWaveInputError>
where
    D: Dimension,
{
    if spectral.raw_dim() != in_plane.raw_dim() {
        return Err(PlaneWaveInputError::ShapeMismatch {
            spectral: spectral.raw_dim().into_dyn(),
            in_plane: in_plane.raw_dim().into_dyn(),
        });
    }

    Ok(())
}

fn first_non_finite_index<S, D>(values: &Array<S, D>) -> Option<Vec<usize>>
where
    S: ComplexField,
    D: Dimension,
{
    values.indexed_iter().find_map(|(index, value)| {
        (!value.is_finite()).then(|| index.into_dimension().as_array_view().to_owned().to_vec())
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::{IxDyn, arr1, arr2};
    use num_complex::Complex64;
    use tmm_units::{AngleUnit, InverseLengthUnit};

    use crate::input::{InPlaneCoordinate, SpectralCoordinate};

    fn intrinsic_coordinates() -> Coordinates {
        Coordinates::new(
            SpectralCoordinate::VacuumWavenumber(InverseLengthUnit::PerCentimetre),
            InPlaneCoordinate::EffectiveIndex,
        )
    }

    fn angle_coordinates() -> Coordinates {
        Coordinates::new(
            SpectralCoordinate::VacuumWavenumber(InverseLengthUnit::PerCentimetre),
            InPlaneCoordinate::IncidentAngle(AngleUnit::Degree),
        )
    }

    #[test]
    fn point_constructs_intrinsic_reference() {
        let input = CoordinateInput::point(intrinsic_coordinates(), 1_000.0, 1.5).unwrap();

        assert_eq!(input.coordinates(), intrinsic_coordinates(),);
        assert_eq!(input.spectral()[()], 1_000.0);
        assert_eq!(input.in_plane()[()], 1.5);
        assert_eq!(input.reference(), CoordinateReference::Intrinsic,);
    }

    #[test]
    fn incident_angle_point_records_incident_side() {
        let input = CoordinateInput::incident_angle_point(
            angle_coordinates(),
            1_000.0,
            30.0,
            IncidentSide::Right,
        )
        .unwrap();

        assert_eq!(input.spectral()[()], 1_000.0);
        assert_eq!(input.in_plane()[()], 30.0);
        assert_eq!(
            input.reference(),
            CoordinateReference::IncidentSide(IncidentSide::Right,),
        );
    }

    #[test]
    fn intrinsic_constructor_rejects_incident_angle() {
        let error = CoordinateInput::point(angle_coordinates(), 1_000.0, 30.0).unwrap_err();

        assert_eq!(error, PlaneWaveInputError::IncidentReferenceRequired,);
    }

    #[test]
    fn incident_referenced_constructor_rejects_intrinsic_coordinate() {
        let error = CoordinateInput::incident_angle_point(
            intrinsic_coordinates(),
            1_000.0,
            1.5,
            IncidentSide::Left,
        )
        .unwrap_err();

        assert_eq!(error, PlaneWaveInputError::UnexpectedIncidentReference,);
    }

    #[test]
    fn samples_preserve_values_and_intrinsic_reference() {
        let spectral = arr1(&[1_000.0, 1_100.0]);

        let in_plane = arr1(&[1.4, 1.5]);

        let input =
            CoordinateInput::samples(intrinsic_coordinates(), spectral.clone(), in_plane.clone())
                .unwrap();

        assert_eq!(input.spectral(), &spectral);
        assert_eq!(input.in_plane(), &in_plane);
        assert_eq!(input.reference(), CoordinateReference::Intrinsic,);
    }

    #[test]
    fn incident_angle_samples_preserve_side() {
        let spectral = arr1(&[1_000.0, 1_100.0]);

        let angles = arr1(&[10.0, 20.0]);

        let input = CoordinateInput::incident_angle_samples(
            angle_coordinates(),
            spectral.clone(),
            angles.clone(),
            IncidentSide::Left,
        )
        .unwrap();

        assert_eq!(input.spectral(), &spectral);
        assert_eq!(input.in_plane(), &angles);
        assert_eq!(
            input.reference(),
            CoordinateReference::IncidentSide(IncidentSide::Left,),
        );
    }

    #[test]
    fn samples_reject_different_lengths() {
        let error = CoordinateInput::samples(
            intrinsic_coordinates(),
            arr1(&[1_000.0, 1_100.0]),
            arr1(&[1.4]),
        )
        .unwrap_err();

        assert_eq!(
            error,
            PlaneWaveInputError::ShapeMismatch {
                spectral: IxDyn(&[2]),
                in_plane: IxDyn(&[1]),
            },
        );
    }

    #[test]
    fn grid_rejects_broadcastable_but_unequal_shapes() {
        let error = CoordinateInput::grid(
            intrinsic_coordinates(),
            arr2(&[[1_000.0], [1_100.0]]),
            arr2(&[[1.0, 1.1]]),
        )
        .unwrap_err();

        assert_eq!(
            error,
            PlaneWaveInputError::ShapeMismatch {
                spectral: IxDyn(&[2, 1]),
                in_plane: IxDyn(&[1, 2]),
            },
        );
    }

    #[test]
    fn spectral_values_report_first_non_finite_index() {
        let error = CoordinateInput::grid(
            intrinsic_coordinates(),
            arr2(&[[1_000.0, 1_100.0], [f64::NAN, f64::INFINITY]]),
            arr2(&[[1.0, 1.1], [1.2, 1.3]]),
        )
        .unwrap_err();

        assert_eq!(
            error,
            PlaneWaveInputError::NonFiniteSpectralValue { index: vec![1, 0] },
        );
    }

    #[test]
    fn in_plane_values_report_first_non_finite_index() {
        let error = CoordinateInput::grid(
            intrinsic_coordinates(),
            arr2(&[[1_000.0, 1_100.0], [1_200.0, 1_300.0]]),
            arr2(&[[1.0, 1.1], [f64::NEG_INFINITY, f64::NAN]]),
        )
        .unwrap_err();

        assert_eq!(
            error,
            PlaneWaveInputError::NonFiniteInPlaneValue { index: vec![1, 0] },
        );
    }

    #[test]
    fn complex_inputs_are_supported() {
        let input = CoordinateInput::point(
            intrinsic_coordinates(),
            Complex64::new(1_000.0, 5.0),
            Complex64::new(1.5, -0.1),
        )
        .unwrap();

        assert_eq!(input.spectral()[()], Complex64::new(1_000.0, 5.0),);

        assert_eq!(input.in_plane()[()], Complex64::new(1.5, -0.1),);
    }

    #[test]
    fn complex_non_finite_value_is_rejected() {
        let error = CoordinateInput::point(
            intrinsic_coordinates(),
            Complex64::new(f64::NAN, 0.0),
            Complex64::new(1.5, 0.0),
        )
        .unwrap_err();

        assert_eq!(
            error,
            PlaneWaveInputError::NonFiniteSpectralValue { index: vec![] },
        );
    }

    #[test]
    fn negative_finite_values_are_structurally_valid() {
        let input =
            CoordinateInput::samples(intrinsic_coordinates(), arr1(&[-1_000.0]), arr1(&[-1.5]));

        assert!(input.is_ok());
    }

    #[test]
    fn empty_inputs_are_accepted_when_shapes_match() {
        let input =
            CoordinateInput::samples(intrinsic_coordinates(), arr1::<f64>(&[]), arr1::<f64>(&[]));

        assert!(input.is_ok());
    }

    #[test]
    fn get_point_extracts_values_and_preserves_metadata() {
        let input = CoordinateInput::incident_angle_grid(
            angle_coordinates(),
            arr2(&[[1_000.0, 1_100.0], [1_200.0, 1_300.0]]),
            arr2(&[[10.0, 20.0], [30.0, 40.0]]),
            IncidentSide::Right,
        )
        .unwrap();

        let point = input.get_point((1, 0)).unwrap();

        assert_eq!(point.coordinates(), angle_coordinates(),);
        assert_eq!(point.spectral()[()], 1_200.0);
        assert_eq!(point.in_plane()[()], 30.0);
        assert_eq!(
            point.reference(),
            CoordinateReference::IncidentSide(IncidentSide::Right,),
        );
    }

    #[test]
    fn get_point_returns_none_for_out_of_bounds_index() {
        let input = CoordinateInput::samples(
            intrinsic_coordinates(),
            arr1(&[1_000.0, 1_100.0]),
            arr1(&[1.4, 1.5]),
        )
        .unwrap();

        assert!(input.get_point(2).is_none());
    }

    #[test]
    fn raw_dimension_matches_sampled_values() {
        let input = CoordinateInput::grid(
            intrinsic_coordinates(),
            arr2(&[[1_000.0, 1_100.0], [1_200.0, 1_300.0]]),
            arr2(&[[1.0, 1.1], [1.2, 1.3]]),
        )
        .unwrap();

        assert_eq!(input.raw_dim(), Ix2(2, 2));
    }

    #[test]
    fn into_parts_preserves_all_components() {
        let spectral = arr1(&[1_000.0, 1_100.0]);

        let in_plane = arr1(&[1.4, 1.5]);

        let input =
            CoordinateInput::samples(intrinsic_coordinates(), spectral.clone(), in_plane.clone())
                .unwrap();

        let (returned_coordinates, returned_values, returned_reference) = input.into_parts();

        let (returned_spectral, returned_in_plane) = returned_values.into_parts();

        assert_eq!(returned_coordinates, intrinsic_coordinates(),);
        assert_eq!(returned_spectral, spectral);
        assert_eq!(returned_in_plane, in_plane);
        assert_eq!(returned_reference, CoordinateReference::Intrinsic,);
    }
}
