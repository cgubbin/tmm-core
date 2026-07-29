use nalgebra::RealField;
use ndarray::{Array, Array1, Array2, Dimension, IntoDimension, Ix0, Ix1, Ix2, arr0};

use super::{IncidentSide, PlaneWaveCoordinates, PlaneWaveInputError, Polarisation};

/// A scalar plane-wave input.
pub type PlaneWavePoint<R> = PlaneWaveInput<R, Ix0>;

/// Caller-facing sampled plane-wave input.
///
/// Every element describes one plane-wave state using a spectral coordinate
/// and an in-plane coordinate. Both arrays must have exactly the same shape;
/// this type performs no implicit broadcasting.
///
/// Coordinate values are interpreted according to [`PlaneWaveCoordinates`]
/// and converted to backend coordinates during input compilation.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveInput<R, D>
where
    D: Dimension,
{
    input: PlaneWaveCoordinatesInput<R, D>,
    polarisation: Polarisation,
    incident_side: IncidentSide,
}

impl<R, D> PlaneWaveInput<R, D>
where
    D: Dimension,
{
    /// Construct and validate a plane-wave input.
    ///
    /// # Errors
    ///
    /// Returns an error when:
    ///
    /// - the spectral and in-plane arrays have different shapes;
    /// - any spectral value is not finite;
    /// - any in-plane value is not finite.
    ///
    /// Arrays that could be broadcast to a common shape are still rejected.
    /// Broadcasting, when desired, must be performed explicitly by the caller.
    pub fn new(
        coordinates: PlaneWaveCoordinates,
        spectral: Array<R, D>,
        in_plane: Array<R, D>,
        polarisation: Polarisation,
        incident_side: IncidentSide,
    ) -> Result<Self, PlaneWaveInputError>
    where
        R: RealField,
    {
        Ok(Self {
            input: PlaneWaveCoordinatesInput::new(coordinates, spectral, in_plane)?,
            polarisation,
            incident_side,
        })
    }

    /// Return the caller-facing coordinate system.
    pub fn coordinates(&self) -> PlaneWaveCoordinates {
        self.input.coordinates()
    }

    /// Return the supplied spectral coordinate values.
    pub fn spectral(&self) -> &Array<R, D> {
        self.input.spectral()
    }

    /// Return the supplied in-plane coordinate values.
    pub fn in_plane(&self) -> &Array<R, D> {
        self.input.in_plane()
    }

    /// Return the requested polarisation.
    pub fn polarisation(&self) -> Polarisation {
        self.polarisation
    }

    /// Return the side from which the stack is illuminated.
    pub fn incident_side(&self) -> IncidentSide {
        self.incident_side
    }

    /// Extract one sampled state as a scalar plane-wave input.
    ///
    /// Returns `None` when `index` lies outside the sampled array.
    pub fn get_point<I>(&self, index: I) -> Option<PlaneWavePoint<R>>
    where
        R: Copy,
        I: IntoDimension<Dim = D>,
    {
        Some(PlaneWavePoint {
            input: self.input.get_point(index)?,
            polarisation: self.polarisation,
            incident_side: self.incident_side,
        })
    }

    /// Consume the input and return its internal components.
    pub(crate) fn into_parts(
        self,
    ) -> (PlaneWaveCoordinatesInput<R, D>, Polarisation, IncidentSide) {
        (self.input, self.polarisation, self.incident_side)
    }
}

impl<R> PlaneWaveInput<R, Ix0>
where
    R: RealField,
{
    /// Construct a scalar plane-wave input.
    pub fn point(
        coordinates: PlaneWaveCoordinates,
        spectral: R,
        in_plane: R,
        polarisation: Polarisation,
        incident_side: IncidentSide,
    ) -> Result<Self, PlaneWaveInputError> {
        Self::new(
            coordinates,
            arr0(spectral),
            arr0(in_plane),
            polarisation,
            incident_side,
        )
    }
}

impl<R> PlaneWaveInput<R, Ix1>
where
    R: RealField,
{
    /// Construct a one-dimensional sequence of plane-wave inputs.
    pub fn samples(
        coordinates: PlaneWaveCoordinates,
        spectral: Array1<R>,
        in_plane: Array1<R>,
        polarisation: Polarisation,
        incident_side: IncidentSide,
    ) -> Result<Self, PlaneWaveInputError> {
        Self::new(coordinates, spectral, in_plane, polarisation, incident_side)
    }
}

impl<R> PlaneWaveInput<R, Ix2>
where
    R: RealField,
{
    /// Construct a two-dimensional grid of plane-wave inputs.
    ///
    /// Both arrays must already have the complete grid shape. Column and row
    /// vectors are not implicitly broadcast.
    pub fn grid(
        coordinates: PlaneWaveCoordinates,
        spectral: Array2<R>,
        in_plane: Array2<R>,
        polarisation: Polarisation,
        incident_side: IncidentSide,
    ) -> Result<Self, PlaneWaveInputError> {
        Self::new(coordinates, spectral, in_plane, polarisation, incident_side)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PlaneWaveCoordinatesInput<R, D>
where
    D: Dimension,
{
    coordinates: PlaneWaveCoordinates,
    values: PlaneWaveCoordinateValues<R, D>,
}

impl<R, D> PlaneWaveCoordinatesInput<R, D>
where
    D: Dimension,
{
    /// Construct and validate a plane-wave coordinate input.
    ///
    /// # Errors
    ///
    /// Returns an error when:
    ///
    /// - the spectral and in-plane arrays have different shapes;
    /// - any spectral value is not finite;
    /// - any in-plane value is not finite.
    ///
    /// Arrays that could be broadcast to a common shape are still rejected.
    /// Broadcasting, when desired, must be performed explicitly by the caller.
    pub fn new(
        coordinates: PlaneWaveCoordinates,
        spectral: Array<R, D>,
        in_plane: Array<R, D>,
    ) -> Result<Self, PlaneWaveInputError>
    where
        R: RealField,
    {
        if spectral.raw_dim() != in_plane.raw_dim() {
            return Err(PlaneWaveInputError::ShapeMismatch {
                spectral: spectral.raw_dim().into_dyn(),
                in_plane: in_plane.raw_dim().into_dyn(),
            });
        }

        if let Some(index) = first_non_finite_index(&spectral) {
            return Err(PlaneWaveInputError::NonFiniteSpectralValue { index });
        }

        if let Some(index) = first_non_finite_index(&in_plane) {
            return Err(PlaneWaveInputError::NonFiniteInPlaneValue { index });
        }

        Ok(Self {
            coordinates,
            values: PlaneWaveCoordinateValues::new(spectral, in_plane),
        })
    }

    /// Return the caller-facing coordinate system.
    pub fn coordinates(&self) -> PlaneWaveCoordinates {
        self.coordinates
    }

    /// Return the supplied spectral coordinate values.
    pub fn spectral(&self) -> &Array<R, D> {
        self.values.spectral()
    }

    /// Return the supplied in-plane coordinate values.
    pub fn in_plane(&self) -> &Array<R, D> {
        self.values.in_plane()
    }

    /// Extract one sampled state as a scalar plane-wave input.
    ///
    /// Returns `None` when `index` lies outside the sampled array.
    pub fn get_point<I>(&self, index: I) -> Option<PlaneWaveCoordinatesInput<R, Ix0>>
    where
        R: Copy,
        I: IntoDimension<Dim = D>,
    {
        Some(PlaneWaveCoordinatesInput {
            coordinates: self.coordinates,
            values: self.values.get_point(index)?,
        })
    }

    /// Consume the input and return its internal components.
    pub(crate) fn into_parts(self) -> (PlaneWaveCoordinates, PlaneWaveCoordinateValues<R, D>) {
        (self.coordinates, self.values)
    }
}

/// Paired spectral and in-plane coordinate values.
///
/// This type is internal to input compilation. Shape and finiteness invariants
/// are established by [`PlaneWaveInput::new`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct PlaneWaveCoordinateValues<R, D>
where
    D: Dimension,
{
    spectral: Array<R, D>,
    in_plane: Array<R, D>,
}

impl<R, D> PlaneWaveCoordinateValues<R, D>
where
    D: Dimension,
{
    pub(super) fn new(spectral: Array<R, D>, in_plane: Array<R, D>) -> Self {
        Self { spectral, in_plane }
    }

    pub(super) fn spectral(&self) -> &Array<R, D> {
        &self.spectral
    }

    pub(super) fn in_plane(&self) -> &Array<R, D> {
        &self.in_plane
    }

    fn get_point<I>(&self, index: I) -> Option<PlaneWaveCoordinateValues<R, Ix0>>
    where
        R: Copy,
        I: IntoDimension<Dim = D>,
    {
        let index = index.into_dimension();

        Some(PlaneWaveCoordinateValues {
            spectral: arr0(*self.spectral.get(index.clone())?),
            in_plane: arr0(*self.in_plane.get(index)?),
        })
    }

    pub(crate) fn raw_dim(&self) -> D {
        self.spectral.raw_dim()
    }

    pub(crate) fn into_parts(self) -> (Array<R, D>, Array<R, D>) {
        (self.spectral, self.in_plane)
    }
}

fn first_non_finite_index<R, D>(values: &Array<R, D>) -> Option<Vec<usize>>
where
    R: RealField,
    D: Dimension,
{
    values.indexed_iter().find_map(|(index, value)| {
        (!value.is_finite()).then(|| index.into_dimension().as_array_view().to_owned().to_vec())
    })
}

#[cfg(test)]
mod tests {
    use ndarray::{IxDyn, arr1, arr2};
    use tmm_units::InverseLengthUnit;

    use crate::input::{InPlaneCoordinate, SpectralCoordinate};

    use super::*;

    fn coordinates() -> PlaneWaveCoordinates {
        PlaneWaveCoordinates::new(
            SpectralCoordinate::VacuumWavenumber(InverseLengthUnit::PerCentimetre),
            InPlaneCoordinate::EffectiveIndex,
        )
    }

    #[test]
    fn point_preserves_values_and_metadata() {
        let input = PlaneWaveInput::point(
            coordinates(),
            1000.0,
            1.5,
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        )
        .unwrap();

        assert_eq!(input.coordinates(), coordinates());
        assert_eq!(input.spectral()[()], 1000.0);
        assert_eq!(input.in_plane()[()], 1.5);
        assert_eq!(input.polarisation(), Polarisation::TransverseElectric,);
        assert_eq!(input.incident_side(), IncidentSide::Left);
    }

    #[test]
    fn samples_preserve_values_and_metadata() {
        let spectral = arr1(&[1000.0, 1100.0]);
        let in_plane = arr1(&[1.4, 1.5]);

        let input = PlaneWaveInput::samples(
            coordinates(),
            spectral.clone(),
            in_plane.clone(),
            Polarisation::TransverseMagnetic,
            IncidentSide::Right,
        )
        .unwrap();

        assert_eq!(input.spectral(), &spectral);
        assert_eq!(input.in_plane(), &in_plane);
        assert_eq!(input.polarisation(), Polarisation::TransverseMagnetic,);
        assert_eq!(input.incident_side(), IncidentSide::Right);
    }

    #[test]
    fn samples_reject_different_lengths() {
        let result = PlaneWaveInput::samples(
            coordinates(),
            arr1(&[1000.0, 1100.0]),
            arr1(&[1.4]),
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        );

        assert_eq!(
            result,
            Err(PlaneWaveInputError::ShapeMismatch {
                spectral: IxDyn(&[2]),
                in_plane: IxDyn(&[1]),
            }),
        );
    }

    #[test]
    fn grid_rejects_broadcastable_but_unequal_shapes() {
        let result = PlaneWaveInput::grid(
            coordinates(),
            arr2(&[[1000.0], [1100.0]]),
            arr2(&[[1.0, 1.1]]),
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        );

        assert_eq!(
            result,
            Err(PlaneWaveInputError::ShapeMismatch {
                spectral: IxDyn(&[2, 1]),
                in_plane: IxDyn(&[1, 2]),
            }),
        );
    }

    #[test]
    fn spectral_values_report_first_non_finite_index() {
        let result = PlaneWaveInput::grid(
            coordinates(),
            arr2(&[[1000.0, 1100.0], [f64::NAN, f64::INFINITY]]),
            arr2(&[[1.0, 1.1], [1.2, 1.3]]),
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        );

        assert_eq!(
            result,
            Err(PlaneWaveInputError::NonFiniteSpectralValue { index: vec![1, 0] },),
        );
    }

    #[test]
    fn in_plane_values_report_first_non_finite_index() {
        let result = PlaneWaveInput::grid(
            coordinates(),
            arr2(&[[1000.0, 1100.0], [1200.0, 1300.0]]),
            arr2(&[[1.0, 1.1], [f64::NEG_INFINITY, f64::NAN]]),
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        );

        assert_eq!(
            result,
            Err(PlaneWaveInputError::NonFiniteInPlaneValue { index: vec![1, 0] },),
        );
    }

    #[test]
    fn negative_finite_values_are_not_rejected_at_structural_validation() {
        let input = PlaneWaveInput::samples(
            coordinates(),
            arr1(&[-1000.0]),
            arr1(&[-1.5]),
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        );

        assert!(input.is_ok());
    }

    #[test]
    fn empty_inputs_are_accepted_when_shapes_match() {
        let input = PlaneWaveInput::samples(
            coordinates(),
            arr1::<f64>(&[]),
            arr1::<f64>(&[]),
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        );

        assert!(input.is_ok());
    }

    #[test]
    fn get_point_extracts_values_and_preserves_metadata() {
        let input = PlaneWaveInput::grid(
            coordinates(),
            arr2(&[[1000.0, 1100.0], [1200.0, 1300.0]]),
            arr2(&[[1.0, 1.1], [1.2, 1.3]]),
            Polarisation::TransverseMagnetic,
            IncidentSide::Right,
        )
        .unwrap();

        let point = input.get_point((1, 0)).unwrap();

        assert_eq!(point.coordinates(), coordinates());
        assert_eq!(point.spectral()[()], 1200.0);
        assert_eq!(point.in_plane()[()], 1.2);
        assert_eq!(point.polarisation(), Polarisation::TransverseMagnetic,);
        assert_eq!(point.incident_side(), IncidentSide::Right);
    }

    #[test]
    fn get_point_returns_none_for_out_of_bounds_index() {
        let input = PlaneWaveInput::samples(
            coordinates(),
            arr1(&[1000.0, 1100.0]),
            arr1(&[1.4, 1.5]),
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        )
        .unwrap();

        assert!(input.get_point(2).is_none());
    }

    #[test]
    fn into_parts_preserves_all_components() {
        let spectral = arr1(&[1000.0, 1100.0]);
        let in_plane = arr1(&[1.4, 1.5]);

        let input = PlaneWaveInput::samples(
            coordinates(),
            spectral.clone(),
            in_plane.clone(),
            Polarisation::TransverseMagnetic,
            IncidentSide::Right,
        )
        .unwrap();

        let (input, polarisation, incident_side) = input.into_parts();

        let (returned_coordinates, values) = input.into_parts();
        let (returned_spectral, returned_in_plane) = values.into_parts();

        assert_eq!(returned_coordinates, coordinates());
        assert_eq!(returned_spectral, spectral);
        assert_eq!(returned_in_plane, in_plane);
        assert_eq!(polarisation, Polarisation::TransverseMagnetic,);
        assert_eq!(incident_side, IncidentSide::Right);
    }
}
