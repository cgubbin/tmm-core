use nalgebra::RealField;
use ndarray::{
    Array, Array0, Array1, Array2, ArrayBase, Data, Dimension, IntoDimension, Ix0, Ix1, Ix2,
    OwnedRepr, arr0,
};

use super::{IncidentSide, PlaneWaveCoordinates, PlaneWaveInputError, Polarisation};

pub type PlaneWavePoint<R> = PlaneWaveInput<R, Ix0>;
pub type PlaneWaveSamples<R> = PlaneWaveInput<R, Ix1>;
pub type PlaneWaveGrid<R> = PlaneWaveInput<R, Ix2>;

/// Caller-facing plane-wave input.
///
/// The spectral and in-plane values:
///
/// - use the parameterisations described by [`PlaneWaveCoordinates`];
/// - have identical dimensions and shapes;
/// - are interpreted elementwise;
/// - have already been broadcast by the caller, if broadcasting was required.
///
/// Each corresponding pair defines one plane-wave state.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveInput<R, D>
where
    D: Dimension,
{
    coordinates: PlaneWaveCoordinates,
    values: PlaneWaveCoordinateValues<R, D>,
    polarisation: Polarisation,
    incident_side: IncidentSide,
}

impl<R, D> PlaneWaveInput<R, D>
where
    D: Dimension,
{
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
            polarisation,
            incident_side,
        })
    }

    pub fn coordinates(&self) -> PlaneWaveCoordinates {
        self.coordinates
    }

    pub fn spectral(&self) -> &Array<R, D> {
        self.values.spectral()
    }

    pub fn in_plane(&self) -> &Array<R, D> {
        self.values.in_plane()
    }

    pub fn polarisation(&self) -> Polarisation {
        self.polarisation
    }

    pub fn incident_side(&self) -> IncidentSide {
        self.incident_side
    }

    pub fn get_point<I>(&self, index: &I) -> Option<PlaneWavePoint<R>>
    where
        R: Copy,
        I: IntoDimension<Dim = D> + Clone,
    {
        Some(PlaneWavePoint {
            coordinates: self.coordinates,
            values: self.values.get_point(index)?,
            polarisation: self.polarisation,
            incident_side: self.incident_side,
        })
    }

    pub fn into_parts(
        self,
    ) -> (
        PlaneWaveCoordinates,
        PlaneWaveCoordinateValues<R, D>,
        Polarisation,
        IncidentSide,
    ) {
        (
            self.coordinates,
            self.values,
            self.polarisation,
            self.incident_side,
        )
    }
}

impl<R> PlaneWaveInput<R, Ix0>
where
    R: RealField,
{
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
pub struct PlaneWaveCoordinateValues<R, D>
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
    pub fn new(spectral: Array<R, D>, in_plane: Array<R, D>) -> Self {
        Self { spectral, in_plane }
    }

    pub fn spectral(&self) -> &Array<R, D> {
        &self.spectral
    }

    pub fn in_plane(&self) -> &Array<R, D> {
        &self.in_plane
    }

    pub fn get_point<I>(&self, index: &I) -> Option<PlaneWaveCoordinateValues<R, Ix0>>
    where
        R: Copy,
        I: IntoDimension<Dim = D> + Clone,
    {
        let index = index.clone().into_dimension();
        Some(PlaneWaveCoordinateValues {
            spectral: arr0(*self.spectral.get(index.clone())?),
            in_plane: arr0(*self.in_plane.get(index.clone())?),
        })
    }

    pub fn raw_dim(&self) -> D
    where
        D: Clone,
    {
        self.spectral.raw_dim()
    }

    pub fn into_parts(self) -> (Array<R, D>, Array<R, D>) {
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
    use ndarray::{arr1, arr2};
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
    fn point_constructs_scalar_arrays() {
        let input = PlaneWaveInput::point(
            coordinates(),
            1000.0,
            1.5,
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        )
        .unwrap();

        assert_eq!(input.spectral()[()], 1000.0);
        assert_eq!(input.in_plane()[()], 1.5);
    }

    #[test]
    fn samples_accept_equal_shapes() {
        let input = PlaneWaveInput::samples(
            coordinates(),
            arr1(&[1000.0, 1100.0]),
            arr1(&[1.4, 1.5]),
            Polarisation::TransverseMagnetic,
            IncidentSide::Right,
        );

        assert!(input.is_ok());
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

        assert!(matches!(
            result,
            Err(PlaneWaveInputError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn grid_rejects_different_shapes_even_when_broadcastable() {
        let result = PlaneWaveInput::grid(
            coordinates(),
            arr2(&[[1000.0], [1100.0]]),
            arr2(&[[1.0, 1.1]]),
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        );

        assert!(matches!(
            result,
            Err(PlaneWaveInputError::ShapeMismatch { .. })
        ));
    }

    #[test]
    fn spectral_values_must_be_finite() {
        let result = PlaneWaveInput::samples(
            coordinates(),
            arr1(&[1000.0, f64::NAN]),
            arr1(&[1.4, 1.5]),
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        );

        assert!(matches!(
            result,
            Err(PlaneWaveInputError::NonFiniteSpectralValue { .. })
        ));
    }

    #[test]
    fn in_plane_values_must_be_finite() {
        let result = PlaneWaveInput::samples(
            coordinates(),
            arr1(&[1000.0, 1100.0]),
            arr1(&[1.4, f64::INFINITY]),
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        );

        assert!(matches!(
            result,
            Err(PlaneWaveInputError::NonFiniteInPlaneValue { .. })
        ));
    }
}
