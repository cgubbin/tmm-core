use nalgebra::ComplexField;
use ndarray::{Array, Dimension};

use crate::TmmError;

pub type PlaneWavePoint<R> = PlaneWaveInput<R, ndarray::Ix0>;

#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveInput<R, D>
where
    D: Dimension,
{
    coordinates: CanonicalCoordinates<R, D>,
    polarisation: Polarisation,
    incident_side: IncidentSide,
}

impl<C, D> PlaneWaveInput<C, D>
where
    D: Dimension,
{
    /// Construct a planar evaluation input.
    ///
    /// `vacuum_wavenumber` and `parallel_wavenumber` must use the same
    /// inverse-length unit. For sampled values, they must also have matching
    /// shapes.
    pub fn new(
        vacuum_wavenumber: Array<C, D>,
        parallel_wavenumber: Array<C, D>,
        polarisation: Polarisation,
        incident_side: IncidentSide,
    ) -> Result<Self, TmmError<D>> {
        Ok(Self {
            coordinates: CanonicalCoordinates::new(vacuum_wavenumber, parallel_wavenumber)?,
            polarisation,
            incident_side,
        })
    }

    /// Return the canonical coordinates.
    pub fn coordinates(&self) -> &CanonicalCoordinates<C, D> {
        &self.coordinates
    }

    /// Return the vacuum wavenumber `k₀`.
    pub fn vacuum_wavenumber(&self) -> &Array<C, D> {
        self.coordinates.vacuum_wavenumber()
    }

    /// Return the conserved parallel wavenumber `k∥`.
    pub fn parallel_wavenumber(&self) -> &Array<C, D> {
        self.coordinates.parallel_wavenumber()
    }

    /// Return the polarisation.
    pub fn polarisation(&self) -> Polarisation {
        self.polarisation
    }

    /// Return the incident side.
    pub fn incident_side(&self) -> IncidentSide {
        self.incident_side
    }

    pub fn from_real(
        vacuum_wavenumber: Array<C::RealField, D>,
        parallel_wavenumber: Array<C::RealField, D>,
        polarisation: Polarisation,
        incident_side: IncidentSide,
    ) -> Result<Self, TmmError<D>>
    where
        C: ComplexField + Copy,
        C::RealField: Copy,
        D: Dimension,
    {
        Self::new(
            vacuum_wavenumber.mapv(C::from_real),
            parallel_wavenumber.mapv(C::from_real),
            polarisation,
            incident_side,
        )
    }

    pub fn point(&self, index: &D) -> Option<PlaneWavePoint<C>>
    where
        C: Copy,
    {
        Some(PlaneWaveInput {
            coordinates: self.coordinates.point(index)?,
            polarisation: self.polarisation,
            incident_side: self.incident_side,
        })
    }

    /// Consume the input and return its coordinates and polarisation.
    pub fn into_components(self) -> (CanonicalCoordinates<C, D>, Polarisation, IncidentSide) {
        (self.coordinates, self.polarisation, self.incident_side)
    }

    /// Consume the input and return its flattened components.
    pub fn into_parts(self) -> (Array<C, D>, Array<C, D>, Polarisation, IncidentSide) {
        let (coordinates, polarisation, incident_side) = self.into_components();

        let (vacuum_wavenumber, parallel_wavenumber) = coordinates.into_parts();

        (
            vacuum_wavenumber,
            parallel_wavenumber,
            polarisation,
            incident_side,
        )
    }
}

/// Minimal plane wave input required to solve a problem
#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalInput<R, D>
where
    D: Dimension,
{
    coordinates: CanonicalCoordinates<R, D>,
    polarisation: Polarisation,
}

impl<C, D> CanonicalInput<C, D>
where
    D: Dimension,
{
    /// Construct a planar evaluation input.
    ///
    /// `vacuum_wavenumber` and `parallel_wavenumber` must use the same
    /// inverse-length unit. For sampled values, they must also have matching
    /// shapes.
    pub fn new(
        vacuum_wavenumber: Array<C, D>,
        parallel_wavenumber: Array<C, D>,
        polarisation: Polarisation,
    ) -> Result<Self, TmmError<D>> {
        Ok(Self {
            coordinates: CanonicalCoordinates::new(vacuum_wavenumber, parallel_wavenumber)?,
            polarisation,
        })
    }

    /// Return the canonical coordinates.
    pub fn coordinates(&self) -> &CanonicalCoordinates<C, D> {
        &self.coordinates
    }

    /// Return the vacuum wavenumber `k₀`.
    pub fn vacuum_wavenumber(&self) -> &Array<C, D> {
        self.coordinates.vacuum_wavenumber()
    }

    /// Return the conserved parallel wavenumber `k∥`.
    pub fn parallel_wavenumber(&self) -> &Array<C, D> {
        self.coordinates.parallel_wavenumber()
    }

    /// Return the polarisation.
    pub fn polarisation(&self) -> Polarisation {
        self.polarisation
    }

    pub fn point(&self, index: &D) -> Option<CanonicalInput<C, ndarray::Ix0>>
    where
        C: Copy,
    {
        Some(CanonicalInput {
            coordinates: self.coordinates.point(index)?,
            polarisation: self.polarisation,
        })
    }

    pub fn from_real(
        vacuum_wavenumber: Array<C::RealField, D>,
        parallel_wavenumber: Array<C::RealField, D>,
        polarisation: Polarisation,
    ) -> Result<Self, TmmError<D>>
    where
        C: ComplexField + Copy,
        C::RealField: Copy,
        D: Dimension,
    {
        Self::new(
            vacuum_wavenumber.mapv(C::from_real),
            parallel_wavenumber.mapv(C::from_real),
            polarisation,
        )
    }

    /// Consume the input and return its coordinates and polarisation.
    pub fn into_components(self) -> (CanonicalCoordinates<C, D>, Polarisation) {
        (self.coordinates, self.polarisation)
    }

    /// Consume the input and return its flattened components.
    pub fn into_parts(self) -> (Array<C, D>, Array<C, D>, Polarisation) {
        let (coordinates, polarisation) = self.into_components();

        let (vacuum_wavenumber, parallel_wavenumber) = coordinates.into_parts();

        (vacuum_wavenumber, parallel_wavenumber, polarisation)
    }
}

/// Polarisation supported by isotropic planar backends.
///
/// In an isotropic stratified system, transverse-electric and
/// transverse-magnetic fields decouple into independent scalar problems.
///
/// An anisotropic backend may require a different input representation because
/// this TE/TM decomposition does not generally remain valid.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Polarisation {
    /// Transverse-electric polarisation.
    ///
    /// The electric field is perpendicular to the plane of incidence.
    TransverseElectric,

    /// Transverse-magnetic polarisation.
    ///
    /// The magnetic field is perpendicular to the plane of incidence.
    TransverseMagnetic,
}

/// Side from which a plane wave is incident on a planar stack.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum IncidentSide {
    /// Incidence from the first exterior medium towards the final exterior
    /// medium.
    Left,

    /// Incidence from the final exterior medium towards the first exterior
    /// medium.
    Right,
}

/// Canonical coordinates used by the core solver.
///
/// Both arrays:
///
/// - are expressed in inverse centimetres;
/// - have identical shape;
/// - are interpreted elementwise;
/// - contain one `(k0, kx)` pair per solved plane-wave state.
#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalCoordinates<R, D>
where
    D: Dimension,
{
    vacuum_wavenumber: Array<R, D>,
    parallel_wavenumber: Array<R, D>,
}

impl<R, D> CanonicalCoordinates<R, D>
where
    D: Dimension,
{
    pub(crate) fn new_unchecked(
        vacuum_wavenumber: Array<R, D>,
        parallel_wavenumber: Array<R, D>,
    ) -> Self {
        debug_assert_eq!(vacuum_wavenumber.shape(), parallel_wavenumber.shape());

        Self {
            vacuum_wavenumber,
            parallel_wavenumber,
        }
    }

    pub fn new(
        vacuum_wavenumber: Array<R, D>,
        parallel_wavenumber: Array<R, D>,
    ) -> Result<Self, TmmError<D>> {
        if vacuum_wavenumber.raw_dim() != parallel_wavenumber.raw_dim() {
            return Err(TmmError::InputArraySizeMismatch {
                vacuum_wavenumber: vacuum_wavenumber.raw_dim(),
                parallel_wavenumber: parallel_wavenumber.raw_dim(),
            });
        }

        Ok(Self {
            vacuum_wavenumber,
            parallel_wavenumber,
        })
    }

    pub fn point(&self, index: &D) -> Option<CanonicalCoordinates<R, ndarray::Ix0>>
    where
        R: Copy,
    {
        Some(CanonicalCoordinates {
            vacuum_wavenumber: ndarray::arr0(*self.vacuum_wavenumber.get(index.clone())?),
            parallel_wavenumber: ndarray::arr0(*self.parallel_wavenumber.get(index.clone())?),
        })
    }

    pub fn vacuum_wavenumber(&self) -> &Array<R, D> {
        &self.vacuum_wavenumber
    }

    pub fn parallel_wavenumber(&self) -> &Array<R, D> {
        &self.parallel_wavenumber
    }

    pub fn into_parts(self) -> (Array<R, D>, Array<R, D>) {
        (self.vacuum_wavenumber, self.parallel_wavenumber)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array0, Array1, Array2, Ix0, Ix1, Ix2, arr0, arr1, arr2};
    use num_complex::Complex64;

    use super::{CanonicalInput, Polarisation};

    type C = Complex64;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    #[test]
    fn scalar_planar_input_exposes_its_components() {
        let vacuum_wavenumber: Array0<C> = arr0(c(1000.0, 2.0));

        let parallel_wavenumber: Array0<C> = arr0(c(250.0, -1.0));

        let input: CanonicalInput<C, Ix0> = CanonicalInput::new(
            vacuum_wavenumber.clone(),
            parallel_wavenumber.clone(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

        assert_eq!(input.vacuum_wavenumber(), &vacuum_wavenumber,);

        assert_eq!(input.parallel_wavenumber(), &parallel_wavenumber,);

        assert_eq!(input.polarisation(), Polarisation::TransverseElectric,);
    }

    #[test]
    fn sampled_planar_input_exposes_its_components() {
        let vacuum_wavenumber: Array1<C> = arr1(&[c(1000.0, 0.0), c(1100.0, 0.0), c(1200.0, 0.0)]);

        let parallel_wavenumber: Array1<C> = arr1(&[c(100.0, 0.0), c(200.0, 0.0), c(300.0, 0.0)]);

        let input: CanonicalInput<C, Ix1> = CanonicalInput::new(
            vacuum_wavenumber.clone(),
            parallel_wavenumber.clone(),
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

        assert_eq!(input.vacuum_wavenumber(), &vacuum_wavenumber,);

        assert_eq!(input.parallel_wavenumber(), &parallel_wavenumber,);

        assert_eq!(input.polarisation(), Polarisation::TransverseMagnetic,);
    }

    #[test]
    fn grid_planar_input_accepts_matching_shapes() {
        let vacuum_wavenumber: Array2<C> = arr2(&[
            [c(1000.0, 0.0), c(1100.0, 0.0)],
            [c(1200.0, 0.0), c(1300.0, 0.0)],
        ]);

        let parallel_wavenumber: Array2<C> = arr2(&[
            [c(100.0, 0.0), c(200.0, 0.0)],
            [c(300.0, 0.0), c(400.0, 0.0)],
        ]);

        let input: CanonicalInput<C, Ix2> = CanonicalInput::new(
            vacuum_wavenumber.clone(),
            parallel_wavenumber.clone(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

        assert_eq!(
            input.vacuum_wavenumber().raw_dim(),
            input.parallel_wavenumber().raw_dim(),
        );

        assert_eq!(input.vacuum_wavenumber(), &vacuum_wavenumber,);

        assert_eq!(input.parallel_wavenumber(), &parallel_wavenumber,);
    }

    #[test]
    fn planar_input_rejects_mismatched_vector_shapes() {
        let vacuum_wavenumber: Array1<C> = arr1(&[c(1000.0, 0.0), c(1100.0, 0.0)]);

        let parallel_wavenumber: Array1<C> = arr1(&[c(100.0, 0.0), c(200.0, 0.0), c(300.0, 0.0)]);

        let result = CanonicalInput::new(
            vacuum_wavenumber,
            parallel_wavenumber,
            Polarisation::TransverseElectric,
        );

        assert!(result.is_err());
    }

    #[test]
    fn planar_input_rejects_mismatched_grid_shapes() {
        let vacuum_wavenumber: Array2<C> = Array2::from_elem((2, 3), c(1000.0, 0.0));

        let parallel_wavenumber: Array2<C> = Array2::from_elem((3, 2), c(100.0, 0.0));

        let result = CanonicalInput::new(
            vacuum_wavenumber,
            parallel_wavenumber,
            Polarisation::TransverseMagnetic,
        );

        assert!(result.is_err());
    }

    #[test]
    fn planar_input_into_parts_preserves_owned_values() {
        let vacuum_wavenumber: Array1<C> = arr1(&[c(1000.0, 1.0), c(1100.0, 2.0)]);

        let parallel_wavenumber: Array1<C> = arr1(&[c(100.0, -1.0), c(200.0, -2.0)]);

        let input: CanonicalInput<C, Ix1> = CanonicalInput::new(
            vacuum_wavenumber.clone(),
            parallel_wavenumber.clone(),
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

        let (returned_vacuum, returned_parallel, returned_polarisation) = input.into_parts();

        assert_eq!(returned_vacuum, vacuum_wavenumber,);

        assert_eq!(returned_parallel, parallel_wavenumber,);

        assert_eq!(returned_polarisation, Polarisation::TransverseMagnetic,);
    }

    #[test]
    fn planar_input_clone_is_independent() {
        let input: CanonicalInput<C, Ix1> = CanonicalInput::new(
            arr1(&[c(1000.0, 0.0), c(1100.0, 0.0)]),
            arr1(&[c(100.0, 0.0), c(200.0, 0.0)]),
            Polarisation::TransverseElectric,
        )
        .unwrap();

        let clone = input.clone();

        assert_eq!(clone, input);

        let (cloned_vacuum, cloned_parallel, cloned_polarisation) = clone.into_parts();

        assert_eq!(cloned_vacuum, arr1(&[c(1000.0, 0.0), c(1100.0, 0.0),]),);

        assert_eq!(cloned_parallel, arr1(&[c(100.0, 0.0), c(200.0, 0.0),]),);

        assert_eq!(cloned_polarisation, Polarisation::TransverseElectric,);

        assert_eq!(
            input.vacuum_wavenumber(),
            &arr1(&[c(1000.0, 0.0), c(1100.0, 0.0),]),
        );
    }
}
