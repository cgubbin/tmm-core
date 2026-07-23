//! Inputs shared by planar electromagnetic backends.
//!
//! This module defines:
//!
//! - [`PlanarInput`], the spectral coordinate and polarisation required to
//!   evaluate an isotropic planar stack;
//! - [`PlaneWaveInput`], a unit-amplitude plane-wave scattering problem;
//! - [`Polarisation`], the TE/TM decomposition used by isotropic backends;
//! - [`IncidentSide`], the side from which the plane wave enters the stack.
//!
//! Both spectral coordinates in [`PlanarInput`] are wavenumbers and must use
//! the same inverse-length unit. The backend does not perform unit conversion
//! or implicit broadcasting.

mod algebraic;

use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, OwnedRepr};

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

/// Spectral and in-plane wavevector coordinates for a planar calculation.
///
/// Both coordinates must use the same inverse-length unit. A typical canonical
/// choice is inverse centimetres.
///
/// The vacuum wavenumber is:
///
/// ```text
/// k₀ = ω / c
/// ```
///
/// The parallel wavenumber is the conserved magnitude of the wavevector
/// component parallel to the interfaces:
///
/// ```text
/// k∥
/// ```
///
/// For an isotropic layer, the normal wavenumber satisfies:
///
/// ```text
/// κ² = ε μ k₀² - k∥²
/// ```
///
/// This type describes an evaluation point. It deliberately contains no
/// incidence direction, excitation amplitude, boundary condition, or
/// derivative request.
///
/// For sampled inputs, both coordinates must have matching shapes. The backend
/// performs no implicit broadcasting.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanarInput<C, D>
where
    D: Dimension,
{
    pub(crate) vacuum_wavenumber: ArrayBase<OwnedRepr<C>, D>,
    pub(crate) parallel_wavenumber: ArrayBase<OwnedRepr<C>, D>,
    polarisation: Polarisation,
}

impl<C, D> PlanarInput<C, D>
where
    D: Dimension,
{
    /// Construct a planar evaluation input.
    ///
    /// `vacuum_wavenumber` and `parallel_wavenumber` must use the same
    /// inverse-length unit. For sampled values, they must also have matching
    /// shapes.
    pub fn new(
        vacuum_wavenumber: ArrayBase<OwnedRepr<C>, D>,
        parallel_wavenumber: ArrayBase<OwnedRepr<C>, D>,
        polarisation: Polarisation,
    ) -> Self {
        assert_eq!(
            vacuum_wavenumber.raw_dim(),
            parallel_wavenumber.raw_dim(),
            "vacuum and parallel wavenumber arrays must have identical shapes",
        );

        Self {
            vacuum_wavenumber,
            parallel_wavenumber,
            polarisation,
        }
    }

    /// Return the vacuum wavenumber `k₀`.
    pub fn vacuum_wavenumber(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.vacuum_wavenumber
    }

    /// Return the conserved parallel wavenumber `k∥`.
    pub fn parallel_wavenumber(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.parallel_wavenumber
    }

    /// Return the polarisation.
    pub fn polarisation(&self) -> Polarisation {
        self.polarisation
    }

    pub fn from_real(
        vacuum_wavenumber: ArrayBase<OwnedRepr<C::RealField>, D>,
        parallel_wavenumber: ArrayBase<OwnedRepr<C::RealField>, D>,
        polarisation: Polarisation,
    ) -> Self
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
    pub fn into_parts(
        self,
    ) -> (
        ArrayBase<OwnedRepr<C>, D>,
        ArrayBase<OwnedRepr<C>, D>,
        Polarisation,
    ) {
        (
            self.vacuum_wavenumber,
            self.parallel_wavenumber,
            self.polarisation,
        )
    }
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

/// Unit-amplitude plane-wave scattering problem.
///
/// The backend returns complex reflection and transmission amplitude
/// coefficients for a unit-amplitude incident wave arriving from
/// [`incident_side`](Self::incident_side).
///
/// The incident amplitude is intentionally omitted. A caller may obtain the
/// reflected and transmitted field amplitudes by multiplying the returned
/// coefficients by its chosen incident amplitude.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveInput<C, D>
where
    D: Dimension,
{
    pub(crate) planar: PlanarInput<C, D>,
    pub(crate) incident_side: IncidentSide,
}

impl<C, D> PlaneWaveInput<C, D>
where
    D: Dimension,
{
    /// Construct a unit-amplitude plane-wave scattering input.
    pub fn new(planar: PlanarInput<C, D>, incident_side: IncidentSide) -> Self {
        Self {
            planar,
            incident_side,
        }
    }

    /// Return the underlying planar evaluation input.
    pub fn planar(&self) -> &PlanarInput<C, D> {
        &self.planar
    }

    /// Return the side from which the wave is incident.
    pub fn incident_side(&self) -> IncidentSide {
        self.incident_side
    }

    /// Consume the input and return its planar coordinates and incident side.
    pub fn into_parts(self) -> (PlanarInput<C, D>, IncidentSide) {
        (self.planar, self.incident_side)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array0, Array1, Array2, Ix0, Ix1, Ix2, arr0, arr1, arr2};
    use num_complex::Complex64;

    use super::{PlanarInput, Polarisation};

    type C = Complex64;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    #[test]
    fn scalar_planar_input_exposes_its_components() {
        let vacuum_wavenumber: Array0<C> = arr0(c(1000.0, 2.0));

        let parallel_wavenumber: Array0<C> = arr0(c(250.0, -1.0));

        let input: PlanarInput<C, Ix0> = PlanarInput::new(
            vacuum_wavenumber.clone(),
            parallel_wavenumber.clone(),
            Polarisation::TransverseElectric,
        );

        assert_eq!(input.vacuum_wavenumber(), &vacuum_wavenumber,);

        assert_eq!(input.parallel_wavenumber(), &parallel_wavenumber,);

        assert_eq!(input.polarisation(), Polarisation::TransverseElectric,);
    }

    #[test]
    fn sampled_planar_input_exposes_its_components() {
        let vacuum_wavenumber: Array1<C> = arr1(&[c(1000.0, 0.0), c(1100.0, 0.0), c(1200.0, 0.0)]);

        let parallel_wavenumber: Array1<C> = arr1(&[c(100.0, 0.0), c(200.0, 0.0), c(300.0, 0.0)]);

        let input: PlanarInput<C, Ix1> = PlanarInput::new(
            vacuum_wavenumber.clone(),
            parallel_wavenumber.clone(),
            Polarisation::TransverseMagnetic,
        );

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

        let input: PlanarInput<C, Ix2> = PlanarInput::new(
            vacuum_wavenumber.clone(),
            parallel_wavenumber.clone(),
            Polarisation::TransverseElectric,
        );

        assert_eq!(
            input.vacuum_wavenumber().raw_dim(),
            input.parallel_wavenumber().raw_dim(),
        );

        assert_eq!(input.vacuum_wavenumber(), &vacuum_wavenumber,);

        assert_eq!(input.parallel_wavenumber(), &parallel_wavenumber,);
    }

    #[test]
    #[should_panic(expected = "vacuum and parallel wavenumber arrays must have identical shapes")]
    fn planar_input_rejects_mismatched_vector_shapes() {
        let vacuum_wavenumber: Array1<C> = arr1(&[c(1000.0, 0.0), c(1100.0, 0.0)]);

        let parallel_wavenumber: Array1<C> = arr1(&[c(100.0, 0.0), c(200.0, 0.0), c(300.0, 0.0)]);

        let _ = PlanarInput::new(
            vacuum_wavenumber,
            parallel_wavenumber,
            Polarisation::TransverseElectric,
        );
    }

    #[test]
    #[should_panic(expected = "vacuum and parallel wavenumber arrays must have identical shapes")]
    fn planar_input_rejects_mismatched_grid_shapes() {
        let vacuum_wavenumber: Array2<C> = Array2::from_elem((2, 3), c(1000.0, 0.0));

        let parallel_wavenumber: Array2<C> = Array2::from_elem((3, 2), c(100.0, 0.0));

        let _ = PlanarInput::new(
            vacuum_wavenumber,
            parallel_wavenumber,
            Polarisation::TransverseMagnetic,
        );
    }

    #[test]
    fn planar_input_into_parts_preserves_owned_values() {
        let vacuum_wavenumber: Array1<C> = arr1(&[c(1000.0, 1.0), c(1100.0, 2.0)]);

        let parallel_wavenumber: Array1<C> = arr1(&[c(100.0, -1.0), c(200.0, -2.0)]);

        let input: PlanarInput<C, Ix1> = PlanarInput::new(
            vacuum_wavenumber.clone(),
            parallel_wavenumber.clone(),
            Polarisation::TransverseMagnetic,
        );

        let (returned_vacuum, returned_parallel, returned_polarisation) = input.into_parts();

        assert_eq!(returned_vacuum, vacuum_wavenumber,);

        assert_eq!(returned_parallel, parallel_wavenumber,);

        assert_eq!(returned_polarisation, Polarisation::TransverseMagnetic,);
    }

    #[test]
    fn planar_input_clone_is_independent() {
        let input: PlanarInput<C, Ix1> = PlanarInput::new(
            arr1(&[c(1000.0, 0.0), c(1100.0, 0.0)]),
            arr1(&[c(100.0, 0.0), c(200.0, 0.0)]),
            Polarisation::TransverseElectric,
        );

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
