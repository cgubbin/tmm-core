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
pub struct PlanarInput<I> {
    vacuum_wavenumber: I,
    parallel_wavenumber: I,
    polarisation: Polarisation,
}

impl<I> PlanarInput<I> {
    /// Construct a planar evaluation input.
    ///
    /// `vacuum_wavenumber` and `parallel_wavenumber` must use the same
    /// inverse-length unit. For sampled values, they must also have matching
    /// shapes.
    pub fn new(vacuum_wavenumber: I, parallel_wavenumber: I, polarisation: Polarisation) -> Self {
        Self {
            vacuum_wavenumber,
            parallel_wavenumber,
            polarisation,
        }
    }

    /// Return the vacuum wavenumber `k₀`.
    pub fn vacuum_wavenumber(&self) -> &I {
        &self.vacuum_wavenumber
    }

    /// Return the conserved parallel wavenumber `k∥`.
    pub fn parallel_wavenumber(&self) -> &I {
        &self.parallel_wavenumber
    }

    /// Return the polarisation.
    pub fn polarisation(&self) -> Polarisation {
        self.polarisation
    }

    /// Consume the input and return its coordinates and polarisation.
    pub fn into_parts(self) -> (I, I, Polarisation) {
        (
            self.vacuum_wavenumber,
            self.parallel_wavenumber,
            self.polarisation,
        )
    }
}

impl<R> PlanarInput<R> {
    pub fn map<J>(&self, mut map: impl FnMut(&R) -> J) -> PlanarInput<J> {
        PlanarInput::new(
            map(self.vacuum_wavenumber()),
            map(self.parallel_wavenumber()),
            self.polarisation(),
        )
    }

    pub fn clone_map<J>(&self, mut map: impl FnMut(R) -> J) -> PlanarInput<J>
    where
        R: Clone,
    {
        PlanarInput::new(
            map(self.vacuum_wavenumber().clone()),
            map(self.parallel_wavenumber().clone()),
            self.polarisation(),
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
pub struct PlaneWaveInput<I> {
    planar: PlanarInput<I>,
    incident_side: IncidentSide,
}

impl<I> PlaneWaveInput<I> {
    /// Construct a unit-amplitude plane-wave scattering input.
    pub fn new(planar: PlanarInput<I>, incident_side: IncidentSide) -> Self {
        Self {
            planar,
            incident_side,
        }
    }

    /// Return the underlying planar evaluation input.
    pub fn planar(&self) -> &PlanarInput<I> {
        &self.planar
    }

    /// Return the side from which the wave is incident.
    pub fn incident_side(&self) -> IncidentSide {
        self.incident_side
    }

    /// Consume the input and return its planar coordinates and incident side.
    pub fn into_parts(self) -> (PlanarInput<I>, IncidentSide) {
        (self.planar, self.incident_side)
    }
}

impl<R, D> PlaneWaveInput<ArrayBase<OwnedRepr<R>, D>> {
    pub(crate) fn complex_planar_input<C>(&self) -> PlanarInput<ArrayBase<OwnedRepr<C>, D>>
    where
        C: ComplexField<RealField = R>,
        C::RealField: Copy,
        D: Dimension,
    {
        self.planar().map(|values| values.mapv(C::from_real))
    }

    pub(crate) fn to_complex<C>(&self) -> PlaneWaveInput<ArrayBase<OwnedRepr<C>, D>>
    where
        C: ComplexField<RealField = R>,
        C::RealField: Copy,
        D: Dimension,
    {
        PlaneWaveInput {
            planar: self.planar().map(|values| values.mapv(C::from_real)),
            incident_side: self.incident_side,
        }
    }
}

impl<R, D> PlanarInput<ArrayBase<OwnedRepr<R>, D>> {
    pub(crate) fn to_complex<C>(&self) -> PlanarInput<ArrayBase<OwnedRepr<C>, D>>
    where
        C: ComplexField<RealField = R>,
        C::RealField: Copy,
        D: Dimension,
    {
        self.map(|values| values.mapv(C::from_real))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn planar_input_exposes_its_components() {
        let input = PlanarInput::new(1000.0, 250.0, Polarisation::TransverseElectric);

        assert_eq!(*input.vacuum_wavenumber(), 1000.0);
        assert_eq!(*input.parallel_wavenumber(), 250.0);
        assert_eq!(input.polarisation(), Polarisation::TransverseElectric);
    }

    #[test]
    fn planar_input_into_parts_preserves_values() {
        let input = PlanarInput::new(1000.0, 250.0, Polarisation::TransverseMagnetic);

        let (vacuum_wavenumber, parallel_wavenumber, polarisation) = input.into_parts();

        assert_eq!(vacuum_wavenumber, 1000.0);
        assert_eq!(parallel_wavenumber, 250.0);
        assert_eq!(polarisation, Polarisation::TransverseMagnetic);
    }

    #[test]
    fn plane_wave_input_exposes_planar_input_and_side() {
        let planar = PlanarInput::new(1000.0, 250.0, Polarisation::TransverseElectric);

        let input = PlaneWaveInput::new(planar, IncidentSide::Right);

        assert_eq!(
            input.planar().polarisation(),
            Polarisation::TransverseElectric
        );
        assert_eq!(*input.planar().vacuum_wavenumber(), 1000.0);
        assert_eq!(input.incident_side(), IncidentSide::Right);
    }

    #[test]
    fn plane_wave_input_into_parts_preserves_values() {
        let planar = PlanarInput::new(1000.0, 250.0, Polarisation::TransverseMagnetic);

        let input = PlaneWaveInput::new(planar, IncidentSide::Left);

        let (planar, side) = input.into_parts();

        assert_eq!(planar.polarisation(), Polarisation::TransverseMagnetic);
        assert_eq!(*planar.parallel_wavenumber(), 250.0);
        assert_eq!(side, IncidentSide::Left);
    }
}
