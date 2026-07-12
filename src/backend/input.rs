/// Polarisation supported by isotropic planar backends.
///
/// In an isotropic stratified system, transverse-electric and
/// transverse-magnetic fields decouple into independent scalar problems.
///
/// A future anisotropic backend may use a different input type because this
/// decomposition does not generally remain valid.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum Polarisation {
    /// Transverse-electric polarisation.
    ///
    /// The electric field is transverse to the plane of incidence.
    TransverseElectric,

    /// Transverse-magnetic polarisation.
    ///
    /// The magnetic field is transverse to the plane of incidence.
    TransverseMagnetic,
}

/// Spectral and in-plane wavevector coordinates for a planar calculation.
///
/// Both coordinates use the same inverse-length unit. A typical choice is
/// inverse centimetres.
///
/// The vacuum wavenumber is
///
/// ```text
/// k₀ = ω / c
/// ```
///
/// and the parallel wavenumber is the conserved wavevector component parallel
/// to the interfaces:
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
/// This type deliberately contains no incidence direction, amplitude,
/// boundary condition, or derivative request.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanarInput<I> {
    pub(crate) vacuum_wavenumber: I,
    pub(crate) parallel_wavenumber: I,
    pub(crate) polarisation: Polarisation,
}

impl<I> PlanarInput<I> {
    /// Construct a planar evaluation input.
    ///
    /// `vacuum_wavenumber` and `parallel_wavenumber` must use the same
    /// inverse-length unit and, for sampled inputs, must have matching shapes.
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

    /// Return the requested polarisation.
    pub fn polarisation(&self) -> Polarisation {
        self.polarisation
    }

    /// Decompose the input into its constituent values.
    pub fn into_parts(self) -> (I, I, Polarisation) {
        (
            self.vacuum_wavenumber,
            self.parallel_wavenumber,
            self.polarisation,
        )
    }
}

/// Physical plane-wave scattering problem.
///
/// The backend returns reflection and transmission amplitude coefficients for
/// a unit-amplitude incident wave from `incident_side`.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveInput<I> {
    planar: PlanarInput<I>,
    incident_side: IncidentSide,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum IncidentSide {
    Left,
    Right,
}

impl<I> PlaneWaveInput<I> {
    /// Construct a plane-wave scattering input.
    pub fn new(planar: PlanarInput<I>, incident_side: IncidentSide) -> Self {
        Self {
            planar,
            incident_side,
        }
    }

    /// Return the underlying planar spectral input.
    pub fn planar(&self) -> &PlanarInput<I> {
        &self.planar
    }

    /// Return the incident side.
    pub fn incident_side(&self) -> IncidentSide {
        self.incident_side
    }

    /// Consume the input and return its components.
    pub fn into_parts(self) -> (PlanarInput<I>, IncidentSide) {
        (self.planar, self.incident_side)
    }
}
