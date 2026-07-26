use tmm_units::{AngleUnit, AngularFrequencyUnit, FrequencyUnit, InverseLengthUnit, LengthUnit};

/// The caller-facing spectral coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpectralCoordinate {
    /// Vacuum angular wavenumber `k₀ = 2π / λ`.
    VacuumAngularWavenumber(InverseLengthUnit),

    /// Spectroscopic vacuum wavenumber `ν̃ = 1 / λ`.
    VacuumWavenumber(InverseLengthUnit),

    /// Ordinary frequency `f`.
    Frequency(FrequencyUnit),

    /// Angular frequency `ω = 2πf`.
    AngularFrequency(AngularFrequencyUnit),

    /// Vacuum wavelength `λ`.
    VacuumWavelength(LengthUnit),
}

/// The caller-facing in-plane coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InPlaneCoordinate {
    /// Parallel angular wavenumber `k∥`.
    ParallelAngularWavenumber(InverseLengthUnit),

    /// Parallel cyclic wavenumber `k∥ / 2π`.
    ParallelWavenumber(InverseLengthUnit),

    /// Effective index `n_eff = k∥ / k₀`.
    EffectiveIndex,

    /// Angle of incidence in the incident medium.
    IncidentAngle(AngleUnit),
}

/// A caller-facing pair of spectral and in-plane coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlaneWaveCoordinates {
    pub spectral: SpectralCoordinate,
    pub in_plane: InPlaneCoordinate,
}

impl PlaneWaveCoordinates {
    pub const fn new(spectral: SpectralCoordinate, in_plane: InPlaneCoordinate) -> Self {
        Self { spectral, in_plane }
    }
}

// impl SpectralCoordinate {
//     pub(crate) fn to_canonical<J>(
//         self,
//         value: J,
//     ) -> Result<J, SpectralTransformError>
//     where
//         J: CoordinateJet,
//     {
//         todo!()
//     }
// }
