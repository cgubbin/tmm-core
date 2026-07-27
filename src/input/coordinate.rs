use tmm_units::{AngleUnit, AngularFrequencyUnit, FrequencyUnit, InverseLengthUnit, LengthUnit};

/// Spectral coordinate used to parameterise a plane-wave input.
///
/// Values supplied for this coordinate are converted internally to the
/// canonical spectral coordinate used by the selected backend.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SpectralCoordinate {
    /// Vacuum angular wavenumber
    ///
    /// `k₀ = 2π / λ`.
    VacuumAngularWavenumber(InverseLengthUnit),

    /// Spectroscopic vacuum wavenumber
    ///
    /// `ν̃ = 1 / λ`.
    VacuumWavenumber(InverseLengthUnit),

    /// Ordinary frequency `f`.
    Frequency(FrequencyUnit),

    /// Angular frequency
    ///
    /// `ω = 2πf`.
    AngularFrequency(AngularFrequencyUnit),

    /// Vacuum wavelength `λ`.
    VacuumWavelength(LengthUnit),
}

/// In-plane coordinate used to parameterise a plane-wave input.
///
/// Effective index and incidence angle are coupled to the spectral coordinate.
/// Incidence angle also requires the refractive index of the incident exterior
/// medium.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InPlaneCoordinate {
    /// Parallel angular wavenumber `k∥`.
    ParallelAngularWavenumber(InverseLengthUnit),

    /// Parallel cyclic wavenumber `k∥ / 2π`.
    ParallelWavenumber(InverseLengthUnit),

    /// Effective index
    ///
    /// `n_eff = k∥ / k₀`.
    EffectiveIndex,

    /// Angle of incidence in the incident exterior medium.
    IncidentAngle(AngleUnit),
}

/// Coordinate system used for a plane-wave input.
///
/// The two coordinates jointly describe each sampled plane-wave state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PlaneWaveCoordinates {
    /// Spectral coordinate supplied by the caller.
    pub spectral: SpectralCoordinate,

    /// In-plane coordinate supplied by the caller.
    pub in_plane: InPlaneCoordinate,
}

impl PlaneWaveCoordinates {
    /// Construct a plane-wave coordinate system.
    pub const fn new(spectral: SpectralCoordinate, in_plane: InPlaneCoordinate) -> Self {
        Self { spectral, in_plane }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinates_preserve_coordinate_choices() {
        let spectral = SpectralCoordinate::VacuumWavelength(LengthUnit::Nanometre);

        let in_plane = InPlaneCoordinate::IncidentAngle(AngleUnit::Degree);

        let coordinates = PlaneWaveCoordinates::new(spectral, in_plane);

        assert_eq!(coordinates.spectral, spectral);
        assert_eq!(coordinates.in_plane, in_plane);
    }
}
