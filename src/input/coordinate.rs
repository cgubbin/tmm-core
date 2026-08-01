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

impl InPlaneCoordinate {
    pub(crate) const fn reference_requirement(self) -> ReferenceRequirement {
        match self {
            Self::IncidentAngle(_) => ReferenceRequirement::IncidentSide,
            _ => ReferenceRequirement::Intrinsic,
        }
    }
}

/// Coordinate system used for a plane-wave input.
///
/// The two coordinates jointly describe each sampled plane-wave state.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Coordinates {
    /// Spectral coordinate supplied by the caller.
    spectral: SpectralCoordinate,

    /// In-plane coordinate supplied by the caller.
    in_plane: InPlaneCoordinate,
}

pub(crate) enum ReferenceRequirement {
    Intrinsic,
    IncidentSide,
}

impl Coordinates {
    /// Construct a plane-wave coordinate system.
    pub const fn new(spectral: SpectralCoordinate, in_plane: InPlaneCoordinate) -> Self {
        Self { spectral, in_plane }
    }

    pub const fn in_plane(&self) -> InPlaneCoordinate {
        self.in_plane
    }

    pub const fn spectral(&self) -> SpectralCoordinate {
        self.spectral
    }

    pub(crate) const fn reference_requirement(self) -> ReferenceRequirement {
        self.in_plane().reference_requirement()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn coordinates_preserve_coordinate_choices() {
        let spectral = SpectralCoordinate::VacuumWavelength(LengthUnit::Nanometre);

        let in_plane = InPlaneCoordinate::IncidentAngle(AngleUnit::Degree);

        let coordinates = Coordinates::new(spectral, in_plane);

        assert_eq!(coordinates.spectral, spectral);
        assert_eq!(coordinates.in_plane, in_plane);
    }
}
