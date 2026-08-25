use lamina_units::{AngleUnit, InverseLengthUnit, SpectralCoordinate};

/// In-plane coordinate used to parameterise a plane-wave input.
///
/// Values are converted internally to the conserved parallel angular
/// wavenumber. When derivatives are requested with respect to
/// [`Parameter::InPlane`], they are derivatives with respect to this
/// caller-facing coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum InPlaneCoordinate {
    /// Parallel angular wavenumber `k∥`.
    ParallelAngularWavenumber(InverseLengthUnit),

    /// Parallel cyclic wavenumber `k∥ / 2π`.
    ParallelWavenumber(InverseLengthUnit),

    /// Effective index
    ///
    /// `n_eff = k∥ / k₀`, where `k₀` is the vacuum angular wavenumber.
    EffectiveIndex,

    /// Angle of incidence in the referenced exterior medium.
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ReferenceRequirement {
    Intrinsic,
    IncidentSide,
}

impl Coordinates {
    /// Construct a plane-wave coordinate system.
    pub const fn new(spectral: SpectralCoordinate, in_plane: InPlaneCoordinate) -> Self {
        Self { spectral, in_plane }
    }

    /// Return the caller-facing in-plane coordinate.
    pub const fn in_plane(&self) -> InPlaneCoordinate {
        self.in_plane
    }

    /// Return the caller-facing spectral coordinate.
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

    use lamina_units::LengthUnit;

    #[test]
    fn coordinates_preserve_coordinate_choices() {
        let spectral = SpectralCoordinate::VacuumWavelength(LengthUnit::Nanometre);

        let in_plane = InPlaneCoordinate::IncidentAngle(AngleUnit::Degree);

        let coordinates = Coordinates::new(spectral, in_plane);

        assert_eq!(coordinates.spectral(), spectral);
        assert_eq!(coordinates.in_plane(), in_plane);
    }
}
