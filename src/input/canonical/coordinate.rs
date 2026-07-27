use crate::input::Polarisation;

/// Minimal canonical input required by an backend solve.
///
/// The backend solve is agnostic over the incident side, which is only needed for end-stage
/// observable computation
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CanonicalSolverInput<J> {
    coordinates: CanonicalCoordinates<J>,
    polarisation: Polarisation,
}

impl<J> CanonicalSolverInput<J> {
    /// Construct canonical solver input.
    pub(crate) const fn new(
        coordinates: CanonicalCoordinates<J>,
        polarisation: Polarisation,
    ) -> Self {
        Self {
            coordinates,
            polarisation,
        }
    }

    /// Return the canonical coordinates.
    pub(crate) fn coordinates(&self) -> &CanonicalCoordinates<J> {
        &self.coordinates
    }

    /// Return the vacuum angular wavenumber `k₀`.
    pub(crate) fn vacuum_angular_wavenumber(&self) -> &J {
        self.coordinates.vacuum_angular_wavenumber()
    }

    /// Return the conserved parallel angular wavenumber `k∥`.
    pub(crate) fn parallel_angular_wavenumber(&self) -> &J {
        self.coordinates.parallel_angular_wavenumber()
    }

    /// Return the polarisation.
    pub(crate) const fn polarisation(&self) -> Polarisation {
        self.polarisation
    }

    /// Consume the input and return its components.
    pub(crate) fn into_parts(self) -> (CanonicalCoordinates<J>, Polarisation) {
        (self.coordinates, self.polarisation)
    }
}

/// Canonical coordinates used by planar numerical backends.
///
/// Both values:
///
/// - are expressed in inverse centimetres;
/// - have compatible sampled shapes;
/// - are interpreted elementwise;
/// - contain one `(k₀, k∥)` pair per solved state.
///
/// `J` is the complete sampled algebraic representation. It may be an array,
/// a zeroth-order jet, a directional jet, or a bivariate jet.
///
/// Shape, finiteness, units, and coordinate conversion have already been
/// validated by the compilation layer.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CanonicalCoordinates<J> {
    vacuum_angular_wavenumber: J,
    parallel_angular_wavenumber: J,
}

impl<J> CanonicalCoordinates<J> {
    /// Construct canonical plane-wave coordinates.
    pub(crate) const fn new(vacuum_angular_wavenumber: J, parallel_angular_wavenumber: J) -> Self {
        Self {
            vacuum_angular_wavenumber,
            parallel_angular_wavenumber,
        }
    }

    /// Return the vacuum angular wavenumber `k₀`.
    pub(crate) fn vacuum_angular_wavenumber(&self) -> &J {
        &self.vacuum_angular_wavenumber
    }

    /// Return the conserved parallel angular wavenumber `k∥`.
    pub(crate) fn parallel_angular_wavenumber(&self) -> &J {
        &self.parallel_angular_wavenumber
    }

    /// Consume the coordinates and return `(k₀, k∥)`.
    pub(crate) fn into_parts(self) -> (J, J) {
        (
            self.vacuum_angular_wavenumber,
            self.parallel_angular_wavenumber,
        )
    }
}
