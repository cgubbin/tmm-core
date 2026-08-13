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
pub struct CanonicalCoordinates<J> {
    vacuum_angular_wavenumber: J,
    parallel_angular_wavenumber: J,
}

impl<J> CanonicalCoordinates<J> {
    /// Construct canonical plane-wave coordinates.
    pub const fn new(vacuum_angular_wavenumber: J, parallel_angular_wavenumber: J) -> Self {
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
