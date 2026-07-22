//! Derivative coordinates and chain-rule transformations.
//!
//! Planar backends expose derivatives with respect to physical spectral
//! coordinates and finite-layer thicknesses.
//!
//! The current isotropic kernels use squared spectral coordinates as their
//! primitive variables:
//!
//! ```text
//! k₀²
//! k∥²
//! ```
//!
//! Derivatives requested with respect to the corresponding linear coordinates
//! are obtained after the primitive backend evaluation using the exact chain
//! rule.
//!
//! Derivative order is not represented in this module. It is selected by the
//! backend method called and encoded by the resulting value type.

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum StructuralDerivativeVariable {
    ParallelWavenumber,
    Thickness(usize),
}

impl From<StructuralDerivativeVariable> for DerivativeVariable {
    fn from(val: StructuralDerivativeVariable) -> Self {
        match val {
            StructuralDerivativeVariable::ParallelWavenumber => {
                DerivativeVariable::ParallelWavenumber
            }
            StructuralDerivativeVariable::Thickness(i) => DerivativeVariable::Thickness(i),
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SpectralDerivativeVariable {
    VacuumWavenumber,
}

impl From<SpectralDerivativeVariable> for DerivativeVariable {
    fn from(val: SpectralDerivativeVariable) -> Self {
        match val {
            SpectralDerivativeVariable::VacuumWavenumber => DerivativeVariable::VacuumWavenumber,
        }
    }
}

impl std::convert::TryFrom<DerivativeVariable> for StructuralDerivativeVariable {
    type Error = &'static str;

    fn try_from(value: DerivativeVariable) -> Result<Self, Self::Error> {
        match value {
            DerivativeVariable::ParallelWavenumber => {
                Ok(StructuralDerivativeVariable::ParallelWavenumber)
            }
            DerivativeVariable::Thickness(i) => Ok(StructuralDerivativeVariable::Thickness(i)),
            DerivativeVariable::VacuumWavenumber => {
                Err("VacuumWavenumber is not a structural derivative")
            }
        }
    }
}

impl std::convert::TryFrom<DerivativeVariable> for SpectralDerivativeVariable {
    type Error = &'static str;

    fn try_from(value: DerivativeVariable) -> Result<Self, Self::Error> {
        match value {
            DerivativeVariable::ParallelWavenumber => {
                Err("ParallelWavenumber is not a spectral derivative")
            }
            DerivativeVariable::Thickness(_) => Err("Thickness is not a spectral derivative"),
            DerivativeVariable::VacuumWavenumber => {
                Ok(SpectralDerivativeVariable::VacuumWavenumber)
            }
        }
    }
}

/// Independent variable with respect to which derivatives are evaluated.
///
/// The current isotropic derivative kernels use squared vacuum and parallel
/// wavenumbers as primitive spectral variables. Linear-coordinate derivatives
/// are obtained from those primitive derivatives using an exact chain-rule
/// transformation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DerivativeVariable {
    /// Vacuum wavenumber `k₀`.
    VacuumWavenumber,

    /// Conserved parallel wavenumber `k∥`.
    ParallelWavenumber,

    /// Physical thickness of a finite layer.
    ///
    /// The index refers to the finite layers in geometric left-to-right order and
    /// excludes the two semi-infinite exterior media.
    Thickness(usize),
}

impl DerivativeVariable {
    pub fn is_spectral(self) -> bool {
        SpectralDerivativeVariable::try_from(self).is_ok()
    }

    pub fn is_structural(self) -> bool {
        StructuralDerivativeVariable::try_from(self).is_ok()
    }
}
