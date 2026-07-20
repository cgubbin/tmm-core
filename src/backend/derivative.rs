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

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{ComplexScalar, backend::PlanarInput};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum StructuralDerivativeVariable {
    ParallelWavenumber,
    ParallelWavenumberSquared,
    Thickness(usize),
}

impl From<StructuralDerivativeVariable> for DerivativeVariable {
    fn from(val: StructuralDerivativeVariable) -> Self {
        match val {
            StructuralDerivativeVariable::ParallelWavenumber => {
                DerivativeVariable::ParallelWavenumber
            }
            StructuralDerivativeVariable::ParallelWavenumberSquared => {
                DerivativeVariable::ParallelWavenumberSquared
            }
            StructuralDerivativeVariable::Thickness(i) => DerivativeVariable::Thickness(i),
        }
    }
}

impl StructuralDerivativeVariable {
    /// Return the primitive variable evaluated directly by the isotropic
    /// derivative kernel.
    ///
    /// Linear spectral variables map to their squared counterparts. Thickness
    /// is already a primitive coordinate.
    pub fn primitive(self) -> Self {
        match self {
            Self::ParallelWavenumber => Self::ParallelWavenumberSquared,
            variable => variable,
        }
    }

    /// Return whether this variable can be evaluated directly by the primitive
    /// isotropic derivative kernel.
    pub fn is_primitive(self) -> bool {
        self == self.primitive()
    }

    /// Construct the chain-rule coefficients needed to transform derivatives
    /// from the primitive coordinate to this requested coordinate.
    ///
    /// For a linear coordinate `y` whose primitive coordinate is `x = y²`,
    ///
    /// ```text
    /// dx/dy   = 2y
    /// d²x/dy² = 2
    /// ```
    ///
    /// Returns `None` when no transformation is required.
    pub(crate) fn chain_rule<C, D>(
        self,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Option<ChainRule<ArrayBase<OwnedRepr<C>, D>>>
    where
        C: ComplexScalar,
        D: Dimension,
    {
        let two = C::one() + C::one();

        match self {
            Self::ParallelWavenumber => Some(ChainRule::new(
                planar.parallel_wavenumber().mapv(|parallel| two * parallel),
                planar.parallel_wavenumber().mapv(|_| two),
            )),

            Self::ParallelWavenumberSquared | Self::Thickness(_) => None,
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum SpectralDerivativeVariable {
    VacuumWavenumber,
    VacuumWavenumberSquared,
}

impl From<SpectralDerivativeVariable> for DerivativeVariable {
    fn from(val: SpectralDerivativeVariable) -> Self {
        match val {
            SpectralDerivativeVariable::VacuumWavenumber => DerivativeVariable::VacuumWavenumber,
            SpectralDerivativeVariable::VacuumWavenumberSquared => {
                DerivativeVariable::VacuumWavenumberSquared
            }
        }
    }
}

impl SpectralDerivativeVariable {
    /// Return the primitive variable evaluated directly by the isotropic
    /// derivative kernel.
    ///
    /// Linear spectral variables map to their squared counterparts. Thickness
    /// is already a primitive coordinate.
    pub fn primitive(self) -> Self {
        match self {
            Self::VacuumWavenumber => Self::VacuumWavenumberSquared,
            variable => variable,
        }
    }

    /// Return whether this variable can be evaluated directly by the primitive
    /// isotropic derivative kernel.
    pub fn is_primitive(self) -> bool {
        self == self.primitive()
    }

    /// Construct the chain-rule coefficients needed to transform derivatives
    /// from the primitive coordinate to this requested coordinate.
    ///
    /// For a linear coordinate `y` whose primitive coordinate is `x = y²`,
    ///
    /// ```text
    /// dx/dy   = 2y
    /// d²x/dy² = 2
    /// ```
    ///
    /// Returns `None` when no transformation is required.
    pub(crate) fn chain_rule<C, D>(
        self,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Option<ChainRule<ArrayBase<OwnedRepr<C>, D>>>
    where
        C: ComplexScalar,
        D: Dimension,
    {
        let two = C::one() + C::one();

        match self {
            Self::VacuumWavenumber => Some(ChainRule::new(
                planar.vacuum_wavenumber().mapv(|k0| two * k0),
                planar.vacuum_wavenumber().mapv(|_| two),
            )),

            Self::VacuumWavenumberSquared => None,
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
            DerivativeVariable::ParallelWavenumberSquared => {
                Ok(StructuralDerivativeVariable::ParallelWavenumberSquared)
            }
            DerivativeVariable::Thickness(i) => Ok(StructuralDerivativeVariable::Thickness(i)),
            DerivativeVariable::VacuumWavenumber => {
                Err("VacuumWavenumber is not a structural derivative")
            }
            DerivativeVariable::VacuumWavenumberSquared => {
                Err("VacuumWavenumberSquared is not a structural derivative")
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
            DerivativeVariable::ParallelWavenumberSquared => {
                Err("ParallelWavenumberSquared is not a spectral derivative")
            }
            DerivativeVariable::Thickness(_) => Err("Thickness is not a spectral derivative"),
            DerivativeVariable::VacuumWavenumber => {
                Ok(SpectralDerivativeVariable::VacuumWavenumber)
            }
            DerivativeVariable::VacuumWavenumberSquared => {
                Ok(SpectralDerivativeVariable::VacuumWavenumberSquared)
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

    /// Squared vacuum wavenumber `k₀²`.
    VacuumWavenumberSquared,

    /// Conserved parallel wavenumber `k∥`.
    ParallelWavenumber,

    /// Squared conserved parallel wavenumber `k∥²`.
    ParallelWavenumberSquared,

    /// Physical thickness of a finite layer.
    ///
    /// The index refers to the finite layers in geometric left-to-right order and
    /// excludes the two semi-infinite exterior media.
    Thickness(usize),
}

impl DerivativeVariable {
    /// Return the primitive variable evaluated directly by the isotropic
    /// derivative kernel.
    ///
    /// Linear spectral variables map to their squared counterparts. Thickness
    /// is already a primitive coordinate.
    pub fn primitive(self) -> Self {
        match self {
            Self::VacuumWavenumber => Self::VacuumWavenumberSquared,
            Self::ParallelWavenumber => Self::ParallelWavenumberSquared,
            variable => variable,
        }
    }

    /// Return whether this variable can be evaluated directly by the primitive
    /// isotropic derivative kernel.
    pub fn is_primitive(self) -> bool {
        self == self.primitive()
    }

    pub fn is_spectral(self) -> bool {
        SpectralDerivativeVariable::try_from(self).is_ok()
    }

    pub fn is_structural(self) -> bool {
        StructuralDerivativeVariable::try_from(self).is_ok()
    }

    /// Construct the chain-rule coefficients needed to transform derivatives
    /// from the primitive coordinate to this requested coordinate.
    ///
    /// For a linear coordinate `y` whose primitive coordinate is `x = y²`,
    ///
    /// ```text
    /// dx/dy   = 2y
    /// d²x/dy² = 2
    /// ```
    ///
    /// Returns `None` when no transformation is required.
    pub(crate) fn chain_rule<C, D>(
        self,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Option<ChainRule<ArrayBase<OwnedRepr<C>, D>>>
    where
        C: ComplexScalar,
        D: Dimension,
    {
        let two = C::one() + C::one();

        match self {
            Self::VacuumWavenumber => Some(ChainRule::new(
                planar.vacuum_wavenumber().mapv(|k0| two * k0),
                planar.vacuum_wavenumber().mapv(|_| two),
            )),

            Self::ParallelWavenumber => Some(ChainRule::new(
                planar.parallel_wavenumber().mapv(|parallel| two * parallel),
                planar.parallel_wavenumber().mapv(|_| two),
            )),

            Self::VacuumWavenumberSquared
            | Self::ParallelWavenumberSquared
            | Self::Thickness(_) => None,
        }
    }
}

/// First and second derivatives of a primitive coordinate with respect to a
/// requested coordinate.
///
/// If `x` is the primitive coordinate and `y` is the requested coordinate,
/// this stores:
///
/// ```text
/// first  = dx/dy
/// second = d²x/dy²
/// ```
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ChainRule<R> {
    pub(crate) first: R,
    pub(crate) second: R,
}

impl<R> ChainRule<R> {
    /// Construct a chain-rule transformation.
    pub(crate) fn new(first: R, second: R) -> Self {
        Self { first, second }
    }

    /// Return `dx/dy`.
    pub(crate) fn first(&self) -> &R {
        &self.first
    }

    /// Return `d²x/dy²`.
    pub(crate) fn second(&self) -> &R {
        &self.second
    }

    /// Consume the transformation and return its coefficients.
    pub(crate) fn into_parts(self) -> (R, R) {
        (self.first, self.second)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr0;
    use num_complex::Complex64;

    use super::*;
    use crate::backend::Polarisation;

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn scalar_input(
        vacuum_wavenumber: f64,
        parallel_wavenumber: f64,
    ) -> PlanarInput<ndarray::Array0<C>> {
        PlanarInput::new(
            arr0(c(vacuum_wavenumber)),
            arr0(c(parallel_wavenumber)),
            Polarisation::TransverseElectric,
        )
    }

    #[test]
    fn linear_vacuum_wavenumber_maps_to_squared_primitive() {
        assert_eq!(
            DerivativeVariable::VacuumWavenumber.primitive(),
            DerivativeVariable::VacuumWavenumberSquared
        );
    }

    #[test]
    fn linear_parallel_wavenumber_maps_to_squared_primitive() {
        assert_eq!(
            DerivativeVariable::ParallelWavenumber.primitive(),
            DerivativeVariable::ParallelWavenumberSquared
        );
    }

    #[test]
    fn squared_and_thickness_variables_are_primitive() {
        assert!(DerivativeVariable::VacuumWavenumberSquared.is_primitive());
        assert!(DerivativeVariable::ParallelWavenumberSquared.is_primitive());
        assert!(DerivativeVariable::Thickness(3).is_primitive());
    }

    #[test]
    fn vacuum_wavenumber_chain_rule_has_expected_coefficients() {
        let input = scalar_input(3.0, 4.0);

        let rule = DerivativeVariable::VacuumWavenumber
            .chain_rule(&input)
            .unwrap();

        assert_eq!(rule.first()[()], c(6.0));
        assert_eq!(rule.second()[()], c(2.0));
    }

    #[test]
    fn parallel_wavenumber_chain_rule_has_expected_coefficients() {
        let input = scalar_input(3.0, 4.0);

        let rule = DerivativeVariable::ParallelWavenumber
            .chain_rule(&input)
            .unwrap();

        assert_eq!(rule.first()[()], c(8.0));
        assert_eq!(rule.second()[()], c(2.0));
    }

    #[test]
    fn primitive_variables_require_no_chain_rule() {
        let input = scalar_input(3.0, 4.0);

        assert!(
            DerivativeVariable::VacuumWavenumberSquared
                .chain_rule(&input)
                .is_none()
        );

        assert!(
            DerivativeVariable::ParallelWavenumberSquared
                .chain_rule(&input)
                .is_none()
        );

        assert!(
            DerivativeVariable::Thickness(1)
                .chain_rule(&input)
                .is_none()
        );
    }
}
