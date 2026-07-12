use crate::{ComplexScalar, backend::PlanarInput};

use ndarray::{ArrayBase, Dimension, OwnedRepr};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DerivativeOrder {
    Value,
    First,
    Second,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DerivativeRequest {
    Value,
    Derivative {
        variable: DerivativeVariable,
        order: DerivativeOrder,
    },
}

/// Independent variable with respect to which derivatives are evaluated.
///
/// Squared spectral variables are the primitive variables used internally by
/// the current isotropic 2×2 backend. Linear-variable derivatives are obtained
/// by an exact chain-rule transformation.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash)]
pub enum DerivativeVariable {
    /// Vacuum wavenumber `k₀`.
    VacuumWavenumber,

    /// Squared vacuum wavenumber `k₀²`.
    VacuumWavenumberSquared,

    /// Parallel wavenumber `k∥`.
    ParallelWavenumber,

    /// Squared parallel wavenumber `k∥²`.
    ParallelWavenumberSquared,

    /// Physical thickness of the finite layer at the given index.
    ///
    /// The index refers to the finite layers in propagation order and excludes
    /// the two semi-infinite exterior media.
    Thickness(usize),
}

impl DerivativeVariable {
    /// Return the primitive variable used by the backend differentiation
    /// kernel.
    ///
    /// Linear spectral coordinates map to their squared coordinates.
    /// Thickness derivatives are already primitive.
    pub fn primitive(self) -> Self {
        match self {
            Self::VacuumWavenumber => Self::VacuumWavenumberSquared,
            Self::ParallelWavenumber => Self::ParallelWavenumberSquared,
            variable => variable,
        }
    }

    /// Return `true` when this variable can be evaluated directly by the
    /// backend derivative kernel.
    pub fn is_primitive(self) -> bool {
        self == self.primitive()
    }
}

#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct ChainRule<C> {
    pub(crate) first: C,
    pub(crate) second: C,
}

impl DerivativeVariable {
    pub(crate) fn chain_rule<C: ComplexScalar, D: Dimension>(
        self,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Option<ChainRule<ArrayBase<OwnedRepr<C>, D>>> {
        let two = C::one() + C::one();

        match self {
            Self::VacuumWavenumber => Some(ChainRule {
                first: planar.vacuum_wavenumber.mapv(|w| two * w),
                second: planar.vacuum_wavenumber.mapv(|_| two),
            }),

            Self::ParallelWavenumber => Some(ChainRule {
                first: planar.parallel_wavenumber.mapv(|w| two * w),
                second: planar.parallel_wavenumber.mapv(|_| two),
            }),

            _ => None,
        }
    }
}
