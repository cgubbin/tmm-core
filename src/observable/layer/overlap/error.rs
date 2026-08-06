use thiserror::Error;

use crate::{FiniteLayerIndex, Polarisation, observable::LayerAggregateError};

/// Failure to construct or evaluate a pair of retained plane-wave
/// solutions.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum OverlapError {
    /// The states were compiled with different polarizations.
    #[error(
        "overlap requires matching polarizations; reference is \
         {reference:?}, comparison is {comparison:?}"
    )]
    PolarisationMismatch {
        reference: Polarisation,
        comparison: Polarisation,
    },

    /// The retained finite-layer counts differ.
    #[error(
        "reference finite-layer count {reference_count} does not match \
         comparison finite-layer count {comparison_count}"
    )]
    LayerCountMismatch {
        reference_count: usize,
        comparison_count: usize,
    },

    /// Corresponding finite layers do not occupy the same physical interval.
    #[error(
        "finite layer {index:?} has incompatible reference and comparison \
         thicknesses"
    )]
    LayerThicknessMismatch { index: FiniteLayerIndex },

    /// The two jet mappings do not assign the same meaning to derivative
    /// components.
    #[error("reference and comparison differential mappings are incompatible")]
    DifferentialMappingMismatch,

    /// A state does not retain the layer data required by pairwise
    /// observables.
    #[error("{operand} state does not retain finite-layer analysis data")]
    LayersNotRetained { operand: PairOperand },

    #[error(transparent)]
    Aggregate(LayerAggregateError),
}

/// Operand involved in a pairwise retained-state operation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PairOperand {
    Reference,
    Comparison,
}

impl std::fmt::Display for PairOperand {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reference => formatter.write_str("reference"),
            Self::Comparison => formatter.write_str("comparison"),
        }
    }
}
