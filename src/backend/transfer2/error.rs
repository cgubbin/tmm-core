/// Entry of a 2×2 transfer matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Transfer2Entry {
    M11,
    M12,
    M21,
    M22,
}

impl std::fmt::Display for Transfer2Entry {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::M11 => formatter.write_str("m11"),
            Self::M12 => formatter.write_str("m12"),
            Self::M21 => formatter.write_str("m21"),
            Self::M22 => formatter.write_str("m22"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Transfer2Error {
    #[error(
        "layer {layer} produced non-finite transfer entry {entry} at sampled \
         index {index:?}; use the scattering-matrix backend for optically \
         thick, strongly absorbing, or strongly evanescent stacks"
    )]
    NonFiniteLayerMatrix {
        layer: usize,
        entry: Transfer2Entry,
        index: Vec<usize>,
    },

    #[error(
        "the transfer matrix became non-finite after layer {layer} in entry \
         {entry} at sampled index {index:?}; use the scattering-matrix backend \
         for optically thick, strongly absorbing, or strongly evanescent stacks"
    )]
    NonFiniteAccumulation {
        layer: usize,
        entry: Transfer2Entry,
        index: Vec<usize>,
    },

    #[error(
        "the completed transfer matrix contains non-finite entry {entry} at \
         sampled index {index:?}; use the scattering-matrix backend for \
         optically thick, strongly absorbing, or strongly evanescent stacks"
    )]
    NonFiniteFinalMatrix {
        entry: Transfer2Entry,
        index: Vec<usize>,
    },
}
