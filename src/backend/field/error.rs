#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum PlaneWaveFieldError<R> {
    #[error("finite-layer wave count {actual} does not match stack layer count {expected}")]
    LayerCountMismatch { expected: usize, actual: usize },

    #[error("requested finite layer {requested} is outside stack with {layer_count} layers")]
    LayerOutOfBounds {
        requested: usize,
        layer_count: usize,
    },

    #[error("field distance must be finite and non-negative, got {distance:?}")]
    InvalidExteriorDistance { distance: R },

    #[error("layer offset {offset:?} is outside [0, {thickness:?}] for layer {layer}")]
    InvalidLayerOffset {
        layer: usize,
        offset: R,
        thickness: R,
    },

    #[error("uniform field sampling requires at least one point")]
    EmptyUniformSampling,

    #[error("one-point layer sampling cannot include both distinct boundaries")]
    AmbiguousSinglePointLayerSampling,

    #[error("incident power flux is zero, negative, or non-finite")]
    InvalidIncidentFlux,

    #[error("spectral derivative function called, but no spectral derivative data found")]
    ExpectedSpectralDerivatives,

    #[error("structural derivative function called, but no structural derivative data found")]
    ExpectedStructuralDerivatives,

    #[error("second derivative method called with no second derivative information")]
    MissingSecondDerivatives,

    #[error("power derivatives required but not supplied")]
    MissingPowerDerivatives,

    #[error("field reconstruction produced a non-finite value or derivative")]
    NonFiniteFieldQuantity,
}
