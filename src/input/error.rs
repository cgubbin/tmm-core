use ndarray::IxDyn;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PlaneWaveInputError {
    #[error("spectral shape {spectral:?} does not match in-plane shape {in_plane:?}")]
    ShapeMismatch { spectral: IxDyn, in_plane: IxDyn },

    #[error("spectral input contains a non-finite value at index {index:?}")]
    NonFiniteSpectralValue { index: Vec<usize> },

    #[error("in-plane input contains a non-finite value at index {index:?}")]
    NonFiniteInPlaneValue { index: Vec<usize> },

    #[error("incident-angle coordinates require an incident-side reference")]
    IncidentReferenceRequired,

    #[error("an incident-side reference is only valid for incident-angle coordinates")]
    UnexpectedIncidentReference,
}
