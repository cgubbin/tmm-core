use ndarray::IxDyn;

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum PlaneWaveInputError {
    #[error("spectral shape {spectral:?} does not match in-plane shape {in_plane:?}")]
    ShapeMismatch { spectral: IxDyn, in_plane: IxDyn },

    #[error("spectral input contains a non-finite value at index {index:?}")]
    NonFiniteSpectralValue { index: Vec<usize> },

    #[error("in-plane input contains a non-finite value at index {index:?}")]
    NonFiniteInPlaneValue { index: Vec<usize> },
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum SpectralTransformError {
    #[error("vacuum wavelength must be strictly positive")]
    NonPositiveWavelength,

    #[error("spectral transformation produced a non-finite value")]
    NonFiniteResult,

    #[error("spectral coordinate is outside its supported domain")]
    OutsideDomain,
}

#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum InPlaneTransformError {
    #[error("incident-angle coordinates require an incident refractive index")]
    MissingIncidentRefractiveIndex,

    #[error("incident refractive index is invalid for this transformation")]
    InvalidIncidentRefractiveIndex,

    #[error("in-plane transformation produced a non-finite value")]
    NonFiniteResult,

    #[error("in-plane coordinate is outside its supported domain")]
    OutsideDomain,
}
