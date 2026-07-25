use ndarray::Dimension;

#[derive(Debug, thiserror::Error)]
pub enum TmmError<D: Dimension> {
    #[error(
        "calculations require the input arrays to be pre-arranged to the same dimension\
    got vacuum_wavenumber: {vacuum_wavenumber}, parallel_wavenumber: {parallel_wavenumber}"
    )]
    InputArraySizeMismatch {
        vacuum_wavenumber: D,
        parallel_wavenumber: D,
    },
}
