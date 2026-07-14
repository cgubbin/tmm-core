#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Transfer2Error {
    #[error(
        "thickness derivative requested for layer {requested}, \
         but the stack contains {layer_count} finite layers"
    )]
    ThicknessLayerOutOfBounds {
        requested: usize,
        layer_count: usize,
    },
}
