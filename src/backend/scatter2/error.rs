#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Scatter2Error {
    #[error(
        "thickness derivative requested for layer {requested}, \
         but the stack contains {layer_count} finite layers"
    )]
    ThicknessLayerOutOfBounds {
        requested: usize,
        layer_count: usize,
    },

    #[error("scattering composition encountered a singular feedback denominator")]
    SingularStarProduct,
}
