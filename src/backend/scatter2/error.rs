#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum Scatter2Error {
    #[error("scattering composition encountered a singular feedback denominator")]
    SingularStarProduct,
}
