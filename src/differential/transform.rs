#[derive(Debug, thiserror::Error)]
pub enum DifferentialTransformError {
    #[error("coordinate transform is singular")]
    SingularTransform,

    #[error("coordinate derivatives have incompatible shapes")]
    ShapeMismatch,

    #[error("coordinate transform contains non-finite values")]
    NonFinite,
}

pub struct CoordinateJacobian<T> {
    // ∂k0/∂u, ∂k0/∂v
    vacuum_wavenumber: [T; 2],

    // ∂kx/∂u, ∂kx/∂v
    parallel_wavenumber: [T; 2],
}

// pub struct CoordinateHessians<T> {
//     vacuum_wavenumber: SymmetricHessian2<T>,
//     parallel_wavenumber: SymmetricHessian2<T>,
// }

// pub trait TransformSpectralDifferential<T> {
//     type Output;

//     fn transform(self, transform: &impl SecondOrderCoordinateTransform<T>) -> Self::Output;
// }
