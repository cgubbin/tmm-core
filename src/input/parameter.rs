/// Physical variable assigned to one jet slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DerivativeParameter {
    Spectral,

    InPlane,

    LayerThickness { layer: usize },
}

/// The numerical information requested from the solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveRequest {
    /// Compute values only.
    Value,

    /// Compute the value and first derivative with respect to one
    /// caller-facing parameter.
    First { parameter: DerivativeParameter },

    /// Compute the value, first derivative, and second derivative with respect
    /// to one caller-facing parameter.
    Second { parameter: DerivativeParameter },

    /// Compute first derivatives with respect to the supplied spectral and
    /// in-plane coordinates.
    CoordinateGradient,

    /// Compute the full Hessian with respect to the supplied spectral and
    /// in-plane coordinates.
    CoordinateHessian,
}
