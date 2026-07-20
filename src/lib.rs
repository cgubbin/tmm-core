#![allow(dead_code)]

pub mod backend;
pub mod material;
mod scalar;
pub mod stack;
mod tensor;

pub use backend::{
    DerivativeVariable, DifferentiablePlaneWaveBackend, IncidentSide, OutgoingModeResidualBackend,
    PlanarInput, PlaneWaveBackend, PlaneWaveInput, PlaneWaveResponse, PlaneWaveResponseDerivatives,
    PlaneWaveResponseDifferential, Polarisation, SpectralDerivativeVariable,
    StructuralDerivativeVariable,
};

pub use backend::scatter2::Scatter2;
pub use backend::transfer2::Transfer2;
pub use material::{
    Constant, DerivativeOrder, DifferentiableMaterial, DifferentiableMeromorphicMaterial,
    EvaluateDifferentiableMaterial, EvaluateDifferentiableMeromorphicMaterial, EvaluateMaterial,
    EvaluateMeromorphicMaterial, Material, MeromorphicMaterial, Sampled, Scalar, SpectralVariable,
};

pub use scalar::ComplexScalar;

pub use stack::{
    DifferentiableMaterialStack, Layer, MaterialStack, Stack, Thickness, ValidationConfig,
};
