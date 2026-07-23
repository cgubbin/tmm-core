#![allow(dead_code)]

pub(crate) mod algebra;
pub mod backend;
pub(crate) mod crystallise;
mod differential;
pub(crate) mod field;
pub mod material;
mod scalar;
pub mod stack;
mod tensor;

pub use backend::{IncidentSide, Polarisation};
// pub use backend::{
//     ArrayJet, ArrayJetFirst, DerivativeVariable, IncidentSide, OutgoingModeResidualBackend,
//     PlanarInput, PlaneWaveAmplitudeDifferential, PlaneWaveAmplitudes, PlaneWaveBackend,
//     PlaneWaveFieldBackend, PlaneWaveInput, PlaneWavePower, PlaneWavePowerDifferential,
//     PlaneWaveResponse, PlaneWaveResponseDerivatives, PlaneWaveResponseDifferential, Polarisation,
//     SpectralDerivativeVariable, StructuralDerivativeVariable,
// };

// pub use backend::scatter2::Scatter2;
// // pub use backend::transfer2::Transfer2;
pub use material::{
    Constant, DerivativeOrder, DifferentiableMaterial, DifferentiableMeromorphicMaterial,
    EvaluateDifferentiableMaterial, EvaluateDifferentiableMeromorphicMaterial, EvaluateMaterial,
    EvaluateMeromorphicMaterial, Material, MeromorphicMaterial, Sampled, Scalar,
};

pub use scalar::ComplexScalar;

// pub use stack::{
//     AnalyticalMaterialStack, DifferentiableMaterialStack, Layer, MaterialStack,
//     MeromorphicMaterialStack, Stack, Thickness, ValidationConfig,
// };
