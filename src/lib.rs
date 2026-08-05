#![allow(dead_code)]
// #![allow(unused_imports)]

pub(crate) mod algebra;
pub mod backend;
pub(crate) mod derivative_parts;
mod differential;
mod domain;
mod error;
mod evaluate;
pub mod field;
mod input;
pub mod material;
mod observable;
mod parameter;
mod response;
// mod sampling;
mod projection;
mod scalar;
pub mod spatial;
pub mod stack;
mod tensor;

#[cfg(test)]
mod test_support;

pub use domain::{ComplexPlane, RealAxis};
pub use error::TmmError;
pub use input::{
    CoordinateInput, Coordinates, InPlaneCoordinate, IncidentSide, Polarisation, SpectralCoordinate,
};

pub use evaluate::PlaneWaveEvaluator;
pub use material::{
    Constant, DerivativeOrder, DifferentiableMaterial, DifferentiableMeromorphicMaterial,
    EvaluateDifferentiableMaterial, EvaluateDifferentiableMeromorphicMaterial, EvaluateMaterial,
    EvaluateMeromorphicMaterial, Material, MeromorphicMaterial, Sampled, Scalar,
};

pub use field::VectorField;
pub use observable::{
    ConstitutiveFields, DirectedPower, DissipationDensity, ElectromagneticFields, EnergyDensity,
    FieldIndexError, InterfacePower, LayerDissipation, LayerPower, ModeResidual,
    PlaneWaveAmplitudes, PlaneWaveDeterminant, PlaneWavePower, StoredEnergy,
};
pub use parameter::{FiniteLayerIndex, Parameter};
pub use response::Response;
pub use scalar::ComplexScalar;

pub use spatial::{SpatialProfile, SpatialProfileError};

pub use stack::{
    AnalyticalMaterialStack, DifferentiableMaterialStack, Layer, MaterialStack,
    MeromorphicMaterialStack, Stack, Thickness, ValidationConfig,
};
