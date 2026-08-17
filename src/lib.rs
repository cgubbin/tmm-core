#![allow(dead_code)]
#![allow(clippy::type_complexity)]

pub(crate) mod algebra;
pub mod backend;
pub(crate) mod derivative_parts;
mod differential;
mod domain;
mod evaluate;
pub mod field;
mod input;
pub mod material;
mod observable;
mod parameter;
mod projection;
mod response;
mod scalar;
mod spatial;
pub mod stack;
mod tensor;
mod waves;

#[cfg(test)]
mod test_support;

pub use algebra::{ModeJet1, ScalarAlgebra, SeedJet};

pub use backend::ExteriorWavevectors;

pub use domain::{ComplexPlane, RealAxis};

pub use input::{
    CanonicalCoordinates, CoordinateGrid, CoordinateInput, CoordinateReference, CoordinateSamples,
    Coordinates, InPlaneCoordinate, IncidentSide, Polarisation, SpectralCoordinate,
};

pub use evaluate::PlaneWaveEvaluator;

pub use material::{
    Constant, DerivativeOrder, DifferentiableMaterial, DifferentiableMeromorphicMaterial,
    EvaluateDifferentiableMaterial, EvaluateDifferentiableMeromorphicMaterial, EvaluateMaterial,
    EvaluateMeromorphicMaterial, Material, MeromorphicMaterial, Sampled, Scalar,
};

pub use field::VectorField;

pub use observable::{
    ConstitutiveFields, DirectedPower, ElectromagneticDissipation, ElectromagneticEnergy,
    ElectromagneticFields, ElectromagneticIntensities, FieldIndexError, InterfacePower,
    LayerDissipation, LayerPower, PlaneWaveAmplitudes, PlaneWaveDeterminant, PlaneWavePower,
};
pub use parameter::{FiniteLayerIndex, Parameter};
pub use response::Response;
pub use scalar::ComplexScalar;

pub use stack::{
    AnalyticalMaterialStack, DifferentiableMaterialStack, Layer, MaterialStack,
    MeromorphicMaterialStack, Stack,
};
