#![allow(dead_code)]

pub mod backend;
mod material;
mod scalar;
mod stack;
mod tensor;

pub use backend::{
    DerivativeVariable, IncidentSide, OutgoingModeBackend, PlanarInput, PlaneWaveBackend,
    PlaneWaveInput, PlaneWaveResponse, PlaneWaveResponseDerivatives, PlaneWaveResponseDifferential,
    Polarisation,
};

pub use backend::transfer2::Transfer2;
pub use material::{Material, Sampled};
pub use scalar::ComplexScalar;
// pub use backend::scatter2::Scatter2;

pub use stack::{Layer, Stack, Thickness, ValidationConfig};
