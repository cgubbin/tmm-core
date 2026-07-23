// mod field;
mod input;
// mod isotropic;
// mod matrix;
// mod mode;
// mod plane_wave;

// #[cfg(test)]
// mod tests;

// pub mod scatter2;
// // pub mod transfer2;

pub(crate) struct RealAxis;
pub(crate) struct ComplexPlane;

pub use input::{IncidentSide, Polarisation};
// pub use derivative::{
//     DerivativeVariable, SpectralDerivativeVariable, StructuralDerivativeVariable,
// };

// pub use field::{
//     ExteriorSampling, FieldPosition, FieldSampling, FieldSamplingRegion, IsotropicFieldState,
//     LayerSampling, PlaneWaveFieldBackend, PlaneWaveFieldError, PlaneWaveFieldResponse,
//     PlaneWaveFieldSample, PlaneWaveFieldSampleOwned, PlaneWaveFieldSampleView, PlaneWaveFields,
//     PlaneWavePowerBalance,
// };

// pub use input::{IncidentSide, PlanarInput, PlaneWaveInput, Polarisation};

// pub use matrix::{
//     ComplexMatrixBackend, ComplexMatrixKxDerivativeBackend, ComplexMatrixSpectralDerivativeBackend,
//     ComplexMatrixThicknessDerivativeBackend, MatrixEvaluation, RawMatrixBackend,
//     RawMatrixKxDerivativeBackend, RawMatrixSpectralDerivativeBackend,
//     RawMatrixThicknessDerivativeBackend,
// };

// pub use mode::{
//     AnalyticResidual, OutgoingModeResidualBackend, OutgoingModeResidualKxDerivativeBackend,
//     OutgoingModeResidualSpectralDerivativeBackend, OutgoingModeResidualThicknessDerivativeBackend,
//     OutgoingModeResponse, OutgoingModeStateBackend,
// };

// pub use plane_wave::{
//     PlaneWaveAmplitudeDifferential, PlaneWaveAmplitudes, PlaneWaveBackend,
//     PlaneWaveKxDerivativeBackend, PlaneWavePower, PlaneWavePowerDifferential, PlaneWaveResponse,
//     PlaneWaveResponseDerivatives, PlaneWaveResponseDifferential,
//     PlaneWaveSpectralDerivativeBackend, PlaneWaveThicknessDerivativeBackend,
// };

// use isotropic::IsotropicLayerQuantities;
