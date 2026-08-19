//! Physical observables derived from plane-wave and modal solutions.
//!
//! This module defines the backend-independent physical quantities exposed by
//! `lamina-core`. Numerical backends produce boundary solutions and retained
//! internal state; the observable layer converts those representations into
//! quantities with direct electromagnetic meaning.
//!
//! The module is organised by where a quantity lives:
//!
//! - [`boundary`] contains states and directional-wave amplitudes at layer
//!   boundaries;
//! - [`interface`] contains quantities defined at interfaces, including
//!   directed power flow;
//! - [`field`] contains local electromagnetic and constitutive fields;
//! - [`layer`] contains layer-resolved and layer-integrated quantities;
//! - [`plane_wave`] contains exterior reflection/transmission amplitudes and
//!   powers;
//! - [`determinant`] contains the scalar outgoing-mode determinant.
//!
//! # Backend independence
//!
//! Public observable types do not expose transfer- or scattering-matrix
//! representations. Both numerical backends reconstruct the same physical
//! quantities through these types.
//!
//! # Derivatives
//!
//! During evaluation, observable structures may contain jet-valued components.
//! Arithmetic and reconstruction therefore propagate the selected derivative
//! algebra through the same physical operations used for values. Differential
//! responses are crystallised only after the requested observable has been
//! assembled.
//!
//! # Projection
//!
//! Sampled observables may also be projected to an owned scalar sample.
//! Projection preserves all derivative components while reducing the sampled
//! ndarray dimension to `Ix0`.

mod boundary;
mod determinant;
mod field;
mod interface;
mod layer;
mod plane_wave;

pub use boundary::{
    BoundaryProjectionError, BoundaryState, BoundaryWaves, LayerBoundaries, LayerBoundaryStates,
    LayerBoundaryWaves,
};
pub use determinant::{PlaneWaveDeterminant, ProjectPlaneWaveModeDeterminant};
pub use field::{
    ConstitutiveFields, ElectromagneticDissipation, ElectromagneticEnergy, ElectromagneticFields,
    ElectromagneticIntensities, FieldIndexError, FieldReconstructionError,
};
pub use interface::{
    DirectedPower, InterfacePower, InterfaceProjectionError, InterfaceStates, Interfaces,
};
pub use layer::{
    AggregateBilinearNormalization, AggregateBilinearOverlap, AggregateEnergy,
    AggregateHermitianOverlap, EnergyConfinement, LayerAggregateError, LayerConfinementError,
    LayerDissipation, LayerEnergy, LayerEnergyError, LayerParticipation, LayerParticipationError,
    LayerPower, LayerProjectionError, Layers,
};
pub use plane_wave::{PlaneWaveAmplitudes, PlaneWavePower};

pub(crate) use boundary::{
    project_layer_boundary_states, project_layer_boundary_waves, project_layer_mode_waves,
};
pub(crate) use field::{
    ConstitutiveFieldReconstructionError, ConstitutiveSamplingContext, ConstitutiveSamplingError,
    FieldSamplingContext, IsotropicConstitutiveParameters, IsotropicConstitutiveSpectralData,
    electromagnetic_dissipation_coefficients,
};
pub(crate) use interface::{
    InterfaceWaveData, assemble_interface_wave_data, exterior_boundary_states,
    exterior_boundary_waves, project_layer_admittances,
};
pub(crate) use layer::{
    BilinearLayerOverlap, BilinearLayerOverlapInput, HermitianLayerOverlap,
    HermitianLayerOverlapInput, LayerOverlapInput, LayerOverlapOperand, OverlapError,
};
pub(crate) use layer::{LayerIntegrationInput, assemble_layer_integration_inputs};
pub(crate) use plane_wave::{Amplitudes, ProjectAmplitudes, ProjectPower};
