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
pub use determinant::PlaneWaveDeterminant;
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
pub(crate) use determinant::ProjectPlaneWaveModeDeterminant;
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
