mod absorption;
mod boundary;
mod components;
mod determinant;
mod energy;
mod field;
mod interface;
mod layer;
mod layer_power;
mod mode;
mod plane_wave;

pub use absorption::DissipationDensity;
pub use boundary::{
    BoundaryProjectionError, BoundaryState, BoundaryWaves, LayerBoundaries, LayerBoundaryStates,
    LayerBoundaryWaves,
};
pub use determinant::PlaneWaveDeterminant;
pub use energy::{EnergyDensity, StoredEnergy};
pub use field::{ConstitutiveFields, ElectromagneticFields, FieldIndexError};
pub use interface::{
    DirectedPower, InterfacePower, InterfaceProjectionError, InterfaceStates, Interfaces,
};
pub use layer::{
    AggregateBilinearNormalization, AggregateBilinearOverlap, AggregateEnergy,
    AggregateHermitianOverlap, BilinearLayerNormalization, EnergyConfinement, LayerAggregateError,
    LayerConfinementError, LayerDissipation, LayerEnergy, LayerEnergyError, LayerParticipation,
    LayerParticipationError, LayerPower, LayerProjectionError, Layers, NormalizedHermitianOverlap,
};
pub use mode::ModeResidual;
pub use plane_wave::{PlaneWaveAmplitudes, PlaneWavePower};

pub(crate) use boundary::{project_layer_boundary_states, project_layer_boundary_waves};
pub(crate) use determinant::ProjectPlaneWaveModeDeterminant;
pub(crate) use interface::{
    InterfaceWaveData, assemble_interface_wave_data, exterior_boundary_states,
    exterior_boundary_waves, project_layer_admittances,
};
pub(crate) use layer::{
    BilinearLayerOverlap, BilinearLayerOverlapInput, HermitianLayerOverlap,
    HermitianLayerOverlapInput, LayerOverlapInput, LayerOverlapOperand, OverlapError, PairOperand,
    project_integrated_field_norms,
};
pub(crate) use layer::{LayerIntegrationInput, assemble_layer_integration_inputs};
pub(crate) use plane_wave::{ProjectAmplitudes, ProjectPower};
