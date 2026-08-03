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
    EnergyDefinition, LayerDissipation, LayerEnergy, LayerEnergyError, LayerPower,
    LayerProjectionError, Layers,
};
pub use mode::ModeResidual;
pub use plane_wave::{PlaneWaveAmplitudes, PlaneWavePower};

pub(crate) use boundary::{project_layer_boundary_states, project_layer_boundary_waves};
pub(crate) use determinant::ProjectPlaneWaveModeDeterminant;
pub(crate) use interface::{
    InterfaceWaveData, assemble_interface_states, assemble_interface_wave_data,
    exterior_boundary_states, exterior_boundary_waves, project_interface_power,
    project_layer_admittances,
};
pub(crate) use layer::{
    IsotropicBrillouinEnergyData, LayerWaveData, assemble_layer_wave_data,
    canonical_energy_normalization, evaluate_brillouin_layer_energy_data,
    evaluate_nondispersive_layer_energy_data, integrate_layer_wave_sequence,
    project_layer_brillouin_energy_sequence, project_layer_dissipation_sequence,
    project_layer_energy_sequence, project_layer_power,
};
pub(crate) use plane_wave::{ProjectAmplitudes, ProjectPower};
