mod data;
mod dissipation;
mod energy;
mod energy_data;
mod field_norm;
mod normalisation;
mod overlap;
mod power;
mod project;
mod state_overlap;

pub(crate) use data::{IntegratedLayerWaveData, LayerWaveData};
pub use dissipation::LayerDissipation;
pub(crate) use energy::canonical_energy_normalization;
pub use energy::{EnergyDefinition, LayerEnergy};
pub(crate) use energy_data::{IsotropicBrillouinEnergyData, evaluate_brillouin_layer_energy_data};
pub(crate) use field_norm::IntegratedFieldNorms;
pub(crate) use overlap::{
    IntegratedWaveProducts, integrate_bilinear_wave_products, integrate_hermitian_wave_products,
};
pub use power::LayerPower;
pub use project::LayerEnergyError;
pub use project::LayerProjectionError;
pub(crate) use project::{
    assemble_layer_wave_data, evaluate_nondispersive_layer_energy_data,
    integrate_layer_wave_sequence, project_layer_brillouin_energy_sequence,
    project_layer_dissipation_sequence, project_layer_energy_sequence, project_layer_power,
};
pub(crate) use state_overlap::IntegratedStateProducts;

/// Layer quantities in physical left-to-right order.
///
/// A stack containing `N` finite layers has `N + 1` interfaces.
#[derive(Clone, Debug, PartialEq)]
pub struct Layers<T> {
    values: Vec<T>,
}

impl<T> Layers<T> {
    pub(crate) fn new(values: Vec<T>) -> Self {
        Self { values }
    }

    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.values.get(index)
    }

    pub fn first(&self) -> Option<&T> {
        self.values.first()
    }

    pub fn last(&self) -> Option<&T> {
        self.values.last()
    }

    pub fn iter(&self) -> impl ExactSizeIterator<Item = &T> {
        self.values.iter()
    }

    pub fn into_inner(self) -> Vec<T> {
        self.values
    }

    pub fn map<U>(self, map: impl FnMut(T) -> U) -> Layers<U> {
        Layers::new(self.values.into_iter().map(map).collect())
    }
}

// pub(crate) fn assemble_interface_states<A>(
//     layers: LayerBoundaries<LayerBoundaryStates<A>>,
//     left_exterior: BoundaryState<A>,
//     right_exterior: BoundaryState<A>,
// ) -> Layers<InterfaceStates<A>> {
//     let layers = layers.into_inner();

//     if layers.is_empty() {
//         return Layers::new(vec![InterfaceStates::new(left_exterior, right_exterior)]);
//     }

//     let interface_count = layers.len() + 1;
//     let mut interfaces = Vec::with_capacity(interface_count);

//     let mut layers = layers.into_iter();

//     let first = layers
//         .next()
//         .expect("non-empty layer collection was checked");

//     let (first_left, first_right) = first.into_parts();

//     interfaces.push(LayerStates::new(left_exterior, first_left));

//     let mut previous_right = first_right;

//     for layer in layers {
//         let (current_left, current_right) = layer.into_parts();

//         interfaces.push(LayerStates::new(previous_right, current_left));

//         previous_right = current_right;
//     }

//     interfaces.push(LayerStates::new(previous_right, right_exterior));

//     Layers::new(interfaces)
// }

// #[cfg(test)]
// mod tests {
//     use super::*;

//     #[test]
//     fn empty_finite_stack_produces_one_exterior_interface() {
//         let interfaces = assemble_interface_states(
//             LayerBoundaries::new(Vec::new()),
//             BoundaryState::new("left field", "left secondary"),
//             BoundaryState::new("right field", "right secondary"),
//         );

//         assert_eq!(interfaces.len(), 1);

//         assert_eq!(interfaces.first().unwrap().left().field(), &"left field",);

//         assert_eq!(interfaces.first().unwrap().right().field(), &"right field",);
//     }

//     #[test]
//     fn two_layers_produce_three_interfaces_in_physical_order() {
//         let layers = LayerBoundaries::new(vec![
//             LayerBoundaryStates::new(BoundaryState::new(10, 11), BoundaryState::new(12, 13)),
//             LayerBoundaryStates::new(BoundaryState::new(20, 21), BoundaryState::new(22, 23)),
//         ]);

//         let interfaces =
//             assemble_interface_states(layers, BoundaryState::new(0, 1), BoundaryState::new(30, 31));

//         assert_eq!(interfaces.len(), 3);

//         let values: Vec<_> = interfaces
//             .iter()
//             .map(|interface| (*interface.left().field(), *interface.right().field()))
//             .collect();

//         assert_eq!(values, vec![(0, 10), (12, 20), (22, 30),],);
//     }
// }
