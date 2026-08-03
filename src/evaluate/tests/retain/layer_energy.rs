use approx::assert_relative_eq;
use ndarray::Array0;

use crate::{
    IncidentSide, PlaneWaveEvaluator, Polarisation,
    backend::Scatter2,
    observable::EnergyDefinition,
    test_support::{
        assertions::VALUE_TOLERANCE,
        planar::{scalar_real_input, two_layer_stack},
    },
};

fn scalar<C: Copy>(data: &Array0<C>) -> C {
    data[()]
}

#[test]
fn nondispersive_energy_returns_one_record_per_layer() {
    let state = PlaneWaveEvaluator::new(Scatter2::new())
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let energy = state
        .nondispersive_layer_energy(IncidentSide::Left)
        .unwrap();

    assert_eq!(energy.value().len(), 2);
}

#[test]
fn nondispersive_layer_energy_is_positive() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for side in [IncidentSide::Left, IncidentSide::Right] {
            let state = evaluator
                .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
                .unwrap();

            let energy = state
                .nondispersive_layer_energy(IncidentSide::Left)
                .unwrap();

            for layer in energy.value().iter() {
                assert!(scalar(layer.electric()) >= 0.0,);

                assert!(scalar(layer.magnetic()) >= 0.0,);

                assert!(scalar(layer.total()) > 0.0,);
            }
        }
    }
}

#[test]
fn nondispersive_energy_total_is_component_sum() {
    let state = PlaneWaveEvaluator::new(Scatter2::new())
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let energy = state
        .nondispersive_layer_energy(IncidentSide::Right)
        .unwrap();

    for layer in energy.value().iter() {
        assert_relative_eq!(
            scalar(layer.total()),
            scalar(layer.electric()) + scalar(layer.magnetic()),
            epsilon = VALUE_TOLERANCE,
            max_relative = VALUE_TOLERANCE,
        );
    }
}
