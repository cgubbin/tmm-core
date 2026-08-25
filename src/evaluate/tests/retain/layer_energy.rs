use approx::assert_relative_eq;
use ndarray::Array0;

use crate::{
    IncidentSide, Polarisation, RealAxisEvaluator,
    backend::Scatter2,
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
    let state = RealAxisEvaluator::new(Scatter2::new())
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let energy = state
        .excitation(IncidentSide::Left)
        .expect("state should be projectable")
        .layer_energy_nondispersive()
        .unwrap();

    assert_eq!(energy.len(), 2);
}

#[test]
fn nondispersive_layer_energy_is_positive() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

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
                .excitation(side)
                .expect("state should be projectable")
                .layer_energy_nondispersive()
                .unwrap();

            for layer in energy.iter() {
                assert!(scalar(layer.electric()) >= 0.0,);

                assert!(scalar(layer.magnetic()) >= 0.0,);

                assert!(scalar(layer.total()) > 0.0,);
            }
        }
    }
}

#[test]
fn nondispersive_energy_total_is_component_sum() {
    let state = RealAxisEvaluator::new(Scatter2::new())
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let energy = state
        .excitation(IncidentSide::Right)
        .expect("state should be projectable")
        .layer_energy_nondispersive()
        .unwrap();

    for layer in energy.iter() {
        assert_relative_eq!(
            scalar(layer.total()),
            scalar(layer.electric()) + scalar(layer.magnetic()),
            epsilon = VALUE_TOLERANCE,
            max_relative = VALUE_TOLERANCE,
        );
    }
}
