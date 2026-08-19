use ndarray::{ArrayBase, Ix0, OwnedRepr};

use crate::{
    FiniteLayerIndex, IncidentSide, Polarisation, RealAxisEvaluator,
    backend::{RetainedIsotropicLayers, scatter2::Scatter2, transfer2::Transfer2},
    test_support::{
        C,
        planar::{scalar_real_input, two_layer_stack},
    },
};

type ComplexArray = ArrayBase<OwnedRepr<C>, Ix0>;

fn complex_scalar(value: &ComplexArray) -> C {
    value[()]
}

#[test]
fn layer_wave_data_returns_one_record_per_finite_layer() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let layers = state
        .raw_layer_integration_inputs_unchecked(IncidentSide::Left)
        .unwrap();

    assert_eq!(layers.len(), 2);
}

#[test]
fn layer_wave_data_uses_each_layers_left_boundary_waves() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let boundaries = state
        .raw_layer_boundary_waves_unchecked(IncidentSide::Right)
        .unwrap();

    let layers = state
        .raw_layer_integration_inputs_unchecked(IncidentSide::Right)
        .unwrap();

    assert_eq!(layers.len(), boundaries.len());

    for index in 0..layers.len() {
        let layer = layers.get(FiniteLayerIndex::new(index)).unwrap();
        let boundary = boundaries.get(crate::FiniteLayerIndex::new(index)).unwrap();

        assert_eq!(
            layer.waves(),
            boundary.left(),
            "layer {index} should use its left boundary waves",
        );

        assert_ne!(
            layer.waves(),
            boundary.right(),
            "test fixture should distinguish layer {index} boundaries",
        );
    }
}

#[test]
fn layer_wave_data_preserves_quantities_in_physical_order() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let layers = state
        .raw_layer_integration_inputs_unchecked(IncidentSide::Left)
        .unwrap();

    let workspace = state.workspace();

    assert_eq!(layers.len(), 2);

    for index in 0..layers.len() {
        let layer = layers.get(FiniteLayerIndex::new(index)).unwrap();

        let expected = workspace
            .layer_quantities(index)
            .expect("retained quantities should exist");

        assert_eq!(
            layer.quantities(),
            expected,
            "quantities should remain aligned at layer {index}",
        );
    }
}

#[test]
fn layer_wave_data_preserves_thicknesses_in_physical_order() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let layers = state
        .raw_layer_integration_inputs_unchecked(IncidentSide::Left)
        .unwrap();

    let workspace = state.workspace();

    for index in 0..layers.len() {
        let expected = workspace
            .layer_thickness(index)
            .expect("retained thickness should exist");

        assert_eq!(
            layers
                .get(FiniteLayerIndex::new(index))
                .unwrap()
                .thickness(),
            expected,
            "thickness should remain aligned at layer {index}",
        );
    }
}

#[test]
fn incident_side_changes_waves_but_not_layer_metadata() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let left = state
        .raw_layer_integration_inputs_unchecked(IncidentSide::Left)
        .unwrap();

    let right = state
        .raw_layer_integration_inputs_unchecked(IncidentSide::Right)
        .unwrap();

    assert_eq!(left.len(), right.len());

    for index in 0..left.len() {
        let left_layer = left.get(FiniteLayerIndex::new(index)).unwrap();
        let right_layer = right.get(FiniteLayerIndex::new(index)).unwrap();

        assert_eq!(left_layer.quantities(), right_layer.quantities(),);

        assert_eq!(left_layer.thickness(), right_layer.thickness(),);
    }

    assert!(
        left.iter()
            .zip(right.iter())
            .any(|(left, right)| left.waves() != right.waves()),
        "opposite incidence directions should produce different layer waves",
    );
}

#[test]
fn transfer_backend_assembles_layer_wave_data() {
    let evaluator = RealAxisEvaluator::new(Transfer2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let layers = state
        .raw_layer_integration_inputs_unchecked(IncidentSide::Left)
        .unwrap();

    assert_eq!(layers.len(), 2);
}
