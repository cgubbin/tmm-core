use crate::{
    IncidentSide, PlaneWaveEvaluator, Polarisation,
    evaluate::query::PlaneWaveExternalQueries,
    test_support::{
        TOLERANCE,
        assertions::{
            assert_boundary_state_jet_close, assert_zero_jet_close, assert_zero_jet_zero,
        },
        jet::unit_jet_like,
        planar::{scalar_real_input, single_layer_stack, two_layer_stack},
    },
};

#[test]
fn one_layer_produces_two_interface_wave_records() {
    let evaluator = PlaneWaveEvaluator::new(crate::backend::scatter2::Scatter2::new());

    let stack = single_layer_stack(1.8, 0.23);

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let interfaces = state.raw_interface_wave_data(IncidentSide::Left).unwrap();

    assert_eq!(interfaces.len(), 2);
}

#[test]
fn two_layers_produce_three_interface_wave_records() {
    let evaluator = PlaneWaveEvaluator::new(crate::backend::scatter2::Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let interfaces = state.raw_interface_wave_data(IncidentSide::Right).unwrap();

    assert_eq!(interfaces.len(), 3);
}

#[test]
fn left_incidence_interface_data_contains_expected_exterior_waves() {
    let evaluator = PlaneWaveEvaluator::new(crate::backend::scatter2::Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let amplitudes = state.raw_amplitudes(IncidentSide::Left);

    let interfaces = state.raw_interface_wave_data(IncidentSide::Left).unwrap();

    let first = interfaces.first().unwrap();
    let last = interfaces.last().unwrap();

    assert_zero_jet_close(
        first.left().waves().forward(),
        &unit_jet_like(first.left().waves().forward().value()),
    );

    assert_zero_jet_close(first.left().waves().backward(), amplitudes.reflection());

    assert_zero_jet_close(last.right().waves().forward(), amplitudes.transmission());

    assert_zero_jet_zero(last.right().waves().backward());
}

#[test]
fn right_incidence_interface_data_contains_expected_exterior_waves() {
    let evaluator = PlaneWaveEvaluator::new(crate::backend::scatter2::Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let amplitudes = state.raw_amplitudes(IncidentSide::Right);

    let interfaces = state.raw_interface_wave_data(IncidentSide::Right).unwrap();

    let first = interfaces.first().unwrap();
    let last = interfaces.last().unwrap();

    assert_zero_jet_zero(first.left().waves().forward());

    assert_zero_jet_close(first.left().waves().backward(), amplitudes.transmission());

    assert_zero_jet_close(last.right().waves().forward(), amplitudes.reflection());

    assert_zero_jet_close(
        last.right().waves().backward(),
        &unit_jet_like(last.right().waves().backward().value()),
    );
}

#[test]
fn every_interface_side_state_matches_its_waves_and_admittance() {
    let evaluator = PlaneWaveEvaluator::new(crate::backend::scatter2::Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let interfaces = state.raw_interface_wave_data(IncidentSide::Left).unwrap();

    for interface in interfaces.iter() {
        for side in [interface.left(), interface.right()] {
            let actual = side.state();

            let expected = side.waves().clone().into_state(side.admittance());

            assert_boundary_state_jet_close(&actual, &expected, TOLERANCE);
        }
    }
}

#[test]
fn interface_wave_data_states_match_interface_state_projection() {
    let evaluator = PlaneWaveEvaluator::new(crate::backend::scatter2::Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let wave_data = state.raw_interface_wave_data(IncidentSide::Right).unwrap();

    let interface_states = state.raw_interface_states(IncidentSide::Right).unwrap();

    assert_eq!(wave_data.len(), interface_states.len(),);

    for (data, states) in wave_data.iter().zip(interface_states.iter()) {
        assert_boundary_state_jet_close(&data.left().state(), states.left(), TOLERANCE);

        assert_boundary_state_jet_close(&data.right().state(), states.right(), TOLERANCE);
    }
}

#[test]
fn transfer_backend_constructs_interface_wave_data() {
    let evaluator = PlaneWaveEvaluator::new(crate::backend::transfer2::Transfer2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let interfaces = state.raw_interface_wave_data(IncidentSide::Left).unwrap();

    assert_eq!(interfaces.len(), 3);
}
