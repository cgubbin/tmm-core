use crate::{
    IncidentSide, Polarisation, RealAxisEvaluator,
    backend::{Scatter2, Transfer2},
    test_support::{
        TOLERANCE,
        assertions::{assert_real_array_close, assert_real_close, assert_real_zero},
        planar::{scalar_real_input, two_layer_stack},
    },
};

#[test]
fn left_incidence_exterior_interface_power_matches_plane_wave_power() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let excitation = state
        .excitation(IncidentSide::Left)
        .expect("state should be projectable");

    let external = excitation.power();

    let interfaces = excitation.interface_power().unwrap();

    let first = interfaces.first().unwrap();

    let last = interfaces.last().unwrap();

    assert_real_close(first.left().forward_flux()[()], 1.0, TOLERANCE);

    assert_real_array_close(
        first.left().backward_flux(),
        &external.reflectance().mapv(|value| -value),
        TOLERANCE,
    );

    assert_real_array_close(
        last.right().forward_flux(),
        external.transmittance(),
        TOLERANCE,
    );

    assert_real_zero(last.right().backward_flux()[()], TOLERANCE);
}

#[test]
fn net_flux_is_continuous_across_every_interface() {
    let evaluator = RealAxisEvaluator::new(Scatter2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseMagnetic,
        )
        .unwrap();

    let excitation = state
        .excitation(IncidentSide::Left)
        .expect("state should be projectable");

    let response = excitation.interface_power().unwrap();

    for interface in response.iter() {
        assert_real_array_close(
            interface.left_net_flux(),
            interface.right_net_flux(),
            TOLERANCE,
        );
    }
}

#[test]
fn transfer_backend_projects_interface_power() {
    let evaluator = RealAxisEvaluator::new(Transfer2::new());

    let state = evaluator
        .retain(
            scalar_real_input(2.5, 0.31),
            &two_layer_stack(),
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let excitation = state
        .excitation(IncidentSide::Left)
        .expect("state should be projectable");

    let response = excitation.interface_power().unwrap();

    assert_eq!(response.len(), 3);
}
