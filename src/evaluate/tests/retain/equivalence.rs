use crate::{
    Parameter, PlaneWaveEvaluator, Polarisation,
    backend::{ExteriorContextProvider, Scatter2, Transfer2},
    test_support::{
        assertions::assert_first_jet_close,
        planar::{scalar_real_input, two_layer_stack},
    },
};

#[test]
fn retained_backends_agree_on_exterior_constitutive_quantities() {
    let stack = two_layer_stack();

    let scatter = PlaneWaveEvaluator::new(Scatter2::new())
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap();

    let transfer = PlaneWaveEvaluator::new(Transfer2::new())
        .retain_first(
            scalar_real_input(2.5, 0.31),
            &stack,
            Polarisation::TransverseElectric,
            Parameter::Spectral,
        )
        .unwrap();

    let scatter = scatter.project_point(&()).unwrap();
    let transfer = transfer.project_point(&()).unwrap();

    let scatter = scatter.workspace().solution().context();
    let transfer = transfer.workspace().solution().context();

    assert_first_jet_close(scatter.left_epsilon(), transfer.left_epsilon());
    assert_first_jet_close(scatter.left_mu(), transfer.left_mu());
    assert_first_jet_close(scatter.right_epsilon(), transfer.right_epsilon());
    assert_first_jet_close(scatter.right_mu(), transfer.right_mu());
}
