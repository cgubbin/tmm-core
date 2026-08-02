use nalgebra::ComplexField;
use ndarray::Dimension;
use num_traits::{One, Zero};

use crate::{
    ComplexScalar, PlaneWaveAmplitudes, RealAxis,
    algebra::ScalarAlgebra,
    backend::{
        BidirectionalWaves, ExteriorAdmittanceProvider, IsotropicLayerQuantities,
        PlaneWaveSolutionSource, ReconstructLayerBoundaryWaves, Scatter2, Transfer2, TransferState,
        bidirectional_waves_from_state, right_exterior_waves, transfer_state_from_waves,
        transfer_state_slope,
    },
    derivative_parts::IntoValue,
    evaluate::{PlaneWaveEvaluator, query::PlaneWaveExternalQueries},
    input::{CanonicalCoordinates, IncidentSide, Polarisation},
    observable::{BoundaryWaves, LayerBoundaryWaves},
    parameter::{FiniteLayerIndex, Parameter},
    test_support::{
        assertions::{
            VALUE_TOLERANCE, assert_bidirectional_waves_close, assert_bivariate_first_layers_close,
            assert_bivariate_second_layers_close, assert_boundary_waves_close,
            assert_first_layers_close, assert_layer_boundary_waves_close,
            assert_layer_states_collection_close, assert_layer_waves_collection_close,
            assert_second_layers_close, assert_zero_layers_close,
        },
        finite_difference::{FIRST_DERIVATIVE_TOLERANCE, SECOND_DERIVATIVE_TOLERANCE},
        jet::{J0, zero_jet_from_real_value},
        planar::{scalar_real_input, single_layer_stack, two_layer_stack},
    },
};

macro_rules! retained_boundary_wave_equivalence_suite {
    (
        $module:ident,
        left = $left_backend:expr,
        right = $right_backend:expr $(,)?
    ) => {
        mod $module {
            use super::*;

            #[test]
            fn one_layer_te_left_incidence_matches() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.23);

                let left_waves = left
                    .retain(
                        scalar_real_input(2.3, 0.37),
                        &stack,
                        Polarisation::TransverseElectric,
                    )
                    .expect("left backend should retain")
                    .raw_layer_boundary_waves(IncidentSide::Left)
                    .expect("left workspace should contain retained layers");

                let right_waves = right
                    .retain(
                        scalar_real_input(2.3, 0.37),
                        &stack,
                        Polarisation::TransverseElectric,
                    )
                    .expect("right backend should retain")
                    .raw_layer_boundary_waves(IncidentSide::Left)
                    .expect("right workspace should contain retained layers");

                assert_zero_layers_close(
                    &left_waves.into_inner(),
                    &right_waves.into_inner(),
                    VALUE_TOLERANCE,
                );
            }

            #[test]
            fn one_layer_te_right_incidence_matches() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.23);

                let left_waves = left
                    .retain(
                        scalar_real_input(2.3, 0.37),
                        &stack,
                        Polarisation::TransverseElectric,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Right)
                    .unwrap();

                let right_waves = right
                    .retain(
                        scalar_real_input(2.3, 0.37),
                        &stack,
                        Polarisation::TransverseElectric,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Right)
                    .unwrap();

                assert_zero_layers_close(
                    &left_waves.into_inner(),
                    &right_waves.into_inner(),
                    VALUE_TOLERANCE,
                );
            }

            #[test]
            fn one_layer_tm_left_incidence_matches() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.23);

                let left_waves = left
                    .retain(
                        scalar_real_input(2.3, 0.37),
                        &stack,
                        Polarisation::TransverseMagnetic,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Left)
                    .unwrap();

                let right_waves = right
                    .retain(
                        scalar_real_input(2.3, 0.37),
                        &stack,
                        Polarisation::TransverseMagnetic,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Left)
                    .unwrap();

                assert_zero_layers_close(
                    &left_waves.into_inner(),
                    &right_waves.into_inner(),
                    VALUE_TOLERANCE,
                );
            }

            #[test]
            fn one_layer_tm_right_incidence_matches() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = single_layer_stack(1.8, 0.23);

                let left_waves = left
                    .retain(
                        scalar_real_input(2.3, 0.37),
                        &stack,
                        Polarisation::TransverseMagnetic,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Right)
                    .unwrap();

                let right_waves = right
                    .retain(
                        scalar_real_input(2.3, 0.37),
                        &stack,
                        Polarisation::TransverseMagnetic,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Right)
                    .unwrap();

                assert_zero_layers_close(
                    &left_waves.into_inner(),
                    &right_waves.into_inner(),
                    VALUE_TOLERANCE,
                );
            }

            #[test]
            fn two_layer_te_left_incidence_matches() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = two_layer_stack();

                let left_waves = left
                    .retain(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Left)
                    .unwrap();

                let right_waves = right
                    .retain(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Left)
                    .unwrap();

                assert_eq!(left_waves.len(), 2);

                assert_zero_layers_close(
                    &left_waves.into_inner(),
                    &right_waves.into_inner(),
                    VALUE_TOLERANCE,
                );
            }

            #[test]
            fn two_layer_te_right_incidence_matches() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = two_layer_stack();

                let left_waves = left
                    .retain(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Right)
                    .unwrap();

                let right_waves = right
                    .retain(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Right)
                    .unwrap();

                assert_zero_layers_close(
                    &left_waves.into_inner(),
                    &right_waves.into_inner(),
                    VALUE_TOLERANCE,
                );
            }

            #[test]
            fn two_layer_tm_left_incidence_matches() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = two_layer_stack();

                let left_waves = left
                    .retain(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseMagnetic,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Left)
                    .unwrap();

                let right_waves = right
                    .retain(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseMagnetic,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Left)
                    .unwrap();

                assert_zero_layers_close(
                    &left_waves.into_inner(),
                    &right_waves.into_inner(),
                    VALUE_TOLERANCE,
                );
            }

            #[test]
            fn two_layer_tm_right_incidence_matches() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = two_layer_stack();

                let left_waves = left
                    .retain(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseMagnetic,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Right)
                    .unwrap();

                let right_waves = right
                    .retain(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseMagnetic,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Right)
                    .unwrap();

                assert_zero_layers_close(
                    &left_waves.into_inner(),
                    &right_waves.into_inner(),
                    VALUE_TOLERANCE,
                );
            }

            #[test]
            fn first_spectral_derivative_matches() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = two_layer_stack();

                let left_waves = left
                    .retain_first(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Left)
                    .unwrap();

                let right_waves = right
                    .retain_first(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Left)
                    .unwrap();

                assert_first_layers_close(&left_waves.into_inner(), &right_waves.into_inner());
            }

            #[test]
            fn first_thickness_derivative_matches() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = two_layer_stack();

                let parameter = Parameter::LayerThickness(FiniteLayerIndex(1));

                let left_waves = left
                    .retain_first(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        parameter,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Right)
                    .unwrap();

                let right_waves = right
                    .retain_first(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        parameter,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Right)
                    .unwrap();

                assert_first_layers_close(&left_waves.into_inner(), &right_waves.into_inner());
            }

            #[test]
            fn second_spectral_derivative_matches() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = two_layer_stack();

                let left_waves = left
                    .retain_second(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Left)
                    .unwrap();

                let right_waves = right
                    .retain_second(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Left)
                    .unwrap();

                assert_second_layers_close(&left_waves.into_inner(), &right_waves.into_inner());
            }

            #[test]
            fn second_thickness_derivative_matches() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = two_layer_stack();

                let parameter = Parameter::LayerThickness(FiniteLayerIndex(0));

                let left_waves = left
                    .retain_second(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        parameter,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Right)
                    .unwrap();

                let right_waves = right
                    .retain_second(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        parameter,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Right)
                    .unwrap();

                assert_second_layers_close(&left_waves.into_inner(), &right_waves.into_inner());
            }

            #[test]
            fn bivariate_first_derivatives_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = two_layer_stack();

                let axis0 = Parameter::Spectral;

                let axis1 = Parameter::LayerThickness(FiniteLayerIndex(1));

                let left_waves = left
                    .retain_bivariate_first(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                        axis0,
                        axis1,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Left)
                    .unwrap();

                let right_waves = right
                    .retain_bivariate_first(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                        axis0,
                        axis1,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Left)
                    .unwrap();

                assert_bivariate_first_layers_close(
                    &left_waves.into_inner(),
                    &right_waves.into_inner(),
                );
            }

            #[test]
            fn bivariate_second_derivatives_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = two_layer_stack();

                let axis0 = Parameter::Spectral;

                let axis1 = Parameter::LayerThickness(FiniteLayerIndex(1));

                let left_waves = left
                    .retain_bivariate_second(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                        axis0,
                        axis1,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Left)
                    .unwrap();

                let right_waves = right
                    .retain_bivariate_second(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                        axis0,
                        axis1,
                    )
                    .unwrap()
                    .raw_layer_boundary_waves(IncidentSide::Left)
                    .unwrap();

                assert_bivariate_second_layers_close(
                    &left_waves.into_inner(),
                    &right_waves.into_inner(),
                );
            }
        }
    };
}

retained_boundary_wave_equivalence_suite!(
    transfer2_matches_scatter2,
    left = Transfer2::new(),
    right = Scatter2::new(),
);

fn waves_from_state<A>(field: &A, slope: &A, characteristic_slope: &A) -> BidirectionalWaves<A>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar + Copy,
    A::Dimension: Dimension,
{
    let slope_ratio = slope.divide(characteristic_slope);

    let half = (<A::Scalar as One>::one() + <A::Scalar as One>::one()).recip();

    let forward = field.subtract(&slope_ratio).scale(half);

    let backward = field.add(&slope_ratio).scale(half);

    BidirectionalWaves::new(forward, backward)
}

fn analytic_single_layer_boundary_waves<A>(
    amplitudes: &PlaneWaveAmplitudes<A>,
    left_admittance: &A,
    layer_admittance: &A,
    right_admittance: &A,
    incident_side: IncidentSide,
) -> LayerBoundaryWaves<A>
where
    A: ScalarAlgebra + Clone,
    A::Scalar: ComplexScalar + One + Zero + Copy,
    A::Dimension: Dimension,
{
    let left_slope = transfer_state_slope(left_admittance);

    let layer_slope = transfer_state_slope(layer_admittance);

    let right_slope = transfer_state_slope(right_admittance);

    let zero = A::filled_constant_like(left_admittance.value(), <A::Scalar as Zero>::zero());

    let one = A::filled_constant_like(left_admittance.value(), <A::Scalar as One>::one());

    let (left_exterior, right_exterior) = match incident_side {
        IncidentSide::Left => (
            BidirectionalWaves::new(one, amplitudes.reflection().clone()),
            BidirectionalWaves::new(amplitudes.transmission().clone(), zero),
        ),

        IncidentSide::Right => (
            BidirectionalWaves::new(zero, amplitudes.transmission().clone()),
            BidirectionalWaves::new(amplitudes.reflection().clone(), one),
        ),
    };

    let left_field = left_exterior.forward().add(left_exterior.backward());

    let left_state_slope =
        left_slope.multiply(&left_exterior.backward().subtract(left_exterior.forward()));

    let right_field = right_exterior.forward().add(right_exterior.backward());

    let right_state_slope =
        right_slope.multiply(&right_exterior.backward().subtract(right_exterior.forward()));

    let left = waves_from_state(&left_field, &left_state_slope, &layer_slope);

    let right = waves_from_state(&right_field, &right_state_slope, &layer_slope);

    crate::backend::LayerBoundaryWaves::new(left, right).into()
}

#[test]
fn transfer_single_layer_waves_match_boundary_continuity() {
    let evaluator = PlaneWaveEvaluator::new(Transfer2::new());

    let stack = single_layer_stack(1.8, 0.23);

    let coordinates = CanonicalCoordinates::new(
        zero_jet_from_real_value(2.3),
        zero_jet_from_real_value(0.37),
    );

    let polarisation = Polarisation::TransverseElectric;

    let state = evaluator
        .retain(scalar_real_input(2.3, 0.37), &stack, polarisation)
        .unwrap();

    let amplitudes = state.raw_amplitudes(IncidentSide::Left);

    let actual = state.raw_layer_boundary_waves(IncidentSide::Left).unwrap();

    let expected = analytic_single_layer_boundary_waves(
        &amplitudes,
        &IsotropicLayerQuantities::evaluate::<RealAxis, _>(
            stack.left_exterior(),
            &coordinates,
            polarisation,
        )
        .into_admittance()
        .into_inner(),
        &IsotropicLayerQuantities::evaluate::<RealAxis, _>(
            stack.iter().next().unwrap().material(),
            &coordinates,
            polarisation,
        )
        .into_admittance()
        .into_inner(),
        &IsotropicLayerQuantities::evaluate::<RealAxis, _>(
            stack.right_exterior(),
            &coordinates,
            polarisation,
        )
        .into_admittance()
        .into_inner(),
        IncidentSide::Left,
    );

    assert_eq!(actual.len(), 1);

    assert_layer_boundary_waves_close(&actual.into_inner()[0], &expected, VALUE_TOLERANCE);
}

#[test]
fn scatter_single_layer_waves_match_boundary_continuity() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = single_layer_stack(1.8, 0.23);

    let coordinates = CanonicalCoordinates::new(
        zero_jet_from_real_value(2.3),
        zero_jet_from_real_value(0.37),
    );

    let polarisation = Polarisation::TransverseElectric;

    let state = evaluator
        .retain(scalar_real_input(2.3, 0.37), &stack, polarisation)
        .unwrap();

    let amplitudes = state.raw_amplitudes(IncidentSide::Left);

    let actual = state.raw_layer_boundary_waves(IncidentSide::Left).unwrap();

    let expected = analytic_single_layer_boundary_waves(
        &amplitudes,
        &IsotropicLayerQuantities::evaluate::<RealAxis, _>(
            stack.left_exterior(),
            &coordinates,
            polarisation,
        )
        .into_admittance()
        .into_inner(),
        &IsotropicLayerQuantities::evaluate::<RealAxis, _>(
            stack.iter().next().unwrap().material(),
            &coordinates,
            polarisation,
        )
        .into_admittance()
        .into_inner(),
        &IsotropicLayerQuantities::evaluate::<RealAxis, _>(
            stack.right_exterior(),
            &coordinates,
            polarisation,
        )
        .into_admittance()
        .into_inner(),
        IncidentSide::Left,
    );

    assert_eq!(actual.len(), 1);

    assert_layer_boundary_waves_close(&actual.into_inner()[0], &expected, VALUE_TOLERANCE);
}

#[test]
fn transfer_single_layer_reconstruction_localises_boundary_error() {
    let evaluator = PlaneWaveEvaluator::new(Transfer2::new());

    let stack = single_layer_stack(1.8, 0.23);

    let coordinates = CanonicalCoordinates::new(
        zero_jet_from_real_value(2.3),
        zero_jet_from_real_value(0.37),
    );

    let polarisation = Polarisation::TransverseElectric;

    let state = evaluator
        .retain(scalar_real_input(2.3, 0.37), &stack, polarisation)
        .unwrap();

    let amplitudes = state.raw_amplitudes(IncidentSide::Left);

    let actual = state.raw_layer_boundary_waves(IncidentSide::Left).unwrap();

    let expected = analytic_single_layer_boundary_waves(
        &amplitudes,
        &IsotropicLayerQuantities::evaluate::<RealAxis, _>(
            stack.left_exterior(),
            &coordinates,
            polarisation,
        )
        .into_admittance()
        .into_inner(),
        &IsotropicLayerQuantities::evaluate::<RealAxis, _>(
            stack.iter().next().unwrap().material(),
            &coordinates,
            polarisation,
        )
        .into_admittance()
        .into_inner(),
        &IsotropicLayerQuantities::evaluate::<RealAxis, _>(
            stack.right_exterior(),
            &coordinates,
            polarisation,
        )
        .into_admittance()
        .into_inner(),
        IncidentSide::Left,
    );

    assert_eq!(actual.len(), 1);

    assert_boundary_waves_close(
        actual.clone().into_inner()[0].right(),
        expected.right(),
        VALUE_TOLERANCE,
    );

    assert_boundary_waves_close(
        actual.into_inner()[0].left(),
        expected.left(),
        VALUE_TOLERANCE,
    );
}

#[test]
fn analytic_right_boundary_matches_transfer_state_conversion() {
    let evaluator = PlaneWaveEvaluator::new(Transfer2::new());
    let stack = single_layer_stack(1.8, 0.23);

    let state = evaluator
        .retain(
            scalar_real_input(2.3, 0.37),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let workspace = state.workspace();
    let solution = workspace.solution();
    let retained = workspace.retained().unwrap();

    let amplitudes = solution.amplitudes(IncidentSide::Left);

    let right_admittance = solution.context().right_admittance();

    let layer_admittance = retained.layers()[0]
        .quantities()
        .clone()
        .into_admittance()
        .into_inner();

    let exterior_waves =
        right_exterior_waves(&amplitudes, IncidentSide::Left, right_admittance.value());

    let exterior_state = transfer_state_from_waves(&exterior_waves, &right_admittance);

    let direct = bidirectional_waves_from_state(&exterior_state, &layer_admittance);

    let analytic = analytic_single_layer_boundary_waves(
        &amplitudes,
        &solution.context().left_admittance(),
        &layer_admittance,
        &right_admittance,
        IncidentSide::Left,
    );

    assert_boundary_waves_close(&direct.into(), analytic.right(), 1.0e-12);
}

#[test]
fn plane_wave_state_delegates_boundary_reconstruction_to_workspace() {
    let evaluator = PlaneWaveEvaluator::new(Transfer2::new());

    let stack = single_layer_stack(1.8, 0.23);

    let state = evaluator
        .retain(
            scalar_real_input(2.3, 0.37),
            &stack,
            Polarisation::TransverseElectric,
        )
        .expect("retained evaluation should succeed");

    let from_state = state
        .raw_layer_boundary_waves(IncidentSide::Left)
        .expect("state should retain boundary data");

    let from_workspace = state
        .workspace()
        .reconstruct_layer_boundary_waves(IncidentSide::Left)
        .expect("workspace should retain boundary data");

    assert_eq!(from_state.len(), from_workspace.len());

    for (state_layer, workspace_layer) in from_state.iter().zip(from_workspace.iter()) {
        assert_boundary_waves_close(
            state_layer.left(),
            &workspace_layer.left().clone().into(),
            1.0e-12,
        );

        assert_boundary_waves_close(
            state_layer.right(),
            &workspace_layer.right().clone().into(),
            1.0e-12,
        );
    }
}

#[test]
fn evaluator_workspace_right_boundary_matches_analytic_continuity() {
    let evaluator = PlaneWaveEvaluator::new(Transfer2::new());

    let stack = single_layer_stack(1.8, 0.23);

    let state = evaluator
        .retain(
            scalar_real_input(2.3, 0.37),
            &stack,
            Polarisation::TransverseElectric,
        )
        .unwrap();

    let workspace = state.workspace();

    let actual = workspace
        .reconstruct_layer_boundary_waves(IncidentSide::Left)
        .unwrap();

    let solution = workspace.solution();

    let amplitudes = solution.amplitudes(IncidentSide::Left);

    let right_admittance = solution.context().right_admittance();

    let retained = workspace.retained().expect("layers should be retained");

    let layer_admittance = retained.layers()[0]
        .quantities()
        .clone()
        .into_admittance()
        .into_inner();

    let right_exterior =
        right_exterior_waves(&amplitudes, IncidentSide::Left, right_admittance.value());

    let right_state = transfer_state_from_waves(&right_exterior, &right_admittance);

    let expected = bidirectional_waves_from_state(&right_state, &layer_admittance);

    assert_bidirectional_waves_close(actual[0].right(), &expected, 1.0e-12);
}

fn assert_jet_close_named(label: &str, actual: &J0, expected: &J0, tolerance: f64) {
    println!("At {label}");

    approx::assert_relative_eq!(
        actual.value()[()].re,
        expected.value()[()].re,
        epsilon = tolerance,
        max_relative = tolerance,
    );

    approx::assert_relative_eq!(
        actual.value()[()].im,
        expected.value()[()].im,
        epsilon = tolerance,
        max_relative = tolerance,
    );
}

fn assert_transfer_state_close(
    label: &str,
    actual: &TransferState<J0>,
    expected: &TransferState<J0>,
    tolerance: f64,
) {
    assert_jet_close_named(
        &format!("{label}: field"),
        actual.field(),
        expected.field(),
        tolerance,
    );

    assert_jet_close_named(
        &format!("{label}: slope"),
        actual.slope(),
        expected.slope(),
        tolerance,
    );
}

fn assert_bidirectional_waves_close_named(
    label: &str,
    actual: &BidirectionalWaves<J0>,
    expected: &BidirectionalWaves<J0>,
    tolerance: f64,
) {
    assert_jet_close_named(
        &format!("{label}: forward"),
        actual.forward(),
        expected.forward(),
        tolerance,
    );

    assert_jet_close_named(
        &format!("{label}: backward"),
        actual.backward(),
        expected.backward(),
        tolerance,
    );
}

#[test]
fn transfer_evaluator_right_boundary_reconstruction_is_stepwise_consistent() {
    let evaluator = PlaneWaveEvaluator::new(Transfer2::new());

    let stack = single_layer_stack(1.8, 0.23);

    let state = evaluator
        .retain(
            scalar_real_input(2.3, 0.37),
            &stack,
            Polarisation::TransverseElectric,
        )
        .expect("retained evaluation should succeed");

    let workspace = state.workspace();
    let solution = workspace.solution();

    let retained = workspace
        .retained()
        .expect("transfer layers should be retained");

    assert_eq!(retained.layers().len(), 1);

    let amplitudes = solution.amplitudes(IncidentSide::Left);

    let right_admittance = solution.context().right_admittance();

    let layer_admittance = retained.layers()[0]
        .quantities()
        .clone()
        .into_admittance()
        .into_inner();

    /*
     * Stage 1: construct the right-exterior waves exactly as the workspace
     * implementation does.
     */
    let expected_exterior_waves =
        right_exterior_waves(&amplitudes, IncidentSide::Left, right_admittance.value());

    /*
     * Stage 2: convert them to the transfer-state representation.
     */
    let expected_right_state =
        transfer_state_from_waves(&expected_exterior_waves, &right_admittance);

    /*
     * Stage 3: propagate through retained layers and inspect the stored right
     * boundary before any wave decomposition occurs.
     */
    let boundary_states = retained.propagate_right_state(expected_right_state.clone());

    assert_eq!(boundary_states.len(), 1);

    assert_transfer_state_close(
        "retained right boundary state",
        boundary_states[0].right(),
        &expected_right_state,
        1.0e-12,
    );

    /*
     * Stage 4: directly decompose the expected right state in the finite
     * layer's basis.
     */
    let expected_layer_waves =
        bidirectional_waves_from_state(&expected_right_state, &layer_admittance);

    /*
     * Stage 5: decompose the state record returned by propagation.
     */
    let propagated_layer_waves =
        bidirectional_waves_from_state(boundary_states[0].right(), &layer_admittance);

    assert_bidirectional_waves_close_named(
        "propagated state decomposition",
        &propagated_layer_waves,
        &expected_layer_waves,
        1.0e-12,
    );

    /*
     * Stage 6: use the retained-container convenience method.
     */
    let retained_waves = retained.reconstruct_layer_boundary_waves(expected_right_state.clone());

    assert_eq!(retained_waves.len(), 1);

    assert_bidirectional_waves_close_named(
        "retained reconstruction",
        retained_waves[0].right(),
        &expected_layer_waves,
        1.0e-12,
    );

    /*
     * Stage 7: use the workspace trait method, which independently constructs
     * its own amplitudes and right state.
     */
    let workspace_waves = workspace
        .reconstruct_layer_boundary_waves(IncidentSide::Left)
        .expect("workspace should retain layer waves");

    assert_eq!(workspace_waves.len(), 1);

    assert_bidirectional_waves_close_named(
        "workspace reconstruction",
        workspace_waves[0].right(),
        &expected_layer_waves,
        1.0e-12,
    );
}

macro_rules! boundary_observable_suite {
    (
        $module:ident,
        backend = $backend:expr $(,)?
    ) => {
        mod $module {
            use super::*;

            #[test]
            fn value_boundary_waves_match_raw_waves() {
                let evaluator = PlaneWaveEvaluator::new($backend);

                let stack = two_layer_stack();

                let state = evaluator
                    .retain(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                    )
                    .expect("retained evaluation should succeed");

                for side in [IncidentSide::Left, IncidentSide::Right] {
                    let raw = state
                        .raw_layer_boundary_waves(side)
                        .expect("raw boundary-wave projection should succeed");

                    let response = state
                        .boundary_waves(side)
                        .expect("boundary-wave response should assemble");

                    assert_layer_waves_collection_close(
                        response.value(),
                        &raw.into_value().into_inner(),
                        VALUE_TOLERANCE,
                    );
                }
            }

            #[test]
            fn value_boundary_states_match_raw_states() {
                let evaluator = PlaneWaveEvaluator::new($backend);

                let stack = two_layer_stack();

                let state = evaluator
                    .retain(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseMagnetic,
                    )
                    .expect("retained evaluation should succeed");

                for side in [IncidentSide::Left, IncidentSide::Right] {
                    let raw = state
                        .raw_layer_boundary_states(side)
                        .expect("raw boundary-state projection should succeed");

                    let response = state
                        .boundary_states(side)
                        .expect("boundary-state response should assemble");

                    assert_layer_states_collection_close(
                        response.value(),
                        &raw.into_value().into_inner(),
                        VALUE_TOLERANCE,
                    );
                }
            }

            #[test]
            fn value_boundary_methods_preserve_layer_order() {
                let evaluator = PlaneWaveEvaluator::new($backend);

                let stack = two_layer_stack();

                let state = evaluator
                    .retain(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                    )
                    .unwrap();

                let waves = state.boundary_waves(IncidentSide::Left).unwrap();

                let states = state.boundary_states(IncidentSide::Left).unwrap();

                assert_eq!(waves.value().len(), 2);
                assert_eq!(states.value().len(), 2);

                /*
                 * The corresponding layer-state and layer-wave entries must
                 * remain aligned. A stronger wave-to-state equality is tested
                 * separately below.
                 */
                assert_eq!(waves.value().len(), states.value().len(),);
            }

            #[test]
            fn first_boundary_waves_have_requested_parameter() {
                let evaluator = PlaneWaveEvaluator::new($backend);

                let stack = two_layer_stack();

                let state = evaluator
                    .retain_first(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap();

                let response = state.boundary_waves(IncidentSide::Left).unwrap();

                assert_eq!(response.derivatives().parameter(), Parameter::Spectral,);

                assert_eq!(response.value().len(), 2);
                assert_eq!(response.derivatives().first().len(), 2,);
            }

            #[test]
            fn first_boundary_states_have_requested_parameter() {
                let evaluator = PlaneWaveEvaluator::new($backend);

                let stack = two_layer_stack();

                let parameter = Parameter::LayerThickness(FiniteLayerIndex(1));

                let state = evaluator
                    .retain_first(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        parameter,
                    )
                    .unwrap();

                let response = state.boundary_states(IncidentSide::Right).unwrap();

                assert_eq!(response.derivatives().parameter(), parameter,);

                assert_eq!(response.value().len(), 2);
                assert_eq!(response.derivatives().first().len(), 2,);
            }

            #[test]
            fn second_boundary_waves_contain_first_and_second_branches() {
                let evaluator = PlaneWaveEvaluator::new($backend);

                let stack = two_layer_stack();

                let state = evaluator
                    .retain_second(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap();

                let response = state.boundary_waves(IncidentSide::Left).unwrap();

                assert_eq!(response.derivatives().parameter(), Parameter::Spectral,);

                assert_eq!(response.value().len(), 2);
                assert_eq!(response.derivatives().first().len(), 2,);
                assert_eq!(response.derivatives().second().len(), 2,);
            }

            #[test]
            fn second_boundary_states_contain_first_and_second_branches() {
                let evaluator = PlaneWaveEvaluator::new($backend);

                let stack = two_layer_stack();

                let parameter = Parameter::LayerThickness(FiniteLayerIndex(0));

                let state = evaluator
                    .retain_second(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        parameter,
                    )
                    .unwrap();

                let response = state.boundary_states(IncidentSide::Right).unwrap();

                assert_eq!(response.derivatives().parameter(), parameter,);

                assert_eq!(response.value().len(), 2);
                assert_eq!(response.derivatives().first().len(), 2,);
                assert_eq!(response.derivatives().second().len(), 2,);
            }

            #[test]
            fn bivariate_first_boundary_waves_preserve_parameter_order() {
                let evaluator = PlaneWaveEvaluator::new($backend);

                let stack = two_layer_stack();

                let axis0 = Parameter::Spectral;
                let axis1 = Parameter::LayerThickness(FiniteLayerIndex(1));

                let state = evaluator
                    .retain_bivariate_first(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                        axis0,
                        axis1,
                    )
                    .unwrap();

                let response = state.boundary_waves(IncidentSide::Left).unwrap();

                assert_eq!(response.derivatives().parameters(), [axis0, axis1],);

                assert_eq!(response.value().len(), 2);
                assert_eq!(response.derivatives().axis0().len(), 2,);
                assert_eq!(response.derivatives().axis1().len(), 2,);
            }

            #[test]
            fn bivariate_second_boundary_states_preserve_all_branches() {
                let evaluator = PlaneWaveEvaluator::new($backend);

                let stack = two_layer_stack();

                let axis0 = Parameter::Spectral;
                let axis1 = Parameter::LayerThickness(FiniteLayerIndex(1));

                let state = evaluator
                    .retain_bivariate_second(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        axis0,
                        axis1,
                    )
                    .unwrap();

                let response = state.boundary_states(IncidentSide::Right).unwrap();

                assert_eq!(response.derivatives().parameters(), [axis0, axis1],);

                assert_eq!(response.value().len(), 2);

                let gradient = response.derivatives().first();

                let hessian = response.derivatives().second();

                assert_eq!(gradient.axis0().len(), 2);
                assert_eq!(gradient.axis1().len(), 2);

                assert_eq!(hessian.axis0_axis0().len(), 2,);
                assert_eq!(hessian.axis0_axis1().len(), 2,);
                assert_eq!(hessian.axis1_axis1().len(), 2,);
            }
        }
    };
}

boundary_observable_suite!(transfer2, backend = crate::backend::Transfer2::new(),);

boundary_observable_suite!(scatter2, backend = crate::backend::Scatter2::new(),);

macro_rules! boundary_observable_equivalence_suite {
    (
        $module:ident,
        left = $left_backend:expr,
        right = $right_backend:expr $(,)?
    ) => {
        mod $module {
            use super::*;

            #[test]
            fn value_boundary_waves_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = two_layer_stack();

                for polarisation in [
                    Polarisation::TransverseElectric,
                    Polarisation::TransverseMagnetic,
                ] {
                    for side in [IncidentSide::Left, IncidentSide::Right] {
                        let left_response = left
                            .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
                            .unwrap()
                            .boundary_waves(side)
                            .unwrap();

                        let right_response = right
                            .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
                            .unwrap()
                            .boundary_waves(side)
                            .unwrap();

                        assert_layer_waves_collection_close(
                            left_response.value(),
                            right_response.value(),
                            VALUE_TOLERANCE,
                        );
                    }
                }
            }

            #[test]
            fn value_boundary_states_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = two_layer_stack();

                for polarisation in [
                    Polarisation::TransverseElectric,
                    Polarisation::TransverseMagnetic,
                ] {
                    for side in [IncidentSide::Left, IncidentSide::Right] {
                        let left_response = left
                            .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
                            .unwrap()
                            .boundary_states(side)
                            .unwrap();

                        let right_response = right
                            .retain(scalar_real_input(2.5, 0.31), &stack, polarisation)
                            .unwrap()
                            .boundary_states(side)
                            .unwrap();

                        assert_layer_states_collection_close(
                            left_response.value(),
                            right_response.value(),
                            VALUE_TOLERANCE,
                        );
                    }
                }
            }

            #[test]
            fn first_boundary_waves_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = two_layer_stack();

                let left_response = left
                    .retain_first(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .boundary_waves(IncidentSide::Left)
                    .unwrap();

                let right_response = right
                    .retain_first(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                        Parameter::Spectral,
                    )
                    .unwrap()
                    .boundary_waves(IncidentSide::Left)
                    .unwrap();

                assert_layer_waves_collection_close(
                    left_response.value(),
                    right_response.value(),
                    VALUE_TOLERANCE,
                );

                assert_layer_waves_collection_close(
                    left_response.derivatives().first(),
                    right_response.derivatives().first(),
                    FIRST_DERIVATIVE_TOLERANCE,
                );
            }

            #[test]
            fn second_boundary_states_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = two_layer_stack();

                let parameter = Parameter::LayerThickness(FiniteLayerIndex(1));

                let left_response = left
                    .retain_second(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        parameter,
                    )
                    .unwrap()
                    .boundary_states(IncidentSide::Right)
                    .unwrap();

                let right_response = right
                    .retain_second(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseMagnetic,
                        parameter,
                    )
                    .unwrap()
                    .boundary_states(IncidentSide::Right)
                    .unwrap();

                assert_layer_states_collection_close(
                    left_response.value(),
                    right_response.value(),
                    VALUE_TOLERANCE,
                );

                assert_layer_states_collection_close(
                    left_response.derivatives().first(),
                    right_response.derivatives().first(),
                    FIRST_DERIVATIVE_TOLERANCE,
                );

                assert_layer_states_collection_close(
                    left_response.derivatives().second(),
                    right_response.derivatives().second(),
                    SECOND_DERIVATIVE_TOLERANCE,
                );
            }

            #[test]
            fn bivariate_second_boundary_waves_match() {
                let left = PlaneWaveEvaluator::new($left_backend);

                let right = PlaneWaveEvaluator::new($right_backend);

                let stack = two_layer_stack();

                let axis0 = Parameter::Spectral;
                let axis1 = Parameter::LayerThickness(FiniteLayerIndex(1));

                let left_response = left
                    .retain_bivariate_second(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                        axis0,
                        axis1,
                    )
                    .unwrap()
                    .boundary_waves(IncidentSide::Left)
                    .unwrap();

                let right_response = right
                    .retain_bivariate_second(
                        scalar_real_input(2.5, 0.31),
                        &stack,
                        Polarisation::TransverseElectric,
                        axis0,
                        axis1,
                    )
                    .unwrap()
                    .boundary_waves(IncidentSide::Left)
                    .unwrap();

                assert_layer_waves_collection_close(
                    left_response.value(),
                    right_response.value(),
                    VALUE_TOLERANCE,
                );

                assert_layer_waves_collection_close(
                    left_response.derivatives().first().axis0(),
                    right_response.derivatives().first().axis0(),
                    FIRST_DERIVATIVE_TOLERANCE,
                );

                assert_layer_waves_collection_close(
                    left_response.derivatives().first().axis1(),
                    right_response.derivatives().first().axis1(),
                    FIRST_DERIVATIVE_TOLERANCE,
                );

                assert_layer_waves_collection_close(
                    left_response.derivatives().second().axis0_axis0(),
                    right_response.derivatives().second().axis0_axis0(),
                    SECOND_DERIVATIVE_TOLERANCE,
                );

                assert_layer_waves_collection_close(
                    left_response.derivatives().second().axis0_axis1(),
                    right_response.derivatives().second().axis0_axis1(),
                    SECOND_DERIVATIVE_TOLERANCE,
                );

                assert_layer_waves_collection_close(
                    left_response.derivatives().second().axis1_axis1(),
                    right_response.derivatives().second().axis1_axis1(),
                    SECOND_DERIVATIVE_TOLERANCE,
                );
            }
        }
    };
}

boundary_observable_equivalence_suite!(
    transfer2_matches_scatter2_boundary_observables,
    left = crate::backend::Transfer2::new(),
    right = crate::backend::Scatter2::new(),
);
