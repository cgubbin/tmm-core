use lamina_units::Length;
use ndarray::{ArrayBase, Dimension, Ix1, OwnedRepr, arr0};

use crate::{
    ComplexPlane, ComplexPlaneEvaluator, ComplexScalar, ExteriorWavevectors, Polarisation,
    algebra::ScalarAlgebra,
    backend::{
        ExteriorContextProvider, ModalSolutionSource, ReconstructExteriorModeWaves,
        ReconstructLayerModeWaves, RetainedIsotropicLayers, Scatter2, Transfer2,
        evaluate_exterior_wavevectors,
    },
    field::VectorField,
    input::{CanonicalCoordinates, CanonicalStack, canonical::CanonicalLayer},
    material::{ConstitutiveEvaluator, ConstitutiveLift},
    spatial::{ExteriorSampling, FieldSampling, LayerSampling},
    test_support::{
        C,
        assertions::assert_complex_close,
        finite_difference::{
            FIRST_DERIVATIVE_TOLERANCE, SECOND_DERIVATIVE_TOLERANCE, VALUE_TOLERANCE,
        },
        jet::{HoloJ0, HoloJ1, HoloJ2, HoloJB2},
        planar::two_layer_stack,
    },
};

type ComplexArray = ArrayBase<OwnedRepr<C>, Ix1>;

const K0: C = C::new(2.5, -0.05);
const K_PARALLEL: C = C::new(0.31, 0.02);

// -----------------------------------------------------------------------------
// Evaluation fixtures
// -----------------------------------------------------------------------------

fn value_coordinates() -> CanonicalCoordinates<HoloJ0> {
    CanonicalCoordinates::new(
        HoloJ0::constant(arr0(K0)),
        HoloJ0::constant(arr0(K_PARALLEL)),
    )
}

fn first_spectral_coordinates() -> CanonicalCoordinates<HoloJ1> {
    CanonicalCoordinates::new(
        HoloJ1::variable(arr0(K0)),
        HoloJ1::constant(arr0(K_PARALLEL)),
    )
}

fn second_spectral_coordinates() -> CanonicalCoordinates<HoloJ2> {
    CanonicalCoordinates::new(
        HoloJ2::variable(arr0(K0)),
        HoloJ2::constant(arr0(K_PARALLEL)),
    )
}

/// Canonical bivariate problem with
///
/// - axis 0 = spectral coordinate;
/// - axis 1 = layer-1 thickness.
fn bivariate_coordinates() -> CanonicalCoordinates<HoloJB2> {
    CanonicalCoordinates::new(
        HoloJB2::variable_axis0(arr0(K0)),
        HoloJB2::constant(arr0(K_PARALLEL)),
    )
}

fn exterior<J, M>(
    stack: &CanonicalStack<M, J>,
    coordinates: &CanonicalCoordinates<J>,
    _polarisation: Polarisation,
) -> ExteriorWavevectors<J>
where
    J: ScalarAlgebra + ConstitutiveLift<ComplexPlane, M> + Clone,
    J::Scalar: ComplexScalar,
    J::Dimension: Dimension,
    ComplexPlane: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
{
    evaluate_exterior_wavevectors::<ComplexPlane, M, J>(
        coordinates,
        stack.left_exterior(),
        stack.right_exterior(),
    )
}

fn first_order_geometry_stack(
    differentiated_layer: usize,
) -> CanonicalStack<crate::Constant<f64>, HoloJ1> {
    let stack = two_layer_stack();

    let layers = stack
        .layers_left_to_right()
        .iter()
        .enumerate()
        .map(|(index, layer)| {
            let thickness_cm = layer.thickness().as_centimetres();

            let value = arr0(C::new(thickness_cm, 0.0));

            let thickness = if index == differentiated_layer {
                HoloJ1::variable(value)
            } else {
                HoloJ1::constant(value)
            };

            CanonicalLayer::new(layer.material().clone(), thickness)
        })
        .collect();

    CanonicalStack::new(
        stack.left_exterior().clone(),
        stack.right_exterior().clone(),
        layers,
    )
}

fn bivariate_geometry_stack() -> CanonicalStack<crate::Constant<f64>, HoloJB2> {
    let stack = two_layer_stack();

    let layers = stack
        .layers_left_to_right()
        .iter()
        .enumerate()
        .map(|(index, layer)| {
            let thickness_cm = layer.thickness().as_centimetres();

            let value = arr0(C::new(thickness_cm, 0.0));

            let thickness = if index == 1 {
                HoloJB2::variable_axis1(value)
            } else {
                HoloJB2::constant(value)
            };

            CanonicalLayer::new(layer.material().clone(), thickness)
        })
        .collect();

    CanonicalStack::new(
        stack.left_exterior().clone(),
        stack.right_exterior().clone(),
        layers,
    )
}

macro_rules! for_each_value_backend {
    ($stack:expr, $evaluator:ident, $body:block) => {{
        {
            let $evaluator =
                ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&$stack, Scatter2::new()).unwrap();

            $body
        }

        {
            let $evaluator =
                ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&$stack, Transfer2::new()).unwrap();

            $body
        }
    }};
}

// -----------------------------------------------------------------------------
// Spatial sampling
// -----------------------------------------------------------------------------

fn sampling() -> FieldSampling<f64> {
    FieldSampling::new()
        .left_exterior(ExteriorSampling::point(Length::zero()))
        .layer(0, LayerSampling::uniform(3))
        .layer(1, LayerSampling::uniform(3))
        .right_exterior(ExteriorSampling::point(Length::zero()))
}

fn interface_sampling() -> FieldSampling<f64> {
    FieldSampling::new()
        .left_exterior(ExteriorSampling::point(Length::zero()))
        .layer_interfaces()
        .right_exterior(ExteriorSampling::point(Length::zero()))
}

// -----------------------------------------------------------------------------
// Field assertions
// -----------------------------------------------------------------------------

fn assert_zero(values: &ComplexArray, tolerance: f64) {
    for &value in values {
        assert!(
            value.norm() <= tolerance,
            "expected zero, got {value:?}; |value| = {:e}",
            value.norm(),
        );
    }
}

fn assert_te_structure(
    electric: &VectorField<C, Ix1>,
    magnetic: &VectorField<C, Ix1>,
    tolerance: f64,
) {
    assert_zero(electric.x(), tolerance);
    assert_zero(electric.z(), tolerance);
    assert_zero(magnetic.y(), tolerance);
}

fn assert_tm_structure(
    electric: &VectorField<C, Ix1>,
    magnetic: &VectorField<C, Ix1>,
    tolerance: f64,
) {
    assert_zero(electric.y(), tolerance);
    assert_zero(magnetic.x(), tolerance);
    assert_zero(magnetic.z(), tolerance);
}

fn assert_all_finite(electric: &VectorField<C, Ix1>, magnetic: &VectorField<C, Ix1>) {
    for component in [
        electric.x(),
        electric.y(),
        electric.z(),
        magnetic.x(),
        magnetic.y(),
        magnetic.z(),
    ] {
        assert!(
            component
                .iter()
                .all(|value| value.re.is_finite() && value.im.is_finite()),
        );
    }
}

fn assert_complex_close_at_interface(
    quantity: &str,
    interface: &str,
    actual: C,
    expected: C,
    tolerance: f64,
) {
    let error = (actual - expected).norm();

    assert!(
        error <= tolerance,
        "{quantity} is discontinuous at {interface}: \
         left = {actual:?}, right = {expected:?}; \
         absolute error = {error:e}",
    );
}

fn assert_te_interface_continuity(
    electric: &VectorField<C, Ix1>,
    magnetic: &VectorField<C, Ix1>,
    tolerance: f64,
) {
    for (name, left, right) in [
        ("left exterior / layer 0", 0, 1),
        ("layer 0 / layer 1", 2, 3),
        ("layer 1 / right exterior", 4, 5),
    ] {
        assert_complex_close_at_interface(
            "Ey",
            name,
            electric.y()[left],
            electric.y()[right],
            tolerance,
        );

        assert_complex_close_at_interface(
            "Hx",
            name,
            magnetic.x()[left],
            magnetic.x()[right],
            tolerance,
        );
    }
}

fn assert_tm_interface_continuity(
    electric: &VectorField<C, Ix1>,
    magnetic: &VectorField<C, Ix1>,
    tolerance: f64,
) {
    for (name, left, right) in [
        ("left exterior / layer 0", 0, 1),
        ("layer 0 / layer 1", 2, 3),
        ("layer 1 / right exterior", 4, 5),
    ] {
        assert_complex_close_at_interface(
            "Ex",
            name,
            electric.x()[left],
            electric.x()[right],
            tolerance,
        );

        assert_complex_close_at_interface(
            "Hy",
            name,
            magnetic.y()[left],
            magnetic.y()[right],
            tolerance,
        );
    }
}

// -----------------------------------------------------------------------------
// Value reconstruction
// -----------------------------------------------------------------------------

#[test]
fn modal_te_fields_from_retained_complex_solution() {
    let stack = two_layer_stack();

    let evaluator =
        ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, Scatter2::new()).unwrap();

    let polarisation = Polarisation::TransverseElectric;

    let coordinates = value_coordinates();

    let exterior = exterior(evaluator.stack(), &coordinates, polarisation);

    let state = evaluator
        .retain(coordinates, exterior, polarisation)
        .unwrap();

    let mode = state.mode().unwrap();

    let spatial_response = mode.fields(&sampling()).unwrap();

    let response = spatial_response.quantity();

    /*
     * left exterior  1
     * layer 0        3
     * layer 1        3
     * right exterior 1
     */
    assert_eq!(response.electric().value().x().shape(), &[8]);
    assert_eq!(response.electric().value().y().shape(), &[8]);
    assert_eq!(response.electric().value().z().shape(), &[8]);

    assert_eq!(response.magnetic().value().x().shape(), &[8]);
    assert_eq!(response.magnetic().value().y().shape(), &[8]);
    assert_eq!(response.magnetic().value().z().shape(), &[8]);

    assert_te_structure(
        response.electric().value(),
        response.magnetic().value(),
        VALUE_TOLERANCE,
    );

    assert_all_finite(response.electric().value(), response.magnetic().value());
}

#[test]
fn modal_tm_fields_from_retained_complex_solution() {
    let stack = two_layer_stack();

    let evaluator =
        ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, Scatter2::new()).unwrap();

    let polarisation = Polarisation::TransverseMagnetic;

    let coordinates = value_coordinates();

    let exterior = exterior(evaluator.stack(), &coordinates, polarisation);

    let state = evaluator
        .retain(coordinates, exterior, polarisation)
        .unwrap();

    let mode = state.mode().unwrap();

    let response = mode.fields(&sampling()).unwrap();

    let response = response.quantity();

    assert_eq!(response.electric().value().x().shape(), &[8]);
    assert_eq!(response.magnetic().value().y().shape(), &[8]);

    assert_tm_structure(
        response.electric().value(),
        response.magnetic().value(),
        VALUE_TOLERANCE,
    );

    assert_all_finite(response.electric().value(), response.magnetic().value());
}

#[test]
fn modal_fields_are_nonzero() {
    let stack = two_layer_stack();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let evaluator =
            ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, Scatter2::new()).unwrap();

        let coordinates = value_coordinates();

        let exterior = exterior(evaluator.stack(), &coordinates, polarisation);

        let state = evaluator
            .retain(coordinates, exterior, polarisation)
            .unwrap();

        let mode = state.mode().unwrap();

        let response = mode.fields(&sampling()).unwrap();

        let response = response.quantity();

        let total = [
            response.electric().value().x(),
            response.electric().value().y(),
            response.electric().value().z(),
            response.magnetic().value().x(),
            response.magnetic().value().y(),
            response.magnetic().value().z(),
        ]
        .into_iter()
        .flat_map(|component| component.iter())
        .map(|value| value.norm_sqr())
        .sum::<f64>();

        assert!(
            total > VALUE_TOLERANCE,
            "modal field candidate must be nonzero",
        );
    }
}

// -----------------------------------------------------------------------------
// Differential reconstruction
// -----------------------------------------------------------------------------

#[test]
fn first_spectral_modal_field_derivative_survives_reconstruction() {
    let stack = two_layer_stack();

    let evaluator =
        ComplexPlaneEvaluator::<HoloJ1, _, _>::compile(&stack, Scatter2::new()).unwrap();

    let polarisation = Polarisation::TransverseElectric;

    let coordinates = first_spectral_coordinates();

    let exterior = exterior(evaluator.stack(), &coordinates, polarisation);

    let state = evaluator
        .retain(coordinates, exterior, polarisation)
        .unwrap();

    let mode = state.mode().unwrap();

    let response = mode.fields(&sampling()).unwrap();

    let response = response.quantity();

    assert_eq!(response.electric().first().x().shape(), &[8]);
    assert_eq!(response.magnetic().first().x().shape(), &[8]);

    assert_te_structure(
        response.electric().value(),
        response.magnetic().value(),
        VALUE_TOLERANCE,
    );

    assert_te_structure(
        response.electric().first(),
        response.magnetic().first(),
        FIRST_DERIVATIVE_TOLERANCE,
    );

    assert_all_finite(response.electric().value(), response.magnetic().value());

    assert_all_finite(response.electric().first(), response.magnetic().first());
}

#[test]
fn thickness_modal_field_derivative_survives_reconstruction() {
    let evaluator =
        ComplexPlaneEvaluator::from_canonical_stack(first_order_geometry_stack(1), Scatter2::new());

    let polarisation = Polarisation::TransverseMagnetic;

    let coordinates = CanonicalCoordinates::new(
        HoloJ1::constant(arr0(K0)),
        HoloJ1::constant(arr0(K_PARALLEL)),
    );

    let exterior = exterior(evaluator.stack(), &coordinates, polarisation);

    let state = evaluator
        .retain(coordinates, exterior, polarisation)
        .unwrap();

    let mode = state.mode().unwrap();

    let response = mode.fields(&sampling()).unwrap();

    let response = response.quantity();

    assert_tm_structure(
        response.electric().value(),
        response.magnetic().value(),
        VALUE_TOLERANCE,
    );

    assert_tm_structure(
        response.electric().first(),
        response.magnetic().first(),
        FIRST_DERIVATIVE_TOLERANCE,
    );

    assert_all_finite(response.electric().first(), response.magnetic().first());
}

#[test]
fn second_spectral_modal_field_derivative_survives_reconstruction() {
    let stack = two_layer_stack();

    let evaluator =
        ComplexPlaneEvaluator::<HoloJ2, _, _>::compile(&stack, Scatter2::new()).unwrap();

    let polarisation = Polarisation::TransverseElectric;

    let coordinates = second_spectral_coordinates();

    let exterior = exterior(evaluator.stack(), &coordinates, polarisation);

    let state = evaluator
        .retain(coordinates, exterior, polarisation)
        .unwrap();

    let mode = state.mode().unwrap();

    let response = mode.fields(&sampling()).unwrap();

    let response = response.quantity();

    assert_te_structure(
        response.electric().value(),
        response.magnetic().value(),
        VALUE_TOLERANCE,
    );

    assert_te_structure(
        response.electric().first(),
        response.magnetic().first(),
        FIRST_DERIVATIVE_TOLERANCE,
    );

    assert_te_structure(
        response.electric().second(),
        response.magnetic().second(),
        SECOND_DERIVATIVE_TOLERANCE,
    );

    assert_all_finite(response.electric().first(), response.magnetic().first());

    assert_all_finite(response.electric().second(), response.magnetic().second());
}

#[test]
fn bivariate_modal_field_derivatives_survive_reconstruction() {
    let evaluator =
        ComplexPlaneEvaluator::from_canonical_stack(bivariate_geometry_stack(), Scatter2::new());

    let polarisation = Polarisation::TransverseMagnetic;

    let coordinates = bivariate_coordinates();

    let exterior = exterior(evaluator.stack(), &coordinates, polarisation);

    let state = evaluator
        .retain(coordinates, exterior, polarisation)
        .unwrap();

    let mode = state.mode().unwrap();

    let response = mode.fields(&sampling()).unwrap();

    let response = response.quantity();

    assert_tm_structure(
        response.electric().value(),
        response.magnetic().value(),
        VALUE_TOLERANCE,
    );

    assert_tm_structure(
        response.electric().axis0(),
        response.magnetic().axis0(),
        FIRST_DERIVATIVE_TOLERANCE,
    );

    assert_tm_structure(
        response.electric().axis1(),
        response.magnetic().axis1(),
        FIRST_DERIVATIVE_TOLERANCE,
    );

    assert_tm_structure(
        response.electric().axis0_axis0(),
        response.magnetic().axis0_axis0(),
        SECOND_DERIVATIVE_TOLERANCE,
    );

    assert_tm_structure(
        response.electric().axis0_axis1(),
        response.magnetic().axis0_axis1(),
        SECOND_DERIVATIVE_TOLERANCE,
    );

    assert_tm_structure(
        response.electric().axis1_axis1(),
        response.magnetic().axis1_axis1(),
        SECOND_DERIVATIVE_TOLERANCE,
    );

    assert_all_finite(response.electric().axis0(), response.magnetic().axis0());

    assert_all_finite(response.electric().axis1(), response.magnetic().axis1());

    assert_all_finite(
        response.electric().axis0_axis0(),
        response.magnetic().axis0_axis0(),
    );

    assert_all_finite(
        response.electric().axis0_axis1(),
        response.magnetic().axis0_axis1(),
    );

    assert_all_finite(
        response.electric().axis1_axis1(),
        response.magnetic().axis1_axis1(),
    );
}

// -----------------------------------------------------------------------------
// Maxwell interface conditions
// -----------------------------------------------------------------------------

#[test]
fn modal_fields_satisfy_tangential_interface_continuity() {
    let stack = two_layer_stack();
    let request = interface_sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for_each_value_backend!(stack, evaluator, {
            let coordinates = value_coordinates();

            let exterior = exterior(evaluator.stack(), &coordinates, polarisation);

            let state = evaluator
                .retain(coordinates, exterior, polarisation)
                .unwrap();

            let mode = state.mode().unwrap();

            let response = mode.fields(&request).unwrap();

            let response = response.quantity();

            assert_eq!(response.electric().value().x().shape(), &[6]);

            match polarisation {
                Polarisation::TransverseElectric => {
                    assert_te_interface_continuity(
                        response.electric().value(),
                        response.magnetic().value(),
                        VALUE_TOLERANCE,
                    );
                }

                Polarisation::TransverseMagnetic => {
                    assert_tm_interface_continuity(
                        response.electric().value(),
                        response.magnetic().value(),
                        VALUE_TOLERANCE,
                    );
                }
            }
        });
    }
}

#[test]
fn first_spectral_modal_field_derivatives_satisfy_tangential_interface_continuity() {
    let stack = two_layer_stack();
    let request = interface_sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        {
            let evaluator =
                ComplexPlaneEvaluator::<HoloJ1, _, _>::compile(&stack, Scatter2::new()).unwrap();

            let coordinates = first_spectral_coordinates();

            let exterior = exterior(evaluator.stack(), &coordinates, polarisation);

            let state = evaluator
                .retain(coordinates, exterior, polarisation)
                .unwrap();

            let mode = state.mode().unwrap();

            let response = mode.fields(&request).unwrap();

            let response = response.quantity();

            match polarisation {
                Polarisation::TransverseElectric => {
                    assert_te_interface_continuity(
                        response.electric().value(),
                        response.magnetic().value(),
                        VALUE_TOLERANCE,
                    );

                    assert_te_interface_continuity(
                        response.electric().first(),
                        response.magnetic().first(),
                        FIRST_DERIVATIVE_TOLERANCE,
                    );
                }

                Polarisation::TransverseMagnetic => {
                    assert_tm_interface_continuity(
                        response.electric().value(),
                        response.magnetic().value(),
                        VALUE_TOLERANCE,
                    );

                    assert_tm_interface_continuity(
                        response.electric().first(),
                        response.magnetic().first(),
                        FIRST_DERIVATIVE_TOLERANCE,
                    );
                }
            }
        }

        {
            let evaluator =
                ComplexPlaneEvaluator::<HoloJ1, _, _>::compile(&stack, Transfer2::new()).unwrap();

            let coordinates = first_spectral_coordinates();

            let exterior = exterior(evaluator.stack(), &coordinates, polarisation);

            let state = evaluator
                .retain(coordinates, exterior, polarisation)
                .unwrap();

            let mode = state.mode().unwrap();

            let response = mode.fields(&request).unwrap();

            let response = response.quantity();

            match polarisation {
                Polarisation::TransverseElectric => {
                    assert_te_interface_continuity(
                        response.electric().value(),
                        response.magnetic().value(),
                        VALUE_TOLERANCE,
                    );

                    assert_te_interface_continuity(
                        response.electric().first(),
                        response.magnetic().first(),
                        FIRST_DERIVATIVE_TOLERANCE,
                    );
                }

                Polarisation::TransverseMagnetic => {
                    assert_tm_interface_continuity(
                        response.electric().value(),
                        response.magnetic().value(),
                        VALUE_TOLERANCE,
                    );

                    assert_tm_interface_continuity(
                        response.electric().first(),
                        response.magnetic().first(),
                        FIRST_DERIVATIVE_TOLERANCE,
                    );
                }
            }
        }
    }
}

#[test]
fn first_thickness_modal_field_derivatives_satisfy_tangential_interface_continuity() {
    let request = interface_sampling();

    for differentiated_layer in [0, 1] {
        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            {
                let evaluator = ComplexPlaneEvaluator::from_canonical_stack(
                    first_order_geometry_stack(differentiated_layer),
                    Scatter2::new(),
                );

                let coordinates = CanonicalCoordinates::new(
                    HoloJ1::constant(arr0(K0)),
                    HoloJ1::constant(arr0(K_PARALLEL)),
                );

                let exterior = exterior(evaluator.stack(), &coordinates, polarisation);

                let state = evaluator
                    .retain(coordinates, exterior, polarisation)
                    .unwrap();

                let mode = state.mode().unwrap();

                let response = mode.fields(&request).unwrap();

                let response = response.quantity();

                match polarisation {
                    Polarisation::TransverseElectric => {
                        assert_te_interface_continuity(
                            response.electric().value(),
                            response.magnetic().value(),
                            VALUE_TOLERANCE,
                        );

                        assert_te_interface_continuity(
                            response.electric().first(),
                            response.magnetic().first(),
                            FIRST_DERIVATIVE_TOLERANCE,
                        );
                    }

                    Polarisation::TransverseMagnetic => {
                        assert_tm_interface_continuity(
                            response.electric().value(),
                            response.magnetic().value(),
                            VALUE_TOLERANCE,
                        );

                        assert_tm_interface_continuity(
                            response.electric().first(),
                            response.magnetic().first(),
                            FIRST_DERIVATIVE_TOLERANCE,
                        );
                    }
                }
            }

            {
                let evaluator = ComplexPlaneEvaluator::from_canonical_stack(
                    first_order_geometry_stack(differentiated_layer),
                    Transfer2::new(),
                );

                let coordinates = CanonicalCoordinates::new(
                    HoloJ1::constant(arr0(K0)),
                    HoloJ1::constant(arr0(K_PARALLEL)),
                );

                let exterior = exterior(evaluator.stack(), &coordinates, polarisation);

                let state = evaluator
                    .retain(coordinates, exterior, polarisation)
                    .unwrap();

                let mode = state.mode().unwrap();

                let response = mode.fields(&request).unwrap();

                let response = response.quantity();

                match polarisation {
                    Polarisation::TransverseElectric => {
                        assert_te_interface_continuity(
                            response.electric().value(),
                            response.magnetic().value(),
                            VALUE_TOLERANCE,
                        );

                        assert_te_interface_continuity(
                            response.electric().first(),
                            response.magnetic().first(),
                            FIRST_DERIVATIVE_TOLERANCE,
                        );
                    }

                    Polarisation::TransverseMagnetic => {
                        assert_tm_interface_continuity(
                            response.electric().value(),
                            response.magnetic().value(),
                            VALUE_TOLERANCE,
                        );

                        assert_tm_interface_continuity(
                            response.electric().first(),
                            response.magnetic().first(),
                            FIRST_DERIVATIVE_TOLERANCE,
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn second_spectral_modal_field_derivatives_satisfy_tangential_interface_continuity() {
    let stack = two_layer_stack();
    let request = interface_sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for backend in [0, 1] {
            if backend == 0 {
                let evaluator =
                    ComplexPlaneEvaluator::<HoloJ2, _, _>::compile(&stack, Scatter2::new())
                        .unwrap();

                let coordinates = second_spectral_coordinates();

                let exterior = exterior(evaluator.stack(), &coordinates, polarisation);

                let state = evaluator
                    .retain(coordinates, exterior, polarisation)
                    .unwrap();

                let mode = state.mode().unwrap();

                let response = mode.fields(&request).unwrap();

                let response = response.quantity();

                match polarisation {
                    Polarisation::TransverseElectric => {
                        assert_te_interface_continuity(
                            response.electric().value(),
                            response.magnetic().value(),
                            VALUE_TOLERANCE,
                        );

                        assert_te_interface_continuity(
                            response.electric().first(),
                            response.magnetic().first(),
                            FIRST_DERIVATIVE_TOLERANCE,
                        );

                        assert_te_interface_continuity(
                            response.electric().second(),
                            response.magnetic().second(),
                            SECOND_DERIVATIVE_TOLERANCE,
                        );
                    }

                    Polarisation::TransverseMagnetic => {
                        assert_tm_interface_continuity(
                            response.electric().value(),
                            response.magnetic().value(),
                            VALUE_TOLERANCE,
                        );

                        assert_tm_interface_continuity(
                            response.electric().first(),
                            response.magnetic().first(),
                            FIRST_DERIVATIVE_TOLERANCE,
                        );

                        assert_tm_interface_continuity(
                            response.electric().second(),
                            response.magnetic().second(),
                            SECOND_DERIVATIVE_TOLERANCE,
                        );
                    }
                }
            } else {
                let evaluator =
                    ComplexPlaneEvaluator::<HoloJ2, _, _>::compile(&stack, Transfer2::new())
                        .unwrap();

                let coordinates = second_spectral_coordinates();

                let exterior = exterior(evaluator.stack(), &coordinates, polarisation);

                let state = evaluator
                    .retain(coordinates, exterior, polarisation)
                    .unwrap();

                let mode = state.mode().unwrap();

                let response = mode.fields(&request).unwrap();

                let response = response.quantity();

                match polarisation {
                    Polarisation::TransverseElectric => {
                        assert_te_interface_continuity(
                            response.electric().value(),
                            response.magnetic().value(),
                            VALUE_TOLERANCE,
                        );

                        assert_te_interface_continuity(
                            response.electric().first(),
                            response.magnetic().first(),
                            FIRST_DERIVATIVE_TOLERANCE,
                        );

                        assert_te_interface_continuity(
                            response.electric().second(),
                            response.magnetic().second(),
                            SECOND_DERIVATIVE_TOLERANCE,
                        );
                    }

                    Polarisation::TransverseMagnetic => {
                        assert_tm_interface_continuity(
                            response.electric().value(),
                            response.magnetic().value(),
                            VALUE_TOLERANCE,
                        );

                        assert_tm_interface_continuity(
                            response.electric().first(),
                            response.magnetic().first(),
                            FIRST_DERIVATIVE_TOLERANCE,
                        );

                        assert_tm_interface_continuity(
                            response.electric().second(),
                            response.magnetic().second(),
                            SECOND_DERIVATIVE_TOLERANCE,
                        );
                    }
                }
            }
        }
    }
}

// -----------------------------------------------------------------------------
// Modal boundary reconstruction
// -----------------------------------------------------------------------------

#[test]
fn modal_last_layer_state_matches_right_exterior_state() {
    let stack = two_layer_stack();

    let evaluator =
        ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, Scatter2::new()).unwrap();

    let polarisation = Polarisation::TransverseElectric;

    let coordinates = value_coordinates();

    let exterior = exterior(evaluator.stack(), &coordinates, polarisation);

    let state = evaluator
        .retain(coordinates, exterior, polarisation)
        .unwrap();

    let workspace = state.workspace();

    let seed = workspace.modal_boundary_solution().unwrap();

    let layers = workspace.reconstruct_layer_mode_waves(&seed).unwrap();

    let exterior = workspace.reconstruct_exterior_mode_waves(&seed).unwrap();

    let last = layers.last().unwrap();

    let layer_admittance = workspace
        .layer_quantities(layers.len() - 1)
        .unwrap()
        .admittance();

    let right_admittance = workspace.solution().context().right_admittance();

    let layer_waves: crate::observable::BoundaryWaves<_> = last.right().clone().into();

    let exterior_waves: crate::observable::BoundaryWaves<_> = exterior.right().clone().into();

    let layer_state = layer_waves.into_state(&layer_admittance);

    let exterior_state = exterior_waves.into_state(right_admittance);

    assert_complex_close(
        layer_state.field().value()[()],
        exterior_state.field().value()[()],
        VALUE_TOLERANCE,
    );

    assert_complex_close(
        layer_state.secondary().value()[()],
        exterior_state.secondary().value()[()],
        VALUE_TOLERANCE,
    );
}

// -----------------------------------------------------------------------------
// Spatial-response metadata
// -----------------------------------------------------------------------------

#[test]
fn modal_field_response_preserves_resolved_sampling() {
    let stack = two_layer_stack();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        for_each_value_backend!(stack, evaluator, {
            let coordinates = value_coordinates();

            let exterior = exterior(evaluator.stack(), &coordinates, polarisation);

            let state = evaluator
                .retain(coordinates, exterior, polarisation)
                .unwrap();

            let requested = sampling();

            let expected = requested.resolve_canonical(evaluator.stack()).unwrap();

            let mode = state.mode().unwrap();

            let response = mode.fields(&requested).unwrap();

            assert_eq!(response.sampling(), &expected);

            assert_eq!(
                response.quantity().electric().value().x().len(),
                response.sampling().len(),
            );

            assert_eq!(
                response.quantity().magnetic().value().x().len(),
                response.sampling().len(),
            );
        });
    }
}

// -----------------------------------------------------------------------------
// Cross-backend modal equivalence
// -----------------------------------------------------------------------------

fn squared_field_difference(
    first_electric: &VectorField<C, Ix1>,
    first_magnetic: &VectorField<C, Ix1>,
    second_electric: &VectorField<C, Ix1>,
    second_magnetic: &VectorField<C, Ix1>,
    sign: C,
) -> f64 {
    [
        (first_electric.x(), second_electric.x()),
        (first_electric.y(), second_electric.y()),
        (first_electric.z(), second_electric.z()),
        (first_magnetic.x(), second_magnetic.x()),
        (first_magnetic.y(), second_magnetic.y()),
        (first_magnetic.z(), second_magnetic.z()),
    ]
    .into_iter()
    .flat_map(|(first, second)| {
        first
            .iter()
            .zip(second.iter())
            .map(move |(&first, &second)| (first - sign * second).norm_sqr())
    })
    .sum()
}

fn modal_relative_sign(
    first_electric: &VectorField<C, Ix1>,
    first_magnetic: &VectorField<C, Ix1>,
    second_electric: &VectorField<C, Ix1>,
    second_magnetic: &VectorField<C, Ix1>,
) -> C {
    let positive = C::new(1.0, 0.0);
    let negative = C::new(-1.0, 0.0);

    let positive_error = squared_field_difference(
        first_electric,
        first_magnetic,
        second_electric,
        second_magnetic,
        positive,
    );

    let negative_error = squared_field_difference(
        first_electric,
        first_magnetic,
        second_electric,
        second_magnetic,
        negative,
    );

    if positive_error <= negative_error {
        positive
    } else {
        negative
    }
}

fn assert_fields_close_with_sign(
    actual_electric: &VectorField<C, Ix1>,
    actual_magnetic: &VectorField<C, Ix1>,
    expected_electric: &VectorField<C, Ix1>,
    expected_magnetic: &VectorField<C, Ix1>,
    sign: C,
    tolerance: f64,
) {
    for (actual, expected) in [
        (actual_electric.x(), expected_electric.x()),
        (actual_electric.y(), expected_electric.y()),
        (actual_electric.z(), expected_electric.z()),
        (actual_magnetic.x(), expected_magnetic.x()),
        (actual_magnetic.y(), expected_magnetic.y()),
        (actual_magnetic.z(), expected_magnetic.z()),
    ] {
        assert_eq!(actual.shape(), expected.shape());

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_complex_close(actual, sign * expected, tolerance);
        }
    }
}

#[test]
fn normalised_modal_fields_agree_between_backends_up_to_global_sign() {
    let stack = two_layer_stack();
    let request = sampling();

    for polarisation in [
        Polarisation::TransverseElectric,
        Polarisation::TransverseMagnetic,
    ] {
        let scatter =
            ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, Scatter2::new()).unwrap();

        let transfer =
            ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, Transfer2::new()).unwrap();

        let scatter_coordinates = value_coordinates();

        let scatter_exterior = exterior(scatter.stack(), &scatter_coordinates, polarisation);

        let scatter_state = scatter
            .retain(scatter_coordinates, scatter_exterior, polarisation)
            .unwrap();

        let transfer_coordinates = value_coordinates();

        let transfer_exterior = exterior(transfer.stack(), &transfer_coordinates, polarisation);

        let transfer_state = transfer
            .retain(transfer_coordinates, transfer_exterior, polarisation)
            .unwrap();

        let scatter_mode = scatter_state.mode().unwrap();

        let transfer_mode = transfer_state.mode().unwrap();

        let scatter = scatter_mode.fields(&request).unwrap();

        let transfer = transfer_mode.fields(&request).unwrap();

        let scatter_fields = scatter.quantity();

        let transfer_fields = transfer.quantity();

        let sign = modal_relative_sign(
            scatter_fields.electric().value(),
            scatter_fields.magnetic().value(),
            transfer_fields.electric().value(),
            transfer_fields.magnetic().value(),
        );

        assert_fields_close_with_sign(
            scatter_fields.electric().value(),
            scatter_fields.magnetic().value(),
            transfer_fields.electric().value(),
            transfer_fields.magnetic().value(),
            sign,
            VALUE_TOLERANCE,
        );
    }
}
