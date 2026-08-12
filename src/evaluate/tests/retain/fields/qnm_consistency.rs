use approx::assert_relative_eq;
use ndarray::{Array1, ArrayBase, Ix0, Ix1, OwnedRepr};
use num_complex::Complex64;

use crate::{
    ComplexPlane, CoordinateInput, FiniteLayerIndex, PlaneWaveEvaluator, Polarisation,
    backend::{scatter2::Scatter2, transfer2::Transfer2},
    evaluate::mode::raw_layer_integration_inputs_unchecked,
    spatial::{FieldSampling, LayerSampling},
    test_support::planar::{scalar_complex_input, two_layer_stack},
};

type C = Complex64;
type ComplexArray = ArrayBase<OwnedRepr<C>, Ix1>;

const QNM_INTEGRATION_POINTS: usize = 4001;
const QNM_INTEGRATION_TOLERANCE: f64 = 2.0e-10;

fn modal_input() -> CoordinateInput<C, Ix0> {
    scalar_complex_input(C::new(2.5, -0.05), C::new(0.31, 0.02))
}

fn sampling() -> FieldSampling<f64> {
    FieldSampling::new()
        .layer(0, LayerSampling::uniform(QNM_INTEGRATION_POINTS))
        .layer(1, LayerSampling::uniform(QNM_INTEGRATION_POINTS))
}

fn bilinear_square(x: &ComplexArray, y: &ComplexArray, z: &ComplexArray) -> Array1<C> {
    assert_eq!(x.len(), y.len());
    assert_eq!(x.len(), z.len());

    Array1::from_iter(x.iter().zip(y.iter()).zip(z.iter()).map(|((&x, &y), &z)| {
        /*
         * Bilinear Cartesian contraction:
         *
         *     V · V = Vx² + Vy² + Vz²
         *
         * Deliberately no complex conjugation.
         */
        x * x + y * y + z * z
    }))
}

fn integrate_uniform_complex(values: &[C], thickness: f64) -> C {
    assert!(
        values.len() >= 2,
        "numerical integration requires at least two samples",
    );

    let dz = thickness / (values.len() - 1) as f64;

    let interior: C = values[1..values.len() - 1].iter().copied().sum();

    (values[0] * 0.5 + interior + values[values.len() - 1] * 0.5) * dz
}

fn assert_complex_close(actual: C, expected: C, tolerance: f64) {
    assert_relative_eq!(
        actual.re,
        expected.re,
        epsilon = tolerance,
        max_relative = tolerance,
    );

    assert_relative_eq!(
        actual.im,
        expected.im,
        epsilon = tolerance,
        max_relative = tolerance,
    );
}

#[test]
fn scatter_te_sampled_qnm_fields_integrate_to_unit_normalisation() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain_modal(modal_input(), &stack, Polarisation::TransverseElectric)
        .expect("complex retained solve should succeed");

    let mode = state.mode().expect("mode should construct");

    let sampling = sampling();

    /*
     * This reconstructs from PlaneWaveMode::solution(), which is already
     * QNM-normalised.
     */
    let response = mode
        .evaluate_fields(&sampling)
        .expect("normalised modal fields should evaluate");

    let resolved = sampling
        .resolve(mode.state().stack())
        .expect("field sampling should resolve");

    /*
     * Sample exactly the constitutive spectral data entering the analytic
     * QNM normalisation:
     *
     *     W_e = epsilon + k0 d epsilon / d k0
     *     W_m = mu      + k0 d mu      / d k0
     *
     * These remain complex. We deliberately do not convert them to
     * Hermitian energy coefficients.
     */
    let factors = mode
        .state()
        .raw_constitutive_spectral_first_parameters::<ComplexPlane>(&resolved)
        .expect("complex constitutive spectral data should sample")
        .into_brillouin_factors();

    let fields = response.value();

    let electric_square = bilinear_square(
        fields.electric().x(),
        fields.electric().y(),
        fields.electric().z(),
    );

    let magnetic_square = bilinear_square(
        fields.magnetic().x(),
        fields.magnetic().y(),
        fields.magnetic().z(),
    );

    let electric_weight = factors.electric().value();

    let magnetic_weight = factors.magnetic().value();

    assert_eq!(electric_weight.len(), electric_square.len(),);

    assert_eq!(magnetic_weight.len(), magnetic_square.len(),);

    /*
     * QNM bilinear-normalisation density:
     *
     *     rho =
     *         W_e (E · E)
     *       + W_m (H · H)
     *
     * No conjugation.
     * No real part.
     * No factor of 1/4.
     */
    let electric_density = Array1::from_iter(
        electric_weight
            .iter()
            .zip(electric_square.iter())
            .map(|(&weight, &field)| weight * field),
    );

    let magnetic_density = Array1::from_iter(
        magnetic_weight
            .iter()
            .zip(magnetic_square.iter())
            .map(|(&weight, &field)| weight * field),
    );

    let total_density = &electric_density - &magnetic_density;

    let stack = mode.state().stack();

    let layer_count = stack.layers_left_to_right().len();

    assert_eq!(
        layer_count, 2,
        "this test fixture is expected to contain two finite layers",
    );

    assert_eq!(total_density.len(), layer_count * QNM_INTEGRATION_POINTS,);

    let mut electric_total = C::new(0.0, 0.0);

    let mut magnetic_total = C::new(0.0, 0.0);

    let mut total = C::new(0.0, 0.0);

    for layer_index in 0..layer_count {
        let start = layer_index * QNM_INTEGRATION_POINTS;

        let end = start + QNM_INTEGRATION_POINTS;

        let thickness = stack
            .layers_left_to_right()
            .get(layer_index)
            .expect("finite layer should exist")
            .thickness()
            .as_centimetres();

        let electric_layer = integrate_uniform_complex(
            &electric_density
                .as_slice()
                .expect("sampled electric density should be contiguous")[start..end],
            thickness,
        );

        let magnetic_layer = integrate_uniform_complex(
            &magnetic_density
                .as_slice()
                .expect("sampled magnetic density should be contiguous")[start..end],
            thickness,
        );

        let total_layer = integrate_uniform_complex(
            &total_density
                .as_slice()
                .expect("sampled total density should be contiguous")[start..end],
            thickness,
        );

        /*
         * Check the numerical decomposition independently for each layer.
         */
        assert_complex_close(
            total_layer,
            electric_layer - magnetic_layer,
            QNM_INTEGRATION_TOLERANCE,
        );

        electric_total += electric_layer;

        magnetic_total += magnetic_layer;

        total += total_layer;
    }

    assert_complex_close(
        total,
        electric_total - magnetic_total,
        QNM_INTEGRATION_TOLERANCE,
    );

    assert_complex_close(total, C::new(1.0, 0.0), QNM_INTEGRATION_TOLERANCE);
}

#[test]
fn scatter_tm_sampled_qnm_fields_integrate_to_unit_normalisation() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain_modal(modal_input(), &stack, Polarisation::TransverseMagnetic)
        .expect("complex retained solve should succeed");

    let mode = state.mode().expect("mode should construct");

    let sampling = sampling();

    /*
     * This reconstructs from PlaneWaveMode::solution(), which is already
     * QNM-normalised.
     */
    let response = mode
        .evaluate_fields(&sampling)
        .expect("normalised modal fields should evaluate");

    let resolved = sampling
        .resolve(mode.state().stack())
        .expect("field sampling should resolve");

    /*
     * Sample exactly the constitutive spectral data entering the analytic
     * QNM normalisation:
     *
     *     W_e = epsilon + k0 d epsilon / d k0
     *     W_m = mu      + k0 d mu      / d k0
     *
     * These remain complex. We deliberately do not convert them to
     * Hermitian energy coefficients.
     */
    let factors = mode
        .state()
        .raw_constitutive_spectral_first_parameters::<ComplexPlane>(&resolved)
        .expect("complex constitutive spectral data should sample")
        .into_brillouin_factors();

    let fields = response.value();

    let electric_square = bilinear_square(
        fields.electric().x(),
        fields.electric().y(),
        fields.electric().z(),
    );

    let magnetic_square = bilinear_square(
        fields.magnetic().x(),
        fields.magnetic().y(),
        fields.magnetic().z(),
    );

    let electric_weight = factors.electric().value();

    let magnetic_weight = factors.magnetic().value();

    assert_eq!(electric_weight.len(), electric_square.len(),);

    assert_eq!(magnetic_weight.len(), magnetic_square.len(),);

    /*
     * QNM bilinear-normalisation density:
     *
     *     rho =
     *         W_e (E · E)
     *       + W_m (H · H)
     *
     * No conjugation.
     * No real part.
     * No factor of 1/4.
     */
    let electric_density = Array1::from_iter(
        electric_weight
            .iter()
            .zip(electric_square.iter())
            .map(|(&weight, &field)| weight * field),
    );

    let magnetic_density = Array1::from_iter(
        magnetic_weight
            .iter()
            .zip(magnetic_square.iter())
            .map(|(&weight, &field)| weight * field),
    );

    let total_density = &electric_density - &magnetic_density;

    let stack = mode.state().stack();

    let layer_count = stack.layers_left_to_right().len();

    assert_eq!(
        layer_count, 2,
        "this test fixture is expected to contain two finite layers",
    );

    assert_eq!(total_density.len(), layer_count * QNM_INTEGRATION_POINTS,);

    let mut electric_total = C::new(0.0, 0.0);

    let mut magnetic_total = C::new(0.0, 0.0);

    let mut total = C::new(0.0, 0.0);

    for layer_index in 0..layer_count {
        let start = layer_index * QNM_INTEGRATION_POINTS;

        let end = start + QNM_INTEGRATION_POINTS;

        let thickness = stack
            .layers_left_to_right()
            .get(layer_index)
            .expect("finite layer should exist")
            .thickness()
            .as_centimetres();

        let electric_layer = integrate_uniform_complex(
            &electric_density
                .as_slice()
                .expect("sampled electric density should be contiguous")[start..end],
            thickness,
        );

        let magnetic_layer = integrate_uniform_complex(
            &magnetic_density
                .as_slice()
                .expect("sampled magnetic density should be contiguous")[start..end],
            thickness,
        );

        let total_layer = integrate_uniform_complex(
            &total_density
                .as_slice()
                .expect("sampled total density should be contiguous")[start..end],
            thickness,
        );

        /*
         * Check the numerical decomposition independently for each layer.
         */
        assert_complex_close(
            total_layer,
            electric_layer - magnetic_layer,
            QNM_INTEGRATION_TOLERANCE,
        );

        electric_total += electric_layer;

        magnetic_total += magnetic_layer;

        total += total_layer;
    }

    assert_complex_close(
        total,
        electric_total - magnetic_total,
        QNM_INTEGRATION_TOLERANCE,
    );

    assert_complex_close(total, C::new(1.0, 0.0), QNM_INTEGRATION_TOLERANCE);
}

#[test]
fn transfer_te_sampled_qnm_fields_integrate_to_unit_normalisation() {
    let evaluator = PlaneWaveEvaluator::new(Transfer2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain_modal(modal_input(), &stack, Polarisation::TransverseElectric)
        .expect("complex retained solve should succeed");

    let mode = state.mode().expect("mode should construct");

    let sampling = sampling();

    /*
     * This reconstructs from PlaneWaveMode::solution(), which is already
     * QNM-normalised.
     */
    let response = mode
        .evaluate_fields(&sampling)
        .expect("normalised modal fields should evaluate");

    let resolved = sampling
        .resolve(mode.state().stack())
        .expect("field sampling should resolve");

    /*
     * Sample exactly the constitutive spectral data entering the analytic
     * QNM normalisation:
     *
     *     W_e = epsilon + k0 d epsilon / d k0
     *     W_m = mu      + k0 d mu      / d k0
     *
     * These remain complex. We deliberately do not convert them to
     * Hermitian energy coefficients.
     */
    let factors = mode
        .state()
        .raw_constitutive_spectral_first_parameters::<ComplexPlane>(&resolved)
        .expect("complex constitutive spectral data should sample")
        .into_brillouin_factors();

    let fields = response.value();

    let electric_square = bilinear_square(
        fields.electric().x(),
        fields.electric().y(),
        fields.electric().z(),
    );

    let magnetic_square = bilinear_square(
        fields.magnetic().x(),
        fields.magnetic().y(),
        fields.magnetic().z(),
    );

    let electric_weight = factors.electric().value();

    let magnetic_weight = factors.magnetic().value();

    assert_eq!(electric_weight.len(), electric_square.len(),);

    assert_eq!(magnetic_weight.len(), magnetic_square.len(),);

    /*
     * QNM bilinear-normalisation density:
     *
     *     rho =
     *         W_e (E · E)
     *       + W_m (H · H)
     *
     * No conjugation.
     * No real part.
     * No factor of 1/4.
     */
    let electric_density = Array1::from_iter(
        electric_weight
            .iter()
            .zip(electric_square.iter())
            .map(|(&weight, &field)| weight * field),
    );

    let magnetic_density = Array1::from_iter(
        magnetic_weight
            .iter()
            .zip(magnetic_square.iter())
            .map(|(&weight, &field)| weight * field),
    );

    let total_density = &electric_density - &magnetic_density;

    let stack = mode.state().stack();

    let layer_count = stack.layers_left_to_right().len();

    assert_eq!(
        layer_count, 2,
        "this test fixture is expected to contain two finite layers",
    );

    assert_eq!(total_density.len(), layer_count * QNM_INTEGRATION_POINTS,);

    let mut electric_total = C::new(0.0, 0.0);

    let mut magnetic_total = C::new(0.0, 0.0);

    let mut total = C::new(0.0, 0.0);

    for layer_index in 0..layer_count {
        let start = layer_index * QNM_INTEGRATION_POINTS;

        let end = start + QNM_INTEGRATION_POINTS;

        let thickness = stack
            .layers_left_to_right()
            .get(layer_index)
            .expect("finite layer should exist")
            .thickness()
            .as_centimetres();

        let electric_layer = integrate_uniform_complex(
            &electric_density
                .as_slice()
                .expect("sampled electric density should be contiguous")[start..end],
            thickness,
        );

        let magnetic_layer = integrate_uniform_complex(
            &magnetic_density
                .as_slice()
                .expect("sampled magnetic density should be contiguous")[start..end],
            thickness,
        );

        let total_layer = integrate_uniform_complex(
            &total_density
                .as_slice()
                .expect("sampled total density should be contiguous")[start..end],
            thickness,
        );

        /*
         * Check the numerical decomposition independently for each layer.
         */
        assert_complex_close(
            total_layer,
            electric_layer - magnetic_layer,
            QNM_INTEGRATION_TOLERANCE,
        );

        electric_total += electric_layer;

        magnetic_total += magnetic_layer;

        total += total_layer;
    }

    assert_complex_close(
        total,
        electric_total - magnetic_total,
        QNM_INTEGRATION_TOLERANCE,
    );

    assert_complex_close(total, C::new(1.0, 0.0), QNM_INTEGRATION_TOLERANCE);
}

#[test]
fn transfer_tm_sampled_qnm_fields_integrate_to_unit_normalisation() {
    let evaluator = PlaneWaveEvaluator::new(Transfer2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain_modal(modal_input(), &stack, Polarisation::TransverseMagnetic)
        .expect("complex retained solve should succeed");

    let mode = state.mode().expect("mode should construct");

    let sampling = sampling();

    /*
     * This reconstructs from PlaneWaveMode::solution(), which is already
     * QNM-normalised.
     */
    let response = mode
        .evaluate_fields(&sampling)
        .expect("normalised modal fields should evaluate");

    let resolved = sampling
        .resolve(mode.state().stack())
        .expect("field sampling should resolve");

    /*
     * Sample exactly the constitutive spectral data entering the analytic
     * QNM normalisation:
     *
     *     W_e = epsilon + k0 d epsilon / d k0
     *     W_m = mu      + k0 d mu      / d k0
     *
     * These remain complex. We deliberately do not convert them to
     * Hermitian energy coefficients.
     */
    let factors = mode
        .state()
        .raw_constitutive_spectral_first_parameters::<ComplexPlane>(&resolved)
        .expect("complex constitutive spectral data should sample")
        .into_brillouin_factors();

    let fields = response.value();

    let electric_square = bilinear_square(
        fields.electric().x(),
        fields.electric().y(),
        fields.electric().z(),
    );

    let magnetic_square = bilinear_square(
        fields.magnetic().x(),
        fields.magnetic().y(),
        fields.magnetic().z(),
    );

    let electric_weight = factors.electric().value();

    let magnetic_weight = factors.magnetic().value();

    assert_eq!(electric_weight.len(), electric_square.len(),);

    assert_eq!(magnetic_weight.len(), magnetic_square.len(),);

    /*
     * QNM bilinear-normalisation density:
     *
     *     rho =
     *         W_e (E · E)
     *       + W_m (H · H)
     *
     * No conjugation.
     * No real part.
     * No factor of 1/4.
     */
    let electric_density = Array1::from_iter(
        electric_weight
            .iter()
            .zip(electric_square.iter())
            .map(|(&weight, &field)| weight * field),
    );

    let magnetic_density = Array1::from_iter(
        magnetic_weight
            .iter()
            .zip(magnetic_square.iter())
            .map(|(&weight, &field)| weight * field),
    );

    let total_density = &electric_density - &magnetic_density;

    let stack = mode.state().stack();

    let layer_count = stack.layers_left_to_right().len();

    assert_eq!(
        layer_count, 2,
        "this test fixture is expected to contain two finite layers",
    );

    assert_eq!(total_density.len(), layer_count * QNM_INTEGRATION_POINTS,);

    let mut electric_total = C::new(0.0, 0.0);

    let mut magnetic_total = C::new(0.0, 0.0);

    let mut total = C::new(0.0, 0.0);

    for layer_index in 0..layer_count {
        let start = layer_index * QNM_INTEGRATION_POINTS;

        let end = start + QNM_INTEGRATION_POINTS;

        let thickness = stack
            .layers_left_to_right()
            .get(layer_index)
            .expect("finite layer should exist")
            .thickness()
            .as_centimetres();

        let electric_layer = integrate_uniform_complex(
            &electric_density
                .as_slice()
                .expect("sampled electric density should be contiguous")[start..end],
            thickness,
        );

        let magnetic_layer = integrate_uniform_complex(
            &magnetic_density
                .as_slice()
                .expect("sampled magnetic density should be contiguous")[start..end],
            thickness,
        );

        let total_layer = integrate_uniform_complex(
            &total_density
                .as_slice()
                .expect("sampled total density should be contiguous")[start..end],
            thickness,
        );

        /*
         * Check the numerical decomposition independently for each layer.
         */
        assert_complex_close(
            total_layer,
            electric_layer - magnetic_layer,
            QNM_INTEGRATION_TOLERANCE,
        );

        electric_total += electric_layer;

        magnetic_total += magnetic_layer;

        total += total_layer;
    }

    assert_complex_close(
        total,
        electric_total - magnetic_total,
        QNM_INTEGRATION_TOLERANCE,
    );

    assert_complex_close(total, C::new(1.0, 0.0), QNM_INTEGRATION_TOLERANCE);
}

#[test]
fn scatter_te_sampled_qnm_components_match_analytic_normalisation_components() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain_modal(modal_input(), &stack, Polarisation::TransverseElectric)
        .expect("retained modal solve should succeed");

    let mode = state.mode().expect("mode construction should succeed");

    let sampling = FieldSampling::new()
        .layer(0, LayerSampling::uniform(QNM_INTEGRATION_POINTS))
        .layer(1, LayerSampling::uniform(QNM_INTEGRATION_POINTS));

    let resolved = sampling.resolve(&stack).expect("sampling should resolve");

    let factors = mode
        .state()
        .raw_constitutive_spectral_first_parameters::<ComplexPlane>(&resolved)
        .expect("constitutive spectral sampling should succeed")
        .into_brillouin_factors();

    /*
     * Preserve the analytic normalization before evaluate_fields consumes
     * the mode.
     */
    let raw_electric = mode.raw_normalisation().electric().value()[()];

    let raw_magnetic = mode.raw_normalisation().magnetic().value()[()];

    let raw_total = mode.raw_normalisation().total().value()[()];

    let expected_electric = raw_electric / raw_total;

    let expected_magnetic = raw_magnetic / raw_total;

    let expected_total = expected_electric - expected_magnetic;

    let response = mode
        .evaluate_fields(&sampling)
        .expect("modal field evaluation should succeed");

    let fields = response.value();

    let electric_square = bilinear_square(
        fields.electric().x(),
        fields.electric().y(),
        fields.electric().z(),
    );

    let magnetic_square = bilinear_square(
        fields.magnetic().x(),
        fields.magnetic().y(),
        fields.magnetic().z(),
    );

    let electric_weight = factors.electric().value();

    let magnetic_weight = factors.magnetic().value();

    let electric_density = Array1::from_iter(
        electric_weight
            .iter()
            .zip(electric_square.iter())
            .map(|(&weight, &field)| weight * field),
    );

    let magnetic_density = Array1::from_iter(
        magnetic_weight
            .iter()
            .zip(magnetic_square.iter())
            .map(|(&weight, &field)| weight * field),
    );

    let thicknesses: Vec<f64> = stack
        .layers_left_to_right()
        .iter()
        .map(|layer| layer.thickness().as_centimetres())
        .collect();

    let mut sampled_electric = C::new(0.0, 0.0);

    let mut sampled_magnetic = C::new(0.0, 0.0);

    for (layer_index, &thickness) in thicknesses.iter().enumerate() {
        let start = layer_index * QNM_INTEGRATION_POINTS;

        let end = start + QNM_INTEGRATION_POINTS;

        sampled_electric += integrate_uniform_complex(
            &electric_density
                .as_slice()
                .expect("electric density should be contiguous")[start..end],
            thickness,
        );

        sampled_magnetic += integrate_uniform_complex(
            &magnetic_density
                .as_slice()
                .expect("magnetic density should be contiguous")[start..end],
            thickness,
        );
    }

    let sampled_total = sampled_electric - sampled_magnetic;

    println!("TE");
    println!("sampled electric = {sampled_electric:?}");
    println!("analytic electric = {expected_electric:?}");
    println!("sampled magnetic = {sampled_magnetic:?}");
    println!("analytic magnetic = {expected_magnetic:?}");
    println!("sampled total = {sampled_total:?}");
    println!("analytic total = {expected_total:?}");

    assert_complex_close(sampled_total, C::new(1.0, 0.0), QNM_INTEGRATION_TOLERANCE);
}

#[test]
fn scatter_tm_sampled_qnm_components_match_analytic_normalisation_components() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain_modal(modal_input(), &stack, Polarisation::TransverseMagnetic)
        .expect("retained modal solve should succeed");

    let mode = state.mode().expect("mode construction should succeed");

    let sampling = FieldSampling::new()
        .layer(0, LayerSampling::uniform(QNM_INTEGRATION_POINTS))
        .layer(1, LayerSampling::uniform(QNM_INTEGRATION_POINTS));

    let resolved = sampling.resolve(&stack).expect("sampling should resolve");

    let factors = mode
        .state()
        .raw_constitutive_spectral_first_parameters::<ComplexPlane>(&resolved)
        .expect("constitutive spectral sampling should succeed")
        .into_brillouin_factors();

    let raw_electric = mode.raw_normalisation().electric().value()[()];

    let raw_magnetic = mode.raw_normalisation().magnetic().value()[()];

    let raw_total = mode.raw_normalisation().total().value()[()];

    let expected_electric = raw_electric / raw_total;

    let expected_magnetic = raw_magnetic / raw_total;

    let expected_total = expected_electric - expected_magnetic;

    let response = mode
        .evaluate_fields(&sampling)
        .expect("modal field evaluation should succeed");

    let fields = response.value();

    let electric_square = bilinear_square(
        fields.electric().x(),
        fields.electric().y(),
        fields.electric().z(),
    );

    let magnetic_square = bilinear_square(
        fields.magnetic().x(),
        fields.magnetic().y(),
        fields.magnetic().z(),
    );

    let electric_weight = factors.electric().value();

    let magnetic_weight = factors.magnetic().value();

    let electric_density = Array1::from_iter(
        electric_weight
            .iter()
            .zip(electric_square.iter())
            .map(|(&weight, &field)| weight * field),
    );

    let magnetic_density = Array1::from_iter(
        magnetic_weight
            .iter()
            .zip(magnetic_square.iter())
            .map(|(&weight, &field)| weight * field),
    );

    let thicknesses: Vec<f64> = stack
        .layers_left_to_right()
        .iter()
        .map(|layer| layer.thickness().as_centimetres())
        .collect();

    let mut sampled_electric = C::new(0.0, 0.0);

    let mut sampled_magnetic = C::new(0.0, 0.0);

    for (layer_index, &thickness) in thicknesses.iter().enumerate() {
        let start = layer_index * QNM_INTEGRATION_POINTS;

        let end = start + QNM_INTEGRATION_POINTS;

        sampled_electric += integrate_uniform_complex(
            &electric_density
                .as_slice()
                .expect("electric density should be contiguous")[start..end],
            thickness,
        );

        sampled_magnetic += integrate_uniform_complex(
            &magnetic_density
                .as_slice()
                .expect("magnetic density should be contiguous")[start..end],
            thickness,
        );
    }

    let sampled_total = sampled_electric - sampled_magnetic;

    println!("TM");
    println!("sampled electric = {sampled_electric:?}");
    println!("analytic electric = {expected_electric:?}");
    println!("sampled magnetic = {sampled_magnetic:?}");
    println!("analytic magnetic = {expected_magnetic:?}");
    println!("sampled total = {sampled_total:?}");
    println!("analytic total = {expected_total:?}");

    assert_complex_close(sampled_total, C::new(1.0, 0.0), QNM_INTEGRATION_TOLERANCE);
}

#[test]
fn scatter_te_sampled_qnm_normalisation_matches_analytic_layers() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain_modal(modal_input(), &stack, Polarisation::TransverseElectric)
        .expect("retained modal solve should succeed");

    let mode = state.mode().expect("mode construction should succeed");

    let sampling = FieldSampling::new()
        .layer(0, LayerSampling::uniform(QNM_INTEGRATION_POINTS))
        .layer(1, LayerSampling::uniform(QNM_INTEGRATION_POINTS));

    let resolved = sampling.resolve(&stack).expect("sampling should resolve");

    let factors = mode
        .state()
        .raw_constitutive_spectral_first_parameters::<ComplexPlane>(&resolved)
        .expect("constitutive spectral sampling should succeed")
        .into_brillouin_factors();

    /*
     * Analytic normalization, but using the already-scaled modal solution.
     * Therefore the sum of these layers should already be unity.
     */
    let coordinates = mode.state().problem().coordinates();

    let analytic_layers =
        raw_layer_integration_inputs_unchecked(mode.solution(), mode.state().workspace())
            .expect("layer inputs should assemble")
            .integrate_bilinear()
            .into_brillouin_layers::<ComplexPlane, _>(
                mode.state()
                    .problem()
                    .stack()
                    .layers()
                    .iter()
                    .map(|layer| layer.material()),
                coordinates.vacuum_angular_wavenumber(),
            )
            .expect("Brillouin layers should assemble")
            .into_qnm_normalisation(
                coordinates.vacuum_angular_wavenumber(),
                coordinates.parallel_angular_wavenumber(),
            );

    let response = mode
        .evaluate_fields(&sampling)
        .expect("modal field evaluation should succeed");

    let fields = response.value();

    let electric_square = bilinear_square(
        fields.electric().x(),
        fields.electric().y(),
        fields.electric().z(),
    );

    let magnetic_square = bilinear_square(
        fields.magnetic().x(),
        fields.magnetic().y(),
        fields.magnetic().z(),
    );

    let electric_weight = factors.electric().value();

    let magnetic_weight = factors.magnetic().value();

    let electric_density = Array1::from_iter(
        electric_weight
            .iter()
            .zip(electric_square.iter())
            .map(|(&weight, &field)| weight * field),
    );

    let magnetic_density = Array1::from_iter(
        magnetic_weight
            .iter()
            .zip(magnetic_square.iter())
            .map(|(&weight, &field)| weight * field),
    );

    let thicknesses: Vec<f64> = stack
        .layers_left_to_right()
        .iter()
        .map(|layer| layer.thickness().as_centimetres())
        .collect();

    assert_eq!(analytic_layers.len(), thicknesses.len(),);

    for (layer_index, &thickness) in thicknesses.iter().enumerate() {
        let start = layer_index * QNM_INTEGRATION_POINTS;

        let end = start + QNM_INTEGRATION_POINTS;

        let sampled_electric =
            integrate_uniform_complex(&electric_density.as_slice().unwrap()[start..end], thickness);

        let sampled_magnetic =
            integrate_uniform_complex(&magnetic_density.as_slice().unwrap()[start..end], thickness);

        let sampled_total = sampled_electric - sampled_magnetic;

        let analytic = analytic_layers
            .get(FiniteLayerIndex::new(layer_index))
            .expect("analytic layer should exist");

        let analytic_electric = analytic.electric().value()[()];

        let analytic_magnetic = analytic.magnetic().value()[()];

        let analytic_total = analytic.total().value()[()];

        println!("layer {layer_index}");
        println!("  electric sampled={sampled_electric:?}, analytic={analytic_electric:?}");
        println!("  magnetic sampled={sampled_magnetic:?}, analytic={analytic_magnetic:?}");
        println!("  total    sampled={sampled_total:?}, analytic={analytic_total:?}");

        assert_complex_close(sampled_total, analytic_total, QNM_INTEGRATION_TOLERANCE);
    }
}

#[test]
fn scatter_tm_sampled_qnm_normalisation_matches_analytic_layers() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain_modal(modal_input(), &stack, Polarisation::TransverseMagnetic)
        .expect("retained modal solve should succeed");

    let mode = state.mode().expect("mode construction should succeed");

    let sampling = FieldSampling::new()
        .layer(0, LayerSampling::uniform(QNM_INTEGRATION_POINTS))
        .layer(1, LayerSampling::uniform(QNM_INTEGRATION_POINTS));

    let resolved = sampling.resolve(&stack).expect("sampling should resolve");

    let factors = mode
        .state()
        .raw_constitutive_spectral_first_parameters::<ComplexPlane>(&resolved)
        .expect("constitutive spectral sampling should succeed")
        .into_brillouin_factors();

    /*
     * Analytic normalization, but using the already-scaled modal solution.
     * Therefore the sum of these layers should already be unity.
     */
    let coordinates = mode.state().problem().coordinates();

    let analytic_layers =
        raw_layer_integration_inputs_unchecked(mode.solution(), mode.state().workspace())
            .expect("layer inputs should assemble")
            .integrate_bilinear()
            .into_brillouin_layers::<ComplexPlane, _>(
                mode.state()
                    .problem()
                    .stack()
                    .layers()
                    .iter()
                    .map(|layer| layer.material()),
                coordinates.vacuum_angular_wavenumber(),
            )
            .expect("Brillouin layers should assemble")
            .into_qnm_normalisation(
                coordinates.vacuum_angular_wavenumber(),
                coordinates.parallel_angular_wavenumber(),
            );

    let response = mode
        .evaluate_fields(&sampling)
        .expect("modal field evaluation should succeed");

    let fields = response.value();

    let electric_square = bilinear_square(
        fields.electric().x(),
        fields.electric().y(),
        fields.electric().z(),
    );

    let magnetic_square = bilinear_square(
        fields.magnetic().x(),
        fields.magnetic().y(),
        fields.magnetic().z(),
    );

    let electric_weight = factors.electric().value();

    let magnetic_weight = factors.magnetic().value();

    let electric_density = Array1::from_iter(
        electric_weight
            .iter()
            .zip(electric_square.iter())
            .map(|(&weight, &field)| weight * field),
    );

    let magnetic_density = Array1::from_iter(
        magnetic_weight
            .iter()
            .zip(magnetic_square.iter())
            .map(|(&weight, &field)| weight * field),
    );

    let thicknesses: Vec<f64> = stack
        .layers_left_to_right()
        .iter()
        .map(|layer| layer.thickness().as_centimetres())
        .collect();

    assert_eq!(analytic_layers.len(), thicknesses.len(),);

    for (layer_index, &thickness) in thicknesses.iter().enumerate() {
        let start = layer_index * QNM_INTEGRATION_POINTS;

        let end = start + QNM_INTEGRATION_POINTS;

        let sampled_electric =
            integrate_uniform_complex(&electric_density.as_slice().unwrap()[start..end], thickness);

        let sampled_magnetic =
            integrate_uniform_complex(&magnetic_density.as_slice().unwrap()[start..end], thickness);

        let sampled_total = sampled_electric - sampled_magnetic;

        let analytic = analytic_layers
            .get(FiniteLayerIndex::new(layer_index))
            .expect("analytic layer should exist");

        let analytic_electric = analytic.electric().value()[()];

        let analytic_magnetic = analytic.magnetic().value()[()];

        let analytic_total = analytic.total().value()[()];

        println!("layer {layer_index}");
        println!("  electric sampled={sampled_electric:?}, analytic={analytic_electric:?}");
        println!("  magnetic sampled={sampled_magnetic:?}, analytic={analytic_magnetic:?}");
        println!("  total    sampled={sampled_total:?}, analytic={analytic_total:?}");

        assert_complex_close(sampled_total, analytic_total, QNM_INTEGRATION_TOLERANCE);
    }
}

#[test]
fn scatter_te_sampled_qnm_normalisation_converges_with_sampling_density() {
    let evaluator = PlaneWaveEvaluator::new(Scatter2::new());

    let stack = two_layer_stack();

    let state = evaluator
        .retain_modal(modal_input(), &stack, Polarisation::TransverseElectric)
        .expect("retained modal solve should succeed");

    /*
     * Reconstruct a fresh mode for each sampling density because
     * evaluate_fields consumes the mode.
     */
    for points in [251usize, 501, 1001, 2001, 4001, 8001, 16001] {
        let mode = state.mode().expect("mode construction should succeed");

        let sampling = FieldSampling::new()
            .layer(0, LayerSampling::uniform(points))
            .layer(1, LayerSampling::uniform(points));

        let resolved = sampling.resolve(&stack).expect("sampling should resolve");

        let factors = mode
            .state()
            .raw_constitutive_spectral_first_parameters::<ComplexPlane>(&resolved)
            .expect("constitutive spectral sampling should succeed")
            .into_brillouin_factors();

        let response = mode
            .evaluate_fields(&sampling)
            .expect("modal fields should evaluate");

        let fields = response.value();

        let electric_square = bilinear_square(
            fields.electric().x(),
            fields.electric().y(),
            fields.electric().z(),
        );

        let magnetic_square = bilinear_square(
            fields.magnetic().x(),
            fields.magnetic().y(),
            fields.magnetic().z(),
        );

        let electric_density = Array1::from_iter(
            factors
                .electric()
                .value()
                .iter()
                .zip(electric_square.iter())
                .map(|(&weight, &field)| weight * field),
        );

        let magnetic_density = Array1::from_iter(
            factors
                .magnetic()
                .value()
                .iter()
                .zip(magnetic_square.iter())
                .map(|(&weight, &field)| weight * field),
        );

        let total_density = &electric_density - &magnetic_density;

        let thicknesses: Vec<f64> = stack
            .layers_left_to_right()
            .iter()
            .map(|layer| layer.thickness().as_centimetres())
            .collect();

        let mut total = C::new(0.0, 0.0);

        for (layer_index, &thickness) in thicknesses.iter().enumerate() {
            let start = layer_index * points;

            let end = start + points;

            total += integrate_uniform_complex(
                &total_density.as_slice().unwrap()[start..end],
                thickness,
            );
        }

        println!(
            "points={points:5}, norm={total:?}, error={:e}",
            (total - C::new(1.0, 0.0)).norm(),
        );
    }
}
