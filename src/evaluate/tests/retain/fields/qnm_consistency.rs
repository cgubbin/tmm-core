use approx::assert_relative_eq;
use ndarray::{Array1, ArrayBase, Ix0, Ix1, OwnedRepr, arr0};
use num_complex::Complex64;

use crate::{
    ComplexPlane, ComplexPlaneEvaluator, ExteriorWavevectors, FiniteLayerIndex, Polarisation,
    backend::{Scatter2, Transfer2, evaluate_exterior_wavevectors},
    evaluate::complex_plane::mode::raw_layer_integration_inputs_unchecked,
    input::{CanonicalCoordinates, CanonicalStack},
    spatial::{FieldSampling, LayerSampling},
    test_support::{jet::HoloJ0, planar::two_layer_stack},
};

type C = Complex64;
type ComplexArray = ArrayBase<OwnedRepr<C>, Ix1>;

const K0: C = C::new(2.5, -0.05);
const K_PARALLEL: C = C::new(0.31, 0.02);

const QNM_INTEGRATION_POINTS: usize = 4001;
const QNM_INTEGRATION_TOLERANCE: f64 = 2.0e-10;
const QNM_COMPONENT_INTEGRATION_TOLERANCE: f64 = 2.0e-4;

// -----------------------------------------------------------------------------
// Complex-plane fixtures
// -----------------------------------------------------------------------------

fn modal_coordinates() -> CanonicalCoordinates<HoloJ0> {
    CanonicalCoordinates::new(
        HoloJ0::constant(arr0(K0)),
        HoloJ0::constant(arr0(K_PARALLEL)),
    )
}

fn principal_exterior<M>(
    stack: &CanonicalStack<M, HoloJ0>,
    coordinates: &CanonicalCoordinates<HoloJ0>,
    _polarisation: Polarisation,
) -> ExteriorWavevectors<HoloJ0>
where
    HoloJ0: crate::material::ConstitutiveLift<ComplexPlane, M> + Clone,
    ComplexPlane: crate::material::ConstitutiveEvaluator<C, Ix0, M>,
{
    evaluate_exterior_wavevectors::<ComplexPlane, M, HoloJ0>(
        coordinates,
        stack.left_exterior(),
        stack.right_exterior(),
    )
}

fn sampling() -> FieldSampling<f64> {
    FieldSampling::new()
        .layer(0, LayerSampling::uniform(QNM_INTEGRATION_POINTS))
        .layer(1, LayerSampling::uniform(QNM_INTEGRATION_POINTS))
}

/// Return finite-layer thicknesses from a canonical scalar stack.
///
/// Canonical thicknesses are already stored in centimetres. Spatial numerical
/// integration uses only their primal real values.
fn canonical_layer_thicknesses<M>(stack: &CanonicalStack<M, HoloJ0>) -> Vec<f64> {
    stack
        .layers()
        .iter()
        .map(|layer| layer.thickness_cm().value()[()].re)
        .collect()
}

// -----------------------------------------------------------------------------
// Numerical integration helpers
// -----------------------------------------------------------------------------

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

fn integrate_density_by_layer(
    density: &Array1<C>,
    thicknesses: &[f64],
    points_per_layer: usize,
) -> Vec<C> {
    assert_eq!(density.len(), thicknesses.len() * points_per_layer,);

    thicknesses
        .iter()
        .enumerate()
        .map(|(layer_index, &thickness)| {
            let start = layer_index * points_per_layer;
            let end = start + points_per_layer;

            integrate_uniform_complex(
                &density
                    .as_slice()
                    .expect("sampled density should be contiguous")[start..end],
                thickness,
            )
        })
        .collect()
}

// -----------------------------------------------------------------------------
// Unit QNM normalization from sampled fields
// -----------------------------------------------------------------------------

macro_rules! sampled_qnm_unit_normalisation_test {
    (
        $name:ident,
        backend = $backend:expr,
        polarisation = $polarisation:expr
    ) => {
        #[test]
        fn $name() {
            let stack = two_layer_stack();

            let evaluator =
                ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, $backend).unwrap();

            let polarisation = $polarisation;

            let coordinates = modal_coordinates();

            let exterior = principal_exterior(evaluator.stack(), &coordinates, polarisation);

            let state = evaluator
                .retain(coordinates, exterior, polarisation)
                .expect("complex retained solve should succeed");

            let mode = state.mode().expect("mode should construct");

            let sampling = sampling();

            /*
             * ComplexPlaneMode is already QNM-normalized.
             */
            let response = mode
                .fields(&sampling)
                .expect("normalized modal fields should evaluate");

            let resolved = sampling
                .resolve_canonical(state.stack())
                .expect("field sampling should resolve");

            /*
             * Sample exactly the constitutive spectral data entering the
             * analytic QNM normalization:
             *
             *     W_e = ε + k₀ ∂ε/∂k₀
             *     W_m = μ + k₀ ∂μ/∂k₀
             *
             * These remain complex. There is no conjugation, real-part
             * projection, or Hermitian energy prefactor.
             */
            let factors = state
                .raw_constitutive_spectral_first_parameters::<ComplexPlane>(&resolved)
                .expect("complex constitutive spectral data should sample")
                .into_brillouin_factors();

            /*
             * The complex-plane field response retains the jet inside each
             * physical field:
             *
             *     ElectromagneticFields<Jet<VectorField<_>>>
             */
            let fields = response.quantity();

            let electric = fields.electric().value();
            let magnetic = fields.magnetic().value();

            let electric_square = bilinear_square(electric.x(), electric.y(), electric.z());

            let magnetic_square = bilinear_square(magnetic.x(), magnetic.y(), magnetic.z());

            let electric_weight = factors.electric().value();

            let magnetic_weight = factors.magnetic().value();

            assert_eq!(electric_weight.len(), electric_square.len(),);

            assert_eq!(magnetic_weight.len(), magnetic_square.len(),);

            /*
             * QNM normalization density:
             *
             *     ρ =
             *         W_e (E · E)
             *       - W_m (H · H)
             *
             * No complex conjugation.
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

            let thicknesses = canonical_layer_thicknesses(state.stack());

            assert_eq!(
                thicknesses.len(),
                2,
                "this fixture should contain two finite layers",
            );

            let electric_layers =
                integrate_density_by_layer(&electric_density, &thicknesses, QNM_INTEGRATION_POINTS);

            let magnetic_layers =
                integrate_density_by_layer(&magnetic_density, &thicknesses, QNM_INTEGRATION_POINTS);

            let total_layers =
                integrate_density_by_layer(&total_density, &thicknesses, QNM_INTEGRATION_POINTS);

            for layer_index in 0..thicknesses.len() {
                assert_complex_close(
                    total_layers[layer_index],
                    electric_layers[layer_index] - magnetic_layers[layer_index],
                    QNM_INTEGRATION_TOLERANCE,
                );
            }

            let electric_total: C = electric_layers.iter().copied().sum();

            let magnetic_total: C = magnetic_layers.iter().copied().sum();

            let total: C = total_layers.iter().copied().sum();

            assert_complex_close(
                total,
                electric_total - magnetic_total,
                QNM_INTEGRATION_TOLERANCE,
            );

            assert_complex_close(total, C::new(1.0, 0.0), QNM_INTEGRATION_TOLERANCE);
        }
    };
}

sampled_qnm_unit_normalisation_test!(
    scatter_te_sampled_qnm_fields_integrate_to_unit_normalisation,
    backend = Scatter2::new(),
    polarisation = Polarisation::TransverseElectric
);

sampled_qnm_unit_normalisation_test!(
    scatter_tm_sampled_qnm_fields_integrate_to_unit_normalisation,
    backend = Scatter2::new(),
    polarisation = Polarisation::TransverseMagnetic
);

sampled_qnm_unit_normalisation_test!(
    transfer_te_sampled_qnm_fields_integrate_to_unit_normalisation,
    backend = Transfer2::new(),
    polarisation = Polarisation::TransverseElectric
);

sampled_qnm_unit_normalisation_test!(
    transfer_tm_sampled_qnm_fields_integrate_to_unit_normalisation,
    backend = Transfer2::new(),
    polarisation = Polarisation::TransverseMagnetic
);

// -----------------------------------------------------------------------------
// Sampled components versus analytic normalization components
// -----------------------------------------------------------------------------

macro_rules! sampled_components_match_analytic_test {
    (
        $name:ident,
        polarisation = $polarisation:expr
    ) => {
        #[test]
        fn $name() {
            let stack = two_layer_stack();

            let evaluator =
                ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, Scatter2::new()).unwrap();

            let polarisation = $polarisation;

            let coordinates = modal_coordinates();

            let exterior = principal_exterior(evaluator.stack(), &coordinates, polarisation);

            let state = evaluator
                .retain(coordinates, exterior, polarisation)
                .expect("retained modal solve should succeed");

            let mode = state.mode().expect("mode construction should succeed");

            let sampling = sampling();

            let resolved = sampling
                .resolve_canonical(state.stack())
                .expect("sampling should resolve");

            let factors = state
                .raw_constitutive_spectral_first_parameters::<ComplexPlane>(&resolved)
                .expect("constitutive spectral sampling should succeed")
                .into_brillouin_factors();

            /*
             * `raw_normalisation` belongs to the arbitrary candidate before
             * the mode was rescaled. If
             *
             *     s² = 1 / N,
             *
             * each normalized component therefore acquires the factor 1/N.
             */
            let raw_electric = mode.seed_normalisation().electric().value()[()];

            let raw_magnetic = mode.seed_normalisation().magnetic().value()[()];

            let raw_total = mode.seed_normalisation().total().value()[()];

            let expected_electric = raw_electric / raw_total;

            let expected_magnetic = raw_magnetic / raw_total;

            let expected_total = expected_electric - expected_magnetic;

            let response = mode
                .fields(&sampling)
                .expect("modal field evaluation should succeed");

            let fields = response.quantity();

            let electric = fields.electric().value();
            let magnetic = fields.magnetic().value();

            let electric_square = bilinear_square(electric.x(), electric.y(), electric.z());

            let magnetic_square = bilinear_square(magnetic.x(), magnetic.y(), magnetic.z());

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

            let thicknesses = canonical_layer_thicknesses(state.stack());

            let sampled_electric: C =
                integrate_density_by_layer(&electric_density, &thicknesses, QNM_INTEGRATION_POINTS)
                    .into_iter()
                    .sum();

            let sampled_magnetic: C =
                integrate_density_by_layer(&magnetic_density, &thicknesses, QNM_INTEGRATION_POINTS)
                    .into_iter()
                    .sum();

            let sampled_total = sampled_electric - sampled_magnetic;

            /*
             * These tests previously only checked the total even though their
             * names claimed to verify the components. Check all three.
             */
            assert_complex_close(
                sampled_electric,
                expected_electric,
                QNM_COMPONENT_INTEGRATION_TOLERANCE,
            );

            assert_complex_close(
                sampled_magnetic,
                expected_magnetic,
                QNM_COMPONENT_INTEGRATION_TOLERANCE,
            );

            assert_complex_close(sampled_total, expected_total, QNM_INTEGRATION_TOLERANCE);

            assert_complex_close(sampled_total, C::new(1.0, 0.0), QNM_INTEGRATION_TOLERANCE);
        }
    };
}

sampled_components_match_analytic_test!(
    scatter_te_sampled_qnm_components_match_analytic_normalisation_components,
    polarisation = Polarisation::TransverseElectric
);

sampled_components_match_analytic_test!(
    scatter_tm_sampled_qnm_components_match_analytic_normalisation_components,
    polarisation = Polarisation::TransverseMagnetic
);

// -----------------------------------------------------------------------------
// Sampled integration versus analytic per-layer normalization
// -----------------------------------------------------------------------------

macro_rules! sampled_layers_match_analytic_test {
    (
        $name:ident,
        polarisation = $polarisation:expr
    ) => {
        #[test]
        fn $name() {
            let stack = two_layer_stack();

            let evaluator =
                ComplexPlaneEvaluator::<HoloJ0, _, _>::compile(&stack, Scatter2::new()).unwrap();

            let polarisation = $polarisation;

            let coordinates = modal_coordinates();

            let exterior = principal_exterior(evaluator.stack(), &coordinates, polarisation);

            let state = evaluator
                .retain(coordinates, exterior, polarisation)
                .expect("retained modal solve should succeed");

            let mode = state.mode().expect("mode construction should succeed");

            let sampling = sampling();

            let resolved = sampling
                .resolve_canonical(state.stack())
                .expect("sampling should resolve");

            let factors = state
                .raw_constitutive_spectral_first_parameters::<ComplexPlane>(&resolved)
                .expect("constitutive spectral sampling should succeed")
                .into_brillouin_factors();

            /*
             * Analytic normalization using the already-scaled modal solution.
             * The aggregate total therefore equals unity.
             */
            let coordinates = state.coordinates();

            let analytic_layers =
                raw_layer_integration_inputs_unchecked(mode.solution(), state.workspace())
                    .expect("layer inputs should assemble")
                    .integrate_bilinear()
                    .into_brillouin_layers::<ComplexPlane, _>(
                        state.stack().layers().iter().map(|layer| layer.material()),
                        coordinates.vacuum_angular_wavenumber(),
                    )
                    .expect("Brillouin layers should assemble")
                    .into_qnm_normalisation(
                        coordinates.vacuum_angular_wavenumber(),
                        coordinates.parallel_angular_wavenumber(),
                    );

            let response = mode
                .fields(&sampling)
                .expect("modal field evaluation should succeed");

            let fields = response.quantity();

            let electric = fields.electric().value();
            let magnetic = fields.magnetic().value();

            let electric_square = bilinear_square(electric.x(), electric.y(), electric.z());

            let magnetic_square = bilinear_square(magnetic.x(), magnetic.y(), magnetic.z());

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

            let thicknesses = canonical_layer_thicknesses(state.stack());

            assert_eq!(analytic_layers.len(), thicknesses.len(),);

            let sampled_electric =
                integrate_density_by_layer(&electric_density, &thicknesses, QNM_INTEGRATION_POINTS);

            let sampled_magnetic =
                integrate_density_by_layer(&magnetic_density, &thicknesses, QNM_INTEGRATION_POINTS);

            for layer_index in 0..thicknesses.len() {
                let sampled_total = sampled_electric[layer_index] - sampled_magnetic[layer_index];

                let analytic = analytic_layers
                    .get(FiniteLayerIndex::new(layer_index))
                    .expect("analytic layer should exist");

                let analytic_electric = analytic.electric().value()[()];

                let analytic_magnetic = analytic.magnetic().value()[()];

                let analytic_total = analytic.total().value()[()];

                assert_complex_close(
                    sampled_electric[layer_index],
                    analytic_electric,
                    QNM_COMPONENT_INTEGRATION_TOLERANCE,
                );

                assert_complex_close(
                    sampled_magnetic[layer_index],
                    analytic_magnetic,
                    QNM_COMPONENT_INTEGRATION_TOLERANCE,
                );

                assert_complex_close(sampled_total, analytic_total, QNM_INTEGRATION_TOLERANCE);
            }
        }
    };
}

sampled_layers_match_analytic_test!(
    scatter_te_sampled_qnm_normalisation_matches_analytic_layers,
    polarisation = Polarisation::TransverseElectric
);

sampled_layers_match_analytic_test!(
    scatter_tm_sampled_qnm_normalisation_matches_analytic_layers,
    polarisation = Polarisation::TransverseMagnetic
);
