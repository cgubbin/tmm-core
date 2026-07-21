use approx::assert_relative_eq;
use ndarray::{Array0, ArrayBase, Data, Dimension, Ix0, arr0};
use num_complex::Complex64;

use crate::{
    Constant, DifferentiableMaterialStack, IncidentSide, Layer, MaterialStack, PlanarInput,
    PlaneWaveInput, Polarisation, Stack, Thickness,
    backend::{
        ExteriorSampling, LayerSampling,
        field::{
            BoundaryWaves, CartesianElectromagneticField, CartesianVector3, IsotropicFieldState,
            PlaneWaveFields, PlaneWavePowerBalance, PlaneWavePowerBalanceDerivative,
        },
    },
    material::model::Lossy,
};

pub(crate) type C = Complex64;
pub(crate) type D = Ix0;

pub(crate) const ABS_TOLERANCE: f64 = 1.0e-9;
pub(crate) const REL_TOLERANCE: f64 = 1.0e-8;

pub(crate) fn c(r: f64, i: f64) -> C {
    C::new(r, i)
}

pub(crate) fn assert_complex_close(actual: C, expected: C) {
    assert_relative_eq!(
        actual.re,
        expected.re,
        epsilon = ABS_TOLERANCE,
        max_relative = REL_TOLERANCE,
    );

    assert_relative_eq!(
        actual.im,
        expected.im,
        epsilon = ABS_TOLERANCE,
        max_relative = REL_TOLERANCE,
    );
}

pub(crate) fn assert_real_close(actual: f64, expected: f64) {
    assert_relative_eq!(
        actual,
        expected,
        epsilon = ABS_TOLERANCE,
        max_relative = REL_TOLERANCE,
    );
}

pub(crate) fn assert_complex_array_close<S1, S2, D>(
    actual: &ArrayBase<S1, D>,
    expected: &ArrayBase<S2, D>,
) where
    S1: Data<Elem = C>,
    S2: Data<Elem = C>,
    D: Dimension,
{
    assert_eq!(actual.raw_dim(), expected.raw_dim());

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_complex_close(*actual, *expected);
    }
}

pub(crate) fn assert_real_array_close<S1, S2, D>(
    actual: &ArrayBase<S1, D>,
    expected: &ArrayBase<S2, D>,
) where
    S1: Data<Elem = f64>,
    S2: Data<Elem = f64>,
    D: Dimension,
{
    assert_eq!(actual.raw_dim(), expected.raw_dim());

    for (actual, expected) in actual.iter().zip(expected.iter()) {
        assert_real_close(*actual, *expected);
    }
}

pub(crate) fn assert_canonical_state_close<D>(
    actual: &IsotropicFieldState<ArrayBase<ndarray::OwnedRepr<C>, D>>,
    expected: &IsotropicFieldState<ArrayBase<ndarray::OwnedRepr<C>, D>>,
) where
    D: Dimension,
{
    assert_complex_array_close(actual.primary(), expected.primary());

    assert_complex_array_close(actual.dual(), expected.dual());
}

pub(crate) fn assert_vector_close<D>(
    actual: &CartesianVector3<C, D>,
    expected: &CartesianVector3<C, D>,
) where
    D: Dimension,
{
    assert_complex_array_close(actual.x(), expected.x());
    assert_complex_array_close(actual.y(), expected.y());
    assert_complex_array_close(actual.z(), expected.z());
}

pub(crate) fn assert_cartesian_fields_close<D>(
    actual: &CartesianElectromagneticField<CartesianVector3<C, D>>,
    expected: &CartesianElectromagneticField<CartesianVector3<C, D>>,
) where
    D: Dimension,
{
    assert_vector_close(actual.electric(), expected.electric());

    assert_vector_close(actual.magnetic(), expected.magnetic());
}

pub(crate) fn assert_fields_close<D>(
    actual: &PlaneWaveFields<C, D>,
    expected: &PlaneWaveFields<C, D>,
) where
    D: Dimension,
{
    assert_eq!(actual.len(), expected.len());

    for (actual, expected) in actual.samples().iter().zip(expected.samples()) {
        assert_eq!(actual.position(), expected.position(),);

        assert_real_close(actual.coordinate(), expected.coordinate());

        assert_canonical_state_close(actual.canonical_state(), expected.canonical_state());

        assert_cartesian_fields_close(actual.cartesian_fields(), expected.cartesian_fields());

        assert_real_array_close(&actual.normal_flux(), &expected.normal_flux());
    }
}

pub(crate) fn assert_power_balance_close<D>(
    actual: &PlaneWavePowerBalance<f64, D>,
    expected: &PlaneWavePowerBalance<f64, D>,
) where
    D: Dimension,
{
    assert_real_array_close(actual.incident_flux(), expected.incident_flux());

    assert_real_array_close(actual.reflected_flux(), expected.reflected_flux());

    assert_real_array_close(actual.transmitted_flux(), expected.transmitted_flux());

    assert_eq!(
        actual.layer_absorptance().len(),
        expected.layer_absorptance().len(),
    );

    for (actual, expected) in actual
        .layer_absorptance()
        .iter()
        .zip(expected.layer_absorptance())
    {
        assert_real_array_close(actual, expected);
    }

    assert_real_array_close(
        actual.total_layer_absorptance(),
        expected.total_layer_absorptance(),
    );

    assert_real_array_close(actual.balance_residual(), expected.balance_residual());
}

pub(crate) fn assert_bidirectional_waves_close<D>(
    actual: &crate::backend::field::BidirectionalWaves<C, D>,
    expected: &crate::backend::field::BidirectionalWaves<C, D>,
) where
    D: Dimension,
{
    assert_complex_array_close(actual.forward(), expected.forward());

    assert_complex_array_close(actual.backward(), expected.backward());
}

pub(crate) fn assert_boundary_waves_close<D>(
    actual: &BoundaryWaves<C, D>,
    expected: &BoundaryWaves<C, D>,
) where
    D: Dimension,
{
    assert_eq!(actual.layers().len(), expected.layers().len(),);

    assert_bidirectional_waves_close(actual.exterior().left(), expected.exterior().left());

    assert_bidirectional_waves_close(actual.exterior().right(), expected.exterior().right());

    for (actual, expected) in actual.layers().iter().zip(expected.layers()) {
        assert_bidirectional_waves_close(actual.left(), expected.left());

        assert_bidirectional_waves_close(actual.right(), expected.right());
    }
}

pub(crate) fn lossless_stack() -> DifferentiableMaterialStack<f64, Complex64> {
    Stack::from_differentiable_materials(Constant::dielectric(1.0), Constant::dielectric(1.0))
        .differentiable_layer(
            Constant::dielectric(2.25),
            Thickness::from_cm(0.31).unwrap(),
        )
        .differentiable_layer(
            Constant::dielectric(4.00),
            Thickness::from_cm(0.17).unwrap(),
        )
        .differentiable_layer(
            Constant::dielectric(1.69),
            Thickness::from_cm(0.23).unwrap(),
        )
        .build()
        .unwrap()
}

pub(crate) fn absorbing_stack() -> DifferentiableMaterialStack<f64, Complex64> {
    Stack::from_differentiable_materials(Constant::dielectric(1.0), Constant::dielectric(1.44))
        .differentiable_layer(
            Lossy::dielectric(c(2.25, 0.15)),
            Thickness::from_cm(0.21).unwrap(),
        )
        .differentiable_layer(
            Lossy::dielectric(c(3.10, 0.35)),
            Thickness::from_cm(0.16).unwrap(),
        )
        .build()
        .unwrap()
}

pub(crate) fn scalar_input(
    polarisation: Polarisation,
    incident_side: IncidentSide,
) -> PlaneWaveInput<Array0<f64>> {
    let vacuum_wavenumber = arr0(8.3);
    let parallel_wavenumber = arr0(1.1);
    let planar = PlanarInput::new(vacuum_wavenumber, parallel_wavenumber, polarisation);

    PlaneWaveInput::new(planar, incident_side)
}

pub(crate) fn boundary_samples() -> crate::backend::field::FieldSampling<f64> {
    use crate::backend::field::FieldSampling;

    FieldSampling::new()
        .left_exterior(ExteriorSampling::point(0.0))
        .layer_interfaces()
        .right_exterior(ExteriorSampling::point(0.0))
}

pub(crate) fn boundary_positions<M>(
    stack: &Stack<M, f64>,
) -> Vec<crate::backend::field::FieldPosition<f64>> {
    use crate::backend::field::FieldPosition;

    let layers = stack.layers_left_to_right();

    let mut positions = Vec::with_capacity(2 + 2 * layers.len());

    positions.push(FieldPosition::LeftExterior { distance: 0.0 });

    for (index, layer) in layers.iter().enumerate() {
        positions.push(FieldPosition::Layer { index, offset: 0.0 });

        positions.push(FieldPosition::Layer {
            index,
            offset: layer.thickness().as_cm(),
        });
    }

    positions.push(FieldPosition::RightExterior { distance: 0.0 });

    positions
}

pub(crate) fn field_positions<M>(
    stack: &Stack<M, f64>,
) -> Vec<crate::backend::field::FieldPosition<f64>> {
    use crate::backend::field::FieldPosition;

    let mut positions = boundary_positions(stack);

    for (index, layer) in stack.layers_left_to_right().iter().enumerate() {
        positions.push(FieldPosition::Layer {
            index,
            offset: 0.37 * layer.thickness().as_cm(),
        });
    }

    positions.push(FieldPosition::LeftExterior { distance: 0.19 });

    positions.push(FieldPosition::RightExterior { distance: 0.27 });

    positions
}

pub(crate) fn field_samples<M>(stack: &Stack<M, f64>) -> crate::backend::field::FieldSampling<f64> {
    use crate::backend::field::FieldSampling;

    let layers = stack.layers_left_to_right();

    let mut sampling = FieldSampling::new();

    for (index, layer) in layers.iter().enumerate() {
        sampling = sampling.layer(
            index,
            LayerSampling::point(0.37 * layer.thickness().as_cm()),
        );
    }

    sampling = sampling.left_exterior(ExteriorSampling::point(0.19));
    sampling = sampling.right_exterior(ExteriorSampling::point(0.27));

    sampling
}

pub(crate) fn assert_field_derivatives_close<D>(
    actual: &PlaneWaveFields<C, D>,
    expected: &PlaneWaveFields<C, D>,
) where
    D: Dimension,
{
    assert_fields_close(actual, expected);

    let actual = actual
        .derivatives()
        .expect("actual fields should contain derivatives");

    let expected = expected
        .derivatives()
        .expect("expected fields should contain derivatives");

    assert_eq!(actual.variable(), expected.variable(),);

    assert_eq!(actual.first().len(), expected.first().len(),);

    for (actual, expected) in actual.first().iter().zip(expected.first()) {
        assert_canonical_state_close(actual.canonical_state(), expected.canonical_state());

        assert_cartesian_fields_close(actual.cartesian_fields(), expected.cartesian_fields());
    }

    match (actual.second(), expected.second()) {
        (None, None) => {}

        (Some(actual), Some(expected)) => {
            assert_eq!(actual.len(), expected.len());

            for (actual, expected) in actual.iter().zip(expected) {
                assert_canonical_state_close(actual.canonical_state(), expected.canonical_state());

                assert_cartesian_fields_close(
                    actual.cartesian_fields(),
                    expected.cartesian_fields(),
                );
            }
        }

        _ => panic!("backends retained different derivative orders"),
    }
}

pub(crate) fn assert_power_derivative_close<D>(
    actual: &PlaneWavePowerBalanceDerivative<f64, D>,
    expected: &PlaneWavePowerBalanceDerivative<f64, D>,
) where
    D: Dimension,
{
    assert_real_array_close(actual.incident_flux(), expected.incident_flux());

    assert_real_array_close(actual.reflected_flux(), expected.reflected_flux());

    assert_real_array_close(actual.transmitted_flux(), expected.transmitted_flux());

    assert_eq!(
        actual.layer_absorptance().len(),
        expected.layer_absorptance().len(),
    );

    for (actual, expected) in actual
        .layer_absorptance()
        .iter()
        .zip(expected.layer_absorptance())
    {
        assert_real_array_close(actual, expected);
    }

    assert_real_array_close(
        actual.total_layer_absorptance(),
        expected.total_layer_absorptance(),
    );

    assert_real_array_close(actual.balance_residual(), expected.balance_residual());
}

pub(crate) fn assert_differentiated_power_balance_close<D>(
    actual: &PlaneWavePowerBalance<f64, D>,
    expected: &PlaneWavePowerBalance<f64, D>,
) where
    D: Dimension,
{
    assert_power_balance_close(actual, expected);

    let actual = actual.derivatives().unwrap();
    let expected = expected.derivatives().unwrap();

    assert_eq!(actual.variable(), expected.variable(),);

    assert_power_derivative_close(actual.first(), expected.first());

    match (actual.second(), expected.second()) {
        (None, None) => {}

        (Some(actual), Some(expected)) => {
            assert_power_derivative_close(actual, expected);
        }

        _ => panic!("backends retained different derivative orders"),
    }
}

pub(crate) fn central_first_complex<F>(mut evaluate: F, x: f64, step: f64) -> Complex64
where
    F: FnMut(f64) -> Complex64,
{
    let upper = evaluate(x + step);
    let lower = evaluate(x - step);

    (upper - lower) / (2.0 * step)
}

pub(crate) fn central_second_complex<F>(mut evaluate: F, x: f64, step: f64) -> Complex64
where
    F: FnMut(f64) -> Complex64,
{
    let upper = evaluate(x + step);
    let centre = evaluate(x);
    let lower = evaluate(x - step);

    (upper - 2.0 * centre + lower) / step.powi(2)
}

pub(crate) fn central_first_real<F>(mut evaluate: F, x: f64, step: f64) -> f64
where
    F: FnMut(f64) -> f64,
{
    let upper = evaluate(x + step);
    let lower = evaluate(x - step);

    (upper - lower) / (2.0 * step)
}

pub(crate) fn central_second_real<F>(mut evaluate: F, x: f64, step: f64) -> f64
where
    F: FnMut(f64) -> f64,
{
    let upper = evaluate(x + step);
    let centre = evaluate(x);
    let lower = evaluate(x - step);

    (upper - 2.0 * centre + lower) / step.powi(2)
}

pub(crate) fn first_derivative_step(value: f64) -> f64 {
    1.0e-5 * value.abs().max(1.0)
}

pub(crate) fn second_derivative_step(value: f64) -> f64 {
    1.0e-4 * value.abs().max(1.0)
}

pub(crate) fn thickness_first_step(thickness_cm: f64) -> f64 {
    1.0e-5 * thickness_cm.abs().max(1.0e-6)
}

pub(crate) fn thickness_second_step(thickness_cm: f64) -> f64 {
    1.0e-4 * thickness_cm.abs().max(1.0e-6)
}

pub(crate) fn with_layer_thickness<M>(
    stack: &Stack<M, f64>,
    index: usize,
    thickness_cm: f64,
) -> Stack<M, f64>
where
    M: Clone,
{
    let mut layers = stack.layers_left_to_right().to_vec();

    let material = layers[index].material().clone();

    layers[index] = Layer::new(material, Thickness::from_cm(thickness_cm).unwrap());

    Stack::new(
        stack.left_exterior().clone(),
        layers,
        stack.right_exterior().clone(),
    )
}

pub(crate) fn with_vacuum_wavenumber(
    input: &PlaneWaveInput<Array0<f64>>,
    vacuum_wavenumber: f64,
) -> PlaneWaveInput<Array0<f64>> {
    let mut input = input.clone();

    input.planar.vacuum_wavenumber = arr0(vacuum_wavenumber);

    input
}

pub(crate) fn with_parallel_wavenumber(
    input: &PlaneWaveInput<Array0<f64>>,
    parallel_wavenumber: f64,
) -> PlaneWaveInput<Array0<f64>> {
    let mut input = input.clone();

    input.planar.parallel_wavenumber = arr0(parallel_wavenumber);

    input
}

pub(crate) const FIRST_DERIVATIVE_ABS_TOLERANCE: f64 = 1.0e-7;
pub(crate) const FIRST_DERIVATIVE_REL_TOLERANCE: f64 = 2.0e-5;

pub(crate) fn assert_real_derivative_close(actual: f64, expected: f64) {
    assert_relative_eq!(
        actual,
        expected,
        epsilon = FIRST_DERIVATIVE_ABS_TOLERANCE,
        max_relative = FIRST_DERIVATIVE_REL_TOLERANCE,
    );
}

pub(crate) fn assert_complex_derivative_close(actual: Complex64, expected: Complex64) {
    assert_relative_eq!(
        actual.re,
        expected.re,
        epsilon = FIRST_DERIVATIVE_ABS_TOLERANCE,
        max_relative = FIRST_DERIVATIVE_REL_TOLERANCE,
    );

    assert_relative_eq!(
        actual.im,
        expected.im,
        epsilon = FIRST_DERIVATIVE_ABS_TOLERANCE,
        max_relative = FIRST_DERIVATIVE_REL_TOLERANCE,
    );
}

pub(crate) const SECOND_DERIVATIVE_ABS_TOLERANCE: f64 = 1.0e-5;
pub(crate) const SECOND_DERIVATIVE_REL_TOLERANCE: f64 = 5.0e-4;

pub(crate) fn assert_real_second_derivative_close(actual: f64, expected: f64) {
    assert_relative_eq!(
        actual,
        expected,
        epsilon = SECOND_DERIVATIVE_ABS_TOLERANCE,
        max_relative = SECOND_DERIVATIVE_REL_TOLERANCE,
    );
}

pub(crate) fn assert_complex_second_derivative_close(actual: Complex64, expected: Complex64) {
    assert_relative_eq!(
        actual.re,
        expected.re,
        epsilon = SECOND_DERIVATIVE_ABS_TOLERANCE,
        max_relative = SECOND_DERIVATIVE_REL_TOLERANCE,
    );

    assert_relative_eq!(
        actual.im,
        expected.im,
        epsilon = SECOND_DERIVATIVE_ABS_TOLERANCE,
        max_relative = SECOND_DERIVATIVE_REL_TOLERANCE,
    );
}
