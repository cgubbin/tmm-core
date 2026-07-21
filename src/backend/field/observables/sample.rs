use crate::{
    ComplexScalar, PlaneWaveInput, Stack,
    backend::{
        FieldPosition, FieldSampling, IsotropicFieldState, PlaneWaveFieldError, PlaneWaveFields,
        algebra::ScalarAlgebra,
        field::{
            BidirectionalWavesGeneric, BoundaryWaveSolution, BoundaryWaves,
            CartesianElectromagneticField,
            boundary::{
                BoundaryWavesGeneric, generic_boundary_first, generic_boundary_second,
                generic_boundary_values,
            },
            observables::{
                context::{
                    AlgebraicFieldContext, spectral_first_context, spectral_second_context,
                    structural_first_context, structural_second_context, value_context,
                },
                fields::AlgebraicFieldSample,
            },
            sampling::{validate_exterior_distance, validate_layer_offset},
        },
        isotropic::IsotropicLayerQuantities,
    },
    material::{EvaluateDifferentiableMaterial, EvaluateMaterial},
};

use ndarray::{ArrayBase, Dimension, OwnedRepr};
use num_traits::Float;

/// Sample fields from a high-level spatial sampling specification.
pub fn sample_plane_wave_field_profile<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    waves: &BoundaryWaveSolution<C, D>,
    sampling: &FieldSampling<C::RealField>,
) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
where
    M: EvaluateMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
{
    let positions = sampling.positions(stack)?;

    let values = match waves {
        BoundaryWaveSolution::Values(values) => values,
        BoundaryWaveSolution::Differentiated(differentiated) => differentiated.values(),
    };
    sample_value_fields(stack, input, values, positions)
}

pub(crate) fn sample_value_fields<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    waves: &BoundaryWaves<C, D>,
    positions: Vec<FieldPosition<C::RealField>>,
) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
where
    M: EvaluateMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
{
    let context = value_context(stack, input);
    let waves = generic_boundary_values(waves);

    let samples = sample_plane_wave_fields_algebraic(&context, &waves, positions)?;

    Ok(PlaneWaveFields::from_values(samples))
}

pub(super) fn sample_first_order_fields_structural<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    waves: &BoundaryWaveSolution<C, D>,
    positions: Vec<FieldPosition<C::RealField>>,
) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
where
    M: EvaluateMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
{
    let differentiated = waves
        .structural()
        .ok_or(PlaneWaveFieldError::ExpectedStructuralDerivatives)?;

    let variable = differentiated
        .variable()
        .try_into()
        .map_err(|_| PlaneWaveFieldError::ExpectedStructuralDerivatives)?;

    let context = structural_first_context(stack, input, variable);
    let waves = generic_boundary_first(waves.values(), differentiated);

    let samples = sample_plane_wave_fields_algebraic(&context, &waves, positions)?;

    Ok(PlaneWaveFields::from_first_order(
        differentiated.variable(),
        samples,
    ))
}

pub(super) fn sample_second_order_fields_structural<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    waves: &BoundaryWaveSolution<C, D>,
    positions: Vec<FieldPosition<C::RealField>>,
) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
where
    M: EvaluateMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
{
    let differentiated = waves
        .structural()
        .ok_or(PlaneWaveFieldError::ExpectedStructuralDerivatives)?;

    let variable = differentiated
        .variable()
        .try_into()
        .map_err(|_| PlaneWaveFieldError::ExpectedStructuralDerivatives)?;

    let waves = generic_boundary_second(waves.values(), differentiated)
        .ok_or(PlaneWaveFieldError::MissingSecondDerivatives)?;

    let context = structural_second_context(stack, input, variable);

    let samples = sample_plane_wave_fields_algebraic(&context, &waves, positions)?;

    Ok(PlaneWaveFields::from_second_order(
        differentiated.variable(),
        samples,
    ))
}

pub(super) fn sample_first_order_fields_spectral<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    waves: &BoundaryWaveSolution<C, D>,
    positions: Vec<FieldPosition<C::RealField>>,
) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
where
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
{
    let differentiated = waves
        .spectral()
        .ok_or(PlaneWaveFieldError::ExpectedSpectralDerivatives)?;

    let variable = differentiated
        .variable()
        .try_into()
        .map_err(|_| PlaneWaveFieldError::ExpectedSpectralDerivatives)?;

    let context = spectral_first_context(stack, input, variable);
    let waves = generic_boundary_first(waves.values(), differentiated);

    let samples = sample_plane_wave_fields_algebraic(&context, &waves, positions)?;

    Ok(PlaneWaveFields::from_first_order(
        differentiated.variable(),
        samples,
    ))
}

pub(super) fn sample_second_order_fields_spectral<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    waves: &BoundaryWaveSolution<C, D>,
    positions: Vec<FieldPosition<C::RealField>>,
) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
where
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
{
    let differentiated = waves
        .spectral()
        .ok_or(PlaneWaveFieldError::ExpectedSpectralDerivatives)?;

    let variable = differentiated
        .variable()
        .try_into()
        .map_err(|_| PlaneWaveFieldError::ExpectedSpectralDerivatives)?;

    let waves = generic_boundary_second(waves.values(), differentiated)
        .ok_or(PlaneWaveFieldError::MissingSecondDerivatives)?;

    let context = spectral_second_context(stack, input, variable);

    let samples = sample_plane_wave_fields_algebraic(&context, &waves, positions)?;

    Ok(PlaneWaveFields::from_second_order(
        differentiated.variable(),
        samples,
    ))
}

fn sample_plane_wave_fields_algebraic<C, D, A>(
    context: &AlgebraicFieldContext<C, D, A>,
    waves: &BoundaryWavesGeneric<A>,
    positions: impl IntoIterator<Item = FieldPosition<C::RealField>>,
) -> Result<Vec<AlgebraicFieldSample<C, D, A>>, PlaneWaveFieldError<C::RealField>>
where
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Clone,
{
    validate_generic_layer_count(context.layers.len(), waves)?;

    positions
        .into_iter()
        .map(|position| sample_position(context, waves, position))
        .collect()
}

pub(super) fn validate_generic_layer_count<A, R>(
    expected: usize,
    waves: &BoundaryWavesGeneric<A>,
) -> Result<(), PlaneWaveFieldError<R>> {
    if expected != waves.len() {
        return Err(PlaneWaveFieldError::LayerCountMismatch {
            expected,
            actual: waves.len(),
        });
    }

    Ok(())
}

fn sample_position<C, D, A>(
    context: &AlgebraicFieldContext<C, D, A>,
    waves: &BoundaryWavesGeneric<A>,
    position: FieldPosition<C::RealField>,
) -> Result<AlgebraicFieldSample<C, D, A>, PlaneWaveFieldError<C::RealField>>
where
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Clone,
{
    match position {
        FieldPosition::LeftExterior { distance } => {
            sample_left_exterior(context, waves, distance, position)
        }

        FieldPosition::Layer { index, offset } => {
            sample_layer(context, waves, index, offset, position)
        }

        FieldPosition::RightExterior { distance } => {
            sample_right_exterior(context, waves, distance, position)
        }
    }
}

fn sample_left_exterior<C, D, A>(
    context: &AlgebraicFieldContext<C, D, A>,
    waves: &BoundaryWavesGeneric<A>,
    distance: C::RealField,
    position: FieldPosition<C::RealField>,
) -> Result<AlgebraicFieldSample<C, D, A>, PlaneWaveFieldError<C::RealField>>
where
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Clone,
{
    validate_exterior_distance(distance)?;

    let local = sample_left_exterior_waves_algebraic::<C, D, A>(
        waves.exterior().left(),
        context.left.kappa(),
        distance,
    );

    let (canonical, cartesian) = fields_from_local_waves(context, &local, &context.left);

    Ok(AlgebraicFieldSample {
        position,
        coordinate: -distance,
        canonical,
        cartesian,
    })
}

fn sample_layer<C, D, A>(
    context: &AlgebraicFieldContext<C, D, A>,
    waves: &BoundaryWavesGeneric<A>,
    index: usize,
    offset: C::RealField,
    position: FieldPosition<C::RealField>,
) -> Result<AlgebraicFieldSample<C, D, A>, PlaneWaveFieldError<C::RealField>>
where
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Clone,
{
    let layer = context
        .layers
        .get(index)
        .ok_or(PlaneWaveFieldError::LayerOutOfBounds {
            requested: index,
            layer_count: context.layers.len(),
        })?;

    validate_layer_offset(index, offset, layer.thickness)?;

    let boundary = waves
        .layer(index)
        .ok_or(PlaneWaveFieldError::LayerOutOfBounds {
            requested: index,
            layer_count: waves.len(),
        })?;

    let local = sample_layer_waves_algebraic::<C, D, A>(
        boundary,
        layer.quantities.kappa(),
        offset,
        layer.thickness,
    );

    let (canonical, cartesian) = fields_from_local_waves(context, &local, &layer.quantities);

    Ok(AlgebraicFieldSample {
        position,
        coordinate: layer.origin + offset,
        canonical,
        cartesian,
    })
}

fn sample_right_exterior<C, D, A>(
    context: &AlgebraicFieldContext<C, D, A>,
    waves: &BoundaryWavesGeneric<A>,
    distance: C::RealField,
    position: FieldPosition<C::RealField>,
) -> Result<AlgebraicFieldSample<C, D, A>, PlaneWaveFieldError<C::RealField>>
where
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Clone,
{
    validate_exterior_distance(distance)?;

    let local = sample_right_exterior_waves_algebraic::<C, D, A>(
        waves.exterior().right(),
        context.right.kappa(),
        distance,
    );

    let (canonical, cartesian) = fields_from_local_waves(context, &local, &context.right);

    Ok(AlgebraicFieldSample {
        position,
        coordinate: context.total_thickness + distance,
        canonical,
        cartesian,
    })
}

fn fields_from_local_waves<C, D, A>(
    context: &AlgebraicFieldContext<C, D, A>,
    waves: &BidirectionalWavesGeneric<A>,
    quantities: &IsotropicLayerQuantities<A>,
) -> (
    IsotropicFieldState<A>,
    CartesianElectromagneticField<A::Vector>,
)
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Clone,
{
    let admittance = quantities.clone().into_admittance().into_inner();

    let canonical = IsotropicFieldState::from_waves::<C, D>(waves, &admittance);

    let cartesian = canonical.cartesian_fields::<C, D>(
        context.polarisation,
        context.planar.parallel_wavenumber(),
        quantities.epsilon(),
        quantities.mu(),
    );

    (canonical, cartesian)
}

fn sample_left_exterior_waves_algebraic<C, D, A>(
    boundary: &crate::backend::field::boundary::BidirectionalWavesGeneric<A>,
    kappa: &A,
    distance: C::RealField,
) -> crate::backend::field::boundary::BidirectionalWavesGeneric<A>
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Clone,
{
    let forward_phase = propagation_phase_algebraic::<C, D, A>(kappa, -distance);

    let backward_phase = propagation_phase_algebraic::<C, D, A>(&kappa.negate(), -distance);

    BidirectionalWavesGeneric::new(
        boundary.forward().multiply(&forward_phase),
        boundary.backward().multiply(&backward_phase),
    )
}

fn sample_layer_waves_algebraic<C, D, A>(
    boundary: &crate::backend::field::boundary::LayerBoundaryWavesGeneric<A>,
    kappa: &A,
    offset: C::RealField,
    thickness: C::RealField,
) -> crate::backend::field::boundary::BidirectionalWavesGeneric<A>
where
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Clone,
{
    let forward_phase = propagation_phase_algebraic::<C, D, A>(kappa, offset);

    let backward_phase = propagation_phase_algebraic::<C, D, A>(kappa, thickness - offset);

    crate::backend::field::boundary::BidirectionalWavesGeneric::new(
        boundary.left().forward().multiply(&forward_phase),
        boundary.right().backward().multiply(&backward_phase),
    )
}

fn sample_right_exterior_waves_algebraic<C, D, A>(
    boundary: &crate::backend::field::boundary::BidirectionalWavesGeneric<A>,
    kappa: &A,
    distance: C::RealField,
) -> crate::backend::field::boundary::BidirectionalWavesGeneric<A>
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Clone,
{
    let forward_phase = propagation_phase_algebraic::<C, D, A>(kappa, distance);

    let backward_phase = propagation_phase_algebraic::<C, D, A>(&kappa.negate(), distance);

    crate::backend::field::boundary::BidirectionalWavesGeneric::new(
        boundary.forward().multiply(&forward_phase),
        boundary.backward().multiply(&backward_phase),
    )
}

fn propagation_phase_algebraic<C, D, A>(kappa: &A, distance: C::RealField) -> A
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D>,
{
    let coefficient = C::i() * C::from_real(distance);

    kappa.scale(coefficient).exp()
}

#[cfg(test)]
mod tests {
    use crate::backend::field::LayerBoundaryWavesGeneric;

    use super::*;

    use approx::assert_relative_eq;
    use ndarray::{Array0, arr0};
    use num_complex::Complex64;

    type C = Complex64;
    type Samples = Array0<C>;

    const TOLERANCE: f64 = 1e-15;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn scalar(real: f64, imaginary: f64) -> Samples {
        arr0(c(real, imaginary))
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
    fn left_exterior_zero_distance_preserves_boundary_waves() {
        let waves = BidirectionalWavesGeneric::new(scalar(0.8, -0.1), scalar(0.2, 0.3));

        let sampled = sample_left_exterior_waves_algebraic::<C, ndarray::Ix0, Array0<_>>(
            &waves,
            &scalar(3.0, 0.0),
            0.0,
        );

        assert_complex_close(sampled.forward()[()], c(0.8, -0.1), TOLERANCE);
        assert_complex_close(sampled.backward()[()], c(0.2, 0.3), TOLERANCE);
    }

    #[test]
    fn right_exterior_zero_distance_preserves_boundary_waves() {
        let waves = BidirectionalWavesGeneric::new(scalar(0.1, -0.6), scalar(0.4, 0.2));

        let sampled = sample_right_exterior_waves_algebraic::<C, ndarray::Ix0, Array0<_>>(
            &waves,
            &scalar(3.0, 0.0),
            0.0,
        );

        assert_complex_close(sampled.forward()[()], c(0.1, -0.6), TOLERANCE);
        assert_complex_close(sampled.backward()[()], c(0.4, 0.2), TOLERANCE);
    }

    #[test]
    fn finite_layer_forward_wave_accumulates_expected_phase() {
        let kappa = scalar(2.0, 0.0);
        let amplitude = c(0.6, -0.1);

        let boundary = LayerBoundaryWavesGeneric::new(
            BidirectionalWavesGeneric::new(Array0::from_elem((), amplitude), scalar(0.0, 0.0)),
            BidirectionalWavesGeneric::new(scalar(0.0, 0.0), scalar(0.0, 0.0)),
        );

        let thickness = 1.0;
        let offset = 0.3;

        let sampled = sample_layer_waves_algebraic(&boundary, &kappa, offset, thickness);

        let expected = amplitude * c(0.0, 2.0 * offset).exp();

        assert_complex_close(sampled.forward()[()], expected, TOLERANCE);

        assert_complex_close(sampled.backward()[()], c(0.0, 0.0), TOLERANCE);
    }

    #[test]
    fn finite_layer_backward_wave_is_referenced_to_right_boundary() {
        let kappa = scalar(2.0, 0.0);
        let amplitude = c(-0.2, 0.5);

        let thickness = 1.0;
        let offset = 0.3;

        let boundary = LayerBoundaryWavesGeneric::new(
            BidirectionalWavesGeneric::new(scalar(0.0, 0.0), scalar(0.0, 0.0)),
            BidirectionalWavesGeneric::new(scalar(0.0, 0.0), Array0::from_elem((), amplitude)),
        );

        let sampled = sample_layer_waves_algebraic(&boundary, &kappa, offset, thickness);

        let expected = amplitude * c(0.0, 2.0 * (thickness - offset)).exp();

        assert_complex_close(sampled.forward()[()], c(0.0, 0.0), TOLERANCE);

        assert_complex_close(sampled.backward()[()], expected, TOLERANCE);
    }

    #[test]
    fn finite_layer_endpoints_reproduce_reference_amplitudes() {
        let thickness = 0.8;

        let left_forward = c(0.7, 0.1);
        let right_backward = c(-0.3, 0.2);

        let boundary = LayerBoundaryWavesGeneric::new(
            BidirectionalWavesGeneric::new(Array0::from_elem((), left_forward), scalar(0.0, 0.0)),
            BidirectionalWavesGeneric::new(scalar(0.0, 0.0), Array0::from_elem((), right_backward)),
        );

        let left = sample_layer_waves_algebraic(&boundary, &scalar(1.4, 0.0), 0.0, thickness);

        let right =
            sample_layer_waves_algebraic(&boundary, &scalar(1.4, 0.0), thickness, thickness);

        assert_complex_close(left.forward()[()], left_forward, TOLERANCE);

        assert_complex_close(right.backward()[()], right_backward, TOLERANCE);
    }

    #[test]
    fn sampling_rejects_negative_exterior_distance() {
        let error = validate_exterior_distance(-0.1).unwrap_err();

        assert!(matches!(
            error,
            PlaneWaveFieldError::InvalidExteriorDistance { .. }
        ));
    }

    #[test]
    fn sampling_rejects_layer_offset_beyond_thickness() {
        let error = validate_layer_offset(0, 1.1, 1.0).unwrap_err();

        assert!(matches!(
            error,
            PlaneWaveFieldError::InvalidLayerOffset { .. }
        ));
    }
}
