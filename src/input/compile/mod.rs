//! Compilation of caller-facing plane-wave problems.
//!
//! This module converts validated caller-facing coordinates and layer
//! parameters into the canonical representation required by the numerical
//! backend.
//!
//! Compilation consists of three largely independent stages:
//!
//! - coordinate compilation;
//! - stack compilation;
//! - assembly of backend input and caller-facing context.
//!
//! Coordinate compilation is performed before stack compilation because the
//! canonical vacuum angular wavenumber may be required when evaluating
//! dispersive material properties.

pub(crate) mod context;
mod coordinates;
mod error;
mod layout;
mod problem;
mod stack;
mod validation;

pub use context::ProjectionConstraintError;
pub(crate) use context::{
    CompilationContext, CoordinateContext, ProjectionConstraint, StackContext,
};
pub use error::CompilePlaneWaveError;
pub(crate) use error::MappingError;
pub(crate) use layout::JetMapping;
pub(crate) use stack::{StackCompileError, StackThicknessJet, compile_canonical_constant_stack};
pub(crate) use validation::ValidationConfig;

use coordinates::CoordinateCompileError;
use stack::compile_stack;
use validation::ValidationError;

use crate::{
    ComplexPlane, ComplexScalar, SeedJet, Stack,
    algebra::ScalarAlgebra,
    domain::RealAxis,
    input::{
        CoordinateReference, Coordinates, InPlaneCoordinate,
        canonical::CanonicalProblem,
        compile::coordinates::CanonicalCoordinateJet,
        coordinate_input::{CoordinateInput, CoordinateValues},
    },
    material::{ConstitutiveEvaluator, ConstitutiveLift},
    parameter::{DerivativeMapping, DerivativeMappingError, Parameter},
};
use nalgebra::ComplexField;
use ndarray::{Array, Dimension};
use num_traits::{Float, FloatConst, FromPrimitive};
use std::fmt::Debug;

/// Aggregate capability required to compile a caller-facing problem into
/// a canonical jet-valued problem.
///
/// This is public only because it appears in bounds on public evaluator
/// implementations. It is not intended as a user-facing extension point.
#[doc(hidden)]
pub trait CompileJet<M, E>:
    SeedJet
    + CanonicalCoordinateJet
    + ScalarAlgebra
    + StackThicknessJet
    + ConstitutiveLift<E, M>
    + JetMapping
where
    Self::Scalar: ComplexScalar,
    <Self::Scalar as ComplexField>::RealField: Copy,
    Self::Dimension: Dimension,
    E: ConstitutiveEvaluator<Self::Scalar, Self::Dimension, M>,
{
}

impl<J, M, E> CompileJet<M, E> for J
where
    J::Scalar: ComplexScalar,
    <J::Scalar as ComplexField>::RealField: Copy,
    J::Dimension: Dimension,
    J: SeedJet
        + CanonicalCoordinateJet
        + ScalarAlgebra
        + StackThicknessJet
        + ConstitutiveLift<E, M>
        + JetMapping,
    E: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
{
}

#[derive(Clone, Debug)]
struct CompiledCore<M, J, R> {
    canonical: CanonicalProblem<M, J>,
    stack_context: StackContext<R>,
    projection_constraint: ProjectionConstraint,
}

pub(crate) fn compile_real<M, J>(
    input: CoordinateInput<<J::Scalar as ComplexField>::RealField, J::Dimension>,
    stack: &Stack<M, <J::Scalar as ComplexField>::RealField>,
    validation: &ValidationConfig<<J::Scalar as ComplexField>::RealField>,
    mapping: &DerivativeMapping,
) -> Result<
    (
        CanonicalProblem<M, J>,
        CompilationContext<<J::Scalar as ComplexField>::RealField, J::Dimension, J::Mapping>,
    ),
    CompilePlaneWaveError<J::Scalar>,
>
where
    J: CompileJet<M, RealAxis>,
    J::Scalar: ComplexScalar,
    <J::Scalar as ComplexField>::RealField: Float + FloatConst + FromPrimitive + Debug + Copy,
    J::Dimension: Dimension,
    M: Clone,
    RealAxis: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
{
    mapping.validate_against_stack(J::VARIABLE_SLOTS, stack.len())?;

    let (metadata, values, reference) = input.into_parts();

    let sampled_shape = values.raw_dim();

    let spectral = complexify(values.spectral());
    let in_plane = complexify(values.in_plane());

    let core = compile_core::<M, J, RealAxis>(
        metadata,
        &spectral,
        &in_plane,
        reference,
        sampled_shape,
        stack,
        validation,
        mapping,
    )?;

    let mapping = J::compile_mapping(mapping).map_err(MappingError::Mapping)?;

    Ok(finish_compilation(metadata, values, mapping, core))
}

pub(crate) fn compile_complex<M, J>(
    input: CoordinateInput<J::Scalar, J::Dimension>,
    stack: &Stack<M, <J::Scalar as ComplexField>::RealField>,
    validation: &ValidationConfig<<J::Scalar as ComplexField>::RealField>,
    mapping: &DerivativeMapping,
) -> Result<
    (
        CanonicalProblem<M, J>,
        CompilationContext<J::Scalar, J::Dimension, J::Mapping>,
    ),
    CompilePlaneWaveError<J::Scalar>,
>
where
    J: CompileJet<M, ComplexPlane>,
    J::Scalar: ComplexScalar,
    <J::Scalar as ComplexField>::RealField: Float + FloatConst + FromPrimitive + Debug + Copy,
    J::Dimension: Dimension,
    M: Clone,
    ComplexPlane: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
{
    mapping.validate_against_stack(J::VARIABLE_SLOTS, stack.len())?;

    let (metadata, values, reference) = input.into_parts();

    if matches!(metadata.in_plane(), InPlaneCoordinate::IncidentAngle(_)) {
        return Err(CompilePlaneWaveError::Coordinates(
            CoordinateCompileError::ComplexIncidentAngleUnsupported,
        ));
    }

    let sampled_shape = values.raw_dim();

    let core = compile_core::<M, J, ComplexPlane>(
        metadata,
        values.spectral(),
        values.in_plane(),
        reference,
        sampled_shape,
        stack,
        validation,
        mapping,
    )?;

    let mapping = J::compile_mapping(mapping).map_err(MappingError::Mapping)?;

    Ok(finish_compilation(metadata, values, mapping, core))
}

#[allow(clippy::too_many_arguments)]
fn compile_core<M, J, E>(
    metadata: Coordinates,
    spectral_values: &Array<J::Scalar, J::Dimension>,
    in_plane_values: &Array<J::Scalar, J::Dimension>,
    reference: CoordinateReference,
    sampled_shape: J::Dimension,
    stack: &Stack<M, <J::Scalar as ComplexField>::RealField>,
    validation: &ValidationConfig<<J::Scalar as ComplexField>::RealField>,
    mapping: &DerivativeMapping,
) -> Result<
    CompiledCore<M, J, <J::Scalar as ComplexField>::RealField>,
    CompilePlaneWaveError<J::Scalar>,
>
where
    J: CompileJet<M, E>,
    J::Scalar: ComplexScalar,
    <J::Scalar as ComplexField>::RealField: Float + FloatConst + FromPrimitive + Debug + Copy,
    J::Dimension: Dimension,
    M: Clone,
    E: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
{
    let compiled_coordinates = coordinates::compile_coordinates::<M, J, E>(
        metadata,
        spectral_values,
        in_plane_values,
        reference,
        stack,
        mapping,
    )?;

    let (canonical_coordinates, projection_constraint) = compiled_coordinates.into_parts();

    let compiled_stack = compile_stack::<M, J>(stack, sampled_shape, validation, mapping)?;

    let (canonical_stack, stack_context) = compiled_stack.into_parts();

    let canonical = CanonicalProblem::new(canonical_coordinates, canonical_stack);

    Ok(CompiledCore {
        canonical,
        stack_context,
        projection_constraint,
    })
}

fn finish_compilation<M, J, S, D>(
    metadata: Coordinates,
    values: CoordinateValues<S, D>,
    mapping: J::Mapping,
    core: CompiledCore<M, J, S::RealField>,
) -> (CanonicalProblem<M, J>, CompilationContext<S, D, J::Mapping>)
where
    S: ComplexField,
    D: Dimension,
    J: JetMapping,
{
    let coordinate_context = CoordinateContext::new(metadata, values);

    let context = CompilationContext::new(
        coordinate_context,
        core.stack_context,
        mapping,
        core.projection_constraint,
    );

    (core.canonical, context)
}

/// Convert sampled real values into the backend complex scalar type.
///
/// Coordinate compilation operates on real caller inputs, whereas jet
/// construction and canonicalisation are performed over the backend complex
/// scalar type.
///
/// The imaginary component of every sample is initialised to zero.
fn complexify<R, C, D>(values: &Array<R, D>) -> Array<C, D>
where
    R: Copy,
    C: ComplexField<RealField = R>,
    D: Dimension,
{
    values.mapv(C::from_real)
}

impl DerivativeMapping {
    fn validate_against_stack(
        &self,
        derivative_dimension: usize,
        finite_layer_count: usize,
    ) -> Result<(), MappingError> {
        if derivative_dimension != self.parameter_count() {
            return Err(DerivativeMappingError::IncompatibleShape {
                assigned_slots: self.parameter_count(),
                derivative_dimension,
            }
            .into());
        }

        if let Some(layer) = self
            .slots()
            .iter()
            .filter_map(|parameter| match parameter {
                Parameter::LayerThickness(layer) if layer.get() >= finite_layer_count => {
                    Some(*layer)
                }

                _ => None,
            })
            .next()
        {
            return Err(MappingError::LayerOutOfBounds {
                layer,
                finite_layer_count,
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use lamina_units::{AngleUnit, FrequencyUnit, InverseLengthUnit, Length};
    use ndarray::{Array, Dimension, Ix1, array};
    use num_complex::Complex64;

    use crate::{
        Constant,
        algebra::{ArrayJet0, ArrayJet1, HolomorphicParameter, RealParameter},
        input::{
            CoordinateInput, Coordinates, InPlaneCoordinate, IncidentSide, SpectralCoordinate,
            compile::coordinates::SpectralInputError,
        },
        parameter::{DerivativeMapping, FiniteLayerIndex, Parameter},
        stack::{Layer, Stack},
    };

    type C = Complex64;
    type R = f64;
    type D = Ix1;

    type RealValueJet = ArrayJet0<C, D, RealParameter>;

    type ComplexValueJet = ArrayJet0<C, D, HolomorphicParameter>;

    type RealFirstJet = ArrayJet1<C, D, RealParameter>;

    const TOLERANCE: f64 = 1.0e-12;

    fn assert_complex_eq(actual: C, expected: C) {
        let error = (actual - expected).norm();

        assert!(
            error <= TOLERANCE,
            "expected {expected:?}, got {actual:?}; \
             absolute error = {error:e}",
        );
    }

    fn assert_complex_array_eq<D>(actual: &Array<C, D>, expected: &Array<C, D>)
    where
        D: Dimension,
    {
        assert_eq!(actual.raw_dim(), expected.raw_dim());

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_complex_eq(actual, expected);
        }
    }

    fn coordinates_with_parallel_wavenumber() -> Coordinates {
        Coordinates::new(
            SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
            InPlaneCoordinate::ParallelAngularWavenumber(InverseLengthUnit::PerCentimetre),
        )
    }

    fn coordinates_with_effective_index() -> Coordinates {
        Coordinates::new(
            SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
            InPlaneCoordinate::EffectiveIndex,
        )
    }

    fn coordinates_with_angle() -> Coordinates {
        Coordinates::new(
            SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
            InPlaneCoordinate::IncidentAngle(AngleUnit::Radian),
        )
    }

    fn frequency_coordinates_with_angle() -> Coordinates {
        Coordinates::new(
            SpectralCoordinate::Frequency(FrequencyUnit::Hertz),
            InPlaneCoordinate::IncidentAngle(AngleUnit::Radian),
        )
    }

    fn real_input_with_parallel_wavenumber() -> CoordinateInput<R, D> {
        CoordinateInput::samples(
            coordinates_with_parallel_wavenumber(),
            array![2.0, 3.0],
            array![0.5, 0.75],
        )
        .unwrap()
    }

    fn complex_input_with_parallel_wavenumber() -> CoordinateInput<C, D> {
        CoordinateInput::samples(
            coordinates_with_parallel_wavenumber(),
            array![C::new(2.0, 0.1), C::new(3.0, -0.2),],
            array![C::new(0.5, 0.3), C::new(0.75, -0.4),],
        )
        .unwrap()
    }

    fn real_angle_input(side: IncidentSide) -> CoordinateInput<R, D> {
        CoordinateInput::incident_angle_samples(
            coordinates_with_angle(),
            array![2.0],
            array![0.25],
            side,
        )
        .unwrap()
    }

    fn complex_angle_input(side: IncidentSide) -> CoordinateInput<C, D> {
        CoordinateInput::incident_angle_samples(
            frequency_coordinates_with_angle(),
            array![C::new(1.0, 0.1)],
            array![C::new(0.2, 0.0)],
            side,
        )
        .unwrap()
    }

    fn no_derivatives() -> DerivativeMapping {
        DerivativeMapping::none()
    }

    fn spectral_derivative() -> DerivativeMapping {
        DerivativeMapping::new([Parameter::Spectral]).unwrap()
    }

    fn out_of_range_layer_mapping(layer_count: usize) -> DerivativeMapping {
        DerivativeMapping::new([Parameter::LayerThickness(FiniteLayerIndex::new(
            layer_count,
        ))])
        .unwrap()
    }

    fn validation() -> ValidationConfig<R> {
        ValidationConfig::strict()
    }

    fn test_stack() -> Stack<Constant<R>, R> {
        Stack::new(
            Constant::dielectric(1.0),
            vec![
                Layer::new(Constant::dielectric(4.0), Length::nanometres(500.0)),
                Layer::new(Constant::dielectric(2.0), Length::micrometres(2.0)),
            ],
            Constant::dielectric(1.0),
        )
    }

    fn test_stack_with_invalid_thickness() -> Stack<Constant<R>, R> {
        Stack::new(
            Constant::dielectric(1.0),
            vec![Layer::new(
                Constant::dielectric(2.0),
                Length::micrometres(-2.0),
            )],
            Constant::dielectric(1.0),
        )
    }

    fn test_stack_with_constant_exterior_index(index: R) -> Stack<Constant<R>, R> {
        let epsilon = index * index;

        Stack::new(
            Constant::dielectric(epsilon),
            vec![
                Layer::new(Constant::dielectric(4.0), Length::nanometres(500.0)),
                Layer::new(Constant::dielectric(2.0), Length::micrometres(2.0)),
            ],
            Constant::dielectric(epsilon),
        )
    }

    #[test]
    fn compile_real_complexifies_backend_coordinates() {
        let input = real_input_with_parallel_wavenumber();
        let stack = test_stack();

        let (canonical, _) =
            compile_real::<_, RealValueJet>(input, &stack, &validation(), &no_derivatives())
                .unwrap();

        let expected_spectral = array![C::new(2.0, 0.0), C::new(3.0, 0.0),];

        let expected_in_plane = array![C::new(0.5, 0.0), C::new(0.75, 0.0),];

        assert_complex_array_eq(
            canonical.coordinates().vacuum_angular_wavenumber().value(),
            &expected_spectral,
        );

        assert_complex_array_eq(
            canonical
                .coordinates()
                .parallel_angular_wavenumber()
                .value(),
            &expected_in_plane,
        );
    }

    #[test]
    fn compile_real_preserves_caller_values_in_context() {
        let input = real_input_with_parallel_wavenumber();

        let expected_spectral = input.spectral().clone();
        let expected_in_plane = input.in_plane().clone();

        let (_, context) =
            compile_real::<_, RealValueJet>(input, &test_stack(), &validation(), &no_derivatives())
                .unwrap();

        assert_eq!(
            context.coordinates().values().spectral(),
            &expected_spectral,
        );

        assert_eq!(
            context.coordinates().values().in_plane(),
            &expected_in_plane,
        );
    }

    #[test]
    fn compile_real_intrinsic_input_has_free_projection_constraint() {
        let (_, context) = compile_real::<_, RealValueJet>(
            real_input_with_parallel_wavenumber(),
            &test_stack(),
            &validation(),
            &no_derivatives(),
        )
        .unwrap();

        assert_eq!(context.projection_constraint(), ProjectionConstraint::Free,);
    }

    #[test]
    fn compile_real_angle_sets_fixed_projection_constraint() {
        let (_, context) = compile_real::<_, RealValueJet>(
            real_angle_input(IncidentSide::Left),
            &test_stack_with_constant_exterior_index(1.5),
            &validation(),
            &no_derivatives(),
        )
        .unwrap();

        assert_eq!(
            context.projection_constraint(),
            ProjectionConstraint::Fixed(IncidentSide::Left,),
        );
    }

    #[test]
    fn compile_real_angle_preserves_right_reference() {
        let (_, context) = compile_real::<_, RealValueJet>(
            real_angle_input(IncidentSide::Right),
            &test_stack_with_constant_exterior_index(1.5),
            &validation(),
            &no_derivatives(),
        )
        .unwrap();

        assert_eq!(
            context.projection_constraint(),
            ProjectionConstraint::Fixed(IncidentSide::Right,),
        );
    }

    #[test]
    fn compile_real_rejects_out_of_range_layer_mapping() {
        let stack = test_stack();

        let error = compile_real::<_, RealFirstJet>(
            real_input_with_parallel_wavenumber(),
            &stack,
            &validation(),
            &out_of_range_layer_mapping(stack.len()),
        )
        .unwrap_err();

        assert!(matches!(error, CompilePlaneWaveError::Mapping(_)));
    }

    #[test]
    fn compile_real_refines_value_mapping() {
        let (_, context) = compile_real::<_, RealValueJet>(
            real_input_with_parallel_wavenumber(),
            &test_stack(),
            &validation(),
            &no_derivatives(),
        )
        .unwrap();

        assert_eq!(context.mapping(), &crate::parameter::ValueMapping,);
    }

    #[test]
    fn compile_real_refines_directional_mapping() {
        let (_, context) = compile_real::<_, RealFirstJet>(
            real_input_with_parallel_wavenumber(),
            &test_stack(),
            &validation(),
            &spectral_derivative(),
        )
        .unwrap();

        assert_eq!(context.mapping().parameter(), Parameter::Spectral,);
    }

    #[test]
    fn compile_complex_preserves_backend_coordinates() {
        let input = complex_input_with_parallel_wavenumber();

        let expected_spectral = input.spectral().clone();
        let expected_in_plane = input.in_plane().clone();

        let (canonical, _) = compile_complex::<_, ComplexValueJet>(
            input,
            &test_stack(),
            &validation(),
            &no_derivatives(),
        )
        .unwrap();

        assert_complex_array_eq(
            canonical.coordinates().vacuum_angular_wavenumber().value(),
            &expected_spectral,
        );

        assert_complex_array_eq(
            canonical
                .coordinates()
                .parallel_angular_wavenumber()
                .value(),
            &expected_in_plane,
        );
    }

    #[test]
    fn compile_complex_preserves_caller_values_in_context() {
        let input = complex_input_with_parallel_wavenumber();

        let expected_spectral = input.spectral().clone();
        let expected_in_plane = input.in_plane().clone();

        let (_, context) = compile_complex::<_, ComplexValueJet>(
            input,
            &test_stack(),
            &validation(),
            &no_derivatives(),
        )
        .unwrap();

        assert_eq!(
            context.coordinates().values().spectral(),
            &expected_spectral,
        );

        assert_eq!(
            context.coordinates().values().in_plane(),
            &expected_in_plane,
        );
    }

    #[test]
    fn compile_complex_intrinsic_input_has_free_projection_constraint() {
        let (_, context) = compile_complex::<_, ComplexValueJet>(
            complex_input_with_parallel_wavenumber(),
            &test_stack(),
            &validation(),
            &no_derivatives(),
        )
        .unwrap();

        assert_eq!(context.projection_constraint(), ProjectionConstraint::Free,);
    }

    #[test]
    fn compile_complex_rejects_incident_angle_even_with_valid_reference() {
        let error = compile_complex::<_, ComplexValueJet>(
            complex_angle_input(IncidentSide::Left),
            &test_stack(),
            &ValidationConfig::permissive(),
            &no_derivatives(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            CompilePlaneWaveError::Coordinates(
                CoordinateCompileError::ComplexIncidentAngleUnsupported
            )
        ));
    }

    #[test]
    fn compile_complex_accepts_complex_effective_index() {
        let spectral = array![C::new(2.0, 0.1)];

        let effective_index = array![C::new(1.5, -0.2)];

        let input = CoordinateInput::samples(
            coordinates_with_effective_index(),
            spectral.clone(),
            effective_index.clone(),
        )
        .unwrap();

        let (canonical, context) = compile_complex::<_, ComplexValueJet>(
            input,
            &test_stack(),
            &validation(),
            &no_derivatives(),
        )
        .unwrap();

        let expected = spectral[0] * effective_index[0];

        assert_complex_eq(
            canonical
                .coordinates()
                .parallel_angular_wavenumber()
                .value()[0],
            expected,
        );

        assert_eq!(context.projection_constraint(), ProjectionConstraint::Free,);
    }

    #[test]
    fn compile_complex_validates_mapping_before_coordinate_policy() {
        type ComplexFirstJet = ArrayJet1<C, D, HolomorphicParameter>;

        let stack = test_stack();

        let error = compile_complex::<_, ComplexFirstJet>(
            complex_angle_input(IncidentSide::Left),
            &stack,
            &validation(),
            &out_of_range_layer_mapping(stack.len()),
        )
        .unwrap_err();

        assert!(matches!(error, CompilePlaneWaveError::Mapping(_)));
    }

    #[test]
    fn compile_core_assembles_coordinates_and_stack() {
        let coordinates = coordinates_with_parallel_wavenumber();

        let spectral = array![C::new(2.0, 0.1), C::new(3.0, -0.2),];

        let in_plane = array![C::new(0.5, 0.3), C::new(0.75, -0.4),];

        let stack = test_stack();

        let core = compile_core::<_, ComplexValueJet, ComplexPlane>(
            coordinates,
            &spectral,
            &in_plane,
            CoordinateReference::Intrinsic,
            spectral.raw_dim(),
            &stack,
            &validation(),
            &no_derivatives(),
        )
        .unwrap();

        assert_complex_array_eq(
            core.canonical
                .coordinates()
                .vacuum_angular_wavenumber()
                .value(),
            &spectral,
        );

        assert_complex_array_eq(
            core.canonical
                .coordinates()
                .parallel_angular_wavenumber()
                .value(),
            &in_plane,
        );

        assert_eq!(core.canonical.stack().layer_count(), stack.len(),);

        assert_eq!(core.projection_constraint, ProjectionConstraint::Free,);
    }

    #[test]
    fn compile_core_preserves_sample_shape_in_stack() {
        let spectral = array![C::new(2.0, 0.0), C::new(3.0, 0.0), C::new(4.0, 0.0),];

        let in_plane = array![C::new(0.5, 0.0), C::new(0.75, 0.0), C::new(1.0, 0.0),];

        let core = compile_core::<_, RealValueJet, RealAxis>(
            coordinates_with_parallel_wavenumber(),
            &spectral,
            &in_plane,
            CoordinateReference::Intrinsic,
            spectral.raw_dim(),
            &test_stack(),
            &validation(),
            &no_derivatives(),
        )
        .unwrap();

        for layer in core.canonical.stack().layers() {
            assert_eq!(layer.thickness_cm().raw_dim(), spectral.raw_dim(),);
        }
    }

    #[test]
    fn compile_core_propagates_coordinate_error() {
        let spectral = array![C::new(0.0, 1.0)];

        let in_plane = array![C::new(0.5, 0.0)];

        let error = compile_core::<_, ComplexValueJet, ComplexPlane>(
            coordinates_with_parallel_wavenumber(),
            &spectral,
            &in_plane,
            CoordinateReference::Intrinsic,
            spectral.raw_dim(),
            &test_stack(),
            &validation(),
            &no_derivatives(),
        )
        .unwrap_err();

        assert!(matches!(
            error,
            CompilePlaneWaveError::Coordinates(CoordinateCompileError::Spectral(
                SpectralInputError::NonPositive { .. }
            ))
        ));
    }

    #[test]
    fn compile_core_propagates_stack_error() {
        let spectral = array![C::new(2.0, 0.0)];

        let in_plane = array![C::new(0.5, 0.0)];

        let error = compile_core::<_, RealValueJet, RealAxis>(
            coordinates_with_parallel_wavenumber(),
            &spectral,
            &in_plane,
            CoordinateReference::Intrinsic,
            spectral.raw_dim(),
            &test_stack_with_invalid_thickness(),
            &validation(),
            &no_derivatives(),
        )
        .unwrap_err();

        assert!(matches!(error, CompilePlaneWaveError::Stack(_)));
    }

    #[test]
    fn compile_core_carries_projection_constraint() {
        let spectral = array![C::new(2.0, 0.0)];

        let angle = array![C::new(0.25, 0.0)];

        let core = compile_core::<_, RealValueJet, RealAxis>(
            coordinates_with_angle(),
            &spectral,
            &angle,
            CoordinateReference::IncidentSide(IncidentSide::Right),
            spectral.raw_dim(),
            &test_stack_with_constant_exterior_index(1.5),
            &validation(),
            &no_derivatives(),
        )
        .unwrap();

        assert_eq!(
            core.projection_constraint,
            ProjectionConstraint::Fixed(IncidentSide::Right,),
        );
    }

    #[test]
    fn compile_core_applies_coordinate_mapping() {
        let spectral = array![C::new(2.0, 0.0)];

        let in_plane = array![C::new(0.5, 0.0)];

        let core = compile_core::<_, RealFirstJet, RealAxis>(
            coordinates_with_parallel_wavenumber(),
            &spectral,
            &in_plane,
            CoordinateReference::Intrinsic,
            spectral.raw_dim(),
            &test_stack(),
            &validation(),
            &spectral_derivative(),
        )
        .unwrap();

        let vacuum_wavenumber = core.canonical.coordinates().vacuum_angular_wavenumber();

        assert_complex_array_eq(vacuum_wavenumber.first(), &Array::ones(spectral.raw_dim()));
    }
}
