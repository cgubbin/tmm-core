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

mod assignment;
mod context;
mod coordinates;
mod error;
mod problem;
mod seed;
mod stack;

pub(crate) use assignment::{ParameterAssignment, ParameterAssignmentError};
pub(crate) use context::{
    CompilationContext, CoordinateContext, ProjectionConstraint, StackContext,
};
pub use error::CompilePlaneWaveError;
pub(crate) use problem::CompiledProblem;
pub(crate) use seed::SeedJet;

use coordinates::{CoordinateCompileError, compile_coordinates};
use stack::{StackCompileError, compile_stack};

use crate::{
    ComplexPlane, ComplexScalar, IncidentSide, Material, MeromorphicMaterial, Polarisation, Stack,
    ValidationConfig,
    algebra::ScalarAlgebra,
    domain::RealAxis,
    input::{
        CanonicalBackendInput, CanonicalCoordinates, InPlaneCoordinate, JetEvaluation,
        PlaneWaveCoordinates, PlaneWaveInput,
        canonical::CanonicalProblem,
        compile::{
            assignment::CoordinateAssignment, coordinates::CanonicalCoordinateJet,
            stack::StackThicknessJet,
        },
        plane_wave::{PlaneWaveCoordinateValues, PlaneWaveCoordinatesInput},
    },
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};
use nalgebra::ComplexField;
use ndarray::{Array, Dimension};
use num_traits::{Float, FloatConst, FromPrimitive};
use std::fmt::Debug;

pub(crate) trait CompileJet<M, E>:
    SeedJet
    + CanonicalCoordinateJet
    + ScalarAlgebra
    + StackThicknessJet
    + ConstitutiveLift<E, M>
    + JetEvaluation
where
    Self::Scalar: ComplexScalar,
    <Self::Scalar as ComplexField>::RealField: Copy,
    Self::Dimension: Dimension,
    E: ConstitutiveEvaluator<Self::Scalar, Self::Dimension, M>,
{
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CoordinateReference {
    Intrinsic,
    IncidentSide(IncidentSide),
}

pub struct CoordinateInput<S, D>
where
    D: Dimension,
{
    coordinates: PlaneWaveCoordinates,
    values: PlaneWaveCoordinateValues<S, D>,
    reference: CoordinateReference,
}

impl<S, D: Dimension> CoordinateInput<S, D> {
    pub(crate) fn new(
        coordinates: PlaneWaveCoordinates,
        values: PlaneWaveCoordinateValues<S, D>,
        reference: CoordinateReference,
    ) -> Self {
        Self {
            coordinates,
            values,
            reference,
        }
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        PlaneWaveCoordinates,
        PlaneWaveCoordinateValues<S, D>,
        CoordinateReference,
    ) {
        (self.coordinates, self.values, self.reference)
    }
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
        + JetEvaluation,
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
    assignment: ParameterAssignment,
) -> Result<
    (
        CanonicalProblem<M, J>,
        CompilationContext<<J::Scalar as ComplexField>::RealField, J::Dimension, J::Assignment>,
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
    assignment.validate(J::VARIABLE_SLOTS, stack.len())?;

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
        &assignment,
    )?;

    let assignment = J::refine_assignment(assignment)?;

    Ok(finish_compilation(metadata, values, assignment, core))
}

pub(crate) fn compile_complex<M, J>(
    input: CoordinateInput<J::Scalar, J::Dimension>,
    stack: &Stack<M, <J::Scalar as ComplexField>::RealField>,
    validation: &ValidationConfig<<J::Scalar as ComplexField>::RealField>,
    assignment: ParameterAssignment,
) -> Result<
    (
        CanonicalProblem<M, J>,
        CompilationContext<J::Scalar, J::Dimension, J::Assignment>,
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
    assignment.validate(J::VARIABLE_SLOTS, stack.len())?;

    let (metadata, values, reference) = input.into_parts();

    if matches!(metadata.in_plane, InPlaneCoordinate::IncidentAngle(_)) {
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
        &assignment,
    )?;

    let assignment = J::refine_assignment(assignment)?;

    Ok(finish_compilation(metadata, values, assignment, core))
}

fn compile_core<M, J, E>(
    metadata: PlaneWaveCoordinates,
    spectral_values: &Array<J::Scalar, J::Dimension>,
    in_plane_values: &Array<J::Scalar, J::Dimension>,
    reference: CoordinateReference,
    sampled_shape: J::Dimension,
    stack: &Stack<M, <J::Scalar as ComplexField>::RealField>,
    validation: &ValidationConfig<<J::Scalar as ComplexField>::RealField>,
    assignment: &ParameterAssignment,
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
        &assignment.coordinates(),
    )?;

    let (canonical_coordinates, projection_constraint) = compiled_coordinates.into_parts();

    let compiled_stack =
        compile_stack::<M, J, _>(stack, sampled_shape, validation, &assignment.thicknesses())?;

    let (canonical_stack, stack_context) = compiled_stack.into_parts();

    let canonical = CanonicalProblem::new(canonical_coordinates, canonical_stack);

    Ok(CompiledCore {
        canonical,
        stack_context,
        projection_constraint,
    })
}

fn finish_compilation<M, J, S, D>(
    metadata: PlaneWaveCoordinates,
    values: PlaneWaveCoordinateValues<S, D>,
    assignment: J::Assignment,
    core: CompiledCore<M, J, S::RealField>,
) -> (
    CanonicalProblem<M, J>,
    CompilationContext<S, D, J::Assignment>,
)
where
    S: ComplexField,
    D: Dimension,
    J: JetEvaluation,
{
    let coordinate_context = CoordinateContext::new(metadata, values);

    let context = CompilationContext::new(
        coordinate_context,
        core.stack_context,
        assignment,
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

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::{Array, Ix1, array};
    use num_complex::Complex64;
    use tmm_units::{AngleUnit, FrequencyUnit, InverseLengthUnit};

    use crate::{
        Constant,
        algebra::{ArrayJet0, RealParameter},
        input::SpectralCoordinate,
        stack::{Layer, Thickness},
    };

    use super::*;

    type C = Complex64;
    type R = f64;
    type D = Ix1;

    const TOLERANCE: f64 = 1.0e-12;

    fn assert_complex_eq(actual: C, expected: C) {
        let error = (actual - expected).norm();

        assert!(
            error <= TOLERANCE,
            "expected {expected:?}, got {actual:?}; \
             absolute error = {error:e}",
        );
    }

    fn assert_complex_array_eq<D: Dimension>(actual: &Array<C, D>, expected: &Array<C, D>) {
        assert_eq!(actual.raw_dim(), expected.raw_dim());

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_complex_eq(actual, expected);
        }
    }

    type TestJet = crate::algebra::Jet0<Array<C, D>>;

    fn metadata_with_beta() -> PlaneWaveCoordinates {
        PlaneWaveCoordinates::new(
            SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
            InPlaneCoordinate::ParallelAngularWavenumber(InverseLengthUnit::PerCentimetre),
        )
    }

    fn metadata_with_angle() -> PlaneWaveCoordinates {
        PlaneWaveCoordinates::new(
            SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
            InPlaneCoordinate::IncidentAngle(AngleUnit::Radian),
        )
    }

    fn real_value() -> PlaneWaveCoordinateValues<R, D> {
        PlaneWaveCoordinateValues::new(array![2.0, 3.0], array![0.5, 0.75])
    }

    fn complex_value() -> PlaneWaveCoordinateValues<C, D> {
        PlaneWaveCoordinateValues::new(
            array![C::new(2.0, 0.1), C::new(3.0, -0.2),],
            array![C::new(0.5, 0.3), C::new(0.75, -0.4),],
        )
    }

    fn assignment() -> ParameterAssignment {
        ParameterAssignment::none()
    }

    fn assignment_with_out_of_range_layer(layer_count: usize) -> ParameterAssignment {
        ParameterAssignment::layer_thickness(layer_count)
    }

    fn validation() -> ValidationConfig<R> {
        ValidationConfig::strict()
    }

    fn test_stack() -> Stack<Constant<f64>, f64> {
        Stack::new(
            Constant::dielectric(1.0),
            vec![
                Layer::new(Constant::dielectric(4.0), Thickness::nanometres(500.0)),
                Layer::new(Constant::dielectric(2.0), Thickness::micrometres(2.0)),
            ],
            Constant::dielectric(1.0),
        )
    }

    fn test_stack_with_invalid_thickness() -> Stack<Constant<f64>, f64> {
        Stack::new(
            Constant::dielectric(1.0),
            vec![Layer::new(
                Constant::dielectric(2.0),
                Thickness::micrometres(-2.0),
            )],
            Constant::dielectric(1.0),
        )
    }

    fn test_stack_with_constant_exterior_index(index: f64) -> Stack<Constant<f64>, f64> {
        let eps = index * index;
        Stack::new(
            Constant::dielectric(eps),
            vec![
                Layer::new(Constant::dielectric(4.0), Thickness::nanometres(500.0)),
                Layer::new(Constant::dielectric(2.0), Thickness::micrometres(2.0)),
            ],
            Constant::dielectric(eps),
        )
    }

    #[test]
    fn compile_complex_rejects_incident_angle_even_with_incident_reference() {
        let input = CoordinateInput {
            coordinates: PlaneWaveCoordinates {
                spectral: SpectralCoordinate::Frequency(FrequencyUnit::Hertz),
                in_plane: InPlaneCoordinate::IncidentAngle(AngleUnit::Radian),
            },
            values: PlaneWaveCoordinateValues::new(
                array![Complex64::new(1.0, 0.1)],
                array![Complex64::new(0.2, 0.0)],
            ),
            reference: CoordinateReference::IncidentSide(IncidentSide::Left),
        };

        let stack = test_stack();

        let error = compile_complex::<_, ArrayJet0<Complex64, Ix1, RealParameter>>(
            input,
            &stack,
            &ValidationConfig::permissive(),
            ParameterAssignment::default(),
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
    fn compile_complex_rejects_intrinsic_incident_angle_as_unsupported() {
        let input = CoordinateInput {
            coordinates: PlaneWaveCoordinates {
                spectral: SpectralCoordinate::Frequency(FrequencyUnit::Hertz),
                in_plane: InPlaneCoordinate::IncidentAngle(AngleUnit::Radian),
            },
            values: PlaneWaveCoordinateValues::new(
                array![Complex64::new(1.0, 0.1)],
                array![Complex64::new(0.2, 0.0)],
            ),
            reference: CoordinateReference::Intrinsic,
        };

        let stack = test_stack();

        let error = compile_complex::<_, ArrayJet0<Complex64, Ix1, RealParameter>>(
            input,
            &stack,
            &ValidationConfig::permissive(),
            ParameterAssignment::default(),
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
    fn compile_real_complexifies_backend_coordinates() {
        let values = real_value();

        let input = CoordinateInput::new(
            metadata_with_beta(),
            values.clone(),
            CoordinateReference::Intrinsic,
        );

        let stack = test_stack();

        let compiled =
            compile_real::<_, TestJet>(input, &stack, &validation(), assignment()).unwrap();

        let canonical = compiled.0;

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
    fn compile_real_preserves_real_caller_values_in_context() {
        let values = real_value();

        let input = CoordinateInput::new(
            metadata_with_beta(),
            values.clone(),
            CoordinateReference::Intrinsic,
        );

        let stack = test_stack();

        let compiled =
            compile_real::<_, TestJet>(input, &stack, &validation(), assignment()).unwrap();

        assert_eq!(compiled.1.coordinates().values(), &values,);
    }

    #[test]
    fn compile_real_propagates_missing_incident_side() {
        let values = PlaneWaveCoordinateValues::new(array![2.0], array![0.25]);

        let input = CoordinateInput::new(
            metadata_with_angle(),
            values,
            CoordinateReference::Intrinsic,
        );

        let stack = test_stack();

        let error =
            compile_real::<_, TestJet>(input, &stack, &validation(), assignment()).unwrap_err();

        assert!(matches!(
            error,
            CompilePlaneWaveError::Coordinates(CoordinateCompileError::MissingIncidentSide)
        ));
    }

    #[test]
    fn compile_real_angle_sets_fixed_projection_constraint() {
        let values = PlaneWaveCoordinateValues::new(array![2.0], array![0.25]);

        let input = CoordinateInput::new(
            metadata_with_angle(),
            values,
            CoordinateReference::IncidentSide(IncidentSide::Left),
        );

        let stack = test_stack_with_constant_exterior_index(1.5);

        let compiled =
            compile_real::<_, TestJet>(input, &stack, &validation(), assignment()).unwrap();

        assert_eq!(
            compiled.1.projection_constraint(),
            ProjectionConstraint::Fixed(IncidentSide::Left),
        );
    }

    #[test]
    fn compile_real_validates_assignment_before_compilation() {
        let input = CoordinateInput::new(
            metadata_with_beta(),
            real_value(),
            CoordinateReference::Intrinsic,
        );

        let stack = test_stack();

        let invalid_assignment = assignment_with_out_of_range_layer(stack.len());

        let error = compile_real::<_, TestJet>(input, &stack, &validation(), invalid_assignment)
            .unwrap_err();

        assert!(matches!(error, CompilePlaneWaveError::Assignment(_)));
    }

    #[test]
    fn compile_complex_preserves_complex_backend_coordinates() {
        let values = complex_value();

        let input = CoordinateInput::new(
            metadata_with_beta(),
            values.clone(),
            CoordinateReference::Intrinsic,
        );

        let stack = test_stack();

        let compiled =
            compile_complex::<_, TestJet>(input, &stack, &validation(), assignment()).unwrap();

        assert_complex_array_eq(
            compiled.0.coordinates().vacuum_angular_wavenumber().value(),
            values.spectral(),
        );

        assert_complex_array_eq(
            compiled
                .0
                .coordinates()
                .parallel_angular_wavenumber()
                .value(),
            values.in_plane(),
        );
    }

    #[test]
    fn compile_complex_preserves_complex_caller_values_in_context() {
        let values = complex_value();

        let input = CoordinateInput::new(
            metadata_with_beta(),
            values.clone(),
            CoordinateReference::Intrinsic,
        );

        let stack = test_stack();

        let compiled =
            compile_complex::<_, TestJet>(input, &stack, &validation(), assignment()).unwrap();

        assert_eq!(compiled.1.coordinates().values(), &values,);
    }

    #[test]
    fn compile_complex_rejects_incident_angle_with_intrinsic_reference() {
        let values =
            PlaneWaveCoordinateValues::new(array![C::new(2.0, 0.1)], array![C::new(0.25, 0.0)]);

        let input = CoordinateInput::new(
            metadata_with_angle(),
            values,
            CoordinateReference::Intrinsic,
        );

        let stack = test_stack();

        let error =
            compile_complex::<_, TestJet>(input, &stack, &validation(), assignment()).unwrap_err();

        assert!(matches!(
            error,
            CompilePlaneWaveError::Coordinates(
                CoordinateCompileError::ComplexIncidentAngleUnsupported
            )
        ));
    }

    #[test]
    fn compile_complex_rejects_incident_angle_even_with_side() {
        let values =
            PlaneWaveCoordinateValues::new(array![C::new(2.0, 0.1)], array![C::new(0.25, 0.2)]);

        let input = CoordinateInput::new(
            metadata_with_angle(),
            values,
            CoordinateReference::IncidentSide(IncidentSide::Left),
        );

        let stack = test_stack();

        let error =
            compile_complex::<_, TestJet>(input, &stack, &validation(), assignment()).unwrap_err();

        assert!(matches!(
            error,
            CompilePlaneWaveError::Coordinates(
                CoordinateCompileError::ComplexIncidentAngleUnsupported
            )
        ));
    }

    #[test]
    fn compile_complex_accepts_complex_effective_index() {
        let metadata = PlaneWaveCoordinates::new(
            SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
            InPlaneCoordinate::EffectiveIndex,
        );

        let values =
            PlaneWaveCoordinateValues::new(array![C::new(2.0, 0.1)], array![C::new(1.5, -0.2)]);

        let input = CoordinateInput::new(metadata, values.clone(), CoordinateReference::Intrinsic);

        let stack = test_stack();

        let compiled =
            compile_complex::<_, TestJet>(input, &stack, &validation(), assignment()).unwrap();

        let expected = values.spectral()[0] * values.in_plane()[0];

        assert_complex_eq(
            compiled
                .0
                .coordinates()
                .parallel_angular_wavenumber()
                .value()[0],
            expected,
        );

        assert_eq!(
            compiled.1.projection_constraint(),
            ProjectionConstraint::Free,
        );
    }

    #[test]
    fn compile_complex_validates_assignment_before_angle_policy() {
        let values =
            PlaneWaveCoordinateValues::new(array![C::new(2.0, 0.1)], array![C::new(0.25, 0.0)]);

        let input = CoordinateInput::new(
            metadata_with_angle(),
            values,
            CoordinateReference::Intrinsic,
        );

        let stack = test_stack();

        let invalid_assignment = assignment_with_out_of_range_layer(stack.len());

        let error = compile_complex::<_, TestJet>(input, &stack, &validation(), invalid_assignment)
            .unwrap_err();

        assert!(matches!(error, CompilePlaneWaveError::Assignment(_)));
    }

    #[test]
    fn compile_core_assembles_coordinates_and_stack() {
        let metadata = metadata_with_beta();

        let spectral = array![C::new(2.0, 0.1), C::new(3.0, -0.2),];

        let in_plane = array![C::new(0.5, 0.3), C::new(0.75, -0.4),];

        let stack = test_stack();

        let assignment = assignment();

        let core = compile_core::<_, TestJet, ComplexPlane>(
            metadata,
            &spectral,
            &in_plane,
            CoordinateReference::Intrinsic,
            spectral.raw_dim(),
            &stack,
            &validation(),
            &assignment,
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
    fn compile_core_preserves_sample_shape_in_compiled_stack() {
        let metadata = metadata_with_beta();

        let spectral = array![C::new(2.0, 0.0), C::new(3.0, 0.0), C::new(4.0, 0.0),];

        let in_plane = array![C::new(0.5, 0.0), C::new(0.75, 0.0), C::new(1.0, 0.0),];

        let stack = test_stack();
        let assignment = assignment();

        let core = compile_core::<_, TestJet, RealAxis>(
            metadata,
            &spectral,
            &in_plane,
            CoordinateReference::Intrinsic,
            spectral.raw_dim(),
            &stack,
            &validation(),
            &assignment,
        )
        .unwrap();

        for layer in core.canonical.stack().layers() {
            assert_eq!(layer.thickness_cm().raw_dim(), spectral.raw_dim(),);
        }
    }

    #[test]
    fn compile_core_propagates_coordinate_error() {
        let metadata = metadata_with_beta();

        let spectral = array![C::new(0.0, 1.0)];
        let in_plane = array![C::new(0.5, 0.0)];

        let stack = test_stack();
        let assignment = assignment();

        let error = compile_core::<_, TestJet, ComplexPlane>(
            metadata,
            &spectral,
            &in_plane,
            CoordinateReference::Intrinsic,
            spectral.raw_dim(),
            &stack,
            &validation(),
            &assignment,
        )
        .unwrap_err();

        assert!(matches!(
            error,
            CompilePlaneWaveError::Coordinates(CoordinateCompileError::Spectral(
                super::coordinates::SpectralInputError::NonPositive { .. }
            ))
        ));
    }

    #[test]
    fn compile_core_propagates_stack_error() {
        let metadata = metadata_with_beta();

        let spectral = array![C::new(2.0, 0.0)];
        let in_plane = array![C::new(0.5, 0.0)];

        let stack = test_stack_with_invalid_thickness();
        let assignment = assignment();

        let error = compile_core::<_, TestJet, RealAxis>(
            metadata,
            &spectral,
            &in_plane,
            CoordinateReference::Intrinsic,
            spectral.raw_dim(),
            &stack,
            &validation(),
            &assignment,
        )
        .unwrap_err();

        assert!(matches!(error, CompilePlaneWaveError::Stack(_)));
    }

    #[test]
    fn compile_core_carries_projection_constraint_from_coordinates() {
        let metadata = metadata_with_angle();

        let spectral = array![C::new(2.0, 0.0)];
        let angle = array![C::new(0.25, 0.0)];

        let stack = test_stack_with_constant_exterior_index(1.5);

        let assignment = assignment();

        let core = compile_core::<_, TestJet, RealAxis>(
            metadata,
            &spectral,
            &angle,
            CoordinateReference::IncidentSide(IncidentSide::Right),
            spectral.raw_dim(),
            &stack,
            &validation(),
            &assignment,
        )
        .unwrap();

        assert_eq!(
            core.projection_constraint,
            ProjectionConstraint::Fixed(IncidentSide::Right),
        );
    }

    #[test]
    fn compile_core_applies_coordinate_assignment() {
        type FirstJet = crate::algebra::Jet1<Array<C, D>>;

        let metadata = metadata_with_beta();

        let spectral = array![C::new(2.0, 0.0)];
        let in_plane = array![C::new(0.5, 0.0)];

        let stack = test_stack();

        let assignment = ParameterAssignment::spectral();

        let core = compile_core::<_, FirstJet, RealAxis>(
            metadata,
            &spectral,
            &in_plane,
            CoordinateReference::Intrinsic,
            spectral.raw_dim(),
            &stack,
            &validation(),
            &assignment,
        )
        .unwrap();

        let omega = core.canonical.coordinates().vacuum_angular_wavenumber();

        assert_complex_array_eq(omega.first(), &Array::ones(spectral.raw_dim()));
    }
}
