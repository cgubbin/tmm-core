mod assignment;
mod context;
mod coordinates;
mod error;
mod mode;
mod plan;
mod problem;
mod seed;
mod stack;

pub(crate) use assignment::ParameterAssignment;
pub(crate) use context::{CompilationContext, CoordinateContext, StackContext};
pub use error::CompileProblemError;
pub(crate) use plan::{CompilationPlan, plan_compilation};
pub(crate) use problem::CompiledProblem;
pub(crate) use seed::SeedJet;

use coordinates::{CoordinateCompileError, compile_in_plane, compile_spectral};
use stack::{StackCompileError, compile_stack};

use crate::{
    ComplexScalar, Material, Stack, ValidationConfig,
    domain::RealAxis,
    input::{
        CanonicalCoordinates, CanonicalPlaneWaveProblem, CanonicalProblem, InPlaneCoordinate,
        PlaneWaveInput,
        compile::{coordinates::CoordinateJet, stack::StackJet},
    },
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};
use nalgebra::ComplexField;
use ndarray::{Array, Dimension};
use num_traits::{Float, FloatConst, FromPrimitive};

pub(crate) fn compile_problem<M, R, C, D, J, Domain>(
    input: PlaneWaveInput<R, D>,
    stack: &Stack<M, R>,
    validation: &ValidationConfig<R>,
    assignment: ParameterAssignment,
) -> Result<(CanonicalPlaneWaveProblem<M, J>, CompilationContext<R, D>), CompileProblemError<R>>
where
    R: Float + FloatConst + FromPrimitive + std::fmt::Debug + Copy,
    C: ComplexScalar<RealField = R>,
    D: Dimension + Clone,
    J: SeedJet<Array<C, D>>
        + CoordinateJet<C, D>
        + StackJet<C, D>
        + ConstitutiveLift<C, D, Domain, M>,
    Domain: ConstitutiveEvaluator<C, D, M>,
    M: Clone + Material<Real = R>,
{
    assignment.validate(J::VARIABLE_SLOTS, stack.len())?;

    let (coordinate_metadata, coordinate_values, polarisation, incident_side) = input.into_parts();

    let sampled_shape = coordinate_values.raw_dim();

    let coordinate_assignment = assignment.coordinates();

    let spectral = compile_spectral::<C, D, J>(
        coordinate_values.spectral(),
        coordinate_metadata.spectral,
        coordinate_assignment.spectral_slot(),
    )?;

    let incident_index = match coordinate_metadata.in_plane {
        InPlaneCoordinate::IncidentAngle(_) => {
            let material = stack.incident_exterior(incident_side);

            Some(
                ConstitutiveLift::refractive_index(material, spectral.vacuum_angular_wavenumber()), // .map_err(|source| CompileProblemError::IncidentIndex { source })?,
            )
        }

        _ => None,
    };

    let parallel_angular_wavenumber = compile_in_plane::<C, D, J>(
        coordinate_values.in_plane(),
        coordinate_metadata.in_plane,
        spectral.vacuum_angular_wavenumber(),
        incident_index.as_ref(),
        coordinate_assignment.in_plane_slot(),
    )?;

    let canonical_coordinates =
        CanonicalCoordinates::new(spectral.into_inner(), parallel_angular_wavenumber);

    let coordinate_context = CoordinateContext::new(
        coordinate_metadata,
        coordinate_values,
        incident_side,
        polarisation,
    );

    let compiled_stack =
        compile_stack::<M, C, D, J>(stack, sampled_shape, validation, assignment.thicknesses())?;

    let (canonical_stack, stack_context) = compiled_stack.into_parts();

    let canonical = CanonicalPlaneWaveProblem::new(
        canonical_coordinates,
        polarisation,
        incident_side,
        canonical_stack,
    );

    let context = CompilationContext::new(coordinate_context, stack_context, assignment);

    Ok((canonical, context))
}

fn complexify<R, C, D>(values: &Array<R, D>) -> Array<C, D>
where
    R: Copy,
    C: ComplexField<RealField = R>,
    D: Dimension,
{
    values.mapv(C::from_real)
}
