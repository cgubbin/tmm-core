mod assignment;
mod context;
mod coordinates;
mod error;
mod problem;
mod seed;
mod stack;

pub(crate) use assignment::{ParameterAssignment, ParameterAssignmentError};
pub(crate) use context::{CompilationContext, CoordinateContext, StackContext};
pub use error::CompilePlaneWaveError;
pub(crate) use problem::CompiledProblem;
pub(crate) use seed::SeedJet;

use coordinates::{CoordinateCompileError, compile_in_plane, compile_spectral};
use stack::{StackCompileError, compile_stack};

use crate::{
    ComplexScalar, Material, Stack, ValidationConfig,
    algebra::ScalarAlgebra,
    domain::RealAxis,
    input::{
        CanonicalBackendInput, CanonicalCoordinates, InPlaneCoordinate, PlaneWaveInput,
        compile::coordinates::CanonicalCoordinateJet,
    },
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};
use nalgebra::ComplexField;
use ndarray::{Array, Dimension};
use num_traits::{Float, FloatConst, FromPrimitive};

pub(crate) trait CompilePlaneWaveJet<C: ComplexField, D: Dimension>:
    SeedJet<Array<C, D>> + CanonicalCoordinateJet<C, D> + ScalarAlgebra<C, D>
{
}

impl<C, D, J> CompilePlaneWaveJet<C, D> for J
where
    C: ComplexField,
    D: Dimension,
    J: SeedJet<Array<C, D>> + CanonicalCoordinateJet<C, D> + ScalarAlgebra<C, D>,
{
}

pub(crate) fn compile_plane_wave_problem<M, C, D, J>(
    input: PlaneWaveInput<C::RealField, D>,
    stack: &Stack<M, C::RealField>,
    validation: &ValidationConfig<C::RealField>,
    assignment: ParameterAssignment,
) -> Result<
    (
        CanonicalBackendInput<M, J>,
        CompilationContext<C::RealField, D>,
    ),
    CompilePlaneWaveError<C::RealField>,
>
where
    C: ComplexScalar,
    C::RealField: Float + FloatConst + FromPrimitive + std::fmt::Debug + Copy,
    D: Dimension + Clone,
    J: SeedJet<Array<C, D>>
        + CanonicalCoordinateJet<C, D>
        + ScalarAlgebra<C, D>
        + ConstitutiveLift<C, D, RealAxis, M>,
    M: Clone + Material<Real = C::RealField>,
{
    //     assignment.validate(J::VARIABLE_SLOTS, stack.len())?;

    //     let (coordinate_metadata, coordinate_values, polarisation, incident_side) = input.into_parts();

    //     let sampled_shape = coordinate_values.raw_dim();

    //     let coordinate_assignment = assignment.coordinates();

    //     let spectral = compile_spectral::<C, D, J>(
    //         coordinate_values.spectral(),
    //         coordinate_metadata.spectral,
    //         coordinate_assignment.spectral_slot(),
    //     )?;

    //     let incident_index = match coordinate_metadata.in_plane {
    //         InPlaneCoordinate::IncidentAngle(_) => {
    //             let material = stack.incident_exterior(incident_side);

    //             Some(
    //                 ConstitutiveLift::refractive_index(material, spectral.vacuum_angular_wavenumber()), // .map_err(|source| CompileProblemError::IncidentIndex { source })?,
    //             )
    //         }

    //         _ => None,
    //     };

    //     let parallel_angular_wavenumber = compile_in_plane::<C, D, J>(
    //         coordinate_values.in_plane(),
    //         coordinate_metadata.in_plane,
    //         spectral.vacuum_angular_wavenumber(),
    //         incident_index.as_ref(),
    //         coordinate_assignment.in_plane_slot(),
    //     )?;

    //     let canonical_coordinates =
    //         CanonicalCoordinates::new(spectral.into_inner(), parallel_angular_wavenumber);

    //     let coordinate_context = CoordinateContext::new(
    //         coordinate_metadata,
    //         coordinate_values,
    //         incident_side,
    //         polarisation,
    //     );

    //     let compiled_stack =
    //         compile_stack::<M, C, D, J>(stack, sampled_shape, validation, assignment.thicknesses())?;

    //     let (canonical_stack, stack_context) = compiled_stack.into_parts();

    //     let canonical =
    //         CanonicalBackendInput::new(canonical_coordinates, polarisation, canonical_stack);

    //     let context = CompilationContext::new(coordinate_context, stack_context, assignment);

    //     Ok((canonical, context))
    todo!()
}

fn complexify<R, C, D>(values: &Array<R, D>) -> Array<C, D>
where
    R: Copy,
    C: ComplexField<RealField = R>,
    D: Dimension,
{
    values.mapv(C::from_real)
}
