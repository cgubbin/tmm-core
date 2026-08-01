//! Caller-facing information retained after canonical compilation.
//!
//! Compilation converts coordinates and layer thicknesses into backend units
//! and seeds derivative jets. The canonical backend input therefore no longer
//! contains all information needed to describe results in the representation
//! originally supplied by the caller.
//!
//! The context types in this module retain that information for derivative
//! interpretation, observable projection, labels, and reporting.

use nalgebra::ComplexField;
use ndarray::{Array, Dimension};

use crate::{
    IncidentSide, Polarisation,
    input::{Coordinates, coordinate_input::CoordinateValues},
    stack::Thickness,
};

/// Caller-facing information associated with a compiled backend problem.
///
/// This context records:
///
/// - the original coordinate kinds, values, and units;
/// - the requested incidence side and polarisation;
/// - finite-layer thicknesses in their caller-facing units;
/// - the mapping from derivative slots to physical parameters.
///
/// The numerical backend does not consume this information. It is retained to
/// interpret derivatives and project solved matrices into caller-facing
/// observables.
#[derive(Clone, Debug, PartialEq)]
pub struct CompilationContext<C, D, M>
where
    C: ComplexField,
    D: Dimension,
{
    coordinates: CoordinateContext<C, D>,
    stack: StackContext<C::RealField>,
    mapping: M,
    constraint: ProjectionConstraint,
}

impl<C, D, M> CompilationContext<C, D, M>
where
    C: ComplexField,
    D: Dimension,
{
    /// Construct context for a successfully compiled problem.
    pub(crate) fn new(
        coordinates: CoordinateContext<C, D>,
        stack: StackContext<C::RealField>,
        mapping: M,
        constraint: ProjectionConstraint,
    ) -> Self {
        Self {
            coordinates,
            stack,
            mapping,
            constraint,
        }
    }

    /// Return the caller-facing plane-wave coordinate context.
    pub(crate) fn coordinates(&self) -> &CoordinateContext<C, D> {
        &self.coordinates
    }

    /// Return the caller-facing finite-layer geometry.
    pub(crate) fn stack(&self) -> &StackContext<C::RealField> {
        &self.stack
    }

    /// Return the mapping from derivative slots to physical parameters.
    pub(crate) fn mapping(&self) -> &M {
        &self.mapping
    }

    /// Return the attached projection constraint
    pub(crate) fn projection_constraint(&self) -> ProjectionConstraint {
        self.constraint
    }

    /// Consume the context and return its components.
    pub(crate) fn into_parts(
        self,
    ) -> (
        CoordinateContext<C, D>,
        StackContext<C::RealField>,
        M,
        ProjectionConstraint,
    ) {
        (self.coordinates, self.stack, self.mapping, self.constraint)
    }
}

#[derive(Clone, Debug, PartialEq, Copy)]
pub enum ProjectionConstraint {
    Free,
    Fixed(IncidentSide),
}

/// Caller-facing geometric description of the compiled finite layers.
///
/// Material models remain in the canonical backend stack. This lightweight
/// context retains only the physical thickness values and units required for
/// derivative interpretation, labels, and reporting.
///
/// Layer order is the same geometric left-to-right order used by the canonical
/// stack.
#[derive(Clone, Debug, PartialEq)]
pub struct StackContext<R> {
    layer_thicknesses: Vec<Thickness<R>>,
}

impl<R> StackContext<R> {
    /// Construct context from finite-layer thicknesses in left-to-right order.
    pub(crate) fn new(layer_thicknesses: Vec<Thickness<R>>) -> Self {
        Self { layer_thicknesses }
    }

    /// Return all finite-layer thicknesses in left-to-right order.
    pub(crate) fn layer_thicknesses(&self) -> &[Thickness<R>] {
        &self.layer_thicknesses
    }

    /// Return the thickness of finite layer `index`.
    pub(crate) fn layer_thickness(&self, index: usize) -> Option<&Thickness<R>> {
        self.layer_thicknesses.get(index)
    }

    /// Return the number of finite layers.
    pub(crate) fn layer_count(&self) -> usize {
        self.layer_thicknesses.len()
    }

    /// Whether the stack contains no finite layers.
    pub(crate) fn is_empty(&self) -> bool {
        self.layer_thicknesses.is_empty()
    }

    /// Consume the context and return the finite-layer thicknesses.
    pub(crate) fn into_layer_thicknesses(self) -> Vec<Thickness<R>> {
        self.layer_thicknesses
    }
}

/// Original plane-wave description retained after canonicalisation.
///
/// Coordinate values remain in the exact representation and units supplied by
/// the caller. The incidence side is retained for observable projection; it
/// does not alter the fixed left-to-right backend traversal of the stack.
///
/// Polarisation is retained for result interpretation and reporting even
/// though it is also present in the canonical backend input.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CoordinateContext<R, D>
where
    D: Dimension,
{
    coordinates: Coordinates,
    values: CoordinateValues<R, D>,
}

impl<R, D> CoordinateContext<R, D>
where
    D: Dimension,
{
    /// Construct caller-facing coordinate context.
    pub(crate) fn new(coordinates: Coordinates, values: CoordinateValues<R, D>) -> Self {
        Self {
            coordinates,
            values,
        }
    }

    /// Return the coordinate parameterisations supplied by the caller.
    pub(crate) fn values(&self) -> &CoordinateValues<R, D> {
        &self.values
    }

    /// Return the coordinate parameterisations supplied by the caller.
    pub(crate) fn coordinates(&self) -> Coordinates {
        self.coordinates
    }

    /// Return the supplied spectral-coordinate values.
    pub(crate) fn spectral_values(&self) -> &Array<R, D> {
        self.values.spectral()
    }

    /// Return the supplied in-plane-coordinate values.
    pub(crate) fn in_plane_values(&self) -> &Array<R, D> {
        self.values.in_plane()
    }

    /// Consume the context and return its caller-facing components.
    pub(crate) fn into_parts(self) -> (Coordinates, Array<R, D>, Array<R, D>) {
        let (spectral_values, in_plane_values) = self.values.into_parts();

        (self.coordinates, spectral_values, in_plane_values)
    }
}

// #[cfg(test)]
// mod tests {
//     use ndarray::arr1;

//     use super::*;
//     use crate::input::{InPlaneCoordinate, Parameter, SpectralCoordinate};
//     use tmm_units::InverseLengthUnit;

//     #[test]
//     fn stack_context_preserves_layer_order() {
//         let thicknesses = vec![
//             Thickness::nanometres(10.0),
//             Thickness::micrometres(2.0),
//             Thickness::centimetres(0.1),
//         ];

//         let context = StackContext::new(thicknesses.clone());

//         assert_eq!(context.layer_thicknesses(), thicknesses.as_slice(),);
//         assert_eq!(context.layer_count(), 3);
//         assert!(!context.is_empty());

//         assert_eq!(context.layer_thickness(0), Some(&thicknesses[0]),);
//         assert_eq!(context.layer_thickness(2), Some(&thicknesses[2]),);
//         assert_eq!(context.layer_thickness(3), None);

//         assert_eq!(context.into_layer_thicknesses(), thicknesses,);
//     }

//     #[test]
//     fn empty_stack_context_is_empty() {
//         let context = StackContext::<f64>::new(Vec::new());

//         assert!(context.is_empty());
//         assert_eq!(context.layer_count(), 0);
//         assert_eq!(context.layer_thickness(0), None);
//     }

//     #[test]
//     fn coordinate_context_preserves_caller_representation() {
//         let spectral = arr1(&[1000.0, 1100.0]);
//         let in_plane = arr1(&[0.1, 0.2]);

//         let coordinates = Coordinates::new(
//             SpectralCoordinate::VacuumWavenumber(InverseLengthUnit::PerCentimetre),
//             InPlaneCoordinate::EffectiveIndex,
//         );

//         let values = CoordinateValues::new(spectral.clone(), in_plane.clone());

//         let context = CoordinateContext::new(coordinates, values, Polarisation::TransverseMagnetic);

//         assert_eq!(context.coordinates(), coordinates);
//         assert_eq!(context.spectral_values(), &spectral);
//         assert_eq!(context.in_plane_values(), &in_plane);
//         assert_eq!(context.polarisation(), Polarisation::TransverseMagnetic,);

//         let (returned_coordinates, returned_spectral, returned_in_plane, returned_polarisation) =
//             context.into_parts();

//         assert_eq!(returned_coordinates, coordinates);
//         assert_eq!(returned_spectral, spectral);
//         assert_eq!(returned_in_plane, in_plane);
//         assert_eq!(returned_polarisation, Polarisation::TransverseMagnetic,);
//     }

//     #[test]
//     fn compilation_context_preserves_all_components() {
//         let coordinate_context = CoordinateContext::new(
//             Coordinates::new(
//                 SpectralCoordinate::VacuumWavenumber(InverseLengthUnit::PerCentimetre),
//                 InPlaneCoordinate::ParallelWavenumber(InverseLengthUnit::PerMetre),
//             ),
//             CoordinateValues::new(arr1(&[1000.0]), arr1(&[100.0])),
//             Polarisation::TransverseElectric,
//         );

//         let stack_context = StackContext::new(vec![Thickness::nanometres(100.0)]);

//         let assignment =
//             ParameterAssignment::new([Parameter::Spectral, Parameter::LayerThickness { layer: 0 }])
//                 .unwrap();

//         let constraint = ProjectionConstraint::Free;

//         let context = CompilationContext::new(
//             coordinate_context.clone(),
//             stack_context.clone(),
//             assignment.clone(),
//             constraint,
//         );

//         assert_eq!(context.coordinates(), &coordinate_context,);
//         assert_eq!(context.stack(), &stack_context);
//         assert_eq!(context.assignment(), &assignment);
//         assert_eq!(context.constraint(), constraint);

//         assert_eq!(
//             context.into_parts(),
//             (coordinate_context, stack_context, assignment, constraint),
//         );
//     }
// }
