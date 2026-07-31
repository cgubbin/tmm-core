//! Assignment of caller-facing parameters to derivative slots.
//!
//! Numerical jets identify independent variables by numbered slots, whereas
//! callers identify them by physical meaning, such as the supplied spectral
//! coordinate or a particular layer thickness.
//!
//! [`ParameterAssignment`] records the correspondence between those two
//! representations. Entry `i` identifies the physical parameter represented
//! by derivative slot `i`.
//!
//! For example, the assignment
//!
//! ```text
//! slot 0 -> spectral coordinate
//! slot 1 -> layer 3 thickness
//! ```
//!
//! causes coordinate compilation to seed the spectral coordinate into slot
//! zero and stack compilation to seed layer 3 into slot one. All other
//! quantities are compiled as constants.
//!
//! Assignments preserve slot order and prohibit assigning the same physical
//! parameter more than once.

use thiserror::Error;

use crate::input::{
    Parameter, SolveRequest, ThicknessSeedError,
    compile::{coordinates::CoordinateVariable, stack::ThicknessSlotMap},
    parameter::FiniteLayerIndex,
};

/// Mapping from jet derivative slots to caller-facing parameters.
///
/// The parameter at `slots[i]` is seeded as the independent variable in jet
/// slot `i`. Parameters absent from the assignment are compiled as constants.
///
/// Assignments maintain the following invariants:
///
/// - slot indices are contiguous and begin at zero;
/// - each parameter appears at most once;
/// - layer indices are validated against the compiled stack before seeding;
/// - the number of assigned parameters matches the selected jet algebra.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ParameterAssignment {
    slots: Vec<Parameter>,
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ParameterAssignmentError {
    #[error(
        "{variable:?} is assigned more than once: \
         slots {first_slot} and {second_slot}"
    )]
    DuplicateVariable {
        variable: Parameter,
        first_slot: usize,
        second_slot: usize,
    },

    #[error(
        "the parameter assignment uses {assigned_slots} slot(s), \
         but the selected jet algebra provides {available_slots}"
    )]
    IncompatibleShape {
        assigned_slots: usize,
        available_slots: usize,
    },

    #[error(
        "layer thickness variable refers to layer {layer}, \
         but the stack contains only {finite_layer_count} layer(s)"
    )]
    LayerOutOfBounds {
        layer: usize,
        finite_layer_count: usize,
    },
}

impl ParameterAssignment {
    /// Compile every problem parameter as a constant.
    pub(crate) const fn none() -> Self {
        Self { slots: Vec::new() }
    }

    /// Create an assignment from variables in slot order.
    ///
    /// The first variable occupies slot zero, the second occupies slot one,
    /// and so forth.
    pub(crate) fn new(
        slots: impl IntoIterator<Item = Parameter>,
    ) -> Result<Self, ParameterAssignmentError> {
        let slots = slots.into_iter().collect::<Vec<_>>();

        let assignment = Self { slots };
        assignment.validate_unique_parameters()?;

        Ok(assignment)
    }

    /// Append a variable in the next available jet slot.
    pub(crate) fn with(mut self, variable: Parameter) -> Result<Self, ParameterAssignmentError> {
        if let Some(first_slot) = self.slot_for(variable) {
            return Err(ParameterAssignmentError::DuplicateVariable {
                variable,
                first_slot,
                second_slot: self.slots.len(),
            });
        }

        self.slots.push(variable);
        Ok(self)
    }

    pub(crate) fn univariate(parameter: Parameter) -> Self {
        match parameter {
            Parameter::Spectral => Self::spectral(),
            Parameter::InPlane => Self::in_plane(),
            Parameter::LayerThickness(FiniteLayerIndex(layer)) => Self::layer_thickness(layer),
        }
    }

    /// Construct a univariate spectral assignment.
    pub(crate) fn spectral() -> Self {
        Self {
            slots: vec![Parameter::Spectral],
        }
    }

    /// Construct a univariate in-plane assignment.
    pub(crate) fn in_plane() -> Self {
        Self {
            slots: vec![Parameter::InPlane],
        }
    }

    /// Construct a univariate layer-thickness assignment.
    pub(crate) fn layer_thickness(layer: usize) -> Self {
        Self {
            slots: vec![Parameter::LayerThickness(FiniteLayerIndex(layer))],
        }
    }

    /// Construct the usual two-coordinate assignment.
    ///
    /// - slot 0: spectral coordinate;
    /// - slot 1: in-plane coordinate.
    pub(crate) fn spectral_and_in_plane() -> Self {
        Self {
            slots: vec![Parameter::Spectral, Parameter::InPlane],
        }
    }

    /// Return the parameter assigned to `slot`.
    pub(crate) fn parameter(&self, slot: usize) -> Option<Parameter> {
        self.slots.get(slot).copied()
    }

    /// Return the slot occupied by `variable`.
    pub(crate) fn slot_for(&self, variable: Parameter) -> Option<usize> {
        self.slots
            .iter()
            .position(|candidate| *candidate == variable)
    }

    pub(crate) fn slots(&self) -> &[Parameter] {
        &self.slots
    }

    pub(crate) fn parameter_count(&self) -> usize {
        self.slots.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    fn validate_unique_parameters(&self) -> Result<(), ParameterAssignmentError> {
        for (first_slot, &first) in self.slots.iter().enumerate() {
            if let Some(second_slot) = self.slots[first_slot + 1..]
                .iter()
                .position(|&second| second == first)
            {
                return Err(ParameterAssignmentError::DuplicateVariable {
                    variable: first,
                    first_slot,
                    second_slot: first_slot + 1 + second_slot,
                });
            }
        }

        Ok(())
    }

    pub(crate) fn validate(
        &self,
        available_slots: usize,
        finite_layer_count: usize,
    ) -> Result<(), ParameterAssignmentError> {
        if available_slots != self.parameter_count() {
            return Err(ParameterAssignmentError::IncompatibleShape {
                assigned_slots: self.parameter_count(),
                available_slots,
            });
        }

        if let Some(layer) = self
            .slots
            .iter()
            .filter_map(|parameter| match parameter {
                Parameter::LayerThickness(FiniteLayerIndex(layer))
                    if *layer >= finite_layer_count =>
                {
                    Some(*layer)
                }

                _ => None,
            })
            .next()
        {
            return Err(ParameterAssignmentError::LayerOutOfBounds {
                layer,
                finite_layer_count,
            });
        }

        Ok(())
    }

    pub(crate) fn coordinates(&self) -> CoordinateAssignment<'_> {
        CoordinateAssignment::new(self)
    }

    pub(crate) fn thicknesses(&self) -> ThicknessAssignment<'_> {
        ThicknessAssignment::new(self)
    }
}

/// Coordinate-specific view over a parameter assignment.
///
/// This translates canonical coordinate variables into their assigned jet
/// slots without exposing coordinate compilation to layer parameters.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CoordinateAssignment<'a> {
    assignment: &'a ParameterAssignment,
}

impl<'a> CoordinateAssignment<'a> {
    pub(crate) const fn new(assignment: &'a ParameterAssignment) -> Self {
        Self { assignment }
    }

    pub(crate) fn spectral_slot(&self) -> Option<usize> {
        self.assignment.slot_for(Parameter::Spectral)
    }

    pub(crate) fn in_plane_slot(&self) -> Option<usize> {
        self.assignment.slot_for(Parameter::InPlane)
    }

    pub(crate) fn slot_for(&self, variable: CoordinateVariable) -> Option<usize> {
        match variable {
            CoordinateVariable::Spectral => self.spectral_slot(),

            CoordinateVariable::InPlane => self.in_plane_slot(),
        }
    }
}

/// Layer-thickness-specific view over a parameter assignment.
#[derive(Clone, Copy, Debug)]
pub struct ThicknessAssignment<'a> {
    assignment: &'a ParameterAssignment,
}

impl<'a> ThicknessAssignment<'a> {
    pub(crate) const fn new(assignment: &'a ParameterAssignment) -> Self {
        Self { assignment }
    }
}

impl<'a> ThicknessSlotMap for ThicknessAssignment<'a> {
    fn slot_for_layer(&self, layer: usize) -> Option<usize> {
        self.assignment
            .slot_for(Parameter::LayerThickness(FiniteLayerIndex(layer)))
    }
}

#[cfg(test)]
mod tests {
    use crate::input::parameter::FiniteLayerIndex;

    use super::*;

    fn spectral() -> Parameter {
        Parameter::Spectral
    }

    fn in_plane() -> Parameter {
        Parameter::InPlane
    }

    fn thickness(layer: usize) -> Parameter {
        Parameter::LayerThickness(FiniteLayerIndex(layer))
    }

    #[test]
    fn none_creates_empty_assignment() {
        let assignment = ParameterAssignment::none();

        assert!(assignment.is_empty());
        assert_eq!(assignment.parameter_count(), 0);
        assert!(assignment.slots().is_empty());
        assert_eq!(assignment.parameter(0), None);
    }

    #[test]
    fn new_preserves_slot_order() {
        let assignment = ParameterAssignment::new([in_plane(), thickness(2), spectral()]).unwrap();

        assert_eq!(assignment.slots(), &[in_plane(), thickness(2), spectral()],);

        assert_eq!(assignment.parameter(0), Some(in_plane()));
        assert_eq!(assignment.parameter(1), Some(thickness(2)));
        assert_eq!(assignment.parameter(2), Some(spectral()));
        assert_eq!(assignment.parameter(3), None);
    }

    #[test]
    fn new_accepts_empty_iterator() {
        let assignment = ParameterAssignment::new(std::iter::empty()).unwrap();

        assert!(assignment.is_empty());
        assert_eq!(assignment, ParameterAssignment::none());
    }

    #[test]
    fn new_rejects_duplicate_parameter() {
        let error = ParameterAssignment::new([spectral(), thickness(1), spectral()]).unwrap_err();

        assert_eq!(
            error,
            ParameterAssignmentError::DuplicateVariable {
                variable: spectral(),
                first_slot: 0,
                second_slot: 2,
            },
        );
    }

    #[test]
    fn new_reports_first_duplicate_pair() {
        let error = ParameterAssignment::new([thickness(0), spectral(), thickness(0), spectral()])
            .unwrap_err();

        assert_eq!(
            error,
            ParameterAssignmentError::DuplicateVariable {
                variable: thickness(0),
                first_slot: 0,
                second_slot: 2,
            },
        );
    }

    #[test]
    fn with_appends_variable_to_next_slot() {
        let assignment = ParameterAssignment::none()
            .with(spectral())
            .unwrap()
            .with(in_plane())
            .unwrap()
            .with(thickness(3))
            .unwrap();

        assert_eq!(assignment.slots(), &[spectral(), in_plane(), thickness(3)],);
    }

    #[test]
    fn with_rejects_duplicate_parameter() {
        let assignment = ParameterAssignment::new([spectral(), in_plane()]).unwrap();

        let error = assignment.with(spectral()).unwrap_err();

        assert_eq!(
            error,
            ParameterAssignmentError::DuplicateVariable {
                variable: spectral(),
                first_slot: 0,
                second_slot: 2,
            },
        );
    }

    #[test]
    fn univariate_constructs_single_slot_assignment() {
        for parameter in [spectral(), in_plane(), thickness(4)] {
            let assignment = ParameterAssignment::univariate(parameter);

            assert_eq!(assignment.slots(), &[parameter]);
            assert_eq!(assignment.parameter_count(), 1);
            assert_eq!(assignment.parameter(0), Some(parameter));
        }
    }

    #[test]
    fn spectral_constructor_assigns_slot_zero() {
        let assignment = ParameterAssignment::spectral();

        assert_eq!(assignment.slots(), &[spectral()]);
        assert_eq!(assignment.slot_for(spectral()), Some(0));
    }

    #[test]
    fn in_plane_constructor_assigns_slot_zero() {
        let assignment = ParameterAssignment::in_plane();

        assert_eq!(assignment.slots(), &[in_plane()]);
        assert_eq!(assignment.slot_for(in_plane()), Some(0));
    }

    #[test]
    fn layer_thickness_constructor_assigns_slot_zero() {
        let assignment = ParameterAssignment::layer_thickness(7);

        assert_eq!(assignment.slots(), &[thickness(7)]);
        assert_eq!(assignment.slot_for(thickness(7)), Some(0));
    }

    #[test]
    fn spectral_and_in_plane_uses_documented_axis_order() {
        let assignment = ParameterAssignment::spectral_and_in_plane();

        assert_eq!(assignment.slots(), &[spectral(), in_plane()],);

        assert_eq!(assignment.slot_for(spectral()), Some(0));
        assert_eq!(assignment.slot_for(in_plane()), Some(1));
    }

    #[test]
    fn slot_for_returns_matching_slot() {
        let assignment = ParameterAssignment::new([thickness(2), spectral(), in_plane()]).unwrap();

        assert_eq!(assignment.slot_for(thickness(2)), Some(0));
        assert_eq!(assignment.slot_for(spectral()), Some(1));
        assert_eq!(assignment.slot_for(in_plane()), Some(2));
    }

    #[test]
    fn slot_for_distinguishes_layer_indices() {
        let assignment = ParameterAssignment::new([thickness(2), thickness(5)]).unwrap();

        assert_eq!(assignment.slot_for(thickness(2)), Some(0));
        assert_eq!(assignment.slot_for(thickness(5)), Some(1));
        assert_eq!(assignment.slot_for(thickness(3)), None);
    }

    #[test]
    fn slot_for_returns_none_for_unassigned_parameter() {
        let assignment = ParameterAssignment::spectral();

        assert_eq!(assignment.slot_for(in_plane()), None);
        assert_eq!(assignment.slot_for(thickness(0)), None);
    }

    #[test]
    fn coordinates_exposes_coordinate_slots() {
        let assignment = ParameterAssignment::new([thickness(3), in_plane(), spectral()]).unwrap();

        let coordinates = assignment.coordinates();

        assert_eq!(coordinates.spectral_slot(), Some(2));
        assert_eq!(coordinates.in_plane_slot(), Some(1));

        assert_eq!(coordinates.slot_for(CoordinateVariable::Spectral), Some(2),);

        assert_eq!(coordinates.slot_for(CoordinateVariable::InPlane), Some(1),);
    }

    #[test]
    fn coordinates_returns_none_for_unassigned_coordinates() {
        let assignment = ParameterAssignment::layer_thickness(0);

        let coordinates = assignment.coordinates();

        assert_eq!(coordinates.spectral_slot(), None);
        assert_eq!(coordinates.in_plane_slot(), None);

        assert_eq!(coordinates.slot_for(CoordinateVariable::Spectral), None,);

        assert_eq!(coordinates.slot_for(CoordinateVariable::InPlane), None,);
    }

    #[test]
    fn thicknesses_exposes_layer_slots() {
        let assignment =
            ParameterAssignment::new([thickness(4), spectral(), thickness(1)]).unwrap();

        let thicknesses = assignment.thicknesses();

        assert_eq!(thicknesses.slot_for_layer(4), Some(0));
        assert_eq!(thicknesses.slot_for_layer(1), Some(2));
        assert_eq!(thicknesses.slot_for_layer(0), None);
        assert_eq!(thicknesses.slot_for_layer(3), None);
    }

    #[test]
    fn validate_accepts_matching_shape() {
        let assignment = ParameterAssignment::spectral_and_in_plane();

        assert_eq!(assignment.validate(2, 0), Ok(()));
    }

    #[test]
    fn validate_accepts_empty_assignment_with_zero_slots() {
        let assignment = ParameterAssignment::none();

        assert_eq!(assignment.validate(0, 0), Ok(()));
    }

    #[test]
    fn validate_rejects_too_few_available_slots() {
        let assignment = ParameterAssignment::spectral_and_in_plane();

        let error = assignment.validate(1, 0).unwrap_err();

        assert_eq!(
            error,
            ParameterAssignmentError::IncompatibleShape {
                assigned_slots: 2,
                available_slots: 1,
            },
        );
    }

    #[test]
    fn validate_rejects_too_many_available_slots() {
        let assignment = ParameterAssignment::spectral();

        let error = assignment.validate(2, 0).unwrap_err();

        assert_eq!(
            error,
            ParameterAssignmentError::IncompatibleShape {
                assigned_slots: 1,
                available_slots: 2,
            },
        );
    }

    #[test]
    fn validate_accepts_first_layer() {
        let assignment = ParameterAssignment::layer_thickness(0);

        assert_eq!(assignment.validate(1, 1), Ok(()));
    }

    #[test]
    fn validate_accepts_last_layer() {
        let assignment = ParameterAssignment::layer_thickness(2);

        assert_eq!(assignment.validate(1, 3), Ok(()));
    }

    #[test]
    fn validate_rejects_layer_equal_to_layer_count() {
        let assignment = ParameterAssignment::layer_thickness(3);

        let error = assignment.validate(1, 3).unwrap_err();

        assert_eq!(
            error,
            ParameterAssignmentError::LayerOutOfBounds {
                layer: 3,
                finite_layer_count: 3,
            },
        );
    }

    #[test]
    fn validate_rejects_layer_greater_than_layer_count() {
        let assignment = ParameterAssignment::layer_thickness(5);

        let error = assignment.validate(1, 3).unwrap_err();

        assert_eq!(
            error,
            ParameterAssignmentError::LayerOutOfBounds {
                layer: 5,
                finite_layer_count: 3,
            },
        );
    }

    #[test]
    fn validate_rejects_thickness_parameter_for_empty_stack() {
        let assignment = ParameterAssignment::layer_thickness(0);

        let error = assignment.validate(1, 0).unwrap_err();

        assert_eq!(
            error,
            ParameterAssignmentError::LayerOutOfBounds {
                layer: 0,
                finite_layer_count: 0,
            },
        );
    }

    #[test]
    fn validate_checks_every_assigned_layer() {
        let assignment =
            ParameterAssignment::new([thickness(0), spectral(), thickness(4)]).unwrap();

        let error = assignment.validate(3, 2).unwrap_err();

        assert_eq!(
            error,
            ParameterAssignmentError::LayerOutOfBounds {
                layer: 4,
                finite_layer_count: 2,
            },
        );
    }

    #[test]
    fn validate_does_not_apply_layer_bounds_to_coordinates() {
        let assignment = ParameterAssignment::spectral_and_in_plane();

        assert_eq!(assignment.validate(2, 0), Ok(()));
    }
}
