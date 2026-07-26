use thiserror::Error;

use crate::input::compile::coordinates::CoordinateVariable;

/// A caller-facing independent variable that may be seeded into a jet slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProblemVariable {
    Spectral,
    InPlane,
    LayerThickness { layer: usize },
}

/// Assignment of caller-facing problem variables to jet slots.
///
/// Entry `slots[i]` identifies the physical variable represented by slot `i`.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ParameterAssignment {
    slots: Vec<ProblemVariable>,
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ParameterAssignmentError {
    #[error(
        "{variable:?} is assigned more than once: \
         slots {first_slot} and {second_slot}"
    )]
    DuplicateVariable {
        variable: ProblemVariable,
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
        "the parameter assignment uses {assigned_slots} slot(s), \
         but the selected jet algebra provides {available_slots}"
    )]
    TooManyVariables {
        assigned_slots: usize,
        available_slots: usize,
    },

    #[error(
        "layer thickness variable refers to layer {layer}, \
         but the stack contains only {layer_count} layer(s)"
    )]
    LayerOutOfBounds { layer: usize, layer_count: usize },
}

impl ParameterAssignment {
    /// Compile every problem parameter as a constant.
    pub const fn none() -> Self {
        Self { slots: Vec::new() }
    }

    /// Create an assignment from variables in slot order.
    ///
    /// The first variable occupies slot zero, the second occupies slot one,
    /// and so forth.
    pub fn new(
        slots: impl IntoIterator<Item = ProblemVariable>,
    ) -> Result<Self, ParameterAssignmentError> {
        let slots = slots.into_iter().collect::<Vec<_>>();

        let assignment = Self { slots };
        assignment.validate_unique_variables()?;

        Ok(assignment)
    }

    /// Append a variable in the next available jet slot.
    pub fn with(mut self, variable: ProblemVariable) -> Result<Self, ParameterAssignmentError> {
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

    /// Construct a univariate spectral assignment.
    pub fn spectral() -> Self {
        Self {
            slots: vec![ProblemVariable::Spectral],
        }
    }

    /// Construct a univariate in-plane assignment.
    pub fn in_plane() -> Self {
        Self {
            slots: vec![ProblemVariable::InPlane],
        }
    }

    /// Construct a univariate layer-thickness assignment.
    pub fn layer_thickness(layer: usize) -> Self {
        Self {
            slots: vec![ProblemVariable::LayerThickness { layer }],
        }
    }

    /// Construct the usual two-coordinate assignment.
    ///
    /// - slot 0: spectral coordinate;
    /// - slot 1: in-plane coordinate.
    pub fn spectral_and_in_plane() -> Self {
        Self {
            slots: vec![ProblemVariable::Spectral, ProblemVariable::InPlane],
        }
    }

    /// Return the variable assigned to `slot`.
    pub fn variable(&self, slot: usize) -> Option<ProblemVariable> {
        self.slots.get(slot).copied()
    }

    /// Return the slot occupied by `variable`.
    pub fn slot_for(&self, variable: ProblemVariable) -> Option<usize> {
        self.slots
            .iter()
            .position(|candidate| *candidate == variable)
    }

    pub fn slots(&self) -> &[ProblemVariable] {
        &self.slots
    }

    pub fn variable_count(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    fn validate_unique_variables(&self) -> Result<(), ParameterAssignmentError> {
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
        layer_count: usize,
    ) -> Result<(), ParameterAssignmentError> {
        if available_slots != self.variable_count() {
            return Err(ParameterAssignmentError::IncompatibleShape {
                assigned_slots: self.variable_count(),
                available_slots,
            });
        }

        self.validate_unique_variables()?;

        for layer in self.slots.iter().filter_map(|variable| match variable {
            ProblemVariable::LayerThickness { layer } if *layer > layer_count => Some(*layer),
            _ => None,
        }) {
            return Err(ParameterAssignmentError::LayerOutOfBounds { layer, layer_count });
        }
        Ok(())
    }

    pub fn coordinates(&self) -> CoordinateAssignment<'_> {
        CoordinateAssignment::new(self)
    }

    pub fn thicknesses(&self) -> ThicknessAssignment<'_> {
        ThicknessAssignment::new(self)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CoordinateAssignment<'a> {
    assignment: &'a ParameterAssignment,
}

impl<'a> CoordinateAssignment<'a> {
    pub(crate) const fn new(assignment: &'a ParameterAssignment) -> Self {
        Self { assignment }
    }

    pub fn spectral_slot(&self) -> Option<usize> {
        self.assignment.slot_for(ProblemVariable::Spectral)
    }

    pub fn in_plane_slot(&self) -> Option<usize> {
        self.assignment.slot_for(ProblemVariable::InPlane)
    }

    pub fn slot_for(&self, variable: CoordinateVariable) -> Option<usize> {
        match variable {
            CoordinateVariable::Spectral => self.spectral_slot(),

            CoordinateVariable::InPlane => self.in_plane_slot(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ThicknessAssignment<'a> {
    assignment: &'a ParameterAssignment,
}

impl<'a> ThicknessAssignment<'a> {
    pub(crate) const fn new(assignment: &'a ParameterAssignment) -> Self {
        Self { assignment }
    }

    pub fn slot_for_layer(&self, layer: usize) -> Option<usize> {
        self.assignment
            .slot_for(ProblemVariable::LayerThickness { layer })
    }
}
