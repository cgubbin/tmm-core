use super::Parameter;

use thiserror::Error;

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
#[doc(hidden)]
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct DerivativeMapping {
    slots: Vec<Parameter>,
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum DerivativeMappingError {
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
         but the selected jet algebra provides {derivative_dimension}"
    )]
    IncompatibleShape {
        assigned_slots: usize,
        derivative_dimension: usize,
    },
}

impl DerivativeMapping {
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
    ) -> Result<Self, DerivativeMappingError> {
        let slots = slots.into_iter().collect::<Vec<_>>();

        let assignment = Self { slots };
        assignment.validate_unique_parameters()?;

        Ok(assignment)
    }

    /// Append a variable in the next available jet slot.
    pub(crate) fn with(mut self, variable: Parameter) -> Result<Self, DerivativeMappingError> {
        if let Some(first_slot) = self.slot_for(variable) {
            return Err(DerivativeMappingError::DuplicateVariable {
                variable,
                first_slot,
                second_slot: self.slots.len(),
            });
        }

        self.slots.push(variable);
        Ok(self)
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

    fn validate_unique_parameters(&self) -> Result<(), DerivativeMappingError> {
        for (first_slot, &first) in self.slots.iter().enumerate() {
            if let Some(second_slot) = self.slots[first_slot + 1..]
                .iter()
                .position(|&second| second == first)
            {
                return Err(DerivativeMappingError::DuplicateVariable {
                    variable: first,
                    first_slot,
                    second_slot: first_slot + 1 + second_slot,
                });
            }
        }

        Ok(())
    }
}
