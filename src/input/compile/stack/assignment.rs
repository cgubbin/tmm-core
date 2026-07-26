use thiserror::Error;

/// Assigns physical layer thicknesses to slots in the global derivative space.
pub trait ThicknessAssignment {
    /// Return the derivative slot occupied by `layer_index`.
    ///
    /// `None` means the layer thickness is constant.
    fn slot_for_layer(&self, layer_index: usize) -> Option<usize>;

    /// Validate the assignment against the stack being compiled.
    fn validate(
        &self,
        layer_count: usize,
        available_slots: usize,
    ) -> Result<(), ThicknessAssignmentError>;
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum ThicknessAssignmentError {
    #[error(
        "thickness derivative requested for layer {layer}, \
         but the stack contains {layer_count} layer(s)"
    )]
    LayerOutOfBounds { layer: usize, layer_count: usize },

    #[error(
        "thickness derivative for layer {layer} uses slot {slot}, \
         but the selected jet algebra provides {available_slots} slot(s)"
    )]
    SlotOutOfBounds {
        layer: usize,
        slot: usize,
        available_slots: usize,
    },

    #[error(
        "multiple layer thicknesses are assigned to derivative slot {slot}: \
         layers {first_layer} and {second_layer}"
    )]
    DuplicateSlot {
        slot: usize,
        first_layer: usize,
        second_layer: usize,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ConstantThicknesses;

impl ThicknessAssignment for ConstantThicknesses {
    fn slot_for_layer(&self, _layer_index: usize) -> Option<usize> {
        None
    }

    fn validate(
        &self,
        _layer_count: usize,
        _available_slots: usize,
    ) -> Result<(), ThicknessAssignmentError> {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerThickness {
    layer: usize,
    slot: usize,
}

impl LayerThickness {
    pub const fn new(layer: usize, slot: usize) -> Self {
        Self { layer, slot }
    }

    pub const fn univariate(layer: usize) -> Self {
        Self::new(layer, 0)
    }

    pub const fn layer(&self) -> usize {
        self.layer
    }

    pub const fn slot(&self) -> usize {
        self.slot
    }
}

impl ThicknessAssignment for LayerThickness {
    fn slot_for_layer(&self, layer_index: usize) -> Option<usize> {
        (layer_index == self.layer).then_some(self.slot)
    }

    fn validate(
        &self,
        layer_count: usize,
        available_slots: usize,
    ) -> Result<(), ThicknessAssignmentError> {
        if self.layer >= layer_count {
            return Err(ThicknessAssignmentError::LayerOutOfBounds {
                layer: self.layer,
                layer_count,
            });
        }

        if self.slot >= available_slots {
            return Err(ThicknessAssignmentError::SlotOutOfBounds {
                layer: self.layer,
                slot: self.slot,
                available_slots,
            });
        }

        Ok(())
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ThicknessVariables {
    assignments: Vec<(usize, usize)>,
}

impl ThicknessVariables {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_layer(mut self, layer: usize, slot: usize) -> Self {
        self.assignments.push((layer, slot));
        self
    }

    pub fn assignments(&self) -> &[(usize, usize)] {
        &self.assignments
    }
}

impl ThicknessAssignment for ThicknessVariables {
    fn slot_for_layer(&self, layer_index: usize) -> Option<usize> {
        self.assignments
            .iter()
            .find_map(|&(layer, slot)| (layer == layer_index).then_some(slot))
    }

    fn validate(
        &self,
        layer_count: usize,
        available_slots: usize,
    ) -> Result<(), ThicknessAssignmentError> {
        for &(layer, slot) in &self.assignments {
            if layer >= layer_count {
                return Err(ThicknessAssignmentError::LayerOutOfBounds { layer, layer_count });
            }

            if slot >= available_slots {
                return Err(ThicknessAssignmentError::SlotOutOfBounds {
                    layer,
                    slot,
                    available_slots,
                });
            }
        }

        for (offset, &(first_layer, first_slot)) in self.assignments.iter().enumerate() {
            for &(second_layer, second_slot) in &self.assignments[offset + 1..] {
                if first_slot == second_slot {
                    return Err(ThicknessAssignmentError::DuplicateSlot {
                        slot: first_slot,
                        first_layer,
                        second_layer,
                    });
                }
            }
        }

        Ok(())
    }
}
