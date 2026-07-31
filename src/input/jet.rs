use crate::{
    algebra::{Jet0, Jet1, Jet2, JetBivariate1, JetBivariate2},
    crystallise::{
        FirstBivariate, FirstDirectional, SecondBivariate, SecondDirectional, ValueOnly,
    },
    input::{Parameter, ParameterAssignment, ParameterAssignmentError},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ValueAssignment;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectionalAssignment {
    parameter: Parameter,
}

impl DirectionalAssignment {
    pub(crate) fn try_from_assignment(
        assignment: ParameterAssignment,
    ) -> Result<Self, ParameterAssignmentError> {
        let [parameter]: [Parameter; 1] = assignment.slots().try_into().map_err(|_| {
            ParameterAssignmentError::IncompatibleShape {
                available_slots: 1,
                assigned_slots: assignment.parameter_count(),
            }
        })?;

        Ok(Self { parameter })
    }

    pub fn parameter(&self) -> &Parameter {
        &self.parameter
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BivariateAssignment {
    first: Parameter,
    second: Parameter,
}

impl BivariateAssignment {
    pub(crate) fn try_from_assignment(
        assignment: ParameterAssignment,
    ) -> Result<Self, ParameterAssignmentError> {
        let [first, second]: [Parameter; 2] = assignment.slots().try_into().map_err(|_| {
            ParameterAssignmentError::IncompatibleShape {
                available_slots: 2,
                assigned_slots: assignment.parameter_count(),
            }
        })?;

        Ok(Self { first, second })
    }

    pub fn parameters(&self) -> (&Parameter, &Parameter) {
        (&self.first, &self.second)
    }
}

pub trait JetEvaluation {
    type Policy: Default;
    type Assignment;

    fn refine_assignment(
        assignment: ParameterAssignment,
    ) -> Result<Self::Assignment, ParameterAssignmentError>;
}

impl<I, P> JetEvaluation for Jet0<I, P> {
    type Policy = ValueOnly;
    type Assignment = ValueAssignment;

    fn refine_assignment(
        assignment: ParameterAssignment,
    ) -> Result<Self::Assignment, ParameterAssignmentError> {
        if assignment.is_empty() {
            Ok(ValueAssignment)
        } else {
            Err(ParameterAssignmentError::IncompatibleShape {
                available_slots: 0,
                assigned_slots: assignment.parameter_count(),
            })
        }
    }
}

impl<I, P> JetEvaluation for Jet1<I, P> {
    type Policy = FirstDirectional;
    type Assignment = DirectionalAssignment;

    fn refine_assignment(
        assignment: ParameterAssignment,
    ) -> Result<Self::Assignment, ParameterAssignmentError> {
        DirectionalAssignment::try_from_assignment(assignment)
    }
}

impl<I, P> JetEvaluation for Jet2<I, P> {
    type Policy = SecondDirectional;
    type Assignment = DirectionalAssignment;

    fn refine_assignment(
        assignment: ParameterAssignment,
    ) -> Result<Self::Assignment, ParameterAssignmentError> {
        DirectionalAssignment::try_from_assignment(assignment)
    }
}

impl<I, P> JetEvaluation for JetBivariate1<I, P> {
    type Policy = FirstBivariate;
    type Assignment = BivariateAssignment;

    fn refine_assignment(
        assignment: ParameterAssignment,
    ) -> Result<Self::Assignment, ParameterAssignmentError> {
        BivariateAssignment::try_from_assignment(assignment)
    }
}

impl<I, P> JetEvaluation for JetBivariate2<I, P> {
    type Policy = SecondBivariate;
    type Assignment = BivariateAssignment;

    fn refine_assignment(
        assignment: ParameterAssignment,
    ) -> Result<Self::Assignment, ParameterAssignmentError> {
        BivariateAssignment::try_from_assignment(assignment)
    }
}
