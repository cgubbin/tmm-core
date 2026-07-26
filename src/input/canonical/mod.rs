mod coordinate;
mod stack;

pub(crate) use coordinate::{CanonicalCoordinates, CanonicalPlaneWaveInput, CanonicalSolverInput};
pub(crate) use stack::{CanonicalLayer, CanonicalStack};

use crate::{IncidentSide, Polarisation};

/// Complete canonical input consumed by a backend solve.
///
/// Coordinates and thicknesses use the same jet algebra and sampled shape.
#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalProblem<M, J> {
    solver_input: CanonicalSolverInput<J>,
    stack: CanonicalStack<M, J>,
}

impl<M, J> CanonicalProblem<M, J> {
    pub(crate) fn input(&self) -> &CanonicalSolverInput<J> {
        &self.solver_input
    }

    pub(crate) fn new(
        coordinates: CanonicalCoordinates<J>,
        polarisation: Polarisation,
        stack: CanonicalStack<M, J>,
    ) -> Self {
        Self {
            solver_input: CanonicalSolverInput::new(coordinates, polarisation),
            stack,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CanonicalPlaneWaveProblem<M, J> {
    problem: CanonicalProblem<M, J>,
    incident_side: IncidentSide,
}

impl<M, J> CanonicalPlaneWaveProblem<M, J> {
    pub(crate) fn problem(&self) -> &CanonicalProblem<M, J> {
        &self.problem
    }

    pub(crate) fn new(
        coordinates: CanonicalCoordinates<J>,
        polarisation: Polarisation,
        incident_side: IncidentSide,
        stack: CanonicalStack<M, J>,
    ) -> Self {
        Self {
            problem: CanonicalProblem::new(coordinates, polarisation, stack),
            incident_side,
        }
    }
}

// impl<M, J> CanonicalSolveInput<M, J> {
//     pub(crate) fn new(coordinates: CanonicalCoordinates<J>, stack: CanonicalStack<M, J>) -> Self {
//         Self { coordinates, stack }
//     }

//     pub fn coordinates(&self) -> &CanonicalCoordinates<J> {
//         &self.coordinates
//     }

//     pub fn coordinates_mut(&mut self) -> &mut CanonicalCoordinates<J> {
//         &mut self.coordinates
//     }

//     pub fn stack(&self) -> &CanonicalStack<M, J> {
//         &self.stack
//     }

//     pub fn stack_mut(&mut self) -> &mut CanonicalStack<M, J> {
//         &mut self.stack
//     }

//     pub fn into_parts(self) -> (CanonicalCoordinates<J>, CanonicalStack<M, J>) {
//         (self.coordinates, self.stack)
//     }
// }
