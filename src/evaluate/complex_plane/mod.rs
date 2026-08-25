//! Canonical complex-plane evaluation.
//!
//! Complex-plane evaluation operates directly on canonical vacuum and
//! in-plane angular wavenumbers. Unlike real-axis evaluation, it performs no
//! caller-coordinate compilation and attaches no physical meaning to jet
//! derivative slots.
//!
//! The evaluator owns a precompiled canonical stack. Callers provide:
//!
//! - canonical complex coordinates;
//! - explicitly branch-selected exterior longitudinal wavevectors;
//! - the polarization;
//! - the jet algebra used to propagate any desired analytic derivatives.
//!
//! This path is intended for complex continuation, outgoing-mode
//! determinants, argument-principle searches, mode refinement, modal
//! reconstruction, and continuation calculations.

pub(super) mod mode;
mod state;

use std::fmt::Debug;

use nalgebra::ComplexField;
use ndarray::{Dimension, Ix0};
use num_traits::{Float, FromPrimitive};

use crate::{
    CanonicalCoordinates, ComplexPlane, ComplexScalar, ExteriorWavevectors, Polarisation, Stack,
    algebra::{Jet, ScalarAlgebra},
    backend::{Backend, PlaneWaveSolution},
    input::{
        CanonicalStack, StackCompileError, StackThicknessJet, ValidationConfig,
        compile_canonical_constant_stack,
    },
    material::{ConstitutiveEvaluator, ConstitutiveLift},
    observable::ProjectPlaneWaveModeDeterminant,
};

pub use mode::{ComplexPlaneMode, QnmCreationError};
pub use state::ComplexPlaneState;

/// Evaluates canonical complex-plane problems using a precompiled stack.
///
/// `J` determines the analytic value/derivative algebra propagated through
/// coordinates, exterior wavevectors, constitutive models, and layer
/// thicknesses.
///
/// [`compile`](Self::compile) constructs a stack with constant geometric jets.
/// [`from_canonical_stack`](Self::from_canonical_stack) accepts an already
/// compiled stack and therefore permits advanced callers to seed geometry
/// derivatives explicitly.
///
/// The evaluator does not perform coordinate conversion, derivative mapping,
/// or differential-response crystallisation.
#[derive(Clone, Debug)]
pub struct ComplexPlaneEvaluator<J, M, B> {
    backend: B,
    stack: CanonicalStack<M, J>,
}

impl<J, M, B> ComplexPlaneEvaluator<J, M, B> {
    pub fn compile(
        stack: &Stack<M, <J::Scalar as ComplexField>::RealField>,
        backend: B,
    ) -> Result<Self, StackCompileError<<J::Scalar as ComplexField>::RealField>>
    where
        J: Jet<Dimension = Ix0> + StackThicknessJet,
        J::Scalar: ComplexField + Copy,
        <J::Scalar as ComplexField>::RealField: Float + FromPrimitive + Copy + Debug,
        J::Dimension: Dimension + Clone,
        M: Clone,
    {
        let stack = compile_canonical_constant_stack::<M, _>(
            stack,
            Ix0(),
            &ValidationConfig::permissive(),
        )?;

        Ok(Self { stack, backend })
    }

    pub fn from_canonical_stack(stack: CanonicalStack<M, J>, backend: B) -> Self {
        Self { stack, backend }
    }

    // pub(crate) fn backend(&self) -> &B {
    //     &self.backend
    // }

    #[cfg(test)]
    pub(crate) fn stack(&self) -> &CanonicalStack<M, J> {
        &self.stack
    }

    pub fn into_parts(self) -> (CanonicalStack<M, J>, B) {
        (self.stack, self.backend)
    }
}

impl<J, M, B> ComplexPlaneEvaluator<J, M, B> {
    /// Solve a canonical complex-plane problem without retaining internal
    /// finite-layer state.
    pub fn solve(
        &self,
        coordinates: &CanonicalCoordinates<J>,
        exterior: &ExteriorWavevectors<J>,
        polarisation: Polarisation,
    ) -> Result<
        PlaneWaveSolution<<B as Backend<J, ComplexPlane>>::Entries>,
        <B as Backend<J, ComplexPlane>>::Error,
    >
    where
        J: ScalarAlgebra + ConstitutiveLift<ComplexPlane, M>,
        J::Scalar: ComplexScalar,
        J::Dimension: Dimension,
        ComplexPlane: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
        B: Backend<J, ComplexPlane>,
    {
        self.backend
            .solve(coordinates, &self.stack, exterior, polarisation)
    }

    pub fn determinant(
        &self,
        coordinates: &CanonicalCoordinates<J>,
        exterior: &ExteriorWavevectors<J>,
        polarisation: Polarisation,
    ) -> Result<
        RawModeDeterminant<<B as Backend<J, ComplexPlane>>::Entries>,
        <B as Backend<J, ComplexPlane>>::Error,
    >
    where
        J: ScalarAlgebra + ConstitutiveLift<ComplexPlane, M>,
        J::Scalar: ComplexScalar,
        J::Dimension: Dimension,
        M: Clone,
        ComplexPlane: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
        B: Backend<J, ComplexPlane>,
        <B as Backend<J, ComplexPlane>>::Entries: ProjectPlaneWaveModeDeterminant,
    {
        Ok(self
            .solve(coordinates, exterior, polarisation)?
            .determinant())
    }

    pub fn retain(
        &self,
        coordinates: CanonicalCoordinates<J>,
        exterior: ExteriorWavevectors<J>,
        polarisation: Polarisation,
    ) -> Result<ComplexPlaneState<'_, J, M, B::Workspace>, B::Error>
    where
        J: ScalarAlgebra + ConstitutiveLift<ComplexPlane, M>,
        J::Scalar: ComplexScalar,
        J::Dimension: Dimension,
        ComplexPlane: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
        B: Backend<J, ComplexPlane>,
    {
        let workspace = self
            .backend
            .retain(&coordinates, &self.stack, &exterior, polarisation)?;

        Ok(ComplexPlaneState::new(
            coordinates,
            exterior,
            &self.stack,
            workspace,
            polarisation,
        ))
    }
}

pub(crate) type RawModeDeterminant<E> = <E as ProjectPlaneWaveModeDeterminant>::Determinant;
