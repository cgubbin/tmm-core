use std::fmt::Debug;

use nalgebra::ComplexField;
use ndarray::{Dimension, Ix0, arr0};
use num_traits::{Float, FromPrimitive, One};
use thiserror::Error;

use crate::{
    CanonicalCoordinates, ComplexPlane, ComplexScalar, ExteriorWavevectors, Parameter,
    Polarisation, Stack,
    algebra::{Jet, ScalarAlgebra},
    backend::Backend,
    evaluate::{PlaneWaveEvaluationError, SolveRequestError},
    input::{
        CanonicalProblem, CanonicalStack, MappingError, StackCompileError, StackThicknessJet,
        ValidationConfig, compile_canonical_constant_stack,
    },
    material::{ConstitutiveEvaluator, ConstitutiveLift},
    observable::ProjectPlaneWaveModeDeterminant,
};

#[derive(Debug, Error)]
pub enum CompileModalError<R> {
    #[error(transparent)]
    Request(#[from] SolveRequestError),

    #[error(transparent)]
    Mapping(#[from] MappingError),

    #[error(transparent)]
    Stack(#[from] StackCompileError<R>),
}

/// Compiles and solves modal problems using a statically
/// selected backend.
///
/// The evaluator does not compute or crystallise observables. It returns a
/// retained [`ModalState`] from which quantities may be requested later.
///
/// This evaluator is designed for advanced consumers in the lamina ecosystem. Contrasting
/// [`PlaneWaveEvaluator`] the stack is pre-compiled. The evaluator is unsuitable for parameter
/// optimisation, and is designed for analysis of systems with fixed geometric and material
/// properties.
///
/// Unlike a [`PlaneWaveEvaluator`], a modal evaluator is crystallised on instantiation for a given
/// derivative order through the jet composition. A new instance must be created to probe a
/// different derivative structure.
///
/// Modal evaluators should be probed in the canonical backend coordinates. No coordinate
/// compilation occurs in the evaluation path
#[derive(Clone, Debug)]
pub struct ModalEvaluator<J, M, B> {
    backend: B,
    stack: CanonicalStack<M, J>,
}

pub enum ModalAnalysisParameter {
    InPlane,
    Spectral,
}

impl From<ModalAnalysisParameter> for Parameter {
    fn from(value: ModalAnalysisParameter) -> Self {
        match value {
            ModalAnalysisParameter::Spectral => Parameter::Spectral,
            ModalAnalysisParameter::InPlane => Parameter::InPlane,
        }
    }
}

impl<J, M, B> ModalEvaluator<J, M, B> {
    fn new(
        stack: &Stack<M, <J::Scalar as ComplexField>::RealField>,
        backend: B,
    ) -> Result<Self, CompileModalError<<J::Scalar as ComplexField>::RealField>>
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

    fn backend(&self) -> &B {
        &self.backend
    }

    fn stack(&self) -> &CanonicalStack<M, J> {
        &self.stack
    }
}

impl<J, M, B> ModalEvaluator<J, M, B> {
    pub fn determinant(
        &self,
        coordinates: CanonicalCoordinates<J>,
        exterior: ExteriorWavevectors<J>,
        polarisation: Polarisation,
    ) -> Result<
        RawModeDeterminant<B, J>,
        PlaneWaveEvaluationError<
            StackCompileError<J::Scalar>,
            <B as Backend<J, ComplexPlane>>::Error,
        >,
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
        let solution = self
            .backend()
            .solve(&coordinates, self.stack(), &exterior, polarisation)
            .map_err(|err| PlaneWaveEvaluationError::Backend { source: err })?;

        Ok(solution.determinant())
    }
}

pub(crate) type QueryEntries<B, J> = <B as Backend<J, ComplexPlane>>::Entries;

pub(crate) type RawModeDeterminant<B, J> =
    <QueryEntries<B, J> as ProjectPlaneWaveModeDeterminant>::Determinant;
