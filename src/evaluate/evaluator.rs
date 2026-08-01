use std::marker::PhantomData;

use crate::{
    ComplexPlane, DifferentiableMaterial, EvaluateMeromorphicMaterial, Material, Polarisation,
    ValidationConfig,
    algebra::{
        ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, ComplexJet,
        RealParameter, ScalarAlgebra,
    },
    backend::{Backend, PlaneWaveSolution},
    domain::RealAxis,
    input::{
        CompilationContext, CompileJet, CompilePlaneWaveError, CoordinateInput, PlaneWaveInput,
        compile_complex, compile_real,
    },
    material::{ConstitutiveEvaluator, ConstitutiveLift},
    parameter::{DerivativeMapping, Parameter},
    scalar::ComplexScalar,
    stack::Stack,
};

use super::{PlaneWaveEvaluationError, PlaneWaveState, RequestError};

/// Compiles and solves plane-wave excitation problems using a statically
/// selected backend.
///
/// The evaluator does not compute or crystallise observables. It returns a
/// retained [`PlaneWaveState`] from which quantities may be requested later.
#[derive(Clone, Debug)]
pub struct PlaneWaveEvaluator<B> {
    backend: B,
}

impl<B> PlaneWaveEvaluator<B> {
    pub const fn new(backend: B) -> Self {
        Self { backend }
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn into_backend(self) -> B {
        self.backend
    }
}

impl<B> From<B> for PlaneWaveEvaluator<B> {
    fn from(backend: B) -> Self {
        Self::new(backend)
    }
}

use nalgebra::ComplexField;
use ndarray::Dimension;
use num_traits::{Float, FloatConst, FromPrimitive};
use std::fmt::Debug;

impl<B> PlaneWaveEvaluator<B> {
    fn solve_real_coordinate_space<J, M, E>(
        &self,
        input: CoordinateInput<<J::Scalar as ComplexField>::RealField, J::Dimension>,
        stack: &Stack<M, <J::Scalar as ComplexField>::RealField>,
        polarisation: Polarisation,
        mapping: &DerivativeMapping,
    ) -> Result<
        PlaneWaveState<M, J, <J::Scalar as ComplexField>::RealField, PlaneWaveSolution<B::Entries>>,
        PlaneWaveEvaluationError<
            CompilePlaneWaveError<J::Scalar>,
            <B as Backend<J, RealAxis>>::Error,
        >,
    >
    where
        J: CompileJet<M, RealAxis>,
        J::Scalar: ComplexScalar,
        <J::Scalar as ComplexField>::RealField: Float + FloatConst + FromPrimitive + Debug + Copy,
        J::Dimension: Dimension,
        M: Clone,
        RealAxis: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
        B: Backend<J, RealAxis>,
    {
        let (canonical_problem, context) =
            compile_real::<M, J>(input, stack, &ValidationConfig::permissive(), mapping)
                .map_err(PlaneWaveEvaluationError::compile)?;

        let workspace = self
            .backend
            .solve(&canonical_problem, polarisation)
            .map_err(|err| PlaneWaveEvaluationError::Backend { source: err })?;

        Ok(PlaneWaveState::new(canonical_problem, workspace, context))
    }

    fn solve_complex_coordinate_space<J, M, E>(
        &self,
        input: CoordinateInput<J::Scalar, J::Dimension>,
        stack: &Stack<M, <J::Scalar as ComplexField>::RealField>,
        polarisation: Polarisation,
        mapping: &DerivativeMapping,
    ) -> Result<
        PlaneWaveState<M, J, J::Scalar, PlaneWaveSolution<B::Entries>>,
        PlaneWaveEvaluationError<
            CompilePlaneWaveError<J::Scalar>,
            <B as Backend<J, ComplexPlane>>::Error,
        >,
    >
    where
        J: CompileJet<M, ComplexPlane>,
        J::Scalar: ComplexScalar,
        <J::Scalar as ComplexField>::RealField: Float + FloatConst + FromPrimitive + Debug + Copy,
        J::Dimension: Dimension,
        M: Clone,
        ComplexPlane: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
        B: Backend<J, ComplexPlane>,
    {
        let (canonical_problem, context) =
            compile_complex::<M, J>(input, stack, &ValidationConfig::permissive(), mapping)
                .map_err(PlaneWaveEvaluationError::compile)?;

        let workspace = self
            .backend
            .solve(&canonical_problem, polarisation)
            .map_err(|err| PlaneWaveEvaluationError::Backend { source: err })?;

        Ok(PlaneWaveState::new(canonical_problem, workspace, context))
    }
}
