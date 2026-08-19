mod excitation;
mod pair;
mod query;
mod result;
mod state;

pub use excitation::RealAxisExcitation;
pub use pair::{RealAxisExcitationPair, RealAxisPairError};
pub(crate) use query::RealAxisExternalQueries;
pub use result::RealAxisResult;
use state::RealAxisState;

use nalgebra::ComplexField;
use ndarray::Dimension;
use num_complex::Complex;
use num_traits::{Float, FloatConst, FromPrimitive};
use std::fmt::Debug;

use crate::{
    Polarisation,
    algebra::{
        ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, Jet, RealParameter,
    },
    backend::{Backend, PlaneWaveSolution, evaluate_exterior_wavevectors},
    domain::RealAxis,
    input::{CompileJet, CompilePlaneWaveError, CoordinateInput, ValidationConfig, compile_real},
    material::ConstitutiveEvaluator,
    parameter::{DerivativeMapping, Parameter},
    scalar::ComplexScalar,
    stack::Stack,
};

use super::{RealAxisEvaluationError, SolveRequestError};

/// Compiles and solves plane-wave excitation problems using a statically
/// selected backend.
///
/// The evaluator does not compute or crystallise observables. It returns a
/// retained [`RealAxisState`] from which quantities may be requested later.
#[derive(Clone, Debug)]
pub struct RealAxisEvaluator<B> {
    backend: B,
}

impl<B> RealAxisEvaluator<B> {
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

impl<B> From<B> for RealAxisEvaluator<B> {
    fn from(backend: B) -> Self {
        Self::new(backend)
    }
}

impl<B> RealAxisEvaluator<B> {
    /// Evaluate external plane-wave quantities over real caller-facing
    /// coordinates.
    ///
    /// This value-only path performs no derivative seeding. The returned result
    /// can be projected into amplitudes and power coefficients for either
    /// incidence side permitted by the compiled projection constraint.
    pub fn evaluate<M, R, D>(
        &self,
        input: CoordinateInput<R, D>,
        stack: &Stack<M, R>,
        polarisation: Polarisation,
    ) -> SolvedRealResult<B, ValueJet<Complex<R>, D>>
    where
        Complex<R>: ComplexScalar<RealField = R>,
        R: Float + FloatConst + FromPrimitive + Debug + Copy + ComplexField,
        D: Dimension,
        M: Clone,
        ValueJet<Complex<R>, D>: Jet<Scalar = Complex<R>, Dimension = D> + CompileJet<M, RealAxis>,
        RealAxis: ConstitutiveEvaluator<Complex<R>, D, M>,
        B: Backend<ValueJet<Complex<R>, D>, RealAxis>,
    {
        let mapping = DerivativeMapping::none();

        self._solve::<ValueJet<Complex<R>, D>, M>(input, stack, polarisation, &mapping)
    }

    pub fn retain<M, R, D>(
        &self,
        input: CoordinateInput<R, D>,
        stack: &Stack<M, R>,
        polarisation: Polarisation,
    ) -> RetainedRealResult<B, ValueJet<Complex<R>, D>, M>
    where
        Complex<R>: ComplexScalar<RealField = R>,
        R: Float + FloatConst + FromPrimitive + Debug + Copy + ComplexField,
        D: Dimension,
        M: Clone,
        ValueJet<Complex<R>, D>: Jet<Scalar = Complex<R>, Dimension = D> + CompileJet<M, RealAxis>,
        RealAxis: ConstitutiveEvaluator<Complex<R>, D, M>,
        B: Backend<ValueJet<Complex<R>, D>, RealAxis>,
    {
        let mapping = DerivativeMapping::none();

        self._retain::<ValueJet<Complex<R>, D>, M>(input, stack, polarisation, &mapping)
    }
}

impl<B> RealAxisEvaluator<B> {
    /// Evaluate values and first derivatives with respect to `parameter`.
    pub fn evaluate_first<M, R, D>(
        &self,
        input: CoordinateInput<R, D>,
        stack: &Stack<M, R>,
        polarisation: Polarisation,
        parameter: Parameter,
    ) -> SolvedRealResult<B, FirstJet<Complex<R>, D>>
    where
        Complex<R>: ComplexScalar<RealField = R>,
        R: Float + FloatConst + FromPrimitive + Debug + Copy + ComplexField,
        D: Dimension,
        M: Clone,
        FirstJet<Complex<R>, D>: Jet<Scalar = Complex<R>, Dimension = D> + CompileJet<M, RealAxis>,
        RealAxis: ConstitutiveEvaluator<Complex<R>, D, M>,
        B: Backend<FirstJet<Complex<R>, D>, RealAxis>,
    {
        let mapping =
            DerivativeMapping::new([parameter]).map_err(SolveRequestError::DerivativeMapping)?;

        self._solve::<FirstJet<Complex<R>, D>, M>(input, stack, polarisation, &mapping)
    }

    /// Retain backend data while computing values and first derivatives with
    /// respect to `parameter`.
    pub fn retain_first<M, R, D>(
        &self,
        input: CoordinateInput<R, D>,
        stack: &Stack<M, R>,
        polarisation: Polarisation,
        parameter: Parameter,
    ) -> RetainedRealResult<B, FirstJet<Complex<R>, D>, M>
    where
        Complex<R>: ComplexScalar<RealField = R>,
        R: Float + FloatConst + FromPrimitive + Debug + Copy + ComplexField,
        D: Dimension,
        M: Clone,
        FirstJet<Complex<R>, D>: Jet<Scalar = Complex<R>, Dimension = D> + CompileJet<M, RealAxis>,
        RealAxis: ConstitutiveEvaluator<Complex<R>, D, M>,
        B: Backend<FirstJet<Complex<R>, D>, RealAxis>,
    {
        let mapping =
            DerivativeMapping::new([parameter]).map_err(SolveRequestError::DerivativeMapping)?;

        self._retain::<FirstJet<Complex<R>, D>, M>(input, stack, polarisation, &mapping)
    }
}

impl<B> RealAxisEvaluator<B> {
    /// Evaluate values and directional derivatives through second order with
    /// respect to `parameter`.
    pub fn evaluate_second<M, R, D>(
        &self,
        input: CoordinateInput<R, D>,
        stack: &Stack<M, R>,
        polarisation: Polarisation,
        parameter: Parameter,
    ) -> SolvedRealResult<B, SecondJet<Complex<R>, D>>
    where
        Complex<R>: ComplexScalar<RealField = R>,
        R: Float + FloatConst + FromPrimitive + Debug + Copy + ComplexField,
        D: Dimension,
        M: Clone,
        SecondJet<Complex<R>, D>: Jet<Scalar = Complex<R>, Dimension = D> + CompileJet<M, RealAxis>,
        RealAxis: ConstitutiveEvaluator<Complex<R>, D, M>,
        B: Backend<SecondJet<Complex<R>, D>, RealAxis>,
    {
        let mapping =
            DerivativeMapping::new([parameter]).map_err(SolveRequestError::DerivativeMapping)?;

        self._solve::<SecondJet<Complex<R>, D>, M>(input, stack, polarisation, &mapping)
    }

    /// Retain backend data while computing directional derivatives through
    /// second order with respect to `parameter`.
    pub fn retain_second<M, R, D>(
        &self,
        input: CoordinateInput<R, D>,
        stack: &Stack<M, R>,
        polarisation: Polarisation,
        parameter: Parameter,
    ) -> RetainedRealResult<B, SecondJet<Complex<R>, D>, M>
    where
        Complex<R>: ComplexScalar<RealField = R>,
        R: Float + FloatConst + FromPrimitive + Debug + Copy + ComplexField,
        D: Dimension,
        M: Clone,
        SecondJet<Complex<R>, D>: Jet<Scalar = Complex<R>, Dimension = D> + CompileJet<M, RealAxis>,
        RealAxis: ConstitutiveEvaluator<Complex<R>, D, M>,
        B: Backend<SecondJet<Complex<R>, D>, RealAxis>,
    {
        let mapping =
            DerivativeMapping::new([parameter]).map_err(SolveRequestError::DerivativeMapping)?;

        self._retain::<SecondJet<Complex<R>, D>, M>(input, stack, polarisation, &mapping)
    }
}

impl<B> RealAxisEvaluator<B> {
    /// Evaluate values and first derivatives with respect to two ordered
    /// caller-facing parameters.
    ///
    /// `axis0` and `axis1` determine the ordering of the returned bivariate
    /// derivative representation.
    pub fn evaluate_bivariate_first<M, R, D>(
        &self,
        input: CoordinateInput<R, D>,
        stack: &Stack<M, R>,
        polarisation: Polarisation,
        axis0: Parameter,
        axis1: Parameter,
    ) -> SolvedRealResult<B, BivariateFirstJet<Complex<R>, D>>
    where
        Complex<R>: ComplexScalar<RealField = R>,
        R: Float + FloatConst + FromPrimitive + Debug + Copy + ComplexField,
        D: Dimension,
        M: Clone,
        BivariateFirstJet<Complex<R>, D>:
            Jet<Scalar = Complex<R>, Dimension = D> + CompileJet<M, RealAxis>,
        RealAxis: ConstitutiveEvaluator<Complex<R>, D, M>,
        B: Backend<BivariateFirstJet<Complex<R>, D>, RealAxis>,
    {
        let mapping =
            DerivativeMapping::new([axis0, axis1]).map_err(SolveRequestError::DerivativeMapping)?;

        self._solve::<BivariateFirstJet<Complex<R>, D>, M>(input, stack, polarisation, &mapping)
    }

    /// Retain backend data while computing first derivatives with respect to
    /// two ordered caller-facing parameters.
    pub fn retain_bivariate_first<M, R, D>(
        &self,
        input: CoordinateInput<R, D>,
        stack: &Stack<M, R>,
        polarisation: Polarisation,
        axis0: Parameter,
        axis1: Parameter,
    ) -> RetainedRealResult<B, BivariateFirstJet<Complex<R>, D>, M>
    where
        Complex<R>: ComplexScalar<RealField = R>,
        R: Float + FloatConst + FromPrimitive + Debug + Copy + ComplexField,
        D: Dimension,
        M: Clone,
        BivariateFirstJet<Complex<R>, D>:
            Jet<Scalar = Complex<R>, Dimension = D> + CompileJet<M, RealAxis>,
        RealAxis: ConstitutiveEvaluator<Complex<R>, D, M>,
        B: Backend<BivariateFirstJet<Complex<R>, D>, RealAxis>,
    {
        let mapping =
            DerivativeMapping::new([axis0, axis1]).map_err(SolveRequestError::DerivativeMapping)?;

        self._retain::<BivariateFirstJet<Complex<R>, D>, M>(input, stack, polarisation, &mapping)
    }
}

impl<B> RealAxisEvaluator<B> {
    /// Evaluate values, a two-parameter gradient, and a symmetric Hessian.
    ///
    /// `axis0` and `axis1` determine the ordering of both the gradient and
    /// Hessian components.
    pub fn evaluate_bivariate_second<M, R, D>(
        &self,
        input: CoordinateInput<R, D>,
        stack: &Stack<M, R>,
        polarisation: Polarisation,
        axis0: Parameter,
        axis1: Parameter,
    ) -> SolvedRealResult<B, BivariateSecondJet<Complex<R>, D>>
    where
        Complex<R>: ComplexScalar<RealField = R>,
        R: Float + FloatConst + FromPrimitive + Debug + Copy + ComplexField,
        D: Dimension,
        M: Clone,
        BivariateSecondJet<Complex<R>, D>:
            Jet<Scalar = Complex<R>, Dimension = D> + CompileJet<M, RealAxis>,
        RealAxis: ConstitutiveEvaluator<Complex<R>, D, M>,
        B: Backend<BivariateSecondJet<Complex<R>, D>, RealAxis>,
    {
        let mapping =
            DerivativeMapping::new([axis0, axis1]).map_err(SolveRequestError::DerivativeMapping)?;

        self._solve::<BivariateSecondJet<Complex<R>, D>, M>(input, stack, polarisation, &mapping)
    }

    /// Retain backend data while computing a two-parameter gradient and
    /// symmetric Hessian.
    pub fn retain_bivariate_second<M, R, D>(
        &self,
        input: CoordinateInput<R, D>,
        stack: &Stack<M, R>,
        polarisation: Polarisation,
        axis0: Parameter,
        axis1: Parameter,
    ) -> RetainedRealResult<B, BivariateSecondJet<Complex<R>, D>, M>
    where
        Complex<R>: ComplexScalar<RealField = R>,
        R: Float + FloatConst + FromPrimitive + Debug + Copy + ComplexField,
        D: Dimension,
        M: Clone,
        BivariateSecondJet<Complex<R>, D>:
            Jet<Scalar = Complex<R>, Dimension = D> + CompileJet<M, RealAxis>,
        RealAxis: ConstitutiveEvaluator<Complex<R>, D, M>,
        B: Backend<BivariateSecondJet<Complex<R>, D>, RealAxis>,
    {
        let mapping =
            DerivativeMapping::new([axis0, axis1]).map_err(SolveRequestError::DerivativeMapping)?;

        self._retain::<BivariateSecondJet<Complex<R>, D>, M>(input, stack, polarisation, &mapping)
    }
}

impl<B> RealAxisEvaluator<B> {
    fn _solve<J, M>(
        &self,
        input: CoordinateInput<<J::Scalar as ComplexField>::RealField, J::Dimension>,
        stack: &Stack<M, <J::Scalar as ComplexField>::RealField>,
        polarisation: Polarisation,
        mapping: &DerivativeMapping,
    ) -> Result<
        RealAxisResult<J, PlaneWaveSolution<B::Entries>>,
        RealAxisEvaluationError<
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
                .map_err(RealAxisEvaluationError::compile)?;

        let exterior = evaluate_exterior_wavevectors::<RealAxis, M, J>(
            canonical_problem.coordinates(),
            canonical_problem.stack().left_exterior(),
            canonical_problem.stack().right_exterior(),
        );

        let solution = self
            .backend
            .solve(
                canonical_problem.coordinates(),
                canonical_problem.stack(),
                &exterior,
                polarisation,
            )
            .map_err(|err| RealAxisEvaluationError::Backend { source: err })?;

        Ok(RealAxisResult::new(solution, context))
    }

    fn _retain<J, M>(
        &self,
        input: CoordinateInput<<J::Scalar as ComplexField>::RealField, J::Dimension>,
        stack: &Stack<M, <J::Scalar as ComplexField>::RealField>,
        polarisation: Polarisation,
        mapping: &DerivativeMapping,
    ) -> Result<
        RealAxisState<J, M, B::Workspace>,
        RealAxisEvaluationError<
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
                .map_err(RealAxisEvaluationError::compile)?;

        let exterior = evaluate_exterior_wavevectors::<RealAxis, M, J>(
            canonical_problem.coordinates(),
            canonical_problem.stack().left_exterior(),
            canonical_problem.stack().right_exterior(),
        );

        let workspace = self
            .backend
            .retain(
                canonical_problem.coordinates(),
                canonical_problem.stack(),
                &exterior,
                polarisation,
            )
            .map_err(|err| RealAxisEvaluationError::Backend { source: err })?;

        Ok(RealAxisState::new(
            canonical_problem,
            workspace,
            context,
            stack.clone(),
            polarisation,
        ))
    }
}

type ValueJet<C, D, P = RealParameter> = ArrayJet0<C, D, P>;

type FirstJet<C, D, P = RealParameter> = ArrayJet1<C, D, P>;

type SecondJet<C, D, P = RealParameter> = ArrayJet2<C, D, P>;

type BivariateFirstJet<C, D, P = RealParameter> = ArrayJetBivariate1<C, D, P>;

type BivariateSecondJet<C, D, P = RealParameter> = ArrayJetBivariate2<C, D, P>;

type SolvedRealResult<B, J> = Result<
    RealAxisResult<J, PlaneWaveSolution<<B as Backend<J, RealAxis>>::Entries>>,
    RealAxisEvaluationError<
        CompilePlaneWaveError<<J as Jet>::Scalar>,
        <B as Backend<J, RealAxis>>::Error,
    >,
>;

type RetainedRealResult<B, J, M> = Result<
    RealAxisState<J, M, <B as Backend<J, RealAxis>>::Workspace>,
    RealAxisEvaluationError<
        CompilePlaneWaveError<<J as Jet>::Scalar>,
        <B as Backend<J, RealAxis>>::Error,
    >,
>;
