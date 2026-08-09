use num_complex::Complex;

use crate::{
    ComplexPlane, Polarisation, ValidationConfig,
    algebra::{
        ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2,
        HolomorphicParameter, Jet, RealParameter,
    },
    backend::{Backend, PlaneWaveSolution},
    domain::RealAxis,
    evaluate::PlaneWaveResult,
    input::{CompileJet, CompilePlaneWaveError, CoordinateInput, compile_complex, compile_real},
    material::ConstitutiveEvaluator,
    parameter::{DerivativeMapping, Parameter},
    scalar::ComplexScalar,
    stack::Stack,
};

use super::{PlaneWaveEvaluationError, PlaneWaveState, SolveRequestError};

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

        self.solve_real_coordinate_space::<ValueJet<Complex<R>, D>, M>(
            input,
            stack,
            polarisation,
            &mapping,
        )
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

        self.retain_real_coordinate_space::<ValueJet<Complex<R>, D>, M>(
            input,
            stack,
            polarisation,
            &mapping,
        )
    }
}

impl<B> PlaneWaveEvaluator<B> {
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

        self.solve_real_coordinate_space::<FirstJet<Complex<R>, D>, M>(
            input,
            stack,
            polarisation,
            &mapping,
        )
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

        self.retain_real_coordinate_space::<FirstJet<Complex<R>, D>, M>(
            input,
            stack,
            polarisation,
            &mapping,
        )
    }
}

impl<B> PlaneWaveEvaluator<B> {
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

        self.solve_real_coordinate_space::<SecondJet<Complex<R>, D>, M>(
            input,
            stack,
            polarisation,
            &mapping,
        )
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

        self.retain_real_coordinate_space::<SecondJet<Complex<R>, D>, M>(
            input,
            stack,
            polarisation,
            &mapping,
        )
    }
}

impl<B> PlaneWaveEvaluator<B> {
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

        self.solve_real_coordinate_space::<BivariateFirstJet<Complex<R>, D>, M>(
            input,
            stack,
            polarisation,
            &mapping,
        )
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

        self.retain_real_coordinate_space::<BivariateFirstJet<Complex<R>, D>, M>(
            input,
            stack,
            polarisation,
            &mapping,
        )
    }
}

impl<B> PlaneWaveEvaluator<B> {
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

        self.solve_real_coordinate_space::<BivariateSecondJet<Complex<R>, D>, M>(
            input,
            stack,
            polarisation,
            &mapping,
        )
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

        self.retain_real_coordinate_space::<BivariateSecondJet<Complex<R>, D>, M>(
            input,
            stack,
            polarisation,
            &mapping,
        )
    }
}

impl<B> PlaneWaveEvaluator<B> {
    /// Evaluate the outgoing modal determinant over complex caller-facing
    /// coordinates.
    ///
    /// This value-only path performs no derivative seeding. It is intended for
    /// complex continuation, contour integration, and mode refinement.
    pub fn evaluate_modal<M, C, D>(
        &self,
        input: CoordinateInput<C, D>,
        stack: &Stack<M, C::RealField>,
        polarisation: Polarisation,
    ) -> SolvedModalResult<B, ModalValueJet<C, D>>
    where
        C: ComplexScalar,
        C::RealField: Float + FloatConst + FromPrimitive + Debug + Copy,
        D: Dimension,
        M: Clone,
        ModalValueJet<C, D>: Jet<Scalar = C, Dimension = D> + CompileJet<M, ComplexPlane>,
        ComplexPlane: ConstitutiveEvaluator<C, D, M>,
        B: Backend<ModalValueJet<C, D>, ComplexPlane>,
    {
        let mapping = DerivativeMapping::none();

        self.solve_complex_coordinate_space::<ModalValueJet<C, D>, M>(
            input,
            stack,
            polarisation,
            &mapping,
        )
    }

    /// Retain backend data for a value-only modal evaluation.
    pub fn retain_modal<M, C, D>(
        &self,
        input: CoordinateInput<C, D>,
        stack: &Stack<M, C::RealField>,
        polarisation: Polarisation,
    ) -> RetainedModalResult<B, ModalValueJet<C, D>, M>
    where
        C: ComplexScalar,
        C::RealField: Float + FloatConst + FromPrimitive + Debug + Copy,
        D: Dimension,
        M: Clone,
        ModalValueJet<C, D>: Jet<Scalar = C, Dimension = D> + CompileJet<M, ComplexPlane>,
        ComplexPlane: ConstitutiveEvaluator<C, D, M>,
        B: Backend<ModalValueJet<C, D>, ComplexPlane>,
    {
        let mapping = DerivativeMapping::none();

        self.retain_complex_coordinate_space::<ModalValueJet<C, D>, M>(
            input,
            stack,
            polarisation,
            &mapping,
        )
    }
}

impl<B> PlaneWaveEvaluator<B> {
    /// Evaluate the modal determinant and its first holomorphic derivative
    /// with respect to `parameter`.
    ///
    /// The selected parameter is analytically continued as a complex variable.
    /// Operations requiring real-parameter or Hermitian derivative semantics
    /// are not available on the returned result.
    pub fn evaluate_modal_first<M, C, D>(
        &self,
        input: CoordinateInput<C, D>,
        stack: &Stack<M, C::RealField>,
        polarisation: Polarisation,
        parameter: Parameter,
    ) -> SolvedModalResult<B, ModalFirstJet<C, D>>
    where
        C: ComplexScalar,
        C::RealField: Float + FloatConst + FromPrimitive + Debug + Copy,
        D: Dimension,
        M: Clone,
        ModalFirstJet<C, D>: Jet<Scalar = C, Dimension = D> + CompileJet<M, ComplexPlane>,
        ComplexPlane: ConstitutiveEvaluator<C, D, M>,
        B: Backend<ModalFirstJet<C, D>, ComplexPlane>,
    {
        let mapping =
            DerivativeMapping::new([parameter]).map_err(SolveRequestError::DerivativeMapping)?;

        self.solve_complex_coordinate_space::<ModalFirstJet<C, D>, M>(
            input,
            stack,
            polarisation,
            &mapping,
        )
    }

    /// Retain backend data while computing a first holomorphic modal
    /// derivative.
    pub fn retain_modal_first<M, C, D>(
        &self,
        input: CoordinateInput<C, D>,
        stack: &Stack<M, C::RealField>,
        polarisation: Polarisation,
        parameter: Parameter,
    ) -> RetainedModalResult<B, ModalFirstJet<C, D>, M>
    where
        C: ComplexScalar,
        C::RealField: Float + FloatConst + FromPrimitive + Debug + Copy,
        D: Dimension,
        M: Clone,
        ModalFirstJet<C, D>: Jet<Scalar = C, Dimension = D> + CompileJet<M, ComplexPlane>,
        ComplexPlane: ConstitutiveEvaluator<C, D, M>,
        B: Backend<ModalFirstJet<C, D>, ComplexPlane>,
    {
        let mapping =
            DerivativeMapping::new([parameter]).map_err(SolveRequestError::DerivativeMapping)?;

        self.retain_complex_coordinate_space::<ModalFirstJet<C, D>, M>(
            input,
            stack,
            polarisation,
            &mapping,
        )
    }
}

impl<B> PlaneWaveEvaluator<B> {
    pub fn evaluate_modal_bivariate_first<M, C, D>(
        &self,
        input: CoordinateInput<C, D>,
        stack: &Stack<M, C::RealField>,
        polarisation: Polarisation,
        axis0: Parameter,
        axis1: Parameter,
    ) -> SolvedModalResult<B, ModalBivariateFirstJet<C, D>>
    where
        C: ComplexScalar,
        C::RealField: Float + FloatConst + FromPrimitive + Debug + Copy,
        D: Dimension,
        M: Clone,
        ModalBivariateFirstJet<C, D>: Jet<Scalar = C, Dimension = D> + CompileJet<M, ComplexPlane>,
        ComplexPlane: ConstitutiveEvaluator<C, D, M>,
        B: Backend<ModalBivariateFirstJet<C, D>, ComplexPlane>,
    {
        let mapping =
            DerivativeMapping::new([axis0, axis1]).map_err(SolveRequestError::DerivativeMapping)?;

        self.solve_complex_coordinate_space::<ModalBivariateFirstJet<C, D>, M>(
            input,
            stack,
            polarisation,
            &mapping,
        )
    }

    pub fn retain_modal_bivariate_first<M, C, D>(
        &self,
        input: CoordinateInput<C, D>,
        stack: &Stack<M, C::RealField>,
        polarisation: Polarisation,
        axis0: Parameter,
        axis1: Parameter,
    ) -> RetainedModalResult<B, ModalBivariateFirstJet<C, D>, M>
    where
        C: ComplexScalar,
        C::RealField: Float + FloatConst + FromPrimitive + Debug + Copy,
        D: Dimension,
        M: Clone,
        ModalBivariateFirstJet<C, D>: Jet<Scalar = C, Dimension = D> + CompileJet<M, ComplexPlane>,
        ComplexPlane: ConstitutiveEvaluator<C, D, M>,
        B: Backend<ModalBivariateFirstJet<C, D>, ComplexPlane>,
    {
        let mapping =
            DerivativeMapping::new([axis0, axis1]).map_err(SolveRequestError::DerivativeMapping)?;

        self.retain_complex_coordinate_space::<ModalBivariateFirstJet<C, D>, M>(
            input,
            stack,
            polarisation,
            &mapping,
        )
    }
}

impl<B> PlaneWaveEvaluator<B> {
    /// Evaluate the modal determinant and its holomorphic derivatives through
    /// second order with respect to `parameter`.
    pub fn evaluate_modal_second<M, C, D>(
        &self,
        input: CoordinateInput<C, D>,
        stack: &Stack<M, C::RealField>,
        polarisation: Polarisation,
        parameter: Parameter,
    ) -> SolvedModalResult<B, ModalSecondJet<C, D>>
    where
        C: ComplexScalar,
        C::RealField: Float + FloatConst + FromPrimitive + Debug + Copy,
        D: Dimension,
        M: Clone,
        ModalSecondJet<C, D>: Jet<Scalar = C, Dimension = D> + CompileJet<M, ComplexPlane>,
        ComplexPlane: ConstitutiveEvaluator<C, D, M>,
        B: Backend<ModalSecondJet<C, D>, ComplexPlane>,
    {
        let mapping =
            DerivativeMapping::new([parameter]).map_err(SolveRequestError::DerivativeMapping)?;

        self.solve_complex_coordinate_space::<ModalSecondJet<C, D>, M>(
            input,
            stack,
            polarisation,
            &mapping,
        )
    }

    /// Retain backend data while computing holomorphic modal derivatives
    /// through second order.
    pub fn retain_modal_second<M, C, D>(
        &self,
        input: CoordinateInput<C, D>,
        stack: &Stack<M, C::RealField>,
        polarisation: Polarisation,
        parameter: Parameter,
    ) -> RetainedModalResult<B, ModalSecondJet<C, D>, M>
    where
        C: ComplexScalar,
        C::RealField: Float + FloatConst + FromPrimitive + Debug + Copy,
        D: Dimension,
        M: Clone,
        ModalSecondJet<C, D>: Jet<Scalar = C, Dimension = D> + CompileJet<M, ComplexPlane>,
        ComplexPlane: ConstitutiveEvaluator<C, D, M>,
        B: Backend<ModalSecondJet<C, D>, ComplexPlane>,
    {
        let mapping =
            DerivativeMapping::new([parameter]).map_err(SolveRequestError::DerivativeMapping)?;

        self.retain_complex_coordinate_space::<ModalSecondJet<C, D>, M>(
            input,
            stack,
            polarisation,
            &mapping,
        )
    }
}

impl<B> PlaneWaveEvaluator<B> {
    pub fn evaluate_modal_bivariate_second<M, C, D>(
        &self,
        input: CoordinateInput<C, D>,
        stack: &Stack<M, C::RealField>,
        polarisation: Polarisation,
        axis0: Parameter,
        axis1: Parameter,
    ) -> SolvedModalResult<B, ModalBivariateSecondJet<C, D>>
    where
        C: ComplexScalar,
        C::RealField: Float + FloatConst + FromPrimitive + Debug + Copy,
        D: Dimension,
        M: Clone,
        ModalBivariateSecondJet<C, D>: Jet<Scalar = C, Dimension = D> + CompileJet<M, ComplexPlane>,
        ComplexPlane: ConstitutiveEvaluator<C, D, M>,
        B: Backend<ModalBivariateSecondJet<C, D>, ComplexPlane>,
    {
        let mapping =
            DerivativeMapping::new([axis0, axis1]).map_err(SolveRequestError::DerivativeMapping)?;

        self.solve_complex_coordinate_space::<ModalBivariateSecondJet<C, D>, M>(
            input,
            stack,
            polarisation,
            &mapping,
        )
    }

    pub fn retain_modal_bivariate_second<M, C, D>(
        &self,
        input: CoordinateInput<C, D>,
        stack: &Stack<M, C::RealField>,
        polarisation: Polarisation,
        axis0: Parameter,
        axis1: Parameter,
    ) -> RetainedModalResult<B, ModalBivariateSecondJet<C, D>, M>
    where
        C: ComplexScalar,
        C::RealField: Float + FloatConst + FromPrimitive + Debug + Copy,
        D: Dimension,
        M: Clone,
        ModalBivariateSecondJet<C, D>: Jet<Scalar = C, Dimension = D> + CompileJet<M, ComplexPlane>,
        ComplexPlane: ConstitutiveEvaluator<C, D, M>,
        B: Backend<ModalBivariateSecondJet<C, D>, ComplexPlane>,
    {
        let mapping =
            DerivativeMapping::new([axis0, axis1]).map_err(SolveRequestError::DerivativeMapping)?;

        self.retain_complex_coordinate_space::<ModalBivariateSecondJet<C, D>, M>(
            input,
            stack,
            polarisation,
            &mapping,
        )
    }
}

impl<B> PlaneWaveEvaluator<B> {
    fn solve_real_coordinate_space<J, M>(
        &self,
        input: CoordinateInput<<J::Scalar as ComplexField>::RealField, J::Dimension>,
        stack: &Stack<M, <J::Scalar as ComplexField>::RealField>,
        polarisation: Polarisation,
        mapping: &DerivativeMapping,
    ) -> Result<
        PlaneWaveResult<J, <J::Scalar as ComplexField>::RealField, PlaneWaveSolution<B::Entries>>,
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

        let solution = self
            .backend
            .solve(&canonical_problem, polarisation)
            .map_err(|err| PlaneWaveEvaluationError::Backend { source: err })?;

        Ok(PlaneWaveResult::new(solution, context))
    }

    fn solve_complex_coordinate_space<J, M>(
        &self,
        input: CoordinateInput<J::Scalar, J::Dimension>,
        stack: &Stack<M, <J::Scalar as ComplexField>::RealField>,
        polarisation: Polarisation,
        mapping: &DerivativeMapping,
    ) -> Result<
        PlaneWaveResult<J, J::Scalar, PlaneWaveSolution<B::Entries>>,
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

        let solution = self
            .backend
            .solve(&canonical_problem, polarisation)
            .map_err(|err| PlaneWaveEvaluationError::Backend { source: err })?;

        Ok(PlaneWaveResult::new(solution, context))
    }

    fn retain_real_coordinate_space<J, M>(
        &self,
        input: CoordinateInput<<J::Scalar as ComplexField>::RealField, J::Dimension>,
        stack: &Stack<M, <J::Scalar as ComplexField>::RealField>,
        polarisation: Polarisation,
        mapping: &DerivativeMapping,
    ) -> Result<
        PlaneWaveState<J, <J::Scalar as ComplexField>::RealField, M, B::Workspace>,
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
            .retain(&canonical_problem, polarisation)
            .map_err(|err| PlaneWaveEvaluationError::Backend { source: err })?;

        Ok(PlaneWaveState::new(
            canonical_problem,
            workspace,
            context,
            stack.clone(),
            polarisation,
        ))
    }

    fn retain_complex_coordinate_space<J, M>(
        &self,
        input: CoordinateInput<J::Scalar, J::Dimension>,
        stack: &Stack<M, <J::Scalar as ComplexField>::RealField>,
        polarisation: Polarisation,
        mapping: &DerivativeMapping,
    ) -> Result<
        PlaneWaveState<J, J::Scalar, M, B::Workspace>,
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
            .retain(&canonical_problem, polarisation)
            .map_err(|err| PlaneWaveEvaluationError::Backend { source: err })?;

        Ok(PlaneWaveState::new(
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

type ModalValueJet<C, D> = ArrayJet0<C, D, HolomorphicParameter>;

type ModalFirstJet<C, D> = ArrayJet1<C, D, HolomorphicParameter>;

type ModalSecondJet<C, D> = ArrayJet2<C, D, HolomorphicParameter>;

type ModalBivariateFirstJet<C, D> = ArrayJetBivariate1<C, D, HolomorphicParameter>;

type ModalBivariateSecondJet<C, D> = ArrayJetBivariate2<C, D, HolomorphicParameter>;

type SolvedRealResult<B, J> = Result<
    PlaneWaveResult<
        J,
        <<J as Jet>::Scalar as ComplexField>::RealField,
        PlaneWaveSolution<<B as Backend<J, RealAxis>>::Entries>,
    >,
    PlaneWaveEvaluationError<
        CompilePlaneWaveError<<J as Jet>::Scalar>,
        <B as Backend<J, RealAxis>>::Error,
    >,
>;

type RetainedRealResult<B, J, M> = Result<
    PlaneWaveState<
        J,
        <<J as Jet>::Scalar as ComplexField>::RealField,
        M,
        <B as Backend<J, RealAxis>>::Workspace,
    >,
    PlaneWaveEvaluationError<
        CompilePlaneWaveError<<J as Jet>::Scalar>,
        <B as Backend<J, RealAxis>>::Error,
    >,
>;

type SolvedModalResult<B, J> = Result<
    PlaneWaveResult<
        J,
        <J as Jet>::Scalar,
        PlaneWaveSolution<<B as Backend<J, ComplexPlane>>::Entries>,
    >,
    PlaneWaveEvaluationError<
        CompilePlaneWaveError<<J as Jet>::Scalar>,
        <B as Backend<J, ComplexPlane>>::Error,
    >,
>;

type RetainedModalResult<B, J, M> = Result<
    PlaneWaveState<J, <J as Jet>::Scalar, M, <B as Backend<J, ComplexPlane>>::Workspace>,
    PlaneWaveEvaluationError<
        CompilePlaneWaveError<<J as Jet>::Scalar>,
        <B as Backend<J, ComplexPlane>>::Error,
    >,
>;
