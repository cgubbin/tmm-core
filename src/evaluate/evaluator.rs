use std::marker::PhantomData;

use crate::{
    DifferentiableMaterial, Material, ValidationConfig,
    algebra::{
        ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, ComplexJet,
        RealParameter, ScalarAlgebra,
    },
    backend::{Backend, BuildPlaneWaveObservables, HasEntries, IntoEntries, PlaneWaveBackend},
    domain::RealAxis,
    input::{
        CompilationContext, CompilePlaneWaveError, CompilePlaneWaveJet, Parameter,
        ParameterAssignment, ParameterAssignmentError, PlaneWaveInput, compile_plane_wave_problem,
    },
    material::ConstitutiveLift,
    scalar::ComplexScalar,
    stack::Stack,
};

use super::{PlaneWaveEvaluationError, PlaneWaveResult, PlaneWaveState, RequestError};

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

use ndarray::Dimension;
use num_traits::{Float, FloatConst, FromPrimitive};
use std::fmt::Debug;

impl<B> PlaneWaveEvaluator<B> {
    fn retain_with_jet<M, J, C, D, W, E>(
        &self,
        input: PlaneWaveInput<C::RealField, D>,
        stack: &Stack<M, C::RealField>,
        assignment: ParameterAssignment,
    ) -> Result<
        PlaneWaveState<M, J, W, CompilationContext<C::RealField, D>>,
        PlaneWaveEvaluationError<CompilePlaneWaveError<C::RealField>, E>,
    >
    where
        B: Backend<J, RealAxis, Workspace = W, Error = E> + PlaneWaveBackend<J>,
        W: IntoEntries,
        W::Entries: BuildPlaneWaveObservables<J>,
        C: ComplexScalar,
        C::RealField: Copy + Float + FloatConst + FromPrimitive + Debug,
        D: Dimension,
        J: ComplexJet
            + CompilePlaneWaveJet<C, D>
            + ScalarAlgebra<C, D>
            + ConstitutiveLift<C, D, RealAxis, M>,
        M: Clone + Material<Real = C::RealField>,
    {
        let (canonical_problem, context) = compile_plane_wave_problem::<M, C, D, J>(
            input,
            stack,
            &ValidationConfig::permissive(),
            assignment,
        )
        .map_err(PlaneWaveEvaluationError::compile)?;

        let workspace = Backend::<J, RealAxis>::retain(&self.backend, canonical_problem.problem())
            .map_err(PlaneWaveEvaluationError::backend)?;

        Ok(PlaneWaveState::new(canonical_problem, workspace, context))
    }

    pub fn retain<M, C, D, W, E>(
        &self,
        input: PlaneWaveInput<C::RealField, D>,
        stack: &Stack<M, C::RealField>,
    ) -> Result<
        PlaneWaveState<M, ArrayJet0<C, D, RealParameter>, W, CompilationContext<C::RealField, D>>,
        PlaneWaveEvaluationError<CompilePlaneWaveError<C::RealField>, E>,
    >
    where
        B: Backend<ArrayJet0<C, D, RealParameter>, RealAxis, Workspace = W, Error = E>
            + PlaneWaveBackend<ArrayJet0<C, D, RealParameter>>,
        W: IntoEntries,
        W::Entries: BuildPlaneWaveObservables<ArrayJet0<C, D, RealParameter>>,
        C: ComplexScalar,
        C::RealField: Copy + Float + FloatConst + FromPrimitive + Debug,
        D: Dimension,
        M: Clone + Material<Real = C::RealField>,
    {
        self.retain_with_jet::<M, ArrayJet0<C, D, RealParameter>, C, D, W, E>(
            input,
            stack,
            ParameterAssignment::none(),
        )
    }

    fn solve_with_jet<M, J, C, D, W, E>(
        &self,
        input: PlaneWaveInput<C::RealField, D>,
        stack: &Stack<M, C::RealField>,
        assignment: ParameterAssignment,
    ) -> Result<
        PlaneWaveResult<J, CompilationContext<C::RealField, D>>,
        PlaneWaveEvaluationError<CompilePlaneWaveError<C::RealField>, E>,
    >
    where
        B: Backend<J, RealAxis, Workspace = W, Error = E> + PlaneWaveBackend<J>,
        W: IntoEntries,
        W::Entries: BuildPlaneWaveObservables<J>,
        C: ComplexScalar,
        C::RealField: Copy + Float + FloatConst + FromPrimitive + Debug,
        D: Dimension,
        J: ComplexJet
            + CompilePlaneWaveJet<C, D>
            + ScalarAlgebra<C, D>
            + ConstitutiveLift<C, D, RealAxis, M>,
        M: Clone + Material<Real = C::RealField>,
    {
        let (canonical_problem, context) = compile_plane_wave_problem::<M, C, D, J>(
            input,
            stack,
            &ValidationConfig::permissive(),
            assignment,
        )
        .map_err(PlaneWaveEvaluationError::compile)?;

        let response = PlaneWaveBackend::<J>::solve_plane_wave(&self.backend, &canonical_problem)
            .map_err(PlaneWaveEvaluationError::backend)?;

        Ok(PlaneWaveResult::new(response.into_external(), context))
    }

    pub fn solve<M, C, D, W, E>(
        &self,
        input: PlaneWaveInput<C::RealField, D>,
        stack: &Stack<M, C::RealField>,
    ) -> Result<
        PlaneWaveResult<ArrayJet0<C, D, RealParameter>, CompilationContext<C::RealField, D>>,
        PlaneWaveEvaluationError<CompilePlaneWaveError<C::RealField>, E>,
    >
    where
        B: Backend<ArrayJet0<C, D, RealParameter>, RealAxis, Workspace = W, Error = E>
            + PlaneWaveBackend<ArrayJet0<C, D, RealParameter>>,
        W: IntoEntries,
        W::Entries: BuildPlaneWaveObservables<ArrayJet0<C, D, RealParameter>>,
        C: ComplexScalar,
        C::RealField: Copy + Float + FloatConst + FromPrimitive + Debug,
        D: Dimension,
        M: Clone + Material<Real = C::RealField>,
    {
        self.solve_with_jet::<M, ArrayJet0<C, D, RealParameter>, C, D, W, E>(
            input,
            stack,
            ParameterAssignment::none(),
        )
    }

    pub fn solve_first<M, C, D, W, E>(
        &self,
        input: PlaneWaveInput<C::RealField, D>,
        stack: &Stack<M, C::RealField>,
        parameter: Parameter,
    ) -> Result<
        PlaneWaveResult<ArrayJet1<C, D, RealParameter>, CompilationContext<C::RealField, D>>,
        PlaneWaveEvaluationError<CompilePlaneWaveError<C::RealField>, E>,
    >
    where
        B: Backend<ArrayJet1<C, D, RealParameter>, RealAxis, Workspace = W, Error = E>
            + PlaneWaveBackend<ArrayJet1<C, D, RealParameter>>,
        W: IntoEntries,
        W::Entries: BuildPlaneWaveObservables<ArrayJet1<C, D, RealParameter>>,
        C: ComplexScalar,
        C::RealField: Copy + Float + FloatConst + FromPrimitive + Debug,
        D: Dimension,
        M: Clone + DifferentiableMaterial<Real = C::RealField>,
    {
        parameter
            .validate(stack.len())
            .map_err(RequestError::ThicknessSeed)?;

        let assignment = parameter.assignment().map_err(RequestError::from)?;

        self.solve_with_jet::<M, ArrayJet1<C, D, RealParameter>, C, D, W, E>(
            input, stack, assignment,
        )
    }

    pub fn retain_first<M, C, D, W, E>(
        &self,
        input: PlaneWaveInput<C::RealField, D>,
        stack: &Stack<M, C::RealField>,
        parameter: Parameter,
    ) -> Result<
        PlaneWaveState<M, ArrayJet1<C, D, RealParameter>, W, CompilationContext<C::RealField, D>>,
        PlaneWaveEvaluationError<CompilePlaneWaveError<C::RealField>, E>,
    >
    where
        B: Backend<ArrayJet1<C, D, RealParameter>, RealAxis, Workspace = W, Error = E>
            + PlaneWaveBackend<ArrayJet1<C, D, RealParameter>>,
        W: IntoEntries,
        W::Entries: BuildPlaneWaveObservables<ArrayJet1<C, D, RealParameter>>,
        C: ComplexScalar,
        C::RealField: Copy + Float + FloatConst + FromPrimitive + Debug,
        D: Dimension,
        M: Clone + DifferentiableMaterial<Real = C::RealField>,
    {
        parameter
            .validate(stack.len())
            .map_err(RequestError::ThicknessSeed)?;

        let assignment = parameter.assignment().map_err(RequestError::from)?;

        self.retain_with_jet::<M, ArrayJet1<C, D, RealParameter>, C, D, W, E>(
            input, stack, assignment,
        )
    }

    pub fn solve_second<M, C, D, W, E>(
        &self,
        input: PlaneWaveInput<C::RealField, D>,
        stack: &Stack<M, C::RealField>,
        parameter: Parameter,
    ) -> Result<
        PlaneWaveResult<ArrayJet2<C, D, RealParameter>, CompilationContext<C::RealField, D>>,
        PlaneWaveEvaluationError<CompilePlaneWaveError<C::RealField>, E>,
    >
    where
        B: Backend<ArrayJet2<C, D, RealParameter>, RealAxis, Workspace = W, Error = E>
            + PlaneWaveBackend<ArrayJet2<C, D, RealParameter>>,
        W: IntoEntries,
        W::Entries: BuildPlaneWaveObservables<ArrayJet2<C, D, RealParameter>>,
        C: ComplexScalar,
        C::RealField: Copy + Float + FloatConst + FromPrimitive + Debug,
        D: Dimension,
        M: Clone + DifferentiableMaterial<Real = C::RealField>,
    {
        parameter
            .validate(stack.len())
            .map_err(RequestError::ThicknessSeed)?;

        let assignment = parameter.assignment().map_err(RequestError::from)?;

        self.solve_with_jet::<M, ArrayJet2<C, D, RealParameter>, C, D, W, E>(
            input, stack, assignment,
        )
    }

    pub fn retain_second<M, C, D, W, E>(
        &self,
        input: PlaneWaveInput<C::RealField, D>,
        stack: &Stack<M, C::RealField>,
        parameter: Parameter,
    ) -> Result<
        PlaneWaveState<M, ArrayJet2<C, D, RealParameter>, W, CompilationContext<C::RealField, D>>,
        PlaneWaveEvaluationError<CompilePlaneWaveError<C::RealField>, E>,
    >
    where
        B: Backend<ArrayJet2<C, D, RealParameter>, RealAxis, Workspace = W, Error = E>
            + PlaneWaveBackend<ArrayJet2<C, D, RealParameter>>,
        W: IntoEntries,
        W::Entries: BuildPlaneWaveObservables<ArrayJet2<C, D, RealParameter>>,
        C: ComplexScalar,
        C::RealField: Copy + Float + FloatConst + FromPrimitive + Debug,
        D: Dimension,
        M: Clone + DifferentiableMaterial<Real = C::RealField>,
    {
        parameter
            .validate(stack.len())
            .map_err(RequestError::ThicknessSeed)?;

        let assignment = parameter.assignment().map_err(RequestError::from)?;

        self.retain_with_jet::<M, ArrayJet2<C, D, RealParameter>, C, D, W, E>(
            input, stack, assignment,
        )
    }

    pub fn solve_coordinate_gradient<M, C, D, W, E>(
        &self,
        input: PlaneWaveInput<C::RealField, D>,
        stack: &Stack<M, C::RealField>,
    ) -> Result<
        PlaneWaveResult<
            ArrayJetBivariate1<C, D, RealParameter>,
            CompilationContext<C::RealField, D>,
        >,
        PlaneWaveEvaluationError<CompilePlaneWaveError<C::RealField>, E>,
    >
    where
        B: Backend<ArrayJetBivariate1<C, D, RealParameter>, RealAxis, Workspace = W, Error = E>
            + PlaneWaveBackend<ArrayJetBivariate1<C, D, RealParameter>>,
        W: IntoEntries,
        W::Entries: BuildPlaneWaveObservables<ArrayJetBivariate1<C, D, RealParameter>>,
        C: ComplexScalar,
        C::RealField: Copy + Float + FloatConst + FromPrimitive + Debug,
        D: Dimension,
        M: Clone + DifferentiableMaterial<Real = C::RealField>,
    {
        let assignment = coordinate_assignment().map_err(RequestError::from)?;

        self.solve_with_jet::<M, ArrayJetBivariate1<C, D, RealParameter>, C, D, W, E>(
            input, stack, assignment,
        )
    }

    pub fn retain_coordinate_gradient<M, C, D, W, E>(
        &self,
        input: PlaneWaveInput<C::RealField, D>,
        stack: &Stack<M, C::RealField>,
    ) -> Result<
        PlaneWaveState<
            M,
            ArrayJetBivariate1<C, D, RealParameter>,
            W,
            CompilationContext<C::RealField, D>,
        >,
        PlaneWaveEvaluationError<CompilePlaneWaveError<C::RealField>, E>,
    >
    where
        B: Backend<ArrayJetBivariate1<C, D, RealParameter>, RealAxis, Workspace = W, Error = E>
            + PlaneWaveBackend<ArrayJetBivariate1<C, D, RealParameter>>,
        W: IntoEntries,
        W::Entries: BuildPlaneWaveObservables<ArrayJetBivariate1<C, D, RealParameter>>,
        C: ComplexScalar,
        C::RealField: Copy + Float + FloatConst + FromPrimitive + Debug,
        D: Dimension,
        M: Clone + DifferentiableMaterial<Real = C::RealField>,
    {
        let assignment = coordinate_assignment().map_err(RequestError::from)?;

        self.retain_with_jet::<M, ArrayJetBivariate1<C, D, RealParameter>, C, D, W, E>(
            input, stack, assignment,
        )
    }

    pub fn solve_coordinate_hessian<M, C, D, W, E>(
        &self,
        input: PlaneWaveInput<C::RealField, D>,
        stack: &Stack<M, C::RealField>,
    ) -> Result<
        PlaneWaveResult<
            ArrayJetBivariate2<C, D, RealParameter>,
            CompilationContext<C::RealField, D>,
        >,
        PlaneWaveEvaluationError<CompilePlaneWaveError<C::RealField>, E>,
    >
    where
        B: Backend<ArrayJetBivariate2<C, D, RealParameter>, RealAxis, Workspace = W, Error = E>
            + PlaneWaveBackend<ArrayJetBivariate2<C, D, RealParameter>>,
        W: IntoEntries,
        W::Entries: BuildPlaneWaveObservables<ArrayJetBivariate2<C, D, RealParameter>>,
        C: ComplexScalar,
        C::RealField: Copy + Float + FloatConst + FromPrimitive + Debug,
        D: Dimension,
        M: Clone + DifferentiableMaterial<Real = C::RealField>,
    {
        let assignment = coordinate_assignment().map_err(RequestError::from)?;

        self.solve_with_jet::<M, ArrayJetBivariate2<C, D, RealParameter>, C, D, W, E>(
            input, stack, assignment,
        )
    }

    pub fn retain_coordinate_hessian<M, C, D, W, E>(
        &self,
        input: PlaneWaveInput<C::RealField, D>,
        stack: &Stack<M, C::RealField>,
    ) -> Result<
        PlaneWaveState<
            M,
            ArrayJetBivariate2<C, D, RealParameter>,
            W,
            CompilationContext<C::RealField, D>,
        >,
        PlaneWaveEvaluationError<CompilePlaneWaveError<C::RealField>, E>,
    >
    where
        B: Backend<ArrayJetBivariate2<C, D, RealParameter>, RealAxis, Workspace = W, Error = E>
            + PlaneWaveBackend<ArrayJetBivariate2<C, D, RealParameter>>,
        W: IntoEntries,
        W::Entries: BuildPlaneWaveObservables<ArrayJetBivariate2<C, D, RealParameter>>,
        C: ComplexScalar,
        C::RealField: Copy + Float + FloatConst + FromPrimitive + Debug,
        D: Dimension,
        M: Clone + DifferentiableMaterial<Real = C::RealField>,
    {
        let assignment = coordinate_assignment().map_err(RequestError::from)?;

        self.retain_with_jet::<M, ArrayJetBivariate2<C, D, RealParameter>, C, D, W, E>(
            input, stack, assignment,
        )
    }
}

fn coordinate_assignment() -> Result<ParameterAssignment, ParameterAssignmentError> {
    ParameterAssignment::new([Parameter::Spectral, Parameter::InPlane])
}
