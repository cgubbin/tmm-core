// use crate::{Backend, ValidationConfig};

// use num_traits::Float;

// #[derive(Clone, Debug, Default)]
// pub struct Evaluator<B, R: Float> {
//     backend: B,
//     validation: ValidationConfig<R>,
// }

// impl<R: Float> Evaluator<R> {
//     pub fn new() -> Self
//     where
//         R: Default,
//     {
//         Self::default()
//     }

//     pub fn with_backend(mut self, backend: Backend) -> Self {
//         self.backend = backend;
//         self
//     }

//     pub fn with_validation(mut self, validation: ValidationConfig<R>) -> Self {
//         self.validation = validation;
//         self
//     }

//     pub fn backend(&self) -> Backend {
//         self.backend
//     }

//     pub fn validation(&self) -> &ValidationConfig<R> {
//         &self.validation
//     }
// }

// // use crate::{
// //     ComplexScalar, Material, Stack, ValidationConfig,
// //     algebra::{
// //         ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, ScalarAlgebra,
// //     },
// //     input::{
// //         CompilationPlan, CompileProblemError, DerivativeParameter, ParameterAssignment,
// //         PlaneWaveInput, SeedJet, SolveRequest, compile_problem, plan_compilation,
// //     },
// //     material::{ConstitutiveEvaluator, ConstitutiveLift},
// // };
// // use ndarray::{Array, Dimension};
// // use num_traits::{Float, FloatConst, FromPrimitive};
// // use std::fmt::Debug;
// // use thiserror::Error;

// // #[derive(Debug, Error)]
// // pub enum EvaluateError<R, E> {
// //     #[error(transparent)]
// //     Compile(#[from] CompileProblemError<R>),

// //     #[error(transparent)]
// //     Backend(#[from] BackendError<E>),
// // }

// // pub fn evaluate_request<M, R, D>(
// //     input: PlaneWaveInput<R, D>,
// //     stack: &Stack<M, R>,
// //     request: SolveRequest,
// // ) -> Result<DynamicPlaneWaveResponse<M::Complex, D>, EvaluateError<R, E>>
// // where
// //     D: Dimension,
// // {
// //     let plan = plan_compilation(request, input.coordinates(), stack.len())?;
// //     let validation = ValidationConfig::permissive();

// //     match plan {
// //         CompilationPlan::Value { assignment } => evaluate_with_jet::<
// //             M,
// //             R,
// //             D,
// //             ArrayJet0<M::Complex, D, _>,
// //         >(input, stack, &validation, assignment)
// //         .map(DynamicPlaneWaveResponse::Value),

// //         CompilationPlan::UnivariateFirst {
// //             assignment,
// //             parameter,
// //         } => evaluate_with_jet::<M, R, D, ArrayJet1<M::Complex, D, _>>(
// //             input,
// //             stack,
// //             &validation,
// //             assignment,
// //         )
// //         .map(|response| DynamicPlaneWaveResponse::First {
// //             parameter,
// //             response,
// //         }),

// //         CompilationPlan::UnivariateSecond {
// //             assignment,
// //             parameter,
// //         } => evaluate_with_jet::<M, R, D, ArrayJet2<M::Complex, D, _>>(
// //             input,
// //             stack,
// //             &validation,
// //             assignment,
// //         )
// //         .map(|response| DynamicPlaneWaveResponse::Second {
// //             parameter,
// //             response,
// //         }),

// //         CompilationPlan::CoordinateGradient { assignment } => {
// //             evaluate_with_jet::<M, R, D, ArrayJetBivariate1<M::Complex, D, _>>(
// //                 input,
// //                 stack,
// //                 &validation,
// //                 assignment,
// //             )
// //             .map(DynamicPlaneWaveResponse::CoordinateGradient)
// //         }

// //         CompilationPlan::CoordinateHessian { assignment } => {
// //             evaluate_with_jet::<M, R, D, ArrayJetBivariate2<M::Complex, D, _>>(
// //                 input,
// //                 stack,
// //                 &validation,
// //                 assignment,
// //             )
// //             .map(DynamicPlaneWaveResponse::CoordinateHessian)
// //         }
// //     }
// // }

// // fn evaluate_with_jet<M, R, C, D, J, Domain>(
// //     input: PlaneWaveInput<R, D>,
// //     stack: &Stack<M, R>,
// //     validation: &ValidationConfig<R>,
// //     assignment: ParameterAssignment,
// // ) -> Result<BackendResult<C, D, J>, EvaluateError<R, M::Error>>
// // where
// //     R: Float + FloatConst + FromPrimitive + Copy + Debug,
// //     C: ComplexScalar<RealField = R> + Copy,
// //     D: Dimension + Clone,
// //     J: ScalarAlgebra<C, D> + ConstitutiveLift<C, D, Domain, M> + SeedJet<Array<C, D>>,
// //     Domain: ConstitutiveEvaluator<C, D, M>,
// //     M: Clone + Material<Real = R>,
// // {
// //     let compiled = compile_problem::<M, R, C, D, J, Domain>(input, stack, validation, assignment)?;

// //     backend.solve(compiled.canonical())
// // }
