use crate::{
    PlaneWaveObservables, algebra::ComplexJet, backend::HasEntries, input::CompilationContext,
};

/// A retained plane-wave solution.
///
/// The state contains the canonical problem, the backend-specific retained
/// workspace, and the metadata required to crystallise derivative-aware
/// quantities into caller-facing coordinates and units.
///
/// No derived quantity is crystallised during evaluation.
#[derive(Clone, Debug)]
pub struct PlaneWaveState<M, J, W, Ctx> {
    problem: CanonicalPlaneWaveProblem<M, J>,
    workspace: W,
    context: Ctx,
}

impl<M, J, W, Ctx> PlaneWaveState<M, J, W, Ctx> {
    pub(crate) fn new(
        problem: CanonicalPlaneWaveProblem<M, J>,
        workspace: W,
        context: Ctx,
    ) -> Self {
        Self {
            problem,
            workspace,
            context,
        }
    }

    /// Return the compiled canonical plane-wave problem.
    pub fn problem(&self) -> &CanonicalPlaneWaveProblem<M, J> {
        &self.problem
    }

    /// Return the retained backend-specific workspace.
    pub fn workspace(&self) -> &W {
        &self.workspace
    }

    /// Return the retained compilation metadata.
    pub fn context(&self) -> &Ctx {
        &self.context
    }

    /// Consume the state and return its components.
    pub fn into_parts(self) -> (CanonicalPlaneWaveProblem<M, J>, W, Ctx) {
        (self.problem, self.workspace, self.context)
    }

    /// Transform the retained workspace while preserving the canonical
    /// problem and compilation context.
    pub fn map_workspace<W2>(self, map: impl FnOnce(W) -> W2) -> PlaneWaveState<M, J, W2, Ctx> {
        PlaneWaveState {
            problem: self.problem,
            workspace: map(self.workspace),
            context: self.context,
        }
    }
}

// impl<M, J, W, Ctx> PlaneWaveState<M, J, W, Ctx>
// where
//     W: HasEntries,
//     W::Entries: BuildPlaneWaveObservables<J>,
//     J: ComplexJet,
// {
//     /// Construct the complete uncrystallised set of plane-wave observables.
//     ///
//     /// This is primarily an internal extension point. Most callers should use
//     /// the crystallised observable accessors.
//     pub fn raw_observable(&self) -> PlaneWaveObservables<J, J::RealJet> {
//         self.workspace
//             .entries()
//             .build_plane_wave_observables(&self.problem)
//     }
// }
// //     /// Compute and crystallise the complete external plane-wave response.
// //     pub fn observables(
// //         &self,
// //     ) -> Result<
// //         <
// //             crate::observables::PlaneWaveObservables<
// //                 J,
// //                 J::RealAlgebra,
// //             > as crate::crystallise::Crystallise<
// //                 Ctx,
// //             >
// //         >::Output,
// //         <
// //             crate::observables::PlaneWaveObservables<
// //                 J,
// //                 J::RealAlgebra,
// //             > as crate::crystallise::Crystallise<
// //                 Ctx,
// //             >
// //         >::Error,
// //     >{
// //         self.raw_observables().crystallise(&self.context)
// //     }
// // }
