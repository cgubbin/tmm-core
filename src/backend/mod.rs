use crate::{
    PlaneWaveObservables, RealAxis,
    algebra::ComplexJet,
    input::{CanonicalBackendInput, CanonicalSolverInput},
};

use ndarray::Dimension;

mod isotropic;
// mod scatter2;

/// Internal field data requested from a backend solve.
///
/// The derivative order is represented by the backend workspace entry type.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum InternalFieldRequest {
    /// Compute only the external response.
    None,

    /// Retain enough data to reconstruct waves at finite-layer boundaries.
    LayerBoundaries,
}

impl InternalFieldRequest {
    pub(crate) const fn is_requested(self) -> bool {
        matches!(self, Self::LayerBoundaries)
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub(crate) struct NoWorkspace;

#[derive(Clone, Debug)]
pub(crate) struct BackendResponse<J, W>
where
    J: ComplexJet,
{
    external: PlaneWaveObservables<J, J::RealJet>,
    internal: W,
}

impl<J, W> BackendResponse<J, W>
where
    J: ComplexJet,
{
    pub(crate) fn external(&self) -> &PlaneWaveObservables<J, J::RealJet> {
        &self.external
    }

    pub(crate) fn internal(&self) -> &W {
        &self.internal
    }

    pub(crate) fn into_external(self) -> PlaneWaveObservables<J, J::RealJet> {
        self.external
    }

    pub(crate) fn into_internal(self) -> W {
        self.internal
    }

    pub(crate) fn into_parts(self) -> (PlaneWaveObservables<J, J::RealJet>, W) {
        (self.external, self.internal)
    }
}

// pub(crate) trait PlaneWaveBackend<J>: Backend<J, RealAxis>
// where
//     J: ComplexJet,
//     <Self::Workspace as HasEntries>::Entries: BuildPlaneWaveObservables<J>,
// {
//     fn solve_plane_wave<M>(
//         &self,
//         input: &CanonicalPlaneWaveProblem<M, J>,
//     ) -> Result<BackendResponse<J, NoWorkspace>, Self::Error> {
//         let problem = input.problem();
//         let entries = Backend::solve(self, problem)?;

//         let observables = entries.build_plane_wave_observables(input);

//         Ok(BackendResponse {
//             external: observables,
//             internal: NoWorkspace,
//         })
//     }

//     fn retain_plane_wave<M>(
//         &self,
//         input: &CanonicalPlaneWaveProblem<M, J>,
//     ) -> Result<BackendResponse<J, Self::Workspace>, Self::Error> {
//         let problem = input.problem();
//         let workspace = Backend::retain(self, problem)?;

//         let observables = workspace.entries().build_plane_wave_observables(input);

//         Ok(BackendResponse {
//             external: observables,
//             internal: workspace,
//         })
//     }
// }

pub(crate) trait Backend<J, Domain> {
    type Workspace: IntoEntries;

    type Error;

    fn solve<M>(
        &self,
        problem: &CanonicalBackendInput<M, J>,
    ) -> Result<<Self::Workspace as HasEntries>::Entries, Self::Error>;

    fn retain<M>(
        &self,
        problem: &CanonicalBackendInput<M, J>,
    ) -> Result<Self::Workspace, Self::Error>;
}

// pub(crate) trait BuildPlaneWaveObservables<J>
// where
//     J: ComplexJet,
// {
//     fn build_plane_wave_observables<M>(
//         &self,
//         input: &CanonicalPlaneWaveProblem<M, J>,
//     ) -> PlaneWaveObservables<J, J::RealJet>;
// }

pub(crate) trait HasEntries {
    type Entries;

    fn entries(&self) -> &Self::Entries;
}

pub(crate) trait IntoEntries: HasEntries {
    fn into_entries(self) -> Self::Entries;
}
