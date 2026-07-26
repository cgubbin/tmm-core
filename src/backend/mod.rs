use crate::{
    PlaneWaveObservables, RealAxis,
    algebra::ComplexJet,
    input::{
        CanonicalPlaneWaveInput, CanonicalPlaneWaveProblem, CanonicalProblem, CanonicalSolverInput,
    },
};

use ndarray::Dimension;

// mod field;
// mod input;
mod isotropic;
// mod matrix;
// mod mode;
// mod plane_wave;

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

pub(crate) trait PlaneWaveBackend: Backend<RealAxis>
where
    Self::Workspace: HasEntries<Entries = Self::Entries>,
{
    fn solve<M, J>(
        &self,
        input: &CanonicalPlaneWaveProblem<M, J>,
    ) -> Result<BackendResponse<J, NoWorkspace>, Self::Error>
    where
        J: ComplexJet,
        Self::Entries: BuildPlaneWaveObservables<J>,
    {
        let problem = input.problem();
        let entries = Backend::solve(self, problem)?;

        let observables = entries.build_plane_wave_observables(input);

        Ok(BackendResponse {
            external: observables,
            internal: NoWorkspace,
        })
    }

    fn retain<M, J>(
        &self,
        input: &CanonicalPlaneWaveProblem<M, J>,
    ) -> Result<BackendResponse<J, Self::Workspace>, Self::Error>
    where
        J: ComplexJet,
        Self::Entries: BuildPlaneWaveObservables<J>,
    {
        let problem = input.problem();
        let workspace = Backend::retain(self, problem)?;

        let observables = workspace.entries().build_plane_wave_observables(input);

        Ok(BackendResponse {
            external: observables,
            internal: workspace,
        })
    }
}

pub(crate) trait Backend<Domain> {
    type Workspace;
    type Entries;

    type Error;

    fn solve<M, J>(&self, problem: &CanonicalProblem<M, J>) -> Result<Self::Entries, Self::Error>;

    fn retain<M, J>(
        &self,
        problem: &CanonicalProblem<M, J>,
    ) -> Result<Self::Workspace, Self::Error>;
}

pub(crate) trait BuildPlaneWaveObservables<J>
where
    J: ComplexJet,
{
    fn build_plane_wave_observables<M>(
        &self,
        input: &CanonicalPlaneWaveProblem<M, J>,
    ) -> PlaneWaveObservables<J, J::RealJet>;
}

pub(crate) trait HasEntries {
    type Entries;

    fn entries(&self) -> &Self::Entries;
}

pub(crate) trait IntoEntries: HasEntries {
    fn into_entries(self) -> Self::Entries;
}

// #[cfg(test)]
// mod tests;

// pub(crate)mod scatter2;
// // pub(crate)mod transfer2;

// pub(crate)use derivative::{
//     DerivativeVariable, SpectralDerivativeVariable, StructuralDerivativeVariable,
// };

// pub(crate)use field::{
//     ExteriorSampling, FieldPosition, FieldSampling, FieldSamplingRegion, IsotropicFieldState,
//     LayerSampling, PlaneWaveFieldBackend, PlaneWaveFieldError, PlaneWaveFieldResponse,
//     PlaneWaveFieldSample, PlaneWaveFieldSampleOwned, PlaneWaveFieldSampleView, PlaneWaveFields,
//     PlaneWavePowerBalance,
// };

// pub(crate)use input::{IncidentSide, PlanarInput, PlaneWaveInput, Polarisation};

// pub(crate)use matrix::{
//     ComplexMatrixBackend, ComplexMatrixKxDerivativeBackend, ComplexMatrixSpectralDerivativeBackend,
//     ComplexMatrixThicknessDerivativeBackend, MatrixEvaluation, RawMatrixBackend,
//     RawMatrixKxDerivativeBackend, RawMatrixSpectralDerivativeBackend,
//     RawMatrixThicknessDerivativeBackend,
// };

// pub(crate)use mode::{
//     AnalyticResidual, OutgoingModeResidualBackend, OutgoingModeResidualKxDerivativeBackend,
//     OutgoingModeResidualSpectralDerivativeBackend, OutgoingModeResidualThicknessDerivativeBackend,
//     OutgoingModeResponse, OutgoingModeStateBackend,
// };

// pub(crate)use plane_wave::{
//     PlaneWaveAmplitudeDifferential, PlaneWaveAmplitudes, PlaneWaveBackend,
//     PlaneWaveKxDerivativeBackend, PlaneWavePower, PlaneWavePowerDifferential, PlaneWaveResponse,
//     PlaneWaveResponseDerivatives, PlaneWaveResponseDifferential,
//     PlaneWaveSpectralDerivativeBackend, PlaneWaveThicknessDerivativeBackend,
// };

// use isotropic::IsotropicLayerQuantities;
