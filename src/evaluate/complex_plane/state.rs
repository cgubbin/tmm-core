use nalgebra::ComplexField;
use ndarray::Dimension;
use num_traits::{FromPrimitive, One};

use crate::{
    CanonicalCoordinates, ComplexPlane, ComplexScalar, ExteriorWavevectors, Polarisation,
    algebra::{ComplexJet, Jet, JetStack, ScalarAlgebra, ScalarAlgebraExpRelExt},
    backend::{
        ExteriorContextProvider, ModalSolutionSource, PlaneWaveEntries, PlaneWaveSolutionSource,
        PlaneWaveSolutionView, ReconstructLayerModeWaves, RetainedIsotropicLayers,
    },
    input::{CanonicalStack, JetMapping},
    material::{ConstitutiveDerivativeEvaluator, ConstitutiveSpectralFirstLift},
    observable::{
        AggregateBilinearNormalization, ConstitutiveSamplingContext, ConstitutiveSamplingError,
        IsotropicConstitutiveParameters, IsotropicConstitutiveSpectralData,
        ProjectPlaneWaveModeDeterminant,
    },
    spatial::ResolvedFieldSampling,
};

use super::{ComplexPlaneMode, QnmCreationError, RawModeDeterminant};

#[derive(Clone, Debug)]
pub struct ComplexPlaneState<'a, J, M, W>
where
    J: Jet,
{
    coordinates: CanonicalCoordinates<J>,
    exterior: ExteriorWavevectors<J>,
    stack: &'a CanonicalStack<M, J>,
    workspace: W,
    polarisation: Polarisation,
}

impl<'a, J, M, W> ComplexPlaneState<'a, J, M, W>
where
    J: Jet,
{
    pub(crate) fn new(
        coordinates: CanonicalCoordinates<J>,
        exterior: ExteriorWavevectors<J>,
        stack: &'a CanonicalStack<M, J>,
        workspace: W,
        polarisation: Polarisation,
    ) -> Self {
        Self {
            coordinates,
            exterior,
            stack,
            workspace,
            polarisation,
        }
    }

    pub fn coordinates(&self) -> &CanonicalCoordinates<J> {
        &self.coordinates
    }

    pub fn exterior(&self) -> &ExteriorWavevectors<J> {
        &self.exterior
    }

    pub fn stack(&self) -> &CanonicalStack<M, J> {
        self.stack
    }

    pub fn workspace(&self) -> &W {
        &self.workspace
    }

    pub fn polarisation(&self) -> Polarisation {
        self.polarisation
    }

    pub fn solution(&self) -> PlaneWaveSolutionView<'_, W::Entries>
    where
        W: PlaneWaveSolutionSource,
    {
        self.workspace.solution()
    }

    pub fn mode(&self) -> Result<ComplexPlaneMode<'_, J, M, W>, QnmCreationError>
    where
        J: ComplexJet
            + ScalarAlgebra
            + ScalarAlgebraExpRelExt
            + ConstitutiveSpectralFirstLift<ComplexPlane, M>
            + Clone
            + JetMapping,
        J::RealJet: ScalarAlgebra,
        J::Scalar: ComplexScalar,
        <J::RealJet as Jet>::Scalar: FromPrimitive + One,
        J::Dimension: Dimension,
        ComplexPlane: ConstitutiveDerivativeEvaluator<J::Scalar, J::Dimension, M>,
        W: PlaneWaveSolutionSource
            + ReconstructLayerModeWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>
            + ModalSolutionSource<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
    {
        ComplexPlaneMode::new(self)
    }

    pub(crate) fn raw_constitutive_parameters(
        &self,
        sampling: &ResolvedFieldSampling<<J::Scalar as ComplexField>::RealField>,
    ) -> Result<IsotropicConstitutiveParameters<J::Stacked>, ConstitutiveSamplingError>
    where
        W: PlaneWaveSolutionSource + RetainedIsotropicLayers<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
        J: JetStack + Clone,
        J::Dimension: Dimension,
    {
        let context = ConstitutiveSamplingContext::new(self.workspace());

        context.sample(sampling)
    }

    pub(crate) fn raw_constitutive_spectral_first_parameters<E>(
        &self,
        sampling: &ResolvedFieldSampling<<J::Scalar as ComplexField>::RealField>,
    ) -> Result<IsotropicConstitutiveSpectralData<J::Stacked>, ConstitutiveSamplingError>
    where
        W: PlaneWaveSolutionSource + RetainedIsotropicLayers<Algebra = J>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = J>,
        J: JetStack + Clone,
        J::Dimension: Dimension,
        J::Scalar: ComplexScalar,
        E: ConstitutiveDerivativeEvaluator<J::Scalar, J::Dimension, M>,
        J: ConstitutiveSpectralFirstLift<E, M>,
    {
        let context = ConstitutiveSamplingContext::new(self.workspace());

        context.sample_spectral_first(sampling, self.stack())
    }
}

impl<'a, J, M, W> ComplexPlaneState<'a, J, M, W>
where
    J: Jet,
    W: PlaneWaveSolutionSource,
{
    pub fn determinant(&self) -> RawModeDeterminant<W::Entries>
    where
        W::Entries: ProjectPlaneWaveModeDeterminant,
    {
        self.solution().determinant()
    }
}
