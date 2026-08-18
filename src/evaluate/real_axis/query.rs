use nalgebra::ComplexField;
use ndarray::Dimension;

use crate::{
    IncidentSide,
    algebra::Jet,
    backend::PlaneWaveSolutionSource,
    differential::IntoDifferentialResponse,
    input::JetMapping,
    observable::{ProjectAmplitudes, ProjectPlaneWaveModeDeterminant, ProjectPower},
};

use super::{RealAxisResult, RealAxisState};

#[doc(hidden)]
pub trait PlaneWaveQuery<J>
where
    J: Jet + JetMapping,
{
    type Source: PlaneWaveSolutionSource;

    fn source(&self) -> &Self::Source;

    fn mapping(&self) -> &J::Mapping;
}

impl<J, S> PlaneWaveQuery<J> for RealAxisResult<J, S>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    <J::Scalar as ComplexField>::RealField: ComplexField,
    S: PlaneWaveSolutionSource,
{
    type Source = S;

    fn source(&self) -> &Self::Source {
        self.solution()
    }

    fn mapping(&self) -> &J::Mapping {
        self.context().mapping()
    }
}

impl<J, M, W> PlaneWaveQuery<J> for RealAxisState<J, M, W>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    <J::Scalar as ComplexField>::RealField: ComplexField,
    W: PlaneWaveSolutionSource,
{
    type Source = W;

    fn source(&self) -> &Self::Source {
        self.workspace()
    }

    fn mapping(&self) -> &J::Mapping {
        self.context().mapping()
    }
}

pub(crate) trait RealAxisExternalQueries<J>: PlaneWaveQuery<J>
where
    J: Jet + JetMapping,
    J::Policy: Default,
{
    fn raw_amplitudes(&self, incident_side: IncidentSide) -> RawAmplitudes<Self, J>
    where
        Self: Sized,
        QueryEntries<Self, J>: ProjectAmplitudes,
    {
        self.source().solution().amplitudes(incident_side)
    }

    fn raw_power(&self, incident_side: IncidentSide) -> RawPower<Self, J>
    where
        Self: Sized,
        QueryEntries<Self, J>: ProjectPower,
    {
        self.source().solution().power(incident_side)
    }
}

impl<J, Q> RealAxisExternalQueries<J> for Q
where
    J: Jet + JetMapping,
    J::Policy: Default,
    Q: PlaneWaveQuery<J>,
{
}

pub(crate) type QueryEntries<Q, J> =
    <<Q as PlaneWaveQuery<J>>::Source as PlaneWaveSolutionSource>::Entries;

pub(crate) type RawAmplitudes<Q, J> = <QueryEntries<Q, J> as ProjectAmplitudes>::Amplitudes;

pub(crate) type RawPower<Q, J> = <QueryEntries<Q, J> as ProjectPower>::Power;

pub(crate) type DifferentialResponseFor<J, T> =
    <T as IntoDifferentialResponse<<J as JetMapping>::Policy, <J as JetMapping>::Mapping>>::Output;
