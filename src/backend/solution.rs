use crate::{
    IncidentSide, PlaneWaveAmplitudes,
    observable::{ProjectAmplitudes, ProjectPlaneWaveModeDeterminant, ProjectPower},
};

pub trait PlaneWaveEntries {
    type ExteriorContext;
}

pub struct PlaneWaveSolution<E: PlaneWaveEntries> {
    entries: E,
    context: E::ExteriorContext,
}

impl<E: PlaneWaveEntries> PlaneWaveSolution<E> {
    pub const fn new(entries: E, context: E::ExteriorContext) -> Self {
        Self { entries, context }
    }

    pub fn entries(&self) -> &E {
        &self.entries
    }

    pub fn context(&self) -> &E::ExteriorContext {
        &self.context
    }

    pub fn as_view(&self) -> PlaneWaveSolutionView<'_, E> {
        PlaneWaveSolutionView {
            entries: &self.entries,
            context: &self.context,
        }
    }

    pub(crate) fn replace_entries(&mut self, entries: E) {
        self.entries = entries;
    }

    pub(crate) fn entries_mut(&mut self) -> &mut E {
        &mut self.entries
    }

    pub fn amplitudes(&self, incident_side: IncidentSide) -> E::Amplitudes
    where
        E: ProjectAmplitudes,
    {
        self.entries()
            .project_amplitudes(self.context(), incident_side)
    }

    pub fn power(&self, incident_side: IncidentSide) -> E::Power
    where
        E: ProjectPower,
    {
        self.entries().project_power(self.context(), incident_side)
    }

    pub fn determinant(&self) -> E::Determinant
    where
        E: ProjectPlaneWaveModeDeterminant,
    {
        self.entries().project_determinant(self.context())
    }

    pub fn into_parts(self) -> (E, E::ExteriorContext) {
        (self.entries, self.context)
    }
}

pub struct PlaneWaveSolutionView<'a, E: PlaneWaveEntries> {
    entries: &'a E,
    context: &'a E::ExteriorContext,
}

impl<'a, E: PlaneWaveEntries> PlaneWaveSolutionView<'a, E> {
    pub const fn new(entries: &'a E, context: &'a E::ExteriorContext) -> Self {
        Self { entries, context }
    }

    pub fn entries(&self) -> &'a E {
        self.entries
    }

    pub fn context(&self) -> &'a E::ExteriorContext {
        self.context
    }

    pub fn amplitudes(&self, incident_side: IncidentSide) -> E::Amplitudes
    where
        E: ProjectAmplitudes,
    {
        self.entries()
            .project_amplitudes(self.context(), incident_side)
    }

    pub fn power(&self, incident_side: IncidentSide) -> E::Power
    where
        E: ProjectPower,
    {
        self.entries().project_power(self.context(), incident_side)
    }

    pub fn determinant(&self) -> E::Determinant
    where
        E: ProjectPlaneWaveModeDeterminant,
    {
        self.entries().project_determinant(self.context())
    }
}
