//! Backend-independent plane-wave solution containers.
//!
//! Numerical backends produce backend-specific matrix entries together with
//! exterior-medium context required to interpret those entries physically.
//!
//! [`PlaneWaveSolution`] owns both pieces, while [`PlaneWaveSolutionView`]
//! provides the same projection interface over borrowed data.
//!
//! Observable projections such as amplitudes, power, and modal determinants
//! are implemented separately through projection traits. This keeps backend
//! storage independent from the public physical quantities derived from it.

use crate::{
    IncidentSide, Polarisation,
    observable::{ProjectAmplitudes, ProjectPlaneWaveModeDeterminant, ProjectPower},
};

/// Exterior-medium context required to interpret backend solution entries.
///
/// The context stores the branch-selected exterior propagation quantities,
/// constitutive properties, canonical coordinates, and polarization used by
/// observable projections.
pub trait ExteriorContextProvider {
    /// Scalar algebra representation used by the backend.
    type Algebra;

    /// Return the left exterior characteristic admittance.
    fn left_admittance(&self) -> &Self::Algebra;

    /// Return the right exterior characteristic admittance.
    fn right_admittance(&self) -> &Self::Algebra;

    /// Return the left exterior normal angular wavenumber.
    fn left_kappa(&self) -> &Self::Algebra;

    /// Return the right exterior normal angular wavenumber.
    fn right_kappa(&self) -> &Self::Algebra;

    /// Return the left exterior relative permittivity.
    fn left_epsilon(&self) -> &Self::Algebra;

    /// Return the right exterior relative permittivity.
    fn right_epsilon(&self) -> &Self::Algebra;

    /// Return the left exterior relative permeability.
    fn left_mu(&self) -> &Self::Algebra;

    /// Return the right exterior relative permeability.
    fn right_mu(&self) -> &Self::Algebra;

    /// Return the canonical vacuum angular wavenumber.
    fn vacuum_angular_wavenumber(&self) -> &Self::Algebra;

    /// Return the canonical parallel angular wavenumber.
    fn parallel_angular_wavenumber(&self) -> &Self::Algebra;

    /// Return the selected polarization.
    fn polarisation(&self) -> Polarisation;
}

pub trait PlaneWaveEntries {
    type ExteriorContext: ExteriorContextProvider<Algebra = Self::Algebra>;
    type Algebra;
}

/// A source from which a borrowed plane-wave solution can be obtained.
///
/// Implemented by completed solutions and retained backend workspaces so that
/// projection code can operate uniformly on either representation.
pub trait PlaneWaveSolutionSource {
    type Entries: PlaneWaveEntries;

    fn solution(&self) -> PlaneWaveSolutionView<'_, Self::Entries>;
}

impl<E: PlaneWaveEntries> PlaneWaveSolutionSource for PlaneWaveSolution<E> {
    type Entries = E;

    fn solution(&self) -> PlaneWaveSolutionView<'_, Self::Entries> {
        self.as_view()
    }
}

#[derive(Clone, Debug)]
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
