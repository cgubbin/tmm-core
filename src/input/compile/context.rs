use ndarray::{Array, Dimension};

use crate::{
    IncidentSide, Polarisation,
    input::{
        PlaneWaveCoordinates, compile::assignment::ParameterAssignment,
        plane_wave::PlaneWaveCoordinateValues,
    },
    stack::Thickness,
};

/// Non-canonical information retained alongside a compiled backend problem.
///
/// This contains everything required to interpret backend values and
/// assignment in the caller-facing parameterisation.
#[derive(Clone, Debug, PartialEq)]
pub struct CompilationContext<R, D>
where
    D: Dimension,
{
    coordinates: CoordinateContext<R, D>,
    stack: StackContext<R>,
    assignment: ParameterAssignment,
}

impl<R, D> CompilationContext<R, D>
where
    D: Dimension,
{
    pub fn new(
        coordinates: CoordinateContext<R, D>,
        stack: StackContext<R>,
        assignment: ParameterAssignment,
    ) -> Self {
        Self {
            coordinates,
            stack,
            assignment,
        }
    }

    pub fn coordinates(&self) -> &CoordinateContext<R, D> {
        &self.coordinates
    }

    pub fn stack(&self) -> &StackContext<R> {
        &self.stack
    }

    pub fn assignment(&self) -> &ParameterAssignment {
        &self.assignment
    }

    pub fn into_parts(
        self,
    ) -> (
        CoordinateContext<R, D>,
        StackContext<R>,
        ParameterAssignment,
    ) {
        (self.coordinates, self.stack, self.assignment)
    }
}

/// Caller-facing geometric description of the compiled stack.
///
/// This deliberately excludes material handles. Material models remain in the
/// canonical stack, while this context stores the lightweight geometry needed
/// for labels, derivative interpretation, and reporting.
#[derive(Clone, Debug, PartialEq)]
pub struct StackContext<R> {
    layer_thicknesses: Vec<Thickness<R>>,

    layer_labels: Vec<Option<String>>,
}

impl<R> StackContext<R> {
    pub fn new(layer_thicknesses: Vec<Thickness<R>>) -> Self {
        let len = layer_thicknesses.len();
        Self {
            layer_thicknesses,
            layer_labels: vec![None; len],
        }
    }

    pub fn layer_thicknesses(&self) -> &[Thickness<R>] {
        &self.layer_thicknesses
    }

    pub fn layer_thickness(&self, index: usize) -> Option<&Thickness<R>> {
        self.layer_thicknesses.get(index)
    }

    pub fn layer_count(&self) -> usize {
        self.layer_thicknesses.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layer_thicknesses.is_empty()
    }

    pub fn into_layer_thicknesses(self) -> Vec<Thickness<R>> {
        self.layer_thicknesses
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CoordinateContext<R, D>
where
    D: Dimension,
{
    coordinates: PlaneWaveCoordinates,
    values: PlaneWaveCoordinateValues<R, D>,
    incident_side: IncidentSide,
    polarisation: Polarisation,
}

impl<R, D> CoordinateContext<R, D>
where
    D: Dimension,
{
    pub(crate) fn new(
        coordinates: PlaneWaveCoordinates,
        values: PlaneWaveCoordinateValues<R, D>,
        incident_side: IncidentSide,
        polarisation: Polarisation,
    ) -> Self {
        Self {
            coordinates,
            values,
            incident_side,
            polarisation,
        }
    }

    pub fn coordinates(&self) -> PlaneWaveCoordinates {
        self.coordinates
    }

    pub fn values(&self) -> &PlaneWaveCoordinateValues<R, D> {
        &self.values
    }

    pub fn spectral_values(&self) -> &Array<R, D> {
        self.values.spectral()
    }

    pub fn in_plane_values(&self) -> &Array<R, D> {
        self.values.in_plane()
    }

    pub fn incident_side(&self) -> IncidentSide {
        self.incident_side
    }

    pub fn polarisation(&self) -> Polarisation {
        self.polarisation
    }

    pub fn into_parts(
        self,
    ) -> (
        PlaneWaveCoordinates,
        PlaneWaveCoordinateValues<R, D>,
        IncidentSide,
        Polarisation,
    ) {
        (
            self.coordinates,
            self.values,
            self.incident_side,
            self.polarisation,
        )
    }
}
