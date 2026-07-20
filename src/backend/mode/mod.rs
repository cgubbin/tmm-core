mod residual;

pub use residual::{
    AnalyticResidual, DifferentiableOutgoingModeResidualBackend, OutgoingModeResidualBackend,
    ResidualDerivatives,
};

use crate::{ComplexScalar, PlanarInput};

use nalgebra::ComplexField;
use ndarray::{Array0, Ix0, arr0};

#[derive(Clone, Debug, PartialEq)]
pub struct OutgoingModeResponse<C>
where
    C: ComplexField,
{
    mode: OutgoingMode<C>,
    residual: C,
    amplitudes: OutgoingModeAmplitudes<C>,
}

impl<C> OutgoingModeResponse<C>
where
    C: ComplexField + Copy,
{
    pub(crate) fn new(
        mode: OutgoingMode<C>,
        residual: C,
        amplitudes: OutgoingModeAmplitudes<C>,
    ) -> Self {
        Self {
            mode,
            residual,
            amplitudes,
        }
    }

    pub fn mode(&self) -> &OutgoingMode<C> {
        &self.mode
    }

    pub fn residual(&self) -> C {
        self.residual
    }

    pub fn amplitudes(&self) -> &OutgoingModeAmplitudes<C> {
        &self.amplitudes
    }

    pub fn into_parts(self) -> (OutgoingMode<C>, C, OutgoingModeAmplitudes<C>) {
        (self.mode, self.residual, self.amplitudes)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutgoingMode<C>
where
    C: ComplexField,
{
    input: PlanarInput<Array0<C>>,
    kind: OutgoingModeKind,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum OutgoingModeKind {
    ComplexVacuumWavenumber,
    ComplexParallelWavenumber,
}

#[derive(Clone, Debug, PartialEq)]
pub struct OutgoingModeAmplitudes<C> {
    pub(crate) left_outgoing: Array0<C>,
    pub(crate) right_outgoing: Array0<C>,
}

impl<C: ComplexScalar> OutgoingModeAmplitudes<C> {
    pub(crate) fn normalised(mut left: C, mut right: C) -> Self {
        let left_norm = left.modulus_squared();
        let right_norm = right.modulus_squared();

        let norm = (left_norm.clone() + right_norm.clone()).sqrt();
        let norm = C::from_real(norm);

        left /= norm.clone();
        right /= norm;

        let reference = if left_norm >= right_norm {
            left.clone()
        } else {
            right.clone()
        };

        let magnitude = reference.modulus();

        if magnitude > C::zero().real() {
            let phase = reference / C::from_real(magnitude);
            let correction = phase.conjugate();

            left *= correction.clone();
            right *= correction;
        }

        Self {
            left_outgoing: arr0(left),
            right_outgoing: arr0(right),
        }
    }

    pub(crate) fn left(&self) -> &Array0<C> {
        &self.left_outgoing
    }

    pub(crate) fn right(&self) -> &Array0<C> {
        &self.right_outgoing
    }
}

impl<C> OutgoingMode<C>
where
    C: ComplexField + Copy,
{
    pub fn new(input: PlanarInput<Array0<C>>) -> Self {
        let kind = match (
            input.vacuum_wavenumber()[()].imaginary() == C::zero().real(),
            input.parallel_wavenumber()[()].imaginary() == C::zero().real(),
        ) {
            (true, false) => OutgoingModeKind::ComplexParallelWavenumber,
            (false, true) => OutgoingModeKind::ComplexVacuumWavenumber,
            _ => panic!(),
        };

        Self { input, kind }
    }

    pub fn input(&self) -> &PlanarInput<Array0<C>> {
        &self.input
    }

    pub fn kind(&self) -> OutgoingModeKind {
        self.kind
    }
}

pub trait OutgoingModeStateBackend<C, S>: OutgoingModeResidualBackend<C, Ix0, S>
where
    C: ComplexScalar,
{
    fn outgoing_mode_state(
        &self,
        stack: &S,
        input: &PlanarInput<Array0<C>>,
    ) -> Result<OutgoingModeResponse<C>, Self::Error>;
}
