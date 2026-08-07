use crate::{
    InterfacePower, Response,
    differential::DifferentialResponse,
    field::ScalarField,
    input::CoordinatePoint,
    response::{InterfaceLocation, InterfaceMetadata},
};

use ndarray::{ArrayView1, Dimension};

pub type InterfacePowerResponse<R, ED, D> = Response<
    InterfacePower<ScalarField<R, <ED as Dimension>::Larger>>,
    InterfaceMetadata<R, ED>,
    D,
>;

/// A borrowed interface response at one canonical excitation point.
///
/// The excitation dimensions have been selected, leaving the spatial axis.
/// Both observable values and requested derivatives are retained
pub struct InterfaceProfile<'a, F, D, R> {
    response: DifferentialResponse<F, D>,
    excitation: CoordinatePoint<R>,
    interfaces: ArrayView1<'a, InterfaceLocation<R>>,
}

impl<'a, F, D, R> InterfaceProfile<'a, F, D, R> {
    pub fn response(&self) -> &DifferentialResponse<F, D> {
        &self.response
    }

    pub fn observables(&self) -> &F {
        self.response.value()
    }

    pub fn derivatives(&self) -> &D {
        self.response.derivatives()
    }

    pub fn excitation(&self) -> &CoordinatePoint<R> {
        &self.excitation
    }

    pub fn interfaces(&self) -> ArrayView1<'a, InterfaceLocation<R>> {
        self.interfaces
    }
}
