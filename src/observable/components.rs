#[derive(Clone, Debug, PartialEq)]
pub(super) struct ElectromagneticComponents<R> {
    electric: R,
    magnetic: R,
    coupling: R,
    total: R,
}

impl<R> ElectromagneticComponents<R> {
    pub(super) fn new(electric: R, magnetic: R, coupling: R, total: R) -> Self {
        Self {
            electric,
            magnetic,
            coupling,
            total,
        }
    }

    /// Return the electric contribution to the energy density.
    pub(super) fn electric(&self) -> &R {
        &self.electric
    }

    /// Return the magnetic contribution to the energy density.
    pub(super) fn magnetic(&self) -> &R {
        &self.magnetic
    }

    /// Return the coupling contribution to the energy density.
    pub(super) fn coupling(&self) -> &R {
        &self.coupling
    }

    /// Return the total electromagnetic energy density.
    pub(super) fn total(&self) -> &R {
        &self.total
    }

    /// Consume the value and return its electric, magnetic, coupling and total
    /// components.
    pub(super) fn into_parts(self) -> (R, R, R, R) {
        (self.electric, self.magnetic, self.coupling, self.total)
    }

    /// Transform the storage of every component.
    pub(super) fn map<U>(self, mut f: impl FnMut(R) -> U) -> ElectromagneticComponents<U> {
        ElectromagneticComponents {
            electric: f(self.electric),
            magnetic: f(self.magnetic),
            coupling: f(self.coupling),
            total: f(self.total),
        }
    }
}
