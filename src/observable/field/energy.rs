#[derive(Clone, Debug, PartialEq)]
pub struct ElectromagneticEnergy<R> {
    electric: R,
    magnetic: R,
    total: R,
}

impl<R> ElectromagneticEnergy<R> {
    pub(crate) fn new(electric: R, magnetic: R, total: R) -> Self {
        Self {
            electric,
            magnetic,
            total,
        }
    }

    /// Return the electric contribution to the Energy density.
    pub fn electric(&self) -> &R {
        &self.electric
    }

    /// Return the magnetic contribution to the Energy density.
    pub fn magnetic(&self) -> &R {
        &self.magnetic
    }

    /// Return the total electromagnetic Energy density.
    pub fn total(&self) -> &R {
        &self.total
    }

    /// Consume the value and return its electric, magnetic, coupling and total
    /// components.
    pub fn into_parts(self) -> (R, R, R) {
        (self.electric, self.magnetic, self.total)
    }

    /// Transform the storage of every component.
    pub fn map<U>(self, mut f: impl FnMut(R) -> U) -> ElectromagneticEnergy<U> {
        ElectromagneticEnergy {
            electric: f(self.electric),
            magnetic: f(self.magnetic),
            total: f(self.total),
        }
    }
}
