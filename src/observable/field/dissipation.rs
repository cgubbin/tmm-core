/// Time-averaged volumetric electromagnetic dissipation density.
///
/// Positive values indicate power transferred from the electromagnetic field
/// to the material. Negative values indicate gain.
///
/// `electric` and `magnetic` are the contributions associated with electric
/// and magnetic material loss, and `total` is the complete local dissipation
/// density under the constitutive convention documented by the producing
/// operation.
#[derive(Clone, Debug, PartialEq)]
pub struct ElectromagneticDissipation<R> {
    electric: R,
    magnetic: R,
    total: R,
}

impl<R> ElectromagneticDissipation<R> {
    pub(crate) fn new(electric: R, magnetic: R, total: R) -> Self {
        Self {
            electric,
            magnetic,
            total,
        }
    }

    /// Return the electric contribution to the dissipation density.
    pub fn electric(&self) -> &R {
        &self.electric
    }

    /// Return the magnetic contribution to the dissipation density.
    pub fn magnetic(&self) -> &R {
        &self.magnetic
    }

    /// Return the total electromagnetic dissipation density.
    pub fn total(&self) -> &R {
        &self.total
    }

    /// Consume the value and return its electric, magnetic, coupling and total
    /// components.
    pub fn into_parts(self) -> (R, R, R) {
        (self.electric, self.magnetic, self.total)
    }

    /// Transform the storage of every component.
    pub fn map<U>(self, mut f: impl FnMut(R) -> U) -> ElectromagneticDissipation<U> {
        ElectromagneticDissipation {
            electric: f(self.electric),
            magnetic: f(self.magnetic),
            total: f(self.total),
        }
    }
}
