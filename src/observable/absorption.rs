use crate::observable::components::ElectromagneticComponents;


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
pub struct DissipationDensity<R> {
    components: ElectromagneticComponents<R>,
}

impl<R> DissipationDensity<R> {
    pub(crate) fn new(electric: R, magnetic: R, coupling: R, total: R) -> Self {
        Self {
            components: ElectromagneticComponents::new(electric, magnetic, coupling, total),
        }
    }

    /// Return the electric contribution to the dissipation density.
    pub fn electric(&self) -> &R {
        self.components.electric()
    }

    /// Return the magnetic contribution to the dissipation density.
    pub fn magnetic(&self) -> &R {
        self.components.magnetic()
    }

    /// Return the coupling contribution to the dissipation density.
    pub fn coupling(&self) -> &R {
        self.components.coupling()
    }

    /// Return the total electromagnetic dissipation density.
    pub fn total(&self) -> &R {
        self.components.total()
    }

    /// Consume the value and return its electric, magnetic, coupling and total
    /// components.
    pub fn into_parts(self) -> (R, R, R, R) {
        self.components.into_parts()
    }

    /// Transform the storage of every component.
    pub fn map<U>(self, f: impl FnMut(R) -> U) -> DissipationDensity<U> {
        DissipationDensity {
            components: self.components.map(f),
        }
    }
}

/// Time-averaged electromagnetic power dissipated within one layer.
///
/// The components are spatial integrals of the corresponding dissipation
/// densities through the layer. Positive values indicate absorption and
/// negative values indicate gain.
///
/// In a planar calculation these values are powers per unit transverse area
/// unless the fields have been assigned an additional transverse
/// normalisation.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerDissipation<R> {
    components: ElectromagneticComponents<R>,
}

impl<R> LayerDissipation<R> {
    pub(crate) fn new(electric: R, magnetic: R, coupling: R, total: R) -> Self {
        Self {
            components: ElectromagneticComponents::new(electric, magnetic, coupling, total),
        }
    }

    /// Return the electric layer dissipation.
    pub fn electric(&self) -> &R {
        self.components.electric()
    }

    /// Return the magnetic layer dissipation.
    pub fn magnetic(&self) -> &R {
        self.components.magnetic()
    }

    /// Return the coupling layer dissipation.
    pub fn coupling(&self) -> &R {
        self.components.coupling()
    }

    /// Return the total layer dissipation.
    pub fn total(&self) -> &R {
        self.components.total()
    }

    /// Consume the value and return its electric, magnetic, coupling and total
    /// components.
    pub fn into_parts(self) -> (R, R, R, R) {
        self.components.into_parts()
    }

    /// Transform the storage of every component.
    pub fn map<U>(self, f: impl FnMut(R) -> U) -> LayerDissipation<U> {
        LayerDissipation {
            components: self.components.map(f),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DissipationDensity, LayerDissipation};

    #[test]
    fn energy_density_stores_all_components() {
        let energy = DissipationDensity::new(1, 2, 3, 4);

        assert_eq!(energy.electric(), &1);
        assert_eq!(energy.magnetic(), &2);
        assert_eq!(energy.coupling(), &3);
        assert_eq!(energy.total(), &4);
    }

    #[test]
    fn energy_density_into_parts_preserves_order() {
        let energy = DissipationDensity::new(1, 2, 3, 4);

        assert_eq!(energy.into_parts(), (1, 2, 3, 4));
    }

    #[test]
    fn energy_density_map_transforms_all_components() {
        let energy = DissipationDensity::new(1, 2, 3, 4);

        let mapped = energy.map(|value| format!("density-{value}"));

        assert_eq!(mapped.electric(), "density-1");
        assert_eq!(mapped.magnetic(), "density-2");
        assert_eq!(mapped.coupling(), "density-3");
        assert_eq!(mapped.total(), "density-4");
    }

    #[test]
    fn stored_energy_stores_all_components() {
        let energy = LayerDissipation::new(1, 2, 3, 4);

        assert_eq!(energy.electric(), &1);
        assert_eq!(energy.magnetic(), &2);
        assert_eq!(energy.coupling(), &3);
        assert_eq!(energy.total(), &4);
    }

    #[test]
    fn stored_energy_into_parts_preserves_order() {
        let energy = LayerDissipation::new(1, 2, 3, 4);

        assert_eq!(energy.into_parts(), (1, 2, 3, 4));
    }

    #[test]
    fn stored_energy_map_transforms_all_components() {
        let energy = LayerDissipation::new(1, 2, 3, 4);

        let mapped = energy.map(|value| format!("energy-{value}"));

        assert_eq!(mapped.electric(), "energy-1");
        assert_eq!(mapped.magnetic(), "energy-2");
        assert_eq!(mapped.coupling(), "energy-3");
        assert_eq!(mapped.total(), "energy-4");
    }

    #[test]
    fn maps_consume_non_clone_storage() {
        #[derive(Debug, PartialEq)]
        struct NonClone(i32);

        let density = DissipationDensity::new(NonClone(1), NonClone(2), NonClone(3), NonClone(4));

        let stored = LayerDissipation::new(NonClone(5), NonClone(6), NonClone(7), NonClone(8));

        let density = density.map(|value| value.0 * 10);
        let stored = stored.map(|value| value.0 * 10);

        assert_eq!(density.into_parts(), (10, 20, 30, 40));
        assert_eq!(stored.into_parts(), (50, 60, 70, 80));
    }
}
