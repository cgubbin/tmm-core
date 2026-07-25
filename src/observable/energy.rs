use crate::{
    SpatialProfile, SpatialProfileError,
    field::{ScalarField, ScalarFieldView1},
    observable::components::ElectromagneticComponents,
};

use ndarray::Dimension;

/// Time-averaged electromagnetic energy density.
///
/// The components are evaluated according to the energy convention documented
/// by the operation that produced this value. `total` is the sum of the
/// electric and magnetic contributions.
///
/// Values inherit the field normalisation of the underlying solution.
#[derive(Clone, Debug, PartialEq)]
pub struct EnergyDensity<R> {
    components: ElectromagneticComponents<R>,
}

impl<R, D> SpatialProfile<D::Smaller> for EnergyDensity<ScalarField<R, D>>
where
    D: Dimension,
    D::Smaller: Dimension<Larger = D>,
{
    type Profile<'a>
        = EnergyDensity<ScalarFieldView1<'a, R>>
    where
        Self: 'a;

    fn spatial_profile(
        &self,
        excitation_index: &D::Smaller,
    ) -> Result<Self::Profile<'_>, SpatialProfileError> {
        Ok(EnergyDensity {
            components: self.components.spatial_profile(excitation_index)?,
        })
    }
}

impl<R> EnergyDensity<R> {
    pub(crate) fn new(electric: R, magnetic: R, coupling: R, total: R) -> Self {
        Self {
            components: ElectromagneticComponents::new(electric, magnetic, coupling, total),
        }
    }

    /// Return the electric contribution to the energy density.
    pub fn electric(&self) -> &R {
        self.components.electric()
    }

    /// Return the magnetic contribution to the energy density.
    pub fn magnetic(&self) -> &R {
        self.components.magnetic()
    }

    /// Return the coupling contribution to the energy density.
    pub fn coupling(&self) -> &R {
        self.components.coupling()
    }

    /// Return the total electromagnetic energy density.
    pub fn total(&self) -> &R {
        self.components.total()
    }

    /// Consume the value and return its electric, magnetic, coupling and total
    /// components.
    pub fn into_parts(self) -> (R, R, R, R) {
        self.components.into_parts()
    }

    /// Transform the storage of every component.
    pub fn map<U>(self, f: impl FnMut(R) -> U) -> EnergyDensity<U> {
        EnergyDensity {
            components: self.components.map(f),
        }
    }
}

/// Time-averaged electromagnetic energy stored in a spatial region.
///
/// The components are spatial integrals of the corresponding energy-density
/// contributions under the convention documented by the producing operation.
/// `total` is the sum of the electric and magnetic contributions.
///
/// Values inherit the field normalisation of the underlying solution.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredEnergy<R> {
    components: ElectromagneticComponents<R>,
}

impl<R, D> SpatialProfile<D::Smaller> for StoredEnergy<ScalarField<R, D>>
where
    D: Dimension,
    D::Smaller: Dimension<Larger = D>,
{
    type Profile<'a>
        = StoredEnergy<ScalarFieldView1<'a, R>>
    where
        Self: 'a;

    fn spatial_profile(
        &self,
        excitation_index: &D::Smaller,
    ) -> Result<Self::Profile<'_>, SpatialProfileError> {
        Ok(StoredEnergy {
            components: self.components.spatial_profile(excitation_index)?,
        })
    }
}

impl<R> StoredEnergy<R> {
    pub(crate) fn new(electric: R, magnetic: R, coupling: R, total: R) -> Self {
        Self {
            components: ElectromagneticComponents::new(electric, magnetic, coupling, total),
        }
    }

    /// Return the stored electric energy.
    pub fn electric(&self) -> &R {
        self.components.electric()
    }

    /// Return the stored magnetic energy.
    pub fn magnetic(&self) -> &R {
        self.components.magnetic()
    }

    /// Return the stored coupling energy
    pub fn coupling(&self) -> &R {
        self.components.coupling()
    }

    /// Return the total stored electromagnetic energy.
    pub fn total(&self) -> &R {
        self.components.total()
    }

    /// Consume the value and return its electric, magnetic, coupling and total
    /// components.
    pub fn into_parts(self) -> (R, R, R, R) {
        self.components.into_parts()
    }

    /// Transform the storage of every component.
    pub fn map<U>(self, f: impl FnMut(R) -> U) -> StoredEnergy<U> {
        StoredEnergy {
            components: self.components.map(f),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::field::ScalarField;

    use super::{EnergyDensity, SpatialProfile, StoredEnergy};

    use ndarray::{Array2, Ix1, arr1};

    #[test]
    fn energy_density_stores_all_components() {
        let energy = EnergyDensity::new(1, 2, 3, 4);

        assert_eq!(energy.electric(), &1);
        assert_eq!(energy.magnetic(), &2);
        assert_eq!(energy.coupling(), &3);
        assert_eq!(energy.total(), &4);
    }

    #[test]
    fn energy_density_into_parts_preserves_order() {
        let energy = EnergyDensity::new(1, 2, 3, 4);

        assert_eq!(energy.into_parts(), (1, 2, 3, 4));
    }

    #[test]
    fn energy_density_map_transforms_all_components() {
        let energy = EnergyDensity::new(1, 2, 3, 4);

        let mapped = energy.map(|value| format!("density-{value}"));

        assert_eq!(mapped.electric(), "density-1");
        assert_eq!(mapped.magnetic(), "density-2");
        assert_eq!(mapped.coupling(), "density-3");
        assert_eq!(mapped.total(), "density-4");
    }

    #[test]
    fn stored_energy_stores_all_components() {
        let energy = StoredEnergy::new(1, 2, 3, 4);

        assert_eq!(energy.electric(), &1);
        assert_eq!(energy.magnetic(), &2);
        assert_eq!(energy.coupling(), &3);
        assert_eq!(energy.total(), &4);
    }

    #[test]
    fn stored_energy_into_parts_preserves_order() {
        let energy = StoredEnergy::new(1, 2, 3, 4);

        assert_eq!(energy.into_parts(), (1, 2, 3, 4));
    }

    #[test]
    fn stored_energy_map_transforms_all_components() {
        let energy = StoredEnergy::new(1, 2, 3, 4);

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

        let density = EnergyDensity::new(NonClone(1), NonClone(2), NonClone(3), NonClone(4));

        let stored = StoredEnergy::new(NonClone(5), NonClone(6), NonClone(7), NonClone(8));

        let density = density.map(|value| value.0 * 10);
        let stored = stored.map(|value| value.0 * 10);

        assert_eq!(density.into_parts(), (10, 20, 30, 40));
        assert_eq!(stored.into_parts(), (50, 60, 70, 80));
    }

    #[test]
    fn energy_density_profiles_all_terms() {
        let electric = ScalarField::new(Array2::from_shape_fn((2, 3), |(i, k)| {
            10.0 * i as f64 + k as f64
        }));

        let magnetic = ScalarField::new(Array2::from_shape_fn((2, 3), |(i, k)| {
            100.0 + 10.0 * i as f64 + k as f64
        }));

        let coupling = ScalarField::new(Array2::from_shape_fn((2, 3), |(i, k)| {
            100.0 + 10.0 * i as f64 + k as f64
        }));

        let total = ScalarField::new(Array2::from_shape_fn((2, 3), |(i, k)| {
            200.0 + 10.0 * i as f64 + k as f64
        }));

        let density = EnergyDensity::new(electric, magnetic, coupling, total);

        let profile = density
            .spatial_profile(&Ix1(1))
            .expect("profile should succeed");

        assert_eq!(
            profile.electric().values(),
            arr1(&[10.0, 11.0, 12.0]).view(),
        );
        assert_eq!(
            profile.magnetic().values(),
            arr1(&[110.0, 111.0, 112.0]).view(),
        );
        assert_eq!(
            profile.total().values(),
            arr1(&[210.0, 211.0, 212.0]).view(),
        );
    }
}
