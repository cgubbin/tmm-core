//! Internal directional-wave data used by interface projections.

use ndarray::Dimension;

use crate::{
    ComplexScalar,
    algebra::ScalarAlgebra,
    observable::{BoundaryState, BoundaryWaves},
};

/// Directional waves in the two exterior media.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ExteriorBoundaryWaves<A> {
    left: BoundaryWaves<A>,
    right: BoundaryWaves<A>,
}

impl<A> ExteriorBoundaryWaves<A> {
    pub(crate) const fn new(left: BoundaryWaves<A>, right: BoundaryWaves<A>) -> Self {
        Self { left, right }
    }

    pub(crate) fn left(&self) -> &BoundaryWaves<A> {
        &self.left
    }

    pub(crate) fn right(&self) -> &BoundaryWaves<A> {
        &self.right
    }

    pub(crate) fn into_parts(self) -> (BoundaryWaves<A>, BoundaryWaves<A>) {
        (self.left, self.right)
    }
}

/// Directional waves and characteristic admittance immediately on one side
/// of an interface.
///
/// The canonical boundary state is derived from these two quantities:
///
/// ```text
/// field     = forward + backward
/// secondary = -i Y (backward - forward)
/// ```
///
/// The state is not stored separately, avoiding duplicated representations
/// that could become inconsistent.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct InterfaceSide<A> {
    waves: BoundaryWaves<A>,
    admittance: A,
}

impl<A> InterfaceSide<A> {
    pub(crate) const fn new(waves: BoundaryWaves<A>, admittance: A) -> Self {
        Self { waves, admittance }
    }

    pub(crate) fn waves(&self) -> &BoundaryWaves<A> {
        &self.waves
    }

    pub(crate) fn admittance(&self) -> &A {
        &self.admittance
    }

    /// Derive the canonical state without consuming this side.
    ///
    /// This is primarily useful for diagnostics. Consuming projections should
    /// prefer [`Self::into_state`] to avoid cloning the directional waves.
    pub(crate) fn state(&self) -> BoundaryState<A>
    where
        A: ScalarAlgebra + Clone,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        self.waves.clone().into_state(&self.admittance)
    }

    /// Consume this side and derive its canonical state.
    pub(crate) fn into_state(self) -> BoundaryState<A>
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        self.waves.into_state(&self.admittance)
    }

    pub(crate) fn into_parts(self) -> (BoundaryWaves<A>, A) {
        (self.waves, self.admittance)
    }
}

/// Directional waves and characteristic admittances immediately on both
/// sides of one planar interface.
///
/// This is an internal projection record. Canonical states and normalized
/// power flux are derived from it by the corresponding observable projection.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct InterfaceWaveData<A> {
    left: InterfaceSide<A>,
    right: InterfaceSide<A>,
}

impl<A> InterfaceWaveData<A> {
    pub(crate) const fn new(left: InterfaceSide<A>, right: InterfaceSide<A>) -> Self {
        Self { left, right }
    }

    pub(crate) fn left(&self) -> &InterfaceSide<A> {
        &self.left
    }

    pub(crate) fn right(&self) -> &InterfaceSide<A> {
        &self.right
    }

    pub(crate) fn into_parts(self) -> (InterfaceSide<A>, InterfaceSide<A>) {
        (self.left, self.right)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::observable::BoundaryWaves;

    #[test]
    fn interface_side_stores_waves_and_admittance() {
        let waves = BoundaryWaves::new(1, 2);
        let side = InterfaceSide::new(waves.clone(), 5);

        assert_eq!(side.waves(), &waves);
        assert_eq!(side.admittance(), &5);
    }

    #[test]
    fn interface_side_into_parts_preserves_order() {
        let side = InterfaceSide::new(BoundaryWaves::new(1, 2), 5);

        let (waves, admittance) = side.into_parts();

        assert_eq!(waves.into_parts(), (1, 2));
        assert_eq!(admittance, 5);
    }

    #[test]
    fn interface_wave_data_preserves_side_order() {
        let interface = InterfaceWaveData::new(
            InterfaceSide::new(BoundaryWaves::new(1, 2), 5),
            InterfaceSide::new(BoundaryWaves::new(6, 7), 10),
        );

        let (left, right) = interface.into_parts();

        assert_eq!(left.into_parts().0.into_parts(), (1, 2),);

        assert_eq!(right.into_parts().0.into_parts(), (6, 7),);
    }

    #[test]
    fn consuming_projection_supports_non_clone_storage() {
        #[derive(Debug, PartialEq)]
        struct NonClone(i32);

        let interface = InterfaceWaveData::new(
            InterfaceSide::new(BoundaryWaves::new(NonClone(1), NonClone(2)), NonClone(5)),
            InterfaceSide::new(BoundaryWaves::new(NonClone(6), NonClone(7)), NonClone(10)),
        );

        let (left, right) = interface.into_parts();

        let (left_waves, left_admittance) = left.into_parts();

        let (right_waves, right_admittance) = right.into_parts();

        assert_eq!(left_waves.forward(), &NonClone(1),);
        assert_eq!(left_admittance, NonClone(5),);

        assert_eq!(right_waves.backward(), &NonClone(7),);
        assert_eq!(right_admittance, NonClone(10),);
    }
}
