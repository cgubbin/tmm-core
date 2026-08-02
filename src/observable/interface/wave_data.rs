use ndarray::Dimension;

use crate::{
    ComplexScalar,
    algebra::ScalarAlgebra,
    observable::{BoundaryState, BoundaryWaves},
};

/// Directional and canonical data immediately on one side of an interface.
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

    pub(crate) fn state(&self) -> BoundaryState<A>
    where
        A: ScalarAlgebra + Clone,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        self.waves.clone().into_state(&self.admittance)
    }

    pub(crate) fn into_parts(self) -> (BoundaryWaves<A>, A) {
        (self.waves, self.admittance)
    }

    pub(crate) fn into_waves_state_admittance(self) -> (BoundaryWaves<A>, BoundaryState<A>, A)
    where
        A: ScalarAlgebra + Clone,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        let state = self.waves.clone().into_state(&self.admittance);

        (self.waves, state, self.admittance)
    }
}

/// Directional waves, canonical states, and medium admittances immediately on
/// both sides of one planar interface.
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
    use crate::observable::{BoundaryState, BoundaryWaves};

    #[test]
    fn interface_side_stores_waves_state_and_admittance() {
        let waves = BoundaryWaves::new(1, 2);

        let side = InterfaceSide::new(waves.clone(), 5);

        assert_eq!(side.waves(), &waves);
        assert_eq!(side.admittance(), &5);
    }

    #[test]
    fn interface_side_into_parts_preserves_component_order() {
        let side = InterfaceSide::new(BoundaryWaves::new(1, 2), 5);

        let (waves, admittance) = side.into_parts();

        assert_eq!(waves.into_parts(), (1, 2));
        assert_eq!(admittance, 5);
    }

    #[test]
    fn interface_wave_data_stores_both_sides() {
        let left = InterfaceSide::new(BoundaryWaves::new(1, 2), 5);

        let right = InterfaceSide::new(BoundaryWaves::new(6, 7), 10);

        let interface = InterfaceWaveData::new(left.clone(), right.clone());

        assert_eq!(interface.left(), &left);
        assert_eq!(interface.right(), &right);
    }

    #[test]
    fn interface_wave_data_into_parts_preserves_side_order() {
        let interface = InterfaceWaveData::new(
            InterfaceSide::new(BoundaryWaves::new(1, 2), 5),
            InterfaceSide::new(BoundaryWaves::new(6, 7), 10),
        );

        let (left, right) = interface.into_parts();

        let (left_waves, left_admittance) = left.into_parts();

        let (right_waves, right_admittance) = right.into_parts();

        assert_eq!(left_waves.into_parts(), (1, 2));
        assert_eq!(left_admittance, 5);

        assert_eq!(right_waves.into_parts(), (6, 7));
        assert_eq!(right_admittance, 10);
    }

    #[test]
    fn interface_wave_data_supports_non_clone_storage() {
        #[derive(Debug, PartialEq)]
        struct NonClone(i32);

        let interface = InterfaceWaveData::new(
            InterfaceSide::new(BoundaryWaves::new(NonClone(1), NonClone(2)), NonClone(5)),
            InterfaceSide::new(BoundaryWaves::new(NonClone(6), NonClone(7)), NonClone(10)),
        );

        let (left, right) = interface.into_parts();

        let (left_waves, left_admittance) = left.into_parts();

        let (right_waves, right_admittance) = right.into_parts();

        assert_eq!(left_waves.forward(), &NonClone(1));
        assert_eq!(left_waves.backward(), &NonClone(2));
        assert_eq!(left_admittance, NonClone(5));

        assert_eq!(right_waves.forward(), &NonClone(6));
        assert_eq!(right_waves.backward(), &NonClone(7));
        assert_eq!(right_admittance, NonClone(10));
    }
}
