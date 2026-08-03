//! Internal data used to construct integrated layer observables.

use crate::{
    backend::IsotropicLayerQuantities,
    observable::{
        BoundaryWaves,
        layer::{IntegratedWaveProducts, state_overlap::IntegratedStateProducts},
    },
};

/// Directional waves and homogeneous-medium quantities for one finite layer.
///
/// Waves are defined at the layer's left boundary. `thickness` is expressed
/// in canonical centimetres.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayerWaveData<A> {
    waves: BoundaryWaves<A>,
    quantities: IsotropicLayerQuantities<A>,
    thickness: A,
}

impl<A> LayerWaveData<A> {
    pub(crate) const fn new(
        waves: BoundaryWaves<A>,
        quantities: IsotropicLayerQuantities<A>,
        thickness: A,
    ) -> Self {
        Self {
            waves,
            quantities,
            thickness,
        }
    }

    pub(crate) fn waves(&self) -> &BoundaryWaves<A> {
        &self.waves
    }

    pub(crate) fn quantities(&self) -> &IsotropicLayerQuantities<A> {
        &self.quantities
    }

    pub(crate) fn thickness(&self) -> &A {
        &self.thickness
    }

    pub(crate) fn into_parts(self) -> (BoundaryWaves<A>, IsotropicLayerQuantities<A>, A) {
        (self.waves, self.quantities, self.thickness)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IntegratedLayerWaveData<A> {
    wave_products: IntegratedWaveProducts<A>,
    state_products: IntegratedStateProducts<A>,
    quantities: IsotropicLayerQuantities<A>,
    thickness: A,
}

impl<A> IntegratedLayerWaveData<A> {
    pub(crate) const fn new(
        wave_products: IntegratedWaveProducts<A>,
        state_products: IntegratedStateProducts<A>,
        quantities: IsotropicLayerQuantities<A>,
        thickness: A,
    ) -> Self {
        Self {
            wave_products,
            state_products,
            quantities,
            thickness,
        }
    }

    pub(crate) fn wave_products(&self) -> &IntegratedWaveProducts<A> {
        &self.wave_products
    }

    pub(crate) fn state_products(&self) -> &IntegratedStateProducts<A> {
        &self.state_products
    }

    pub(crate) fn quantities(&self) -> &IsotropicLayerQuantities<A> {
        &self.quantities
    }

    pub(crate) fn thickness(&self) -> &A {
        &self.thickness
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        IntegratedWaveProducts<A>,
        IntegratedStateProducts<A>,
        IsotropicLayerQuantities<A>,
        A,
    ) {
        (
            self.wave_products,
            self.state_products,
            self.quantities,
            self.thickness,
        )
    }
}
