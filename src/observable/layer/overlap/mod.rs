mod bilinear;
mod hermitian;

pub use bilinear::{
    AggregateBilinearOverlap, BilinearLayerOverlapInput, NormalizedBilinearOverlap,
};

pub use hermitian::{
    AggregateHermitianOverlap, HermitianLayerOverlapInput, NormalizedHermitianOverlap,
};

pub(crate) use bilinear::BilinearLayerOverlap;
pub(crate) use hermitian::HermitianLayerOverlap;

use crate::{backend::IsotropicLayerQuantities, observable::BoundaryWaves};

/// Boundary waves and isotropic medium quantities for one overlap operand.
///
/// Waves are expressed at the physical layer's left boundary.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayerOverlapOperand<A> {
    waves: BoundaryWaves<A>,
    quantities: IsotropicLayerQuantities<A>,
}

impl<A> LayerOverlapOperand<A> {
    pub(crate) const fn new(
        waves: BoundaryWaves<A>,
        quantities: IsotropicLayerQuantities<A>,
    ) -> Self {
        Self { waves, quantities }
    }

    fn into_parts(self) -> (BoundaryWaves<A>, IsotropicLayerQuantities<A>) {
        (self.waves, self.quantities)
    }
}
