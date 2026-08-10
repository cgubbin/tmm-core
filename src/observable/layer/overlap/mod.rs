mod bilinear;
mod error;
mod hermitian;

pub use bilinear::{
    AggregateBilinearNormalization, AggregateBilinearOverlap, BilinearLayerNormalization,
};

pub use hermitian::AggregateHermitianOverlap;

pub(crate) use bilinear::{BilinearLayerOverlap, BilinearLayerOverlapInput};
pub(crate) use error::OverlapError;
pub(crate) use hermitian::{HermitianLayerOverlap, HermitianLayerOverlapInput};

use crate::{backend::IsotropicLayerQuantities, observable::BoundaryWaves};

/// Matched left and right solution data for one physical finite layer.
///
/// The two operands must refer to the same physical layer. `thickness` is the
/// common physical integration interval.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayerOverlapInput<A> {
    left: LayerOverlapOperand<A>,
    right: LayerOverlapOperand<A>,
    thickness: A,
}

impl<A> LayerOverlapInput<A> {
    pub(crate) const fn new(
        left: LayerOverlapOperand<A>,
        right: LayerOverlapOperand<A>,
        thickness: A,
    ) -> Self {
        Self {
            left,
            right,
            thickness,
        }
    }

    fn into_parts(self) -> (LayerOverlapOperand<A>, LayerOverlapOperand<A>, A) {
        (self.left, self.right, self.thickness)
    }

    pub(crate) fn into_hermitian(self) -> HermitianLayerOverlapInput<A> {
        HermitianLayerOverlapInput::new(self.left, self.right, self.thickness)
    }

    pub(crate) fn into_bilinear(self) -> BilinearLayerOverlapInput<A> {
        BilinearLayerOverlapInput::new(self.left, self.right, self.thickness)
    }
}

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
