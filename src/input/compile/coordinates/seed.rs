use ndarray::{Array, Dimension};

use crate::input::compile::{
    assignment::CoordinateAssignment,
    coordinates::CoordinateVariable,
    seed::{SeedJet, UnsupportedDerivativeSlot},
};

pub(crate) fn seed_coordinate<R, D, J>(
    values: Array<R, D>,
    slot: Option<usize>,
) -> Result<J, UnsupportedDerivativeSlot>
where
    D: Dimension,
    J: SeedJet<Array<R, D>>,
{
    match slot {
        Some(slot) => J::variable(values, slot),

        None => Ok(J::constant(values)),
    }
}
