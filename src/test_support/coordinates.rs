use crate::input::CanonicalCoordinates;
use crate::{algebra::Jet0, test_support::jet::zero_jet_from_real_value};

use super::{C, TestAlgebra, jet::P};

use ndarray::arr0;

pub fn test_coordinates() -> CanonicalCoordinates<Jet0<TestAlgebra, P>> {
    CanonicalCoordinates::new(
        // k₀
        zero_jet_from_real_value(2.0),
        // k∥
        zero_jet_from_real_value(0.3),
    )
}
