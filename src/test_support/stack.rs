use ndarray::{Array, Ix0};

use crate::{
    Constant,
    algebra::Jet0,
    input::canonical::{CanonicalLayer, CanonicalStack},
    test_support::jet::zero_jet_from_real_value,
};

use super::{C, jet::P};

type TestAlgebra = Array<C, Ix0>;
type TestMaterial = Constant<f64>;

pub fn test_material(relative_permittivity: f64) -> TestMaterial {
    TestMaterial::dielectric(relative_permittivity)
}

pub fn empty_stack() -> CanonicalStack<TestMaterial, Jet0<TestAlgebra, P>> {
    CanonicalStack::new(test_material(1.0), test_material(2.25), Vec::new())
}

pub fn single_layer_stack() -> CanonicalStack<TestMaterial, Jet0<TestAlgebra, P>> {
    CanonicalStack::new(
        test_material(1.0),
        test_material(2.25),
        vec![CanonicalLayer::new(
            test_material(4.0),
            zero_jet_from_real_value(0.1),
        )],
    )
}

pub fn two_layer_stack() -> CanonicalStack<TestMaterial, Jet0<TestAlgebra, P>> {
    CanonicalStack::new(
        test_material(1.0),
        test_material(2.25),
        vec![
            CanonicalLayer::new(test_material(4.0), zero_jet_from_real_value(0.1)),
            CanonicalLayer::new(test_material(6.25), zero_jet_from_real_value(0.2)),
        ],
    )
}

pub fn stack_with_layers(layer_count: usize) -> CanonicalStack<TestMaterial, Jet0<TestAlgebra, P>> {
    let layers = (0..layer_count)
        .map(|index| {
            let index = index as f64;

            CanonicalLayer::new(
                test_material(2.0 + index),
                zero_jet_from_real_value(0.05 * (index + 1.0)),
            )
        })
        .collect();

    CanonicalStack::new(test_material(1.0), test_material(2.25), layers)
}
