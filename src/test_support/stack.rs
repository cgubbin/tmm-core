use ndarray::{Array, Ix0};

use crate::{
    AnalyticalMaterialStack, Constant, MaterialStack, Stack, Thickness,
    algebra::Jet0,
    input::canonical::{CanonicalLayer, CanonicalStack},
    test_support::{jet::zero_jet_from_real_value, material_model::Drude},
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

pub fn absorbing_single_layer_stack() -> AnalyticalMaterialStack<C> {
    Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum())
        .analytical_layer(
            Drude::new(1.0, 15000.0, 20.0).unwrap(),
            Thickness::micrometres(100.0),
        )
        .finalise()
}

pub fn absorbing_two_layer_stack() -> AnalyticalMaterialStack<C> {
    Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(1.7))
        .analytical_layer(
            Drude::new(1.0, 15000.0, 20.0).unwrap(),
            Thickness::micrometres(0.35),
        )
        .analytical_layer(
            Drude::new(1.0, 30000.0, 30.0).unwrap(),
            Thickness::micrometres(0.7),
        )
        .finalise()
}

pub fn two_layer_stack_with_lossless_first_layer() -> AnalyticalMaterialStack<C> {
    Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(1.7))
        .analytical_layer(Constant::dielectric(2.0), Thickness::micrometres(0.35))
        .analytical_layer(
            Drude::new(1.0, 30000.0, 30.0).unwrap(),
            Thickness::micrometres(0.7),
        )
        .finalise()
}
