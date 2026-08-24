use lamina_units::Length;
use ndarray::{Array, Ix0, arr0};
use num_complex::Complex64;

use crate::{
    AnalyticalMaterialStack, Constant, Stack,
    algebra::Jet0,
    input::canonical::{CanonicalLayer, CanonicalStack},
    test_support::{
        jet::{RealJ0, real_j0_from_real},
        material_model::{Drude, magnetic_loss::MagneticDrudeLorentz},
    },
};

use super::{C, jet::P};

type TestAlgebra = Array<C, Ix0>;
type TestMaterial = Constant<f64>;

pub fn test_material(relative_permittivity: f64) -> TestMaterial {
    TestMaterial::dielectric(relative_permittivity)
}

pub fn canonical_empty_stack() -> CanonicalStack<TestMaterial, Jet0<TestAlgebra, P>> {
    CanonicalStack::new(test_material(1.0), test_material(2.25), Vec::new())
}

pub fn canonical_single_layer_stack() -> CanonicalStack<TestMaterial, Jet0<TestAlgebra, P>> {
    CanonicalStack::new(
        test_material(1.0),
        test_material(2.25),
        vec![CanonicalLayer::new(
            test_material(4.0),
            real_j0_from_real(0.1),
        )],
    )
}

pub fn canonical_two_layer_stack() -> CanonicalStack<TestMaterial, Jet0<TestAlgebra, P>> {
    CanonicalStack::new(
        test_material(1.0),
        test_material(2.25),
        vec![
            CanonicalLayer::new(test_material(4.0), real_j0_from_real(0.1)),
            CanonicalLayer::new(test_material(6.25), real_j0_from_real(0.2)),
        ],
    )
}

pub fn canonical_stack_with_layers(
    layer_count: usize,
) -> CanonicalStack<TestMaterial, Jet0<TestAlgebra, P>> {
    let layers = (0..layer_count)
        .map(|index| {
            let index = index as f64;

            CanonicalLayer::new(
                test_material(2.0 + index),
                real_j0_from_real(0.05 * (index + 1.0)),
            )
        })
        .collect();

    CanonicalStack::new(test_material(1.0), test_material(2.25), layers)
}

pub fn absorbing_single_layer_stack() -> AnalyticalMaterialStack<C> {
    Stack::from_analytical_materials(Constant::vacuum(), Constant::vacuum())
        .analytical_layer(
            Drude::new(1.0, 15000.0, 20.0).unwrap(),
            Length::micrometres(100.0),
        )
        .finalise()
}

pub fn absorbing_two_layer_stack() -> AnalyticalMaterialStack<C> {
    Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(1.7))
        .analytical_layer(
            Drude::new(1.0, 15000.0, 20.0).unwrap(),
            Length::micrometres(0.35),
        )
        .analytical_layer(
            Drude::new(1.0, 30000.0, 30.0).unwrap(),
            Length::micrometres(0.7),
        )
        .finalise()
}

pub fn two_layer_stack_with_lossless_first_layer() -> AnalyticalMaterialStack<C> {
    Stack::from_analytical_materials(Constant::vacuum(), Constant::dielectric(1.7))
        .analytical_layer(Constant::dielectric(2.0), Length::micrometres(0.35))
        .analytical_layer(
            Drude::new(1.0, 30000.0, 30.0).unwrap(),
            Length::micrometres(0.7),
        )
        .finalise()
}

pub fn electric_loss_stack() -> AnalyticalMaterialStack<C> {
    Stack::from_analytical_materials(Constant::dielectric(1.0), Constant::dielectric(2.0))
        .analytical_layer(
            Drude::new(1.0, 15000.0, 20.0).unwrap(),
            Length::micrometres(100.0),
        )
        .finalise()
}

pub fn magnetic_loss_stack() -> AnalyticalMaterialStack<C> {
    Stack::from_analytical_materials(Constant::dielectric(1.0), Constant::dielectric(2.0))
        .analytical_layer(
            MagneticDrudeLorentz::new(1.0, 15000.0, 20.0, vec![]).unwrap(),
            Length::micrometres(100.0),
        )
        .finalise()
}

pub fn differentiable_lossless_two_layer_stack() -> AnalyticalMaterialStack<C> {
    Stack::from_analytical_materials(Constant::dielectric(1.0), Constant::dielectric(2.0))
        .analytical_layer(
            Drude::new(1.0, 15000.0, 0.0).unwrap(),
            Length::micrometres(100.0),
        )
        .analytical_layer(
            Drude::new(2.0, 5000.0, 0.0).unwrap(),
            Length::micrometres(100.0),
        )
        .finalise()
}

pub(crate) type BoundaryTestJet = RealJ0;

pub(crate) fn boundary_test_jet(value: Complex64) -> BoundaryTestJet {
    Jet0::new(arr0(value))
}

/// An asymmetric one-layer stack.
///
/// Asymmetric exteriors and nonzero parallel wavenumber make sign, incidence,
/// and exterior-normalisation errors visible.
pub(crate) fn boundary_test_single_layer_stack() -> CanonicalStack<Constant<f64>, BoundaryTestJet> {
    CanonicalStack::new(
        Constant::new(1.0, 1.0),
        Constant::new(2.25, 1.0),
        vec![CanonicalLayer::new(
            Constant::new(3.24, 1.0),
            boundary_test_jet(Complex64::new(0.23, 0.0)),
        )],
    )
}

/// An asymmetric two-layer stack.
///
/// The layers use different admittances and thicknesses, so reversing physical
/// order or confusing adjacent layer bases should fail visibly.
pub(crate) fn boundary_test_two_layer_stack() -> CanonicalStack<Constant<f64>, BoundaryTestJet> {
    CanonicalStack::new(
        Constant::new(1.0, 1.0),
        Constant::new(2.56, 1.0),
        vec![
            CanonicalLayer::new(
                Constant::new(2.25, 1.0),
                boundary_test_jet(Complex64::new(0.17, 0.0)),
            ),
            CanonicalLayer::new(
                Constant::new(4.0, 1.0),
                boundary_test_jet(Complex64::new(0.29, 0.0)),
            ),
        ],
    )
}

/// A zero-thickness layer for exact boundary-propagation invariants.
pub(crate) fn boundary_test_zero_thickness_stack() -> CanonicalStack<Constant<f64>, BoundaryTestJet>
{
    CanonicalStack::new(
        Constant::new(1.0, 1.0),
        Constant::new(2.25, 1.0),
        vec![CanonicalLayer::new(
            Constant::new(3.24, 1.0),
            boundary_test_jet(Complex64::new(0.0, 0.0)),
        )],
    )
}

/// Empty stack for retained-container edge cases.
pub(crate) fn boundary_test_empty_stack() -> CanonicalStack<Constant<f64>, BoundaryTestJet> {
    CanonicalStack::new(
        Constant::new(1.0, 1.0),
        Constant::new(2.25, 1.0),
        Vec::new(),
    )
}
