//! Shared planar-system fixtures for evaluator tests.

use lamina_units::{InverseLengthUnit, Length};
use ndarray::{Ix0, arr0};
use num_complex::Complex64;

use crate::{
    Constant, CoordinateInput, Coordinates, InPlaneCoordinate, Polarisation, SpectralCoordinate,
    Stack,
    algebra::{ArrayJet0, Jet0, RealParameter},
    input::canonical::{CanonicalCoordinates, CanonicalLayer, CanonicalStack},
    stack::Layer,
};

pub type C = Complex64;
pub type R = f64;

/// Scalar vacuum angular wavenumber used by most evaluator tests.
pub const K0: R = 2.0;

/// Scalar conserved parallel angular wavenumber.
pub const K_PARALLEL: R = 0.0;

/// Thickness of the finite-layer fixture, in centimetres.
pub const FILM_THICKNESS_CM: R = 0.125;

/// Construct a purely real complex scalar.
pub fn c(value: R) -> C {
    C::new(value, 0.0)
}

/// Vacuum.
pub fn vacuum() -> Constant<R> {
    Constant::new(1.0, 1.0)
}

/// A nondispersive dielectric with refractive index `n`.
///
/// Relative permeability is unity and relative permittivity is `n²`.
pub fn dielectric(n: R) -> Constant<R> {
    Constant::new(n * n, 1.0)
}

/// Vacuum-to-dielectric interface with no finite layers.
pub fn dielectric_interface(right_index: R) -> Stack<Constant<R>, R> {
    Stack::new(vacuum(), Vec::new(), dielectric(right_index))
}

/// Dielectric slab between vacuum exterior half-spaces.
pub fn single_layer_stack(film_index: R, thickness_cm: R) -> Stack<Constant<R>, R> {
    Stack::new(
        vacuum(),
        vec![Layer::new(
            dielectric(film_index),
            Length::centimetres(thickness_cm),
        )],
        vacuum(),
    )
}

/// Two finite layers, useful for checking geometric layer indexing.
pub fn two_layer_stack() -> Stack<Constant<R>, R> {
    Stack::new(
        vacuum(),
        vec![
            Layer::new(dielectric(1.5), Length::centimetres(0.10)),
            Layer::new(dielectric(2.0), Length::centimetres(0.20)),
        ],
        dielectric(1.25),
    )
}

/// Scalar real-coordinate input using canonical angular wavenumbers.
///
/// Rename the constructors here if your public input API differs.
pub fn scalar_real_input(k0: R, k_parallel: R) -> CoordinateInput<R, ndarray::Ix0> {
    CoordinateInput::point(
        Coordinates::new(
            SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
            InPlaneCoordinate::ParallelAngularWavenumber(InverseLengthUnit::PerCentimetre),
        ),
        k0,
        k_parallel,
    )
    .unwrap()
}

pub fn sampled_real_input(k0: &'_ [R], k_parallel: &'_ [R]) -> CoordinateInput<R, ndarray::Ix1> {
    CoordinateInput::samples(
        Coordinates::new(
            SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
            InPlaneCoordinate::ParallelAngularWavenumber(InverseLengthUnit::PerCentimetre),
        ),
        ndarray::arr1(k0),
        ndarray::arr1(k_parallel),
    )
    .unwrap()
}

/// Scalar complex-coordinate modal input.
pub fn scalar_complex_input(k0: C, k_parallel: C) -> CoordinateInput<C, ndarray::Ix0> {
    CoordinateInput::point(
        Coordinates::new(
            SpectralCoordinate::VacuumAngularWavenumber(InverseLengthUnit::PerCentimetre),
            InPlaneCoordinate::ParallelAngularWavenumber(InverseLengthUnit::PerCentimetre),
        ),
        k0,
        k_parallel,
    )
    .unwrap()
}

/// Fresnel amplitudes at normal incidence for the scattering convention used
/// by the isotropic backend.
///
/// The characteristic admittance is:
///
/// - `Y = n` for TE when `μ = 1`;
/// - `Y = 1 / n` for TM when `ε = n²`.
pub fn fresnel_amplitudes(left_index: R, right_index: R, polarisation: Polarisation) -> (C, C) {
    let left_admittance = match polarisation {
        Polarisation::TransverseElectric => left_index,
        Polarisation::TransverseMagnetic => 1.0 / left_index,
    };

    let right_admittance = match polarisation {
        Polarisation::TransverseElectric => right_index,
        Polarisation::TransverseMagnetic => 1.0 / right_index,
    };

    let denominator = left_admittance + right_admittance;

    let reflection = (left_admittance - right_admittance) / denominator;

    let transmission = 2.0 * left_admittance / denominator;

    (c(reflection), c(transmission))
}

/// Power coefficients from the backend's amplitude normalisation.
pub fn fresnel_power(left_index: R, right_index: R, polarisation: Polarisation) -> (R, R, R) {
    let (reflection, transmission) = fresnel_amplitudes(left_index, right_index, polarisation);

    let left_admittance = match polarisation {
        Polarisation::TransverseElectric => left_index,
        Polarisation::TransverseMagnetic => 1.0 / left_index,
    };

    let right_admittance = match polarisation {
        Polarisation::TransverseElectric => right_index,
        Polarisation::TransverseMagnetic => 1.0 / right_index,
    };

    let reflectance = reflection.norm_sqr();

    let transmittance = transmission.norm_sqr() * right_admittance / left_admittance;

    let absorptance = 1.0 - reflectance - transmittance;

    (reflectance, transmittance, absorptance)
}

pub(crate) type BoundaryTestScalar = Complex64;
pub(crate) type BoundaryTestJet = ArrayJet0<Complex64, Ix0, RealParameter>;

pub(crate) fn boundary_test_jet(value: Complex64) -> BoundaryTestJet {
    Jet0::new(arr0(value))
}

pub(crate) fn boundary_test_coordinates() -> CanonicalCoordinates<BoundaryTestJet> {
    CanonicalCoordinates::new(
        boundary_test_jet(Complex64::new(2.3, 0.0)),
        boundary_test_jet(Complex64::new(0.37, 0.0)),
    )
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
