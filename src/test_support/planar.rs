//! Shared planar-system fixtures for evaluator tests.

use lamina_units::{InverseLengthUnit, Length};
use ndarray::Dimension;
use num_complex::Complex64;

use crate::{
    ComplexPlane, ComplexScalar, Constant, ConstitutiveEvaluator, ConstitutiveLift,
    CoordinateInput, Coordinates, ExteriorWavevectors, InPlaneCoordinate, Polarisation,
    ScalarAlgebra, SpectralCoordinate, Stack,
    backend::evaluate_exterior_wavevectors,
    input::canonical::{CanonicalCoordinates, CanonicalStack},
    stack::Layer,
    test_support::jet::{HoloJ0, RealJ0, holo_j0, real_j0},
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

pub fn real_canonical_coordinates(k0: f64, k_parallel: f64) -> CanonicalCoordinates<RealJ0> {
    CanonicalCoordinates::new(real_j0(c(k0)), real_j0(c(k_parallel)))
}

pub fn canonical_complex_coordinates(k0: C, k_parallel: C) -> CanonicalCoordinates<HoloJ0> {
    CanonicalCoordinates::new(holo_j0(k0), holo_j0(k_parallel))
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

pub fn principal_exterior_wavevectors<M, J>(
    stack: &CanonicalStack<M, J>,
    coordinates: &CanonicalCoordinates<J>,
) -> ExteriorWavevectors<J>
where
    J: ScalarAlgebra + ConstitutiveLift<ComplexPlane, M> + Clone,
    J::Scalar: ComplexScalar,
    J::Dimension: Dimension,
    ComplexPlane: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
{
    evaluate_exterior_wavevectors::<ComplexPlane, _, _>(
        coordinates,
        stack.left_exterior(),
        stack.right_exterior(),
    )
}
