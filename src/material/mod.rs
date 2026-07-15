//! Material models and optical response functions.
//!
//! Materials are evaluated pointwise. Optional vectorised evaluation over
//! `ndarray` inputs is provided by adapter methods rather than by the core
//! material trait.

pub mod builder;
pub mod enums;
pub mod model;
pub mod sample;
pub mod tensor;

pub use builder::DrudeLorentzBuilder;
pub use enums::IsotropicMaterial;
pub use model::{Constant, DrudeLorentz};
pub use sample::Scalar;

pub use sample::{Sampled, TensorSampled};
use tensor::{DiagonalTensorMaterial, TensorMaterial};

use crate::ComplexScalar;

use num_traits::{One, Zero};

/// Differentiation variable for material response functions.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SpectralVariable {
    /// Differentiate with respect to `k0`.
    VacuumWavenumber,

    /// Differentiate with respect to `k0²`.
    VacuumWavenumberSquared,
}

/// Highest derivative order requested.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum DerivativeOrder {
    First,
    Second,
}

/// Pointwise optical material model.
///
/// The input variable is currently named `frequency`, but in your existing code
/// this is numerically a wavenumber in cm⁻¹. I would eventually make that a
/// newtype, but this keeps the migration small.
pub trait Material {
    type Real: One + Zero;

    fn is_dispersive(&self) -> bool;

    fn static_permittivity(&self) -> Self::Real;

    fn refractive_index<I, C>(&self, wavenumber: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: Sampled<Elem = C>;

    fn relative_permittivity<I, C>(&self, wavenumber: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: Sampled<Elem = C>;

    fn relative_permittivity_derivative<I, C>(
        &self,
        wavenumber: I,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: Sampled<Elem = C>;

    fn relative_permeability<I, C>(&self, wavenumber: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: Sampled<Elem = C>,
    {
        wavenumber.map(|_| C::from_real(Self::Real::one()))
    }

    fn relative_permeability_derivative<I, C>(
        &self,
        wavenumber: I,
        _order: DerivativeOrder,
        _variable: SpectralVariable,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: Sampled<Elem = C>,
    {
        wavenumber.map(|_| C::from_real(Self::Real::zero()))
    }
}
