//! Backend-facing material evaluation traits.
//!
//! The public material model traits allow the caller to select any compatible
//! complex scalar type. That flexibility prevents direct type erasure.
//!
//! These traits fix the complex scalar type at the stack/backend boundary,
//! allowing heterogeneous materials to be stored behind material handles while
//! preserving static dispatch for homogeneous stacks.

use super::{
    ComplexScalar, DerivativeOrder, DifferentiableMaterial, DifferentiableMeromorphicMaterial,
    Material, MeromorphicMaterial, Sampled, SpectralVariable,
};

/// Real-axis evaluation of an isotropic material using a fixed complex scalar.
///
/// Backends should generally depend on this trait rather than directly on
/// [`Material`]. Every concrete [`Material`] implementation receives a blanket
/// implementation.
pub trait EvaluateMaterial<C>
where
    C: ComplexScalar<RealField = Self::Real> + Copy,
{
    /// Real scalar used for the spectral coordinate.
    type Real: Copy;

    /// Evaluate the complex refractive index on the real spectral axis.
    fn evaluate_relative_permeability<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>;

    /// Evaluate relative permittivity on the real spectral axis.
    fn evaluate_relative_permittivity<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>;
}

impl<M, C> EvaluateMaterial<C> for M
where
    M: Material,
    M::Real: Copy,
    C: ComplexScalar<RealField = M::Real> + Copy,
{
    type Real = M::Real;

    fn evaluate_relative_permeability<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
    {
        self.relative_permeability::<I, C>(vacuum_wavenumber)
    }

    fn evaluate_relative_permittivity<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
    {
        self.relative_permittivity::<I, C>(vacuum_wavenumber)
    }
}

/// Real-axis material derivatives using a fixed complex scalar.
pub trait EvaluateDifferentiableMaterial<C>: EvaluateMaterial<C>
where
    C: ComplexScalar<RealField = Self::Real> + Copy,
{
    /// Evaluate a derivative of relative permittivity.
    fn evaluate_relative_permittivity_derivative<I>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>;

    fn evaluate_relative_permeability_derivative<I>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>;
}

impl<M, C> EvaluateDifferentiableMaterial<C> for M
where
    M: DifferentiableMaterial,
    M::Real: Copy,
    C: ComplexScalar<RealField = M::Real> + Copy,
{
    fn evaluate_relative_permittivity_derivative<I>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
    {
        self.relative_permittivity_derivative::<I, C>(vacuum_wavenumber, order, variable)
    }

    fn evaluate_relative_permeability_derivative<I>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
    {
        self.relative_permeability_derivative::<I, C>(vacuum_wavenumber, order, variable)
    }
}

/// Complex-frequency constitutive evaluation using a fixed complex scalar.
pub trait EvaluateMeromorphicMaterial<C>: EvaluateMaterial<C>
where
    C: ComplexScalar<RealField = Self::Real> + Copy,
{
    /// Evaluate relative permittivity at complex vacuum wavenumber.
    fn evaluate_relative_permittivity_complex<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>;

    fn evaluate_relative_permeability_complex<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>;
}

impl<M, C> EvaluateMeromorphicMaterial<C> for M
where
    M: MeromorphicMaterial,
    M::Real: Copy,
    C: ComplexScalar<RealField = M::Real> + Copy,
{
    fn evaluate_relative_permittivity_complex<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
    {
        self.relative_permittivity_complex(vacuum_wavenumber)
    }

    fn evaluate_relative_permeability_complex<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
    {
        self.relative_permeability_complex(vacuum_wavenumber)
    }
}

/// Complex-frequency derivatives using a fixed complex scalar.
pub trait EvaluateDifferentiableMeromorphicMaterial<C>:
    EvaluateDifferentiableMaterial<C> + EvaluateMeromorphicMaterial<C>
where
    C: ComplexScalar<RealField = Self::Real> + Copy,
{
    /// Evaluate a derivative of relative permittivity at complex vacuum
    /// wavenumber.
    fn evaluate_relative_permittivity_complex_derivative<I>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>;

    fn evaluate_relative_permeability_complex_derivative<I>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>;
}

impl<M, C> EvaluateDifferentiableMeromorphicMaterial<C> for M
where
    M: DifferentiableMeromorphicMaterial,
    M::Real: Copy,
    C: ComplexScalar<RealField = M::Real> + Copy,
{
    fn evaluate_relative_permittivity_complex_derivative<I>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
    {
        self.relative_permittivity_complex_derivative(vacuum_wavenumber, order, variable)
    }

    fn evaluate_relative_permeability_complex_derivative<I>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
    {
        self.relative_permeability_complex_derivative(vacuum_wavenumber, order, variable)
    }
}
