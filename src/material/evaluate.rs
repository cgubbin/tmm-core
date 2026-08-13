//! Backend-facing material evaluation traits.
//!
//! The public material model traits allow the caller to select any compatible
//! complex scalar type. That flexibility prevents direct type erasure.
//!
//! These traits fix the complex scalar type at the stack/backend boundary,
//! allowing heterogeneous materials to be stored behind material handles while
//! preserving static dispatch for homogeneous stacks.

use super::{
    DerivativeOrder, DifferentiableMaterial, DifferentiableMeromorphicMaterial, Material,
    MeromorphicMaterial, Sampled,
};
use crate::ComplexScalar;

/// Real-axis evaluation of an isotropic material using a fixed complex scalar.
///
/// Backends should generally depend on this trait rather than directly on
/// [`Material`]. Every concrete [`Material`] implementation receives a blanket
/// implementation.
///
/// This is the backend-facing counterpart of [`Material`].
pub trait EvaluateMaterial<C>
where
    C: ComplexScalar<RealField = Self::Real>,
{
    /// Real scalar used for the spectral coordinate.
    type Real: Copy;

    /// Evaluate relative permeability on the real vacuum-angular-wavenumber axis.
    ///
    /// `vacuum_angular_wavenumber` is `k₀` expressed in `cm⁻¹`.
    fn evaluate_relative_permeability<I>(&self, vacuum_angular_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>;

    /// Evaluate relative permittivity on the real vacuum-angular-wavenumber axis.
    ///
    /// `vacuum_angular_wavenumber` is `k₀` expressed in `cm⁻¹`.
    fn evaluate_relative_permittivity<I>(&self, vacuum_angular_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>;
}

impl<M, C> EvaluateMaterial<C> for M
where
    M: Material<Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
{
    type Real = M::Real;

    fn evaluate_relative_permeability<I>(&self, vacuum_angular_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
    {
        self.relative_permeability::<I, C>(vacuum_angular_wavenumber)
    }

    fn evaluate_relative_permittivity<I>(&self, vacuum_angular_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
    {
        self.relative_permittivity::<I, C>(vacuum_angular_wavenumber)
    }
}

/// Real-axis material derivatives using a fixed complex scalar.
///
/// Backends should generally depend on this trait rather than directly on
/// [`DifferentiableMaterial`]. Material authors should implement [`DifferentiableMaterial`];
/// this trait is provided automatically through a blanket implementation.
///
/// This is the backend-facing counterpart of [`DifferentiableMaterial`].
pub trait EvaluateDifferentiableMaterial<C>: EvaluateMaterial<C>
where
    C: ComplexScalar<RealField = Self::Real>,
{
    /// Evaluate a derivative of relative permittivity with respect to `k₀`.
    fn evaluate_relative_permittivity_derivative<I>(
        &self,
        vacuum_angular_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>;

    /// Evaluate a derivative of relative permeability with respect to `k₀`.
    fn evaluate_relative_permeability_derivative<I>(
        &self,
        vacuum_angular_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>;
}

impl<M, C> EvaluateDifferentiableMaterial<C> for M
where
    M: DifferentiableMaterial,
    M::Real: Copy,
    C: ComplexScalar<RealField = M::Real>,
{
    fn evaluate_relative_permittivity_derivative<I>(
        &self,
        vacuum_angular_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
    {
        self.relative_permittivity_derivative::<I, C>(vacuum_angular_wavenumber, order)
    }

    fn evaluate_relative_permeability_derivative<I>(
        &self,
        vacuum_angular_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
    {
        self.relative_permeability_derivative::<I, C>(vacuum_angular_wavenumber, order)
    }
}

/// Complex-frequency constitutive evaluation using a fixed complex scalar.
///
/// Backends should generally depend on this trait rather than directly on
/// [`MeromorphicMaterial`]. Material authors should implement [`MeromorphicMaterial`];
/// this trait is provided automatically through a blanket implementation.
///
/// This is the backend-facing counterpart of [`MeromorphicMaterial`].
pub trait EvaluateMeromorphicMaterial<C>: EvaluateMaterial<C>
where
    C: ComplexScalar<RealField = Self::Real>,
{
    /// Evaluate relative permittivity in the complex vacuum-angular-wavenumber plane.
    ///
    /// `vacuum_angular_wavenumber` is `k₀` expressed in `cm⁻¹`.
    fn evaluate_relative_permittivity_complex<I>(
        &self,
        vacuum_angular_wavenumber: I,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>;

    /// Evaluate relative permeability in the complex vacuum-angular-wavenumber plane.
    ///
    /// `vacuum_angular_wavenumber` is `k₀` expressed in `cm⁻¹`.
    fn evaluate_relative_permeability_complex<I>(
        &self,
        vacuum_angular_wavenumber: I,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>;
}

impl<M, C> EvaluateMeromorphicMaterial<C> for M
where
    M: MeromorphicMaterial,
    M::Real: Copy,
    C: ComplexScalar<RealField = M::Real>,
{
    fn evaluate_relative_permittivity_complex<I>(
        &self,
        vacuum_angular_wavenumber: I,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
    {
        self.relative_permittivity_complex(vacuum_angular_wavenumber)
    }

    fn evaluate_relative_permeability_complex<I>(
        &self,
        vacuum_angular_wavenumber: I,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
    {
        self.relative_permeability_complex(vacuum_angular_wavenumber)
    }
}

/// Complex-frequency constitutive derivative evaluation using a fixed complex scalar.
///
/// Backends should generally depend on this trait rather than directly on
/// [`DifferentiableMeromorphicMaterial`]. Material authors should implement [`DifferentiableMeromorphicMaterial`];
/// this trait is provided automatically through a blanket implementation.
///
/// This is the backend-facing counterpart of [`DifferentiableMeromorphicMaterial`].
pub trait EvaluateDifferentiableMeromorphicMaterial<C>:
    EvaluateDifferentiableMaterial<C> + EvaluateMeromorphicMaterial<C>
where
    C: ComplexScalar<RealField = Self::Real>,
{
    /// Evaluate a derivative of relative permittivity with respect to `k₀`.
    fn evaluate_relative_permittivity_complex_derivative<I>(
        &self,
        vacuum_angular_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>;

    /// Evaluate a derivative of relative permeability with respect to `k₀`.
    fn evaluate_relative_permeability_complex_derivative<I>(
        &self,
        vacuum_angular_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>;
}

impl<M, C> EvaluateDifferentiableMeromorphicMaterial<C> for M
where
    M: DifferentiableMeromorphicMaterial,
    M::Real: Copy,
    C: ComplexScalar<RealField = M::Real>,
{
    fn evaluate_relative_permittivity_complex_derivative<I>(
        &self,
        vacuum_angular_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
    {
        self.relative_permittivity_complex_derivative(vacuum_angular_wavenumber, order)
    }

    fn evaluate_relative_permeability_complex_derivative<I>(
        &self,
        vacuum_angular_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
    {
        self.relative_permeability_complex_derivative(vacuum_angular_wavenumber, order)
    }
}
