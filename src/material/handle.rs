//! Type-erased isotropic material handles.
//!
//! Each handle guarantees a specific set of material capabilities. A stack
//! containing a material that lacks one of those capabilities cannot be
//! constructed using the corresponding handle.

use std::{any::type_name, fmt, sync::Arc};

use crate::ComplexScalar;

use super::{
    DerivativeOrder, DifferentiableMaterial, DifferentiableMeromorphicMaterial,
    EvaluateDifferentiableMaterial, EvaluateDifferentiableMeromorphicMaterial, EvaluateMaterial,
    EvaluateMeromorphicMaterial, Material, MeromorphicMaterial, Sampled,
    erased::{
        DifferentiableMaterialPoint, DifferentiableMeromorphicMaterialPoint, MaterialPoint,
        MeromorphicMaterialPoint,
    },
};

/// Type-erased real-axis isotropic material.
#[derive(Clone)]
pub struct MaterialHandle<R, C>
where
    C: ComplexScalar<RealField = R>,
{
    type_name: &'static str,
    inner: Arc<dyn MaterialPoint<R, C>>,
}

impl<M, R, C> From<M> for MaterialHandle<R, C>
where
    M: Material<Real = R> + Send + Sync + 'static,
    R: Copy + 'static,
    C: ComplexScalar<RealField = R> + 'static,
{
    fn from(material: M) -> Self {
        Self::new(material)
    }
}

impl<R, C> MaterialHandle<R, C>
where
    R: Copy + 'static,
    C: ComplexScalar<RealField = R> + 'static,
{
    /// Erase a concrete real-axis material.
    pub fn new<M>(material: M) -> Self
    where
        M: Material<Real = R> + Send + Sync + 'static,
    {
        Self {
            type_name: type_name::<M>(),
            inner: Arc::new(material),
        }
    }

    /// Return the erased concrete Rust type name.
    pub fn concrete_type_name(&self) -> &'static str {
        self.type_name
    }
}

impl<R, C> fmt::Debug for MaterialHandle<R, C>
where
    C: ComplexScalar<RealField = R>,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MaterialHandle")
            .field("type", &self.type_name)
            .finish_non_exhaustive()
    }
}

impl<R, C> EvaluateMaterial<C> for MaterialHandle<R, C>
where
    R: Copy,
    C: ComplexScalar<RealField = R>,
{
    type Real = R;

    fn evaluate_relative_permeability<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = R>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permeability_at(k0))
    }

    fn evaluate_relative_permittivity<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = R>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permittivity_at(k0))
    }
}

/// Type-erased material supporting real-axis derivatives.
#[derive(Clone)]
pub struct DifferentiableMaterialHandle<R, C>
where
    C: ComplexScalar<RealField = R>,
{
    type_name: &'static str,
    inner: Arc<dyn DifferentiableMaterialPoint<R, C>>,
}

impl<R, C> DifferentiableMaterialHandle<R, C>
where
    R: Copy + 'static,
    C: ComplexScalar<RealField = R> + 'static,
{
    /// Erase a concrete differentiable material.
    pub fn new<M>(material: M) -> Self
    where
        M: DifferentiableMaterial<Real = R> + Send + Sync + 'static,
    {
        Self {
            type_name: type_name::<M>(),
            inner: Arc::new(material),
        }
    }

    /// Return the erased concrete Rust type name.
    pub fn concrete_type_name(&self) -> &'static str {
        self.type_name
    }
}

impl<M, R, C> From<M> for DifferentiableMaterialHandle<R, C>
where
    M: DifferentiableMaterial<Real = R> + Send + Sync + 'static,
    R: Copy + 'static,
    C: ComplexScalar<RealField = R> + 'static,
{
    fn from(material: M) -> Self {
        Self::new(material)
    }
}

impl<R, C> fmt::Debug for DifferentiableMaterialHandle<R, C>
where
    C: ComplexScalar<RealField = R>,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DifferentiableMaterialHandle")
            .field("type", &self.type_name)
            .finish_non_exhaustive()
    }
}

impl<R, C> EvaluateMaterial<C> for DifferentiableMaterialHandle<R, C>
where
    R: Copy,
    C: ComplexScalar<RealField = R>,
{
    type Real = R;

    fn evaluate_relative_permeability<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = R>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permeability_at(k0))
    }

    fn evaluate_relative_permittivity<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = R>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permittivity_at(k0))
    }
}

impl<R, C> EvaluateDifferentiableMaterial<C> for DifferentiableMaterialHandle<R, C>
where
    R: Copy,
    C: ComplexScalar<RealField = R>,
{
    fn evaluate_relative_permittivity_derivative<I>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = R>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permittivity_derivative_at(k0, order))
    }

    fn evaluate_relative_permeability_derivative<I>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permeability_derivative_at(k0, order))
    }
}

/// Type-erased material supporting complex-frequency continuation.
#[derive(Clone)]
pub struct MeromorphicMaterialHandle<R, C>
where
    C: ComplexScalar<RealField = R>,
{
    type_name: &'static str,
    inner: Arc<dyn MeromorphicMaterialPoint<R, C>>,
}

impl<R, C> MeromorphicMaterialHandle<R, C>
where
    R: Copy + 'static,
    C: ComplexScalar<RealField = R> + 'static,
{
    /// Erase a concrete meromorphic material.
    pub fn new<M>(material: M) -> Self
    where
        M: MeromorphicMaterial<Real = R> + Send + Sync + 'static,
    {
        Self {
            type_name: type_name::<M>(),
            inner: Arc::new(material),
        }
    }

    /// Return the erased concrete Rust type name.
    pub fn concrete_type_name(&self) -> &'static str {
        self.type_name
    }
}

impl<M, R, C> From<M> for MeromorphicMaterialHandle<R, C>
where
    M: MeromorphicMaterial<Real = R> + Send + Sync + 'static,
    R: Copy + 'static,
    C: ComplexScalar<RealField = R> + 'static,
{
    fn from(material: M) -> Self {
        Self::new(material)
    }
}

impl<R, C> fmt::Debug for MeromorphicMaterialHandle<R, C>
where
    C: ComplexScalar<RealField = R>,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MeromorphicMaterialHandle")
            .field("type", &self.type_name)
            .finish_non_exhaustive()
    }
}

impl<R, C> EvaluateMaterial<C> for MeromorphicMaterialHandle<R, C>
where
    R: Copy,
    C: ComplexScalar<RealField = R>,
{
    type Real = R;

    fn evaluate_relative_permeability<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = R>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permeability_at(k0))
    }

    fn evaluate_relative_permittivity<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = R>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permittivity_at(k0))
    }
}

impl<R, C> EvaluateMeromorphicMaterial<C> for MeromorphicMaterialHandle<R, C>
where
    R: Copy,
    C: ComplexScalar<RealField = R>,
{
    fn evaluate_relative_permittivity_complex<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permittivity_complex_at(k0))
    }

    fn evaluate_relative_permeability_complex<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permeability_complex_at(k0))
    }
}

/// Type-erased material supporting real derivatives, complex continuation, and
/// complex derivatives.
#[derive(Clone)]
pub struct AnalyticalMaterialHandle<R, C>
where
    C: ComplexScalar<RealField = R>,
{
    type_name: &'static str,
    inner: Arc<dyn DifferentiableMeromorphicMaterialPoint<R, C>>,
}

impl<M, R, C> From<M> for AnalyticalMaterialHandle<R, C>
where
    M: DifferentiableMeromorphicMaterial<Real = R> + Send + Sync + 'static,
    R: Copy + 'static,
    C: ComplexScalar<RealField = R> + 'static,
{
    fn from(material: M) -> Self {
        Self::new(material)
    }
}

impl<R, C> AnalyticalMaterialHandle<R, C>
where
    R: Copy + 'static,
    C: ComplexScalar<RealField = R> + 'static,
{
    /// Erase a fully analytical material.
    pub fn new<M>(material: M) -> Self
    where
        M: DifferentiableMeromorphicMaterial<Real = R> + Send + Sync + 'static,
    {
        Self {
            type_name: type_name::<M>(),
            inner: Arc::new(material),
        }
    }

    /// Return the erased concrete Rust type name.
    pub fn concrete_type_name(&self) -> &'static str {
        self.type_name
    }
}

impl<R, C> fmt::Debug for AnalyticalMaterialHandle<R, C>
where
    C: ComplexScalar<RealField = R>,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnalyticalMaterialHandle")
            .field("type", &self.type_name)
            .finish_non_exhaustive()
    }
}

impl<R, C> EvaluateMaterial<C> for AnalyticalMaterialHandle<R, C>
where
    R: Copy,
    C: ComplexScalar<RealField = R>,
{
    type Real = R;

    fn evaluate_relative_permeability<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = R>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permeability_at(k0))
    }

    fn evaluate_relative_permittivity<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = R>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permittivity_at(k0))
    }
}

impl<R, C> EvaluateDifferentiableMaterial<C> for AnalyticalMaterialHandle<R, C>
where
    R: Copy,
    C: ComplexScalar<RealField = R>,
{
    fn evaluate_relative_permittivity_derivative<I>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = R>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permittivity_derivative_at(k0, order))
    }

    fn evaluate_relative_permeability_derivative<I>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permeability_derivative_at(k0, order))
    }
}

impl<R, C> EvaluateMeromorphicMaterial<C> for AnalyticalMaterialHandle<R, C>
where
    R: Copy,
    C: ComplexScalar<RealField = R>,
{
    fn evaluate_relative_permittivity_complex<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permittivity_complex_at(k0))
    }

    fn evaluate_relative_permeability_complex<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permeability_complex_at(k0))
    }
}

impl<R, C> EvaluateDifferentiableMeromorphicMaterial<C> for AnalyticalMaterialHandle<R, C>
where
    R: Copy,
    C: ComplexScalar<RealField = R>,
{
    fn evaluate_relative_permittivity_complex_derivative<I>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
    {
        vacuum_wavenumber.map(|k0| {
            self.inner
                .relative_permittivity_complex_derivative_at(k0, order)
        })
    }

    fn evaluate_relative_permeability_complex_derivative<I>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
    {
        vacuum_wavenumber.map(|k0| {
            self.inner
                .relative_permeability_complex_derivative_at(k0, order)
        })
    }
}
