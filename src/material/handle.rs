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
pub struct MaterialHandle<C>
where
    C: ComplexScalar,
{
    type_name: &'static str,
    inner: Arc<dyn MaterialPoint<C>>,
}

impl<M, C> From<M> for MaterialHandle<C>
where
    M: Material<Real = C::RealField> + Send + Sync + 'static,
    C: ComplexScalar + 'static,
    C::RealField: Copy + 'static,
{
    fn from(material: M) -> Self {
        Self::new(material)
    }
}

impl<C> MaterialHandle<C>
where
    C: ComplexScalar + 'static,
    C::RealField: Copy + 'static,
{
    /// Erase a concrete real-axis material.
    pub fn new<M>(material: M) -> Self
    where
        M: Material<Real = C::RealField> + Send + Sync + 'static,
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

impl<C> fmt::Debug for MaterialHandle<C>
where
    C: ComplexScalar,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MaterialHandle")
            .field("type", &self.type_name)
            .finish_non_exhaustive()
    }
}

impl<C> EvaluateMaterial<C> for MaterialHandle<C>
where
    C: ComplexScalar,
    C::RealField: Copy,
{
    type Real = C::RealField;

    fn evaluate_relative_permeability<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permeability_at(k0))
    }

    fn evaluate_relative_permittivity<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = Self::Real>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permittivity_at(k0))
    }
}

/// Type-erased material supporting real-axis derivatives.
#[derive(Clone)]
pub struct DifferentiableMaterialHandle<C>
where
    C: ComplexScalar,
{
    type_name: &'static str,
    inner: Arc<dyn DifferentiableMaterialPoint<C>>,
}

impl<C> DifferentiableMaterialHandle<C>
where
    C: ComplexScalar + 'static,
    C::RealField: Copy + 'static,
{
    /// Erase a concrete differentiable material.
    pub fn new<M>(material: M) -> Self
    where
        M: DifferentiableMaterial<Real = C::RealField> + Send + Sync + 'static,
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

impl<M, C> From<M> for DifferentiableMaterialHandle<C>
where
    M: DifferentiableMaterial<Real = C::RealField> + Send + Sync + 'static,
    C: ComplexScalar + 'static,
    C::RealField: Copy + 'static,
{
    fn from(material: M) -> Self {
        Self::new(material)
    }
}

impl<C> fmt::Debug for DifferentiableMaterialHandle<C>
where
    C: ComplexScalar,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DifferentiableMaterialHandle")
            .field("type", &self.type_name)
            .finish_non_exhaustive()
    }
}

impl<C> EvaluateMaterial<C> for DifferentiableMaterialHandle<C>
where
    C: ComplexScalar,
    C::RealField: Copy,
{
    type Real = C::RealField;

    fn evaluate_relative_permeability<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C::RealField>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permeability_at(k0))
    }

    fn evaluate_relative_permittivity<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C::RealField>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permittivity_at(k0))
    }
}

impl<C> EvaluateDifferentiableMaterial<C> for DifferentiableMaterialHandle<C>
where
    C::RealField: Copy,
    C: ComplexScalar,
{
    fn evaluate_relative_permittivity_derivative<I>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C::RealField>,
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
pub struct MeromorphicMaterialHandle<C>
where
    C: ComplexScalar,
{
    type_name: &'static str,
    inner: Arc<dyn MeromorphicMaterialPoint<C>>,
}

impl<C> MeromorphicMaterialHandle<C>
where
    C: ComplexScalar + 'static,
    C::RealField: Copy + 'static,
{
    /// Erase a concrete meromorphic material.
    pub fn new<M>(material: M) -> Self
    where
        M: MeromorphicMaterial<Real = C::RealField> + Send + Sync + 'static,
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

impl<M, C> From<M> for MeromorphicMaterialHandle<C>
where
    M: MeromorphicMaterial<Real = C::RealField> + Send + Sync + 'static,
    C::RealField: Copy + 'static,
    C: ComplexScalar + 'static,
{
    fn from(material: M) -> Self {
        Self::new(material)
    }
}

impl<C> fmt::Debug for MeromorphicMaterialHandle<C>
where
    C: ComplexScalar,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("MeromorphicMaterialHandle")
            .field("type", &self.type_name)
            .finish_non_exhaustive()
    }
}

impl<C> EvaluateMaterial<C> for MeromorphicMaterialHandle<C>
where
    C::RealField: Copy,
    C: ComplexScalar,
{
    type Real = C::RealField;

    fn evaluate_relative_permeability<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C::RealField>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permeability_at(k0))
    }

    fn evaluate_relative_permittivity<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C::RealField>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permittivity_at(k0))
    }
}

impl<C> EvaluateMeromorphicMaterial<C> for MeromorphicMaterialHandle<C>
where
    C::RealField: Copy,
    C: ComplexScalar,
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
pub struct AnalyticalMaterialHandle<C>
where
    C: ComplexScalar,
{
    type_name: &'static str,
    inner: Arc<dyn DifferentiableMeromorphicMaterialPoint<C>>,
}

impl<M, C> From<M> for AnalyticalMaterialHandle<C>
where
    M: DifferentiableMeromorphicMaterial<Real = C::RealField> + Send + Sync + 'static,
    C::RealField: Copy + 'static,
    C: ComplexScalar + 'static,
{
    fn from(material: M) -> Self {
        Self::new(material)
    }
}

impl<C> AnalyticalMaterialHandle<C>
where
    C::RealField: Copy + 'static,
    C: ComplexScalar + 'static,
{
    /// Erase a fully analytical material.
    pub fn new<M>(material: M) -> Self
    where
        M: DifferentiableMeromorphicMaterial<Real = C::RealField> + Send + Sync + 'static,
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

impl<C> fmt::Debug for AnalyticalMaterialHandle<C>
where
    C: ComplexScalar,
{
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("AnalyticalMaterialHandle")
            .field("type", &self.type_name)
            .finish_non_exhaustive()
    }
}

impl<C> EvaluateMaterial<C> for AnalyticalMaterialHandle<C>
where
    C::RealField: Copy,
    C: ComplexScalar,
{
    type Real = C::RealField;

    fn evaluate_relative_permeability<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C::RealField>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permeability_at(k0))
    }

    fn evaluate_relative_permittivity<I>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C::RealField>,
    {
        vacuum_wavenumber.map(|k0| self.inner.relative_permittivity_at(k0))
    }
}

impl<C> EvaluateDifferentiableMaterial<C> for AnalyticalMaterialHandle<C>
where
    C::RealField: Copy,
    C: ComplexScalar,
{
    fn evaluate_relative_permittivity_derivative<I>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C::RealField>,
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

impl<C> EvaluateMeromorphicMaterial<C> for AnalyticalMaterialHandle<C>
where
    C::RealField: Copy,
    C: ComplexScalar,
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

impl<C> EvaluateDifferentiableMeromorphicMaterial<C> for AnalyticalMaterialHandle<C>
where
    C::RealField: Copy,
    C: ComplexScalar,
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
