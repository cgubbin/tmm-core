//! Object-safe pointwise material interfaces.
//!
//! These traits are implementation details of the public material handles.
//! Concrete material implementations are adapted automatically through blanket
//! implementations.

use super::{
    DerivativeOrder, DifferentiableMaterial, DifferentiableMeromorphicMaterial, Material,
    MeromorphicMaterial, Scalar,
};
use crate::ComplexScalar;

pub(crate) trait MaterialPoint<C>: Send + Sync
where
    C: ComplexScalar,
{
    fn relative_permeability_at(&self, vacuum_wavenumber: C::RealField) -> C;

    fn relative_permittivity_at(&self, vacuum_wavenumber: C::RealField) -> C;
}

impl<M, C> MaterialPoint<C> for M
where
    M: Material<Real = C::RealField> + Send + Sync,
    C: ComplexScalar,
    C::RealField: Copy,
{
    fn relative_permeability_at(&self, vacuum_wavenumber: C::RealField) -> C {
        self.relative_permeability::<_, C>(Scalar(vacuum_wavenumber))
    }

    fn relative_permittivity_at(&self, vacuum_wavenumber: C::RealField) -> C {
        self.relative_permittivity::<_, C>(Scalar(vacuum_wavenumber))
    }
}

pub(crate) trait DifferentiableMaterialPoint<C>: MaterialPoint<C>
where
    C: ComplexScalar,
{
    fn relative_permittivity_derivative_at(
        &self,
        vacuum_wavenumber: C::RealField,
        order: DerivativeOrder,
    ) -> C;

    fn relative_permeability_derivative_at(
        &self,
        vacuum_wavenumber: C::RealField,
        order: DerivativeOrder,
    ) -> C;
}

impl<M, C> DifferentiableMaterialPoint<C> for M
where
    M: DifferentiableMaterial<Real = C::RealField> + Send + Sync,
    C: ComplexScalar,
    C::RealField: Copy,
{
    fn relative_permittivity_derivative_at(
        &self,
        vacuum_wavenumber: C::RealField,
        order: DerivativeOrder,
    ) -> C {
        self.relative_permittivity_derivative::<_, C>(Scalar(vacuum_wavenumber), order)
    }

    fn relative_permeability_derivative_at(
        &self,
        vacuum_wavenumber: C::RealField,
        order: DerivativeOrder,
    ) -> C {
        self.relative_permeability_derivative::<_, C>(Scalar(vacuum_wavenumber), order)
    }
}

pub(crate) trait MeromorphicMaterialPoint<C>: MaterialPoint<C>
where
    C: ComplexScalar,
{
    fn relative_permittivity_complex_at(&self, vacuum_wavenumber: C) -> C;
    fn relative_permeability_complex_at(&self, vacuum_wavenumber: C) -> C;
}

impl<M, C> MeromorphicMaterialPoint<C> for M
where
    M: MeromorphicMaterial<Real = C::RealField> + Send + Sync,
    C: ComplexScalar,
    C::RealField: Copy,
{
    fn relative_permittivity_complex_at(&self, vacuum_wavenumber: C) -> C {
        self.relative_permittivity_complex(Scalar(vacuum_wavenumber))
    }

    fn relative_permeability_complex_at(&self, vacuum_wavenumber: C) -> C {
        self.relative_permeability_complex(Scalar(vacuum_wavenumber))
    }
}

pub(crate) trait DifferentiableMeromorphicMaterialPoint<C>:
    DifferentiableMaterialPoint<C> + MeromorphicMaterialPoint<C>
where
    C: ComplexScalar,
{
    fn relative_permittivity_complex_derivative_at(
        &self,
        vacuum_wavenumber: C,
        order: DerivativeOrder,
    ) -> C;

    fn relative_permeability_complex_derivative_at(
        &self,
        vacuum_wavenumber: C,
        order: DerivativeOrder,
    ) -> C;
}

impl<M, C> DifferentiableMeromorphicMaterialPoint<C> for M
where
    M: DifferentiableMeromorphicMaterial<Real = C::RealField> + Send + Sync,
    C: ComplexScalar,
    C::RealField: Copy,
{
    fn relative_permittivity_complex_derivative_at(
        &self,
        vacuum_wavenumber: C,
        order: DerivativeOrder,
    ) -> C {
        self.relative_permittivity_complex_derivative(Scalar(vacuum_wavenumber), order)
    }

    fn relative_permeability_complex_derivative_at(
        &self,
        vacuum_wavenumber: C,
        order: DerivativeOrder,
    ) -> C {
        self.relative_permeability_complex_derivative(Scalar(vacuum_wavenumber), order)
    }
}
