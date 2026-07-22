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

pub(crate) trait MaterialPoint<R, C>: Send + Sync
where
    C: ComplexScalar<RealField = R>,
{
    fn relative_permeability_at(&self, vacuum_wavenumber: R) -> C;

    fn relative_permittivity_at(&self, vacuum_wavenumber: R) -> C;
}

impl<M, R, C> MaterialPoint<R, C> for M
where
    M: Material<Real = R> + Send + Sync,
    R: Copy,
    C: ComplexScalar<RealField = R>,
{
    fn relative_permeability_at(&self, vacuum_wavenumber: R) -> C {
        self.relative_permeability::<_, C>(Scalar(vacuum_wavenumber))
    }

    fn relative_permittivity_at(&self, vacuum_wavenumber: R) -> C {
        self.relative_permittivity::<_, C>(Scalar(vacuum_wavenumber))
    }
}

pub(crate) trait DifferentiableMaterialPoint<R, C>: MaterialPoint<R, C>
where
    C: ComplexScalar<RealField = R>,
{
    fn relative_permittivity_derivative_at(
        &self,
        vacuum_wavenumber: R,
        order: DerivativeOrder,
    ) -> C;

    fn relative_permeability_derivative_at(
        &self,
        vacuum_wavenumber: R,
        order: DerivativeOrder,
    ) -> C;
}

impl<M, R, C> DifferentiableMaterialPoint<R, C> for M
where
    M: DifferentiableMaterial<Real = R> + Send + Sync,
    R: Copy,
    C: ComplexScalar<RealField = R>,
{
    fn relative_permittivity_derivative_at(
        &self,
        vacuum_wavenumber: R,
        order: DerivativeOrder,
    ) -> C {
        self.relative_permittivity_derivative::<_, C>(Scalar(vacuum_wavenumber), order)
    }

    fn relative_permeability_derivative_at(
        &self,
        vacuum_wavenumber: R,
        order: DerivativeOrder,
    ) -> C {
        self.relative_permeability_derivative::<_, C>(Scalar(vacuum_wavenumber), order)
    }
}

pub(crate) trait MeromorphicMaterialPoint<R, C>: MaterialPoint<R, C>
where
    C: ComplexScalar<RealField = R>,
{
    fn relative_permittivity_complex_at(&self, vacuum_wavenumber: C) -> C;
    fn relative_permeability_complex_at(&self, vacuum_wavenumber: C) -> C;
}

impl<M, R, C> MeromorphicMaterialPoint<R, C> for M
where
    M: MeromorphicMaterial<Real = R> + Send + Sync,
    R: Copy,
    C: ComplexScalar<RealField = R>,
{
    fn relative_permittivity_complex_at(&self, vacuum_wavenumber: C) -> C {
        self.relative_permittivity_complex(Scalar(vacuum_wavenumber))
    }

    fn relative_permeability_complex_at(&self, vacuum_wavenumber: C) -> C {
        self.relative_permeability_complex(Scalar(vacuum_wavenumber))
    }
}

pub(crate) trait DifferentiableMeromorphicMaterialPoint<R, C>:
    DifferentiableMaterialPoint<R, C> + MeromorphicMaterialPoint<R, C>
where
    C: ComplexScalar<RealField = R>,
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

impl<M, R, C> DifferentiableMeromorphicMaterialPoint<R, C> for M
where
    M: DifferentiableMeromorphicMaterial<Real = R> + Send + Sync,
    R: Copy,
    C: ComplexScalar<RealField = R>,
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
