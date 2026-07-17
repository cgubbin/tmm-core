use crate::{
    ComplexScalar,
    material::{
        DerivativeOrder, EvaluateDifferentiableMaterial, EvaluateDifferentiableMeromorphicMaterial,
        EvaluateMaterial, EvaluateMeromorphicMaterial, SpectralVariable,
    },
};

use ndarray::{ArrayBase, Dimension, OwnedRepr};

pub(crate) struct RealAxis;
pub(crate) struct ComplexPlane;

pub(crate) trait ConstitutiveEvaluator<C, D, M>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn relative_permittivity(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    ) -> ArrayBase<OwnedRepr<C>, D>;

    fn relative_permeability(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    ) -> ArrayBase<OwnedRepr<C>, D>;
}

pub(crate) trait ConstitutiveDerivativeEvaluator<C, D, M>:
    ConstitutiveEvaluator<C, D, M>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn relative_permittivity_derivative(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> ArrayBase<OwnedRepr<C>, D>;

    fn relative_permeability_derivative(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> ArrayBase<OwnedRepr<C>, D>;
}

impl<C, D, M> ConstitutiveEvaluator<C, D, M> for RealAxis
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
    M: EvaluateMaterial<C, Real = C::RealField>,
{
    fn relative_permittivity(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    ) -> ArrayBase<OwnedRepr<C>, D> {
        material.evaluate_relative_permittivity(vacuum_wavenumber.mapv(|value| value.real()))
    }

    fn relative_permeability(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    ) -> ArrayBase<OwnedRepr<C>, D> {
        material.evaluate_relative_permeability(vacuum_wavenumber.mapv(|value| value.real()))
    }
}

impl<C, D, M> ConstitutiveDerivativeEvaluator<C, D, M> for RealAxis
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
{
    fn relative_permittivity_derivative(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> ArrayBase<OwnedRepr<C>, D> {
        material.evaluate_relative_permittivity_derivative(
            vacuum_wavenumber.mapv(|value| value.real()),
            order,
            variable,
        )
    }

    fn relative_permeability_derivative(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> ArrayBase<OwnedRepr<C>, D> {
        material.evaluate_relative_permeability_derivative(
            vacuum_wavenumber.mapv(|value| value.real()),
            order,
            variable,
        )
    }
}

impl<C, D, M> ConstitutiveEvaluator<C, D, M> for ComplexPlane
where
    C: ComplexScalar,
    D: Dimension,
    M: EvaluateMeromorphicMaterial<C, Real = C::RealField>,
{
    fn relative_permittivity(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    ) -> ArrayBase<OwnedRepr<C>, D> {
        material.evaluate_relative_permittivity_complex(vacuum_wavenumber.clone())
    }

    fn relative_permeability(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    ) -> ArrayBase<OwnedRepr<C>, D> {
        material.evaluate_relative_permeability_complex(vacuum_wavenumber.clone())
    }
}

impl<C, D, M> ConstitutiveDerivativeEvaluator<C, D, M> for ComplexPlane
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
    M: EvaluateDifferentiableMeromorphicMaterial<C, Real = C::RealField>,
{
    fn relative_permeability_derivative(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> ArrayBase<OwnedRepr<C>, D> {
        material.evaluate_relative_permeability_complex_derivative(
            vacuum_wavenumber.clone(),
            order,
            variable,
        )
    }

    fn relative_permittivity_derivative(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> ArrayBase<OwnedRepr<C>, D> {
        material.evaluate_relative_permittivity_complex_derivative(
            vacuum_wavenumber.clone(),
            order,
            variable,
        )
    }
}
