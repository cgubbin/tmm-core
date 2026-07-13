use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::{
        DerivativeVariable, PlanarInput, Polarisation,
        isotropic::{
            IsotropicLayerFirstDerivatives, IsotropicLayerQuantities,
            IsotropicLayerSecondDerivatives,
        },
        jet::ArrayJet,
    },
    material::Material,
};

#[derive(Clone, Debug, PartialEq)]
pub struct IsotropicLayerAdmittance<C, D: Dimension>(ArrayBase<OwnedRepr<C>, D>);

impl<C, D: Dimension> IsotropicLayerAdmittance<C, D> {
    pub(crate) fn from_quantities(q: &IsotropicLayerQuantities<C, D>) -> Self
    where
        C: ComplexScalar,
    {
        Self(q.kappa.clone() / q.factor.view())
    }

    /// Compute isotropic layer quantities for a sampled input grid.
    pub(crate) fn evaluate<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self
    where
        C: ComplexScalar,
        M: Material<Real = C::RealField>,
    {
        let q = IsotropicLayerQuantities::new(material, planar);
        Self::from_quantities(&q)
    }

    pub fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.0
    }

    pub fn into_inner(self) -> ArrayBase<OwnedRepr<C>, D> {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IsotropicLayerAdmittanceFirstDerivative<C, D: Dimension>(ArrayBase<OwnedRepr<C>, D>);

impl<C, D: Dimension> IsotropicLayerAdmittanceFirstDerivative<C, D> {
    pub(crate) fn from_inner(inner: ArrayBase<OwnedRepr<C>, D>) -> Self {
        Self(inner)
    }

    pub(crate) fn from_quantities(
        q: &IsotropicLayerQuantities<C, D>,
        dq: &IsotropicLayerFirstDerivatives<C, D>,
    ) -> Self
    where
        C: ComplexScalar,
    {
        Self(
            dq.dkappa.clone() / q.factor.view()
                - q.kappa.clone() * dq.dfactor.view() / q.factor.mapv(|x| x * x),
        )
    }

    pub(crate) fn vacuum_wavenumber_squared_from_quantities<M>(
        material: &M,
        q: &IsotropicLayerQuantities<C, D>,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
        polarisation: Polarisation,
    ) -> Self
    where
        C: ComplexScalar,
        M: Material<Real = C::RealField>,
    {
        let dq = IsotropicLayerFirstDerivatives::with_respect_to_vacuum_wavenumber_squared(
            material,
            q,
            vacuum_wavenumber,
            polarisation,
        );

        Self::from_quantities(q, &dq)
    }

    pub(crate) fn parallel_wavenumber_squared_from_quantities(
        q: &IsotropicLayerQuantities<C, D>,
    ) -> Self
    where
        C: ComplexScalar,
    {
        let dq = IsotropicLayerFirstDerivatives::with_respect_to_parallel_wavenumber_squared(&q);

        Self::from_quantities(q, &dq)
    }

    pub(super) fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IsotropicLayerAdmittanceSecondDerivative<C, D: Dimension>
where
    D: Dimension,
{
    pub first: IsotropicLayerAdmittanceFirstDerivative<C, D>,
    pub second: ArrayBase<OwnedRepr<C>, D>,
}

impl<C, D: Dimension> IsotropicLayerAdmittanceSecondDerivative<C, D> {
    pub(crate) fn from_inner(
        first: ArrayBase<OwnedRepr<C>, D>,
        second: ArrayBase<OwnedRepr<C>, D>,
    ) -> Self {
        Self {
            second,
            first: IsotropicLayerAdmittanceFirstDerivative::from_inner(first),
        }
    }

    pub(crate) fn from_quantities(
        q: &IsotropicLayerQuantities<C, D>,
        derivatives: &IsotropicLayerSecondDerivatives<C, D>,
    ) -> Self
    where
        C: ComplexScalar,
    {
        let two = C::one() + C::one();

        let factor_squared = q.factor.mapv(|x| x * x);
        let factor_cubed = q.factor.mapv(|x| x * x * x);

        let first =
            IsotropicLayerAdmittanceFirstDerivative::from_quantities(&q, &derivatives.first);

        let second = derivatives.ddkappa.clone() / q.factor.view()
            - (derivatives.first.dkappa.clone() * derivatives.first.dfactor.view())
                .mapv(|x| two * x)
                / factor_squared.view()
            - q.kappa.clone() * derivatives.ddfactor.view() / factor_squared.view()
            + q.kappa.clone() * derivatives.first.dfactor.mapv(|x| x * x).mapv(|x| two * x)
                / factor_cubed.view();

        Self { first, second }
    }

    pub(crate) fn vacuum_wavenumber_squared_from_quantities<M>(
        material: &M,
        q: &IsotropicLayerQuantities<C, D>,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
        polarisation: Polarisation,
    ) -> Self
    where
        C: ComplexScalar,
        M: Material<Real = C::RealField>,
    {
        let derivatives =
            IsotropicLayerSecondDerivatives::with_respect_to_vacuum_wavenumber_squared(
                material,
                q,
                vacuum_wavenumber,
                polarisation,
            );

        Self::from_quantities(q, &derivatives)
    }

    pub(crate) fn parallel_wavenumber_squared_from_quantities(
        q: &IsotropicLayerQuantities<C, D>,
    ) -> Self
    where
        C: ComplexScalar,
    {
        let derivatives =
            IsotropicLayerSecondDerivatives::with_respect_to_parallel_wavenumber_squared(q);

        Self::from_quantities(q, &derivatives)
    }

    pub(super) fn first(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        self.first.value()
    }

    pub(super) fn second(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.second
    }
}

pub(crate) struct AdmittanceEvaluation<C, D>
where
    D: Dimension,
{
    value: IsotropicLayerAdmittance<C, D>,
    first: Option<IsotropicLayerAdmittanceFirstDerivative<C, D>>,
    second: Option<IsotropicLayerAdmittanceSecondDerivative<C, D>>,
}

impl<C, D> AdmittanceEvaluation<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn value_only(value: IsotropicLayerAdmittance<C, D>) -> Self {
        Self {
            value,
            first: None,
            second: None,
        }
    }

    pub(crate) fn first_derivative(
        value: IsotropicLayerAdmittance<C, D>,
        first: IsotropicLayerAdmittanceFirstDerivative<C, D>,
    ) -> Self {
        Self {
            value,
            first: Some(first),
            second: None,
        }
    }

    pub(crate) fn second_derivative(
        value: IsotropicLayerAdmittance<C, D>,
        second: IsotropicLayerAdmittanceSecondDerivative<C, D>,
    ) -> Self {
        Self {
            value,
            first: None,
            second: Some(second),
        }
    }

    pub(crate) fn jets(&self) -> ArrayJet<C, D> {
        match (&self.first, &self.second) {
            (_, Some(second)) => ArrayJet::with_second(
                self.value.value().clone(),
                second.first().clone(),
                second.second().clone(),
            ),

            (Some(first), None) => {
                ArrayJet::with_first(self.value.value().clone(), first.value().clone())
            }

            (None, None) => ArrayJet::value_only(self.value.value().clone()),
        }
    }

    pub(crate) fn evaluate_first<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Self
    where
        M: Material<Real = C::RealField>,
        C: ComplexScalar,
        D: Dimension,
    {
        let q = IsotropicLayerQuantities::new(material, planar);
        let value = IsotropicLayerAdmittance::from_quantities(&q);

        match variable.primitive() {
            DerivativeVariable::VacuumWavenumberSquared => {
                let first =
                IsotropicLayerAdmittanceFirstDerivative::vacuum_wavenumber_squared_from_quantities(
                    material,
                    &q,
                    planar.vacuum_wavenumber(),
                    planar.polarisation(),
                );

                Self::first_derivative(value, first)
            }

            DerivativeVariable::ParallelWavenumberSquared => {
                let first =
                IsotropicLayerAdmittanceFirstDerivative::
                    parallel_wavenumber_squared_from_quantities(&q);

                Self::first_derivative(value, first)
            }

            DerivativeVariable::Thickness(_) => {
                let zero = value.value().mapv(|_| C::zero());

                Self::first_derivative(
                    value,
                    IsotropicLayerAdmittanceFirstDerivative::from_inner(zero),
                )
            }

            DerivativeVariable::VacuumWavenumber | DerivativeVariable::ParallelWavenumber => {
                unreachable!("linear variables must be transformed from primitives")
            }
        }
    }

    pub(crate) fn evaluate_second<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Self
    where
        M: Material<Real = C::RealField>,
    {
        let q = IsotropicLayerQuantities::new(material, planar);
        let value = IsotropicLayerAdmittance::from_quantities(&q);

        match variable.primitive() {
            DerivativeVariable::VacuumWavenumberSquared => {
                let derivative =
                IsotropicLayerAdmittanceSecondDerivative::vacuum_wavenumber_squared_from_quantities(
                    material,
                    &q,
                    planar.vacuum_wavenumber(),
                    planar.polarisation(),
                );

                Self::second_derivative(value, derivative)
            }

            DerivativeVariable::ParallelWavenumberSquared => {
                let derivative =
                IsotropicLayerAdmittanceSecondDerivative::
                    parallel_wavenumber_squared_from_quantities(&q);

                Self::second_derivative(value, derivative)
            }

            DerivativeVariable::Thickness(_) => {
                let zero = value.value().mapv(|_| C::zero());

                Self::second_derivative(
                    value,
                    IsotropicLayerAdmittanceSecondDerivative::from_inner(zero.clone(), zero),
                )
            }

            DerivativeVariable::VacuumWavenumber | DerivativeVariable::ParallelWavenumber => {
                unreachable!("linear variables must be transformed from primitives")
            }
        }
    }
}
