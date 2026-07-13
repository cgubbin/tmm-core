mod admittance;
mod derivatives;

use ndarray::{ArrayBase, Dimension, OwnedRepr};

pub(crate) use admittance::{
    AdmittanceEvaluation, IsotropicLayerAdmittance,
};
pub(crate) use derivatives::{IsotropicLayerFirstDerivatives, IsotropicLayerSecondDerivatives};

use crate::{
    ComplexScalar,
    backend::{PlanarInput, Polarisation},
    material::{Material, Scalar},
};

/// Material and propagation quantities used by the isotropic 2×2 kernel.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IsotropicLayerQuantities<C, D>
where
    D: Dimension,
{
    pub(crate) epsilon: ArrayBase<OwnedRepr<C>, D>,
    pub(crate) mu: ArrayBase<OwnedRepr<C>, D>,
    pub(crate) kappa: ArrayBase<OwnedRepr<C>, D>,
    pub(crate) factor: ArrayBase<OwnedRepr<C>, D>,
}

impl<C, D> IsotropicLayerQuantities<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Compute isotropic layer quantities for a sampled input grid.
    pub(crate) fn new<M>(material: &M, planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>) -> Self
    where
        M: Material<Real = C::RealField>,
    {
        let epsilon = planar
            .vacuum_wavenumber
            .mapv(|k0| material.relative_permittivity(Scalar(k0)));
        let mu = planar
            .vacuum_wavenumber
            .mapv(|k0| material.relative_permeability(Scalar(k0)));

        let kappa = epsilon.clone() * mu.clone() * planar.vacuum_wavenumber.mapv(|k0| k0 * k0)
            - planar.parallel_wavenumber.mapv(|kp| kp * kp);

        let kappa = kappa.mapv(|x| x.sqrt());

        let factor = match planar.polarisation {
            Polarisation::TransverseElectric => mu.clone(),
            Polarisation::TransverseMagnetic => epsilon.clone(),
        };

        IsotropicLayerQuantities {
            epsilon,
            mu,
            kappa,
            factor,
        }
    }

    pub(crate) fn admittance(&self) -> IsotropicLayerAdmittance<C, D> {
        IsotropicLayerAdmittance::from_quantities(self)
    }
}
