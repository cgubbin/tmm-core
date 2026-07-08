use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    material::{Material, Scalar},
};

use super::Polarisation;

/// Material and propagation quantities used by the isotropic 2×2 kernel.
#[derive(Clone, Debug, PartialEq)]
pub struct IsotropicLayerQuantities<C, D>
where
    D: Dimension,
{
    pub epsilon: ArrayBase<OwnedRepr<C>, D>,
    pub mu: ArrayBase<OwnedRepr<C>, D>,
    pub kappa: ArrayBase<OwnedRepr<C>, D>,
    pub factor: ArrayBase<OwnedRepr<C>, D>,
}

/// Compute isotropic layer quantities for a sampled input grid.
pub fn isotropic_layer_quantities<M, C, D>(
    material: &M,
    wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    propagation_constant_squared: &ArrayBase<OwnedRepr<C>, D>,
    polarisation: Polarisation,
) -> IsotropicLayerQuantities<C, D>
where
    M: Material<Real = C::RealField>,
    C: ComplexScalar,
    D: Dimension,
{
    let epsilon = wavenumber.mapv(|w| material.relative_permittivity(Scalar(w)));
    let mu = wavenumber.mapv(|w| material.relative_permeability(Scalar(w)));

    let kappa = epsilon.clone() * mu.clone() * wavenumber.mapv(|w| w * w)
        - propagation_constant_squared.clone();

    let kappa = kappa.mapv(|x| x.sqrt());

    let factor = match polarisation {
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
