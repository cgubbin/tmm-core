use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    material::{Material, Scalar},
    stack::Thickness,
};

use super::{Matrix2, Polarisation};

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

pub fn isotropic_layer_matrix<M, C, D>(
    material: &M,
    thickness: Thickness<C::RealField>,
    wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    propagation_constant_squared: &ArrayBase<OwnedRepr<C>, D>,
    polarisation: Polarisation,
) -> Matrix2<C, D>
where
    M: Material<Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let q = isotropic_layer_quantities(
        material,
        wavenumber,
        propagation_constant_squared,
        polarisation,
    );

    let d = C::from_real(thickness.as_cm());

    let kd = q.kappa.mapv(|k| k * d);
    let coskd = kd.mapv(|x| x.cos());
    let sinkd = kd.mapv(|x| x.sin());

    Matrix2::new(
        coskd.clone(),
        -sinkd.clone() * q.factor.view() / q.kappa.view(),
        sinkd * q.kappa.view() / q.factor.view(),
        coskd,
    )
}

pub fn isotropic_layer_thickness_derivative<M, C, D>(
    material: &M,
    thickness: Thickness<C::RealField>,
    wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    propagation_constant_squared: &ArrayBase<OwnedRepr<C>, D>,
    polarisation: Polarisation,
) -> Matrix2<C, D>
where
    M: Material<Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let q = isotropic_layer_quantities(
        material,
        wavenumber,
        propagation_constant_squared,
        polarisation,
    );

    let d = C::from_real(thickness.as_cm());

    let kd = q.kappa.mapv(|k| k * d);
    let coskd = kd.mapv(|x| x.cos());
    let sinkd = kd.mapv(|x| x.sin());

    Matrix2::new(
        -q.kappa.clone() * sinkd.clone(),
        -q.factor.clone() * coskd.clone(),
        q.kappa.mapv(|k| k * k) * coskd / q.factor.view(),
        -q.kappa * sinkd,
    )
}

pub fn isotropic_layer_thickness_second_derivative<M, C, D>(
    material: &M,
    thickness: Thickness<C::RealField>,
    wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    propagation_constant_squared: &ArrayBase<OwnedRepr<C>, D>,
    polarisation: Polarisation,
) -> Matrix2<C, D>
where
    M: Material<Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let q = isotropic_layer_quantities(
        material,
        wavenumber,
        propagation_constant_squared,
        polarisation,
    );

    let d = C::from_real(thickness.as_cm());

    let kd = q.kappa.mapv(|k| k * d);
    let coskd = kd.mapv(|x| x.cos());
    let sinkd = kd.mapv(|x| x.sin());

    let k2 = q.kappa.mapv(|k| k * k);
    let k3 = q.kappa.mapv(|k| k * k * k);

    Matrix2::new(
        -k2.clone() * coskd.clone(),
        q.kappa.clone() * q.factor.clone() * sinkd.clone(),
        -k3 * sinkd / q.factor.view(),
        -k2 * coskd,
    )
}
