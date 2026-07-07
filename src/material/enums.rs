use crate::{
    ComplexScalar,
    material::{
        Constant, DerivativeOrder, DiagonalTensorMaterial, DrudeLorentz, Material, Sampled, Scalar,
        SpectralVariable, TensorMaterial, TensorSampled,
    },
    tensor::{Tensor3, diagonal3},
};

use num_traits::{One, Zero};

#[derive(Clone, Debug, PartialEq)]
pub enum IsotropicMaterial<R> {
    Constant(Constant<R>),
    DrudeLorentz(DrudeLorentz<R>),
}

impl<R> From<Constant<R>> for IsotropicMaterial<R> {
    fn from(value: Constant<R>) -> Self {
        Self::Constant(value)
    }
}

impl<R> From<DrudeLorentz<R>> for IsotropicMaterial<R> {
    fn from(value: DrudeLorentz<R>) -> Self {
        Self::DrudeLorentz(value)
    }
}

impl<R> Material for IsotropicMaterial<R>
where
    R: One + Zero,
    Constant<R>: Material<Real = R>,
    DrudeLorentz<R>: Material<Real = R>,
{
    type Real = R;

    fn is_dispersive(&self) -> bool {
        match self {
            Self::Constant(material) => Material::is_dispersive(material),
            Self::DrudeLorentz(material) => Material::is_dispersive(material),
        }
    }

    fn static_permittivity(&self) -> Self::Real {
        match self {
            Self::Constant(material) => material.static_permittivity(),
            Self::DrudeLorentz(material) => material.static_permittivity(),
        }
    }

    fn relative_permittivity<I, C>(&self, wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
        C: ComplexScalar<RealField = Self::Real>,
    {
        match self {
            Self::Constant(material) => material.relative_permittivity(wavenumber),
            Self::DrudeLorentz(material) => material.relative_permittivity(wavenumber),
        }
    }

    fn relative_permittivity_derivative<I, C>(
        &self,
        wavenumber: I,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
        C: ComplexScalar<RealField = Self::Real>,
    {
        match self {
            Self::Constant(material) => {
                material.relative_permittivity_derivative(wavenumber, order, variable)
            }
            Self::DrudeLorentz(material) => {
                material.relative_permittivity_derivative(wavenumber, order, variable)
            }
        }
    }

    fn relative_permeability<I, C>(&self, wavenumber: I) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
        C: ComplexScalar<RealField = Self::Real>,
    {
        match self {
            Self::Constant(material) => material.relative_permeability(wavenumber),
            Self::DrudeLorentz(material) => material.relative_permeability(wavenumber),
        }
    }

    fn relative_permeability_derivative<I, C>(
        &self,
        wavenumber: I,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> I::Mapped<C>
    where
        I: Sampled<Elem = C>,
        C: ComplexScalar<RealField = Self::Real>,
    {
        match self {
            Self::Constant(material) => {
                material.relative_permeability_derivative(wavenumber, order, variable)
            }
            Self::DrudeLorentz(material) => {
                material.relative_permeability_derivative(wavenumber, order, variable)
            }
        }
    }
}

pub type PrincipalTensorMaterial<R> =
    DiagonalTensorMaterial<IsotropicMaterial<R>, IsotropicMaterial<R>, IsotropicMaterial<R>>;

#[derive(Clone, Debug, PartialEq)]
pub enum MaterialModel<R> {
    Isotropic(IsotropicMaterial<R>),
    Diagonal(PrincipalTensorMaterial<R>),
}

impl<R> From<IsotropicMaterial<R>> for MaterialModel<R> {
    fn from(value: IsotropicMaterial<R>) -> Self {
        Self::Isotropic(value)
    }
}

impl<R> From<Constant<R>> for MaterialModel<R> {
    fn from(value: Constant<R>) -> Self {
        Self::Isotropic(value.into())
    }
}

impl<R> From<DrudeLorentz<R>> for MaterialModel<R> {
    fn from(value: DrudeLorentz<R>) -> Self {
        Self::Isotropic(value.into())
    }
}

impl<R> From<PrincipalTensorMaterial<R>> for MaterialModel<R> {
    fn from(value: PrincipalTensorMaterial<R>) -> Self {
        Self::Diagonal(value)
    }
}

impl<R> TensorMaterial for MaterialModel<R>
where
    IsotropicMaterial<R>: Material<Real = R>,
    PrincipalTensorMaterial<R>: TensorMaterial<Real = R>,
{
    type Real = R;

    fn is_dispersive(&self) -> bool {
        match self {
            Self::Isotropic(material) => Material::is_dispersive(material),
            Self::Diagonal(material) => TensorMaterial::is_dispersive(material),
        }
    }

    fn static_permittivity_tensor<C>(&self) -> Tensor3<C>
    where
        C: ComplexScalar<RealField = Self::Real>,
    {
        match self {
            Self::Isotropic(material) => {
                let eps = material.static_permittivity();
                let eps = C::from_real(eps);
                diagonal3(eps, eps, eps)
            }
            Self::Diagonal(material) => material.static_permittivity_tensor(),
        }
    }

    fn relative_permittivity_tensor<I, C>(&self, wavenumber: I) -> I::TensorOutput<C>
    where
        I: TensorSampled<Elem = C>,
        C: ComplexScalar<RealField = Self::Real>,
    {
        match self {
            Self::Isotropic(material) => wavenumber.map_tensor3(|w| {
                let eps = material.relative_permittivity(Scalar(w));
                diagonal3(eps, eps, eps)
            }),
            Self::Diagonal(material) => material.relative_permittivity_tensor(wavenumber),
        }
    }

    fn relative_permittivity_tensor_derivative<I, C>(
        &self,
        wavenumber: I,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> I::TensorOutput<C>
    where
        I: TensorSampled<Elem = C>,
        C: ComplexScalar<RealField = Self::Real>,
    {
        match self {
            Self::Isotropic(material) => wavenumber.map_tensor3(|w| {
                let deps = material.relative_permittivity_derivative(Scalar(w), order, variable);
                diagonal3(deps, deps, deps)
            }),
            Self::Diagonal(material) => {
                material.relative_permittivity_tensor_derivative(wavenumber, order, variable)
            }
        }
    }

    fn relative_permeability_tensor<I, C>(&self, wavenumber: I) -> I::TensorOutput<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: TensorSampled<Elem = C>,
    {
        match self {
            Self::Isotropic(material) => wavenumber.map_tensor3(|w| {
                let mu = material.relative_permeability(Scalar(w));
                diagonal3(mu, mu, mu)
            }),
            Self::Diagonal(material) => material.relative_permeability_tensor(wavenumber),
        }
    }

    fn relative_permeability_tensor_derivative<I, C>(
        &self,
        wavenumber: I,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> I::TensorOutput<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: TensorSampled<Elem = C>,
    {
        match self {
            Self::Isotropic(material) => wavenumber.map_tensor3(|w| {
                let dmu = material.relative_permeability_derivative(Scalar(w), order, variable);
                diagonal3(dmu, dmu, dmu)
            }),
            Self::Diagonal(material) => {
                material.relative_permittivity_tensor_derivative(wavenumber, order, variable)
            }
        }
    }
}
