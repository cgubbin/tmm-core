use super::{DerivativeOrder, Material, Scalar, SpectralVariable, TensorSampled};

use crate::{
    ComplexScalar,
    tensor::{Tensor3, diagonal3},
};

pub trait TensorMaterial {
    type Real;

    fn relative_permittivity_tensor<I, C>(&self, wavenumber: I) -> I::TensorOutput<C>
    where
        I: TensorSampled<Elem = C>,
        C: ComplexScalar<RealField = Self::Real>;

    fn relative_permittivity_tensor_derivative<I, C>(
        &self,
        wavenumber: I,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> I::TensorOutput<C>
    where
        I: TensorSampled<Elem = C>,
        C: ComplexScalar<RealField = Self::Real>;

    fn relative_permeability_tensor<I, C>(&self, wavenumber: I) -> I::TensorOutput<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: TensorSampled<Elem = C>;

    fn relative_permeability_tensor_derivative<I, C>(
        &self,
        wavenumber: I,
        order: DerivativeOrder,
        variable: SpectralVariable,
    ) -> I::TensorOutput<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: TensorSampled<Elem = C>;
}

impl<M> TensorMaterial for M
where
    M: Material,
{
    type Real = M::Real;

    fn relative_permittivity_tensor<I, C>(&self, wavenumber: I) -> I::TensorOutput<C>
    where
        I: TensorSampled<Elem = C>,
        C: ComplexScalar<RealField = Self::Real>,
    {
        wavenumber.map_tensor3(|w| {
            let eps = self.relative_permittivity(Scalar(w));
            diagonal3(eps, eps, eps)
        })
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
        wavenumber.map_tensor3(|w| {
            let deps = self.relative_permittivity_derivative(Scalar(w), order, variable);
            diagonal3(deps, deps, deps)
        })
    }

    fn relative_permeability_tensor<I, C>(&self, wavenumber: I) -> I::TensorOutput<C>
    where
        I: TensorSampled<Elem = C>,
        C: ComplexScalar<RealField = Self::Real>,
    {
        wavenumber.map_tensor3(|w| {
            let eps = self.relative_permeability(Scalar(w));
            diagonal3(eps, eps, eps)
        })
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
        wavenumber.map_tensor3(|w| {
            let eps = self.relative_permeability_derivative(Scalar(w), order, variable);
            diagonal3(eps, eps, eps)
        })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DiagonalTensorMaterial<Mx, My, Mz> {
    pub xx: Mx,
    pub yy: My,
    pub zz: Mz,
}

impl<Mx, My, Mz> DiagonalTensorMaterial<Mx, My, Mz> {
    pub fn new(xx: Mx, yy: My, zz: Mz) -> Self {
        Self { xx, yy, zz }
    }
}

impl<Mo, Me> DiagonalTensorMaterial<Mo, Mo, Me>
where
    Mo: Clone,
{
    pub fn uniaxial(ordinary: Mo, extraordinary: Me) -> Self {
        Self {
            xx: ordinary.clone(),
            yy: ordinary,
            zz: extraordinary,
        }
    }
}

impl<Mx, My, Mz> TensorMaterial for DiagonalTensorMaterial<Mx, My, Mz>
where
    Mx: Material,
    My: Material<Real = Mx::Real>,
    Mz: Material<Real = Mx::Real>,
{
    type Real = Mx::Real;

    fn relative_permittivity_tensor<I, C>(&self, wavenumber: I) -> I::TensorOutput<C>
    where
        I: TensorSampled<Elem = C>,
        C: ComplexScalar<RealField = Self::Real>,
    {
        wavenumber.map_tensor3(|w| {
            diagonal3(
                self.xx.relative_permittivity(Scalar(w)),
                self.yy.relative_permittivity(Scalar(w)),
                self.zz.relative_permittivity(Scalar(w)),
            )
        })
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
        wavenumber.map_tensor3(|w| {
            diagonal3(
                self.xx
                    .relative_permittivity_derivative(Scalar(w), order, variable),
                self.yy
                    .relative_permittivity_derivative(Scalar(w), order, variable),
                self.zz
                    .relative_permittivity_derivative(Scalar(w), order, variable),
            )
        })
    }

    fn relative_permeability_tensor<I, C>(&self, wavenumber: I) -> I::TensorOutput<C>
    where
        C: ComplexScalar<RealField = Self::Real> + Copy,
        I: TensorSampled<Elem = C>,
    {
        wavenumber.map_tensor3(|w| {
            diagonal3(
                self.xx.relative_permeability(Scalar(w)),
                self.yy.relative_permeability(Scalar(w)),
                self.zz.relative_permeability(Scalar(w)),
            )
        })
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
        wavenumber.map_tensor3(|w| {
            diagonal3(
                self.xx
                    .relative_permeability_derivative(Scalar(w), order, variable),
                self.yy
                    .relative_permeability_derivative(Scalar(w), order, variable),
                self.zz
                    .relative_permeability_derivative(Scalar(w), order, variable),
            )
        })
    }
}

pub struct RotatedTensorMaterial<M: TensorMaterial> {
    pub principal: M,
    pub rotation: nalgebra::Rotation3<M::Real>,
}
