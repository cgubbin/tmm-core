use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::backend::jet::{
    JetAdditive, JetBilinear, JetConjugate, JetConstant, JetCrossProduct, JetHermitianProduct,
    JetOneLike, JetRealPart, JetScaleBy, JetZeroLike,
};

#[derive(Clone, Debug, PartialEq)]
pub struct SpectralJet<A> {
    value: A,

    dk0: A,
    dkx: A,

    dk0_dk0: A,
    dk0_dkx: A,
    dkx_dkx: A,
}

pub type ArraySpectralJet<C, D> = SpectralJet<ArrayBase<OwnedRepr<C>, D>>;

pub struct SpectralGradientRef<'a, A> {
    pub dk0: &'a A,
    pub dkx: &'a A,
}

pub struct SpectralHessianRef<'a, A> {
    pub dk0_dk0: &'a A,
    pub dk0_dkx: &'a A,
    pub dkx_dkx: &'a A,
}

impl<A> SpectralJet<A> {
    pub fn value(&self) -> &A {
        &self.value
    }

    pub fn dk0(&self) -> &A {
        &self.dk0
    }

    pub fn dkx(&self) -> &A {
        &self.dkx
    }

    pub fn dk0_dk0(&self) -> &A {
        &self.dk0_dk0
    }

    pub fn dk0_dkx(&self) -> &A {
        &self.dk0_dkx
    }

    pub fn dkx_dkx(&self) -> &A {
        &self.dkx_dkx
    }

    pub fn spectral_gradient(&self) -> SpectralGradientRef<'_, A> {
        SpectralGradientRef {
            dk0: self.dk0(),
            dkx: self.dkx(),
        }
    }

    pub fn spectral_hessian(&self) -> SpectralHessianRef<'_, A> {
        SpectralHessianRef {
            dk0_dk0: self.dk0_dk0(),
            dk0_dkx: self.dk0_dkx(),
            dkx_dkx: self.dkx_dkx(),
        }
    }

    pub fn from_parts(value: A, dk0: A, dkx: A, dk0_dk0: A, dk0_dkx: A, dkx_dkx: A) -> Self {
        Self {
            value,
            dk0,
            dkx,
            dk0_dk0,
            dk0_dkx,
            dkx_dkx,
        }
    }
}

impl<I> SpectralJet<I>
where
    I: JetAdditive,
{
    pub(crate) fn add(&self, rhs: &Self) -> Self {
        Self {
            value: self.value.jet_add(&rhs.value),
            dk0: self.dk0.jet_add(&rhs.dk0),
            dkx: self.dkx.jet_add(&rhs.dkx),
            dk0_dk0: self.dk0_dk0.jet_add(&rhs.dk0_dk0),
            dk0_dkx: self.dk0_dkx.jet_add(&rhs.dk0_dkx),
            dkx_dkx: self.dkx_dkx.jet_add(&rhs.dkx_dkx),
        }
    }

    pub(crate) fn subtract(&self, rhs: &Self) -> Self {
        Self {
            value: self.value.jet_subtract(&rhs.value),
            dk0: self.dk0.jet_subtract(&rhs.dk0),
            dkx: self.dkx.jet_subtract(&rhs.dkx),
            dk0_dk0: self.dk0_dk0.jet_subtract(&rhs.dk0_dk0),
            dk0_dkx: self.dk0_dkx.jet_subtract(&rhs.dk0_dkx),
            dkx_dkx: self.dkx_dkx.jet_subtract(&rhs.dkx_dkx),
        }
    }

    pub(crate) fn negate(&self) -> Self {
        Self {
            value: self.value.jet_negate(),
            dk0: self.dk0.jet_negate(),
            dkx: self.dkx.jet_negate(),
            dk0_dk0: self.dk0_dk0.jet_negate(),
            dk0_dkx: self.dk0_dkx.jet_negate(),
            dkx_dkx: self.dkx_dkx.jet_negate(),
        }
    }
}

impl<I> SpectralJet<I>
where
    I: JetBilinear,
{
    /// Multiply two second-order jets using the product rules.
    pub(crate) fn multiply(&self, rhs: &Self) -> Self {
        let value = self.value.jet_add(&rhs.value);

        let dk0 = self
            .dk0
            .jet_multiply(&rhs.value)
            .jet_add(&self.value.jet_multiply(&rhs.dk0));

        let dkx = self
            .dkx
            .jet_multiply(&rhs.value)
            .jet_add(&self.value.jet_multiply(&rhs.dkx));

        let dk0_dk0 = self
            .dk0_dk0
            .jet_multiply(&rhs.value)
            .jet_add(&(&self.dk0.jet_multiply(&rhs.dk0)).jet_double())
            .jet_add(&self.value.jet_multiply(&rhs.dk0_dk0));

        let dk0_dkx = self
            .dk0_dkx
            .jet_multiply(&rhs.value)
            .jet_add(&self.dk0.jet_multiply(&rhs.dkx))
            .jet_add(&self.dkx.jet_multiply(&rhs.dk0))
            .jet_add(&self.value.jet_multiply(&rhs.dk0_dkx));

        let dkx_dkx = self
            .dkx_dkx
            .jet_multiply(&rhs.value)
            .jet_add(&(&self.dkx.jet_multiply(&rhs.dkx)).jet_double())
            .jet_add(&self.value.jet_multiply(&rhs.dkx_dkx));

        Self {
            value,
            dk0,
            dkx,
            dk0_dk0,
            dk0_dkx,
            dkx_dkx,
        }
    }
}

impl<I> SpectralJet<I>
where
    I: JetConstant + JetZeroLike,
{
    pub(crate) fn constant(value: I) -> Self {
        let zeros = I::jet_zeros_like(&value);
        Self {
            value,
            dk0: zeros.clone(),
            dkx: zeros.clone(),
            dk0_dk0: zeros.clone(),
            dk0_dkx: zeros.clone(),
            dkx_dkx: zeros,
        }
    }

    /// Construct a constant second-order jet with zero derivatives.
    pub(crate) fn constant_like(source: &I, value: I::Scalar) -> Self {
        Self::constant(source.constant_like(value))
    }

    pub fn vacuum_wavenumber(value: I) -> Self
    where
        I: JetOneLike,
    {
        let zeros = I::jet_zeros_like(&value);
        let ones = I::jet_ones_like(&value);

        Self {
            value,
            dk0: ones,
            dkx: zeros.clone(),
            dk0_dk0: zeros.clone(),
            dk0_dkx: zeros.clone(),
            dkx_dkx: zeros.clone(),
        }
    }

    pub fn parallel_wavenumber(value: I) -> Self
    where
        I: JetOneLike,
    {
        let zeros = I::jet_zeros_like(&value);
        let ones = I::jet_ones_like(&value);

        Self {
            value,
            dk0: zeros.clone(),
            dkx: ones,
            dk0_dk0: zeros.clone(),
            dk0_dkx: zeros.clone(),
            dkx_dkx: zeros.clone(),
        }
    }

    pub fn from_k0_derivatives(value: I, first: I, second: I) -> Self {
        let zeros = I::jet_zeros_like(&value);

        Self {
            value,
            dk0: first,
            dkx: zeros.clone(),
            dk0_dk0: second,
            dk0_dkx: zeros.clone(),
            dkx_dkx: zeros,
        }
    }
}

impl<I> SpectralJet<I>
where
    I: JetScaleBy,
{
    /// Construct a constant second-order jet with zero derivatives.
    pub(crate) fn scale_by(&self, value: I::Scalar) -> Self {
        Self {
            value: self.value.jet_scale_by(value),
            dk0: self.dk0.jet_scale_by(value),
            dkx: self.dkx.jet_scale_by(value),
            dk0_dk0: self.dk0_dk0.jet_scale_by(value),
            dk0_dkx: self.dk0_dkx.jet_scale_by(value),
            dkx_dkx: self.dkx_dkx.jet_scale_by(value),
        }
    }
}

impl<I> SpectralJet<I>
where
    I: JetConjugate,
{
    pub fn conjugate(&self) -> Self {
        Self {
            value: self.value.jet_conjugate(),

            dk0: self.dk0.jet_conjugate(),
            dkx: self.dkx.jet_conjugate(),

            dk0_dk0: self.dk0_dk0.jet_conjugate(),
            dk0_dkx: self.dk0_dkx.jet_conjugate(),
            dkx_dkx: self.dkx_dkx.jet_conjugate(),
        }
    }
}

impl<I> SpectralJet<I>
where
    I: JetRealPart,
{
    pub fn real_part(&self) -> SpectralJet<I::RealOutput> {
        SpectralJet {
            value: self.value.jet_real(),

            dk0: self.dk0.jet_real(),
            dkx: self.dkx.jet_real(),

            dk0_dk0: self.dk0_dk0.jet_real(),
            dk0_dkx: self.dk0_dkx.jet_real(),
            dkx_dkx: self.dkx_dkx.jet_real(),
        }
    }
}

impl<I> SpectralJet<I>
where
    I: JetCrossProduct + JetAdditive,
{
    /// Compute the cross product of two bivariate second-order jets.
    pub(crate) fn cross(&self, rhs: &Self) -> Self {
        let value = self.value().jet_cross(rhs.value());

        let dk0 = self
            .dk0()
            .jet_cross(rhs.value())
            .jet_add(&self.value().jet_cross(rhs.dk0()));

        let dkx = self
            .dkx()
            .jet_cross(rhs.value())
            .jet_add(&self.value().jet_cross(rhs.dkx()));

        let mixed_k0 = self.dk0().jet_cross(rhs.dk0());

        let dk0_dk0 = self
            .dk0_dk0()
            .jet_cross(rhs.value())
            .jet_add(&mixed_k0)
            .jet_add(&mixed_k0)
            .jet_add(&self.value().jet_cross(rhs.dk0_dk0()));

        let mixed_kx = self.dkx().jet_cross(rhs.dkx());

        let dkx_dkx = self
            .dkx_dkx()
            .jet_cross(rhs.value())
            .jet_add(&mixed_kx)
            .jet_add(&mixed_kx)
            .jet_add(&self.value().jet_cross(rhs.dkx_dkx()));

        let dk0_dkx = self
            .dk0_dkx()
            .jet_cross(rhs.value())
            .jet_add(&self.dk0().jet_cross(rhs.dkx()))
            .jet_add(&self.dkx().jet_cross(rhs.dk0()))
            .jet_add(&self.value().jet_cross(rhs.dk0_dkx()));

        Self::from_parts(value, dk0, dkx, dk0_dk0, dk0_dkx, dkx_dkx)
    }
}

impl<I> SpectralJet<I>
where
    I: JetHermitianProduct,
    I::Output: JetAdditive,
{
    /// Compute the Hermitian product of two bivariate second-order jets.
    pub(crate) fn hermitian_dot_product(&self, rhs: &Self) -> SpectralJet<I::Output> {
        let value = self.value().jet_hermitian_product(rhs.value());

        let dk0 = self
            .dk0()
            .jet_hermitian_product(rhs.value())
            .jet_add(&self.value().jet_hermitian_product(rhs.dk0()));

        let dkx = self
            .dkx()
            .jet_hermitian_product(rhs.value())
            .jet_add(&self.value().jet_hermitian_product(rhs.dkx()));

        let mixed_k0 = self.dk0().jet_hermitian_product(rhs.dk0());

        let dk0_dk0 = self
            .dk0_dk0()
            .jet_hermitian_product(rhs.value())
            .jet_add(&mixed_k0)
            .jet_add(&mixed_k0)
            .jet_add(&self.value().jet_hermitian_product(rhs.dk0_dk0()));

        let mixed_kx = self.dkx().jet_hermitian_product(rhs.dkx());

        let dkx_dkx = self
            .dkx_dkx()
            .jet_hermitian_product(rhs.value())
            .jet_add(&mixed_kx)
            .jet_add(&mixed_kx)
            .jet_add(&self.value().jet_hermitian_product(rhs.dkx_dkx()));

        let dk0_dkx = self
            .dk0_dkx()
            .jet_hermitian_product(rhs.value())
            .jet_add(&self.dk0().jet_hermitian_product(rhs.dkx()))
            .jet_add(&self.dkx().jet_hermitian_product(rhs.dk0()))
            .jet_add(&self.value().jet_hermitian_product(rhs.dk0_dkx()));

        SpectralJet::from_parts(value, dk0, dkx, dk0_dk0, dk0_dkx, dkx_dkx)
    }
}

impl<C, D> ArraySpectralJet<C, D>
where
    C: ComplexField,
    D: Dimension,
{
    pub fn map_unary<F, F1, F2>(&self, function: F, first: F1, second: F2) -> Self
    where
        F: Fn(&C) -> C,
        F1: Fn(&C) -> C,
        F2: Fn(&C) -> C,
    {
        let value = self.value.mapv(|x| function(&x));
        let g1 = self.value.mapv(|x| first(&x));
        let g2 = self.value.mapv(|x| second(&x));

        let dk0 = &g1 * &self.dk0;

        let dkx = &g1 * &self.dkx;

        let dk0_dk0 = &g2 * &self.dk0 * &self.dk0 + &g1 * &self.dk0_dk0;

        let dk0_dkx = &g2 * &self.dk0 * &self.dkx + &g1 * &self.dk0_dkx;

        let dkx_dkx = &g2 * &self.dkx * &self.dkx + &g1 * &self.dkx_dkx;

        Self {
            value,
            dk0,
            dkx,
            dk0_dk0,
            dk0_dkx,
            dkx_dkx,
        }
    }

    pub fn exp(&self) -> Self
    where
        C: Copy,
    {
        self.map_unary(|x| x.exp(), |x| x.exp(), |x| x.exp())
    }

    pub fn sin(&self) -> Self
    where
        C: Copy,
    {
        self.map_unary(|x| x.sin(), |x| x.cos(), |x| -x.sin())
    }

    pub fn cos(&self) -> Self
    where
        C: Copy,
    {
        self.map_unary(|x| x.cos(), |x| -x.sin(), |x| -x.cos())
    }

    pub fn sqrt(&self) -> Self
    where
        C: Copy,
    {
        self.map_unary(
            |x| x.sqrt(),
            |x| C::one() / ((C::one() + C::one()) * x.sqrt()),
            |x| {
                let two = C::one() + C::one();
                let four = two * two;

                -C::one() / (four * x.sqrt() * *x)
            },
        )
    }

    pub fn reciprocal(&self) -> Self
    where
        C: Copy,
    {
        self.map_unary(
            |x| C::one() / *x,
            |x| -C::one() / (*x * *x),
            |x| {
                let two = C::one() + C::one();
                two / (*x * *x * *x)
            },
        )
    }
}
