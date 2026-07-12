//! Analytic first- and second-order scattering-matrix derivatives.

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{ComplexScalar, backend::DerivativeVariable};

use super::ScatterMatrix2;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ArrayJet<C, D>
where
    D: Dimension,
{
    value: ArrayBase<OwnedRepr<C>, D>,
    first: ArrayBase<OwnedRepr<C>, D>,
    second: ArrayBase<OwnedRepr<C>, D>,
}

impl<C, D> ArrayJet<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn new(
        value: ArrayBase<OwnedRepr<C>, D>,
        first: ArrayBase<OwnedRepr<C>, D>,
        second: ArrayBase<OwnedRepr<C>, D>,
    ) -> Self {
        Self {
            value,
            first,
            second,
        }
    }

    pub(crate) fn constant(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        let zero = value.mapv(|_| C::zero());

        Self::new(value, zero.clone(), zero)
    }

    pub(crate) fn add(&self, rhs: &Self) -> Self {
        Self::new(
            self.value.clone() + rhs.value.view(),
            self.first.clone() + rhs.first.view(),
            self.second.clone() + rhs.second.view(),
        )
    }

    pub(crate) fn subtract(&self, rhs: &Self) -> Self {
        Self::new(
            self.value.clone() - rhs.value.view(),
            self.first.clone() - rhs.first.view(),
            self.second.clone() - rhs.second.view(),
        )
    }

    pub(crate) fn multiply(&self, rhs: &Self) -> Self {
        let two = C::one() + C::one();

        Self::new(
            self.value.clone() * rhs.value.view(),
            self.first.clone() * rhs.value.view() + self.value.clone() * rhs.first.view(),
            self.second.clone() * rhs.value.view()
                + (self.first.clone() * rhs.first.view()).mapv(|x| two * x)
                + self.value.clone() * rhs.second.view(),
        )
    }

    pub(crate) fn reciprocal(&self) -> Self {
        let two = C::one() + C::one();

        let value_squared = self.value.mapv(|x| x * x);
        let value_cubed = self.value.mapv(|x| x * x * x);

        Self::new(
            self.value.mapv(|x| C::one() / x),
            -self.first.clone() / value_squared.view(),
            self.first.mapv(|x| two * x * x) / value_cubed.view()
                - self.second.clone() / value_squared,
        )
    }

    pub(crate) fn divide(&self, rhs: &Self) -> Self {
        self.multiply(&rhs.reciprocal())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct ScatterJet2<C, D>
where
    D: Dimension,
{
    s11: ArrayJet<C, D>,
    s12: ArrayJet<C, D>,
    s21: ArrayJet<C, D>,
    s22: ArrayJet<C, D>,
}

impl<C, D> ScatterJet2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn from_value(value: ScatterMatrix2<C, D>) -> Self {
        Self {
            s11: ArrayJet::constant(value.s11().clone()),
            s12: ArrayJet::constant(value.s12().clone()),
            s21: ArrayJet::constant(value.s21().clone()),
            s22: ArrayJet::constant(value.s22().clone()),
        }
    }

    pub(crate) fn new(
        value: ScatterMatrix2<C, D>,
        first: ScatterMatrix2<C, D>,
        second: ScatterMatrix2<C, D>,
    ) -> Self {
        Self {
            s11: ArrayJet::new(
                value.s11().clone(),
                first.s11().clone(),
                second.s11().clone(),
            ),
            s12: ArrayJet::new(
                value.s12().clone(),
                first.s12().clone(),
                second.s12().clone(),
            ),
            s21: ArrayJet::new(
                value.s21().clone(),
                first.s21().clone(),
                second.s21().clone(),
            ),
            s22: ArrayJet::new(
                value.s22().clone(),
                first.s22().clone(),
                second.s22().clone(),
            ),
        }
    }

    pub(crate) fn star(&self, right: &Self) -> Self {
        let one = ArrayJet::constant(self.s11.value.mapv(|_| C::one()));

        let denominator = one.subtract(&right.s11.multiply(&self.s22));

        let s11 = self.s11.add(
            &self
                .s12
                .multiply(&right.s11)
                .multiply(&self.s21)
                .divide(&denominator),
        );

        let s12 = self.s12.multiply(&right.s12).divide(&denominator);

        let s21 = right.s21.multiply(&self.s21).divide(&denominator);

        let s22 = right.s22.add(
            &right
                .s21
                .multiply(&self.s22)
                .multiply(&right.s12)
                .divide(&denominator),
        );

        Self { s11, s12, s21, s22 }
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        ScatterMatrix2<C, D>,
        ScatterMatrix2<C, D>,
        ScatterMatrix2<C, D>,
    ) {
        let value = ScatterMatrix2::new(
            self.s11.value,
            self.s12.value,
            self.s21.value,
            self.s22.value,
        );

        let first = ScatterMatrix2::new(
            self.s11.first,
            self.s12.first,
            self.s21.first,
            self.s22.first,
        );

        let second = ScatterMatrix2::new(
            self.s11.second,
            self.s12.second,
            self.s21.second,
            self.s22.second,
        );

        (value, first, second)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ScatterDerivatives<C, D>
where
    D: Dimension,
{
    variable: DerivativeVariable,
    first: ScatterMatrix2<C, D>,
    second: Option<ScatterMatrix2<C, D>>,
}

impl<C, D> ScatterDerivatives<C, D>
where
    D: Dimension,
{
    pub fn new(variable: DerivativeVariable, first: ScatterMatrix2<C, D>) -> Self {
        Self {
            variable,
            first,
            second: None,
        }
    }

    pub fn with_second(mut self, second: ScatterMatrix2<C, D>) -> Self {
        self.second = Some(second);
        self
    }

    pub fn variable(&self) -> DerivativeVariable {
        self.variable
    }

    pub fn first(&self) -> &ScatterMatrix2<C, D> {
        &self.first
    }

    pub fn second(&self) -> Option<&ScatterMatrix2<C, D>> {
        self.second.as_ref()
    }
}
