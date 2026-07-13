//! Analytic first- and second-order scattering-matrix derivatives.

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::{DerivativeVariable, jet::ArrayJet},
};

use super::ScatterMatrix2;

#[derive(Clone, Debug)]
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
            s11: ArrayJet::value_only(value.s11().clone()),
            s12: ArrayJet::value_only(value.s12().clone()),
            s21: ArrayJet::value_only(value.s21().clone()),
            s22: ArrayJet::value_only(value.s22().clone()),
        }
    }

    pub(crate) fn new(
        value: ScatterMatrix2<C, D>,
        first: ScatterMatrix2<C, D>,
        second: ScatterMatrix2<C, D>,
    ) -> Self {
        Self {
            s11: ArrayJet::with_second(
                value.s11().clone(),
                first.s11().clone(),
                second.s11().clone(),
            ),
            s12: ArrayJet::with_second(
                value.s12().clone(),
                first.s12().clone(),
                second.s12().clone(),
            ),
            s21: ArrayJet::with_second(
                value.s21().clone(),
                first.s21().clone(),
                second.s21().clone(),
            ),
            s22: ArrayJet::with_second(
                value.s22().clone(),
                first.s22().clone(),
                second.s22().clone(),
            ),
        }
    }

    pub(crate) fn star(&self, right: &Self) -> Self {
        let one = ArrayJet::value_only(self.s11.value().mapv(|_| C::one()));

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
            self.s11.value().clone(),
            self.s12.value().clone(),
            self.s21.value().clone(),
            self.s22.value().clone(),
        );

        let first = ScatterMatrix2::new(
            self.s11.first().clone(),
            self.s12.first().clone(),
            self.s21.first().clone(),
            self.s22.first().clone(),
        );

        let second = ScatterMatrix2::new(
            self.s11.second().clone(),
            self.s12.second().clone(),
            self.s21.second().clone(),
            self.s22.second().clone(),
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
