//! Redheffer accumulation for scalar-channel scattering matrices.

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::{DerivativeVariable, MatrixDerivatives, MatrixEvaluation},
};

use super::{ScatterMatrix2, derivatives::ScatterJet2, matrix::star_product};

pub(crate) struct ScatterAccumulator<C, D>
where
    D: Dimension,
{
    value: ScatterMatrix2<C, D>,
    jet: Option<ScatterJet2<C, D>>,
    variable: Option<DerivativeVariable>,
    second_requested: bool,
}

impl<C, D> ScatterAccumulator<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn new(shape_source: &ArrayBase<OwnedRepr<C>, D>) -> Self {
        Self {
            value: ScatterMatrix2::identity_like(shape_source),
            jet: None,
            variable: None,
            second_requested: false,
        }
    }

    pub(crate) fn update(&mut self, element: &ScatterMatrix2<C, D>) {
        self.value = star_product(&self.value, element);
    }

    pub(crate) fn update_first(
        &mut self,
        variable: DerivativeVariable,
        value: ScatterMatrix2<C, D>,
        first: ScatterMatrix2<C, D>,
    ) {
        self.variable = Some(variable);

        let second = ScatterMatrix2::zeros_like(value.s11());
        let element = ScatterJet2::new(value, first, second);

        let current = self
            .jet
            .take()
            .unwrap_or_else(|| ScatterJet2::from_value(self.value.clone()));

        let next = current.star(&element);
        let (value, _, _) = next.clone().into_parts();

        self.value = value;
        self.jet = Some(next);
    }

    pub(crate) fn update_second(
        &mut self,
        variable: DerivativeVariable,
        value: ScatterMatrix2<C, D>,
        first: ScatterMatrix2<C, D>,
        second: ScatterMatrix2<C, D>,
    ) {
        self.variable = Some(variable);
        self.second_requested = true;

        let element = ScatterJet2::new(value, first, second);

        let current = self
            .jet
            .take()
            .unwrap_or_else(|| ScatterJet2::from_value(self.value.clone()));

        let next = current.star(&element);
        let (value, _, _) = next.clone().into_parts();

        self.value = value;
        self.jet = Some(next);
    }

    pub(crate) fn finish(self) -> MatrixEvaluation<ScatterMatrix2<C, D>> {
        let Some(jet) = self.jet else {
            return MatrixEvaluation::new(self.value);
        };

        let variable = self
            .variable
            .expect("a derivative jet must have a variable");

        let (value, first, second) = jet.into_parts();

        let derivatives = if self.second_requested {
            MatrixDerivatives::new(variable, first).with_second(second)
        } else {
            MatrixDerivatives::new(variable, first)
        };

        MatrixEvaluation::with_derivatives(value, derivatives)
    }
}
