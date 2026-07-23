use crate::{
    algebra::{ArrayJet, ArrayJetBivariate, ArrayJetFirst},
    crystallise::{DirectionalFirstParts, DirectionalSecondParts, SpectralSecondParts},
};

use ndarray::{ArrayBase, Dimension, OwnedRepr};

/// Extract the value represented by an algebraic storage type.
pub(crate) trait IntoValue {
    type Value;

    fn into_value(self) -> Self::Value;
}

/// Extract a value and one first directional derivative.
pub(crate) trait IntoDirectionalFirstParts {
    type Value;

    fn into_directional_first_parts(self) -> DirectionalFirstParts<Self::Value>;
}

/// Extract a value and first and second directional derivatives.
pub(crate) trait IntoDirectionalSecondParts {
    type Value;

    fn into_directional_second_parts(self) -> DirectionalSecondParts<Self::Value>;
}

/// Extract a value and the canonical spectral gradient and Hessian.
pub(crate) trait IntoSpectralSecondParts {
    type Value;

    fn into_spectral_second_parts(self) -> SpectralSecondParts<Self::Value>;
}

impl<T, D> IntoValue for ArrayBase<OwnedRepr<T>, D>
where
    D: Dimension,
{
    type Value = Self;

    fn into_value(self) -> Self::Value {
        self
    }
}

impl<T, D, P> IntoValue for ArrayJetFirst<T, D, P>
where
    D: Dimension,
{
    type Value = ArrayBase<OwnedRepr<T>, D>;

    fn into_value(self) -> Self::Value {
        let (value, ..) = self.into_parts();
        value
    }
}

impl<T, D, P> IntoDirectionalFirstParts for ArrayJetFirst<T, D, P>
where
    D: Dimension,
{
    type Value = ArrayBase<OwnedRepr<T>, D>;

    fn into_directional_first_parts(self) -> DirectionalFirstParts<Self::Value> {
        let (value, first) = self.into_parts();
        DirectionalFirstParts::new(value, first)
    }
}

impl<T, D, P> IntoValue for ArrayJet<T, D, P>
where
    D: Dimension,
{
    type Value = ArrayBase<OwnedRepr<T>, D>;

    fn into_value(self) -> Self::Value {
        let (value, ..) = self.into_parts();
        value
    }
}

impl<T, D, P> IntoDirectionalFirstParts for ArrayJet<T, D, P>
where
    D: Dimension,
{
    type Value = ArrayBase<OwnedRepr<T>, D>;

    fn into_directional_first_parts(self) -> DirectionalFirstParts<Self::Value> {
        let (value, first, ..) = self.into_parts();
        DirectionalFirstParts::new(value, first)
    }
}

impl<T, D, P> IntoDirectionalSecondParts for ArrayJet<T, D, P>
where
    D: Dimension,
{
    type Value = ArrayBase<OwnedRepr<T>, D>;

    fn into_directional_second_parts(self) -> DirectionalSecondParts<Self::Value> {
        let (value, first, second) = self.into_parts();
        DirectionalSecondParts::new(value, first, second)
    }
}

impl<T, D, P> IntoValue for ArrayJetBivariate<T, D, P>
where
    D: Dimension,
{
    type Value = ArrayBase<OwnedRepr<T>, D>;

    fn into_value(self) -> Self::Value {
        let (value, ..) = self.into_parts();
        value
    }
}

impl<T, D, P> IntoSpectralSecondParts for ArrayJetBivariate<T, D, P>
where
    D: Dimension,
{
    type Value = ArrayBase<OwnedRepr<T>, D>;

    fn into_spectral_second_parts(self) -> SpectralSecondParts<Self::Value> {
        let (value, gradient, hessian) = self.into_parts();

        let (vacuum_wavenumber, parallel_wavenumber) = gradient.into_parts();

        let (
            vacuum_wavenumber_vacuum_wavenumber,
            vacuum_wavenumber_parallel_wavenumber,
            parallel_wavenumber_parallel_wavenumber,
        ) = hessian.into_parts();

        SpectralSecondParts::new(
            value,
            vacuum_wavenumber,
            parallel_wavenumber,
            vacuum_wavenumber_vacuum_wavenumber,
            vacuum_wavenumber_parallel_wavenumber,
            parallel_wavenumber_parallel_wavenumber,
        )
    }
}
