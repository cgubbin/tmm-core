use crate::{
    derivative_parts::{
        BivariateFirstParts, BivariateSecondParts, DirectionalFirstParts, DirectionalSecondParts,
        IntoBivariateFirst, IntoBivariateSecond, IntoFirst, IntoSecond, IntoValue, ValuePart,
    },
    observable::PlaneWaveDeterminant,
};

/// Extract the value components of a determinant
impl<J> IntoValue for PlaneWaveDeterminant<J>
where
    J: IntoValue,
{
    type Value = PlaneWaveDeterminant<J::Value>;

    fn into_value(self) -> ValuePart<Self::Value> {
        let determinant = self.into_inner();

        ValuePart::new(PlaneWaveDeterminant::new(
            determinant.into_value().into_inner(),
        ))
    }
}

/// Separate a determinant into a value and first directional derivative
impl<J> IntoFirst for PlaneWaveDeterminant<J>
where
    J: IntoFirst,
{
    fn into_first(self) -> DirectionalFirstParts<Self::Value> {
        let determinant = self.into_inner();

        let (value, first) = determinant.into_first().into_parts();

        DirectionalFirstParts::new(
            PlaneWaveDeterminant::new(value),
            PlaneWaveDeterminant::new(first),
        )
    }
}

/// Separate a determinant into values and directional derivatives through second
/// order.
impl<J> IntoSecond for PlaneWaveDeterminant<J>
where
    J: IntoSecond,
{
    fn into_second(self) -> DirectionalSecondParts<Self::Value> {
        let determinant = self.into_inner();

        let (value, first, second) = determinant.into_second().into_parts();

        DirectionalSecondParts::new(
            PlaneWaveDeterminant::new(value),
            PlaneWaveDeterminant::new(first),
            PlaneWaveDeterminant::new(second),
        )
    }
}

/// Separate a determinant into values and first derivatives with respect to two
/// coordinates.
impl<J> IntoBivariateFirst for PlaneWaveDeterminant<J>
where
    J: IntoBivariateFirst,
{
    fn into_bivariate_first(self) -> BivariateFirstParts<Self::Value> {
        let determinant = self.into_inner();

        let (value, axis0, axis1) = determinant.into_bivariate_first().into_parts();

        BivariateFirstParts::new(
            PlaneWaveDeterminant::new(value),
            PlaneWaveDeterminant::new(axis0),
            PlaneWaveDeterminant::new(axis1),
        )
    }
}

/// Separate a determinant into values, a bivariate gradient, and a symmetric
/// bivariate Hessian.
impl<J> IntoBivariateSecond for PlaneWaveDeterminant<J>
where
    J: IntoBivariateSecond,
{
    fn into_bivariate_second(self) -> BivariateSecondParts<Self::Value> {
        let determinant = self.into_inner();

        let (value, axis0, axis1, axis0_axis0, axis0_axis1, axis1_axis1) =
            determinant.into_bivariate_second().into_parts();

        BivariateSecondParts::new(
            PlaneWaveDeterminant::new(value),
            PlaneWaveDeterminant::new(axis0),
            PlaneWaveDeterminant::new(axis1),
            PlaneWaveDeterminant::new(axis0_axis0),
            PlaneWaveDeterminant::new(axis0_axis1),
            PlaneWaveDeterminant::new(axis1_axis1),
        )
    }
}
