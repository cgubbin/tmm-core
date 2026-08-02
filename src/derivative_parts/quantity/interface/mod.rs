mod power;
mod state;

use crate::{
    derivative_parts::{
        BivariateFirstParts, BivariateSecondParts, DirectionalFirstParts, DirectionalSecondParts,
        IntoBivariateFirst, IntoBivariateSecond, IntoFirst, IntoSecond, IntoValue, ValuePart,
    },
    observable::Interfaces,
};

impl<A> IntoValue for Interfaces<A>
where
    A: IntoValue,
{
    type Value = Interfaces<A::Value>;

    fn into_value(self) -> ValuePart<Self::Value> {
        let value = self.into_inner();
        ValuePart::new(Interfaces::new(
            value
                .into_iter()
                .map(|each| each.into_value().into_inner())
                .collect(),
        ))
    }
}

impl<A> IntoFirst for Interfaces<A>
where
    A: IntoFirst,
{
    fn into_first(self) -> DirectionalFirstParts<Self::Value> {
        let (value, first) = self
            .into_inner()
            .into_iter()
            .map(|each| each.into_first().into_parts())
            .collect();

        DirectionalFirstParts::new(Interfaces::new(value), Interfaces::new(first))
    }
}

impl<A> IntoSecond for Interfaces<A>
where
    A: IntoSecond,
{
    fn into_second(self) -> DirectionalSecondParts<Self::Value> {
        let (value, first, second) = self
            .into_inner()
            .into_iter()
            .map(|each| each.into_second().into_parts())
            .collect();

        DirectionalSecondParts::new(
            Interfaces::new(value),
            Interfaces::new(first),
            Interfaces::new(second),
        )
    }
}

impl<A> IntoBivariateFirst for Interfaces<A>
where
    A: IntoBivariateFirst,
{
    fn into_bivariate_first(self) -> BivariateFirstParts<Self::Value> {
        let (value, axis0, axis1) = self
            .into_inner()
            .into_iter()
            .map(|each| each.into_bivariate_first().into_parts())
            .collect();

        BivariateFirstParts::new(
            Interfaces::new(value),
            Interfaces::new(axis0),
            Interfaces::new(axis1),
        )
    }
}

impl<A> IntoBivariateSecond for Interfaces<A>
where
    A: IntoBivariateSecond,
{
    fn into_bivariate_second(self) -> BivariateSecondParts<Self::Value> {
        let (value, axis0, axis1, axis0_axis0, axis0_axis1, axis1_axis1) = self
            .into_inner()
            .into_iter()
            .map(|each| each.into_bivariate_second().into_parts())
            .collect();

        BivariateSecondParts::new(
            Interfaces::new(value),
            Interfaces::new(axis0),
            Interfaces::new(axis1),
            Interfaces::new(axis0_axis0),
            Interfaces::new(axis0_axis1),
            Interfaces::new(axis1_axis1),
        )
    }
}
