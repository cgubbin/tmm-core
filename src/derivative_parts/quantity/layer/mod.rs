mod dissipation;
mod power;

use crate::{
    derivative_parts::{
        BivariateFirstParts, BivariateSecondParts, DirectionalFirstParts, DirectionalSecondParts,
        IntoBivariateFirst, IntoBivariateSecond, IntoFirst, IntoSecond, IntoValue, ValuePart,
    },
    observable::Layers,
};

impl<A> IntoValue for Layers<A>
where
    A: IntoValue,
{
    type Value = Layers<A::Value>;

    fn into_value(self) -> ValuePart<Self::Value> {
        let value = self.into_inner();
        ValuePart::new(Layers::new(
            value
                .into_iter()
                .map(|each| each.into_value().into_inner())
                .collect(),
        ))
    }
}

impl<A> IntoFirst for Layers<A>
where
    A: IntoFirst,
{
    fn into_first(self) -> DirectionalFirstParts<Self::Value> {
        let (value, first) = self
            .into_inner()
            .into_iter()
            .map(|each| each.into_first().into_parts())
            .collect();

        DirectionalFirstParts::new(Layers::new(value), Layers::new(first))
    }
}

impl<A> IntoSecond for Layers<A>
where
    A: IntoSecond,
{
    fn into_second(self) -> DirectionalSecondParts<Self::Value> {
        let (value, first, second) = self
            .into_inner()
            .into_iter()
            .map(|each| each.into_second().into_parts())
            .collect();

        DirectionalSecondParts::new(Layers::new(value), Layers::new(first), Layers::new(second))
    }
}

impl<A> IntoBivariateFirst for Layers<A>
where
    A: IntoBivariateFirst,
{
    fn into_bivariate_first(self) -> BivariateFirstParts<Self::Value> {
        let (value, axis0, axis1) = self
            .into_inner()
            .into_iter()
            .map(|each| each.into_bivariate_first().into_parts())
            .collect();

        BivariateFirstParts::new(Layers::new(value), Layers::new(axis0), Layers::new(axis1))
    }
}

impl<A> IntoBivariateSecond for Layers<A>
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
            Layers::new(value),
            Layers::new(axis0),
            Layers::new(axis1),
            Layers::new(axis0_axis0),
            Layers::new(axis0_axis1),
            Layers::new(axis1_axis1),
        )
    }
}

#[cfg(test)]
mod tests {
    use crate::LayerPower;

    use super::*;

    #[derive(Clone, Copy, Debug, PartialEq)]
    struct Probe {
        value: f64,
        axis0: f64,
        axis1: f64,
        axis0_axis0: f64,
        axis0_axis1: f64,
        axis1_axis1: f64,
    }

    impl Probe {
        const fn new(
            value: f64,
            axis0: f64,
            axis1: f64,
            axis0_axis0: f64,
            axis0_axis1: f64,
            axis1_axis1: f64,
        ) -> Self {
            Self {
                value,
                axis0,
                axis1,
                axis0_axis0,
                axis0_axis1,
                axis1_axis1,
            }
        }
    }

    impl IntoValue for Probe {
        type Value = f64;

        fn into_value(self) -> ValuePart<Self::Value> {
            ValuePart::new(self.value)
        }
    }

    impl IntoFirst for Probe {
        fn into_first(self) -> DirectionalFirstParts<Self::Value> {
            DirectionalFirstParts::new(self.value, self.axis0)
        }
    }

    impl IntoSecond for Probe {
        fn into_second(self) -> DirectionalSecondParts<Self::Value> {
            DirectionalSecondParts::new(self.value, self.axis0, self.axis0_axis0)
        }
    }

    impl IntoBivariateFirst for Probe {
        fn into_bivariate_first(self) -> BivariateFirstParts<Self::Value> {
            BivariateFirstParts::new(self.value, self.axis0, self.axis1)
        }
    }

    impl IntoBivariateSecond for Probe {
        fn into_bivariate_second(self) -> BivariateSecondParts<Self::Value> {
            BivariateSecondParts::new(
                self.value,
                self.axis0,
                self.axis1,
                self.axis0_axis0,
                self.axis0_axis1,
                self.axis1_axis1,
            )
        }
    }

    fn probe(offset: f64) -> Probe {
        Probe::new(
            offset + 1.0,
            offset + 2.0,
            offset + 3.0,
            offset + 4.0,
            offset + 5.0,
            offset + 6.0,
        )
    }

    fn layer_power(offset: f64) -> LayerPower<Probe> {
        LayerPower::new(probe(offset), probe(offset + 10.0), probe(offset + 20.0))
    }

    #[test]
    fn layers_bivariate_second_transposes_every_layer() {
        let layers = Layers::new(vec![layer_power(0.0), layer_power(100.0)]);

        let (value, axis0, axis1, axis0_axis0, axis0_axis1, axis1_axis1) =
            layers.into_bivariate_second().into_parts();

        for branch in [
            &value,
            &axis0,
            &axis1,
            &axis0_axis0,
            &axis0_axis1,
            &axis1_axis1,
        ] {
            assert_eq!(branch.len(), 2);
        }

        assert_eq!(value.get(0).unwrap().left_flux(), &1.0,);

        assert_eq!(axis0_axis1.get(0).unwrap().absorbed(), &25.0,);

        assert_eq!(axis1_axis1.get(1).unwrap().right_flux(), &116.0,);
    }
}
