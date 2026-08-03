use crate::{
    derivative_parts::{
        BivariateFirstParts, BivariateSecondParts, DirectionalFirstParts, DirectionalSecondParts,
        IntoBivariateFirst, IntoBivariateSecond, IntoFirst, IntoSecond, IntoValue, ValuePart,
    },
    observable::LayerEnergy,
};

impl<R> IntoValue for LayerEnergy<R>
where
    R: IntoValue,
{
    type Value = LayerEnergy<R::Value>;

    fn into_value(self) -> ValuePart<Self::Value> {
        let (electric, magnetic, total) = self.into_parts();

        ValuePart::new(LayerEnergy::new(
            electric.into_value().into_inner(),
            magnetic.into_value().into_inner(),
            total.into_value().into_inner(),
        ))
    }
}

impl<R> IntoFirst for LayerEnergy<R>
where
    R: IntoFirst,
{
    fn into_first(self) -> DirectionalFirstParts<Self::Value> {
        let (electric, magnetic, total) = self.into_parts();

        let (electric, electric_first) = electric.into_first().into_parts();

        let (magnetic, magnetic_first) = magnetic.into_first().into_parts();

        let (total, total_first) = total.into_first().into_parts();

        DirectionalFirstParts::new(
            LayerEnergy::new(electric, magnetic, total),
            LayerEnergy::new(electric_first, magnetic_first, total_first),
        )
    }
}

impl<R> IntoSecond for LayerEnergy<R>
where
    R: IntoSecond,
{
    fn into_second(self) -> DirectionalSecondParts<Self::Value> {
        let (electric, magnetic, total) = self.into_parts();

        let (electric, electric_first, electric_second) = electric.into_second().into_parts();

        let (magnetic, magnetic_first, magnetic_second) = magnetic.into_second().into_parts();

        let (total, total_first, total_second) = total.into_second().into_parts();

        DirectionalSecondParts::new(
            LayerEnergy::new(electric, magnetic, total),
            LayerEnergy::new(electric_first, magnetic_first, total_first),
            LayerEnergy::new(electric_second, magnetic_second, total_second),
        )
    }
}

impl<R> IntoBivariateFirst for LayerEnergy<R>
where
    R: IntoBivariateFirst,
{
    fn into_bivariate_first(self) -> BivariateFirstParts<Self::Value> {
        let (electric, magnetic, total) = self.into_parts();

        let (electric, electric_axis0, electric_axis1) =
            electric.into_bivariate_first().into_parts();

        let (magnetic, magnetic_axis0, magnetic_axis1) =
            magnetic.into_bivariate_first().into_parts();

        let (total, total_axis0, total_axis1) = total.into_bivariate_first().into_parts();

        BivariateFirstParts::new(
            LayerEnergy::new(electric, magnetic, total),
            LayerEnergy::new(electric_axis0, magnetic_axis0, total_axis0),
            LayerEnergy::new(electric_axis1, magnetic_axis1, total_axis1),
        )
    }
}

impl<R> IntoBivariateSecond for LayerEnergy<R>
where
    R: IntoBivariateSecond,
{
    fn into_bivariate_second(self) -> BivariateSecondParts<Self::Value> {
        let (electric, magnetic, total) = self.into_parts();

        let (
            electric,
            electric_axis0,
            electric_axis1,
            electric_axis0_axis0,
            electric_axis0_axis1,
            electric_axis1_axis1,
        ) = electric.into_bivariate_second().into_parts();

        let (
            magnetic,
            magnetic_axis0,
            magnetic_axis1,
            magnetic_axis0_axis0,
            magnetic_axis0_axis1,
            magnetic_axis1_axis1,
        ) = magnetic.into_bivariate_second().into_parts();

        let (
            total,
            total_axis0,
            total_axis1,
            total_axis0_axis0,
            total_axis0_axis1,
            total_axis1_axis1,
        ) = total.into_bivariate_second().into_parts();

        BivariateSecondParts::new(
            LayerEnergy::new(electric, magnetic, total),
            LayerEnergy::new(electric_axis0, magnetic_axis0, total_axis0),
            LayerEnergy::new(electric_axis1, magnetic_axis1, total_axis1),
            LayerEnergy::new(
                electric_axis0_axis0,
                magnetic_axis0_axis0,
                total_axis0_axis0,
            ),
            LayerEnergy::new(
                electric_axis0_axis1,
                magnetic_axis0_axis1,
                total_axis0_axis1,
            ),
            LayerEnergy::new(
                electric_axis1_axis1,
                magnetic_axis1_axis1,
                total_axis1_axis1,
            ),
        )
    }
}

#[cfg(test)]
mod tests {
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

    fn layer_power(offset: f64) -> LayerEnergy<Probe> {
        LayerEnergy::new(probe(offset), probe(offset + 10.0), probe(offset + 20.0))
    }

    fn assert_layer_power(actual: &LayerEnergy<f64>, left: f64, right: f64, total: f64) {
        assert_eq!(actual.electric(), &left);
        assert_eq!(actual.magnetic(), &right);
        assert_eq!(actual.total(), &total);
    }

    #[test]
    fn layer_power_into_value_preserves_component_order() {
        let value = layer_power(0.0).into_value().into_inner();

        assert_layer_power(&value, 1.0, 11.0, 21.0);
    }

    #[test]
    fn layer_power_into_first_preserves_all_components() {
        let (value, first) = layer_power(0.0).into_first().into_parts();

        assert_layer_power(&value, 1.0, 11.0, 21.0);

        assert_layer_power(&first, 2.0, 12.0, 22.0);
    }

    #[test]
    fn layer_power_into_second_preserves_all_components() {
        let (value, first, second) = layer_power(0.0).into_second().into_parts();

        assert_layer_power(&value, 1.0, 11.0, 21.0);

        assert_layer_power(&first, 2.0, 12.0, 22.0);

        assert_layer_power(&second, 4.0, 14.0, 24.0);
    }

    #[test]
    fn layer_power_into_bivariate_first_preserves_axes() {
        let (value, axis0, axis1) = layer_power(0.0).into_bivariate_first().into_parts();

        assert_layer_power(&value, 1.0, 11.0, 21.0);

        assert_layer_power(&axis0, 2.0, 12.0, 22.0);

        assert_layer_power(&axis1, 3.0, 13.0, 23.0);
    }

    #[test]
    fn layer_power_into_bivariate_second_preserves_all_branches() {
        let (value, axis0, axis1, axis0_axis0, axis0_axis1, axis1_axis1) =
            layer_power(0.0).into_bivariate_second().into_parts();

        assert_layer_power(&value, 1.0, 11.0, 21.0);

        assert_layer_power(&axis0, 2.0, 12.0, 22.0);

        assert_layer_power(&axis1, 3.0, 13.0, 23.0);

        assert_layer_power(&axis0_axis0, 4.0, 14.0, 24.0);

        assert_layer_power(&axis0_axis1, 5.0, 15.0, 25.0);

        assert_layer_power(&axis1_axis1, 6.0, 16.0, 26.0);
    }
}
