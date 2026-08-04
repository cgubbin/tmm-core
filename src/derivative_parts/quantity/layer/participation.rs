use crate::{
    derivative_parts::{
        BivariateFirstParts, BivariateSecondParts, DirectionalFirstParts, DirectionalSecondParts,
        IntoBivariateFirst, IntoBivariateSecond, IntoFirst, IntoSecond, IntoValue, ValuePart,
    },
    observable::LayerParticipation,
};

impl<R> IntoValue for LayerParticipation<R>
where
    R: IntoValue,
{
    type Value = LayerParticipation<R::Value>;

    fn into_value(self) -> ValuePart<Self::Value> {
        let (electric, magnetic, total) = self.into_parts();

        ValuePart::new(LayerParticipation::new(
            electric.into_value().into_inner(),
            magnetic.into_value().into_inner(),
            total.into_value().into_inner(),
        ))
    }
}

impl<R> IntoFirst for LayerParticipation<R>
where
    R: IntoFirst,
{
    fn into_first(self) -> DirectionalFirstParts<Self::Value> {
        let (electric, magnetic, total) = self.into_parts();

        let (electric, electric_first) = electric.into_first().into_parts();

        let (magnetic, magnetic_first) = magnetic.into_first().into_parts();

        let (total, total_first) = total.into_first().into_parts();

        DirectionalFirstParts::new(
            LayerParticipation::new(electric, magnetic, total),
            LayerParticipation::new(electric_first, magnetic_first, total_first),
        )
    }
}

impl<R> IntoSecond for LayerParticipation<R>
where
    R: IntoSecond,
{
    fn into_second(self) -> DirectionalSecondParts<Self::Value> {
        let (electric, magnetic, total) = self.into_parts();

        let (electric, electric_first, electric_second) = electric.into_second().into_parts();

        let (magnetic, magnetic_first, magnetic_second) = magnetic.into_second().into_parts();

        let (total, total_first, total_second) = total.into_second().into_parts();

        DirectionalSecondParts::new(
            LayerParticipation::new(electric, magnetic, total),
            LayerParticipation::new(electric_first, magnetic_first, total_first),
            LayerParticipation::new(electric_second, magnetic_second, total_second),
        )
    }
}

impl<R> IntoBivariateFirst for LayerParticipation<R>
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
            LayerParticipation::new(electric, magnetic, total),
            LayerParticipation::new(electric_axis0, magnetic_axis0, total_axis0),
            LayerParticipation::new(electric_axis1, magnetic_axis1, total_axis1),
        )
    }
}

impl<R> IntoBivariateSecond for LayerParticipation<R>
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
            LayerParticipation::new(electric, magnetic, total),
            LayerParticipation::new(electric_axis0, magnetic_axis0, total_axis0),
            LayerParticipation::new(electric_axis1, magnetic_axis1, total_axis1),
            LayerParticipation::new(
                electric_axis0_axis0,
                magnetic_axis0_axis0,
                total_axis0_axis0,
            ),
            LayerParticipation::new(
                electric_axis0_axis1,
                magnetic_axis0_axis1,
                total_axis0_axis1,
            ),
            LayerParticipation::new(
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

    fn layer_participation(offset: f64) -> LayerParticipation<Probe> {
        LayerParticipation::new(probe(offset), probe(offset + 10.0), probe(offset + 20.0))
    }

    fn assert_layer_participation(
        actual: &LayerParticipation<f64>,
        left: f64,
        right: f64,
        total: f64,
    ) {
        assert_eq!(actual.electric(), &left);
        assert_eq!(actual.magnetic(), &right);
        assert_eq!(actual.total(), &total);
    }

    #[test]
    fn layer_participation_into_value_preserves_component_order() {
        let value = layer_participation(0.0).into_value().into_inner();

        assert_layer_participation(&value, 1.0, 11.0, 21.0);
    }

    #[test]
    fn layer_participation_into_first_preserves_all_components() {
        let (value, first) = layer_participation(0.0).into_first().into_parts();

        assert_layer_participation(&value, 1.0, 11.0, 21.0);

        assert_layer_participation(&first, 2.0, 12.0, 22.0);
    }

    #[test]
    fn layer_participation_into_second_preserves_all_components() {
        let (value, first, second) = layer_participation(0.0).into_second().into_parts();

        assert_layer_participation(&value, 1.0, 11.0, 21.0);

        assert_layer_participation(&first, 2.0, 12.0, 22.0);

        assert_layer_participation(&second, 4.0, 14.0, 24.0);
    }

    #[test]
    fn layer_participation_into_bivariate_first_preserves_axes() {
        let (value, axis0, axis1) = layer_participation(0.0).into_bivariate_first().into_parts();

        assert_layer_participation(&value, 1.0, 11.0, 21.0);

        assert_layer_participation(&axis0, 2.0, 12.0, 22.0);

        assert_layer_participation(&axis1, 3.0, 13.0, 23.0);
    }

    #[test]
    fn layer_participation_into_bivariate_second_preserves_all_branches() {
        let (value, axis0, axis1, axis0_axis0, axis0_axis1, axis1_axis1) = layer_participation(0.0)
            .into_bivariate_second()
            .into_parts();

        assert_layer_participation(&value, 1.0, 11.0, 21.0);

        assert_layer_participation(&axis0, 2.0, 12.0, 22.0);

        assert_layer_participation(&axis1, 3.0, 13.0, 23.0);

        assert_layer_participation(&axis0_axis0, 4.0, 14.0, 24.0);

        assert_layer_participation(&axis0_axis1, 5.0, 15.0, 25.0);

        assert_layer_participation(&axis1_axis1, 6.0, 16.0, 26.0);
    }
}
