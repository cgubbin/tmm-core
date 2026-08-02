use crate::{
    derivative_parts::{
        BivariateFirstParts, BivariateSecondParts, DirectionalFirstParts, DirectionalSecondParts,
        IntoBivariateFirst, IntoBivariateSecond, IntoFirst, IntoSecond, IntoValue, ValuePart,
    },
    observable::{BoundaryWaves, LayerBoundaryWaves},
};

impl<A> IntoValue for BoundaryWaves<A>
where
    A: IntoValue,
{
    type Value = BoundaryWaves<A::Value>;

    fn into_value(self) -> ValuePart<Self::Value> {
        let (forward, backward) = self.into_parts();
        ValuePart::new(BoundaryWaves::new(
            forward.into_value().into_inner(),
            backward.into_value().into_inner(),
        ))
    }
}

impl<A> IntoFirst for BoundaryWaves<A>
where
    A: IntoFirst,
{
    fn into_first(self) -> DirectionalFirstParts<Self::Value> {
        let (forward, backward) = self.into_parts();

        let (forward, forward_first) = forward.into_first().into_parts();
        let (backward, backward_first) = backward.into_first().into_parts();

        DirectionalFirstParts::new(
            BoundaryWaves::new(forward, backward),
            BoundaryWaves::new(forward_first, backward_first),
        )
    }
}

impl<A> IntoSecond for BoundaryWaves<A>
where
    A: IntoSecond,
{
    fn into_second(self) -> DirectionalSecondParts<Self::Value> {
        let (forward, backward) = self.into_parts();

        let (forward, forward_first, forward_second) = forward.into_second().into_parts();
        let (backward, backward_first, backward_second) = backward.into_second().into_parts();

        DirectionalSecondParts::new(
            BoundaryWaves::new(forward, backward),
            BoundaryWaves::new(forward_first, backward_first),
            BoundaryWaves::new(forward_second, backward_second),
        )
    }
}

impl<A> IntoBivariateFirst for BoundaryWaves<A>
where
    A: IntoBivariateFirst,
{
    fn into_bivariate_first(self) -> BivariateFirstParts<Self::Value> {
        let (forward, backward) = self.into_parts();

        let (forward, forward_axis0, forward_axis1) = forward.into_bivariate_first().into_parts();
        let (backward, backward_axis0, backward_axis1) =
            backward.into_bivariate_first().into_parts();

        BivariateFirstParts::new(
            BoundaryWaves::new(forward, backward),
            BoundaryWaves::new(forward_axis0, backward_axis0),
            BoundaryWaves::new(forward_axis1, backward_axis1),
        )
    }
}

impl<A> IntoBivariateSecond for BoundaryWaves<A>
where
    A: IntoBivariateSecond,
{
    fn into_bivariate_second(self) -> BivariateSecondParts<Self::Value> {
        let (forward, backward) = self.into_parts();

        let (
            forward,
            forward_axis0,
            forward_axis1,
            forward_axis0_axis0,
            forward_axis0_axis1,
            forward_axis1_axis1,
        ) = forward.into_bivariate_second().into_parts();
        let (
            backward,
            backward_axis0,
            backward_axis1,
            backward_axis0_axis0,
            backward_axis0_axis1,
            backward_axis1_axis1,
        ) = backward.into_bivariate_second().into_parts();

        BivariateSecondParts::new(
            BoundaryWaves::new(forward, backward),
            BoundaryWaves::new(forward_axis0, backward_axis0),
            BoundaryWaves::new(forward_axis1, backward_axis1),
            BoundaryWaves::new(forward_axis0_axis0, backward_axis0_axis0),
            BoundaryWaves::new(forward_axis0_axis1, backward_axis0_axis1),
            BoundaryWaves::new(forward_axis1_axis1, backward_axis1_axis1),
        )
    }
}

impl<A> IntoValue for LayerBoundaryWaves<A>
where
    A: IntoValue,
{
    type Value = LayerBoundaryWaves<A::Value>;

    fn into_value(self) -> ValuePart<Self::Value> {
        let (left, right) = self.into_parts();
        ValuePart::new(LayerBoundaryWaves::new(
            left.into_value().into_inner(),
            right.into_value().into_inner(),
        ))
    }
}

impl<A> IntoFirst for LayerBoundaryWaves<A>
where
    A: IntoFirst,
{
    fn into_first(self) -> DirectionalFirstParts<Self::Value> {
        let (left, right) = self.into_parts();

        let (left, left_first) = left.into_first().into_parts();
        let (right, right_first) = right.into_first().into_parts();

        DirectionalFirstParts::new(
            LayerBoundaryWaves::new(left, right),
            LayerBoundaryWaves::new(left_first, right_first),
        )
    }
}

impl<A> IntoSecond for LayerBoundaryWaves<A>
where
    A: IntoSecond,
{
    fn into_second(self) -> DirectionalSecondParts<Self::Value> {
        let (left, right) = self.into_parts();

        let (left, left_first, left_second) = left.into_second().into_parts();
        let (right, right_first, right_second) = right.into_second().into_parts();

        DirectionalSecondParts::new(
            LayerBoundaryWaves::new(left, right),
            LayerBoundaryWaves::new(left_first, right_first),
            LayerBoundaryWaves::new(left_second, right_second),
        )
    }
}

impl<A> IntoBivariateFirst for LayerBoundaryWaves<A>
where
    A: IntoBivariateFirst,
{
    fn into_bivariate_first(self) -> BivariateFirstParts<Self::Value> {
        let (left, right) = self.into_parts();

        let (left, left_axis0, left_axis1) = left.into_bivariate_first().into_parts();
        let (right, right_axis0, right_axis1) = right.into_bivariate_first().into_parts();

        BivariateFirstParts::new(
            LayerBoundaryWaves::new(left, right),
            LayerBoundaryWaves::new(left_axis0, right_axis0),
            LayerBoundaryWaves::new(left_axis1, right_axis1),
        )
    }
}

impl<A> IntoBivariateSecond for LayerBoundaryWaves<A>
where
    A: IntoBivariateSecond,
{
    fn into_bivariate_second(self) -> BivariateSecondParts<Self::Value> {
        let (left, right) = self.into_parts();

        let (left, left_axis0, left_axis1, left_axis0_axis0, left_axis0_axis1, left_axis1_axis1) =
            left.into_bivariate_second().into_parts();
        let (
            right,
            right_axis0,
            right_axis1,
            right_axis0_axis0,
            right_axis0_axis1,
            right_axis1_axis1,
        ) = right.into_bivariate_second().into_parts();

        BivariateSecondParts::new(
            LayerBoundaryWaves::new(left, right),
            LayerBoundaryWaves::new(left_axis0, right_axis0),
            LayerBoundaryWaves::new(left_axis1, right_axis1),
            LayerBoundaryWaves::new(left_axis0_axis0, right_axis0_axis0),
            LayerBoundaryWaves::new(left_axis0_axis1, right_axis0_axis1),
            LayerBoundaryWaves::new(left_axis1_axis1, right_axis1_axis1),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deliberately simple derivative-bearing quantity.
    ///
    /// Every component can be assigned a distinct value, making structural
    /// transposition errors easy to detect.
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

    fn assert_boundary(actual: &BoundaryWaves<f64>, forward: f64, backward: f64) {
        assert_eq!(actual.forward(), &forward);
        assert_eq!(actual.backward(), &backward);
    }

    fn assert_layer_boundary(
        actual: &LayerBoundaryWaves<f64>,
        left_forward: f64,
        left_backward: f64,
        right_forward: f64,
        right_backward: f64,
    ) {
        assert_boundary(actual.left(), left_forward, left_backward);

        assert_boundary(actual.right(), right_forward, right_backward);
    }

    #[test]
    fn boundary_waves_into_value_preserves_directional_order() {
        let waves = BoundaryWaves::new(probe(0.0), probe(10.0));

        let value = waves.into_value().into_inner();

        assert_boundary(&value, 1.0, 11.0);
    }

    #[test]
    fn boundary_waves_into_first_preserves_directional_order() {
        let waves = BoundaryWaves::new(probe(0.0), probe(10.0));

        let (value, first) = waves.into_first().into_parts();

        assert_boundary(&value, 1.0, 11.0);
        assert_boundary(&first, 2.0, 12.0);
    }

    #[test]
    fn boundary_waves_into_second_preserves_every_branch() {
        let waves = BoundaryWaves::new(probe(0.0), probe(10.0));

        let (value, first, second) = waves.into_second().into_parts();

        assert_boundary(&value, 1.0, 11.0);
        assert_boundary(&first, 2.0, 12.0);
        assert_boundary(&second, 4.0, 14.0);
    }

    #[test]
    fn boundary_waves_into_bivariate_first_preserves_axes() {
        let waves = BoundaryWaves::new(probe(0.0), probe(10.0));

        let (value, axis0, axis1) = waves.into_bivariate_first().into_parts();

        assert_boundary(&value, 1.0, 11.0);
        assert_boundary(&axis0, 2.0, 12.0);
        assert_boundary(&axis1, 3.0, 13.0);
    }

    #[test]
    fn boundary_waves_into_bivariate_second_preserves_every_branch() {
        let waves = BoundaryWaves::new(probe(0.0), probe(10.0));

        let (value, axis0, axis1, axis0_axis0, axis0_axis1, axis1_axis1) =
            waves.into_bivariate_second().into_parts();

        assert_boundary(&value, 1.0, 11.0);
        assert_boundary(&axis0, 2.0, 12.0);
        assert_boundary(&axis1, 3.0, 13.0);
        assert_boundary(&axis0_axis0, 4.0, 14.0);
        assert_boundary(&axis0_axis1, 5.0, 15.0);
        assert_boundary(&axis1_axis1, 6.0, 16.0);
    }

    #[test]
    fn layer_boundary_waves_into_value_preserves_boundary_and_direction() {
        let waves = LayerBoundaryWaves::new(
            BoundaryWaves::new(probe(0.0), probe(10.0)),
            BoundaryWaves::new(probe(100.0), probe(1_000.0)),
        );

        let value = waves.into_value().into_inner();

        assert_layer_boundary(&value, 1.0, 11.0, 101.0, 1_001.0);
    }

    #[test]
    fn layer_boundary_waves_into_first_preserves_boundary_and_direction() {
        let waves = LayerBoundaryWaves::new(
            BoundaryWaves::new(probe(0.0), probe(10.0)),
            BoundaryWaves::new(probe(100.0), probe(1_000.0)),
        );

        let (value, first) = waves.into_first().into_parts();

        assert_layer_boundary(&value, 1.0, 11.0, 101.0, 1_001.0);

        assert_layer_boundary(&first, 2.0, 12.0, 102.0, 1_002.0);
    }

    #[test]
    fn layer_boundary_waves_into_second_preserves_boundary_and_order() {
        let waves = LayerBoundaryWaves::new(
            BoundaryWaves::new(probe(0.0), probe(10.0)),
            BoundaryWaves::new(probe(100.0), probe(1_000.0)),
        );

        let (value, first, second) = waves.into_second().into_parts();

        assert_layer_boundary(&value, 1.0, 11.0, 101.0, 1_001.0);

        assert_layer_boundary(&first, 2.0, 12.0, 102.0, 1_002.0);

        assert_layer_boundary(&second, 4.0, 14.0, 104.0, 1_004.0);
    }

    #[test]
    fn layer_boundary_waves_into_bivariate_first_preserves_axes() {
        let waves = LayerBoundaryWaves::new(
            BoundaryWaves::new(probe(0.0), probe(10.0)),
            BoundaryWaves::new(probe(100.0), probe(1_000.0)),
        );

        let (value, axis0, axis1) = waves.into_bivariate_first().into_parts();

        assert_layer_boundary(&value, 1.0, 11.0, 101.0, 1_001.0);

        assert_layer_boundary(&axis0, 2.0, 12.0, 102.0, 1_002.0);

        assert_layer_boundary(&axis1, 3.0, 13.0, 103.0, 1_003.0);
    }

    #[test]
    fn layer_boundary_waves_into_bivariate_second_preserves_all_indices() {
        let waves = LayerBoundaryWaves::new(
            BoundaryWaves::new(probe(0.0), probe(10.0)),
            BoundaryWaves::new(probe(100.0), probe(1_000.0)),
        );

        let (value, axis0, axis1, axis0_axis0, axis0_axis1, axis1_axis1) =
            waves.into_bivariate_second().into_parts();

        assert_layer_boundary(&value, 1.0, 11.0, 101.0, 1_001.0);

        assert_layer_boundary(&axis0, 2.0, 12.0, 102.0, 1_002.0);

        assert_layer_boundary(&axis1, 3.0, 13.0, 103.0, 1_003.0);

        assert_layer_boundary(&axis0_axis0, 4.0, 14.0, 104.0, 1_004.0);

        assert_layer_boundary(&axis0_axis1, 5.0, 15.0, 105.0, 1_005.0);

        assert_layer_boundary(&axis1_axis1, 6.0, 16.0, 106.0, 1_006.0);
    }
}
