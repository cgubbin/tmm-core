use crate::{
    derivative_parts::{
        BivariateFirstParts, BivariateSecondParts, DirectionalFirstParts, DirectionalSecondParts,
        IntoBivariateFirst, IntoBivariateSecond, IntoFirst, IntoSecond, IntoValue, ValuePart,
    },
    observable::{BoundaryState, LayerBoundaryStates},
};

impl<A> IntoValue for BoundaryState<A>
where
    A: IntoValue,
{
    type Value = BoundaryState<A::Value>;

    fn into_value(self) -> ValuePart<Self::Value> {
        let (field, secondary) = self.into_parts();
        ValuePart::new(BoundaryState::new(
            field.into_value().into_inner(),
            secondary.into_value().into_inner(),
        ))
    }
}

impl<A> IntoFirst for BoundaryState<A>
where
    A: IntoFirst,
{
    fn into_first(self) -> DirectionalFirstParts<Self::Value> {
        let (field, secondary) = self.into_parts();

        let (field, field_first) = field.into_first().into_parts();
        let (secondary, secondary_first) = secondary.into_first().into_parts();

        DirectionalFirstParts::new(
            BoundaryState::new(field, secondary),
            BoundaryState::new(field_first, secondary_first),
        )
    }
}

impl<A> IntoSecond for BoundaryState<A>
where
    A: IntoSecond,
{
    fn into_second(self) -> DirectionalSecondParts<Self::Value> {
        let (field, secondary) = self.into_parts();

        let (field, field_first, field_second) = field.into_second().into_parts();
        let (secondary, secondary_first, secondary_second) = secondary.into_second().into_parts();

        DirectionalSecondParts::new(
            BoundaryState::new(field, secondary),
            BoundaryState::new(field_first, secondary_first),
            BoundaryState::new(field_second, secondary_second),
        )
    }
}

impl<A> IntoBivariateFirst for BoundaryState<A>
where
    A: IntoBivariateFirst,
{
    fn into_bivariate_first(self) -> BivariateFirstParts<Self::Value> {
        let (field, secondary) = self.into_parts();

        let (field, field_axis0, field_axis1) = field.into_bivariate_first().into_parts();
        let (secondary, secondary_axis0, secondary_axis1) =
            secondary.into_bivariate_first().into_parts();

        BivariateFirstParts::new(
            BoundaryState::new(field, secondary),
            BoundaryState::new(field_axis0, secondary_axis0),
            BoundaryState::new(field_axis1, secondary_axis1),
        )
    }
}

impl<A> IntoBivariateSecond for BoundaryState<A>
where
    A: IntoBivariateSecond,
{
    fn into_bivariate_second(self) -> BivariateSecondParts<Self::Value> {
        let (field, secondary) = self.into_parts();

        let (
            field,
            field_axis0,
            field_axis1,
            field_axis0_axis0,
            field_axis0_axis1,
            field_axis1_axis1,
        ) = field.into_bivariate_second().into_parts();
        let (
            secondary,
            secondary_axis0,
            secondary_axis1,
            secondary_axis0_axis0,
            secondary_axis0_axis1,
            secondary_axis1_axis1,
        ) = secondary.into_bivariate_second().into_parts();

        BivariateSecondParts::new(
            BoundaryState::new(field, secondary),
            BoundaryState::new(field_axis0, secondary_axis0),
            BoundaryState::new(field_axis1, secondary_axis1),
            BoundaryState::new(field_axis0_axis0, secondary_axis0_axis0),
            BoundaryState::new(field_axis0_axis1, secondary_axis0_axis1),
            BoundaryState::new(field_axis1_axis1, secondary_axis1_axis1),
        )
    }
}

impl<A> IntoValue for LayerBoundaryStates<A>
where
    A: IntoValue,
{
    type Value = LayerBoundaryStates<A::Value>;

    fn into_value(self) -> ValuePart<Self::Value> {
        let (forward, backward) = self.into_parts();
        ValuePart::new(LayerBoundaryStates::new(
            forward.into_value().into_inner(),
            backward.into_value().into_inner(),
        ))
    }
}

impl<A> IntoFirst for LayerBoundaryStates<A>
where
    A: IntoFirst,
{
    fn into_first(self) -> DirectionalFirstParts<Self::Value> {
        let (forward, backward) = self.into_parts();

        let (forward, forward_first) = forward.into_first().into_parts();
        let (backward, backward_first) = backward.into_first().into_parts();

        DirectionalFirstParts::new(
            LayerBoundaryStates::new(forward, backward),
            LayerBoundaryStates::new(forward_first, backward_first),
        )
    }
}

impl<A> IntoSecond for LayerBoundaryStates<A>
where
    A: IntoSecond,
{
    fn into_second(self) -> DirectionalSecondParts<Self::Value> {
        let (forward, backward) = self.into_parts();

        let (forward, forward_first, forward_second) = forward.into_second().into_parts();
        let (backward, backward_first, backward_second) = backward.into_second().into_parts();

        DirectionalSecondParts::new(
            LayerBoundaryStates::new(forward, backward),
            LayerBoundaryStates::new(forward_first, backward_first),
            LayerBoundaryStates::new(forward_second, backward_second),
        )
    }
}

impl<A> IntoBivariateFirst for LayerBoundaryStates<A>
where
    A: IntoBivariateFirst,
{
    fn into_bivariate_first(self) -> BivariateFirstParts<Self::Value> {
        let (forward, backward) = self.into_parts();

        let (forward, forward_axis0, forward_axis1) = forward.into_bivariate_first().into_parts();
        let (backward, backward_axis0, backward_axis1) =
            backward.into_bivariate_first().into_parts();

        BivariateFirstParts::new(
            LayerBoundaryStates::new(forward, backward),
            LayerBoundaryStates::new(forward_axis0, backward_axis0),
            LayerBoundaryStates::new(forward_axis1, backward_axis1),
        )
    }
}

impl<A> IntoBivariateSecond for LayerBoundaryStates<A>
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
            LayerBoundaryStates::new(forward, backward),
            LayerBoundaryStates::new(forward_axis0, backward_axis0),
            LayerBoundaryStates::new(forward_axis1, backward_axis1),
            LayerBoundaryStates::new(forward_axis0_axis0, backward_axis0_axis0),
            LayerBoundaryStates::new(forward_axis0_axis1, backward_axis0_axis1),
            LayerBoundaryStates::new(forward_axis1_axis1, backward_axis1_axis1),
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

    fn assert_state(actual: &BoundaryState<f64>, field: f64, secondary: f64) {
        assert_eq!(actual.field(), &field);
        assert_eq!(actual.secondary(), &secondary);
    }

    fn assert_layer_states(
        actual: &LayerBoundaryStates<f64>,
        left_field: f64,
        left_secondary: f64,
        right_field: f64,
        right_secondary: f64,
    ) {
        assert_state(actual.left(), left_field, left_secondary);

        assert_state(actual.right(), right_field, right_secondary);
    }

    #[test]
    fn boundary_state_into_value_preserves_component_order() {
        let state = BoundaryState::new(probe(0.0), probe(10.0));

        let value = state.into_value().into_inner();

        assert_state(&value, 1.0, 11.0);
    }

    #[test]
    fn boundary_state_into_first_preserves_component_order() {
        let state = BoundaryState::new(probe(0.0), probe(10.0));

        let (value, first) = state.into_first().into_parts();

        assert_state(&value, 1.0, 11.0);
        assert_state(&first, 2.0, 12.0);
    }

    #[test]
    fn boundary_state_into_second_preserves_every_branch() {
        let state = BoundaryState::new(probe(0.0), probe(10.0));

        let (value, first, second) = state.into_second().into_parts();

        assert_state(&value, 1.0, 11.0);
        assert_state(&first, 2.0, 12.0);
        assert_state(&second, 4.0, 14.0);
    }

    #[test]
    fn boundary_state_into_bivariate_first_preserves_axes() {
        let state = BoundaryState::new(probe(0.0), probe(10.0));

        let (value, axis0, axis1) = state.into_bivariate_first().into_parts();

        assert_state(&value, 1.0, 11.0);
        assert_state(&axis0, 2.0, 12.0);
        assert_state(&axis1, 3.0, 13.0);
    }

    #[test]
    fn boundary_state_into_bivariate_second_preserves_every_branch() {
        let state = BoundaryState::new(probe(0.0), probe(10.0));

        let (value, axis0, axis1, axis0_axis0, axis0_axis1, axis1_axis1) =
            state.into_bivariate_second().into_parts();

        assert_state(&value, 1.0, 11.0);
        assert_state(&axis0, 2.0, 12.0);
        assert_state(&axis1, 3.0, 13.0);
        assert_state(&axis0_axis0, 4.0, 14.0);
        assert_state(&axis0_axis1, 5.0, 15.0);
        assert_state(&axis1_axis1, 6.0, 16.0);
    }

    #[test]
    fn layer_boundary_states_into_value_preserves_boundary_and_component() {
        let states = LayerBoundaryStates::new(
            BoundaryState::new(probe(0.0), probe(10.0)),
            BoundaryState::new(probe(100.0), probe(1_000.0)),
        );

        let value = states.into_value().into_inner();

        assert_layer_states(&value, 1.0, 11.0, 101.0, 1_001.0);
    }

    #[test]
    fn layer_boundary_states_into_first_preserves_boundary_and_component() {
        let states = LayerBoundaryStates::new(
            BoundaryState::new(probe(0.0), probe(10.0)),
            BoundaryState::new(probe(100.0), probe(1_000.0)),
        );

        let (value, first) = states.into_first().into_parts();

        assert_layer_states(&value, 1.0, 11.0, 101.0, 1_001.0);

        assert_layer_states(&first, 2.0, 12.0, 102.0, 1_002.0);
    }

    #[test]
    fn layer_boundary_states_into_second_preserves_boundary_and_component() {
        let states = LayerBoundaryStates::new(
            BoundaryState::new(probe(0.0), probe(10.0)),
            BoundaryState::new(probe(100.0), probe(1_000.0)),
        );

        let (value, first, second) = states.into_second().into_parts();

        assert_layer_states(&value, 1.0, 11.0, 101.0, 1_001.0);

        assert_layer_states(&first, 2.0, 12.0, 102.0, 1_002.0);

        assert_layer_states(&second, 4.0, 14.0, 104.0, 1_004.0);
    }

    #[test]
    fn layer_boundary_states_into_bivariate_first_preserves_axes() {
        let states = LayerBoundaryStates::new(
            BoundaryState::new(probe(0.0), probe(10.0)),
            BoundaryState::new(probe(100.0), probe(1_000.0)),
        );

        let (value, axis0, axis1) = states.into_bivariate_first().into_parts();

        assert_layer_states(&value, 1.0, 11.0, 101.0, 1_001.0);

        assert_layer_states(&axis0, 2.0, 12.0, 102.0, 1_002.0);

        assert_layer_states(&axis1, 3.0, 13.0, 103.0, 1_003.0);
    }

    #[test]
    fn layer_boundary_states_into_bivariate_second_preserves_all_indices() {
        let states = LayerBoundaryStates::new(
            BoundaryState::new(probe(0.0), probe(10.0)),
            BoundaryState::new(probe(100.0), probe(1_000.0)),
        );

        let (value, axis0, axis1, axis0_axis0, axis0_axis1, axis1_axis1) =
            states.into_bivariate_second().into_parts();

        assert_layer_states(&value, 1.0, 11.0, 101.0, 1_001.0);

        assert_layer_states(&axis0, 2.0, 12.0, 102.0, 1_002.0);

        assert_layer_states(&axis1, 3.0, 13.0, 103.0, 1_003.0);

        assert_layer_states(&axis0_axis0, 4.0, 14.0, 104.0, 1_004.0);

        assert_layer_states(&axis0_axis1, 5.0, 15.0, 105.0, 1_005.0);

        assert_layer_states(&axis1_axis1, 6.0, 16.0, 106.0, 1_006.0);
    }
}
