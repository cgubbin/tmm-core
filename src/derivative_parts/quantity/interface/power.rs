use crate::{
    derivative_parts::{
        BivariateFirstParts, BivariateSecondParts, DirectionalFirstParts, DirectionalSecondParts,
        IntoBivariateFirst, IntoBivariateSecond, IntoFirst, IntoSecond, IntoValue, ValuePart,
    },
    observable::{DirectedPower, InterfacePower},
};

impl<R> IntoValue for DirectedPower<R>
where
    R: IntoValue,
{
    type Value = DirectedPower<R::Value>;

    fn into_value(self) -> ValuePart<Self::Value> {
        let (forward, backward, net) = self.into_parts();

        ValuePart::new(DirectedPower::new(
            forward.into_value().into_inner(),
            backward.into_value().into_inner(),
            net.into_value().into_inner(),
        ))
    }
}

impl<R> IntoFirst for DirectedPower<R>
where
    R: IntoFirst,
{
    fn into_first(self) -> DirectionalFirstParts<Self::Value> {
        let (forward, backward, net) = self.into_parts();

        let (forward, forward_first) = forward.into_first().into_parts();

        let (backward, backward_first) = backward.into_first().into_parts();

        let (net, net_first) = net.into_first().into_parts();

        DirectionalFirstParts::new(
            DirectedPower::new(forward, backward, net),
            DirectedPower::new(forward_first, backward_first, net_first),
        )
    }
}

impl<R> IntoSecond for DirectedPower<R>
where
    R: IntoSecond,
{
    fn into_second(self) -> DirectionalSecondParts<Self::Value> {
        let (forward, backward, net) = self.into_parts();

        let (forward, forward_first, forward_second) = forward.into_second().into_parts();

        let (backward, backward_first, backward_second) = backward.into_second().into_parts();

        let (net, net_first, net_second) = net.into_second().into_parts();

        DirectionalSecondParts::new(
            DirectedPower::new(forward, backward, net),
            DirectedPower::new(forward_first, backward_first, net_first),
            DirectedPower::new(forward_second, backward_second, net_second),
        )
    }
}

impl<R> IntoBivariateFirst for DirectedPower<R>
where
    R: IntoBivariateFirst,
{
    fn into_bivariate_first(self) -> BivariateFirstParts<Self::Value> {
        let (forward, backward, net) = self.into_parts();

        let (forward, forward_axis0, forward_axis1) = forward.into_bivariate_first().into_parts();

        let (backward, backward_axis0, backward_axis1) =
            backward.into_bivariate_first().into_parts();

        let (net, net_axis0, net_axis1) = net.into_bivariate_first().into_parts();

        BivariateFirstParts::new(
            DirectedPower::new(forward, backward, net),
            DirectedPower::new(forward_axis0, backward_axis0, net_axis0),
            DirectedPower::new(forward_axis1, backward_axis1, net_axis1),
        )
    }
}

impl<R> IntoBivariateSecond for DirectedPower<R>
where
    R: IntoBivariateSecond,
{
    fn into_bivariate_second(self) -> BivariateSecondParts<Self::Value> {
        let (forward, backward, net) = self.into_parts();

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

        let (net, net_axis0, net_axis1, net_axis0_axis0, net_axis0_axis1, net_axis1_axis1) =
            net.into_bivariate_second().into_parts();

        BivariateSecondParts::new(
            DirectedPower::new(forward, backward, net),
            DirectedPower::new(forward_axis0, backward_axis0, net_axis0),
            DirectedPower::new(forward_axis1, backward_axis1, net_axis1),
            DirectedPower::new(forward_axis0_axis0, backward_axis0_axis0, net_axis0_axis0),
            DirectedPower::new(forward_axis0_axis1, backward_axis0_axis1, net_axis0_axis1),
            DirectedPower::new(forward_axis1_axis1, backward_axis1_axis1, net_axis1_axis1),
        )
    }
}

impl<A> IntoValue for InterfacePower<A>
where
    A: IntoValue,
{
    type Value = InterfacePower<A::Value>;

    fn into_value(self) -> ValuePart<Self::Value> {
        let (left, right) = self.into_parts();
        ValuePart::new(InterfacePower::new(
            left.into_value().into_inner(),
            right.into_value().into_inner(),
        ))
    }
}

impl<A> IntoFirst for InterfacePower<A>
where
    A: IntoFirst,
{
    fn into_first(self) -> DirectionalFirstParts<Self::Value> {
        let (left, right) = self.into_parts();

        let (left, left_first) = left.into_first().into_parts();
        let (right, right_first) = right.into_first().into_parts();

        DirectionalFirstParts::new(
            InterfacePower::new(left, right),
            InterfacePower::new(left_first, right_first),
        )
    }
}

impl<A> IntoSecond for InterfacePower<A>
where
    A: IntoSecond,
{
    fn into_second(self) -> DirectionalSecondParts<Self::Value> {
        let (left, right) = self.into_parts();

        let (left, left_first, left_second) = left.into_second().into_parts();
        let (right, right_first, right_second) = right.into_second().into_parts();

        DirectionalSecondParts::new(
            InterfacePower::new(left, right),
            InterfacePower::new(left_first, right_first),
            InterfacePower::new(left_second, right_second),
        )
    }
}

impl<A> IntoBivariateFirst for InterfacePower<A>
where
    A: IntoBivariateFirst,
{
    fn into_bivariate_first(self) -> BivariateFirstParts<Self::Value> {
        let (left, right) = self.into_parts();

        let (left, left_axis0, left_axis1) = left.into_bivariate_first().into_parts();
        let (right, right_axis0, right_axis1) = right.into_bivariate_first().into_parts();

        BivariateFirstParts::new(
            InterfacePower::new(left, right),
            InterfacePower::new(left_axis0, right_axis0),
            InterfacePower::new(left_axis1, right_axis1),
        )
    }
}

impl<A> IntoBivariateSecond for InterfacePower<A>
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
            InterfacePower::new(left, right),
            InterfacePower::new(left_axis0, right_axis0),
            InterfacePower::new(left_axis1, right_axis1),
            InterfacePower::new(left_axis0_axis0, right_axis0_axis0),
            InterfacePower::new(left_axis0_axis1, right_axis0_axis1),
            InterfacePower::new(left_axis1_axis1, right_axis1_axis1),
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

    fn directed_power(offset: f64) -> DirectedPower<Probe> {
        DirectedPower::new(probe(offset), probe(offset + 10.0), probe(offset + 20.0))
    }

    fn interface_power(left_offset: f64, right_offset: f64) -> InterfacePower<Probe> {
        InterfacePower::new(directed_power(left_offset), directed_power(right_offset))
    }

    fn assert_directed_power(actual: &DirectedPower<f64>, forward: f64, backward: f64, net: f64) {
        assert_eq!(actual.forward_flux(), &forward);
        assert_eq!(actual.backward_flux(), &backward);
        assert_eq!(actual.net_flux(), &net);
    }

    fn assert_interface_power(
        actual: &InterfacePower<f64>,
        left: (f64, f64, f64),
        right: (f64, f64, f64),
    ) {
        assert_directed_power(actual.left(), left.0, left.1, left.2);

        assert_directed_power(actual.right(), right.0, right.1, right.2);
    }

    // ---------------------------------------------------------------------
    // DirectedPower
    // ---------------------------------------------------------------------

    #[test]
    fn directed_power_into_value_preserves_flux_order() {
        let power = directed_power(0.0);

        let value = power.into_value().into_inner();

        assert_directed_power(&value, 1.0, 11.0, 21.0);
    }

    #[test]
    fn directed_power_into_first_preserves_every_flux() {
        let power = directed_power(0.0);

        let (value, first) = power.into_first().into_parts();

        assert_directed_power(&value, 1.0, 11.0, 21.0);

        assert_directed_power(&first, 2.0, 12.0, 22.0);
    }

    #[test]
    fn directed_power_into_second_preserves_every_flux_and_order() {
        let power = directed_power(0.0);

        let (value, first, second) = power.into_second().into_parts();

        assert_directed_power(&value, 1.0, 11.0, 21.0);

        assert_directed_power(&first, 2.0, 12.0, 22.0);

        assert_directed_power(&second, 4.0, 14.0, 24.0);
    }

    #[test]
    fn directed_power_into_bivariate_first_preserves_axes() {
        let power = directed_power(0.0);

        let (value, axis0, axis1) = power.into_bivariate_first().into_parts();

        assert_directed_power(&value, 1.0, 11.0, 21.0);

        assert_directed_power(&axis0, 2.0, 12.0, 22.0);

        assert_directed_power(&axis1, 3.0, 13.0, 23.0);
    }

    #[test]
    fn directed_power_into_bivariate_second_preserves_all_branches() {
        let power = directed_power(0.0);

        let (value, axis0, axis1, axis0_axis0, axis0_axis1, axis1_axis1) =
            power.into_bivariate_second().into_parts();

        assert_directed_power(&value, 1.0, 11.0, 21.0);

        assert_directed_power(&axis0, 2.0, 12.0, 22.0);

        assert_directed_power(&axis1, 3.0, 13.0, 23.0);

        assert_directed_power(&axis0_axis0, 4.0, 14.0, 24.0);

        assert_directed_power(&axis0_axis1, 5.0, 15.0, 25.0);

        assert_directed_power(&axis1_axis1, 6.0, 16.0, 26.0);
    }

    // ---------------------------------------------------------------------
    // InterfacePower
    // ---------------------------------------------------------------------

    #[test]
    fn interface_power_into_value_preserves_side_and_flux_order() {
        let power = interface_power(0.0, 100.0);

        let value = power.into_value().into_inner();

        assert_interface_power(&value, (1.0, 11.0, 21.0), (101.0, 111.0, 121.0));
    }

    #[test]
    fn interface_power_into_first_preserves_side_and_flux_order() {
        let power = interface_power(0.0, 100.0);

        let (value, first) = power.into_first().into_parts();

        assert_interface_power(&value, (1.0, 11.0, 21.0), (101.0, 111.0, 121.0));

        assert_interface_power(&first, (2.0, 12.0, 22.0), (102.0, 112.0, 122.0));
    }

    #[test]
    fn interface_power_into_second_preserves_side_and_flux_order() {
        let power = interface_power(0.0, 100.0);

        let (value, first, second) = power.into_second().into_parts();

        assert_interface_power(&value, (1.0, 11.0, 21.0), (101.0, 111.0, 121.0));

        assert_interface_power(&first, (2.0, 12.0, 22.0), (102.0, 112.0, 122.0));

        assert_interface_power(&second, (4.0, 14.0, 24.0), (104.0, 114.0, 124.0));
    }

    #[test]
    fn interface_power_into_bivariate_first_preserves_axes_and_sides() {
        let power = interface_power(0.0, 100.0);

        let (value, axis0, axis1) = power.into_bivariate_first().into_parts();

        assert_interface_power(&value, (1.0, 11.0, 21.0), (101.0, 111.0, 121.0));

        assert_interface_power(&axis0, (2.0, 12.0, 22.0), (102.0, 112.0, 122.0));

        assert_interface_power(&axis1, (3.0, 13.0, 23.0), (103.0, 113.0, 123.0));
    }

    #[test]
    fn interface_power_into_bivariate_second_preserves_all_indices() {
        let power = interface_power(0.0, 100.0);

        let (value, axis0, axis1, axis0_axis0, axis0_axis1, axis1_axis1) =
            power.into_bivariate_second().into_parts();

        assert_interface_power(&value, (1.0, 11.0, 21.0), (101.0, 111.0, 121.0));

        assert_interface_power(&axis0, (2.0, 12.0, 22.0), (102.0, 112.0, 122.0));

        assert_interface_power(&axis1, (3.0, 13.0, 23.0), (103.0, 113.0, 123.0));

        assert_interface_power(&axis0_axis0, (4.0, 14.0, 24.0), (104.0, 114.0, 124.0));

        assert_interface_power(&axis0_axis1, (5.0, 15.0, 25.0), (105.0, 115.0, 125.0));

        assert_interface_power(&axis1_axis1, (6.0, 16.0, 26.0), (106.0, 116.0, 126.0));
    }
}
