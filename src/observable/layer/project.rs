use crate::{
    InterfacePower, LayerPower,
    algebra::ScalarAlgebra,
    observable::{Interfaces, layer::Layers},
};

pub(crate) fn project_layer_power<R>(
    interfaces: Interfaces<InterfacePower<R>>,
) -> Layers<LayerPower<R>>
where
    R: ScalarAlgebra,
{
    let interfaces = interfaces.into_inner();

    let mut layers = Vec::with_capacity(interfaces.len().saturating_sub(1));

    for pair in interfaces.windows(2) {
        let left_flux = pair[0].right_net_flux().clone();

        let right_flux = pair[1].left_net_flux().clone();

        let absorbed = left_flux.subtract(&right_flux);

        layers.push(LayerPower::new(left_flux, right_flux, absorbed));
    }

    Layers::new(layers)
}

#[cfg(test)]
mod tests {
    use ndarray::{Ix0, arr0};

    use super::*;
    use crate::{
        algebra::{ArrayJet0, Jet0, RealParameter},
        observable::{DirectedPower, InterfacePower, Interfaces},
        test_support::{TOLERANCE, assertions::assert_real_close},
    };

    type A = ArrayJet0<f64, Ix0, RealParameter>;

    fn jet(value: f64) -> A {
        Jet0::new(arr0(value))
    }

    fn scalar(value: &A) -> f64 {
        value.value()[()]
    }

    fn interface_power(left_net_flux: f64, right_net_flux: f64) -> InterfacePower<A> {
        InterfacePower::new(
            DirectedPower::new(
                jet(100.0 + left_net_flux),
                jet(200.0 + left_net_flux),
                jet(left_net_flux),
            ),
            DirectedPower::new(
                jet(300.0 + right_net_flux),
                jet(400.0 + right_net_flux),
                jet(right_net_flux),
            ),
        )
    }

    #[test]
    fn empty_interface_sequence_produces_no_layers() {
        let layers = project_layer_power::<A>(Interfaces::new(Vec::new()));

        assert!(layers.is_empty());
    }

    #[test]
    fn one_interface_produces_no_finite_layers() {
        let layers = project_layer_power(Interfaces::new(vec![interface_power(0.8, 0.8)]));

        assert!(layers.is_empty());
    }

    #[test]
    fn two_interfaces_produce_one_layer() {
        let layers = project_layer_power(Interfaces::new(vec![
            interface_power(0.9, 0.8),
            interface_power(0.6, 0.5),
        ]));

        assert_eq!(layers.len(), 1);

        let layer = layers.get(0).unwrap();

        /*
         * Use the finite-layer sides:
         *
         * left boundary  = interface 0 right
         * right boundary = interface 1 left
         */
        assert_real_close(scalar(layer.left_flux()), 0.8, TOLERANCE);
        assert_real_close(scalar(layer.right_flux()), 0.6, TOLERANCE);
        assert_real_close(scalar(layer.absorbed()), 0.2, TOLERANCE);
    }

    #[test]
    fn multiple_layers_preserve_physical_order() {
        let layers = project_layer_power(Interfaces::new(vec![
            interface_power(1.0, 0.9),
            interface_power(0.8, 0.7),
            interface_power(0.4, 0.3),
            interface_power(0.1, 0.0),
        ]));

        assert_eq!(layers.len(), 3);

        let first = layers.get(0).unwrap();
        let second = layers.get(1).unwrap();
        let third = layers.get(2).unwrap();

        assert_real_close(scalar(first.left_flux()), 0.9, TOLERANCE);
        assert_real_close(scalar(first.right_flux()), 0.8, TOLERANCE);
        assert_real_close(scalar(first.absorbed()), 0.1, TOLERANCE);

        assert_real_close(scalar(second.left_flux()), 0.7, TOLERANCE);
        assert_real_close(scalar(second.right_flux()), 0.4, TOLERANCE);
        assert_real_close(scalar(second.absorbed()), 0.3, TOLERANCE);

        assert_real_close(scalar(third.left_flux()), 0.3, TOLERANCE);
        assert_real_close(scalar(third.right_flux()), 0.1, TOLERANCE);
        assert_real_close(scalar(third.absorbed()), 0.2, TOLERANCE);
    }

    #[test]
    fn absorption_uses_global_flux_difference_for_negative_flux() {
        /*
         * This represents right incidence. Flux remains globally signed:
         *
         * left boundary  = -0.4
         * right boundary = -0.7
         *
         * absorbed = -0.4 - (-0.7) = +0.3
         */
        let layers = project_layer_power(Interfaces::new(vec![
            interface_power(-0.5, -0.4),
            interface_power(-0.7, -0.8),
        ]));

        let layer = layers.get(0).unwrap();

        assert_real_close(scalar(layer.left_flux()), -0.4, TOLERANCE);
        assert_real_close(scalar(layer.right_flux()), -0.7, TOLERANCE);
        assert_real_close(scalar(layer.absorbed()), 0.3, TOLERANCE);
    }

    #[test]
    fn lossless_layer_has_zero_absorption() {
        let layers = project_layer_power(Interfaces::new(vec![
            interface_power(0.75, 0.75),
            interface_power(0.75, 0.75),
        ]));

        let layer = layers.get(0).unwrap();

        assert_real_close(scalar(layer.absorbed()), 0.0, TOLERANCE);
    }

    #[test]
    fn projection_uses_finite_layer_sides_not_exterior_sides() {
        let layers = project_layer_power(Interfaces::new(vec![
            /*
             * Deliberately discontinuous marker values. The projection
             * must use the right side of the left interface.
             */
            interface_power(100.0, 0.8),
            /*
             * It must use the left side of the right interface.
             */
            interface_power(0.6, 200.0),
        ]));

        let layer = layers.get(0).unwrap();

        assert_real_close(scalar(layer.left_flux()), 0.8, TOLERANCE);
        assert_real_close(scalar(layer.right_flux()), 0.6, TOLERANCE);
        assert_real_close(scalar(layer.absorbed()), 0.2, TOLERANCE);
    }

    #[test]
    fn summed_layer_absorption_telescopes_when_interfaces_are_continuous() {
        let layers = project_layer_power(Interfaces::new(vec![
            interface_power(1.0, 1.0),
            interface_power(0.8, 0.8),
            interface_power(0.5, 0.5),
            interface_power(0.2, 0.2),
        ]));

        let total: f64 = layers.iter().map(|layer| scalar(layer.absorbed())).sum();

        assert_real_close(total, 0.8, TOLERANCE);
    }
}
