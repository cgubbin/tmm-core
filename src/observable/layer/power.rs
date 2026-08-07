
use crate::{
    InterfacePower,
    algebra::ScalarAlgebra,
    observable::{Interfaces, Layers},
};

/// Normalized power balance across one finite layer.
///
/// Fluxes use the global left-to-right sign convention. `absorbed` is the
/// normalized power removed within the layer:
///
/// ```text
/// absorbed = left_flux - right_flux
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct LayerPower<R> {
    left_flux: R,
    right_flux: R,
    absorbed: R,
}

impl<R> LayerPower<R> {
    pub(crate) const fn new(left_flux: R, right_flux: R, absorbed: R) -> Self {
        Self {
            left_flux,
            right_flux,
            absorbed,
        }
    }

    /// Return the net flux at the layer's left boundary.
    pub fn left_flux(&self) -> &R {
        &self.left_flux
    }

    /// Return the net flux at the layer's right boundary.
    pub fn right_flux(&self) -> &R {
        &self.right_flux
    }

    /// Return the normalized power dissipated within the finite layer.
    ///
    /// For passive media this is non-negative. It is obtained from the signed
    /// flux difference:
    ///
    /// ```text
    /// dissipated = left_flux - right_flux
    /// ```
    pub fn absorbed(&self) -> &R {
        &self.absorbed
    }

    /// Consume the result and return `(left_flux, right_flux, absorbed)`.
    pub fn into_parts(self) -> (R, R, R) {
        (self.left_flux, self.right_flux, self.absorbed)
    }

    /// Return the residual of the layer power-balance identity.
    ///
    /// ```text
    /// left_flux - right_flux - absorbed
    /// ```
    ///
    /// An internally consistent result should be zero up to numerical error.
    pub fn conservation_residual(&self) -> R
    where
        R: ScalarAlgebra,
    {
        self.left_flux
            .clone()
            .subtract(&self.right_flux)
            .subtract(&self.absorbed)
    }

    /// Transform every stored power component.
    pub fn map<U>(self, mut map: impl FnMut(R) -> U) -> LayerPower<U> {
        LayerPower {
            left_flux: map(self.left_flux),
            right_flux: map(self.right_flux),
            absorbed: map(self.absorbed),
        }
    }
}

impl<R> Interfaces<InterfacePower<R>> {
    pub(crate) fn into_layer_power(self) -> Layers<LayerPower<R>>
    where
        R: ScalarAlgebra,
    {
        let interfaces = self.into_inner();

        let mut layers = Vec::with_capacity(interfaces.len().saturating_sub(1));

        for pair in interfaces.windows(2) {
            let left_flux = pair[0].right_net_flux().clone();

            let right_flux = pair[1].left_net_flux().clone();

            let absorbed = left_flux.subtract(&right_flux);

            layers.push(LayerPower::new(left_flux, right_flux, absorbed));
        }

        Layers::new(layers)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Ix0, arr0};

    use super::*;
    use crate::{
        algebra::{ArrayJet0, Jet0, RealParameter},
        observable::{DirectedPower, InterfacePower},
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
    fn layer_power_stores_fluxes_and_absorption() {
        let power = LayerPower::new(jet(0.9), jet(0.6), jet(0.3));

        assert_eq!(scalar(power.left_flux()), 0.9);
        assert_eq!(scalar(power.right_flux()), 0.6);
        assert_eq!(scalar(power.absorbed()), 0.3);
    }

    #[test]
    fn layer_power_into_parts_preserves_order() {
        let power = LayerPower::new(jet(1.0), jet(2.0), jet(3.0));

        let (left, right, absorbed) = power.into_parts();

        assert_eq!(scalar(&left), 1.0);
        assert_eq!(scalar(&right), 2.0);
        assert_eq!(scalar(&absorbed), 3.0);
    }

    #[test]
    fn layer_power_map_transforms_every_component() {
        let power = LayerPower::new(1, 2, 3);

        let mapped = power.map(|value| format!("power-{value}"));

        assert_eq!(mapped.left_flux(), "power-1");
        assert_eq!(mapped.right_flux(), "power-2");
        assert_eq!(mapped.absorbed(), "power-3");
    }

    #[test]
    fn layer_power_map_supports_non_clone_storage() {
        #[derive(Debug, PartialEq)]
        struct NonClone(i32);

        let power = LayerPower::new(NonClone(1), NonClone(2), NonClone(3));

        let mapped = power.map(|value| value.0 * 10);

        assert_eq!(mapped.left_flux(), &10);
        assert_eq!(mapped.right_flux(), &20);
        assert_eq!(mapped.absorbed(), &30);
    }

    #[test]
    fn interface_power_projects_adjacent_internal_fluxes() {
        let interfaces = Interfaces::new(vec![
            interface_power(100.0, 0.9),
            interface_power(0.7, 200.0),
        ]);

        let layers = interfaces.into_layer_power();

        let layer = layers.first().unwrap();

        assert_eq!(scalar(layer.left_flux()), 0.9);
        assert_eq!(scalar(layer.right_flux()), 0.7);

        approx::assert_relative_eq!(scalar(&layer.absorbed()), 0.2, epsilon = 1e-15);
    }

    #[test]
    fn one_interface_produces_no_finite_layer_power() {
        let layers = Interfaces::new(vec![interface_power(1.0, 1.0)]).into_layer_power();

        assert!(layers.is_empty());
    }

    #[test]
    fn conservation_residual_is_zero_for_consistent_components() {
        let power = LayerPower::new(jet(0.9), jet(0.6), jet(0.3));

        approx::assert_relative_eq!(scalar(&power.conservation_residual()), 0.0, epsilon = 1e-15);
    }
}
