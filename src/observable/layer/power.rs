use ndarray::Dimension;

use crate::{
    SpatialProfile, SpatialProfileError,
    field::{ScalarField, ScalarFieldView1},
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
    pub fn dissipated(&self) -> &R {
        &self.absorbed
    }

    pub(crate) fn absorbed(&self) -> &R {
        &self.absorbed
    }

    pub fn into_parts(self) -> (R, R, R) {
        (self.left_flux, self.right_flux, self.absorbed)
    }

    pub fn conservation_residual(&self) -> R
    where
        R: for<'a> std::ops::Sub<&'a R, Output = R> + Clone,
    {
        self.left_flux.clone() - &self.right_flux - &self.absorbed
    }

    pub fn map<U>(self, mut map: impl FnMut(R) -> U) -> LayerPower<U> {
        LayerPower {
            left_flux: map(self.left_flux),
            right_flux: map(self.right_flux),
            absorbed: map(self.absorbed),
        }
    }
}

impl<R, D> SpatialProfile<D::Smaller> for LayerPower<ScalarField<R, D>>
where
    D: Dimension,
    D::Smaller: Dimension<Larger = D>,
{
    type Profile<'a>
        = LayerPower<ScalarFieldView1<'a, R>>
    where
        Self: 'a;

    fn spatial_profile(
        &self,
        excitation_index: &D::Smaller,
    ) -> Result<Self::Profile<'_>, SpatialProfileError> {
        Ok(LayerPower {
            left_flux: self.left_flux.profile_last_axis(excitation_index)?,
            right_flux: self.right_flux.profile_last_axis(excitation_index)?,
            absorbed: self.absorbed.profile_last_axis(excitation_index)?,
        })
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
}
