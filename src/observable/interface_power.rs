use crate::{
    SpatialProfile, SpatialProfileError,
    field::{ScalarField, ScalarFieldView1},
};

use ndarray::Dimension;

/// Power-flux quantities associated with a pair of direction-labelled waves.
///
/// All quantities use the same signed convention: positive flux points in the
/// global forward stack direction and negative flux points in the reverse
/// direction.
///
/// `net_flux` is the physical time-averaged normal Poynting flux. In lossless
/// propagating media it equals `forward_flux + backward_flux`. In lossy or
/// evanescent media, interference terms may prevent the directional values
/// from forming a complete additive decomposition.
#[derive(Clone, Debug, PartialEq)]
pub struct DirectedPower<R> {
    forward_flux: R,
    backward_flux: R,
    net_flux: R,
}

impl<R, D> SpatialProfile<D::Smaller> for DirectedPower<ScalarField<R, D>>
where
    D: Dimension,
    D::Smaller: Dimension<Larger = D>,
{
    type Profile<'a>
        = DirectedPower<ScalarFieldView1<'a, R>>
    where
        Self: 'a;

    fn spatial_profile(
        &self,
        excitation_index: &D::Smaller,
    ) -> Result<Self::Profile<'_>, SpatialProfileError> {
        Ok(DirectedPower {
            forward_flux: self.forward_flux.profile_last_axis(excitation_index)?,
            backward_flux: self.backward_flux.profile_last_axis(excitation_index)?,
            net_flux: self.net_flux.profile_last_axis(excitation_index)?,
        })
    }
}

impl<R> DirectedPower<R> {
    pub(crate) fn new(forward_flux: R, backward_flux: R, net_flux: R) -> Self {
        Self {
            forward_flux,
            backward_flux,
            net_flux,
        }
    }

    /// Return the flux associated with the forward-labelled wave.
    pub fn forward_flux(&self) -> &R {
        &self.forward_flux
    }

    /// Return the flux associated with the backward-labelled wave.
    ///
    /// This is a signed quantity and is normally negative for a propagating
    /// wave carrying energy in the reverse stack direction.
    pub fn backward_flux(&self) -> &R {
        &self.backward_flux
    }

    /// Return the physical time-averaged normal Poynting flux.
    pub fn net_flux(&self) -> &R {
        &self.net_flux
    }

    pub fn into_parts(self) -> (R, R, R) {
        (self.forward_flux, self.backward_flux, self.net_flux)
    }

    pub fn map<U>(self, mut f: impl FnMut(R) -> U) -> DirectedPower<U> {
        DirectedPower {
            forward_flux: f(self.forward_flux),
            backward_flux: f(self.backward_flux),
            net_flux: f(self.net_flux),
        }
    }
}

/// Normal power flux immediately on either side of an interface.
#[derive(Clone, Debug, PartialEq)]
pub struct InterfacePower<R> {
    left: DirectedPower<R>,
    right: DirectedPower<R>,
}

impl<R, D> SpatialProfile<D::Smaller> for InterfacePower<ScalarField<R, D>>
where
    D: Dimension,
    D::Smaller: Dimension<Larger = D>,
{
    type Profile<'a>
        = InterfacePower<ScalarFieldView1<'a, R>>
    where
        Self: 'a;

    fn spatial_profile(
        &self,
        excitation_index: &D::Smaller,
    ) -> Result<Self::Profile<'_>, SpatialProfileError> {
        Ok(InterfacePower {
            left: self.left.spatial_profile(excitation_index)?,
            right: self.right.spatial_profile(excitation_index)?,
        })
    }
}

impl<R> InterfacePower<R> {
    pub(crate) fn new(left: DirectedPower<R>, right: DirectedPower<R>) -> Self {
        Self { left, right }
    }

    pub fn left(&self) -> &DirectedPower<R> {
        &self.left
    }

    pub fn right(&self) -> &DirectedPower<R> {
        &self.right
    }

    pub fn left_net_flux(&self) -> &R {
        self.left.net_flux()
    }

    pub fn right_net_flux(&self) -> &R {
        self.right.net_flux()
    }

    pub fn into_parts(self) -> (DirectedPower<R>, DirectedPower<R>) {
        (self.left, self.right)
    }

    pub fn map<U>(self, mut f: impl FnMut(R) -> U) -> InterfacePower<U> {
        InterfacePower {
            left: self.left.map(&mut f),
            right: self.right.map(f),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{DirectedPower, InterfacePower};

    #[test]
    fn directed_power_stores_all_fluxes() {
        let power = DirectedPower::new(1, 2, 3);

        assert_eq!(power.forward_flux(), &1);
        assert_eq!(power.backward_flux(), &2);
        assert_eq!(power.net_flux(), &3);
    }

    #[test]
    fn directed_power_into_parts_preserves_order() {
        let power = DirectedPower::new(1, 2, 3);

        assert_eq!(power.into_parts(), (1, 2, 3));
    }

    #[test]
    fn directed_power_map_transforms_all_fluxes() {
        let power = DirectedPower::new(1, 2, 3);

        let mapped = power.map(|value| format!("flux-{value}"));

        assert_eq!(mapped.forward_flux(), "flux-1");
        assert_eq!(mapped.backward_flux(), "flux-2");
        assert_eq!(mapped.net_flux(), "flux-3");
    }

    #[test]
    fn interface_power_stores_both_sides() {
        let left = DirectedPower::new(1, 2, 3);
        let right = DirectedPower::new(4, 5, 6);

        let interface = InterfacePower::new(left.clone(), right.clone());

        assert_eq!(interface.left(), &left);
        assert_eq!(interface.right(), &right);
        assert_eq!(interface.left_net_flux(), &3);
        assert_eq!(interface.right_net_flux(), &6);
    }

    #[test]
    fn interface_power_into_parts_preserves_side_order() {
        let interface =
            InterfacePower::new(DirectedPower::new(1, 2, 3), DirectedPower::new(4, 5, 6));

        let (left, right) = interface.into_parts();

        assert_eq!(left, DirectedPower::new(1, 2, 3));
        assert_eq!(right, DirectedPower::new(4, 5, 6));
    }

    #[test]
    fn interface_power_map_transforms_both_sides() {
        let interface =
            InterfacePower::new(DirectedPower::new(1, 2, 3), DirectedPower::new(4, 5, 6));

        let mapped = interface.map(|value| value.to_string());

        assert_eq!(mapped.left().forward_flux(), "1");
        assert_eq!(mapped.left().backward_flux(), "2");
        assert_eq!(mapped.left().net_flux(), "3");
        assert_eq!(mapped.right().forward_flux(), "4");
        assert_eq!(mapped.right().backward_flux(), "5");
        assert_eq!(mapped.right().net_flux(), "6");
    }

    #[test]
    fn mapping_consumes_non_clone_storage() {
        #[derive(Debug, PartialEq)]
        struct NonClone(i32);

        let interface = InterfacePower::new(
            DirectedPower::new(NonClone(1), NonClone(2), NonClone(3)),
            DirectedPower::new(NonClone(4), NonClone(5), NonClone(6)),
        );

        let mapped = interface.map(|value| value.0 * 10);

        assert_eq!(mapped.left().forward_flux(), &10);
        assert_eq!(mapped.left().backward_flux(), &20);
        assert_eq!(mapped.left().net_flux(), &30);
        assert_eq!(mapped.right().forward_flux(), &40);
        assert_eq!(mapped.right().backward_flux(), &50);
        assert_eq!(mapped.right().net_flux(), &60);
    }
}
