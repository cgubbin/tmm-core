use crate::{
    SpatialProfile, SpatialProfileError,
    field::{ScalarField, ScalarFieldView1},
};

use ndarray::Dimension;

/// Time-averaged power associated with a single layer.
///
/// `left_power` is the power entering the layer through its left interface,
/// `right_power` is the power leaving through its right interface, and
/// `absorbed` is the net power dissipated within the layer.
///
/// For passive media,
///
/// ```text
/// left_power = right_power + absorbed
/// ```
///
/// up to numerical error.
///
/// Active media are represented by negative values of `absorbed`.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerPower<R> {
    left_power: R,
    right_power: R,
    absorbed: R,
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
            left_power: self.left_power.profile_last_axis(excitation_index)?,
            right_power: self.right_power.profile_last_axis(excitation_index)?,
            absorbed: self.absorbed.profile_last_axis(excitation_index)?,
        })
    }
}

impl<R> LayerPower<R> {
    pub(crate) fn new(left_power: R, right_power: R, absorbed: R) -> Self {
        Self {
            left_power,
            right_power,
            absorbed,
        }
    }

    /// Return the power at the left of the layer
    pub fn left_power(&self) -> &R {
        &self.left_power
    }

    /// Return the power at the right of the layer
    pub fn right_power(&self) -> &R {
        &self.right_power
    }

    /// Return the power drop over the layer
    pub fn absorbed(&self) -> &R {
        &self.absorbed
    }

    pub fn into_parts(self) -> (R, R, R) {
        (self.left_power, self.right_power, self.absorbed)
    }

    pub fn conservation_residual(&self) -> R
    where
        R: for<'a> std::ops::Sub<&'a R, Output = R> + Clone,
    {
        self.left_power.clone() - &self.right_power - &self.absorbed
    }

    pub fn map<U>(self, mut f: impl FnMut(R) -> U) -> LayerPower<U> {
        LayerPower {
            left_power: f(self.left_power),
            right_power: f(self.right_power),
            absorbed: f(self.absorbed),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::LayerPower;

    #[test]
    fn stores_all_components() {
        let power = LayerPower::new(1, 2, 3);

        assert_eq!(power.left_power(), &1);
        assert_eq!(power.right_power(), &2);
        assert_eq!(power.absorbed(), &3);
    }

    #[test]
    fn into_parts_preserves_order() {
        let power = LayerPower::new(1, 2, 3);

        assert_eq!(power.into_parts(), (1, 2, 3));
    }

    #[test]
    fn map_transforms_all_components() {
        let power = LayerPower::new(1, 2, 3);

        let mapped = power.map(|value| value.to_string());

        assert_eq!(mapped.left_power(), "1");
        assert_eq!(mapped.right_power(), "2");
        assert_eq!(mapped.absorbed(), "3");
    }

    #[test]
    fn accessors_work_for_non_copy_storage() {
        let power = LayerPower::new(
            String::from("left"),
            String::from("right"),
            String::from("absorbed"),
        );

        assert_eq!(power.left_power(), "left");
        assert_eq!(power.right_power(), "right");
        assert_eq!(power.absorbed(), "absorbed");
    }

    #[test]
    fn map_consumes_non_clone_storage() {
        #[derive(Debug, PartialEq)]
        struct NonClone(i32);

        let power = LayerPower::new(NonClone(1), NonClone(2), NonClone(3));

        let mapped = power.map(|value| value.0 * 10);

        assert_eq!(mapped.left_power(), &10);
        assert_eq!(mapped.right_power(), &20);
        assert_eq!(mapped.absorbed(), &30);
    }

    #[test]
    fn clone_and_partial_eq_round_trip() {
        let power = LayerPower::new(1, 2, 3);

        assert_eq!(power.clone(), power);
    }
}
