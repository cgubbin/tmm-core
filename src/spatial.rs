//! Extraction of one-dimensional spatial profiles.
//!
//! Spatially sampled quantities use the final ndarray axis for position.
//! Every preceding axis describes an excitation coordinate, such as vacuum
//! wavenumber or in-plane wavenumber.
//!
//! [`SpatialProfile::profile`] selects one index on each excitation axis and
//! retains the final spatial axis as a one-dimensional borrowed view.

use ndarray::{ArrayView, ArrayView1, Axis, Dimension, Ix1};

/// Error returned when selecting a spatial profile.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SpatialProfileError {
    /// The stored array has no axis available for position.
    #[error("a spatially sampled field must have at least one axis")]
    InvalidFieldDimension,

    /// The number of supplied excitation indices is incorrect.
    #[error("expected {expected} excitation indices, received {actual}")]
    ExcitationRank { expected: usize, actual: usize },

    /// An excitation index is outside its corresponding axis.
    #[error("excitation index {index} is outside axis {axis}, which has length {length}")]
    ExcitationIndexOutOfBounds {
        axis: usize,
        index: usize,
        length: usize,
    },
}

/// A value from which a one-dimensional spatial profile can be borrowed.
///
/// Implementations must interpret the final stored axis as the spatial axis.
/// All preceding axes are selected using `excitation_index`.
pub trait SpatialProfile<ED: Dimension> {
    /// The one-dimensional profile type.
    type Profile<'a>
    where
        Self: 'a;

    /// Select a spatial profile.
    ///
    /// The number of supplied indices must equal the number of stored axes
    /// minus one.
    fn spatial_profile(
        &self,
        excitation_index: &ED,
    ) -> Result<Self::Profile<'_>, SpatialProfileError>;
}

/// Select the final-axis profile from an ndarray view.
///
/// This is the common leaf operation used by scalar, vector, and tensor
/// fields.
pub(crate) fn array_profile<'a, C, ED>(
    values: ArrayView<'a, C, ED::Larger>,
    excitation_index: &ED,
) -> Result<ArrayView1<'a, C>, SpatialProfileError>
where
    ED: Dimension,
    ED::Larger: Dimension,
{
    let mut profile = values.into_dyn();

    for (axis, &index) in excitation_index.slice().iter().enumerate() {
        let length = profile.len_of(Axis(0));

        if index >= length {
            return Err(SpatialProfileError::ExcitationIndexOutOfBounds {
                axis,
                index,
                length,
            });
        }

        // Each selection removes the current first axis. The next
        // original excitation axis therefore becomes Axis(0).
        profile.index_axis_inplace(Axis(0), index);
    }

    profile
        .into_dimensionality::<Ix1>()
        .map_err(|_| SpatialProfileError::InvalidFieldDimension)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array3, IntoDimension, array};

    #[test]
    fn extracts_last_axis_from_three_dimensional_array() {
        let values = Array3::from_shape_fn((2, 3, 4), |(i, j, k)| 100 * i + 10 * j + k);

        let profile = array_profile(values.view(), &[1, 2].into_dimension()).unwrap();

        assert_eq!(profile, array![120, 121, 122, 123]);
    }

    #[test]
    fn accepts_empty_index_for_one_dimensional_array() {
        let values = array![1, 2, 3];

        let profile = array_profile(values.view(), &[].into_dimension()).unwrap();

        assert_eq!(profile, values.view());
    }

    #[test]
    fn rejects_out_of_bounds_excitation_index() {
        let values = Array3::<f64>::zeros((2, 3, 4));

        let error = array_profile::<f64, ndarray::Ix2>(values.view(), &([1, 3].into_dimension()))
            .unwrap_err();

        assert_eq!(
            error,
            SpatialProfileError::ExcitationIndexOutOfBounds {
                axis: 1,
                index: 3,
                length: 3,
            },
        );
    }
}
