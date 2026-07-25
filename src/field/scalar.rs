//! Scalar-valued array fields.

use ndarray::{Array, ArrayView, ArrayView1, Dimension, Ix1};
use std::ops::{Index, IndexMut};

use crate::{SpatialProfileError, spatial::array_profile};

/// A scalar-valued field stored in an `ndarray`.
///
/// `D` describes the complete array dimension. The type does not assign
/// physical meaning to any axis.
///
/// # Examples
///
/// ```
/// use ndarray::array;
/// use tmm_core::field::ScalarField;
///
/// let field = ScalarField::new(array![1.0, 2.0, 3.0]);
///
/// assert_eq!(field.len(), 3);
/// assert_eq!(field[1], 2.0);
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct ScalarField<C, D>
where
    D: Dimension,
{
    values: Array<C, D>,
}

/// A borrowed one-dimensional scalar-field view.
///
/// This is the natural representation of a spatial profile after all
/// excitation axes have been selected.
#[derive(Clone, Copy, Debug)]
pub struct ScalarFieldView1<'a, C> {
    values: ArrayView1<'a, C>,
}

impl<C, D> ScalarField<C, D>
where
    D: Dimension,
{
    /// Construct a scalar field from an owned array.
    pub fn new(values: Array<C, D>) -> Self {
        Self { values }
    }

    /// Borrow the underlying array.
    pub fn values(&self) -> &Array<C, D> {
        &self.values
    }

    /// Mutably borrow the underlying array.
    pub fn values_mut(&mut self) -> &mut Array<C, D> {
        &mut self.values
    }

    /// Consume the field and return the underlying array.
    pub fn into_values(self) -> Array<C, D> {
        self.values
    }

    /// Borrow the underlying array as an ndarray view.
    pub fn view(&self) -> ArrayView<'_, C, D> {
        self.values.view()
    }

    /// Return the field shape.
    pub fn shape(&self) -> &[usize] {
        self.values.shape()
    }

    /// Return the field dimension.
    pub fn raw_dim(&self) -> D {
        self.values.raw_dim()
    }

    /// Return the number of axes.
    pub fn ndim(&self) -> usize {
        self.values.ndim()
    }

    /// Return the total number of values.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Return `true` when the field contains no values.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Iterate over scalar values in ndarray logical order.
    pub fn iter(&self) -> impl Iterator<Item = &C> {
        self.values.iter()
    }

    /// Mutably iterate over scalar values in ndarray logical order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut C> {
        self.values.iter_mut()
    }

    /// Apply a function to every value and return a new scalar field.
    pub fn map<B, F>(&self, map: F) -> ScalarField<B, D>
    where
        F: FnMut(&C) -> B,
    {
        ScalarField::new(self.values.map(map))
    }

    /// Apply a function to every copied value and return a new scalar field.
    pub fn mapv<B, F>(&self, map: F) -> ScalarField<B, D>
    where
        C: Clone,
        F: FnMut(C) -> B,
    {
        ScalarField::new(self.values.mapv(map))
    }

    pub fn profile_last_axis(
        &self,
        excitation_index: &D::Smaller,
    ) -> Result<ScalarFieldView1<'_, C>, SpatialProfileError>
    where
        D::Smaller: Dimension<Larger = D>,
    {
        let values = array_profile(self.values.view(), excitation_index)?;

        Ok(ScalarFieldView1::new(values))
    }
}

impl<C> ScalarField<C, Ix1> {
    /// Borrow this one-dimensional field as a [`ScalarFieldView1`].
    pub fn view1(&self) -> ScalarFieldView1<'_, C> {
        ScalarFieldView1::new(self.values.view())
    }
}

impl<'a, C> ScalarFieldView1<'a, C> {
    /// Construct a one-dimensional scalar-field view.
    pub fn new(values: ArrayView1<'a, C>) -> Self {
        Self { values }
    }

    /// Return the underlying ndarray view.
    pub fn values(&self) -> ArrayView1<'a, C> {
        self.values
    }

    /// Return the profile length.
    pub fn len(&self) -> usize {
        self.values.len()
    }

    /// Return `true` when the profile is empty.
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Return the value at `index`, or `None` when it is out of bounds.
    pub fn get(&self, index: usize) -> Option<&C> {
        self.values.get(index)
    }

    /// Iterate over the profile values.
    pub fn iter(&self) -> impl Iterator<Item = &C> + '_ {
        self.values.iter()
    }

    /// Copy the profile into an owned scalar field.
    pub fn to_owned(&self) -> ScalarField<C, Ix1>
    where
        C: Clone,
    {
        ScalarField::new(self.values.to_owned())
    }

    /// Map the borrowed profile into an owned scalar field.
    pub fn map<B, F>(&self, map: F) -> ScalarField<B, Ix1>
    where
        F: FnMut(&C) -> B,
    {
        ScalarField::new(self.values.map(map))
    }
}

impl<C, D> From<Array<C, D>> for ScalarField<C, D>
where
    D: Dimension,
{
    fn from(values: Array<C, D>) -> Self {
        Self::new(values)
    }
}

impl<C, D> From<ScalarField<C, D>> for Array<C, D>
where
    D: Dimension,
{
    fn from(field: ScalarField<C, D>) -> Self {
        field.into_values()
    }
}

impl<C, D> AsRef<Array<C, D>> for ScalarField<C, D>
where
    D: Dimension,
{
    fn as_ref(&self) -> &Array<C, D> {
        self.values()
    }
}

impl<C, D, I> Index<I> for ScalarField<C, D>
where
    D: Dimension,
    I: ndarray::NdIndex<D>,
{
    type Output = C;

    fn index(&self, index: I) -> &Self::Output {
        &self.values[index]
    }
}

impl<C, D, I> IndexMut<I> for ScalarField<C, D>
where
    D: Dimension,
    I: ndarray::NdIndex<D>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.values[index]
    }
}

impl<'a, C> Index<usize> for ScalarFieldView1<'a, C> {
    type Output = C;

    fn index(&self, index: usize) -> &Self::Output {
        &self.values[index]
    }
}

impl<'a, C> IntoIterator for &'a ScalarField<C, Ix1> {
    type Item = &'a C;
    type IntoIter = ndarray::iter::Iter<'a, C, Ix1>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter()
    }
}

impl<'a, C> IntoIterator for &'a mut ScalarField<C, Ix1> {
    type Item = &'a mut C;
    type IntoIter = ndarray::iter::IterMut<'a, C, Ix1>;

    fn into_iter(self) -> Self::IntoIter {
        self.values.iter_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array2, array};

    #[test]
    fn construction_preserves_values() {
        let field = ScalarField::new(array![1, 2, 3]);

        assert_eq!(field.values(), &array![1, 2, 3]);
        assert_eq!(field.shape(), &[3]);
        assert_eq!(field.ndim(), 1);
        assert_eq!(field.len(), 3);
        assert!(!field.is_empty());
    }

    #[test]
    fn supports_multidimensional_arrays() {
        let field = ScalarField::new(Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap());

        assert_eq!(field.shape(), &[2, 2]);
        assert_eq!(field[[1, 0]], 3);
    }

    #[test]
    fn indexing_can_mutate_values() {
        let mut field = ScalarField::new(array![1, 2, 3]);

        field[1] = 10;

        assert_eq!(field[1], 10);
    }

    #[test]
    fn map_preserves_shape() {
        let field = ScalarField::new(array![1, 2, 3]);
        let mapped = field.map(|value| value.to_string());

        assert_eq!(
            mapped.values(),
            &array!["1".to_owned(), "2".to_owned(), "3".to_owned(),],
        );
    }

    #[test]
    fn view1_borrows_values() {
        let field = ScalarField::new(array![1, 2, 3]);
        let view = field.view1();

        assert_eq!(view.len(), 3);
        assert_eq!(view[1], 2);
        assert_eq!(view.get(3), None);
    }

    #[test]
    fn view1_can_be_copied_to_owned_field() {
        let field = ScalarField::new(array![1, 2, 3]);
        let owned = field.view1().to_owned();

        assert_eq!(owned, field);
    }

    #[test]
    fn into_array_recovers_storage() {
        let field = ScalarField::new(array![1, 2, 3]);

        let values: Array<_, Ix1> = field.into();

        assert_eq!(values, array![1, 2, 3]);
    }

    #[test]
    fn iteration_visits_all_values() {
        let field = ScalarField::new(array![1, 2, 3]);

        let values: Vec<_> = field.iter().copied().collect();

        assert_eq!(values, vec![1, 2, 3]);
    }
}

#[cfg(test)]
mod spatial_profile_tests {
    use super::*;
    use crate::{SpatialProfile, SpatialProfileError};
    use ndarray::{Array1, Array2, Array3, IntoDimension, Ix0, Ix1, Ix2, arr1, arr2, array};

    #[test]
    fn extracts_profile_from_scalar_field() {
        let field = ScalarField::new(Array3::from_shape_fn((2, 3, 4), |(i, j, k)| {
            100 * i + 10 * j + k
        }));

        let profile = field.profile_last_axis(&[1, 2].into_dimension()).unwrap();

        assert_eq!(profile.values(), array![120, 121, 122, 123].view(),);
    }

    #[test]
    fn extracts_profile_from_one_dimensional_field() {
        let field = ScalarField::new(array![1, 2, 3]);

        let profile = field.profile_last_axis(&[].into_dimension()).unwrap();

        assert_eq!(profile.values(), field.values().view());
    }

    #[test]
    fn scalar_field_profiles_point_response() {
        // Excitation dimension: Ix0
        // Stored field dimension: Ix1
        let values = arr1(&[10.0, 11.0, 12.0]);

        let field = ScalarField::new(values);

        let profile = field
            .profile_last_axis(&Ix0())
            .expect("point profile should succeed");

        assert_eq!(profile.values(), arr1(&[10.0, 11.0, 12.0]).view());
    }

    #[test]
    fn scalar_field_profiles_one_dimensional_excitation() {
        // Shape: (excitation, position)
        let values = arr2(&[[10.0, 11.0, 12.0], [20.0, 21.0, 22.0], [30.0, 31.0, 32.0]]);

        let field = ScalarField::new(values);

        let profile = field
            .profile_last_axis(&Ix1(1))
            .expect("profile should succeed");

        assert_eq!(profile.values(), arr1(&[20.0, 21.0, 22.0]).view());
    }

    #[test]
    fn scalar_field_profiles_two_dimensional_excitation() {
        // Shape: (spectral, in_plane, position)
        let values = Array3::from_shape_fn((2, 3, 4), |(i, j, k)| {
            100.0 * i as f64 + 10.0 * j as f64 + k as f64
        });

        let field = ScalarField::new(values);

        let profile = field
            .profile_last_axis(&Ix2(1, 2))
            .expect("profile should succeed");

        assert_eq!(profile.values(), arr1(&[120.0, 121.0, 122.0, 123.0]).view(),);
    }

    #[test]
    fn scalar_field_profile_rejects_out_of_bounds_index() {
        let values = Array2::<f64>::zeros((2, 3));
        let field = ScalarField::new(values);

        let error = field
            .profile_last_axis(&Ix1(2))
            .expect_err("index should be out of bounds");

        assert!(matches!(
            error,
            SpatialProfileError::ExcitationIndexOutOfBounds { .. }
        ));
    }

    #[test]
    fn scalar_field_profile_is_a_view() {
        let values = arr2(&[[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]]);

        let field = ScalarField::new(values);

        let profile = field
            .profile_last_axis(&Ix1(1))
            .expect("profile should succeed");

        let source_ptr = field.values().as_ptr();

        // The selected second row starts three elements after the array base.
        let expected_ptr = unsafe { source_ptr.add(3) };

        assert_eq!(profile.values().as_ptr(), expected_ptr);
    }
}
