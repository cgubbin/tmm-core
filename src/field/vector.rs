//! Cartesian vector-valued array fields.

use crate::{
    algebra::{
        JetAdditive, JetConjugate, JetCrossProduct, JetHermitianProduct, JetMultiplyByScalar,
        JetRealPart, JetScaleBy,
    },
    field::FieldShapeError,
    spatial::{SpatialProfileError, array_profile},
};
use nalgebra::ComplexField;
use ndarray::{Array, ArrayView, ArrayView1, Dimension, Ix1};
use num_traits::Zero;

/// One Cartesian vector value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VectorValue<C> {
    pub x: C,
    pub y: C,
    pub z: C,
}

impl<C> VectorValue<C> {
    /// Construct a Cartesian vector value.
    pub fn new(x: C, y: C, z: C) -> Self {
        Self { x, y, z }
    }

    /// Map each component into a new vector value.
    pub fn map<B, F>(self, mut map: F) -> VectorValue<B>
    where
        F: FnMut(C) -> B,
    {
        VectorValue {
            x: map(self.x),
            y: map(self.y),
            z: map(self.z),
        }
    }

    /// Convert the vector into an array in `x`, `y`, `z` order.
    pub fn into_array(self) -> [C; 3] {
        [self.x, self.y, self.z]
    }
}

impl<C> From<[C; 3]> for VectorValue<C> {
    fn from([x, y, z]: [C; 3]) -> Self {
        Self { x, y, z }
    }
}

impl<C> From<VectorValue<C>> for [C; 3] {
    fn from(value: VectorValue<C>) -> Self {
        value.into_array()
    }
}

/// A Cartesian vector field.
///
/// Every component must have the same shape.
#[derive(Clone, Debug, PartialEq)]
pub struct VectorField<C, D>
where
    D: Dimension,
{
    x: Array<C, D>,
    y: Array<C, D>,
    z: Array<C, D>,
}

/// A borrowed one-dimensional Cartesian vector-field view.
#[derive(Clone, Copy, Debug)]
pub struct VectorFieldView1<'a, C> {
    x: ArrayView1<'a, C>,
    y: ArrayView1<'a, C>,
    z: ArrayView1<'a, C>,
}

impl<C, D> VectorField<C, D>
where
    D: Dimension,
{
    /// Construct a vector field after checking component shapes.
    pub fn new(x: Array<C, D>, y: Array<C, D>, z: Array<C, D>) -> Result<Self, FieldShapeError> {
        let expected = x.shape();

        if y.shape() != expected {
            return Err(FieldShapeError::new("y", expected, y.shape()));
        }

        if z.shape() != expected {
            return Err(FieldShapeError::new("z", expected, z.shape()));
        }

        Ok(Self { x, y, z })
    }

    /// Construct a vector field without checking component shapes.
    ///
    /// This is intended for internal code where matching shapes are already
    /// guaranteed by construction.
    pub(crate) fn new_unchecked(x: Array<C, D>, y: Array<C, D>, z: Array<C, D>) -> Self {
        debug_assert_eq!(x.shape(), y.shape());
        debug_assert_eq!(x.shape(), z.shape());

        Self { x, y, z }
    }

    pub fn x(&self) -> &Array<C, D> {
        &self.x
    }

    pub fn y(&self) -> &Array<C, D> {
        &self.y
    }

    pub fn z(&self) -> &Array<C, D> {
        &self.z
    }

    pub fn x_mut(&mut self) -> &mut Array<C, D> {
        &mut self.x
    }

    pub fn y_mut(&mut self) -> &mut Array<C, D> {
        &mut self.y
    }

    pub fn z_mut(&mut self) -> &mut Array<C, D> {
        &mut self.z
    }

    /// Borrow all component arrays.
    pub fn components(&self) -> (&Array<C, D>, &Array<C, D>, &Array<C, D>) {
        (&self.x, &self.y, &self.z)
    }

    /// Consume the field and return its component arrays.
    pub fn into_components(self) -> (Array<C, D>, Array<C, D>, Array<C, D>) {
        (self.x, self.y, self.z)
    }

    /// Return ndarray views of all components.
    pub fn view(
        &self,
    ) -> (
        ArrayView<'_, C, D>,
        ArrayView<'_, C, D>,
        ArrayView<'_, C, D>,
    ) {
        (self.x.view(), self.y.view(), self.z.view())
    }

    /// Return the shared component shape.
    pub fn shape(&self) -> &[usize] {
        self.x.shape()
    }

    /// Return the shared component dimension.
    pub fn raw_dim(&self) -> D {
        self.x.raw_dim()
    }

    /// Return the number of axes in each component.
    pub fn ndim(&self) -> usize {
        self.x.ndim()
    }

    /// Return the number of vectors stored by the field.
    pub fn len(&self) -> usize {
        self.x.len()
    }

    /// Return `true` when the field contains no vectors.
    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    /// Return references to the vector at `index`.
    pub fn get<I>(&self, index: I) -> Option<VectorValue<&C>>
    where
        I: ndarray::NdIndex<D> + Clone,
    {
        Some(VectorValue {
            x: self.x.get(index.clone())?,
            y: self.y.get(index.clone())?,
            z: self.z.get(index)?,
        })
    }

    /// Apply a function independently to every scalar component.
    pub fn map<B, F>(&self, mut map: F) -> VectorField<B, D>
    where
        F: FnMut(&C) -> B,
    {
        VectorField::new_unchecked(self.x.map(&mut map), self.y.map(&mut map), self.z.map(map))
    }

    /// Apply a function to each complete Cartesian vector.
    pub fn map_vectors<B, F>(&self, mut map: F) -> ScalarField<B, D>
    where
        F: FnMut(VectorValue<&C>) -> B,
    {
        let values = ndarray::Zip::from(&self.x)
            .and(&self.y)
            .and(&self.z)
            .map_collect(|x, y, z| map(VectorValue { x, y, z }));

        ScalarField::new(values)
    }

    /// Construct a zero vector with the same sampling shape as `values`.
    pub(crate) fn zeros_like(values: &Array<C, D>) -> Self
    where
        C: Clone + Zero,
    {
        let dimension = values.raw_dim();

        Self::new_unchecked(
            Array::from_elem(dimension.clone(), C::zero()),
            Array::from_elem(dimension.clone(), C::zero()),
            Array::from_elem(dimension, C::zero()),
        )
    }

    /// Return the pointwise complex conjugate
    pub(crate) fn conjugate(&self) -> Self
    where
        C: ComplexField + Copy,
    {
        self.map(|value| value.conjugate())
    }

    /// Calculate the pointwise bilinear Cartesian dot product.
    ///
    /// Complex conjugation is not applied. Use [`Self::hermitian_dot`] for a conjugating inner
    /// product
    pub(crate) fn dot(&self, rhs: &Self) -> Array<C, D>
    where
        C: ComplexField + Copy,
    {
        self.x.clone() * rhs.x.view()
            + self.y.clone() * rhs.y.view()
            + self.z.clone() * rhs.z.view()
    }

    /// Compute the pointwise Hermitian inner product.
    ///
    /// This evaluates `conjugate(self) · rhs`, and is conjugate-linear in
    /// `self` and linear in `rhs`.
    pub(crate) fn hermitian_dot(&self, rhs: &Self) -> Array<C, D>
    where
        C: ComplexField + Copy,
    {
        self.conjugate().dot(rhs)
    }

    /// Compute the pointwise bilinear Cartesian cross product.
    ///
    /// At every sampling index this returns:
    ///
    /// ```text
    /// [
    ///     self.y * rhs.z - self.z * rhs.y,
    ///     self.z * rhs.x - self.x * rhs.z,
    ///     self.x * rhs.y - self.y * rhs.x,
    /// ]
    /// ```
    ///
    /// Complex conjugation is not applied.
    pub(crate) fn cross(&self, rhs: &Self) -> Self
    where
        C: ComplexField + Copy,
    {
        Self::new_unchecked(
            self.y.clone() * rhs.z.view() - self.z.clone() * rhs.y.view(),
            self.z.clone() * rhs.x.view() - self.x.clone() * rhs.z.view(),
            self.x.clone() * rhs.y.view() - self.y.clone() * rhs.x.view(),
        )
    }

    /// Return the pointwise squared Euclidean magnitude.
    ///
    /// For complex-valued vectors this is:
    ///
    /// ```text
    /// |x|² + |y|² + |z|²
    /// ```
    ///
    /// The result is real and non-negative up to floating-point roundoff.
    pub(crate) fn magnitude_squared(&self) -> Array<C::RealField, D>
    where
        C: ComplexField + Copy,
    {
        self.x.mapv(|value| value.modulus_squared())
            + self.y.mapv(|value| value.modulus_squared())
            + self.z.mapv(|value| value.modulus_squared())
    }

    pub(crate) fn profile_last_axis(
        &self,
        excitation_index: &D::Smaller,
    ) -> Result<VectorFieldView1<'_, C>, SpatialProfileError>
    where
        D::Smaller: Dimension<Larger = D>,
    {
        let x = array_profile(self.x.view(), excitation_index)?;

        let y = array_profile(self.y.view(), excitation_index)?;

        let z = array_profile(self.z.view(), excitation_index)?;

        // VectorField guarantees matching component shapes, so profile
        // extraction preserves matching lengths.
        Ok(VectorFieldView1::new_unchecked(x, y, z))
    }
}

use crate::field::ScalarField;

impl<C> VectorField<C, Ix1> {
    /// Borrow this one-dimensional vector field as a profile view.
    pub fn view1(&self) -> VectorFieldView1<'_, C> {
        VectorFieldView1::new(self.x.view(), self.y.view(), self.z.view())
            .expect("owned vector-field components have matching shapes")
    }
}

impl<'a, C> VectorFieldView1<'a, C> {
    /// Construct a vector-field profile view after checking lengths.
    pub fn new(
        x: ArrayView1<'a, C>,
        y: ArrayView1<'a, C>,
        z: ArrayView1<'a, C>,
    ) -> Result<Self, FieldShapeError> {
        let expected = x.shape();

        if y.shape() != expected {
            return Err(FieldShapeError::new("y", expected, y.shape()));
        }

        if z.shape() != expected {
            return Err(FieldShapeError::new("z", expected, z.shape()));
        }

        Ok(Self { x, y, z })
    }

    pub(crate) fn new_unchecked(
        x: ArrayView1<'a, C>,
        y: ArrayView1<'a, C>,
        z: ArrayView1<'a, C>,
    ) -> Self {
        debug_assert_eq!(x.shape(), y.shape());
        debug_assert_eq!(x.shape(), z.shape());

        Self { x, y, z }
    }

    pub fn x(&self) -> ArrayView1<'a, C> {
        self.x
    }

    pub fn y(&self) -> ArrayView1<'a, C> {
        self.y
    }

    pub fn z(&self) -> ArrayView1<'a, C> {
        self.z
    }

    pub fn components(&self) -> (ArrayView1<'a, C>, ArrayView1<'a, C>, ArrayView1<'a, C>) {
        (self.x, self.y, self.z)
    }

    pub fn len(&self) -> usize {
        self.x.len()
    }

    pub fn is_empty(&self) -> bool {
        self.x.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<VectorValue<&C>> {
        Some(VectorValue {
            x: self.x.get(index)?,
            y: self.y.get(index)?,
            z: self.z.get(index)?,
        })
    }

    pub fn to_owned(&self) -> VectorField<C, Ix1>
    where
        C: Clone,
    {
        VectorField::new_unchecked(self.x.to_owned(), self.y.to_owned(), self.z.to_owned())
    }

    pub fn map<B, F>(&self, mut map: F) -> VectorField<B, Ix1>
    where
        F: FnMut(&C) -> B,
    {
        VectorField::new_unchecked(self.x.map(&mut map), self.y.map(&mut map), self.z.map(map))
    }
}

impl<C, D> std::ops::Add<&VectorField<C, D>> for VectorField<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = VectorField<C, D>;

    fn add(self, rhs: &Self) -> Self::Output {
        VectorField::new_unchecked(
            self.x + rhs.x.view(),
            self.y + rhs.y.view(),
            self.z + rhs.z.view(),
        )
    }
}

impl<C, D> std::ops::Add for VectorField<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = VectorField<C, D>;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new_unchecked(
            self.x + rhs.x.view(),
            self.y + rhs.y.view(),
            self.z + rhs.z.view(),
        )
    }
}

impl<C, D> std::ops::Add<&VectorField<C, D>> for &VectorField<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = VectorField<C, D>;

    fn add(self, rhs: &VectorField<C, D>) -> Self::Output {
        VectorField::new_unchecked(&self.x + &rhs.x, &self.y + &rhs.y, &self.z + &rhs.z)
    }
}

impl<C, D> std::ops::Sub<&VectorField<C, D>> for VectorField<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = VectorField<C, D>;

    fn sub(self, rhs: &Self) -> Self::Output {
        VectorField::new_unchecked(
            self.x - rhs.x.view(),
            self.y - rhs.y.view(),
            self.z - rhs.z.view(),
        )
    }
}

impl<C, D> std::ops::Sub for VectorField<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = VectorField<C, D>;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new_unchecked(
            self.x - rhs.x.view(),
            self.y - rhs.y.view(),
            self.z - rhs.z.view(),
        )
    }
}

impl<C, D> std::ops::Sub<&VectorField<C, D>> for &VectorField<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = VectorField<C, D>;

    fn sub(self, rhs: &VectorField<C, D>) -> Self::Output {
        VectorField::new_unchecked(&self.x - &rhs.x, &self.y - &rhs.y, &self.z - &rhs.z)
    }
}

impl<C, D> std::ops::Mul<Array<C, D>> for VectorField<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = VectorField<C, D>;

    fn mul(self, factor: Array<C, D>) -> Self::Output {
        assert_eq!(
            self.x.raw_dim(),
            factor.raw_dim(),
            "Cartesian vector and scalar field must have identical shapes",
        );
        Self::new_unchecked(
            self.x * factor.view(),
            self.y * factor.view(),
            self.z * factor.view(),
        )
    }
}

impl<C, D> std::ops::Mul<&Array<C, D>> for VectorField<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = VectorField<C, D>;

    fn mul(self, factor: &Array<C, D>) -> Self::Output {
        debug_assert_eq!(self.x.raw_dim(), factor.raw_dim());
        Self::new_unchecked(
            self.x * factor.view(),
            self.y * factor.view(),
            self.z * factor.view(),
        )
    }
}

impl<C, D> std::ops::Mul<&Array<C, D>> for &VectorField<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = VectorField<C, D>;

    fn mul(self, factor: &Array<C, D>) -> Self::Output {
        VectorField::new_unchecked(&self.x * factor, &self.y * factor, &self.z * factor)
    }
}

impl<C, D> std::ops::Mul<C> for VectorField<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = VectorField<C, D>;

    fn mul(self, factor: C) -> Self::Output {
        Self::new_unchecked(
            self.x.mapv(|v| v * factor),
            self.y.mapv(|v| v * factor),
            self.z.mapv(|v| v * factor),
        )
    }
}

impl<C, D> std::ops::Mul<C> for &VectorField<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = VectorField<C, D>;

    fn mul(self, factor: C) -> Self::Output {
        VectorField::new_unchecked(
            self.x.mapv(|value| value * factor),
            self.y.mapv(|value| value * factor),
            self.z.mapv(|value| value * factor),
        )
    }
}

impl<C, D> std::ops::Neg for VectorField<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = VectorField<C, D>;

    fn neg(self) -> Self::Output {
        Self::new_unchecked(-self.x, -self.y, -self.z)
    }
}

impl<C, D> JetAdditive for VectorField<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn jet_add(&self, rhs: &Self) -> Self {
        self + rhs
    }

    fn jet_subtract(&self, rhs: &Self) -> Self {
        self - rhs
    }

    fn jet_negate(&self) -> Self {
        -self.clone()
    }
}

impl<T, D> JetMultiplyByScalar<Array<T, D>> for VectorField<T, D>
where
    T: ComplexField + Copy,
    D: Dimension,
{
    fn jet_multiply_by_scalar(&self, scalar: &Array<T, D>) -> Self {
        self * scalar
    }
}

impl<C, D> JetScaleBy for VectorField<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Scalar = C;

    fn jet_scale_by(&self, value: C) -> Self {
        self * value
    }
}

impl<C, D> JetCrossProduct for VectorField<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn jet_cross(&self, rhs: &Self) -> Self {
        self.cross(rhs)
    }
}

impl<C, D> JetHermitianProduct for VectorField<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = Array<C, D>;

    fn jet_hermitian_product(&self, rhs: &Self) -> Self::Output {
        self.hermitian_dot(rhs)
    }
}

impl<C, D> JetConjugate for VectorField<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn jet_conjugate(&self) -> Self {
        self.conjugate()
    }
}

impl<C, D> JetRealPart for VectorField<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type RealOutput = VectorField<C::RealField, D>;

    fn jet_real(&self) -> Self::RealOutput {
        self.map(|x| nalgebra::ComplexField::real(*x))
    }

    fn jet_imaginary(&self) -> Self::RealOutput {
        self.map(|x| nalgebra::ComplexField::imaginary(*x))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array1, Array2, Ix1, arr1, array};
    use num_complex::Complex64;

    type C = Complex64;
    type D = Ix1;

    const TOLERANCE: f64 = 1.0e-12;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn vector(x: &[C], y: &[C], z: &[C]) -> VectorField<C, D> {
        VectorField::new_unchecked(
            Array1::from_vec(x.to_vec()),
            Array1::from_vec(y.to_vec()),
            Array1::from_vec(z.to_vec()),
        )
    }

    fn assert_real_close(actual: f64, expected: f64) {
        let error = (actual - expected).abs();

        assert!(
            error <= TOLERANCE,
            "expected {expected:e}, got {actual:e}; \
             absolute error = {error:e}",
        );
    }

    fn assert_complex_close(actual: C, expected: C) {
        let error = (actual - expected).norm();

        assert!(
            error <= TOLERANCE,
            "expected {expected:?}, got {actual:?}; \
             absolute error = {error:e}",
        );
    }

    fn assert_real_array_close(actual: &Array1<f64>, expected: &Array1<f64>) {
        assert_eq!(actual.raw_dim(), expected.raw_dim());

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_real_close(actual, expected);
        }
    }

    fn assert_complex_array_close(actual: &Array1<C>, expected: &Array1<C>) {
        assert_eq!(actual.raw_dim(), expected.raw_dim());

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_complex_close(actual, expected);
        }
    }

    #[test]
    fn constructor_accepts_matching_shapes() {
        let field = VectorField::new(array![1, 2], array![3, 4], array![5, 6]).unwrap();

        assert_eq!(field.shape(), &[2]);
        assert_eq!(field.get(1), Some(VectorValue::new(&2, &4, &6)),);
    }

    #[test]
    fn constructor_rejects_mismatched_y_shape() {
        let error = VectorField::new(array![1, 2], array![3], array![5, 6]).unwrap_err();

        assert_eq!(error.component(), "y");
        assert_eq!(error.expected(), &[2]);
        assert_eq!(error.actual(), &[1]);
    }

    #[test]
    fn constructor_rejects_mismatched_z_shape() {
        let error = VectorField::new(array![1, 2], array![3, 4], array![5]).unwrap_err();

        assert_eq!(error.component(), "z");
    }

    #[test]
    fn supports_multidimensional_components() {
        let component = Array2::from_shape_vec((2, 2), vec![1, 2, 3, 4]).unwrap();

        let field = VectorField::new(component.clone(), component.clone(), component).unwrap();

        assert_eq!(field.get([1, 0]), Some(VectorValue::new(&3, &3, &3)),);
    }

    #[test]
    fn view1_borrows_components() {
        let field = VectorField::new(array![1, 2], array![3, 4], array![5, 6]).unwrap();

        let view = field.view1();

        assert_eq!(view.len(), 2);
        assert_eq!(view.get(0), Some(VectorValue::new(&1, &3, &5)),);
    }

    #[test]
    fn view1_can_be_copied_to_owned_field() {
        let field = VectorField::new(array![1, 2], array![3, 4], array![5, 6]).unwrap();

        assert_eq!(field.view1().to_owned(), field);
    }

    #[test]
    fn map_applies_to_every_component() {
        let field = VectorField::new(array![1, 2], array![3, 4], array![5, 6]).unwrap();

        let mapped = field.map(|value| value * 2);

        assert_eq!(mapped.x(), &array![2, 4]);
        assert_eq!(mapped.y(), &array![6, 8]);
        assert_eq!(mapped.z(), &array![10, 12]);
    }

    #[test]
    fn map_vectors_produces_scalar_field() {
        let field = VectorField::new(array![1, 2], array![3, 4], array![5, 6]).unwrap();

        let sums = field.map_vectors(|vector| vector.x + vector.y + vector.z);

        assert_eq!(sums.values(), &array![9, 12]);
    }

    #[test]
    fn conjugate_applies_to_every_component() {
        let vector = vector(&[c(1.0, 2.0)], &[c(3.0, -4.0)], &[c(-5.0, 6.0)]);

        let conjugate = vector.conjugate();

        assert_eq!(conjugate.x(), &arr1(&[c(1.0, -2.0)]));
        assert_eq!(conjugate.y(), &arr1(&[c(3.0, 4.0)]));
        assert_eq!(conjugate.z(), &arr1(&[c(-5.0, -6.0)]));
    }

    #[test]
    fn addition_is_componentwise_for_owned_rhs() {
        let lhs = vector(&[c(1.0, 2.0)], &[c(3.0, 4.0)], &[c(5.0, 6.0)]);

        let rhs = vector(&[c(7.0, 8.0)], &[c(9.0, 10.0)], &[c(11.0, 12.0)]);

        let sum = lhs + rhs;

        assert_eq!(sum.x(), &arr1(&[c(8.0, 10.0)]));
        assert_eq!(sum.y(), &arr1(&[c(12.0, 14.0)]));
        assert_eq!(sum.z(), &arr1(&[c(16.0, 18.0)]));
    }

    #[test]
    fn addition_is_componentwise_for_borrowed_rhs() {
        let lhs = vector(&[c(1.0, 2.0)], &[c(3.0, 4.0)], &[c(5.0, 6.0)]);

        let rhs = vector(&[c(7.0, 8.0)], &[c(9.0, 10.0)], &[c(11.0, 12.0)]);

        let sum = lhs + &rhs;

        assert_eq!(sum.x(), &arr1(&[c(8.0, 10.0)]));
        assert_eq!(sum.y(), &arr1(&[c(12.0, 14.0)]));
        assert_eq!(sum.z(), &arr1(&[c(16.0, 18.0)]));
    }

    #[test]
    fn subtraction_is_componentwise_for_owned_rhs() {
        let lhs = vector(&[c(1.0, 2.0)], &[c(3.0, 4.0)], &[c(5.0, 6.0)]);

        let rhs = vector(&[c(-7.0, -8.0)], &[c(-9.0, -10.0)], &[c(-11.0, -12.0)]);

        let diff = lhs - rhs;

        assert_eq!(diff.x(), &arr1(&[c(8.0, 10.0)]));
        assert_eq!(diff.y(), &arr1(&[c(12.0, 14.0)]));
        assert_eq!(diff.z(), &arr1(&[c(16.0, 18.0)]));
    }

    #[test]
    fn subtraction_is_componentwise_for_borrowed_rhs() {
        let lhs = vector(&[c(1.0, 2.0)], &[c(3.0, 4.0)], &[c(5.0, 6.0)]);

        let rhs = vector(&[c(-7.0, -8.0)], &[c(-9.0, -10.0)], &[c(-11.0, -12.0)]);

        let diff = lhs - &rhs;

        assert_eq!(diff.x(), &arr1(&[c(8.0, 10.0)]));
        assert_eq!(diff.y(), &arr1(&[c(12.0, 14.0)]));
        assert_eq!(diff.z(), &arr1(&[c(16.0, 18.0)]));
    }

    #[test]
    fn negation_produces_expected_output() {
        let value = vector(&[c(1.0, 2.0)], &[c(3.0, 4.0)], &[c(5.0, 6.0)]);

        let neg = -value.clone();

        assert_eq!(neg.x(), -value.x());
        assert_eq!(neg.y(), -value.y());
        assert_eq!(neg.z(), -value.z());
    }

    #[test]
    fn scalar_multiplication_is_componentwise() {
        let vector = vector(&[c(1.0, 2.0)], &[c(3.0, 4.0)], &[c(5.0, 6.0)]);

        let scaled = vector * c(2.0, 0.0);

        assert_eq!(scaled.x(), &arr1(&[c(2.0, 4.0)]));
        assert_eq!(scaled.y(), &arr1(&[c(6.0, 8.0)]));
        assert_eq!(scaled.z(), &arr1(&[c(10.0, 12.0)]));
    }

    #[test]
    fn owned_array_multiplication_is_pointwise() {
        let vector = vector(
            &[c(1.0, 0.0), c(2.0, 0.0)],
            &[c(3.0, 0.0), c(4.0, 0.0)],
            &[c(5.0, 0.0), c(6.0, 0.0)],
        );

        let factor = arr1(&[c(2.0, 0.0), c(3.0, 0.0)]);

        let scaled = vector * factor;

        assert_eq!(scaled.x(), &arr1(&[c(2.0, 0.0), c(6.0, 0.0)]));
        assert_eq!(scaled.y(), &arr1(&[c(6.0, 0.0), c(12.0, 0.0)]));
        assert_eq!(scaled.z(), &arr1(&[c(10.0, 0.0), c(18.0, 0.0)]));
    }

    #[test]
    fn borrowed_array_multiplication_is_pointwise() {
        let vector = vector(
            &[c(1.0, 0.0), c(2.0, 0.0)],
            &[c(3.0, 0.0), c(4.0, 0.0)],
            &[c(5.0, 0.0), c(6.0, 0.0)],
        );

        let factor = arr1(&[c(2.0, 0.0), c(3.0, 0.0)]);

        let scaled = vector * &factor;

        assert_eq!(scaled.x(), &arr1(&[c(2.0, 0.0), c(6.0, 0.0)]));
        assert_eq!(scaled.y(), &arr1(&[c(6.0, 0.0), c(12.0, 0.0)]));
        assert_eq!(scaled.z(), &arr1(&[c(10.0, 0.0), c(18.0, 0.0)]));
    }

    #[test]
    fn dot_product_is_bilinear_without_conjugation() {
        let lhs = vector(&[c(1.0, 1.0)], &[c(2.0, -1.0)], &[c(-1.0, 2.0)]);

        let rhs = vector(&[c(3.0, -2.0)], &[c(1.0, 4.0)], &[c(2.0, 1.0)]);

        let expected =
            c(1.0, 1.0) * c(3.0, -2.0) + c(2.0, -1.0) * c(1.0, 4.0) + c(-1.0, 2.0) * c(2.0, 1.0);

        assert_complex_array_close(&lhs.dot(&rhs), &arr1(&[expected]));
    }

    #[test]
    fn hermitian_dot_conjugates_lhs() {
        let rhs = vector(&[c(1.0, 1.0)], &[c(2.0, -1.0)], &[c(-1.0, 2.0)]);

        let lhs = vector(&[c(3.0, -2.0)], &[c(1.0, 4.0)], &[c(2.0, 1.0)]);

        let expected = c(1.0, 1.0) * c(3.0, -2.0).conj()
            + c(2.0, -1.0) * c(1.0, 4.0).conj()
            + c(-1.0, 2.0) * c(2.0, 1.0).conj();

        assert_complex_array_close(&lhs.hermitian_dot(&rhs), &arr1(&[expected]));
    }

    #[test]
    fn into_components_returns_owned_components_in_cartesian_order() {
        let x = arr1(&[c(1.0, 2.0)]);
        let y = arr1(&[c(3.0, 4.0)]);
        let z = arr1(&[c(5.0, 6.0)]);

        let vector = VectorField::new(x.clone(), y.clone(), z.clone()).unwrap();

        let (actual_x, actual_y, actual_z) = vector.into_components();

        assert_eq!(actual_x, x);
        assert_eq!(actual_y, y);
        assert_eq!(actual_z, z);
    }

    #[test]
    fn basis_cross_products_have_right_handed_orientation() {
        let zero = c(0.0, 0.0);
        let one = c(1.0, 0.0);

        let x = vector(&[one], &[zero], &[zero]);
        let y = vector(&[zero], &[one], &[zero]);
        let z = vector(&[zero], &[zero], &[one]);

        assert_eq!(x.cross(&y), z);

        let y = vector(&[zero], &[one], &[zero]);
        let z = vector(&[zero], &[zero], &[one]);
        let x = vector(&[one], &[zero], &[zero]);

        assert_eq!(y.cross(&z), x);

        let z = vector(&[zero], &[zero], &[one]);
        let x = vector(&[one], &[zero], &[zero]);
        let y = vector(&[zero], &[one], &[zero]);

        assert_eq!(z.cross(&x), y);
    }

    #[test]
    fn cross_product_is_antisymmetric() {
        let lhs = vector(&[c(1.0, 2.0)], &[c(3.0, 4.0)], &[c(5.0, 6.0)]);

        let rhs = vector(&[c(7.0, 8.0)], &[c(9.0, 10.0)], &[c(11.0, 12.0)]);

        let lhs_cross_rhs = lhs.cross(&rhs);
        let rhs_cross_lhs = rhs.cross(&lhs);

        assert_complex_array_close(lhs_cross_rhs.x(), &rhs_cross_lhs.x().mapv(|value| -value));
        assert_complex_array_close(lhs_cross_rhs.y(), &rhs_cross_lhs.y().mapv(|value| -value));
        assert_complex_array_close(lhs_cross_rhs.z(), &rhs_cross_lhs.z().mapv(|value| -value));
    }

    #[test]
    fn magnitude_squared_sums_component_moduli() {
        let vector = vector(
            &[c(3.0, 4.0), c(1.0, 0.0)],
            &[c(0.0, 2.0), c(0.0, 2.0)],
            &[c(1.0, 0.0), c(2.0, 0.0)],
        );

        assert_real_array_close(
            &vector.magnitude_squared(),
            &arr1(&[25.0 + 4.0 + 1.0, 1.0 + 4.0 + 4.0]),
        );
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed\n  left: [1]\n right: [2]")]
    fn constructor_panics_on_mismatched_x_and_y_shapes() {
        VectorField::new_unchecked(
            arr1(&[c(1.0, 0.0)]),
            arr1(&[c(2.0, 0.0), c(3.0, 0.0)]),
            arr1(&[c(4.0, 0.0)]),
        );
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed\n  left: [1]\n right: [2]")]
    fn constructor_rejects_mismatched_x_and_z_shapes() {
        VectorField::new_unchecked(
            arr1(&[c(1.0, 0.0)]),
            arr1(&[c(2.0, 0.0)]),
            arr1(&[c(3.0, 0.0), c(4.0, 0.0)]),
        );
    }

    #[test]
    fn new_constructor_rejects_mismatched_x_and_y_shapes() {
        let result = VectorField::new(
            arr1(&[c(1.0, 0.0)]),
            arr1(&[c(2.0, 0.0), c(3.0, 0.0)]),
            arr1(&[c(4.0, 0.0)]),
        );

        assert!(result.is_err());
    }

    #[test]
    fn new_constructor_rejects_mismatched_x_and_z_shapes() {
        let result = VectorField::new(
            arr1(&[c(1.0, 0.0)]),
            arr1(&[c(4.0, 0.0)]),
            arr1(&[c(2.0, 0.0), c(3.0, 0.0)]),
        );

        assert!(result.is_err());
    }

    #[test]
    fn new_constructor_accepts_equal_shapes() {
        let result = VectorField::new(
            arr1(&[c(2.0, 0.0), c(3.0, 0.0)]),
            arr1(&[c(2.0, 0.0), c(3.0, 0.0)]),
            arr1(&[c(2.0, 0.0), c(3.0, 0.0)]),
        );

        assert!(result.is_ok());
    }

    #[test]
    fn hermitian_self_product_equals_magnitude_squared() {
        let value = vector(
            &[c(3.0, 4.0), c(1.0, -2.0)],
            &[c(0.0, 2.0), c(-3.0, 1.0)],
            &[c(1.0, 0.0), c(2.0, 2.0)],
        );

        let inner = value.hermitian_dot(&value);
        let magnitude = value.magnitude_squared();

        assert_real_array_close(&inner.mapv(|value| value.real()), &magnitude);

        assert!(
            inner
                .iter()
                .all(|value| { value.imaginary().abs() <= TOLERANCE },),
        );
    }

    #[test]
    fn hermitian_dot_has_conjugate_symmetry() {
        let lhs = vector(&[c(1.0, 1.0)], &[c(2.0, -1.0)], &[c(-1.0, 2.0)]);

        let rhs = vector(&[c(3.0, -2.0)], &[c(1.0, 4.0)], &[c(2.0, 1.0)]);

        let lhs_rhs = lhs.hermitian_dot(&rhs);

        let rhs_lhs = rhs.hermitian_dot(&lhs).mapv(|value| value.conjugate());

        assert_complex_array_close(&lhs_rhs, &rhs_lhs);
    }

    #[test]
    fn cross_product_is_bilinearly_orthogonal_to_both_operands() {
        let lhs = vector(
            &[c(1.0, 2.0), c(-0.5, 0.25)],
            &[c(3.0, -1.0), c(2.0, 1.5)],
            &[c(-2.0, 0.5), c(0.75, -1.0)],
        );

        let rhs = vector(
            &[c(0.5, -1.0), c(1.0, 2.0)],
            &[c(-1.5, 0.25), c(-0.5, 0.75)],
            &[c(2.0, 1.0), c(3.0, -0.25)],
        );

        let cross = lhs.cross(&rhs);

        let zero = arr1(&[C::zero(), C::zero()]);

        assert_complex_array_close(&lhs.dot(&cross), &zero);

        assert_complex_array_close(&rhs.dot(&cross), &zero);
    }

    #[test]
    fn cross_product_is_evaluated_independently_at_each_sample() {
        let zero = c(0.0, 0.0);
        let one = c(1.0, 0.0);

        let lhs = vector(&[one, zero], &[zero, one], &[zero, zero]);

        let rhs = vector(&[zero, zero], &[one, zero], &[zero, one]);

        let result = lhs.cross(&rhs);

        assert_eq!(result.x(), &arr1(&[zero, one]),);

        assert_eq!(result.y(), &arr1(&[zero, zero]),);

        assert_eq!(result.z(), &arr1(&[one, zero]),);
    }

    #[test]
    fn conjugation_is_an_involution() {
        let value = vector(&[c(1.0, 2.0)], &[c(3.0, -4.0)], &[c(-5.0, 6.0)]);

        assert_eq!(value.conjugate().conjugate(), value,);
    }

    #[test]
    fn conjugation_distributes_over_cross_product() {
        let lhs = vector(&[c(1.0, 2.0)], &[c(3.0, -4.0)], &[c(-5.0, 6.0)]);

        let rhs = vector(&[c(7.0, -8.0)], &[c(9.0, 10.0)], &[c(11.0, -12.0)]);

        assert_eq!(
            lhs.cross(&rhs).conjugate(),
            lhs.conjugate().cross(&rhs.conjugate(),),
        );
    }

    #[test]
    fn jet_payload_traits_delegate_to_cartesian_operations() {
        let lhs = vector(&[c(1.0, 2.0)], &[c(3.0, -4.0)], &[c(-5.0, 6.0)]);

        let rhs = vector(&[c(7.0, -8.0)], &[c(9.0, 10.0)], &[c(11.0, -12.0)]);

        let scalar = arr1(&[c(2.0, -1.0)]);

        assert_eq!(JetAdditive::jet_add(&lhs, &rhs,), &lhs + &rhs,);

        assert_eq!(JetCrossProduct::jet_cross(&lhs, &rhs,), lhs.cross(&rhs),);

        assert_eq!(
            JetMultiplyByScalar::jet_multiply_by_scalar(&lhs, &scalar,),
            &lhs * &scalar,
        );

        assert_eq!(JetConjugate::jet_conjugate(&lhs,), lhs.conjugate(),);

        assert_eq!(
            JetHermitianProduct::jet_hermitian_product(&lhs, &rhs,),
            lhs.hermitian_dot(&rhs),
        );
    }
}

#[cfg(test)]
mod spatial_profile_tests {
    use super::*;
    use ndarray::{Array2, Array3, IntoDimension, Ix2, arr1, array};

    #[test]
    fn extracts_all_vector_components_at_same_coordinate() {
        let x = Array2::from_shape_fn((2, 3), |(i, k)| 100 * i + k);

        let y = Array2::from_shape_fn((2, 3), |(i, k)| 1_000 + 100 * i + k);

        let z = Array2::from_shape_fn((2, 3), |(i, k)| 2_000 + 100 * i + k);

        let field = VectorField::new(x, y, z).unwrap();
        let profile = field.profile_last_axis(&[1].into_dimension()).unwrap();

        assert_eq!(profile.x(), array![100, 101, 102].view());
        assert_eq!(profile.y(), array![1_100, 1_101, 1_102].view(),);
        assert_eq!(profile.z(), array![2_100, 2_101, 2_102].view(),);
    }

    #[test]
    fn vector_field_profiles_all_components() {
        let x = Array3::from_shape_fn((2, 2, 3), |(i, j, k)| {
            100.0 * i as f64 + 10.0 * j as f64 + k as f64
        });

        let y = x.mapv(|value| value + 1_000.0);
        let z = x.mapv(|value| value + 2_000.0);

        let field = VectorField::new(x, y, z).unwrap();

        let profile = field
            .profile_last_axis(&Ix2(1, 0))
            .expect("profile should succeed");

        assert_eq!(profile.x(), arr1(&[100.0, 101.0, 102.0]).view(),);
        assert_eq!(profile.y(), arr1(&[1100.0, 1101.0, 1102.0]).view(),);
        assert_eq!(profile.z(), arr1(&[2100.0, 2101.0, 2102.0]).view(),);
    }
}
