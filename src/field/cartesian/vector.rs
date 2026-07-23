/// A pointwise Cartesian three-vector field over an ndarray sampling domain.
///
/// Each component is an owned ndarray with dimension `D`. All three arrays
/// have identical shapes. At sampling index `i`, the value is therefore:
///
/// ```text
/// [x[i], y[i], z[i]].
/// ```
///
/// Arithmetic, scalar multiplication, dot products and cross products are
/// evaluated independently at every sampling index. Differently shaped
/// sampling domains are not broadcast.
///
/// The Cartesian basis is right-handed:
///
/// ```text
/// e_x × e_y = e_z
/// e_y × e_z = e_x
/// e_z × e_x = e_y
/// ```
///
/// For complex-valued vectors, [`Self::dot`] is bilinear and applies no
/// conjugation. [`Self::hermitian_dot`] uses the convention
/// `conjugate(self) · rhs`.
use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, OwnedRepr};
use num_traits::Zero;

use crate::algebra::{
    JetAdditive, JetConjugate, JetCrossProduct, JetHermitianProduct, JetMultiplyByScalar,
    JetRealPart, JetScaleBy,
};

/// A pointwise Cartesian three-vector over an ndarray sampling domain.
///
/// Each component is stored as an owned ndarray with dimension `D`. All three
/// component arrays have the same shape.
///
/// The vector represents one Cartesian vector at every point in the sampling
/// domain. Algebraic operations are therefore applied pointwise.
#[derive(Clone, Debug, PartialEq)]
pub struct CartesianVector3<T, D>
where
    D: Dimension,
{
    x: ArrayBase<OwnedRepr<T>, D>,
    y: ArrayBase<OwnedRepr<T>, D>,
    z: ArrayBase<OwnedRepr<T>, D>,
}

impl<T, D> CartesianVector3<T, D>
where
    D: Dimension,
{
    /// Construct a Cartesian vector from its component arrays.
    ///
    /// This constructor is crate-private so public field-producing APIs can
    /// guarantee that all components use the same sampling shape.
    pub(crate) fn new(
        x: ArrayBase<OwnedRepr<T>, D>,
        y: ArrayBase<OwnedRepr<T>, D>,
        z: ArrayBase<OwnedRepr<T>, D>,
    ) -> Self {
        assert_eq!(
            x.raw_dim(),
            y.raw_dim(),
            "Cartesian x and y components must have identical shapes",
        );

        assert_eq!(
            x.raw_dim(),
            z.raw_dim(),
            "Cartesian x and z components must have identical shapes",
        );

        Self { x, y, z }
    }

    /// Return the Cartesian x component.
    pub fn x(&self) -> &ArrayBase<OwnedRepr<T>, D> {
        &self.x
    }

    /// Return the Cartesian y component.
    pub fn y(&self) -> &ArrayBase<OwnedRepr<T>, D> {
        &self.y
    }

    /// Return the Cartesian z component.
    pub fn z(&self) -> &ArrayBase<OwnedRepr<T>, D> {
        &self.z
    }

    /// Return the Cartesian components in `[x, y, z]` order.
    pub fn components(&self) -> [&ArrayBase<OwnedRepr<T>, D>; 3] {
        [&self.x, &self.y, &self.z]
    }

    /// Consume the vector and return its component arrays in `(x, y, z)` order.
    pub(crate) fn into_components(
        self,
    ) -> (
        ArrayBase<OwnedRepr<T>, D>,
        ArrayBase<OwnedRepr<T>, D>,
        ArrayBase<OwnedRepr<T>, D>,
    ) {
        (self.x, self.y, self.z)
    }

    /// Construct a zero vector with the same sampling shape as `values`.
    pub(crate) fn zeros_like(values: &ArrayBase<OwnedRepr<T>, D>) -> Self
    where
        T: Clone + Zero,
    {
        let dimension = values.raw_dim();

        Self::new(
            ArrayBase::from_elem(dimension.clone(), T::zero()),
            ArrayBase::from_elem(dimension.clone(), T::zero()),
            ArrayBase::from_elem(dimension, T::zero()),
        )
    }

    /// Apply a pointwise scalar mapping to every Cartesian component.
    pub fn map<U, F>(&self, mut f: F) -> CartesianVector3<U, D>
    where
        F: FnMut(T) -> U,
        T: Clone,
    {
        CartesianVector3::new(self.x.mapv(&mut f), self.y.mapv(&mut f), self.z.mapv(f))
    }

    /// Return the pointwise complex conjugate
    pub fn conjugate(&self) -> Self
    where
        T: ComplexField + Copy,
    {
        self.map(|value| value.conjugate())
    }

    /// Calculate the pointwise bilinear Cartesian dot product.
    ///
    /// Complex conjugation is not applied. Use [`Self::hermitian_dot`] for a conjugating inner
    /// product
    pub fn dot(&self, rhs: &Self) -> ArrayBase<OwnedRepr<T>, D>
    where
        T: ComplexField + Copy,
    {
        self.x.clone() * rhs.x.view()
            + self.y.clone() * rhs.y.view()
            + self.z.clone() * rhs.z.view()
    }

    /// Compute the pointwise Hermitian inner product.
    ///
    /// This evaluates `conjugate(self) · rhs`, and is conjugate-linear in
    /// `self` and linear in `rhs`.
    pub fn hermitian_dot(&self, rhs: &Self) -> ArrayBase<OwnedRepr<T>, D>
    where
        T: ComplexField + Copy,
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
    pub fn cross(&self, rhs: &Self) -> Self
    where
        T: ComplexField + Copy,
    {
        Self::new(
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
    pub fn magnitude_squared(&self) -> ArrayBase<OwnedRepr<T::RealField>, D>
    where
        T: ComplexField + Copy,
    {
        self.x.mapv(|value| value.modulus_squared())
            + self.y.mapv(|value| value.modulus_squared())
            + self.z.mapv(|value| value.modulus_squared())
    }
}

impl<C, D> std::ops::Add<&CartesianVector3<C, D>> for CartesianVector3<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = CartesianVector3<C, D>;

    fn add(self, rhs: &Self) -> Self::Output {
        CartesianVector3::new(
            self.x + rhs.x.view(),
            self.y + rhs.y.view(),
            self.z + rhs.z.view(),
        )
    }
}

impl<C, D> std::ops::Add for CartesianVector3<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = CartesianVector3<C, D>;

    fn add(self, rhs: Self) -> Self::Output {
        Self::new(
            self.x + rhs.x.view(),
            self.y + rhs.y.view(),
            self.z + rhs.z.view(),
        )
    }
}

impl<C, D> std::ops::Add<&CartesianVector3<C, D>> for &CartesianVector3<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = CartesianVector3<C, D>;

    fn add(self, rhs: &CartesianVector3<C, D>) -> Self::Output {
        CartesianVector3::new(&self.x + &rhs.x, &self.y + &rhs.y, &self.z + &rhs.z)
    }
}

impl<C, D> std::ops::Sub<&CartesianVector3<C, D>> for CartesianVector3<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = CartesianVector3<C, D>;

    fn sub(self, rhs: &Self) -> Self::Output {
        CartesianVector3::new(
            self.x - rhs.x.view(),
            self.y - rhs.y.view(),
            self.z - rhs.z.view(),
        )
    }
}

impl<C, D> std::ops::Sub for CartesianVector3<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = CartesianVector3<C, D>;

    fn sub(self, rhs: Self) -> Self::Output {
        Self::new(
            self.x - rhs.x.view(),
            self.y - rhs.y.view(),
            self.z - rhs.z.view(),
        )
    }
}

impl<C, D> std::ops::Sub<&CartesianVector3<C, D>> for &CartesianVector3<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = CartesianVector3<C, D>;

    fn sub(self, rhs: &CartesianVector3<C, D>) -> Self::Output {
        CartesianVector3::new(&self.x - &rhs.x, &self.y - &rhs.y, &self.z - &rhs.z)
    }
}

impl<C, D> std::ops::Mul<ArrayBase<OwnedRepr<C>, D>> for CartesianVector3<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = CartesianVector3<C, D>;

    fn mul(self, factor: ArrayBase<OwnedRepr<C>, D>) -> Self::Output {
        assert_eq!(
            self.x.raw_dim(),
            factor.raw_dim(),
            "Cartesian vector and scalar field must have identical shapes",
        );
        Self::new(
            self.x * factor.view(),
            self.y * factor.view(),
            self.z * factor.view(),
        )
    }
}

impl<C, D> std::ops::Mul<&ArrayBase<OwnedRepr<C>, D>> for CartesianVector3<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = CartesianVector3<C, D>;

    fn mul(self, factor: &ArrayBase<OwnedRepr<C>, D>) -> Self::Output {
        debug_assert_eq!(self.x.raw_dim(), factor.raw_dim());
        Self::new(
            self.x * factor.view(),
            self.y * factor.view(),
            self.z * factor.view(),
        )
    }
}

impl<C, D> std::ops::Mul<&ArrayBase<OwnedRepr<C>, D>> for &CartesianVector3<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = CartesianVector3<C, D>;

    fn mul(self, factor: &ArrayBase<OwnedRepr<C>, D>) -> Self::Output {
        CartesianVector3::new(&self.x * factor, &self.y * factor, &self.z * factor)
    }
}

impl<C, D> std::ops::Mul<C> for CartesianVector3<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = CartesianVector3<C, D>;

    fn mul(self, factor: C) -> Self::Output {
        Self::new(
            self.x.mapv(|v| v * factor),
            self.y.mapv(|v| v * factor),
            self.z.mapv(|v| v * factor),
        )
    }
}

impl<C, D> std::ops::Mul<C> for &CartesianVector3<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = CartesianVector3<C, D>;

    fn mul(self, factor: C) -> Self::Output {
        CartesianVector3::new(
            self.x.mapv(|value| value * factor),
            self.y.mapv(|value| value * factor),
            self.z.mapv(|value| value * factor),
        )
    }
}

impl<C, D> std::ops::Neg for CartesianVector3<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = CartesianVector3<C, D>;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl<C, D> JetAdditive for CartesianVector3<C, D>
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

impl<T, D> JetMultiplyByScalar<ArrayBase<OwnedRepr<T>, D>> for CartesianVector3<T, D>
where
    T: ComplexField + Copy,
    D: Dimension,
{
    fn jet_multiply_by_scalar(&self, scalar: &ArrayBase<OwnedRepr<T>, D>) -> Self {
        self * scalar
    }
}

impl<C, D> JetScaleBy for CartesianVector3<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Scalar = C;

    fn jet_scale_by(&self, value: C) -> Self {
        self * value
    }
}

impl<C, D> JetCrossProduct for CartesianVector3<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn jet_cross(&self, rhs: &Self) -> Self {
        self.cross(rhs)
    }
}

impl<C, D> JetHermitianProduct for CartesianVector3<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Output = ArrayBase<OwnedRepr<C>, D>;

    fn jet_hermitian_product(&self, rhs: &Self) -> Self::Output {
        self.hermitian_dot(rhs)
    }
}

impl<C, D> JetConjugate for CartesianVector3<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn jet_conjugate(&self) -> Self {
        self.conjugate()
    }
}

impl<C, D> JetRealPart for CartesianVector3<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type RealOutput = CartesianVector3<C::RealField, D>;

    fn jet_real(&self) -> Self::RealOutput {
        self.map(nalgebra::ComplexField::real)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Ix1, arr1};
    use num_complex::Complex64;

    use super::*;

    type C = Complex64;
    type D = Ix1;

    const TOLERANCE: f64 = 1.0e-12;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn vector(x: &[C], y: &[C], z: &[C]) -> CartesianVector3<C, D> {
        CartesianVector3::new(
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
    fn components_are_preserved() {
        let x = arr1(&[c(1.0, 2.0), c(3.0, 4.0)]);
        let y = arr1(&[c(5.0, 6.0), c(7.0, 8.0)]);
        let z = arr1(&[c(9.0, 10.0), c(11.0, 12.0)]);

        let vector = CartesianVector3::new(x.clone(), y.clone(), z.clone());

        assert_eq!(vector.x(), &x);
        assert_eq!(vector.y(), &y);
        assert_eq!(vector.z(), &z);
        assert_eq!(vector.components(), [&x, &y, &z]);
    }

    #[test]
    fn zeros_like_preserves_shape() {
        let values = arr1(&[c(1.0, 2.0), c(3.0, 4.0), c(5.0, 6.0)]);

        let zero = CartesianVector3::zeros_like(&values);

        let expected = arr1(&[C::new(0.0, 0.0), C::new(0.0, 0.0), C::new(0.0, 0.0)]);

        assert_eq!(zero.x(), &expected);
        assert_eq!(zero.y(), &expected);
        assert_eq!(zero.z(), &expected);
    }

    #[test]
    fn map_applies_to_every_component() {
        let vector = vector(&[c(1.0, 2.0)], &[c(3.0, 4.0)], &[c(5.0, 6.0)]);

        let real = vector.map(|value| value.re);

        assert_eq!(real.x(), &arr1(&[1.0]));
        assert_eq!(real.y(), &arr1(&[3.0]));
        assert_eq!(real.z(), &arr1(&[5.0]));
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

        let vector = CartesianVector3::new(x.clone(), y.clone(), z.clone());

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
    #[should_panic(expected = "Cartesian x and y components must have identical shapes")]
    fn constructor_rejects_mismatched_x_and_y_shapes() {
        CartesianVector3::new(
            arr1(&[c(1.0, 0.0)]),
            arr1(&[c(2.0, 0.0), c(3.0, 0.0)]),
            arr1(&[c(4.0, 0.0)]),
        );
    }

    #[test]
    #[should_panic(expected = "Cartesian x and z components must have identical shapes")]
    fn constructor_rejects_mismatched_x_and_z_shapes() {
        CartesianVector3::new(
            arr1(&[c(1.0, 0.0)]),
            arr1(&[c(2.0, 0.0)]),
            arr1(&[c(3.0, 0.0), c(4.0, 0.0)]),
        );
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
