mod algebra;

pub(crate) use algebra::CartesianVectorAlgebra;

use ndarray::{ArrayBase, Dimension, OwnedRepr};
use num_traits::Zero;

use crate::{
    ComplexScalar,
    backend::jet::{
        Jet, JetAdditive, JetConjugate, JetCrossProduct, JetFirst, JetHermitianProduct,
        JetRealPart, JetScaleBy,
    },
};

pub(crate) type CartesianField<C, D> = CartesianElectromagneticField<CartesianVector3<C, D>>;

pub(super) type CartesianFieldFirst<C, D> =
    CartesianElectromagneticField<JetFirst<CartesianVector3<C, D>>>;

impl<C, D> CartesianFieldFirst<C, D>
where
    D: Dimension,
{
    pub(super) fn split(self) -> (CartesianField<C, D>, CartesianField<C, D>) {
        let (electric, magnetic) = self.into_parts();

        let (electric, electric_first) = electric.into_parts();
        let (magnetic, magnetic_first) = magnetic.into_parts();

        (
            CartesianField::new(electric, magnetic),
            CartesianField::new(electric_first, magnetic_first),
        )
    }
}

type CartesianFieldSecond<C, D> = CartesianElectromagneticField<Jet<CartesianVector3<C, D>>>;

impl<C, D> CartesianFieldSecond<C, D>
where
    D: Dimension,
{
    pub(super) fn split(
        self,
    ) -> (
        CartesianField<C, D>,
        CartesianField<C, D>,
        CartesianField<C, D>,
    ) {
        let (electric, magnetic) = self.into_parts();

        let (electric, electric_first, electric_second) = electric.into_parts();
        let (magnetic, magnetic_first, magnetic_second) = magnetic.into_parts();

        (
            CartesianField::new(electric, magnetic),
            CartesianField::new(electric_first, magnetic_first),
            CartesianField::new(electric_second, magnetic_second),
        )
    }
}

/// Pointwise Cartesian electric and magnetic phasor fields.
///
/// The field uses the electromagnetic normalization chosen by the producing
/// backend. The electric and magnetic vectors share the same ndarray sampling
/// shape.
///
/// The complex Poynting vector uses:
///
/// ```text
/// S = 1/2 E × H*
/// ```
///
/// and the time-averaged Poynting vector is its real part.
#[derive(Clone, Debug, PartialEq)]
pub struct CartesianElectromagneticField<V> {
    electric: V,
    magnetic: V,
}

impl<V> CartesianElectromagneticField<V> {
    pub fn new(electric: V, magnetic: V) -> Self {
        Self { electric, magnetic }
    }

    pub fn electric(&self) -> &V {
        &self.electric
    }

    pub fn magnetic(&self) -> &V {
        &self.magnetic
    }

    /// Return the pointwise squared electric-field magnitude.
    pub fn electric_magnitude_squared<C, D>(&self) -> V::RealScalarField
    where
        C: ComplexScalar,
        D: Dimension,
        V: CartesianVectorAlgebra<C, D>,
    {
        magnitude_squared(&self.electric)
    }

    /// Return the pointwise squared magnetic-field magnitude.
    pub fn magnetic_magnitude_squared<C, D>(&self) -> V::RealScalarField
    where
        C: ComplexScalar,
        D: Dimension,
        V: CartesianVectorAlgebra<C, D>,
    {
        magnitude_squared(&self.magnetic)
    }

    /// Return the pointwise complex Poynting vector.
    ///
    /// This evaluates `1/2 E × H*`.
    pub fn complex_poynting_vector<C, D>(&self) -> V
    where
        C: ComplexScalar,
        D: Dimension,
        V: CartesianVectorAlgebra<C, D>,
    {
        complex_poynting::<C, D, V>(&self.electric, &self.magnetic)
    }

    pub fn time_averaged_poynting_vector<C, D>(&self) -> V::RealVector
    where
        C: ComplexScalar,
        D: Dimension,
        V: CartesianVectorAlgebra<C, D>,
    {
        time_averaged_poynting::<C, D, V>(&self.electric, &self.magnetic)
    }

    pub fn into_parts(self) -> (V, V) {
        (self.electric, self.magnetic)
    }
}

pub(crate) fn complex_poynting<C, D, A>(electric: &A, magnetic: &A) -> A
where
    C: ComplexScalar,
    D: Dimension,
    A: CartesianVectorAlgebra<C, D>,
{
    let half = C::one() / (C::one() + C::one());

    electric.cross(&magnetic.conjugate()).scale_by(half)
}

pub(crate) fn time_averaged_poynting<C, D, A>(electric: &A, magnetic: &A) -> A::RealVector
where
    C: ComplexScalar,
    D: Dimension,
    A: CartesianVectorAlgebra<C, D>,
{
    complex_poynting::<C, D, A>(electric, magnetic).real_part()
}

pub(crate) fn magnitude_squared<C, D, A>(vector: &A) -> A::RealScalarField
where
    C: ComplexScalar,
    D: Dimension,
    A: CartesianVectorAlgebra<C, D>,
{
    A::scalar_real_part(vector.hermitian_dot(vector))
}

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
        debug_assert_eq!(x.raw_dim(), y.raw_dim());
        debug_assert_eq!(x.raw_dim(), z.raw_dim());

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

    /// Construct a zero vector with the same sampling shape as `values`.
    pub(crate) fn zeros_like(values: &ArrayBase<OwnedRepr<T>, D>) -> Self
    where
        T: Clone + Zero,
        D: Dimension,
    {
        let zero = values.mapv(|_| T::zero());

        Self::new(zero.clone(), zero.clone(), zero)
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
        T: ComplexScalar,
    {
        self.map(|value| value.conjugate())
    }

    /// Calculate the pointwise bilinear Cartesian dot product.
    ///
    /// Complex conjugation is not applied. Use [`Self::hermitian_dot`] for a conjugating inner
    /// product
    pub fn dot(&self, rhs: &Self) -> ArrayBase<OwnedRepr<T>, D>
    where
        T: ComplexScalar,
    {
        self.x.clone() * rhs.x.view()
            + self.y.clone() * rhs.y.view()
            + self.z.clone() * rhs.z.view()
    }

    /// Calculate the pointwise Hermitian Cartesian inner product.
    ///
    /// This evaluates:
    ///
    /// ```text
    /// self · conjugate(rhs)
    /// ```
    pub fn hermitian_dot(&self, rhs: &Self) -> ArrayBase<OwnedRepr<T>, D>
    where
        T: ComplexScalar,
    {
        self.dot(&rhs.conjugate())
    }

    /// Calculate the pointwise bilinear Cartesian cross product.
    ///
    /// Complex conjugation is not applied.
    pub fn cross(&self, rhs: &Self) -> Self
    where
        T: ComplexScalar,
    {
        Self::new(
            self.y.clone() * rhs.z.view() - self.z.clone() * rhs.y.view(),
            self.z.clone() * rhs.x.view() - self.x.clone() * rhs.z.view(),
            self.x.clone() * rhs.y.view() - self.y.clone() * rhs.x.view(),
        )
    }

    /// Return the pointwise squared Euclidean magnitude.
    ///
    /// For a complex vector this is:
    ///
    /// ```text
    /// |x|² + |y|² + |z|².
    /// ```
    pub fn magnitude_squared(&self) -> ArrayBase<OwnedRepr<T::RealField>, D>
    where
        T: ComplexScalar,
    {
        self.x.mapv(|value| value.modulus_squared())
            + self.y.mapv(|value| value.modulus_squared())
            + self.z.mapv(|value| value.modulus_squared())
    }
}

impl<C, D> std::ops::Add<&CartesianVector3<C, D>> for CartesianVector3<C, D>
where
    C: ComplexScalar,
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
    C: ComplexScalar,
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

impl<C, D> std::ops::Sub<&CartesianVector3<C, D>> for CartesianVector3<C, D>
where
    C: ComplexScalar,
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
    C: ComplexScalar,
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

impl<C, D> std::ops::Mul<ArrayBase<OwnedRepr<C>, D>> for CartesianVector3<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    type Output = CartesianVector3<C, D>;

    fn mul(self, factor: ArrayBase<OwnedRepr<C>, D>) -> Self::Output {
        debug_assert_eq!(self.x.raw_dim(), factor.raw_dim());
        Self::new(
            self.x * factor.view(),
            self.y * factor.view(),
            self.z * factor.view(),
        )
    }
}

impl<C, D> std::ops::Mul<&ArrayBase<OwnedRepr<C>, D>> for CartesianVector3<C, D>
where
    C: ComplexScalar,
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

impl<C, D> std::ops::Mul<C> for CartesianVector3<C, D>
where
    C: ComplexScalar,
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

impl<C, D> std::ops::Neg for CartesianVector3<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    type Output = CartesianVector3<C, D>;

    fn neg(self) -> Self::Output {
        Self::new(-self.x, -self.y, -self.z)
    }
}

impl<C, D> JetAdditive for CartesianVector3<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn jet_add(&self, rhs: &Self) -> Self {
        self.clone() + rhs
    }

    fn jet_subtract(&self, rhs: &Self) -> Self {
        self.clone() - rhs
    }

    fn jet_negate(&self) -> Self {
        -self.clone()
    }
}

impl<C, D> JetScaleBy for CartesianVector3<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    type Scalar = C;

    fn jet_scale_by(&self, value: C) -> Self {
        self.clone() * value
    }
}

impl<C, D> JetCrossProduct for CartesianVector3<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn jet_cross(&self, rhs: &Self) -> Self {
        self.cross(rhs)
    }
}

impl<C, D> JetHermitianProduct for CartesianVector3<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    type Output = ArrayBase<OwnedRepr<C>, D>;

    fn jet_hermitian_product(&self, rhs: &Self) -> Self::Output {
        self.hermitian_dot(rhs)
    }
}

impl<C, D> JetConjugate for CartesianVector3<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn jet_conjugate(&self) -> Self {
        self.conjugate()
    }
}

impl<C, D> JetRealPart for CartesianVector3<C, D>
where
    C: ComplexScalar,
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
    fn hermitian_dot_conjugates_rhs() {
        let lhs = vector(&[c(1.0, 1.0)], &[c(2.0, -1.0)], &[c(-1.0, 2.0)]);

        let rhs = vector(&[c(3.0, -2.0)], &[c(1.0, 4.0)], &[c(2.0, 1.0)]);

        let expected = c(1.0, 1.0) * c(3.0, -2.0).conj()
            + c(2.0, -1.0) * c(1.0, 4.0).conj()
            + c(-1.0, 2.0) * c(2.0, 1.0).conj();

        assert_complex_array_close(&lhs.hermitian_dot(&rhs), &arr1(&[expected]));
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
    fn electromagnetic_field_preserves_components() {
        let electric = vector(&[c(1.0, 0.0)], &[c(2.0, 0.0)], &[c(3.0, 0.0)]);

        let magnetic = vector(&[c(4.0, 0.0)], &[c(5.0, 0.0)], &[c(6.0, 0.0)]);

        let field = CartesianElectromagneticField::new(electric.clone(), magnetic.clone());

        assert_eq!(field.electric(), &electric);
        assert_eq!(field.magnetic(), &magnetic);
    }

    #[test]
    fn complex_poynting_vector_is_half_e_cross_conjugate_h() {
        let electric = vector(&[c(1.0, 2.0)], &[c(3.0, -1.0)], &[c(2.0, 4.0)]);

        let magnetic = vector(&[c(-2.0, 1.0)], &[c(1.0, 3.0)], &[c(4.0, -2.0)]);

        let expected = electric.cross(&magnetic.conjugate()) * c(0.5, 0.0);

        let field = CartesianElectromagneticField::new(electric, magnetic);

        let actual = field.complex_poynting_vector();

        assert_complex_array_close(actual.x(), expected.x());
        assert_complex_array_close(actual.y(), expected.y());
        assert_complex_array_close(actual.z(), expected.z());
    }

    #[test]
    fn time_averaged_poynting_vector_is_real_part() {
        let electric = vector(&[c(1.0, 2.0)], &[c(3.0, -1.0)], &[c(2.0, 4.0)]);

        let magnetic = vector(&[c(-2.0, 1.0)], &[c(1.0, 3.0)], &[c(4.0, -2.0)]);

        let field = CartesianElectromagneticField::new(electric, magnetic);

        let complex = field.complex_poynting_vector();
        let averaged = field.time_averaged_poynting_vector();

        assert_real_array_close(averaged.x(), &complex.x().mapv(|value| value.re));
        assert_real_array_close(averaged.y(), &complex.y().mapv(|value| value.re));
        assert_real_array_close(averaged.z(), &complex.z().mapv(|value| value.re));
    }

    #[test]
    fn electromagnetic_magnitudes_delegate_to_vectors() {
        let electric = vector(&[c(3.0, 4.0)], &[c(0.0, 2.0)], &[c(1.0, 0.0)]);

        let magnetic = vector(&[c(1.0, 0.0)], &[c(2.0, 0.0)], &[c(0.0, 3.0)]);

        let expected_electric = electric.magnitude_squared();
        let expected_magnetic = magnetic.magnitude_squared();

        let field = CartesianElectromagneticField::new(electric, magnetic);

        assert_real_array_close(&field.electric_magnitude_squared(), &expected_electric);

        assert_real_array_close(&field.magnetic_magnitude_squared(), &expected_magnetic);
    }

    #[test]
    fn electromagnetic_field_into_parts_preserves_vectors() {
        let electric = vector(&[c(1.0, 0.0)], &[c(2.0, 0.0)], &[c(3.0, 0.0)]);

        let magnetic = vector(&[c(4.0, 0.0)], &[c(5.0, 0.0)], &[c(6.0, 0.0)]);

        let field = CartesianElectromagneticField::new(electric.clone(), magnetic.clone());

        let (actual_electric, actual_magnetic) = field.into_parts();

        assert_eq!(actual_electric, electric);
        assert_eq!(actual_magnetic, magnetic);
    }
}
