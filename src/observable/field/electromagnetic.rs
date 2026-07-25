use ndarray::{Axis, Dimension, Ix0, Ix1, arr0};
use num_traits::One;
use std::marker::PhantomData;

use super::FieldIndexError;
use crate::algebra::RealCartesianVectorAlgebra;
use crate::field::{VectorField, VectorFieldView1};
use crate::{SpatialProfile, SpatialProfileError};

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
pub struct ElectromagneticFields<V> {
    electric: V,
    magnetic: V,
}

impl<V> ElectromagneticFields<V> {
    pub(crate) fn new(electric: V, magnetic: V) -> Self {
        Self { electric, magnetic }
    }

    pub fn electric(&self) -> &V {
        &self.electric
    }

    pub fn magnetic(&self) -> &V {
        &self.magnetic
    }

    pub fn into_parts(self) -> (V, V) {
        (self.electric, self.magnetic)
    }

    pub fn map_vectors<U>(self, f: impl Fn(V) -> U) -> ElectromagneticFields<U> {
        ElectromagneticFields {
            electric: f(self.electric),
            magnetic: f(self.magnetic),
        }
    }
}

impl<C, D> SpatialProfile<D::Smaller> for ElectromagneticFields<VectorField<C, D>>
where
    D: Dimension,
    D::Smaller: Dimension<Larger = D>,
{
    type Profile<'a>
        = ElectromagneticFields<VectorFieldView1<'a, C>>
    where
        Self: 'a;

    fn spatial_profile(
        &self,
        excitation_index: &D::Smaller,
    ) -> Result<Self::Profile<'_>, SpatialProfileError> {
        Ok(ElectromagneticFields {
            electric: self.electric.profile_last_axis(excitation_index)?,
            magnetic: self.magnetic.profile_last_axis(excitation_index)?,
        })
    }
}

impl<V> ElectromagneticFields<V> {
    /// Return the pointwise squared electric-field magnitude.
    pub fn electric_magnitude_squared(&self) -> V::RealScalarField
    where
        V: RealCartesianVectorAlgebra,
    {
        self.electric.magnitude_squared()
    }

    /// Return the pointwise squared magnetic-field magnitude.
    pub fn magnetic_magnitude_squared(&self) -> V::RealScalarField
    where
        V: RealCartesianVectorAlgebra,
    {
        self.magnetic.magnitude_squared()
    }

    /// Return the pointwise complex Poynting vector.
    ///
    /// This evaluates `1/2 E × H*`.
    ///
    /// Fields use the exp(-iωt) phasor convention.
    pub fn complex_poynting_vector(&self) -> V
    where
        V: RealCartesianVectorAlgebra,
        V::Coefficient: One
            + Copy
            + std::ops::Add<Output = V::Coefficient>
            + std::ops::Div<Output = V::Coefficient>,
    {
        self.electric
            .cross(&self.magnetic.conjugated())
            .scale_by_constant(half())
    }

    /// Return the time average poynting vector
    ///
    /// This evaluates `1/2 Re(E × H*)`.
    ///
    /// Fields use the exp(-iωt) phasor convention.
    pub fn time_averaged_poynting_vector(&self) -> V::RealVector
    where
        V: RealCartesianVectorAlgebra,
        V::Coefficient: One
            + Copy
            + std::ops::Add<Output = V::Coefficient>
            + std::ops::Div<Output = V::Coefficient>,
    {
        self.complex_poynting_vector().real()
    }
}

fn quarter<C>() -> C
where
    C: One + Copy + std::ops::Add<Output = C> + std::ops::Div<Output = C>,
{
    let one = C::one();
    let two = one + one;

    one / (two + two)
}

fn half<C>() -> C
where
    C: One + Copy + std::ops::Add<Output = C> + std::ops::Div<Output = C>,
{
    let one = C::one();

    one / (one + one)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Array3, Ix1, Ix2, arr1};
    use num_complex::Complex64;

    use crate::{
        algebra::{Jet0, RealParameter},
        field::VectorField,
        spatial::SpatialProfile,
    };

    use super::*;

    type C = Complex64;
    type D = Ix1;
    type Vector = Jet0<VectorField<C, D>, RealParameter>;
    type Field = ElectromagneticFields<Vector>;

    const TOLERANCE: f64 = 1.0e-12;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn scalar_vector(value: f64) -> Vector {
        Jet0::new(VectorField::new_unchecked(
            arr1(&[c(value, 0.0)]),
            arr1(&[c(value + 0.1, 0.0)]),
            arr1(&[c(value + 0.2, 0.0)]),
        ))
    }

    fn vector(x: &[C], y: &[C], z: &[C]) -> Vector {
        Jet0::new(VectorField::new_unchecked(
            Array1::from_vec(x.to_vec()),
            Array1::from_vec(y.to_vec()),
            Array1::from_vec(z.to_vec()),
        ))
    }

    fn assert_real_close(actual: f64, expected: f64) {
        let error = (actual - expected).abs();

        assert!(
            error <= TOLERANCE,
            "expected {expected:e}, \
             got {actual:e}; \
             absolute error = {error:e}",
        );
    }

    fn assert_complex_close(actual: C, expected: C) {
        let error = (actual - expected).norm();

        assert!(
            error <= TOLERANCE,
            "expected {expected:?}, \
             got {actual:?}; \
             absolute error = {error:e}",
        );
    }

    fn assert_real_array_close(actual: &Array1<f64>, expected: &Array1<f64>) {
        assert_eq!(actual.raw_dim(), expected.raw_dim(),);

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_real_close(actual, expected);
        }
    }

    fn assert_complex_array_close(actual: &Array1<C>, expected: &Array1<C>) {
        assert_eq!(actual.raw_dim(), expected.raw_dim(),);

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_complex_close(actual, expected);
        }
    }

    fn assert_vector_close(actual: &Vector, expected: &Vector) {
        assert_complex_array_close(actual.x(), expected.x());

        assert_complex_array_close(actual.y(), expected.y());

        assert_complex_array_close(actual.z(), expected.z());
    }

    fn assert_field_equals(actual: &Field, electric: &Vector, magnetic: &Vector) {
        assert_vector_close(actual.electric(), electric);

        assert_vector_close(actual.magnetic(), magnetic);
    }

    #[test]
    fn construction_preserves_electric_and_magnetic_fields() {
        let electric = scalar_vector(1.0);
        let magnetic = scalar_vector(2.0);

        let field = Field::new(electric.clone(), magnetic.clone());

        assert_eq!(field.electric(), &electric,);

        assert_eq!(field.magnetic(), &magnetic,);

        assert_eq!(field.into_parts(), (electric, magnetic),);
    }

    #[test]
    fn magnitude_methods_use_corresponding_vector() {
        let electric = vector(
            &[c(3.0, 4.0), c(1.0, 0.0)],
            &[c(0.0, 2.0), c(0.0, 2.0)],
            &[c(1.0, 0.0), c(2.0, 0.0)],
        );

        let magnetic = vector(
            &[c(1.0, 0.0), c(2.0, 0.0)],
            &[c(2.0, 0.0), c(3.0, 0.0)],
            &[c(3.0, 0.0), c(4.0, 0.0)],
        );

        let field = Field::new(electric, magnetic);

        assert_real_array_close(&field.electric_magnitude_squared(), &arr1(&[30.0, 9.0]));

        assert_real_array_close(&field.magnetic_magnitude_squared(), &arr1(&[14.0, 29.0]));
    }

    #[test]
    fn complex_poynting_is_half_e_cross_conjugate_h() {
        let zero = c(0.0, 0.0);

        let electric = vector(&[c(1.0, 1.0)], &[zero], &[zero]);

        let magnetic = vector(&[zero], &[c(2.0, 1.0)], &[zero]);

        let field = Field::new(electric, magnetic);

        let result = field.complex_poynting_vector();

        let expected_z = c(0.5, 0.0) * c(1.0, 1.0) * c(2.0, 1.0).conj();

        assert_complex_array_close(result.x(), &arr1(&[zero]));

        assert_complex_array_close(result.y(), &arr1(&[zero]));

        assert_complex_array_close(result.z(), &arr1(&[expected_z]));
    }

    #[test]
    fn time_averaged_poynting_is_real_part_of_complex_poynting() {
        let zero = c(0.0, 0.0);

        let field = Field::new(
            vector(&[c(1.0, 1.0)], &[zero], &[zero]),
            vector(&[zero], &[c(2.0, 1.0)], &[zero]),
        );

        let complex = field.complex_poynting_vector();

        let averaged = field.time_averaged_poynting_vector();

        assert_real_array_close(averaged.x(), &complex.x().mapv(|value| value.re));

        assert_real_array_close(averaged.y(), &complex.y().mapv(|value| value.re));

        assert_real_array_close(averaged.z(), &complex.z().mapv(|value| value.re));
    }

    #[test]
    fn electromagnetic_fields_profile_electric_and_magnetic_fields() {
        let base = Array3::from_shape_fn((2, 2, 3), |(i, j, k)| {
            100.0 * i as f64 + 10.0 * j as f64 + k as f64
        });

        let electric = VectorField::new(
            base.clone(),
            base.mapv(|value| value + 1000.0),
            base.mapv(|value| value + 2000.0),
        )
        .unwrap();

        let magnetic = VectorField::new(
            base.mapv(|value| value + 3000.0),
            base.mapv(|value| value + 4000.0),
            base.mapv(|value| value + 5000.0),
        )
        .unwrap();

        let fields: ElectromagneticFields<VectorField<f64, ndarray::Ix3>> =
            ElectromagneticFields::new(electric, magnetic);

        let profile = fields
            .spatial_profile(&Ix2(1, 1))
            .expect("profile should succeed");

        assert_eq!(profile.electric().x(), arr1(&[110.0, 111.0, 112.0]).view(),);
        assert_eq!(
            profile.electric().y(),
            arr1(&[1110.0, 1111.0, 1112.0]).view(),
        );
        assert_eq!(
            profile.magnetic().z(),
            arr1(&[5110.0, 5111.0, 5112.0]).view(),
        );
    }
}
