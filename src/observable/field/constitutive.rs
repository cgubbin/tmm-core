

/// Pointwise Cartesian electric displacement and magnetic induction phasor fields.
///
/// The field uses the electromagnetic normalization chosen by the producing
/// backend. The electric displacement and magnetic induction vectors share the same ndarray sampling
/// shape.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstitutiveFields<V> {
    electric_displacement: V,
    magnetic_induction: V,
}

impl<V> ConstitutiveFields<V> {
    pub(crate) fn new(electric_displacement: V, magnetic_induction: V) -> Self {
        Self {
            electric_displacement,
            magnetic_induction,
        }
    }

    pub fn electric_displacement(&self) -> &V {
        &self.electric_displacement
    }

    pub fn magnetic_induction(&self) -> &V {
        &self.magnetic_induction
    }

    pub fn into_parts(self) -> (V, V) {
        (self.electric_displacement, self.magnetic_induction)
    }

    pub fn map_vectors<U>(self, f: impl Fn(V) -> U) -> ConstitutiveFields<U> {
        ConstitutiveFields {
            electric_displacement: f(self.electric_displacement),
            magnetic_induction: f(self.magnetic_induction),
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Ix1, arr1};
    use num_complex::Complex64;

    use crate::field::VectorField;

    use super::*;

    type C = Complex64;
    type D = Ix1;
    type Vector = VectorField<C, D>;
    type Field = ConstitutiveFields<Vector>;

    const TOLERANCE: f64 = 1.0e-12;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn scalar_vector(value: f64) -> Vector {
        VectorField::new_unchecked(
            arr1(&[c(value, 0.0)]),
            arr1(&[c(value + 0.1, 0.0)]),
            arr1(&[c(value + 0.2, 0.0)]),
        )
    }

    fn vector(x: &[C], y: &[C], z: &[C]) -> Vector {
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

    fn assert_field_equals(
        actual: &Field,
        electric_displacement: &Vector,
        magnetic_induction: &Vector,
    ) {
        assert_vector_close(actual.electric_displacement(), electric_displacement);

        assert_vector_close(actual.magnetic_induction(), magnetic_induction);
    }

    #[test]
    fn construction_preserves_electric_and_magnetic_fields() {
        let electric_displacement = scalar_vector(1.0);
        let magnetic_induction = scalar_vector(2.0);

        let field = Field::new(electric_displacement.clone(), magnetic_induction.clone());

        assert_eq!(field.electric_displacement(), &electric_displacement,);

        assert_eq!(field.magnetic_induction(), &magnetic_induction,);

        assert_eq!(
            field.into_parts(),
            (electric_displacement, magnetic_induction),
        );
    }
}
