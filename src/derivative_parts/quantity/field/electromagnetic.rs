use crate::{
    derivative_parts::{
        BivariateFirstParts, BivariateSecondParts, DirectionalFirstParts, DirectionalSecondParts,
        IntoBivariateFirst, IntoBivariateSecond, IntoFirst, IntoSecond, IntoValue, ValuePart,
    },
    observable::ElectromagneticFields,
};

impl<J> IntoValue for ElectromagneticFields<J>
where
    J: IntoValue,
{
    type Value = ElectromagneticFields<J::Value>;

    fn into_value(self) -> ValuePart<Self::Value> {
        let (electric, magnetic) = self.into_parts();

        ValuePart::new(ElectromagneticFields::new(
            electric.into_value().into_inner(),
            magnetic.into_value().into_inner(),
        ))
    }
}

impl<J> IntoFirst for ElectromagneticFields<J>
where
    J: IntoFirst,
{
    fn into_first(self) -> DirectionalFirstParts<Self::Value> {
        let (electric, magnetic) = self.into_parts();

        let (electric_value, electric_first) = electric.into_first().into_parts();
        let (magnetic_value, magnetic_first) = magnetic.into_first().into_parts();

        DirectionalFirstParts::new(
            ElectromagneticFields::new(electric_value, magnetic_value),
            ElectromagneticFields::new(electric_first, magnetic_first),
        )
    }
}

impl<J> IntoSecond for ElectromagneticFields<J>
where
    J: IntoSecond,
{
    fn into_second(self) -> DirectionalSecondParts<Self::Value> {
        let (electric, magnetic) = self.into_parts();

        let (electric_value, electric_first, electric_second) = electric.into_second().into_parts();

        let (magnetic_value, magnetic_first, magnetic_second) = magnetic.into_second().into_parts();

        DirectionalSecondParts::new(
            ElectromagneticFields::new(electric_value, magnetic_value),
            ElectromagneticFields::new(electric_first, magnetic_first),
            ElectromagneticFields::new(electric_second, magnetic_second),
        )
    }
}

impl<J> IntoBivariateFirst for ElectromagneticFields<J>
where
    J: IntoBivariateFirst,
{
    fn into_bivariate_first(self) -> BivariateFirstParts<Self::Value> {
        let (electric, magnetic) = self.into_parts();

        let (electric_value, electric_axis0, electric_axis1) =
            electric.into_bivariate_first().into_parts();

        let (magnetic_value, magnetic_axis0, magnetic_axis1) =
            magnetic.into_bivariate_first().into_parts();

        BivariateFirstParts::new(
            ElectromagneticFields::new(electric_value, magnetic_value),
            ElectromagneticFields::new(electric_axis0, magnetic_axis0),
            ElectromagneticFields::new(electric_axis1, magnetic_axis1),
        )
    }
}

impl<J> IntoBivariateSecond for ElectromagneticFields<J>
where
    J: IntoBivariateSecond,
{
    fn into_bivariate_second(self) -> BivariateSecondParts<Self::Value> {
        let (electric, magnetic) = self.into_parts();

        let (
            electric_value,
            electric_axis0,
            electric_axis1,
            electric_axis0_axis0,
            electric_axis0_axis1,
            electric_axis1_axis1,
        ) = electric.into_bivariate_second().into_parts();

        let (
            magnetic_value,
            magnetic_axis0,
            magnetic_axis1,
            magnetic_axis0_axis0,
            magnetic_axis0_axis1,
            magnetic_axis1_axis1,
        ) = magnetic.into_bivariate_second().into_parts();

        BivariateSecondParts::new(
            ElectromagneticFields::new(electric_value, magnetic_value),
            ElectromagneticFields::new(electric_axis0, magnetic_axis0),
            ElectromagneticFields::new(electric_axis1, magnetic_axis1),
            ElectromagneticFields::new(electric_axis0_axis0, magnetic_axis0_axis0),
            ElectromagneticFields::new(electric_axis0_axis1, magnetic_axis0_axis1),
            ElectromagneticFields::new(electric_axis1_axis1, magnetic_axis1_axis1),
        )
    }
}

#[cfg(test)]
mod decomposition_tests {
    use ndarray::{Ix1, arr1};
    use num_complex::Complex64;

    use crate::{
        algebra::{Jet0, Jet1, Jet2, JetBivariate1, JetBivariate2, RealParameter},
        derivative_parts::{
            BivariateFirstParts, BivariateSecondParts, DirectionalFirstParts,
            DirectionalSecondParts, IntoBivariateFirst, IntoBivariateSecond, IntoFirst, IntoSecond,
            IntoValue,
        },
        differential::{BivariateGradient, BivariateHessian},
        field::VectorField,
    };

    use super::ElectromagneticFields;

    type C = Complex64;
    type D = Ix1;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn vector(base: f64) -> VectorField<C, D> {
        VectorField::new_unchecked(
            arr1(&[c(base + 1.0), c(base + 2.0)]),
            arr1(&[c(base + 3.0), c(base + 4.0)]),
            arr1(&[c(base + 5.0), c(base + 6.0)]),
        )
    }

    fn assert_vector_eq(actual: &VectorField<C, D>, expected_base: f64) {
        assert_eq!(
            actual.x(),
            &arr1(&[c(expected_base + 1.0), c(expected_base + 2.0),]),
        );

        assert_eq!(
            actual.y(),
            &arr1(&[c(expected_base + 3.0), c(expected_base + 4.0),]),
        );

        assert_eq!(
            actual.z(),
            &arr1(&[c(expected_base + 5.0), c(expected_base + 6.0),]),
        );
    }

    fn assert_fields_eq(
        actual: &ElectromagneticFields<VectorField<C, D>>,
        electric_base: f64,
        magnetic_base: f64,
    ) {
        assert_vector_eq(actual.electric(), electric_base);
        assert_vector_eq(actual.magnetic(), magnetic_base);
    }

    // ---------------------------------------------------------------------
    // Value
    // ---------------------------------------------------------------------

    #[test]
    fn electromagnetic_fields_into_value_extracts_both_vectors() {
        let fields = ElectromagneticFields::new(
            Jet0::<_, RealParameter>::new(vector(10.0)),
            Jet0::<_, RealParameter>::new(vector(20.0)),
        );

        let value = fields.into_value().into_inner();

        assert_fields_eq(&value, 10.0, 20.0);
    }

    // ---------------------------------------------------------------------
    // Directional first
    // ---------------------------------------------------------------------

    #[test]
    fn electromagnetic_fields_into_first_preserves_e_and_h_parts() {
        let fields = ElectromagneticFields::new(
            Jet1::<_, RealParameter>::from_parts(vector(10.0), vector(110.0)),
            Jet1::<_, RealParameter>::from_parts(vector(20.0), vector(120.0)),
        );

        let (value, first) = fields.into_first().into_parts();

        assert_fields_eq(&value, 10.0, 20.0);
        assert_fields_eq(&first, 110.0, 120.0);
    }

    // ---------------------------------------------------------------------
    // Directional second
    // ---------------------------------------------------------------------

    #[test]
    fn electromagnetic_fields_into_second_preserves_all_orders() {
        let fields = ElectromagneticFields::new(
            Jet2::<_, RealParameter>::from_parts(vector(10.0), vector(110.0), vector(210.0)),
            Jet2::<_, RealParameter>::from_parts(vector(20.0), vector(120.0), vector(220.0)),
        );

        let (value, first, second) = fields.into_second().into_parts();

        assert_fields_eq(&value, 10.0, 20.0);
        assert_fields_eq(&first, 110.0, 120.0);
        assert_fields_eq(&second, 210.0, 220.0);
    }

    // ---------------------------------------------------------------------
    // Bivariate first
    // ---------------------------------------------------------------------

    #[test]
    fn electromagnetic_fields_into_bivariate_first_preserves_axes() {
        let fields = ElectromagneticFields::new(
            JetBivariate1::<_, RealParameter>::from_parts(
                vector(10.0),
                BivariateGradient::new(vector(110.0), vector(210.0)),
            ),
            JetBivariate1::<_, RealParameter>::from_parts(
                vector(20.0),
                BivariateGradient::new(vector(120.0), vector(220.0)),
            ),
        );

        let (value, axis0, axis1) = fields.into_bivariate_first().into_parts();

        assert_fields_eq(&value, 10.0, 20.0);
        assert_fields_eq(&axis0, 110.0, 120.0);
        assert_fields_eq(&axis1, 210.0, 220.0);
    }

    // ---------------------------------------------------------------------
    // Bivariate second
    // ---------------------------------------------------------------------

    #[test]
    fn electromagnetic_fields_into_bivariate_second_preserves_all_components() {
        let fields = ElectromagneticFields::new(
            JetBivariate2::<_, RealParameter>::from_parts(
                vector(10.0),
                BivariateGradient::new(vector(110.0), vector(210.0)),
                BivariateHessian::new(vector(310.0), vector(410.0), vector(510.0)),
            ),
            JetBivariate2::<_, RealParameter>::from_parts(
                vector(20.0),
                BivariateGradient::new(vector(120.0), vector(220.0)),
                BivariateHessian::new(vector(320.0), vector(420.0), vector(520.0)),
            ),
        );

        let (value, axis0, axis1, axis0_axis0, axis0_axis1, axis1_axis1) =
            fields.into_bivariate_second().into_parts();

        assert_fields_eq(&value, 10.0, 20.0);

        assert_fields_eq(&axis0, 110.0, 120.0);
        assert_fields_eq(&axis1, 210.0, 220.0);

        assert_fields_eq(&axis0_axis0, 310.0, 320.0);

        assert_fields_eq(&axis0_axis1, 410.0, 420.0);

        assert_fields_eq(&axis1_axis1, 510.0, 520.0);
    }
}
