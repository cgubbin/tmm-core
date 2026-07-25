// use crate::{
//     algebra::{Jet, JetBivariate, JetFirst},
//     field::VectorField,
//     observable::ElectromagneticFields,
// };

// use ndarray::Dimension;

// pub(crate) type ElectromagneticFieldValues<C, D> = ElectromagneticFields<VectorField<C, D>>;
// pub(crate) type ElectromagneticFieldFirst<C, D, P> =
//     ElectromagneticFields<JetFirst<VectorField<C, D>, P>>;
// pub(crate) type ElectromagneticFieldSecond<C, D, P> =
//     ElectromagneticFields<Jet<VectorField<C, D>, P>>;
// pub(crate) type ElectromagneticFieldBivariate<C, D, P> =
//     ElectromagneticFields<JetBivariate<VectorField<C, D>, P>>;

// impl<C, D, P> ElectromagneticFieldFirst<C, D, P>
// where
//     D: Dimension,
// {
//     pub(crate) fn split(
//         self,
//     ) -> (
//         ElectromagneticFieldValues<C, D>,
//         ElectromagneticFieldValues<C, D>,
//     ) {
//         let (electric, magnetic) = self.into_parts();

//         let (electric_value, electric_first) = electric.into_parts();
//         let (magnetic_value, magnetic_first) = magnetic.into_parts();

//         (
//             ElectromagneticFields::new(electric_value, magnetic_value),
//             ElectromagneticFields::new(electric_first, magnetic_first),
//         )
//     }
// }

// impl<C, D, P> ElectromagneticFieldSecond<C, D, P>
// where
//     D: Dimension,
// {
//     pub(crate) fn split(
//         self,
//     ) -> (
//         ElectromagneticFieldValues<C, D>,
//         ElectromagneticFieldValues<C, D>,
//         ElectromagneticFieldValues<C, D>,
//     ) {
//         let (electric, magnetic) = self.into_parts();

//         let (electric_value, electric_first, electric_second) = electric.into_parts();
//         let (magnetic_value, magnetic_first, magnetic_second) = magnetic.into_parts();

//         (
//             ElectromagneticFields::new(electric_value, magnetic_value),
//             ElectromagneticFields::new(electric_first, magnetic_first),
//             ElectromagneticFields::new(electric_second, magnetic_second),
//         )
//     }
// }

// #[derive(Clone, Debug, PartialEq)]
// pub(crate) struct ElectromagneticFieldBivariateParts<F> {
//     pub(crate) value: F,
//     pub(crate) x: F,
//     pub(crate) y: F,
//     pub(crate) xx: F,
//     pub(crate) xy: F,
//     pub(crate) yy: F,
// }

// impl<C, D, P> ElectromagneticFieldBivariate<C, D, P>
// where
//     D: Dimension,
// {
//     pub(crate) fn split(
//         self,
//     ) -> ElectromagneticFieldBivariateParts<ElectromagneticFieldValues<C, D>> {
//         let (electric, magnetic) = self.into_parts();

//         let (electric_value, electric_gradient, electric_hessian) = electric.into_parts();
//         let (magnetic_value, magnetic_gradient, magnetic_hessian) = magnetic.into_parts();

//         let (electric_x, electric_y) = electric_gradient.into_parts();
//         let (magnetic_x, magnetic_y) = magnetic_gradient.into_parts();
//         let (electric_xx, electric_xy, electric_yy) = electric_hessian.into_parts();
//         let (magnetic_xx, magnetic_xy, magnetic_yy) = magnetic_hessian.into_parts();
//         ElectromagneticFieldBivariateParts {
//             value: ElectromagneticFields::new(electric_value, magnetic_value),
//             x: ElectromagneticFields::new(electric_x, magnetic_x),
//             y: ElectromagneticFields::new(electric_y, magnetic_y),
//             xx: ElectromagneticFields::new(electric_xx, magnetic_xx),
//             xy: ElectromagneticFields::new(electric_xy, magnetic_xy),
//             yy: ElectromagneticFields::new(electric_yy, magnetic_yy),
//         }
//     }
// }

// #[cfg(test)]
// mod tests {

//     use ndarray::{Array1, Ix1, arr1};
//     use num_complex::Complex64;

//     use crate::algebra::{Jet, JetBivariate, JetFirst, RealParameter};
//     use crate::field::VectorField;

//     use super::*;

//     type C = Complex64;
//     type D = Ix1;
//     type Vector = VectorField<C, D>;
//     type Field = ElectromagneticFields<Vector>;

//     type FirstVector = JetFirst<Vector, RealParameter>;
//     type FirstField = ElectromagneticFields<FirstVector>;

//     type SecondVector = Jet<Vector, RealParameter>;
//     type SecondField = ElectromagneticFields<SecondVector>;

//     type BivariateVector = JetBivariate<Vector, RealParameter>;
//     type BivariateField = ElectromagneticFields<BivariateVector>;

//     const TOLERANCE: f64 = 1.0e-12;

//     fn c(real: f64, imaginary: f64) -> C {
//         C::new(real, imaginary)
//     }

//     fn scalar_vector(value: f64) -> Vector {
//         VectorField::new_unchecked(
//             arr1(&[c(value, 0.0)]),
//             arr1(&[c(value + 0.1, 0.0)]),
//             arr1(&[c(value + 0.2, 0.0)]),
//         )
//     }

//     fn assert_complex_close(actual: C, expected: C) {
//         let error = (actual - expected).norm();

//         assert!(
//             error <= TOLERANCE,
//             "expected {expected:?}, \
//              got {actual:?}; \
//              absolute error = {error:e}",
//         );
//     }

//     fn assert_complex_array_close(actual: &Array1<C>, expected: &Array1<C>) {
//         assert_eq!(actual.raw_dim(), expected.raw_dim(),);

//         for (&actual, &expected) in actual.iter().zip(expected.iter()) {
//             assert_complex_close(actual, expected);
//         }
//     }

//     fn assert_vector_close(actual: &Vector, expected: &Vector) {
//         assert_complex_array_close(actual.x(), expected.x());

//         assert_complex_array_close(actual.y(), expected.y());

//         assert_complex_array_close(actual.z(), expected.z());
//     }

//     fn assert_field_equals(actual: &Field, electric: &Vector, magnetic: &Vector) {
//         assert_vector_close(actual.electric(), electric);

//         assert_vector_close(actual.magnetic(), magnetic);
//     }

//     #[test]
//     fn first_order_split_transposes_field_and_jet_layers() {
//         let electric_value = scalar_vector(1.0);
//         let electric_first = scalar_vector(2.0);
//         let magnetic_value = scalar_vector(3.0);
//         let magnetic_first = scalar_vector(4.0);

//         let field: FirstField = ElectromagneticFields::new(
//             JetFirst::from_parts(electric_value.clone(), electric_first.clone()),
//             JetFirst::from_parts(magnetic_value.clone(), magnetic_first.clone()),
//         );

//         let (value, first) = field.split();

//         assert_field_equals(&value, &electric_value, &magnetic_value);

//         assert_field_equals(&first, &electric_first, &magnetic_first);
//     }

//     #[test]
//     fn second_order_split_transposes_field_and_jet_layers() {
//         let electric_value = scalar_vector(1.0);
//         let electric_first = scalar_vector(2.0);
//         let electric_second = scalar_vector(3.0);

//         let magnetic_value = scalar_vector(4.0);
//         let magnetic_first = scalar_vector(5.0);
//         let magnetic_second = scalar_vector(6.0);

//         let field: SecondField = ElectromagneticFields::new(
//             Jet::from_parts(
//                 electric_value.clone(),
//                 electric_first.clone(),
//                 electric_second.clone(),
//             ),
//             Jet::from_parts(
//                 magnetic_value.clone(),
//                 magnetic_first.clone(),
//                 magnetic_second.clone(),
//             ),
//         );

//         let (value, first, second) = field.split();

//         assert_field_equals(&value, &electric_value, &magnetic_value);

//         assert_field_equals(&first, &electric_first, &magnetic_first);

//         assert_field_equals(&second, &electric_second, &magnetic_second);
//     }

//     #[test]
//     fn bivariate_split_transposes_all_field_and_jet_components() {
//         let electric_value = scalar_vector(1.0);
//         let electric_x = scalar_vector(2.0);
//         let electric_y = scalar_vector(3.0);
//         let electric_xx = scalar_vector(4.0);
//         let electric_xy = scalar_vector(5.0);
//         let electric_yy = scalar_vector(6.0);

//         let magnetic_value = scalar_vector(7.0);
//         let magnetic_x = scalar_vector(8.0);
//         let magnetic_y = scalar_vector(9.0);
//         let magnetic_xx = scalar_vector(10.0);
//         let magnetic_xy = scalar_vector(11.0);
//         let magnetic_yy = scalar_vector(12.0);

//         let field: BivariateField = ElectromagneticFields::new(
//             JetBivariate::from_components(
//                 electric_value.clone(),
//                 electric_x.clone(),
//                 electric_y.clone(),
//                 electric_xx.clone(),
//                 electric_xy.clone(),
//                 electric_yy.clone(),
//             ),
//             JetBivariate::from_components(
//                 magnetic_value.clone(),
//                 magnetic_x.clone(),
//                 magnetic_y.clone(),
//                 magnetic_xx.clone(),
//                 magnetic_xy.clone(),
//                 magnetic_yy.clone(),
//             ),
//         );

//         let parts = field.split();

//         assert_field_equals(&parts.value, &electric_value, &magnetic_value);

//         assert_field_equals(&parts.x, &electric_x, &magnetic_x);

//         assert_field_equals(&parts.y, &electric_y, &magnetic_y);

//         assert_field_equals(&parts.xx, &electric_xx, &magnetic_xx);

//         assert_field_equals(&parts.xy, &electric_xy, &magnetic_xy);

//         assert_field_equals(&parts.yy, &electric_yy, &magnetic_yy);
//     }
// }
