use nalgebra::ComplexField;
use ndarray::{Array, ArrayBase, Dimension, OwnedRepr};
use std::fmt::Debug;

use super::{
    ArrayJet, ArrayJetBivariate, ArrayJetFirst, FirstOrderExpansion, RealParameter,
    SecondOrderExpansion,
};

pub(crate) trait ScalarAlgebra<T, D>: Clone + Sized + std::fmt::Debug
where
    D: Dimension,
{
    // type Vector: CartesianVectorAlgebra<Coefficient = T, ScalarField = Self>;

    fn value(&self) -> &ArrayBase<OwnedRepr<T>, D>;

    fn lift_constant(value: ArrayBase<OwnedRepr<T>, D>) -> Self;

    fn filled_constant_like(source: &ArrayBase<OwnedRepr<T>, D>, value: T) -> Self;

    // fn into_cartesian_vector(x: Self, y: Self, z: Self) -> Self::Vector;

    fn zero_like(&self) -> Self;

    fn constant(&self, value: T) -> Self {
        Self::filled_constant_like(self.value(), value)
    }

    fn add(&self, rhs: &Self) -> Self;
    fn subtract(&self, rhs: &Self) -> Self;
    fn negate(&self) -> Self;

    fn multiply(&self, rhs: &Self) -> Self;
    fn reciprocal(&self) -> Self;
    fn scale(&self, coefficient: T) -> Self;

    fn exp(&self) -> Self;
    fn sin(&self) -> Self;
    fn cos(&self) -> Self;
    fn sqrt(&self) -> Self;

    fn all_finite(&self) -> bool;

    fn square(&self) -> Self {
        self.multiply(self)
    }

    fn divide(&self, rhs: &Self) -> Self {
        self.multiply(&rhs.reciprocal())
    }
}

/// Operations that are valid when the active differentiation coordinates are
/// real.
///
/// This trait is deliberately not implemented for jets parameterised by
/// [`super::HolomorphicParameter`].
pub(crate) trait RealScalarAlgebra<T, D>: ScalarAlgebra<T, D>
where
    D: Dimension,
{
    type RealField;

    fn conjugated(&self) -> Self;

    fn real(&self) -> Self::RealField;

    fn magnitude_squared(&self) -> Self::RealField {
        self.multiply(&self.conjugated()).real()
    }
}

// -------------------------------------------------------------------------
// Variable-seeding capabilities
// -------------------------------------------------------------------------

pub(crate) trait UnivariateVariableAlgebra<T, D>: ScalarAlgebra<T, D>
where
    D: Dimension,
{
    fn variable(value: ArrayBase<OwnedRepr<T>, D>) -> Self;
}

pub(crate) trait BivariateVariableAlgebra<T, D>: ScalarAlgebra<T, D>
where
    D: Dimension,
{
    fn variable_x(value: ArrayBase<OwnedRepr<T>, D>) -> Self;

    fn variable_y(value: ArrayBase<OwnedRepr<T>, D>) -> Self;
}

// -------------------------------------------------------------------------
// Sampled unary-function composition
// -------------------------------------------------------------------------

pub(crate) trait FirstOrderFunctionAlgebra<C, D>: ScalarAlgebra<C, D>
where
    D: Dimension,
{
    fn compose_sampled_function(
        argument: &Self,
        expansion: FirstOrderExpansion<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self;
}

pub(crate) trait SecondOrderFunctionAlgebra<C, D>: ScalarAlgebra<C, D>
where
    D: Dimension,
{
    fn compose_sampled_function(
        argument: &Self,
        expansion: SecondOrderExpansion<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self;
}

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------

fn complex_is_finite<C>(value: &C) -> bool
where
    C: ComplexField + Copy,
{
    value.real().is_finite() && value.imaginary().is_finite()
}

fn array_is_finite<C, D>(value: &ArrayBase<OwnedRepr<C>, D>) -> bool
where
    C: ComplexField + Copy,
    D: Dimension,
{
    value.iter().all(complex_is_finite)
}

// -------------------------------------------------------------------------
// Plain sampled arrays
// -------------------------------------------------------------------------

impl<C, D> ScalarAlgebra<C, D> for ArrayBase<OwnedRepr<C>, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn value(&self) -> &Self {
        self
    }

    fn lift_constant(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        value
    }

    fn filled_constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        Array::from_elem(source.raw_dim(), value)
    }

    fn zero_like(&self) -> Self {
        self.mapv(|_| C::zero())
    }

    fn add(&self, rhs: &Self) -> Self {
        self.clone() + rhs.view()
    }

    fn subtract(&self, rhs: &Self) -> Self {
        self.clone() - rhs.view()
    }

    fn negate(&self) -> Self {
        -self.clone()
    }

    fn multiply(&self, rhs: &Self) -> Self {
        self.clone() * rhs.view()
    }

    fn reciprocal(&self) -> Self {
        self.mapv(|value| C::one() / value)
    }

    fn scale(&self, coefficient: C) -> Self {
        self.mapv(|value| value * coefficient)
    }

    fn exp(&self) -> Self {
        self.mapv(ComplexField::exp)
    }

    fn sin(&self) -> Self {
        self.mapv(ComplexField::sin)
    }

    fn cos(&self) -> Self {
        self.mapv(ComplexField::cos)
    }

    fn sqrt(&self) -> Self {
        self.mapv(ComplexField::sqrt)
    }

    fn all_finite(&self) -> bool {
        array_is_finite(self)
    }
}

impl<C, D> RealScalarAlgebra<C, D> for ArrayBase<OwnedRepr<C>, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type RealField = ArrayBase<OwnedRepr<C::RealField>, D>;

    fn conjugated(&self) -> Self {
        self.mapv(ComplexField::conjugate)
    }

    fn real(&self) -> Self::RealField {
        self.mapv(|value| value.real())
    }

    fn magnitude_squared(&self) -> Self::RealField {
        self.mapv(|value| value.modulus_squared())
    }
}

// -------------------------------------------------------------------------
// First-order univariate jets
// -------------------------------------------------------------------------

impl<C, D, P> ScalarAlgebra<C, D> for ArrayJetFirst<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArrayJetFirst::value(self)
    }

    fn lift_constant(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJetFirst::constant(value)
    }

    fn filled_constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        ArrayJetFirst::constant_like(source, value)
    }

    fn zero_like(&self) -> Self {
        ArrayJetFirst::constant_like(self.value(), C::zero())
    }

    fn add(&self, rhs: &Self) -> Self {
        ArrayJetFirst::add(self, rhs)
    }

    fn subtract(&self, rhs: &Self) -> Self {
        ArrayJetFirst::subtract(self, rhs)
    }

    fn negate(&self) -> Self {
        ArrayJetFirst::negate(self)
    }

    fn multiply(&self, rhs: &Self) -> Self {
        ArrayJetFirst::multiply(self, rhs)
    }

    fn reciprocal(&self) -> Self {
        ArrayJetFirst::reciprocal(self)
    }

    fn scale(&self, coefficient: C) -> Self {
        ArrayJetFirst::scale_by(self, coefficient)
    }

    fn exp(&self) -> Self {
        self.clone().exp()
    }

    fn sin(&self) -> Self {
        self.clone().sin()
    }

    fn cos(&self) -> Self {
        self.clone().cos()
    }

    fn sqrt(&self) -> Self {
        self.clone().sqrt()
    }

    fn all_finite(&self) -> bool {
        array_is_finite(self.value()) && array_is_finite(self.first())
    }
}

impl<C, D> RealScalarAlgebra<C, D> for ArrayJetFirst<C, D, RealParameter>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type RealField = ArrayJetFirst<C::RealField, D, RealParameter>;

    fn conjugated(&self) -> Self {
        ArrayJetFirst::conjugated(self)
    }

    fn real(&self) -> Self::RealField {
        ArrayJetFirst::real(self)
    }
}

impl<C, D, P> UnivariateVariableAlgebra<C, D> for ArrayJetFirst<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn variable(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJetFirst::variable(value)
    }
}

impl<C, D, P> FirstOrderFunctionAlgebra<C, D> for ArrayJetFirst<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn compose_sampled_function(
        argument: &Self,
        expansion: FirstOrderExpansion<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self {
        ArrayJetFirst::compose_sampled_function(argument, expansion)
    }
}

// -------------------------------------------------------------------------
// Second-order univariate jets
// -------------------------------------------------------------------------

impl<C, D, P> ScalarAlgebra<C, D> for ArrayJet<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArrayJet::value(self)
    }

    fn lift_constant(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJet::constant(value)
    }

    fn filled_constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        ArrayJet::constant_like(source, value)
    }

    fn zero_like(&self) -> Self {
        ArrayJet::constant_like(self.value(), C::zero())
    }

    fn add(&self, rhs: &Self) -> Self {
        ArrayJet::add(self, rhs)
    }

    fn subtract(&self, rhs: &Self) -> Self {
        ArrayJet::subtract(self, rhs)
    }

    fn negate(&self) -> Self {
        ArrayJet::negate(self)
    }

    fn multiply(&self, rhs: &Self) -> Self {
        ArrayJet::multiply(self, rhs)
    }

    fn reciprocal(&self) -> Self {
        ArrayJet::reciprocal(self)
    }

    fn scale(&self, coefficient: C) -> Self {
        ArrayJet::scale_by(self, coefficient)
    }

    fn exp(&self) -> Self {
        self.clone().exp()
    }

    fn sin(&self) -> Self {
        self.clone().sin()
    }

    fn cos(&self) -> Self {
        self.clone().cos()
    }

    fn sqrt(&self) -> Self {
        self.clone().sqrt()
    }

    fn all_finite(&self) -> bool {
        array_is_finite(self.value())
            && array_is_finite(self.first())
            && array_is_finite(self.second())
    }
}

impl<C, D> RealScalarAlgebra<C, D> for ArrayJet<C, D, RealParameter>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type RealField = ArrayJet<C::RealField, D, RealParameter>;

    fn conjugated(&self) -> Self {
        ArrayJet::conjugated(self)
    }

    fn real(&self) -> Self::RealField {
        ArrayJet::real(self)
    }
}

impl<C, D, P> UnivariateVariableAlgebra<C, D> for ArrayJet<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn variable(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJet::variable(value)
    }
}

impl<C, D, P> SecondOrderFunctionAlgebra<C, D> for ArrayJet<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn compose_sampled_function(
        argument: &Self,
        expansion: SecondOrderExpansion<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self {
        ArrayJet::compose_sampled_function(argument, expansion)
    }
}

// -------------------------------------------------------------------------
// Second-order bivariate jets
// -------------------------------------------------------------------------

impl<C, D, P> ScalarAlgebra<C, D> for ArrayJetBivariate<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArrayJetBivariate::value(self)
    }

    fn lift_constant(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJetBivariate::constant(value)
    }

    fn filled_constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        ArrayJetBivariate::constant_like(source, value)
    }

    fn zero_like(&self) -> Self {
        ArrayJetBivariate::constant_like(self.value(), C::zero())
    }

    fn add(&self, rhs: &Self) -> Self {
        ArrayJetBivariate::add(self, rhs)
    }

    fn subtract(&self, rhs: &Self) -> Self {
        ArrayJetBivariate::subtract(self, rhs)
    }

    fn negate(&self) -> Self {
        ArrayJetBivariate::negate(self)
    }

    fn multiply(&self, rhs: &Self) -> Self {
        ArrayJetBivariate::multiply(self, rhs)
    }

    fn reciprocal(&self) -> Self {
        ArrayJetBivariate::reciprocal(self)
    }

    fn scale(&self, coefficient: C) -> Self {
        ArrayJetBivariate::scale_by(self, coefficient)
    }

    fn exp(&self) -> Self {
        self.clone().exp()
    }

    fn sin(&self) -> Self {
        self.clone().sin()
    }

    fn cos(&self) -> Self {
        self.clone().cos()
    }

    fn sqrt(&self) -> Self {
        self.clone().sqrt()
    }

    fn all_finite(&self) -> bool {
        array_is_finite(self.value())
            && array_is_finite(self.x())
            && array_is_finite(self.y())
            && array_is_finite(self.xx())
            && array_is_finite(self.xy())
            && array_is_finite(self.yy())
    }
}

impl<C, D> RealScalarAlgebra<C, D> for ArrayJetBivariate<C, D, RealParameter>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type RealField = ArrayJetBivariate<C::RealField, D, RealParameter>;

    fn conjugated(&self) -> Self {
        ArrayJetBivariate::conjugated(self)
    }

    fn real(&self) -> Self::RealField {
        ArrayJetBivariate::real(self)
    }
}

impl<C, D, P> BivariateVariableAlgebra<C, D> for ArrayJetBivariate<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn variable_x(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJetBivariate::variable_x(value)
    }

    fn variable_y(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJetBivariate::variable_y(value)
    }
}

impl<C, D, P> SecondOrderFunctionAlgebra<C, D> for ArrayJetBivariate<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn compose_sampled_function(
        argument: &Self,
        expansion: SecondOrderExpansion<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self {
        ArrayJetBivariate::compose_sampled_function(argument, expansion)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use ndarray::{Array1, Ix1, array};
    use num_complex::Complex64;

    use crate::algebra::{
        ArrayJet, ArrayJetBivariate, ArrayJetFirst, Jet, JetBivariate, JetFirst, RealParameter,
    };

    type C = Complex64;
    type D = Ix1;
    type Array = Array1<C>;
    type RealArray = Array1<f64>;

    type First = ArrayJetFirst<C, D, RealParameter>;

    type Second = ArrayJet<C, D, RealParameter>;

    type Bivariate = ArrayJetBivariate<C, D, RealParameter>;

    const EPSILON: f64 = 1.0e-11;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn assert_complex_close(actual: C, expected: C) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = EPSILON,
            max_relative = EPSILON,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = EPSILON,
            max_relative = EPSILON,
        );
    }

    fn assert_complex_array_close(actual: &Array, expected: &Array) {
        assert_eq!(actual.raw_dim(), expected.raw_dim(),);

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_complex_close(*actual, *expected);
        }
    }

    fn assert_real_array_close(actual: &RealArray, expected: &RealArray) {
        assert_eq!(actual.raw_dim(), expected.raw_dim(),);

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(actual, expected, epsilon = EPSILON, max_relative = EPSILON,);
        }
    }

    fn values() -> Array {
        array![c(1.0, 0.5), c(2.0, -0.25), c(-0.5, 1.5),]
    }

    fn other_values() -> Array {
        array![c(0.5, -1.0), c(-0.75, 0.5), c(2.0, 0.25),]
    }

    fn first_derivative() -> Array {
        array![c(0.2, 0.3), c(-0.4, 0.1), c(0.7, -0.2),]
    }

    fn other_first_derivative() -> Array {
        array![c(-0.1, 0.4), c(0.6, -0.3), c(-0.2, 0.8),]
    }

    fn second_derivative() -> Array {
        array![c(0.8, -0.1), c(-0.2, 0.5), c(0.3, 0.9),]
    }

    fn other_second_derivative() -> Array {
        array![c(-0.3, 0.7), c(0.4, 0.2), c(-0.6, -0.1),]
    }

    // ---------------------------------------------------------------------
    // Plain arrays
    // ---------------------------------------------------------------------

    #[test]
    fn array_scalar_algebra_lifts_values_unchanged() {
        let source = values();

        let result = <Array as ScalarAlgebra<C, D>>::lift_constant(source.clone());

        assert_eq!(result, source);
    }
}
//     #[test]
//     fn array_scalar_algebra_constructs_constants() {
//         let source = values();
//         let constant = c(3.0, -2.0);

//         let result = <Array as ScalarAlgebra<C, D>>::filled_constant_like(&source, constant);

//         assert_eq!(result, Array::from_elem(source.raw_dim(), constant,),);
//     }

//     #[test]
//     fn array_scalar_algebra_zero_like_preserves_shape() {
//         let source = values();

//         let result = <Array as ScalarAlgebra<C, D>>::zero_like(&source);

//         assert_eq!(result.raw_dim(), source.raw_dim(),);

//         assert!(result.iter().all(|value| *value == C::default(),),);
//     }

//     #[test]
//     fn array_scalar_algebra_delegates_arithmetic() {
//         let left = values();
//         let right = other_values();

//         let sum = <Array as ScalarAlgebra<C, D>>::add(&left, &right);

//         let difference = <Array as ScalarAlgebra<C, D>>::subtract(&left, &right);

//         let product = <Array as ScalarAlgebra<C, D>>::multiply(&left, &right);

//         let quotient = <Array as ScalarAlgebra<C, D>>::divide(&left, &right);

//         assert_complex_array_close(&sum, &(left.clone() + &right));

//         assert_complex_array_close(&difference, &(left.clone() - &right));

//         assert_complex_array_close(&product, &(left.clone() * &right));

//         assert_complex_array_close(&quotient, &(left / right));
//     }

//     #[test]
//     fn array_scalar_algebra_delegates_unary_functions() {
//         let source = values();

//         let exponential = <Array as ScalarAlgebra<C, D>>::exp(&source);

//         let sine = <Array as ScalarAlgebra<C, D>>::sin(&source);

//         let cosine = <Array as ScalarAlgebra<C, D>>::cos(&source);

//         let square_root = <Array as ScalarAlgebra<C, D>>::sqrt(&source);

//         assert_complex_array_close(&exponential, &source.mapv(C::exp));

//         assert_complex_array_close(&sine, &source.mapv(C::sin));

//         assert_complex_array_close(&cosine, &source.mapv(C::cos));

//         assert_complex_array_close(&square_root, &source.mapv(C::sqrt));
//     }

//     #[test]
//     fn array_real_scalar_algebra_delegates_real_operations() {
//         let source = values();

//         let conjugated = <Array as RealScalarAlgebra<C, D>>::conjugated(&source);

//         let real = <Array as RealScalarAlgebra<C, D>>::real(&source);

//         let magnitude_squared = <Array as RealScalarAlgebra<C, D>>::magnitude_squared(&source);

//         assert_complex_array_close(&conjugated, &source.mapv(|value| value.conj()));

//         assert_real_array_close(&real, &source.mapv(|value| value.re));

//         assert_real_array_close(&magnitude_squared, &source.mapv(|value| value.norm_sqr()));
//     }

//     #[test]
//     fn array_scalar_algebra_builds_cartesian_vector() {
//         let x = values();
//         let y = other_values();
//         let z = first_derivative();

//         let vector =
//             <Array as ScalarAlgebra<C, D>>::into_cartesian_vector(x.clone(), y.clone(), z.clone());

//         assert_eq!(vector.x(), &x);
//         assert_eq!(vector.y(), &y);
//         assert_eq!(vector.z(), &z);
//     }

//     // ---------------------------------------------------------------------
//     // Constant lifting
//     // ---------------------------------------------------------------------

//     #[test]
//     fn first_order_lift_constant_zeros_derivative() {
//         let value = values();

//         let result = <First as ScalarAlgebra<C, D>>::lift_constant(value.clone());

//         assert_eq!(result.value(), &value);

//         assert!(result.first().iter().all(|value| *value == C::default()),);
//     }

//     #[test]
//     fn second_order_lift_constant_zeros_all_derivatives() {
//         let value = values();

//         let result = <Second as ScalarAlgebra<C, D>>::lift_constant(value.clone());

//         assert_eq!(result.value(), &value);

//         assert!(result.first().iter().all(|value| *value == C::default()),);

//         assert!(result.second().iter().all(|value| *value == C::default()),);
//     }

//     #[test]
//     fn bivariate_lift_constant_zeros_gradient_and_hessian() {
//         let value = values();

//         let result = <Bivariate as ScalarAlgebra<C, D>>::lift_constant(value.clone());

//         assert_eq!(result.value(), &value);

//         for derivative in [
//             result.x(),
//             result.y(),
//             result.xx(),
//             result.xy(),
//             result.yy(),
//         ] {
//             assert!(derivative.iter().all(|value| { *value == C::default() },),);
//         }
//     }

//     // ---------------------------------------------------------------------
//     // Variable seeding
//     // ---------------------------------------------------------------------

//     #[test]
//     fn first_order_variable_seeds_unit_first_derivative() {
//         let value = values();

//         let result = <First as UnivariateVariableAlgebra<C, D>>::variable(value.clone());

//         assert_eq!(result.value(), &value);

//         assert!(
//             result
//                 .first()
//                 .iter()
//                 .all(|value| *value == C::new(1.0, 0.0)),
//         );
//     }

//     #[test]
//     fn second_order_variable_seeds_first_but_not_second_derivative() {
//         let value = values();

//         let result = <Second as UnivariateVariableAlgebra<C, D>>::variable(value.clone());

//         assert_eq!(result.value(), &value);

//         assert!(
//             result
//                 .first()
//                 .iter()
//                 .all(|value| *value == C::new(1.0, 0.0)),
//         );

//         assert!(result.second().iter().all(|value| *value == C::default()),);
//     }

//     #[test]
//     fn bivariate_variable_x_seeds_only_x() {
//         let value = values();

//         let result = <Bivariate as BivariateVariableAlgebra<C, D>>::variable_x(value.clone());

//         assert_eq!(result.value(), &value);

//         assert!(result.x().iter().all(|value| *value == C::new(1.0, 0.0)),);

//         for derivative in [result.y(), result.xx(), result.xy(), result.yy()] {
//             assert!(derivative.iter().all(|value| { *value == C::default() },),);
//         }
//     }

//     #[test]
//     fn bivariate_variable_y_seeds_only_y() {
//         let value = values();

//         let result = <Bivariate as BivariateVariableAlgebra<C, D>>::variable_y(value.clone());

//         assert_eq!(result.value(), &value);

//         assert!(result.y().iter().all(|value| *value == C::new(1.0, 0.0)),);

//         for derivative in [result.x(), result.xx(), result.xy(), result.yy()] {
//             assert!(derivative.iter().all(|value| { *value == C::default() },),);
//         }
//     }

//     // ---------------------------------------------------------------------
//     // Arithmetic delegation
//     // ---------------------------------------------------------------------

//     #[test]
//     fn first_order_scalar_algebra_matches_inherent_jet_arithmetic() {
//         let left = JetFirst::from_parts(values(), first_derivative());

//         let right = JetFirst::from_parts(other_values(), other_first_derivative());

//         let via_trait = <First as ScalarAlgebra<C, D>>::multiply(&left, &right);

//         let inherent = JetFirst::multiply(&left, &right);

//         assert_eq!(via_trait, inherent);
//     }

//     #[test]
//     fn second_order_scalar_algebra_matches_inherent_jet_arithmetic() {
//         let left = Jet::from_parts(values(), first_derivative(), second_derivative());

//         let right = Jet::from_parts(
//             other_values(),
//             other_first_derivative(),
//             other_second_derivative(),
//         );

//         let via_trait = <Second as ScalarAlgebra<C, D>>::multiply(&left, &right);

//         let inherent = Jet::multiply(&left, &right);

//         assert_eq!(via_trait, inherent);
//     }

//     #[test]
//     fn bivariate_scalar_algebra_matches_inherent_jet_arithmetic() {
//         let left = JetBivariate::from_components(
//             values(),
//             first_derivative(),
//             other_first_derivative(),
//             second_derivative(),
//             other_second_derivative(),
//             values(),
//         );

//         let right = JetBivariate::from_components(
//             other_values(),
//             other_first_derivative(),
//             first_derivative(),
//             other_second_derivative(),
//             second_derivative(),
//             other_values(),
//         );

//         let via_trait = <Bivariate as ScalarAlgebra<C, D>>::multiply(&left, &right);

//         let inherent = JetBivariate::multiply(&left, &right);

//         assert_eq!(via_trait, inherent);
//     }

//     #[test]
//     fn scalar_algebra_default_square_uses_multiplication() {
//         let source = Jet::from_parts(values(), first_derivative(), second_derivative());

//         let via_trait = <Second as ScalarAlgebra<C, D>>::square(&source);

//         let expected = Jet::multiply(&source, &source);

//         assert_eq!(via_trait, expected);
//     }

//     #[test]
//     fn scalar_algebra_default_divide_uses_reciprocal() {
//         let left = Jet::from_parts(values(), first_derivative(), second_derivative());

//         let right = Jet::from_parts(
//             other_values(),
//             other_first_derivative(),
//             other_second_derivative(),
//         );

//         let via_trait = <Second as ScalarAlgebra<C, D>>::divide(&left, &right);

//         let expected = Jet::multiply(&left, &Jet::reciprocal(&right));

//         assert_eq!(via_trait, expected);
//     }

//     // ---------------------------------------------------------------------
//     // Real scalar operations
//     // ---------------------------------------------------------------------

//     #[test]
//     fn first_order_real_scalar_algebra_matches_inherent_operations() {
//         let source = JetFirst::from_parts(values(), first_derivative());

//         let conjugated = <First as RealScalarAlgebra<C, D>>::conjugated(&source);

//         let real = <First as RealScalarAlgebra<C, D>>::real(&source);

//         let magnitude_squared = <First as RealScalarAlgebra<C, D>>::magnitude_squared(&source);

//         assert_eq!(conjugated, JetFirst::conjugated(&source),);

//         assert_eq!(real, JetFirst::real(&source),);

//         assert_eq!(
//             magnitude_squared,
//             JetFirst::multiply(&source, &JetFirst::conjugated(&source),).real(),
//         );
//     }

//     #[test]
//     fn second_order_real_scalar_algebra_matches_inherent_operations() {
//         let source = Jet::from_parts(values(), first_derivative(), second_derivative());

//         let conjugated = <Second as RealScalarAlgebra<C, D>>::conjugated(&source);

//         let real = <Second as RealScalarAlgebra<C, D>>::real(&source);

//         let magnitude_squared = <Second as RealScalarAlgebra<C, D>>::magnitude_squared(&source);

//         assert_eq!(conjugated, Jet::conjugated(&source),);

//         assert_eq!(real, Jet::real(&source),);

//         assert_eq!(
//             magnitude_squared,
//             Jet::multiply(&source, &Jet::conjugated(&source),).real(),
//         );
//     }

//     #[test]
//     fn bivariate_real_scalar_algebra_matches_inherent_operations() {
//         let source = JetBivariate::from_components(
//             values(),
//             first_derivative(),
//             other_first_derivative(),
//             second_derivative(),
//             other_second_derivative(),
//             values(),
//         );

//         let conjugated = <Bivariate as RealScalarAlgebra<C, D>>::conjugated(&source);

//         let real = <Bivariate as RealScalarAlgebra<C, D>>::real(&source);

//         let magnitude_squared = <Bivariate as RealScalarAlgebra<C, D>>::magnitude_squared(&source);

//         assert_eq!(conjugated, JetBivariate::conjugated(&source),);

//         assert_eq!(real, JetBivariate::real(&source),);

//         assert_eq!(
//             magnitude_squared,
//             JetBivariate::multiply(&source, &JetBivariate::conjugated(&source,),).real(),
//         );
//     }

//     // ---------------------------------------------------------------------
//     // Cartesian vector construction
//     // ---------------------------------------------------------------------

//     #[test]
//     fn first_order_scalar_jets_pack_into_vector_jet() {
//         let x = JetFirst::from_parts(values(), first_derivative());

//         let y = JetFirst::from_parts(other_values(), other_first_derivative());

//         let z = JetFirst::from_parts(second_derivative(), other_second_derivative());

//         let result =
//             <First as ScalarAlgebra<C, D>>::into_cartesian_vector(x.clone(), y.clone(), z.clone());

//         assert_eq!(result.value().x(), x.value(),);

//         assert_eq!(result.value().y(), y.value(),);

//         assert_eq!(result.value().z(), z.value(),);

//         assert_eq!(result.first().x(), x.first(),);

//         assert_eq!(result.first().y(), y.first(),);

//         assert_eq!(result.first().z(), z.first(),);
//     }

//     #[test]
//     fn second_order_scalar_jets_pack_into_vector_jet() {
//         let x = Jet::from_parts(values(), first_derivative(), second_derivative());

//         let y = Jet::from_parts(
//             other_values(),
//             other_first_derivative(),
//             other_second_derivative(),
//         );

//         let z = Jet::from_parts(second_derivative(), other_second_derivative(), values());

//         let result =
//             <Second as ScalarAlgebra<C, D>>::into_cartesian_vector(x.clone(), y.clone(), z.clone());

//         assert_eq!(result.value().x(), x.value(),);

//         assert_eq!(result.value().y(), y.value(),);

//         assert_eq!(result.value().z(), z.value(),);

//         assert_eq!(result.first().x(), x.first(),);

//         assert_eq!(result.first().y(), y.first(),);

//         assert_eq!(result.first().z(), z.first(),);

//         assert_eq!(result.second().x(), x.second(),);

//         assert_eq!(result.second().y(), y.second(),);

//         assert_eq!(result.second().z(), z.second(),);
//     }

//     #[test]
//     fn bivariate_scalar_jets_pack_into_vector_jet() {
//         let x = JetBivariate::from_components(
//             values(),
//             first_derivative(),
//             other_first_derivative(),
//             second_derivative(),
//             other_second_derivative(),
//             values(),
//         );

//         let y = JetBivariate::from_components(
//             other_values(),
//             other_first_derivative(),
//             first_derivative(),
//             other_second_derivative(),
//             values(),
//             second_derivative(),
//         );

//         let z = JetBivariate::from_components(
//             second_derivative(),
//             other_second_derivative(),
//             values(),
//             first_derivative(),
//             other_values(),
//             other_first_derivative(),
//         );

//         let result = <Bivariate as ScalarAlgebra<C, D>>::into_cartesian_vector(
//             x.clone(),
//             y.clone(),
//             z.clone(),
//         );

//         assert_eq!(result.value().x(), x.value(),);

//         assert_eq!(result.value().y(), y.value(),);

//         assert_eq!(result.value().z(), z.value(),);

//         assert_eq!(result.x().x(), x.x());
//         assert_eq!(result.x().y(), y.x());
//         assert_eq!(result.x().z(), z.x());

//         assert_eq!(result.y().x(), x.y());
//         assert_eq!(result.y().y(), y.y());
//         assert_eq!(result.y().z(), z.y());

//         assert_eq!(result.xx().x(), x.xx());
//         assert_eq!(result.xx().y(), y.xx());
//         assert_eq!(result.xx().z(), z.xx());

//         assert_eq!(result.xy().x(), x.xy());
//         assert_eq!(result.xy().y(), y.xy());
//         assert_eq!(result.xy().z(), z.xy());

//         assert_eq!(result.yy().x(), x.yy());
//         assert_eq!(result.yy().y(), y.yy());
//         assert_eq!(result.yy().z(), z.yy());
//     }

//     // ---------------------------------------------------------------------
//     // Finiteness
//     // ---------------------------------------------------------------------

//     #[test]
//     fn all_finite_accepts_finite_arrays_and_jets() {
//         let array = values();

//         let first = JetFirst::from_parts(values(), first_derivative());

//         let second = Jet::from_parts(values(), first_derivative(), second_derivative());

//         let bivariate = JetBivariate::from_components(
//             values(),
//             first_derivative(),
//             other_first_derivative(),
//             second_derivative(),
//             other_second_derivative(),
//             values(),
//         );

//         assert!(<Array as ScalarAlgebra<C, D>>::all_finite(&array),);

//         assert!(<First as ScalarAlgebra<C, D>>::all_finite(&first),);

//         assert!(<Second as ScalarAlgebra<C, D>>::all_finite(&second),);

//         assert!(<Bivariate as ScalarAlgebra<C, D>>::all_finite(&bivariate),);
//     }

//     #[test]
//     fn all_finite_checks_first_order_derivative() {
//         let mut derivative = first_derivative();

//         derivative[1] = c(f64::INFINITY, 0.0);

//         let source = JetFirst::from_parts(values(), derivative);

//         assert!(!<First as ScalarAlgebra<C, D>>::all_finite(&source),);
//     }

//     #[test]
//     fn all_finite_checks_second_order_derivative() {
//         let mut derivative = second_derivative();

//         derivative[1] = c(0.0, f64::NAN);

//         let source = Jet::from_parts(values(), first_derivative(), derivative);

//         assert!(!<Second as ScalarAlgebra<C, D>>::all_finite(&source),);
//     }

//     #[test]
//     fn all_finite_checks_every_bivariate_component() {
//         fn finite_jet() -> Bivariate {
//             JetBivariate::from_components(
//                 values(),
//                 first_derivative(),
//                 other_first_derivative(),
//                 second_derivative(),
//                 other_second_derivative(),
//                 values(),
//             )
//         }

//         let positions = ["value", "x", "y", "xx", "xy", "yy"];

//         for position in positions {
//             let source = finite_jet();

//             let (mut value, mut x, mut y, mut xx, mut xy, mut yy) = source.into_components();

//             match position {
//                 "value" => {
//                     value[0] = c(f64::NAN, 0.0);
//                 }
//                 "x" => {
//                     x[0] = c(f64::NAN, 0.0);
//                 }
//                 "y" => {
//                     y[0] = c(f64::NAN, 0.0);
//                 }
//                 "xx" => {
//                     xx[0] = c(f64::NAN, 0.0);
//                 }
//                 "xy" => {
//                     xy[0] = c(f64::NAN, 0.0);
//                 }
//                 "yy" => {
//                     yy[0] = c(f64::NAN, 0.0);
//                 }
//                 _ => unreachable!(),
//             }

//             let source = JetBivariate::from_components(value, x, y, xx, xy, yy);

//             assert!(
//                 !<Bivariate as ScalarAlgebra<C, D>>::all_finite(&source),
//                 "all_finite ignored {position}",
//             );
//         }
//     }

//     // ---------------------------------------------------------------------
//     // Generic usability
//     // ---------------------------------------------------------------------

//     fn generic_scalar_expression<S>(x: &S, y: &S) -> S
//     where
//         S: ScalarAlgebra<C, D>,
//     {
//         let numerator = x.multiply(y).add(&x.sin());

//         let denominator = y.square().add(&y.constant(c(2.0, 0.0)));

//         numerator.divide(&denominator.sqrt())
//     }

//     #[test]
//     fn generic_scalar_expression_accepts_every_scalar_representation() {
//         let array_x = values();
//         let array_y = other_values();

//         let first_x = JetFirst::from_parts(values(), first_derivative());

//         let first_y = JetFirst::from_parts(other_values(), other_first_derivative());

//         let second_x = Jet::from_parts(values(), first_derivative(), second_derivative());

//         let second_y = Jet::from_parts(
//             other_values(),
//             other_first_derivative(),
//             other_second_derivative(),
//         );

//         let bivariate_x = JetBivariate::from_components(
//             values(),
//             first_derivative(),
//             other_first_derivative(),
//             second_derivative(),
//             other_second_derivative(),
//             values(),
//         );

//         let bivariate_y = JetBivariate::from_components(
//             other_values(),
//             other_first_derivative(),
//             first_derivative(),
//             other_second_derivative(),
//             values(),
//             second_derivative(),
//         );

//         let array_result = generic_scalar_expression(&array_x, &array_y);

//         let first_result = generic_scalar_expression(&first_x, &first_y);

//         let second_result = generic_scalar_expression(&second_x, &second_y);

//         let bivariate_result = generic_scalar_expression(&bivariate_x, &bivariate_y);

//         assert_complex_array_close(first_result.value(), &array_result);

//         assert_complex_array_close(second_result.value(), &array_result);

//         assert_complex_array_close(bivariate_result.value(), &array_result);
//     }
// }
