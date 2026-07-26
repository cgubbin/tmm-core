use nalgebra::ComplexField;
use ndarray::{Array, ArrayBase, Dimension, OwnedRepr};
use std::fmt::Debug;

use super::{
    ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, FirstOrderExpansion,
    RealParameter, SecondOrderExpansion,
};

pub trait ComplexJet {
    type RealJet;
}

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
pub(crate) trait RealScalarAlgebra<T, D>: ScalarAlgebra<T, D> + ComplexJet
where
    D: Dimension,
{
    fn conjugated(&self) -> Self;

    fn real(&self) -> Self::RealJet;

    fn magnitude_squared(&self) -> Self::RealJet {
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

impl<C, D, P> ScalarAlgebra<C, D> for ArrayJet0<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArrayJet0::value(self)
    }

    fn lift_constant(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJet0::constant(value)
    }

    fn filled_constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        ArrayJet0::constant_like(source, value)
    }

    fn zero_like(&self) -> Self {
        ArrayJet0::constant_like(self.value(), C::zero())
    }

    fn add(&self, rhs: &Self) -> Self {
        ArrayJet0::add(self, rhs)
    }

    fn subtract(&self, rhs: &Self) -> Self {
        ArrayJet0::subtract(self, rhs)
    }

    fn negate(&self) -> Self {
        ArrayJet0::negate(self)
    }

    fn multiply(&self, rhs: &Self) -> Self {
        ArrayJet0::multiply(self, rhs)
    }

    fn reciprocal(&self) -> Self {
        ArrayJet0::reciprocal(self)
    }

    fn scale(&self, coefficient: C) -> Self {
        ArrayJet0::scale_by(self, coefficient)
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
    }
}

impl<C, D, P> ComplexJet for ArrayJet0<C, D, P>
where
    C: ComplexField,
{
    type RealJet = ArrayJet0<C::RealField, D, P>;
}

impl<C, D> RealScalarAlgebra<C, D> for ArrayJet0<C, D, RealParameter>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn conjugated(&self) -> Self {
        ArrayJet0::conjugated(self)
    }

    fn real(&self) -> Self::RealJet {
        ArrayJet0::real(self)
    }
}

// -------------------------------------------------------------------------
// First-order univariate jets
// -------------------------------------------------------------------------

impl<C, D, P> ScalarAlgebra<C, D> for ArrayJet1<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArrayJet1::value(self)
    }

    fn lift_constant(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJet1::constant(value)
    }

    fn filled_constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        ArrayJet1::constant_like(source, value)
    }

    fn zero_like(&self) -> Self {
        ArrayJet1::constant_like(self.value(), C::zero())
    }

    fn add(&self, rhs: &Self) -> Self {
        ArrayJet1::add(self, rhs)
    }

    fn subtract(&self, rhs: &Self) -> Self {
        ArrayJet1::subtract(self, rhs)
    }

    fn negate(&self) -> Self {
        ArrayJet1::negate(self)
    }

    fn multiply(&self, rhs: &Self) -> Self {
        ArrayJet1::multiply(self, rhs)
    }

    fn reciprocal(&self) -> Self {
        ArrayJet1::reciprocal(self)
    }

    fn scale(&self, coefficient: C) -> Self {
        ArrayJet1::scale_by(self, coefficient)
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

impl<C, D, P> ComplexJet for ArrayJet1<C, D, P>
where
    C: ComplexField,
{
    type RealJet = ArrayJet1<C::RealField, D, P>;
}

impl<C, D> RealScalarAlgebra<C, D> for ArrayJet1<C, D, RealParameter>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn conjugated(&self) -> Self {
        ArrayJet1::conjugated(self)
    }

    fn real(&self) -> Self::RealJet {
        ArrayJet1::real(self)
    }
}

impl<C, D, P> UnivariateVariableAlgebra<C, D> for ArrayJet1<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn variable(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJet1::variable(value)
    }
}

impl<C, D, P> FirstOrderFunctionAlgebra<C, D> for ArrayJet1<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn compose_sampled_function(
        argument: &Self,
        expansion: FirstOrderExpansion<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self {
        ArrayJet1::compose_sampled_function(argument, expansion)
    }
}

// -------------------------------------------------------------------------
// Second-order univariate jets
// -------------------------------------------------------------------------

impl<C, D, P> ScalarAlgebra<C, D> for ArrayJet2<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArrayJet2::value(self)
    }

    fn lift_constant(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJet2::constant(value)
    }

    fn filled_constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        ArrayJet2::constant_like(source, value)
    }

    fn zero_like(&self) -> Self {
        ArrayJet2::constant_like(self.value(), C::zero())
    }

    fn add(&self, rhs: &Self) -> Self {
        ArrayJet2::add(self, rhs)
    }

    fn subtract(&self, rhs: &Self) -> Self {
        ArrayJet2::subtract(self, rhs)
    }

    fn negate(&self) -> Self {
        ArrayJet2::negate(self)
    }

    fn multiply(&self, rhs: &Self) -> Self {
        ArrayJet2::multiply(self, rhs)
    }

    fn reciprocal(&self) -> Self {
        ArrayJet2::reciprocal(self)
    }

    fn scale(&self, coefficient: C) -> Self {
        ArrayJet2::scale_by(self, coefficient)
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

impl<C, D, P> ComplexJet for ArrayJet2<C, D, P>
where
    C: ComplexField,
{
    type RealJet = ArrayJet2<C::RealField, D, P>;
}

impl<C, D> RealScalarAlgebra<C, D> for ArrayJet2<C, D, RealParameter>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn conjugated(&self) -> Self {
        ArrayJet2::conjugated(self)
    }

    fn real(&self) -> Self::RealJet {
        ArrayJet2::real(self)
    }
}

impl<C, D, P> UnivariateVariableAlgebra<C, D> for ArrayJet2<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn variable(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJet2::variable(value)
    }
}

impl<C, D, P> SecondOrderFunctionAlgebra<C, D> for ArrayJet2<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn compose_sampled_function(
        argument: &Self,
        expansion: SecondOrderExpansion<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self {
        ArrayJet2::compose_sampled_function(argument, expansion)
    }
}

// -------------------------------------------------------------------------
// Second-order bivariate jets
// -------------------------------------------------------------------------

impl<C, D, P> ScalarAlgebra<C, D> for ArrayJetBivariate1<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArrayJetBivariate1::value(self)
    }

    fn lift_constant(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJetBivariate1::constant(value)
    }

    fn filled_constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        ArrayJetBivariate1::constant_like(source, value)
    }

    fn zero_like(&self) -> Self {
        ArrayJetBivariate1::constant_like(self.value(), C::zero())
    }

    fn add(&self, rhs: &Self) -> Self {
        ArrayJetBivariate1::add(self, rhs)
    }

    fn subtract(&self, rhs: &Self) -> Self {
        ArrayJetBivariate1::subtract(self, rhs)
    }

    fn negate(&self) -> Self {
        ArrayJetBivariate1::negate(self)
    }

    fn multiply(&self, rhs: &Self) -> Self {
        ArrayJetBivariate1::multiply(self, rhs)
    }

    fn reciprocal(&self) -> Self {
        ArrayJetBivariate1::reciprocal(self)
    }

    fn scale(&self, coefficient: C) -> Self {
        ArrayJetBivariate1::scale_by(self, coefficient)
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
        array_is_finite(self.value()) && array_is_finite(self.x()) && array_is_finite(self.y())
    }
}

impl<C, D, P> ComplexJet for ArrayJetBivariate1<C, D, P>
where
    C: ComplexField,
{
    type RealJet = ArrayJetBivariate1<C::RealField, D, P>;
}

impl<C, D> RealScalarAlgebra<C, D> for ArrayJetBivariate1<C, D, RealParameter>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn conjugated(&self) -> Self {
        ArrayJetBivariate1::conjugated(self)
    }

    fn real(&self) -> Self::RealJet {
        ArrayJetBivariate1::real(self)
    }
}

impl<C, D, P> BivariateVariableAlgebra<C, D> for ArrayJetBivariate1<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn variable_x(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJetBivariate1::variable_x(value)
    }

    fn variable_y(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJetBivariate1::variable_y(value)
    }
}

impl<C, D, P> FirstOrderFunctionAlgebra<C, D> for ArrayJetBivariate1<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn compose_sampled_function(
        argument: &Self,
        expansion: FirstOrderExpansion<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self {
        ArrayJetBivariate1::compose_sampled_function(argument, expansion)
    }
}

impl<C, D, P> ScalarAlgebra<C, D> for ArrayJetBivariate2<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        ArrayJetBivariate2::value(self)
    }

    fn lift_constant(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJetBivariate2::constant(value)
    }

    fn filled_constant_like(source: &ArrayBase<OwnedRepr<C>, D>, value: C) -> Self {
        ArrayJetBivariate2::constant_like(source, value)
    }

    fn zero_like(&self) -> Self {
        ArrayJetBivariate2::constant_like(self.value(), C::zero())
    }

    fn add(&self, rhs: &Self) -> Self {
        ArrayJetBivariate2::add(self, rhs)
    }

    fn subtract(&self, rhs: &Self) -> Self {
        ArrayJetBivariate2::subtract(self, rhs)
    }

    fn negate(&self) -> Self {
        ArrayJetBivariate2::negate(self)
    }

    fn multiply(&self, rhs: &Self) -> Self {
        ArrayJetBivariate2::multiply(self, rhs)
    }

    fn reciprocal(&self) -> Self {
        ArrayJetBivariate2::reciprocal(self)
    }

    fn scale(&self, coefficient: C) -> Self {
        ArrayJetBivariate2::scale_by(self, coefficient)
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

impl<C, D, P> ComplexJet for ArrayJetBivariate2<C, D, P>
where
    C: ComplexField,
{
    type RealJet = ArrayJetBivariate2<C::RealField, D, P>;
}

impl<C, D> RealScalarAlgebra<C, D> for ArrayJetBivariate2<C, D, RealParameter>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn conjugated(&self) -> Self {
        ArrayJetBivariate2::conjugated(self)
    }

    fn real(&self) -> Self::RealJet {
        ArrayJetBivariate2::real(self)
    }
}

impl<C, D, P> BivariateVariableAlgebra<C, D> for ArrayJetBivariate2<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn variable_x(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJetBivariate2::variable_x(value)
    }

    fn variable_y(value: ArrayBase<OwnedRepr<C>, D>) -> Self {
        ArrayJetBivariate2::variable_y(value)
    }
}

impl<C, D, P> SecondOrderFunctionAlgebra<C, D> for ArrayJetBivariate2<C, D, P>
where
    C: ComplexField + Copy,
    D: Dimension,
    P: Clone + Debug,
{
    fn compose_sampled_function(
        argument: &Self,
        expansion: SecondOrderExpansion<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self {
        ArrayJetBivariate2::compose_sampled_function(argument, expansion)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use approx::assert_relative_eq;
    use ndarray::{Array1, Ix1, array};
    use num_complex::Complex64;

    use crate::algebra::{
        ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate2, Jet1, Jet2, JetBivariate2,
        RealParameter,
    };

    type C = Complex64;
    type D = Ix1;
    type Array = Array1<C>;
    type RealArray = Array1<f64>;

    type Zero = ArrayJet0<C, D, RealParameter>;

    type First = ArrayJet1<C, D, RealParameter>;

    type Second = ArrayJet2<C, D, RealParameter>;

    type Bivariate = ArrayJetBivariate2<C, D, RealParameter>;

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

        let result = <Zero as ScalarAlgebra<C, D>>::lift_constant(source.clone());

        assert_eq!(result.into_inner(), source);
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
//         let left = Jet1::from_parts(values(), first_derivative());

//         let right = Jet1::from_parts(other_values(), other_first_derivative());

//         let via_trait = <First as ScalarAlgebra<C, D>>::multiply(&left, &right);

//         let inherent = Jet1::multiply(&left, &right);

//         assert_eq!(via_trait, inherent);
//     }

//     #[test]
//     fn second_order_scalar_algebra_matches_inherent_jet_arithmetic() {
//         let left = Jet2::from_parts(values(), first_derivative(), second_derivative());

//         let right = Jet2::from_parts(
//             other_values(),
//             other_first_derivative(),
//             other_second_derivative(),
//         );

//         let via_trait = <Second as ScalarAlgebra<C, D>>::multiply(&left, &right);

//         let inherent = Jet2::multiply(&left, &right);

//         assert_eq!(via_trait, inherent);
//     }

//     #[test]
//     fn bivariate_scalar_algebra_matches_inherent_jet_arithmetic() {
//         let left = JetBivariate2::from_components(
//             values(),
//             first_derivative(),
//             other_first_derivative(),
//             second_derivative(),
//             other_second_derivative(),
//             values(),
//         );

//         let right = JetBivariate2::from_components(
//             other_values(),
//             other_first_derivative(),
//             first_derivative(),
//             other_second_derivative(),
//             second_derivative(),
//             other_values(),
//         );

//         let via_trait = <Bivariate as ScalarAlgebra<C, D>>::multiply(&left, &right);

//         let inherent = JetBivariate2::multiply(&left, &right);

//         assert_eq!(via_trait, inherent);
//     }

//     #[test]
//     fn scalar_algebra_default_square_uses_multiplication() {
//         let source = Jet2::from_parts(values(), first_derivative(), second_derivative());

//         let via_trait = <Second as ScalarAlgebra<C, D>>::square(&source);

//         let expected = Jet2::multiply(&source, &source);

//         assert_eq!(via_trait, expected);
//     }

//     #[test]
//     fn scalar_algebra_default_divide_uses_reciprocal() {
//         let left = Jet2::from_parts(values(), first_derivative(), second_derivative());

//         let right = Jet2::from_parts(
//             other_values(),
//             other_first_derivative(),
//             other_second_derivative(),
//         );

//         let via_trait = <Second as ScalarAlgebra<C, D>>::divide(&left, &right);

//         let expected = Jet2::multiply(&left, &Jet::reciprocal(&right));

//         assert_eq!(via_trait, expected);
//     }

//     // ---------------------------------------------------------------------
//     // Real scalar operations
//     // ---------------------------------------------------------------------

//     #[test]
//     fn first_order_real_scalar_algebra_matches_inherent_operations() {
//         let source = Jet1::from_parts(values(), first_derivative());

//         let conjugated = <First as RealScalarAlgebra<C, D>>::conjugated(&source);

//         let real = <First as RealScalarAlgebra<C, D>>::real(&source);

//         let magnitude_squared = <First as RealScalarAlgebra<C, D>>::magnitude_squared(&source);

//         assert_eq!(conjugated, Jet1::conjugated(&source),);

//         assert_eq!(real, Jet1::real(&source),);

//         assert_eq!(
//             magnitude_squared,
//             Jet1::multiply(&source, &Jet1::conjugated(&source),).real(),
//         );
//     }

//     #[test]
//     fn second_order_real_scalar_algebra_matches_inherent_operations() {
//         let source = Jet2::from_parts(values(), first_derivative(), second_derivative());

//         let conjugated = <Second as RealScalarAlgebra<C, D>>::conjugated(&source);

//         let real = <Second as RealScalarAlgebra<C, D>>::real(&source);

//         let magnitude_squared = <Second as RealScalarAlgebra<C, D>>::magnitude_squared(&source);

//         assert_eq!(conjugated, Jet2::conjugated(&source),);

//         assert_eq!(real, Jet2::real(&source),);

//         assert_eq!(
//             magnitude_squared,
//             Jet2::multiply(&source, &Jet::conjugated(&source),).real(),
//         );
//     }

//     #[test]
//     fn bivariate_real_scalar_algebra_matches_inherent_operations() {
//         let source = JetBivariate2::from_components(
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

//         assert_eq!(conjugated, JetBivariate2::conjugated(&source),);

//         assert_eq!(real, JetBivariate2::real(&source),);

//         assert_eq!(
//             magnitude_squared,
//             JetBivariate2::multiply(&source, &JetBivariate::conjugated(&source,),).real(),
//         );
//     }

//     // ---------------------------------------------------------------------
//     // Cartesian vector construction
//     // ---------------------------------------------------------------------

//     #[test]
//     fn first_order_scalar_jets_pack_into_vector_jet() {
//         let x = Jet1::from_parts(values(), first_derivative());

//         let y = Jet1::from_parts(other_values(), other_first_derivative());

//         let z = Jet1::from_parts(second_derivative(), other_second_derivative());

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
//         let x = Jet2::from_parts(values(), first_derivative(), second_derivative());

//         let y = Jet2::from_parts(
//             other_values(),
//             other_first_derivative(),
//             other_second_derivative(),
//         );

//         let z = Jet2::from_parts(second_derivative(), other_second_derivative(), values());

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
//         let x = JetBivariate2::from_components(
//             values(),
//             first_derivative(),
//             other_first_derivative(),
//             second_derivative(),
//             other_second_derivative(),
//             values(),
//         );

//         let y = JetBivariate2::from_components(
//             other_values(),
//             other_first_derivative(),
//             first_derivative(),
//             other_second_derivative(),
//             values(),
//             second_derivative(),
//         );

//         let z = JetBivariate2::from_components(
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

//         let first = Jet1::from_parts(values(), first_derivative());

//         let second = Jet2::from_parts(values(), first_derivative(), second_derivative());

//         let bivariate = JetBivariate2::from_components(
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

//         let source = Jet1::from_parts(values(), derivative);

//         assert!(!<First as ScalarAlgebra<C, D>>::all_finite(&source),);
//     }

//     #[test]
//     fn all_finite_checks_second_order_derivative() {
//         let mut derivative = second_derivative();

//         derivative[1] = c(0.0, f64::NAN);

//         let source = Jet2::from_parts(values(), first_derivative(), derivative);

//         assert!(!<Second as ScalarAlgebra<C, D>>::all_finite(&source),);
//     }

//     #[test]
//     fn all_finite_checks_every_bivariate_component() {
//         fn finite_jet() -> Bivariate {
//             JetBivariate2::from_components(
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

//             let source = JetBivariate2::from_components(value, x, y, xx, xy, yy);

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

//         let first_x = Jet1::from_parts(values(), first_derivative());

//         let first_y = Jet1::from_parts(other_values(), other_first_derivative());

//         let second_x = Jet2::from_parts(values(), first_derivative(), second_derivative());

//         let second_y = Jet2::from_parts(
//             other_values(),
//             other_first_derivative(),
//             other_second_derivative(),
//         );

//         let bivariate_x = JetBivariate2::from_components(
//             values(),
//             first_derivative(),
//             other_first_derivative(),
//             second_derivative(),
//             other_second_derivative(),
//             values(),
//         );

//         let bivariate_y = JetBivariate2::from_components(
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
