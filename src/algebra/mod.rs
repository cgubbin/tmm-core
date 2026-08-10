mod cartesian;
mod exprel;
mod jet_bivariate_one;
mod jet_bivariate_two;
mod jet_one;
mod jet_two;
mod jet_zero;
mod scalar;
mod scale;
mod stack;

pub(crate) use exprel::{exprel, exprel_first, exprel_second};
pub(crate) use jet_bivariate_one::{ArrayJetBivariate1, JetBivariate1};
pub(crate) use jet_bivariate_two::{ArrayJetBivariate2, JetBivariate2};
pub(crate) use jet_one::{ArrayJet1, FirstOrderExpansion, Jet1};
pub(crate) use jet_two::{ArrayJet2, Jet2, SecondOrderExpansion};
pub(crate) use jet_zero::{ArrayJet0, Jet0};
pub(crate) use scalar::{
    ComplexJet, Jet, RealScalarAlgebra, ScalarAlgebra, ScalarAlgebraExpRelExt,
};
pub(crate) use stack::JetStack;

pub(crate) use cartesian::{
    CartesianScalarAlgebra, CartesianVectorAlgebra, RealCartesianVectorAlgebra,
};
pub(crate) use scale::ScaleBy;

use nalgebra::ComplexField;
use ndarray::{Array, Dimension};

/// A directional derivative with respect to a real scalar parameter.
///
/// The value and derivative may themselves be complex.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RealParameter;

/// A holomorphic derivative with respect to a complex scalar parameter.
///
/// Only operations preserving holomorphicity should be available.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HolomorphicParameter;

/// Construct a zero value with the same representation and sampled shape.
pub trait JetZeroLike: Clone {
    fn jet_zeros_like(shape_source: &Self) -> Self;
}

/// Construct a unity value with the same representation and sampled shape.
pub trait JetOneLike: Clone {
    fn jet_ones_like(shape_source: &Self) -> Self;
}

/// Additive operations supported componentwise by jets.
pub trait JetAdditive: Sized {
    /// Add two values.
    fn jet_add(&self, rhs: &Self) -> Self;

    /// Subtract `rhs` from this value.
    fn jet_subtract(&self, rhs: &Self) -> Self;

    /// Negate this value.
    fn jet_negate(&self) -> Self;

    /// Return twice this value.
    fn jet_double(&self) -> Self {
        self.jet_add(self)
    }
}

/// A bilinear product for which the ordinary product rule applies.
pub trait JetBilinear: JetAdditive {
    /// Multiply or compose two values using a bilinear operation.
    fn jet_multiply(&self, rhs: &Self) -> Self;

    /// Multiply or compose two values using a bilinear operation.
    fn jet_square(&self) -> Self {
        self.jet_multiply(self)
    }
}

/// Scale a value by a scalar.
pub trait JetScaleBy: Clone {
    type Scalar: Copy;

    fn jet_scale_by(&self, value: Self::Scalar) -> Self;
}

/// Construct a scalar constant with the same sampled shape.
pub trait JetConstant: Clone {
    type Scalar: Copy;

    fn jet_constant_like(&self, value: Self::Scalar) -> Self;
}

/// Elementwise field operations used by reciprocal and division.
///
/// This trait is appropriate for sampled scalar arrays, not matrices.
pub trait JetField: JetBilinear {
    fn jet_elementwise_reciprocal(&self) -> Self;
}

/// A bilinear cross product between values of the same type.
///
/// Implementations must be bilinear in both operands so that the ordinary
/// first- and second-order product rules are valid.
pub trait JetCrossProduct: Clone {
    fn jet_cross(&self, rhs: &Self) -> Self;
}

pub trait JetHermitianProduct: Clone {
    type Output;

    fn jet_hermitian_product(&self, rhs: &Self) -> Self::Output;
}

pub trait JetConjugate {
    fn jet_conjugate(&self) -> Self;
}

pub trait JetRealPart {
    type RealOutput;

    fn jet_real(&self) -> Self::RealOutput;
    fn jet_imaginary(&self) -> Self::RealOutput;
}

pub trait JetMultiplyByScalar<S> {
    fn jet_multiply_by_scalar(&self, scalar: &S) -> Self;
}

impl<C, D> JetZeroLike for Array<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn jet_zeros_like(shape_source: &Self) -> Self {
        Array::zeros(shape_source.raw_dim())
    }
}

impl<C, D> JetOneLike for Array<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn jet_ones_like(shape_source: &Self) -> Self {
        Array::ones(shape_source.raw_dim())
    }
}

impl<C, D> JetAdditive for Array<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn jet_add(&self, rhs: &Self) -> Self {
        self.clone() + rhs.view()
    }

    fn jet_subtract(&self, rhs: &Self) -> Self {
        self.clone() - rhs.view()
    }

    fn jet_negate(&self) -> Self {
        -self.clone()
    }
}

impl<C, D> JetBilinear for Array<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn jet_multiply(&self, rhs: &Self) -> Self {
        self.clone() * rhs.view()
    }
}

impl<C, D> JetScaleBy for Array<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Scalar = C;

    fn jet_scale_by(&self, value: Self::Scalar) -> Self {
        self.mapv(|x| x * value)
    }
}

impl<C, D> JetConstant for Array<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Scalar = C;

    fn jet_constant_like(&self, value: C) -> Self {
        Array::from_elem(self.raw_dim(), value)
    }
}

impl<C, D> JetRealPart for Array<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type RealOutput = Array<C::RealField, D>;

    fn jet_real(&self) -> Self::RealOutput {
        self.mapv(|z| z.real())
    }

    fn jet_imaginary(&self) -> Self::RealOutput {
        self.mapv(|z| z.imaginary())
    }
}

impl<C, D> JetConjugate for Array<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn jet_conjugate(&self) -> Self {
        self.mapv(|z| z.conjugate())
    }
}

impl<C, D> JetField for Array<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn jet_elementwise_reciprocal(&self) -> Self {
        self.mapv(|x| C::one() / x)
    }
}

#[cfg(test)]
mod noncommutative_tests {
    use super::*;

    use crate::algebra::jet_bivariate_two::JetBivariate2;
    use crate::algebra::jet_one::Jet1;
    use crate::algebra::jet_two::Jet2;

    /// A small exact-arithmetic matrix type used to verify that jet product
    /// rules preserve operand ordering.
    ///
    /// Matrix multiplication is deliberately noncommutative:
    ///
    /// ```text
    /// A B != B A
    /// ```
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Matrix2 {
        entries: [[i64; 2]; 2],
    }

    impl Matrix2 {
        const fn new(a11: i64, a12: i64, a21: i64, a22: i64) -> Self {
            Self {
                entries: [[a11, a12], [a21, a22]],
            }
        }

        fn add(self, rhs: Self) -> Self {
            Self::new(
                self.entries[0][0] + rhs.entries[0][0],
                self.entries[0][1] + rhs.entries[0][1],
                self.entries[1][0] + rhs.entries[1][0],
                self.entries[1][1] + rhs.entries[1][1],
            )
        }

        fn subtract(self, rhs: Self) -> Self {
            Self::new(
                self.entries[0][0] - rhs.entries[0][0],
                self.entries[0][1] - rhs.entries[0][1],
                self.entries[1][0] - rhs.entries[1][0],
                self.entries[1][1] - rhs.entries[1][1],
            )
        }

        fn negate(self) -> Self {
            Self::new(
                -self.entries[0][0],
                -self.entries[0][1],
                -self.entries[1][0],
                -self.entries[1][1],
            )
        }

        fn multiply(self, rhs: Self) -> Self {
            let [[a, b], [c, d]] = self.entries;
            let [[e, f], [g, h]] = rhs.entries;

            Self::new(a * e + b * g, a * f + b * h, c * e + d * g, c * f + d * h)
        }

        fn double(self) -> Self {
            self.add(self)
        }
    }

    impl JetAdditive for Matrix2 {
        fn jet_add(&self, rhs: &Self) -> Self {
            (*self).add(*rhs)
        }

        fn jet_subtract(&self, rhs: &Self) -> Self {
            (*self).subtract(*rhs)
        }

        fn jet_negate(&self) -> Self {
            (*self).negate()
        }

        fn jet_double(&self) -> Self {
            (*self).double()
        }
    }

    impl JetBilinear for Matrix2 {
        fn jet_multiply(&self, rhs: &Self) -> Self {
            (*self).multiply(*rhs)
        }
    }

    fn assert_noncommutative(left: Matrix2, right: Matrix2) {
        assert_ne!(
            left.multiply(right),
            right.multiply(left),
            "test matrices must not commute",
        );
    }

    // Each component is intentionally distinct so that an incorrect operand
    // order is unlikely to produce the expected result accidentally.

    const F: Matrix2 = Matrix2::new(1, 2, 0, 1);
    const FX: Matrix2 = Matrix2::new(0, 1, 3, 0);
    const FY: Matrix2 = Matrix2::new(2, 0, 1, 4);
    const FXX: Matrix2 = Matrix2::new(1, 3, 2, 0);
    const FXY: Matrix2 = Matrix2::new(0, 2, 5, 1);
    const FYY: Matrix2 = Matrix2::new(3, 1, 0, 2);

    const G: Matrix2 = Matrix2::new(2, 0, 1, 3);
    const GX: Matrix2 = Matrix2::new(1, 4, 0, 2);
    const GY: Matrix2 = Matrix2::new(0, 3, 2, 1);
    const GXX: Matrix2 = Matrix2::new(4, 0, 1, 1);
    const GXY: Matrix2 = Matrix2::new(2, 1, 3, 0);
    const GYY: Matrix2 = Matrix2::new(1, 2, 4, 3);

    // ---------------------------------------------------------------------
    // First-order univariate jet
    // ---------------------------------------------------------------------

    #[test]
    fn jet_first_multiplication_preserves_operand_order() {
        assert_noncommutative(F, G);
        assert_noncommutative(FX, G);
        assert_noncommutative(F, GX);

        let left: Jet1<Matrix2, RealParameter> = Jet1::from_parts(F, FX);

        let right: Jet1<Matrix2, RealParameter> = Jet1::from_parts(G, GX);

        let result = left.multiply(&right);

        let expected_value = F.multiply(G);

        let expected_first = FX.multiply(G).add(F.multiply(GX));

        assert_eq!(result.value(), &expected_value);
        assert_eq!(result.first(), &expected_first);

        // These are representative incorrect formulas. The assertions make
        // the ordering requirement explicit and ensure the chosen fixtures
        // can actually distinguish the implementations.
        let reversed_value = G.multiply(F);

        let reversed_first = G.multiply(FX).add(GX.multiply(F));

        assert_ne!(expected_value, reversed_value);
        assert_ne!(expected_first, reversed_first);
    }

    // ---------------------------------------------------------------------
    // Second-order univariate jet
    // ---------------------------------------------------------------------

    #[test]
    fn jet_multiplication_preserves_operand_order() {
        assert_noncommutative(F, G);
        assert_noncommutative(FX, GX);
        assert_noncommutative(FXX, G);
        assert_noncommutative(F, GXX);

        let left: Jet2<Matrix2, RealParameter> = Jet2::from_parts(F, FX, FXX);

        let right: Jet2<Matrix2, RealParameter> = Jet2::from_parts(G, GX, GXX);

        let result = left.multiply(&right);

        let expected_value = F.multiply(G);

        let expected_first = FX.multiply(G).add(F.multiply(GX));

        let expected_second = FXX
            .multiply(G)
            .add(FX.multiply(GX))
            .add(FX.multiply(GX))
            .add(F.multiply(GXX));

        assert_eq!(result.value(), &expected_value);
        assert_eq!(result.first(), &expected_first);
        assert_eq!(result.second(), &expected_second);

        // Incorrect formulas that reverse one or more bilinear operands.
        let reversed_value = G.multiply(F);

        let reversed_first = G.multiply(FX).add(GX.multiply(F));

        let reversed_second = G
            .multiply(FXX)
            .add(GX.multiply(FX))
            .add(GX.multiply(FX))
            .add(GXX.multiply(F));

        assert_ne!(expected_value, reversed_value);
        assert_ne!(expected_first, reversed_first);
        assert_ne!(expected_second, reversed_second);
    }

    #[test]
    fn jet_second_derivative_uses_two_identical_ordered_mixed_terms() {
        let left: Jet2<Matrix2, RealParameter> = Jet2::from_parts(F, FX, FXX);

        let right: Jet2<Matrix2, RealParameter> = Jet2::from_parts(G, GX, GXX);

        let result = left.multiply(&right);

        let correct = FXX
            .multiply(G)
            .add(FX.multiply(GX))
            .add(FX.multiply(GX))
            .add(F.multiply(GXX));

        // This superficially resembles a symmetrised product rule, but it is
        // incorrect. Both mixed terms must be f' g', not f' g' + g' f'.
        let incorrectly_symmetrised = FXX
            .multiply(G)
            .add(FX.multiply(GX))
            .add(GX.multiply(FX))
            .add(F.multiply(GXX));

        assert_eq!(result.second(), &correct);
        assert_ne!(correct, incorrectly_symmetrised);
    }

    // ---------------------------------------------------------------------
    // Second-order bivariate jet
    // ---------------------------------------------------------------------

    #[test]
    fn bivariate_jet_multiplication_preserves_operand_order() {
        let left: JetBivariate2<Matrix2, RealParameter> =
            JetBivariate2::from_components(F, FX, FY, FXX, FXY, FYY);

        let right: JetBivariate2<Matrix2, RealParameter> =
            JetBivariate2::from_components(G, GX, GY, GXX, GXY, GYY);

        let result = left.multiply(&right);

        let expected_value = F.multiply(G);

        let expected_x = FX.multiply(G).add(F.multiply(GX));

        let expected_y = FY.multiply(G).add(F.multiply(GY));

        let expected_xx = FXX
            .multiply(G)
            .add(FX.multiply(GX))
            .add(FX.multiply(GX))
            .add(F.multiply(GXX));

        let expected_xy = FXY
            .multiply(G)
            .add(FX.multiply(GY))
            .add(FY.multiply(GX))
            .add(F.multiply(GXY));

        let expected_yy = FYY
            .multiply(G)
            .add(FY.multiply(GY))
            .add(FY.multiply(GY))
            .add(F.multiply(GYY));

        assert_eq!(result.value(), &expected_value);
        assert_eq!(result.axis0(), &expected_x);
        assert_eq!(result.axis1(), &expected_y);
        assert_eq!(result.axis0_axis0(), &expected_xx);
        assert_eq!(result.axis0_axis1(), &expected_xy);
        assert_eq!(result.axis1_axis1(), &expected_yy);
    }

    #[test]
    fn bivariate_jet_pure_second_derivatives_repeat_ordered_mixed_terms() {
        let left: JetBivariate2<Matrix2, RealParameter> =
            JetBivariate2::from_components(F, FX, FY, FXX, FXY, FYY);

        let right: JetBivariate2<Matrix2, RealParameter> =
            JetBivariate2::from_components(G, GX, GY, GXX, GXY, GYY);

        let result = left.multiply(&right);

        let correct_xx = FXX
            .multiply(G)
            .add(FX.multiply(GX))
            .add(FX.multiply(GX))
            .add(F.multiply(GXX));

        let incorrectly_symmetrised_xx = FXX
            .multiply(G)
            .add(FX.multiply(GX))
            .add(GX.multiply(FX))
            .add(F.multiply(GXX));

        let correct_yy = FYY
            .multiply(G)
            .add(FY.multiply(GY))
            .add(FY.multiply(GY))
            .add(F.multiply(GYY));

        let incorrectly_symmetrised_yy = FYY
            .multiply(G)
            .add(FY.multiply(GY))
            .add(GY.multiply(FY))
            .add(F.multiply(GYY));

        assert_eq!(result.axis0_axis0(), &correct_xx);
        assert_eq!(result.axis1_axis1(), &correct_yy);

        assert_ne!(correct_xx, incorrectly_symmetrised_xx);
        assert_ne!(correct_yy, incorrectly_symmetrised_yy);
    }

    #[test]
    fn bivariate_jet_mixed_derivative_preserves_both_cross_orders() {
        let left: JetBivariate2<Matrix2, RealParameter> =
            JetBivariate2::from_components(F, FX, FY, FXX, FXY, FYY);

        let right: JetBivariate2<Matrix2, RealParameter> =
            JetBivariate2::from_components(G, GX, GY, GXX, GXY, GYY);

        let result = left.multiply(&right);

        let expected = FXY
            .multiply(G)
            .add(FX.multiply(GY))
            .add(FY.multiply(GX))
            .add(F.multiply(GXY));

        assert_eq!(result.axis0_axis1(), &expected);

        // Each incorrect candidate reverses a different operand pair.
        let reversed_outer_terms = G
            .multiply(FXY)
            .add(FX.multiply(GY))
            .add(FY.multiply(GX))
            .add(GXY.multiply(F));

        let reversed_cross_terms = FXY
            .multiply(G)
            .add(GY.multiply(FX))
            .add(GX.multiply(FY))
            .add(F.multiply(GXY));

        let exchanged_cross_terms = FXY
            .multiply(G)
            .add(FY.multiply(GX))
            .add(FX.multiply(GY))
            .add(F.multiply(GXY));

        assert_ne!(expected, reversed_outer_terms);
        assert_ne!(expected, reversed_cross_terms);

        // Addition is commutative for this payload, so exchanging the two
        // correctly ordered cross terms does not change the result. What
        // matters is the order within each multiplication.
        assert_eq!(expected, exchanged_cross_terms);
    }

    // ---------------------------------------------------------------------
    // Marker independence
    // ---------------------------------------------------------------------

    #[test]
    fn holomorphic_jet_first_uses_the_same_ordered_product_rule() {
        let left: Jet1<Matrix2, HolomorphicParameter> = Jet1::from_parts(F, FX);

        let right: Jet1<Matrix2, HolomorphicParameter> = Jet1::from_parts(G, GX);

        let result = left.multiply(&right);

        assert_eq!(result.value(), &F.multiply(G));

        assert_eq!(result.first(), &FX.multiply(G).add(F.multiply(GX)),);
    }

    #[test]
    fn holomorphic_jet_uses_the_same_ordered_product_rule() {
        let left: Jet2<Matrix2, HolomorphicParameter> = Jet2::from_parts(F, FX, FXX);

        let right: Jet2<Matrix2, HolomorphicParameter> = Jet2::from_parts(G, GX, GXX);

        let result = left.multiply(&right);

        assert_eq!(result.value(), &F.multiply(G));

        assert_eq!(result.first(), &FX.multiply(G).add(F.multiply(GX)),);

        assert_eq!(
            result.second(),
            &FXX.multiply(G)
                .add(FX.multiply(GX))
                .add(FX.multiply(GX))
                .add(F.multiply(GXX)),
        );
    }

    #[test]
    fn holomorphic_bivariate_jet_uses_the_same_ordered_product_rule() {
        let left: JetBivariate2<Matrix2, HolomorphicParameter> =
            JetBivariate2::from_components(F, FX, FY, FXX, FXY, FYY);

        let right: JetBivariate2<Matrix2, HolomorphicParameter> =
            JetBivariate2::from_components(G, GX, GY, GXX, GXY, GYY);

        let result = left.multiply(&right);

        assert_eq!(result.value(), &F.multiply(G));

        assert_eq!(result.axis0(), &FX.multiply(G).add(F.multiply(GX)),);

        assert_eq!(result.axis1(), &FY.multiply(G).add(F.multiply(GY)),);

        assert_eq!(
            result.axis0_axis0(),
            &FXX.multiply(G)
                .add(FX.multiply(GX))
                .add(FX.multiply(GX))
                .add(F.multiply(GXX)),
        );

        assert_eq!(
            result.axis0_axis1(),
            &FXY.multiply(G)
                .add(FX.multiply(GY))
                .add(FY.multiply(GX))
                .add(F.multiply(GXY)),
        );

        assert_eq!(
            result.axis1_axis1(),
            &FYY.multiply(G)
                .add(FY.multiply(GY))
                .add(FY.multiply(GY))
                .add(F.multiply(GYY)),
        );
    }
}
