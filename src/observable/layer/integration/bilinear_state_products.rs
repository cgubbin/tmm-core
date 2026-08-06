use ndarray::Dimension;

use crate::{
    ComplexScalar, algebra::ScalarAlgebra, observable::layer::integration::IntegratedWaveProducts,
};

/// Spatially integrated bilinear cross-products of two canonical isotropic
/// states.
///
/// Neither state is conjugated, unlike a Hermitian product:
///
/// ```text
/// field_field
///     = ∫ left_field right_field dz
///
/// secondary_secondary
///     = ∫ left_secondary right_secondary dz
///
/// field_secondary
///     = ∫ left_field right_secondary dz
///
/// secondary_field
///     = ∫ left_secondary right_field dz
/// ```
///
/// For the reciprocal scalar isotropic formulation used here, exchanging the
/// two operands transposes the mixed products and leaves the complete field
/// contraction unchanged.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IntegratedBilinearCrossStateProducts<A> {
    field_field: A,
    secondary_secondary: A,
    field_secondary: A,
    secondary_field: A,
}

impl<A> IntegratedBilinearCrossStateProducts<A> {
    pub(crate) const fn new(
        field_field: A,
        secondary_secondary: A,
        field_secondary: A,
        secondary_field: A,
    ) -> Self {
        Self {
            field_field,
            secondary_secondary,
            field_secondary,
            secondary_field,
        }
    }

    pub(crate) fn field_field(&self) -> &A {
        &self.field_field
    }

    pub(crate) fn secondary_secondary(&self) -> &A {
        &self.secondary_secondary
    }

    pub(crate) fn field_secondary(&self) -> &A {
        &self.field_secondary
    }

    pub(crate) fn secondary_field(&self) -> &A {
        &self.secondary_field
    }

    pub(crate) fn into_parts(self) -> (A, A, A, A) {
        (
            self.field_field,
            self.secondary_secondary,
            self.field_secondary,
            self.secondary_field,
        )
    }

    pub(crate) fn map<B>(
        self,
        mut map: impl FnMut(A) -> B,
    ) -> IntegratedBilinearCrossStateProducts<B> {
        IntegratedBilinearCrossStateProducts::new(
            map(self.field_field),
            map(self.secondary_secondary),
            map(self.field_secondary),
            map(self.secondary_field),
        )
    }
}

/// Transform integrated directional-wave products into integrated canonical
/// state products.
pub(crate) fn project_integrated_bilinear_state_products<A>(
    products: &IntegratedWaveProducts<A>,
    admittance: &A,
) -> IntegratedBilinearCrossStateProducts<A>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    let cross = project_integrated_bilinear_cross_state_products(products, admittance, admittance);

    let (field_field, secondary_secondary, field_secondary, secondary_field) = cross.into_parts();

    IntegratedBilinearCrossStateProducts::new(
        field_field,
        secondary_secondary,
        field_secondary,
        secondary_field,
    )
}

/// Transform integrated bilinear directional-wave cross-products into
/// integrated canonical-state cross-products.
///
/// ```text
/// field     = forward + backward
/// secondary = ξ(backward - forward)
/// ξ         = -iY
/// ```
pub(crate) fn project_integrated_bilinear_cross_state_products<A>(
    products: &IntegratedWaveProducts<A>,
    left_admittance: &A,
    right_admittance: &A,
) -> IntegratedBilinearCrossStateProducts<A>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    let i = <A::Scalar as ComplexScalar>::i();

    let left_slope = left_admittance.scale(-i);

    let right_slope = right_admittance.scale(-i);

    let ff = products.forward_forward();
    let bb = products.backward_backward();
    let fb = products.forward_backward();
    let bf = products.backward_forward();

    let field_field = ff.add(fb).add(bf).add(bb);

    /*
     * (b_l - f_l)* (b_r - f_r)
     *
     * = bb - bf - fb + ff
     */
    let secondary_secondary = bb
        .subtract(bf)
        .subtract(fb)
        .add(ff)
        .multiply(&left_slope.multiply(&right_slope));

    /*
     * (f_l + b_l) ξ_r(b_r - f_r)
     *
     * = ξ_r(fb - ff + bb - bf)
     */
    let field_secondary = fb.subtract(ff).add(bb).subtract(bf).multiply(&right_slope);

    /*
     * ξ_l(b_l - f_l) (f_r + b_r)
     *
     * = ξ_l (bf + bb - ff - fb)
     */
    let secondary_field = bf.add(bb).subtract(ff).subtract(fb).multiply(&left_slope);

    IntegratedBilinearCrossStateProducts::new(
        field_field,
        secondary_secondary,
        field_secondary,
        secondary_field,
    )
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        algebra::{ArrayJet0, Jet, Jet0, RealParameter},
        observable::layer::integration::IntegratedWaveProducts,
    };

    type A = ArrayJet0<Complex64, Ix0, RealParameter>;

    const TOLERANCE: f64 = 1.0e-12;

    fn jet(value: Complex64) -> A {
        Jet0::new(arr0(value))
    }

    fn scalar(value: &A) -> Complex64 {
        value.value()[()]
    }

    fn products() -> IntegratedWaveProducts<A> {
        IntegratedWaveProducts::new(
            jet(Complex64::new(2.0, 1.0)),
            jet(Complex64::new(3.0, -2.0)),
            jet(Complex64::new(5.0, 4.0)),
            jet(Complex64::new(-1.0, 6.0)),
        )
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn bilinear_state_projection_matches_direct_expansion() {
        let products = products();

        let left_admittance = Complex64::new(1.3, -0.4);
        let right_admittance = Complex64::new(0.8, 0.2);

        let state = project_integrated_bilinear_cross_state_products(
            &products,
            &jet(left_admittance),
            &jet(right_admittance),
        );

        let ff = Complex64::new(2.0, 1.0);
        let bb = Complex64::new(3.0, -2.0);
        let fb = Complex64::new(5.0, 4.0);
        let bf = Complex64::new(-1.0, 6.0);

        let i = Complex64::new(0.0, 1.0);
        let left_slope = -i * left_admittance;
        let right_slope = -i * right_admittance;

        let expected_field_field = ff + fb + bf + bb;

        let expected_secondary_secondary = left_slope * right_slope * (bb - bf - fb + ff);

        let expected_field_secondary = right_slope * (fb - ff + bb - bf);

        let expected_secondary_field = left_slope * (bf + bb - ff - fb);

        assert_complex_close(scalar(state.field_field()), expected_field_field);

        assert_complex_close(
            scalar(state.secondary_secondary()),
            expected_secondary_secondary,
        );

        assert_complex_close(scalar(state.field_secondary()), expected_field_secondary);

        assert_complex_close(scalar(state.secondary_field()), expected_secondary_field);
    }

    #[test]
    fn exchanging_operands_preserves_symmetric_products_and_swaps_mixed_products() {
        let products = products();

        let left_admittance = jet(Complex64::new(1.3, -0.4));
        let right_admittance = jet(Complex64::new(0.8, 0.2));

        let left_right = project_integrated_bilinear_cross_state_products(
            &products,
            &left_admittance,
            &right_admittance,
        );

        let swapped_products = IntegratedWaveProducts::new(
            products.forward_forward().clone(),
            products.backward_backward().clone(),
            products.backward_forward().clone(),
            products.forward_backward().clone(),
        );

        let right_left = project_integrated_bilinear_cross_state_products(
            &swapped_products,
            &right_admittance,
            &left_admittance,
        );

        assert_complex_close(
            scalar(left_right.field_field()),
            scalar(right_left.field_field()),
        );

        assert_complex_close(
            scalar(left_right.secondary_secondary()),
            scalar(right_left.secondary_secondary()),
        );

        assert_complex_close(
            scalar(left_right.field_secondary()),
            scalar(right_left.secondary_field()),
        );

        assert_complex_close(
            scalar(left_right.secondary_field()),
            scalar(right_left.field_secondary()),
        );
    }

    #[test]
    fn zero_admittances_zero_all_secondary_products() {
        let state = project_integrated_bilinear_cross_state_products(
            &products(),
            &jet(Complex64::new(0.0, 0.0)),
            &jet(Complex64::new(0.0, 0.0)),
        );

        assert_complex_close(
            scalar(state.secondary_secondary()),
            Complex64::new(0.0, 0.0),
        );

        assert_complex_close(scalar(state.field_secondary()), Complex64::new(0.0, 0.0));

        assert_complex_close(scalar(state.secondary_field()), Complex64::new(0.0, 0.0));
    }

    #[test]
    fn into_parts_preserves_documented_order() {
        let state = IntegratedBilinearCrossStateProducts::new(
            jet(Complex64::new(1.0, 0.0)),
            jet(Complex64::new(2.0, 0.0)),
            jet(Complex64::new(3.0, 0.0)),
            jet(Complex64::new(4.0, 0.0)),
        );

        let (field_field, secondary_secondary, field_secondary, secondary_field) =
            state.into_parts();

        assert_eq!(scalar(&field_field), Complex64::new(1.0, 0.0));
        assert_eq!(scalar(&secondary_secondary), Complex64::new(2.0, 0.0));
        assert_eq!(scalar(&field_secondary), Complex64::new(3.0, 0.0));
        assert_eq!(scalar(&secondary_field), Complex64::new(4.0, 0.0));
    }
}
