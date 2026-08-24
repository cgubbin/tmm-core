//! Analytic integration of directional-wave products through homogeneous
//! finite layers.
//!
//! A directional wave is represented using the convention
//!
//! ```text
//! a(z) = a⁺ exp(+i k z) + a⁻ exp(-i k z),
//! ```
//!
//! where:
//!
//! - `z` is measured from the layer's left boundary;
//! - `k` is the normal wavevector;
//! - `a⁺` and `a⁻` are the forward- and backward-labelled amplitudes at the
//!   left boundary.
//!
//! This module analytically integrates the four products arising from the two
//! directional branches. It does not perform field sampling or apply
//! electromagnetic constitutive weights.

use ndarray::Dimension;

use crate::{
    ComplexScalar,
    algebra::{RealScalarAlgebra, ScalarAlgebra, ScalarAlgebraExpRelExt},
    observable::BoundaryWaves,
};

/// Spatially integrated products of two directional-wave decompositions.
///
/// The four entries correspond to:
///
/// ```text
/// forward_forward
///     = ∫ left_forward(z) · right_forward(z) dz
///
/// backward_backward
///     = ∫ left_backward(z) · right_backward(z) dz
///
/// forward_backward
///     = ∫ left_forward(z) · right_backward(z) dz
///
/// backward_forward
///     = ∫ left_backward(z) · right_forward(z) dz
/// ```
///
/// The interpretation of the left factor depends on the projection:
///
/// - [`integrate_hermitian_wave_products`] conjugates the left factor;
/// - [`integrate_bilinear_wave_products`] does not conjugate either factor.
///
/// The Hermitian cross terms are generally complex, so all four entries retain
/// the original complex algebra representation.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IntegratedWaveProducts<A> {
    forward_forward: A,
    backward_backward: A,
    forward_backward: A,
    backward_forward: A,
}

impl<A> IntegratedWaveProducts<A> {
    pub(crate) const fn new(
        forward_forward: A,
        backward_backward: A,
        forward_backward: A,
        backward_forward: A,
    ) -> Self {
        Self {
            forward_forward,
            backward_backward,
            forward_backward,
            backward_forward,
        }
    }

    pub(crate) fn forward_forward(&self) -> &A {
        &self.forward_forward
    }

    pub(crate) fn backward_backward(&self) -> &A {
        &self.backward_backward
    }

    pub(crate) fn forward_backward(&self) -> &A {
        &self.forward_backward
    }

    pub(crate) fn backward_forward(&self) -> &A {
        &self.backward_forward
    }
}

/// Analytically integrate `exp(alpha * z)` over `0 <= z <= thickness`.
///
/// The integral is evaluated as:
///
/// ```text
/// thickness * exprel(alpha * thickness),
/// ```
///
/// where:
///
/// ```text
/// exprel(x) = (exp(x) - 1) / x
/// ```
///
/// with its analytic continuation at zero. This form remains well-conditioned
/// when `alpha * thickness` is small and gives the exact limiting value
/// `thickness` when `alpha` is zero.
fn integrate_exponential<A>(alpha: &A, thickness: &A) -> A
where
    A: ScalarAlgebra + ScalarAlgebraExpRelExt,
{
    let argument = alpha.multiply(thickness);

    thickness.multiply(&argument.exprel())
}

pub(crate) fn integrate_bilinear_wave_products<A>(
    waves: &BoundaryWaves<A>,
    kappa: &A,
    thickness: &A,
) -> IntegratedWaveProducts<A>
where
    A: ScalarAlgebra + ScalarAlgebraExpRelExt,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    integrate_bilinear_cross_wave_products(waves, waves, kappa, kappa, thickness)
}

/// Analytically integrate Hermitian directional-wave products.
///
/// For:
///
/// ```text
/// f(z) = f exp(+i k z)
/// b(z) = b exp(-i k z),
/// ```
///
/// the returned products are:
///
/// ```text
/// forward_forward   = ∫ f(z)* f(z) dz
/// backward_backward = ∫ b(z)* b(z) dz
/// forward_backward  = ∫ f(z)* b(z) dz
/// backward_forward  = ∫ b(z)* f(z) dz
/// ```
///
/// The amplitudes and normal wavevector are defined at the layer's left
/// boundary.
///
/// For real layer thickness, `backward_forward` is the complex conjugate of
/// `forward_backward`.
pub(crate) fn integrate_hermitian_wave_products<A>(
    waves: &BoundaryWaves<A>,
    kappa: &A,
    thickness: &A,
) -> IntegratedWaveProducts<A>
where
    A: RealScalarAlgebra + ScalarAlgebraExpRelExt,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    integrate_hermitian_cross_wave_products(waves, waves, kappa, kappa, thickness)
}

/// Analytically integrate bilinear directional-wave products.
///
/// No complex conjugation is applied. For:
///
/// ```text
/// l(z) = l⁺ exp(+i k_l z) + l⁻ exp(-i k_l z)
/// r(z) = r⁺ exp(+i k_r z) + r⁻ exp(-i k_r z),
/// ```
///
/// this computes:
///
/// ```text
/// forward_forward   = ∫ l⁺(z) r⁺(z) dz
/// backward_backward = ∫ l⁻(z) r⁻(z) dz
/// forward_backward  = ∫ l⁺(z) r⁻(z) dz
/// backward_forward  = ∫ l⁻(z) r⁺(z) dz.
/// ```
///
/// The function does not define the physical relation between the left and
/// right wave sets. In a modal-normalisation calculation, the caller is
/// responsible for providing the appropriate primal and dual or adjoint
/// fields, propagation branches, and boundary conventions.
pub(crate) fn integrate_bilinear_cross_wave_products<A>(
    left_waves: &BoundaryWaves<A>,
    right_waves: &BoundaryWaves<A>,
    left_normal_wavevector: &A,
    right_normal_wavevector: &A,
    thickness: &A,
) -> IntegratedWaveProducts<A>
where
    A: ScalarAlgebra + ScalarAlgebraExpRelExt,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    let i = <A::Scalar as ComplexScalar>::i();

    let sum = left_normal_wavevector.add(right_normal_wavevector);

    let left_minus_right = left_normal_wavevector.subtract(right_normal_wavevector);

    /*
     * exp(+ik_l z) exp(+ik_r z)
     */
    let forward_forward_exponent = sum.scale(i);

    /*
     * exp(-ik_l z) exp(-ik_r z)
     */
    let backward_backward_exponent = sum.scale(-i);

    /*
     * exp(+ik_l z) exp(-ik_r z)
     */
    let forward_backward_exponent = left_minus_right.scale(i);

    /*
     * exp(-ik_l z) exp(+ik_r z)
     */
    let backward_forward_exponent = left_minus_right.scale(-i);

    let forward_forward = left_waves
        .forward()
        .multiply(right_waves.forward())
        .multiply(&integrate_exponential(&forward_forward_exponent, thickness));

    let backward_backward = left_waves
        .backward()
        .multiply(right_waves.backward())
        .multiply(&integrate_exponential(
            &backward_backward_exponent,
            thickness,
        ));

    let forward_backward = left_waves
        .forward()
        .multiply(right_waves.backward())
        .multiply(&integrate_exponential(
            &forward_backward_exponent,
            thickness,
        ));

    let backward_forward = left_waves
        .backward()
        .multiply(right_waves.forward())
        .multiply(&integrate_exponential(
            &backward_forward_exponent,
            thickness,
        ));

    IntegratedWaveProducts::new(
        forward_forward,
        backward_backward,
        forward_backward,
        backward_forward,
    )
}

/// Analytically integrate Hermitian branch-pair products between two
/// directional-wave solutions.
///
/// The left factor is complex-conjugated:
///
/// ```text
/// forward_forward  = ∫ left_forward*  right_forward  dz
/// backward_backward = ∫ left_backward* right_backward dz
/// forward_backward = ∫ left_forward*  right_backward dz
/// backward_forward = ∫ left_backward* right_forward  dz
/// ```
///
/// The two solutions may have different normal wavevectors. They must refer
/// to the same physical layer and therefore use the same integration
/// thickness.
pub(crate) fn integrate_hermitian_cross_wave_products<A>(
    left_waves: &BoundaryWaves<A>,
    right_waves: &BoundaryWaves<A>,
    left_kappa: &A,
    right_kappa: &A,
    thickness: &A,
) -> IntegratedWaveProducts<A>
where
    A: RealScalarAlgebra + ScalarAlgebraExpRelExt,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    let i = <A::Scalar as ComplexScalar>::i();

    let left_kappa_conjugated = left_kappa.conjugated();

    /*
     * exp(-i left_kappa* z)
     * exp(+i right_kappa z)
     */
    let forward_forward_exponent = right_kappa.subtract(&left_kappa_conjugated).scale(i);

    /*
     * exp(+i left_kappa* z)
     * exp(-i right_kappa z)
     */
    let backward_backward_exponent = left_kappa_conjugated.subtract(right_kappa).scale(i);

    /*
     * exp(-i left_kappa* z)
     * exp(-i right_kappa z)
     */
    let forward_backward_exponent = left_kappa_conjugated.add(right_kappa).scale(-i);

    /*
     * exp(+i left_kappa* z)
     * exp(+i right_kappa z)
     */
    let backward_forward_exponent = left_kappa_conjugated.add(right_kappa).scale(i);

    let forward_forward = left_waves
        .forward()
        .hermitian_product(right_waves.forward())
        .multiply(&integrate_exponential(&forward_forward_exponent, thickness));

    let backward_backward = left_waves
        .backward()
        .hermitian_product(right_waves.backward())
        .multiply(&integrate_exponential(
            &backward_backward_exponent,
            thickness,
        ));

    let forward_backward = left_waves
        .forward()
        .hermitian_product(right_waves.backward())
        .multiply(&integrate_exponential(
            &forward_backward_exponent,
            thickness,
        ));

    let backward_forward = left_waves
        .backward()
        .hermitian_product(right_waves.forward())
        .multiply(&integrate_exponential(
            &backward_forward_exponent,
            thickness,
        ));

    IntegratedWaveProducts::new(
        forward_forward,
        backward_backward,
        forward_backward,
        backward_forward,
    )
}

#[cfg(test)]
mod zero_order_tests {
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        algebra::{ArrayJet0, Jet0, RealParameter},
        observable::BoundaryWaves,
        test_support::assertions::{assert_complex_close, assert_zero_jet_close},
    };

    type C = Complex64;
    type A = ArrayJet0<C, Ix0, RealParameter>;

    const TOLERANCE: f64 = 2.0e-11;
    const QUADRATURE_TOLERANCE: f64 = 2.0e-9;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn jet(value: C) -> A {
        Jet0::new(arr0(value))
    }

    fn complex_jet(re: f64, im: f64) -> A {
        Jet0::new(arr0(c(re, im)))
    }

    fn value(value: &A) -> C {
        value.value()[()]
    }

    fn direct_exponential_integral(alpha: C, thickness: f64) -> C {
        if alpha.norm() < 1.0e-10 {
            let argument = alpha * thickness;

            C::new(thickness, 0.0)
                * (C::new(1.0, 0.0)
                    + argument / 2.0
                    + argument * argument / 6.0
                    + argument * argument * argument / 24.0)
        } else {
            ((alpha * thickness).exp() - C::new(1.0, 0.0)) / alpha
        }
    }

    fn integrate_simpson(function: impl Fn(f64) -> C, start: f64, end: f64, intervals: usize) -> C {
        assert_eq!(
            intervals % 2,
            0,
            "Simpson integration requires an even interval count",
        );

        let step = (end - start) / intervals as f64;

        let mut sum = function(start) + function(end);

        for index in 1..intervals {
            let position = start + index as f64 * step;

            let weight = if index % 2 == 0 { 2.0 } else { 4.0 };

            sum += function(position) * weight;
        }

        sum * step / 3.0
    }

    #[test]
    fn integrated_products_store_all_components() {
        let products = IntegratedWaveProducts::new(1, 2, 3, 4);

        assert_eq!(products.forward_forward(), &1);
        assert_eq!(products.backward_backward(), &2);
        assert_eq!(products.forward_backward(), &3);
        assert_eq!(products.backward_forward(), &4);
    }

    #[test]
    fn zero_thickness_produces_zero_hermitian_products() {
        let waves = BoundaryWaves::new(jet(c(1.0, 0.2)), jet(c(-0.3, 0.4)));

        let products =
            integrate_hermitian_wave_products(&waves, &jet(c(2.0, 0.3)), &jet(c(0.0, 0.0)));

        for product in [
            products.forward_forward(),
            products.backward_backward(),
            products.forward_backward(),
            products.backward_forward(),
        ] {
            assert_complex_close(value(product), c(0.0, 0.0), TOLERANCE);
        }
    }

    #[test]
    fn zero_thickness_produces_zero_bilinear_products() {
        let products = integrate_bilinear_cross_wave_products(
            &BoundaryWaves::new(jet(c(1.0, 0.2)), jet(c(-0.3, 0.4))),
            &BoundaryWaves::new(jet(c(0.6, -0.1)), jet(c(0.2, 0.7))),
            &jet(c(2.0, 0.3)),
            &jet(c(1.7, -0.2)),
            &jet(c(0.0, 0.0)),
        );

        for product in [
            products.forward_forward(),
            products.backward_backward(),
            products.forward_backward(),
            products.backward_forward(),
        ] {
            assert_complex_close(value(product), c(0.0, 0.0), TOLERANCE);
        }
    }

    #[test]
    fn zero_wavevector_integrates_constant_hermitian_products() {
        let forward = c(1.0, 0.5);
        let backward = c(-0.25, 0.75);
        let thickness = 0.8;

        let products = integrate_hermitian_wave_products(
            &BoundaryWaves::new(jet(forward), jet(backward)),
            &jet(c(0.0, 0.0)),
            &jet(c(thickness, 0.0)),
        );

        assert_complex_close(
            value(products.forward_forward()),
            forward.conj() * forward * thickness,
            TOLERANCE,
        );

        assert_complex_close(
            value(products.backward_backward()),
            backward.conj() * backward * thickness,
            TOLERANCE,
        );

        assert_complex_close(
            value(products.forward_backward()),
            forward.conj() * backward * thickness,
            TOLERANCE,
        );

        assert_complex_close(
            value(products.backward_forward()),
            backward.conj() * forward * thickness,
            TOLERANCE,
        );
    }

    #[test]
    fn lossless_diagonal_products_are_norm_squared_times_thickness() {
        let forward = c(0.8, 0.3);
        let backward = c(-0.2, 0.5);
        let thickness = 1.7;

        let products = integrate_hermitian_wave_products(
            &BoundaryWaves::new(jet(forward), jet(backward)),
            &jet(c(2.4, 0.0)),
            &jet(c(thickness, 0.0)),
        );

        assert_complex_close(
            value(products.forward_forward()),
            c(forward.norm_sqr() * thickness, 0.0),
            TOLERANCE,
        );

        assert_complex_close(
            value(products.backward_backward()),
            c(backward.norm_sqr() * thickness, 0.0),
            TOLERANCE,
        );
    }

    #[test]
    fn hermitian_cross_products_are_mutual_conjugates() {
        let products = integrate_hermitian_wave_products(
            &BoundaryWaves::new(jet(c(0.8, 0.3)), jet(c(-0.2, 0.5))),
            &jet(c(2.4, 0.35)),
            &jet(c(1.7, 0.0)),
        );

        assert_complex_close(
            value(products.backward_forward()),
            value(products.forward_backward()).conj(),
            TOLERANCE,
        );
    }

    #[test]
    fn hermitian_products_match_closed_form() {
        let forward = c(0.8, 0.3);
        let backward = c(-0.2, 0.5);
        let wavevector = c(2.4, 0.35);
        let thickness = 1.7;

        let products = integrate_hermitian_wave_products(
            &BoundaryWaves::new(jet(forward), jet(backward)),
            &jet(wavevector),
            &jet(c(thickness, 0.0)),
        );

        let difference = wavevector - wavevector.conj();

        let sum = wavevector + wavevector.conj();

        let integral_forward_forward = direct_exponential_integral(C::i() * difference, thickness);

        let integral_backward_backward =
            direct_exponential_integral(-C::i() * difference, thickness);

        let integral_forward_backward = direct_exponential_integral(-C::i() * sum, thickness);

        let integral_backward_forward = direct_exponential_integral(C::i() * sum, thickness);

        assert_complex_close(
            value(products.forward_forward()),
            forward.conj() * forward * integral_forward_forward,
            TOLERANCE,
        );

        assert_complex_close(
            value(products.backward_backward()),
            backward.conj() * backward * integral_backward_backward,
            TOLERANCE,
        );

        assert_complex_close(
            value(products.forward_backward()),
            forward.conj() * backward * integral_forward_backward,
            TOLERANCE,
        );

        assert_complex_close(
            value(products.backward_forward()),
            backward.conj() * forward * integral_backward_forward,
            TOLERANCE,
        );
    }

    #[test]
    fn bilinear_products_match_closed_form() {
        let left_forward = c(0.7, 0.1);
        let left_backward = c(-0.3, 0.2);

        let right_forward = c(0.5, -0.4);
        let right_backward = c(0.2, 0.6);

        let left_wavevector = c(1.8, 0.2);
        let right_wavevector = c(2.1, -0.1);
        let thickness = 0.9;

        let products = integrate_bilinear_cross_wave_products(
            &BoundaryWaves::new(jet(left_forward), jet(left_backward)),
            &BoundaryWaves::new(jet(right_forward), jet(right_backward)),
            &jet(left_wavevector),
            &jet(right_wavevector),
            &jet(c(thickness, 0.0)),
        );

        let sum = left_wavevector + right_wavevector;

        let difference = left_wavevector - right_wavevector;

        assert_complex_close(
            value(products.forward_forward()),
            left_forward * right_forward * direct_exponential_integral(C::i() * sum, thickness),
            TOLERANCE,
        );

        assert_complex_close(
            value(products.backward_backward()),
            left_backward * right_backward * direct_exponential_integral(-C::i() * sum, thickness),
            TOLERANCE,
        );

        assert_complex_close(
            value(products.forward_backward()),
            left_forward
                * right_backward
                * direct_exponential_integral(C::i() * difference, thickness),
            TOLERANCE,
        );

        assert_complex_close(
            value(products.backward_forward()),
            left_backward
                * right_forward
                * direct_exponential_integral(-C::i() * difference, thickness),
            TOLERANCE,
        );
    }

    #[test]
    fn near_zero_exponent_uses_correct_analytic_limit() {
        let alpha = c(1.0e-12, -2.0e-12);

        let thickness = 0.7;

        let actual = integrate_exponential(&jet(alpha), &jet(c(thickness, 0.0)));

        let argument = alpha * thickness;

        let expected = c(thickness, 0.0)
            * (c(1.0, 0.0)
                + argument / 2.0
                + argument * argument / 6.0
                + argument * argument * argument / 24.0);

        assert_complex_close(value(&actual), expected, 1.0e-14);
    }

    #[test]
    fn hermitian_products_match_numerical_quadrature() {
        let forward = c(0.8, 0.3);
        let backward = c(-0.2, 0.5);
        let wavevector = c(2.4, 0.35);
        let thickness = 1.7;

        let products = integrate_hermitian_wave_products(
            &BoundaryWaves::new(jet(forward), jet(backward)),
            &jet(wavevector),
            &jet(c(thickness, 0.0)),
        );

        let forward_at = |position: f64| forward * (C::i() * wavevector * position).exp();

        let backward_at = |position: f64| backward * (-C::i() * wavevector * position).exp();

        let expected_forward_forward = integrate_simpson(
            |position| forward_at(position).conj() * forward_at(position),
            0.0,
            thickness,
            10_000,
        );

        let expected_backward_backward = integrate_simpson(
            |position| backward_at(position).conj() * backward_at(position),
            0.0,
            thickness,
            10_000,
        );

        let expected_forward_backward = integrate_simpson(
            |position| forward_at(position).conj() * backward_at(position),
            0.0,
            thickness,
            10_000,
        );

        let expected_backward_forward = integrate_simpson(
            |position| backward_at(position).conj() * forward_at(position),
            0.0,
            thickness,
            10_000,
        );

        assert_complex_close(
            value(products.forward_forward()),
            expected_forward_forward,
            QUADRATURE_TOLERANCE,
        );

        assert_complex_close(
            value(products.backward_backward()),
            expected_backward_backward,
            QUADRATURE_TOLERANCE,
        );

        assert_complex_close(
            value(products.forward_backward()),
            expected_forward_backward,
            QUADRATURE_TOLERANCE,
        );

        assert_complex_close(
            value(products.backward_forward()),
            expected_backward_forward,
            QUADRATURE_TOLERANCE,
        );
    }

    #[test]
    fn bilinear_products_match_numerical_quadrature() {
        let left_forward = c(0.7, 0.1);
        let left_backward = c(-0.3, 0.2);

        let right_forward = c(0.5, -0.4);
        let right_backward = c(0.2, 0.6);

        let left_wavevector = c(1.8, 0.2);
        let right_wavevector = c(2.1, -0.1);
        let thickness = 0.9;

        let products = integrate_bilinear_cross_wave_products(
            &BoundaryWaves::new(jet(left_forward), jet(left_backward)),
            &BoundaryWaves::new(jet(right_forward), jet(right_backward)),
            &jet(left_wavevector),
            &jet(right_wavevector),
            &jet(c(thickness, 0.0)),
        );

        let left_forward_at =
            |position: f64| left_forward * (C::i() * left_wavevector * position).exp();

        let left_backward_at =
            |position: f64| left_backward * (-C::i() * left_wavevector * position).exp();

        let right_forward_at =
            |position: f64| right_forward * (C::i() * right_wavevector * position).exp();

        let right_backward_at =
            |position: f64| right_backward * (-C::i() * right_wavevector * position).exp();

        assert_complex_close(
            value(products.forward_forward()),
            integrate_simpson(
                |position| left_forward_at(position) * right_forward_at(position),
                0.0,
                thickness,
                10_000,
            ),
            QUADRATURE_TOLERANCE,
        );

        assert_complex_close(
            value(products.backward_backward()),
            integrate_simpson(
                |position| left_backward_at(position) * right_backward_at(position),
                0.0,
                thickness,
                10_000,
            ),
            QUADRATURE_TOLERANCE,
        );

        assert_complex_close(
            value(products.forward_backward()),
            integrate_simpson(
                |position| left_forward_at(position) * right_backward_at(position),
                0.0,
                thickness,
                10_000,
            ),
            QUADRATURE_TOLERANCE,
        );

        assert_complex_close(
            value(products.backward_forward()),
            integrate_simpson(
                |position| left_backward_at(position) * right_forward_at(position),
                0.0,
                thickness,
                10_000,
            ),
            QUADRATURE_TOLERANCE,
        );
    }

    #[test]
    fn total_hermitian_overlap_is_independent_of_boundary_origin() {
        let left_forward = c(0.8, 0.3);
        let left_backward = c(-0.2, 0.5);
        let wavevector = c(2.4, 0.35);
        let thickness = 1.7;

        let left_products = integrate_hermitian_wave_products(
            &BoundaryWaves::new(jet(left_forward), jet(left_backward)),
            &jet(wavevector),
            &jet(c(thickness, 0.0)),
        );

        let right_forward = left_forward * (C::i() * wavevector * thickness).exp();

        let right_backward = left_backward * (-C::i() * wavevector * thickness).exp();

        /*
         * Reversing the local spatial coordinate exchanges the directional
         * branches.
         */
        let right_products = integrate_hermitian_wave_products(
            &BoundaryWaves::new(jet(right_backward), jet(right_forward)),
            &jet(wavevector),
            &jet(c(thickness, 0.0)),
        );

        let left_total = value(left_products.forward_forward())
            + value(left_products.backward_backward())
            + value(left_products.forward_backward())
            + value(left_products.backward_forward());

        let right_total = value(right_products.forward_forward())
            + value(right_products.backward_backward())
            + value(right_products.forward_backward())
            + value(right_products.backward_forward());

        assert_complex_close(left_total, right_total, QUADRATURE_TOLERANCE);
    }

    #[test]
    fn hermitian_cross_product_reduces_to_self_product() {
        let left_forward = c(0.8, 0.3);
        let left_backward = c(-0.2, 0.5);
        let waves = BoundaryWaves::new(jet(left_forward), jet(left_backward));

        let kappa = complex_jet(1.3, 0.2);
        let thickness = complex_jet(0.7, 0.0);

        let self_product = integrate_hermitian_wave_products(&waves, &kappa, &thickness);

        let cross_product =
            integrate_hermitian_cross_wave_products(&waves, &waves, &kappa, &kappa, &thickness);

        assert_eq!(cross_product, self_product);
    }

    #[test]
    fn swapping_cross_product_operands_conjugates_and_transposes_branches() {
        let left_forward = c(0.7, 0.1);
        let left_backward = c(-0.3, 0.2);

        let right_forward = c(0.5, -0.4);
        let right_backward = c(0.2, 0.6);

        let left = BoundaryWaves::new(jet(left_forward), jet(left_backward));
        let right = BoundaryWaves::new(jet(right_forward), jet(right_backward));

        let left_kappa = complex_jet(1.3, 0.2);
        let right_kappa = complex_jet(0.9, 0.1);
        let thickness = complex_jet(0.7, 0.0);

        let left_right = integrate_hermitian_cross_wave_products(
            &left,
            &right,
            &left_kappa,
            &right_kappa,
            &thickness,
        );

        let right_left = integrate_hermitian_cross_wave_products(
            &right,
            &left,
            &right_kappa,
            &left_kappa,
            &thickness,
        );

        assert_zero_jet_close(
            left_right.forward_forward(),
            &right_left.forward_forward().conjugated(),
        );

        assert_zero_jet_close(
            left_right.backward_backward(),
            &right_left.backward_backward().conjugated(),
        );

        assert_zero_jet_close(
            left_right.forward_backward(),
            &right_left.backward_forward().conjugated(),
        );

        assert_zero_jet_close(
            left_right.backward_forward(),
            &right_left.forward_backward().conjugated(),
        );
    }
}

#[cfg(test)]
mod derivative_tests {
    use ndarray::{Ix0, arr0};

    use super::*;

    use crate::{
        algebra::{
            ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, Jet0,
            RealParameter,
        },
        differential::{BivariateGradient, BivariateHessian},
        test_support::{C, TOLERANCE, assertions::assert_complex_close},
    };

    type A = ArrayJet0<C, Ix0, RealParameter>;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn jet(value: C) -> A {
        Jet0::new(arr0(value))
    }

    fn value(value: &A) -> C {
        value.value()[()]
    }

    type A1 = ArrayJet1<C, Ix0, RealParameter>;
    type A2 = ArrayJet2<C, Ix0, RealParameter>;
    type AB1 = ArrayJetBivariate1<C, Ix0, RealParameter>;
    type AB2 = ArrayJetBivariate2<C, Ix0, RealParameter>;

    const FIRST_DERIVATIVE_TOLERANCE: f64 = 2.0e-8;
    const SECOND_DERIVATIVE_TOLERANCE: f64 = 2.0e-5;

    fn jet1(value: C, first: C) -> A1 {
        A1::from_parts(arr0(value), arr0(first))
    }

    fn jet2(value: C, first: C, second: C) -> A2 {
        A2::from_parts(arr0(value), arr0(first), arr0(second))
    }

    fn bivariate1(value: C, axis0: C, axis1: C) -> AB1 {
        AB1::from_parts(
            arr0(value),
            BivariateGradient::new(arr0(axis0), arr0(axis1)),
        )
    }

    fn bivariate2(
        value: C,
        axis0: C,
        axis1: C,
        axis0_axis0: C,
        axis0_axis1: C,
        axis1_axis1: C,
    ) -> AB2 {
        AB2::from_parts(
            arr0(value),
            BivariateGradient::new(arr0(axis0), arr0(axis1)),
            BivariateHessian::new(arr0(axis0_axis0), arr0(axis0_axis1), arr0(axis1_axis1)),
        )
    }

    fn scalar1_first(value: &A1) -> C {
        value.first()[()]
    }

    fn scalar2_value(value: &A2) -> C {
        value.value()[()]
    }

    fn scalar2_first(value: &A2) -> C {
        value.first()[()]
    }

    fn scalar2_second(value: &A2) -> C {
        value.second()[()]
    }

    fn scalar_b1_axis0(value: &AB1) -> C {
        value.axis0()[()]
    }

    fn scalar_b1_axis1(value: &AB1) -> C {
        value.axis1()[()]
    }

    fn scalar_b2_axis0_axis0(value: &AB2) -> C {
        value.axis0_axis0()[()]
    }

    fn scalar_b2_axis0_axis1(value: &AB2) -> C {
        value.axis0_axis1()[()]
    }

    fn scalar_b2_axis1_axis1(value: &AB2) -> C {
        value.axis1_axis1()[()]
    }

    fn hermitian_value_products(forward: C, backward: C, wavevector: C, thickness: f64) -> [C; 4] {
        let products = integrate_hermitian_wave_products(
            &BoundaryWaves::new(jet(forward), jet(backward)),
            &jet(wavevector),
            &jet(c(thickness, 0.0)),
        );

        [
            value(products.forward_forward()),
            value(products.backward_backward()),
            value(products.forward_backward()),
            value(products.backward_forward()),
        ]
    }

    fn bilinear_value_products(
        left_forward: C,
        left_backward: C,
        right_forward: C,
        right_backward: C,
        left_wavevector: C,
        right_wavevector: C,
        thickness: C,
    ) -> [C; 4] {
        let products = integrate_bilinear_cross_wave_products(
            &BoundaryWaves::new(jet(left_forward), jet(left_backward)),
            &BoundaryWaves::new(jet(right_forward), jet(right_backward)),
            &jet(left_wavevector),
            &jet(right_wavevector),
            &jet(thickness),
        );

        [
            value(products.forward_forward()),
            value(products.backward_backward()),
            value(products.forward_backward()),
            value(products.backward_forward()),
        ]
    }

    fn assert_product_arrays_close(actual: [C; 4], expected: [C; 4], tolerance: f64) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_complex_close(actual, expected, tolerance);
        }
    }

    fn direct_exponential_integral(alpha: C, thickness: f64) -> C {
        if alpha.norm() < 1.0e-10 {
            let argument = alpha * thickness;

            C::new(thickness, 0.0)
                * (C::new(1.0, 0.0)
                    + argument / 2.0
                    + argument * argument / 6.0
                    + argument * argument * argument / 24.0)
        } else {
            ((alpha * thickness).exp() - C::new(1.0, 0.0)) / alpha
        }
    }

    #[test]
    fn exponential_integral_first_derivative_wrt_exponent_matches_finite_difference() {
        let alpha = c(0.3, -0.2);
        let thickness = c(0.8, 0.0);

        let result =
            integrate_exponential(&jet1(alpha, c(1.0, 0.0)), &jet1(thickness, c(0.0, 0.0)));

        let step = 1.0e-6;

        let plus = direct_exponential_integral(alpha + c(step, 0.0), thickness.re);

        let minus = direct_exponential_integral(alpha - c(step, 0.0), thickness.re);

        let expected = (plus - minus) / (2.0 * step);

        assert_complex_close(scalar1_first(&result), expected, FIRST_DERIVATIVE_TOLERANCE);
    }

    #[test]
    fn exponential_integral_first_derivative_wrt_thickness_matches_integrand_at_endpoint() {
        let alpha = c(0.3, -0.2);
        let thickness = c(0.8, 0.0);

        let result =
            integrate_exponential(&jet1(alpha, c(0.0, 0.0)), &jet1(thickness, c(1.0, 0.0)));

        let expected = (alpha * thickness).exp();

        assert_complex_close(scalar1_first(&result), expected, FIRST_DERIVATIVE_TOLERANCE);
    }

    #[test]
    fn exponential_integral_second_derivative_wrt_exponent_matches_finite_difference() {
        let alpha = c(0.3, -0.2);
        let thickness = c(0.8, 0.0);

        let result = integrate_exponential(
            &jet2(alpha, c(1.0, 0.0), c(0.0, 0.0)),
            &jet2(thickness, c(0.0, 0.0), c(0.0, 0.0)),
        );

        let step = 2.0e-4;

        let plus = direct_exponential_integral(alpha + c(step, 0.0), thickness.re);

        let centre = direct_exponential_integral(alpha, thickness.re);

        let minus = direct_exponential_integral(alpha - c(step, 0.0), thickness.re);

        let expected = (plus - 2.0 * centre + minus) / step.powi(2);

        assert_complex_close(
            scalar2_second(&result),
            expected,
            SECOND_DERIVATIVE_TOLERANCE,
        );
    }

    #[test]
    fn exponential_integral_second_derivative_wrt_thickness_is_alpha_times_endpoint_integrand() {
        let alpha = c(0.3, -0.2);
        let thickness = c(0.8, 0.0);

        let result = integrate_exponential(
            &jet2(alpha, c(0.0, 0.0), c(0.0, 0.0)),
            &jet2(thickness, c(1.0, 0.0), c(0.0, 0.0)),
        );

        let expected = alpha * (alpha * thickness).exp();

        assert_complex_close(
            scalar2_second(&result),
            expected,
            SECOND_DERIVATIVE_TOLERANCE,
        );
    }

    #[test]
    fn exponential_integral_mixed_derivative_matches_closed_form() {
        let alpha = c(0.3, -0.2);
        let thickness = c(0.8, 0.0);

        let result = integrate_exponential(
            &bivariate2(
                alpha,
                c(1.0, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
            ),
            &bivariate2(
                thickness,
                c(0.0, 0.0),
                c(1.0, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
            ),
        );

        /*
         * d/dd I(alpha, d) = exp(alpha d)
         *
         * Therefore:
         *
         * d²/(d alpha dd) I = d exp(alpha d).
         */
        let expected = thickness * (alpha * thickness).exp();

        assert_complex_close(
            scalar_b2_axis0_axis1(&result),
            expected,
            SECOND_DERIVATIVE_TOLERANCE,
        );
    }

    #[test]
    fn exponential_integral_derivatives_have_correct_zero_exponent_limits() {
        let thickness = 0.8;

        let result = integrate_exponential(
            &jet2(c(0.0, 0.0), c(1.0, 0.0), c(0.0, 0.0)),
            &jet2(c(thickness, 0.0), c(0.0, 0.0), c(0.0, 0.0)),
        );

        /*
         * I(alpha, d)
         *   = d + alpha d²/2 + alpha² d³/6 + ...
         *
         * I_alpha(0, d)       = d²/2
         * I_alpha_alpha(0, d) = d³/3
         */
        assert_complex_close(scalar2_value(&result), c(thickness, 0.0), TOLERANCE);

        assert_complex_close(
            scalar2_first(&result),
            c(thickness.powi(2) / 2.0, 0.0),
            TOLERANCE,
        );

        assert_complex_close(
            scalar2_second(&result),
            c(thickness.powi(3) / 3.0, 0.0),
            TOLERANCE,
        );
    }

    #[test]
    fn hermitian_overlap_first_derivative_matches_real_parameter_finite_difference() {
        let forward = c(0.8, 0.3);
        let backward = c(-0.2, 0.5);
        let wavevector = c(2.4, 0.35);
        let thickness = 1.7;

        let forward_first = c(0.12, -0.07);
        let backward_first = c(-0.04, 0.09);
        let wavevector_first = c(0.08, -0.03);
        let thickness_first = 0.11;

        let products = integrate_hermitian_wave_products(
            &BoundaryWaves::new(jet1(forward, forward_first), jet1(backward, backward_first)),
            &jet1(wavevector, wavevector_first),
            &jet1(c(thickness, 0.0), c(thickness_first, 0.0)),
        );

        let analytic = [
            scalar1_first(products.forward_forward()),
            scalar1_first(products.backward_backward()),
            scalar1_first(products.forward_backward()),
            scalar1_first(products.backward_forward()),
        ];

        let evaluate = |parameter: f64| {
            hermitian_value_products(
                forward + forward_first * parameter,
                backward + backward_first * parameter,
                wavevector + wavevector_first * parameter,
                thickness + thickness_first * parameter,
            )
        };

        let step = 1.0e-6;
        let plus = evaluate(step);
        let minus = evaluate(-step);

        let expected = std::array::from_fn(|index| (plus[index] - minus[index]) / (2.0 * step));

        assert_product_arrays_close(analytic, expected, FIRST_DERIVATIVE_TOLERANCE);
    }

    #[test]
    fn hermitian_overlap_second_derivative_matches_real_parameter_finite_difference() {
        let forward = c(0.8, 0.3);
        let backward = c(-0.2, 0.5);
        let wavevector = c(2.4, 0.35);
        let thickness = 1.7;

        let forward_first = c(0.12, -0.07);
        let backward_first = c(-0.04, 0.09);
        let wavevector_first = c(0.08, -0.03);
        let thickness_first = 0.11;

        let products = integrate_hermitian_wave_products(
            &BoundaryWaves::new(
                jet2(forward, forward_first, c(0.0, 0.0)),
                jet2(backward, backward_first, c(0.0, 0.0)),
            ),
            &jet2(wavevector, wavevector_first, c(0.0, 0.0)),
            &jet2(c(thickness, 0.0), c(thickness_first, 0.0), c(0.0, 0.0)),
        );

        let analytic = [
            scalar2_second(products.forward_forward()),
            scalar2_second(products.backward_backward()),
            scalar2_second(products.forward_backward()),
            scalar2_second(products.backward_forward()),
        ];

        let evaluate = |parameter: f64| {
            hermitian_value_products(
                forward + forward_first * parameter,
                backward + backward_first * parameter,
                wavevector + wavevector_first * parameter,
                thickness + thickness_first * parameter,
            )
        };

        let step = 2.0e-4;

        let plus = evaluate(step);
        let centre = evaluate(0.0);
        let minus = evaluate(-step);

        let expected = std::array::from_fn(|index| {
            (plus[index] - 2.0 * centre[index] + minus[index]) / step.powi(2)
        });

        assert_product_arrays_close(analytic, expected, SECOND_DERIVATIVE_TOLERANCE);
    }

    #[test]
    fn hermitian_overlap_cross_terms_remain_conjugate_through_second_order() {
        let products = integrate_hermitian_wave_products(
            &BoundaryWaves::new(
                jet2(c(0.8, 0.3), c(0.12, -0.07), c(-0.02, 0.04)),
                jet2(c(-0.2, 0.5), c(-0.04, 0.09), c(0.03, -0.01)),
            ),
            &jet2(c(2.4, 0.35), c(0.08, -0.03), c(0.01, 0.02)),
            &jet2(c(1.7, 0.0), c(0.11, 0.0), c(-0.03, 0.0)),
        );

        assert_complex_close(
            scalar2_value(products.backward_forward()),
            scalar2_value(products.forward_backward()).conj(),
            TOLERANCE,
        );

        assert_complex_close(
            scalar2_first(products.backward_forward()),
            scalar2_first(products.forward_backward()).conj(),
            FIRST_DERIVATIVE_TOLERANCE,
        );

        assert_complex_close(
            scalar2_second(products.backward_forward()),
            scalar2_second(products.forward_backward()).conj(),
            SECOND_DERIVATIVE_TOLERANCE,
        );
    }

    #[test]
    fn bilinear_overlap_first_derivative_matches_finite_difference() {
        let left_forward = c(0.7, 0.1);
        let left_backward = c(-0.3, 0.2);
        let right_forward = c(0.5, -0.4);
        let right_backward = c(0.2, 0.6);

        let left_wavevector = c(1.8, 0.2);
        let right_wavevector = c(2.1, -0.1);
        let thickness = c(0.9, 0.0);

        let left_forward_first = c(0.03, -0.02);
        let left_backward_first = c(-0.04, 0.01);
        let right_forward_first = c(0.02, 0.05);
        let right_backward_first = c(-0.01, 0.04);

        let left_wavevector_first = c(0.06, -0.03);

        let right_wavevector_first = c(-0.02, 0.04);

        let thickness_first = c(0.08, 0.0);

        let products = integrate_bilinear_cross_wave_products(
            &BoundaryWaves::new(
                jet1(left_forward, left_forward_first),
                jet1(left_backward, left_backward_first),
            ),
            &BoundaryWaves::new(
                jet1(right_forward, right_forward_first),
                jet1(right_backward, right_backward_first),
            ),
            &jet1(left_wavevector, left_wavevector_first),
            &jet1(right_wavevector, right_wavevector_first),
            &jet1(thickness, thickness_first),
        );

        let analytic = [
            scalar1_first(products.forward_forward()),
            scalar1_first(products.backward_backward()),
            scalar1_first(products.forward_backward()),
            scalar1_first(products.backward_forward()),
        ];

        let evaluate = |parameter: f64| {
            bilinear_value_products(
                left_forward + left_forward_first * parameter,
                left_backward + left_backward_first * parameter,
                right_forward + right_forward_first * parameter,
                right_backward + right_backward_first * parameter,
                left_wavevector + left_wavevector_first * parameter,
                right_wavevector + right_wavevector_first * parameter,
                thickness + thickness_first * parameter,
            )
        };

        let step = 1.0e-6;
        let plus = evaluate(step);
        let minus = evaluate(-step);

        let expected = std::array::from_fn(|index| (plus[index] - minus[index]) / (2.0 * step));

        assert_product_arrays_close(analytic, expected, FIRST_DERIVATIVE_TOLERANCE);
    }

    #[test]
    fn bilinear_overlap_second_derivative_matches_finite_difference() {
        let left_forward = c(0.7, 0.1);
        let left_backward = c(-0.3, 0.2);
        let right_forward = c(0.5, -0.4);
        let right_backward = c(0.2, 0.6);

        let left_wavevector = c(1.8, 0.2);
        let right_wavevector = c(2.1, -0.1);
        let thickness = c(0.9, 0.0);

        let left_forward_first = c(0.03, -0.02);
        let left_backward_first = c(-0.04, 0.01);
        let right_forward_first = c(0.02, 0.05);
        let right_backward_first = c(-0.01, 0.04);

        let left_wavevector_first = c(0.06, -0.03);

        let right_wavevector_first = c(-0.02, 0.04);

        let thickness_first = c(0.08, 0.0);

        let products = integrate_bilinear_cross_wave_products(
            &BoundaryWaves::new(
                jet2(left_forward, left_forward_first, c(0.0, 0.0)),
                jet2(left_backward, left_backward_first, c(0.0, 0.0)),
            ),
            &BoundaryWaves::new(
                jet2(right_forward, right_forward_first, c(0.0, 0.0)),
                jet2(right_backward, right_backward_first, c(0.0, 0.0)),
            ),
            &jet2(left_wavevector, left_wavevector_first, c(0.0, 0.0)),
            &jet2(right_wavevector, right_wavevector_first, c(0.0, 0.0)),
            &jet2(thickness, thickness_first, c(0.0, 0.0)),
        );

        let analytic = [
            scalar2_second(products.forward_forward()),
            scalar2_second(products.backward_backward()),
            scalar2_second(products.forward_backward()),
            scalar2_second(products.backward_forward()),
        ];

        let evaluate = |parameter: f64| {
            bilinear_value_products(
                left_forward + left_forward_first * parameter,
                left_backward + left_backward_first * parameter,
                right_forward + right_forward_first * parameter,
                right_backward + right_backward_first * parameter,
                left_wavevector + left_wavevector_first * parameter,
                right_wavevector + right_wavevector_first * parameter,
                thickness + thickness_first * parameter,
            )
        };

        let step = 2.0e-4;
        let plus = evaluate(step);
        let centre = evaluate(0.0);
        let minus = evaluate(-step);

        let expected = std::array::from_fn(|index| {
            (plus[index] - 2.0 * centre[index] + minus[index]) / step.powi(2)
        });

        assert_product_arrays_close(analytic, expected, SECOND_DERIVATIVE_TOLERANCE);
    }

    #[test]
    fn hermitian_overlap_bivariate_gradient_matches_independent_finite_differences() {
        let forward = c(0.8, 0.3);
        let backward = c(-0.2, 0.5);
        let wavevector = c(2.4, 0.35);
        let thickness = 1.7;

        /*
         * Axis 0 changes amplitudes and wavevector.
         * Axis 1 changes thickness and amplitudes differently.
         */
        let forward_axis0 = c(0.12, -0.07);
        let backward_axis0 = c(-0.04, 0.09);
        let wavevector_axis0 = c(0.08, -0.03);

        let forward_axis1 = c(-0.03, 0.06);
        let backward_axis1 = c(0.05, -0.02);
        let thickness_axis1 = 0.11;

        let products = integrate_hermitian_wave_products(
            &BoundaryWaves::new(
                bivariate1(forward, forward_axis0, forward_axis1),
                bivariate1(backward, backward_axis0, backward_axis1),
            ),
            &bivariate1(wavevector, wavevector_axis0, c(0.0, 0.0)),
            &bivariate1(c(thickness, 0.0), c(0.0, 0.0), c(thickness_axis1, 0.0)),
        );

        let actual_axis0 = [
            scalar_b1_axis0(products.forward_forward()),
            scalar_b1_axis0(products.backward_backward()),
            scalar_b1_axis0(products.forward_backward()),
            scalar_b1_axis0(products.backward_forward()),
        ];

        let actual_axis1 = [
            scalar_b1_axis1(products.forward_forward()),
            scalar_b1_axis1(products.backward_backward()),
            scalar_b1_axis1(products.forward_backward()),
            scalar_b1_axis1(products.backward_forward()),
        ];

        let evaluate = |axis0: f64, axis1: f64| {
            hermitian_value_products(
                forward + forward_axis0 * axis0 + forward_axis1 * axis1,
                backward + backward_axis0 * axis0 + backward_axis1 * axis1,
                wavevector + wavevector_axis0 * axis0,
                thickness + thickness_axis1 * axis1,
            )
        };

        let step = 1.0e-6;

        let axis0_plus = evaluate(step, 0.0);
        let axis0_minus = evaluate(-step, 0.0);

        let expected_axis0 =
            std::array::from_fn(|index| (axis0_plus[index] - axis0_minus[index]) / (2.0 * step));

        let axis1_plus = evaluate(0.0, step);
        let axis1_minus = evaluate(0.0, -step);

        let expected_axis1 =
            std::array::from_fn(|index| (axis1_plus[index] - axis1_minus[index]) / (2.0 * step));

        assert_product_arrays_close(actual_axis0, expected_axis0, FIRST_DERIVATIVE_TOLERANCE);

        assert_product_arrays_close(actual_axis1, expected_axis1, FIRST_DERIVATIVE_TOLERANCE);
    }

    #[test]
    fn bilinear_overlap_bivariate_hessian_matches_finite_differences() {
        let left_forward = c(0.7, 0.1);
        let left_backward = c(-0.3, 0.2);
        let right_forward = c(0.5, -0.4);
        let right_backward = c(0.2, 0.6);

        let left_wavevector = c(1.8, 0.2);
        let right_wavevector = c(2.1, -0.1);
        let thickness = c(0.9, 0.0);

        let left_forward_axis0 = c(0.03, -0.02);
        let left_backward_axis0 = c(-0.04, 0.01);
        let left_wavevector_axis0 = c(0.06, -0.03);

        let right_forward_axis1 = c(0.02, 0.05);
        let right_backward_axis1 = c(-0.01, 0.04);
        let right_wavevector_axis1 = c(-0.02, 0.04);
        let thickness_axis1 = c(0.08, 0.0);

        let products = integrate_bilinear_cross_wave_products(
            &BoundaryWaves::new(
                bivariate2(
                    left_forward,
                    left_forward_axis0,
                    c(0.0, 0.0),
                    c(0.0, 0.0),
                    c(0.0, 0.0),
                    c(0.0, 0.0),
                ),
                bivariate2(
                    left_backward,
                    left_backward_axis0,
                    c(0.0, 0.0),
                    c(0.0, 0.0),
                    c(0.0, 0.0),
                    c(0.0, 0.0),
                ),
            ),
            &BoundaryWaves::new(
                bivariate2(
                    right_forward,
                    c(0.0, 0.0),
                    right_forward_axis1,
                    c(0.0, 0.0),
                    c(0.0, 0.0),
                    c(0.0, 0.0),
                ),
                bivariate2(
                    right_backward,
                    c(0.0, 0.0),
                    right_backward_axis1,
                    c(0.0, 0.0),
                    c(0.0, 0.0),
                    c(0.0, 0.0),
                ),
            ),
            &bivariate2(
                left_wavevector,
                left_wavevector_axis0,
                c(0.0, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
            ),
            &bivariate2(
                right_wavevector,
                c(0.0, 0.0),
                right_wavevector_axis1,
                c(0.0, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
            ),
            &bivariate2(
                thickness,
                c(0.0, 0.0),
                thickness_axis1,
                c(0.0, 0.0),
                c(0.0, 0.0),
                c(0.0, 0.0),
            ),
        );

        let actual_axis0_axis0 = [
            scalar_b2_axis0_axis0(products.forward_forward()),
            scalar_b2_axis0_axis0(products.backward_backward()),
            scalar_b2_axis0_axis0(products.forward_backward()),
            scalar_b2_axis0_axis0(products.backward_forward()),
        ];

        let actual_axis0_axis1 = [
            scalar_b2_axis0_axis1(products.forward_forward()),
            scalar_b2_axis0_axis1(products.backward_backward()),
            scalar_b2_axis0_axis1(products.forward_backward()),
            scalar_b2_axis0_axis1(products.backward_forward()),
        ];

        let actual_axis1_axis1 = [
            scalar_b2_axis1_axis1(products.forward_forward()),
            scalar_b2_axis1_axis1(products.backward_backward()),
            scalar_b2_axis1_axis1(products.forward_backward()),
            scalar_b2_axis1_axis1(products.backward_forward()),
        ];

        let evaluate = |axis0: f64, axis1: f64| {
            bilinear_value_products(
                left_forward + left_forward_axis0 * axis0,
                left_backward + left_backward_axis0 * axis0,
                right_forward + right_forward_axis1 * axis1,
                right_backward + right_backward_axis1 * axis1,
                left_wavevector + left_wavevector_axis0 * axis0,
                right_wavevector + right_wavevector_axis1 * axis1,
                thickness + thickness_axis1 * axis1,
            )
        };

        let step = 2.0e-4;

        let centre = evaluate(0.0, 0.0);

        let axis0_plus = evaluate(step, 0.0);
        let axis0_minus = evaluate(-step, 0.0);

        let expected_axis0_axis0 = std::array::from_fn(|index| {
            (axis0_plus[index] - 2.0 * centre[index] + axis0_minus[index]) / step.powi(2)
        });

        let axis1_plus = evaluate(0.0, step);
        let axis1_minus = evaluate(0.0, -step);

        let expected_axis1_axis1 = std::array::from_fn(|index| {
            (axis1_plus[index] - 2.0 * centre[index] + axis1_minus[index]) / step.powi(2)
        });

        let plus_plus = evaluate(step, step);
        let plus_minus = evaluate(step, -step);
        let minus_plus = evaluate(-step, step);
        let minus_minus = evaluate(-step, -step);

        let expected_axis0_axis1 = std::array::from_fn(|index| {
            (plus_plus[index] - plus_minus[index] - minus_plus[index] + minus_minus[index])
                / (4.0 * step.powi(2))
        });

        assert_product_arrays_close(
            actual_axis0_axis0,
            expected_axis0_axis0,
            SECOND_DERIVATIVE_TOLERANCE,
        );

        assert_product_arrays_close(
            actual_axis0_axis1,
            expected_axis0_axis1,
            SECOND_DERIVATIVE_TOLERANCE,
        );

        assert_product_arrays_close(
            actual_axis1_axis1,
            expected_axis1_axis1,
            SECOND_DERIVATIVE_TOLERANCE,
        );
    }

    #[test]
    fn near_zero_phase_overlap_derivatives_remain_finite_and_match_finite_difference() {
        let forward = c(0.8, 0.3);
        let backward = c(-0.2, 0.5);
        let wavevector = c(1.0e-11, -2.0e-11);
        let thickness = 0.7;

        let wavevector_first = c(3.0e-12, 1.0e-12);

        let products = integrate_hermitian_wave_products(
            &BoundaryWaves::new(jet1(forward, c(0.0, 0.0)), jet1(backward, c(0.0, 0.0))),
            &jet1(wavevector, wavevector_first),
            &jet1(c(thickness, 0.0), c(0.0, 0.0)),
        );

        let analytic = [
            scalar1_first(products.forward_forward()),
            scalar1_first(products.backward_backward()),
            scalar1_first(products.forward_backward()),
            scalar1_first(products.backward_forward()),
        ];

        for value in analytic {
            assert!(value.re.is_finite());
            assert!(value.im.is_finite());
        }

        let step = 1.0e-4;

        let plus = hermitian_value_products(
            forward,
            backward,
            wavevector + wavevector_first * step,
            thickness,
        );

        let minus = hermitian_value_products(
            forward,
            backward,
            wavevector - wavevector_first * step,
            thickness,
        );

        let expected = std::array::from_fn(|index| (plus[index] - minus[index]) / (2.0 * step));

        assert_product_arrays_close(analytic, expected, 1.0e-10);
    }
}

#[cfg(test)]
mod bilinear_tests {

    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        algebra::{ArrayJet0, Jet0, RealParameter},
        observable::BoundaryWaves,
    };

    type A = ArrayJet0<Complex64, Ix0, RealParameter>;

    const TOLERANCE: f64 = 1.0e-11;

    fn jet(value: Complex64) -> A {
        Jet0::new(arr0(value))
    }

    fn scalar(value: &A) -> Complex64 {
        value.value()[()]
    }

    fn waves(forward: Complex64, backward: Complex64) -> BoundaryWaves<A> {
        BoundaryWaves::new(jet(forward), jet(backward))
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64) {
        crate::test_support::assertions::assert_complex_close(actual, expected, TOLERANCE)
    }

    fn exponential_integral(exponent: Complex64, thickness: f64) -> Complex64 {
        if exponent.norm() < 1.0e-12 {
            Complex64::new(thickness, 0.0)
        } else {
            ((exponent * thickness).exp() - Complex64::new(1.0, 0.0)) / exponent
        }
    }

    #[test]
    fn bilinear_wave_products_vanish_for_zero_thickness() {
        let left = waves(Complex64::new(2.0, 1.0), Complex64::new(3.0, -2.0));

        let right = waves(Complex64::new(-1.0, 4.0), Complex64::new(5.0, 2.0));

        let products = integrate_bilinear_cross_wave_products(
            &left,
            &right,
            &jet(Complex64::new(0.7, 0.2)),
            &jet(Complex64::new(0.4, -0.1)),
            &jet(Complex64::new(0.0, 0.0)),
        );

        assert_complex_close(scalar(products.forward_forward()), Complex64::new(0.0, 0.0));

        assert_complex_close(
            scalar(products.backward_backward()),
            Complex64::new(0.0, 0.0),
        );

        assert_complex_close(
            scalar(products.forward_backward()),
            Complex64::new(0.0, 0.0),
        );

        assert_complex_close(
            scalar(products.backward_forward()),
            Complex64::new(0.0, 0.0),
        );
    }

    #[test]
    fn bilinear_wave_products_reduce_to_amplitude_products_for_zero_wavevectors() {
        let left = waves(Complex64::new(2.0, 1.0), Complex64::new(3.0, -2.0));

        let right = waves(Complex64::new(-1.0, 4.0), Complex64::new(5.0, 2.0));

        let thickness = 0.8;

        let products = integrate_bilinear_cross_wave_products(
            &left,
            &right,
            &jet(Complex64::new(0.0, 0.0)),
            &jet(Complex64::new(0.0, 0.0)),
            &jet(Complex64::new(thickness, 0.0)),
        );

        assert_complex_close(
            scalar(products.forward_forward()),
            Complex64::new(2.0, 1.0) * Complex64::new(-1.0, 4.0) * thickness,
        );

        assert_complex_close(
            scalar(products.backward_backward()),
            Complex64::new(3.0, -2.0) * Complex64::new(5.0, 2.0) * thickness,
        );

        assert_complex_close(
            scalar(products.forward_backward()),
            Complex64::new(2.0, 1.0) * Complex64::new(5.0, 2.0) * thickness,
        );

        assert_complex_close(
            scalar(products.backward_forward()),
            Complex64::new(3.0, -2.0) * Complex64::new(-1.0, 4.0) * thickness,
        );
    }

    #[test]
    fn equal_wavevectors_make_mixed_products_constant() {
        let left = waves(Complex64::new(2.0, 1.0), Complex64::new(3.0, -2.0));

        let right = waves(Complex64::new(-1.0, 4.0), Complex64::new(5.0, 2.0));

        let kappa = Complex64::new(0.7, 0.2);
        let thickness = 0.8;

        let products = integrate_bilinear_cross_wave_products(
            &left,
            &right,
            &jet(kappa),
            &jet(kappa),
            &jet(Complex64::new(thickness, 0.0)),
        );

        assert_complex_close(
            scalar(products.forward_backward()),
            Complex64::new(2.0, 1.0) * Complex64::new(5.0, 2.0) * thickness,
        );

        assert_complex_close(
            scalar(products.backward_forward()),
            Complex64::new(3.0, -2.0) * Complex64::new(-1.0, 4.0) * thickness,
        );
    }

    #[test]
    fn opposite_wavevectors_make_same_direction_products_constant() {
        let left = waves(Complex64::new(2.0, 1.0), Complex64::new(3.0, -2.0));

        let right = waves(Complex64::new(-1.0, 4.0), Complex64::new(5.0, 2.0));

        let left_kappa = Complex64::new(0.7, 0.2);
        let right_kappa = -left_kappa;
        let thickness = 0.8;

        let products = integrate_bilinear_cross_wave_products(
            &left,
            &right,
            &jet(left_kappa),
            &jet(right_kappa),
            &jet(Complex64::new(thickness, 0.0)),
        );

        assert_complex_close(
            scalar(products.forward_forward()),
            Complex64::new(2.0, 1.0) * Complex64::new(-1.0, 4.0) * thickness,
        );

        assert_complex_close(
            scalar(products.backward_backward()),
            Complex64::new(3.0, -2.0) * Complex64::new(5.0, 2.0) * thickness,
        );
    }

    #[test]
    fn bilinear_wave_products_match_direct_exponential_integrals() {
        let lf = Complex64::new(2.0, 1.0);
        let lb = Complex64::new(3.0, -2.0);

        let rf = Complex64::new(-1.0, 4.0);
        let rb = Complex64::new(5.0, 2.0);

        let left_kappa = Complex64::new(0.7, 0.2);
        let right_kappa = Complex64::new(0.4, -0.1);
        let thickness = 0.8;

        let products = integrate_bilinear_cross_wave_products(
            &waves(lf, lb),
            &waves(rf, rb),
            &jet(left_kappa),
            &jet(right_kappa),
            &jet(Complex64::new(thickness, 0.0)),
        );

        let i = Complex64::new(0.0, 1.0);

        assert_complex_close(
            scalar(products.forward_forward()),
            lf * rf * exponential_integral(i * (left_kappa + right_kappa), thickness),
        );

        assert_complex_close(
            scalar(products.backward_backward()),
            lb * rb * exponential_integral(-i * (left_kappa + right_kappa), thickness),
        );

        assert_complex_close(
            scalar(products.forward_backward()),
            lf * rb * exponential_integral(i * (left_kappa - right_kappa), thickness),
        );

        assert_complex_close(
            scalar(products.backward_forward()),
            lb * rf * exponential_integral(-i * (left_kappa - right_kappa), thickness),
        );
    }

    #[test]
    fn exchanging_operands_transposes_mixed_wave_products() {
        let left = waves(Complex64::new(2.0, 1.0), Complex64::new(3.0, -2.0));

        let right = waves(Complex64::new(-1.0, 4.0), Complex64::new(5.0, 2.0));

        let left_kappa = jet(Complex64::new(0.7, 0.2));
        let right_kappa = jet(Complex64::new(0.4, -0.1));
        let thickness = jet(Complex64::new(0.8, 0.0));

        let left_right = integrate_bilinear_cross_wave_products(
            &left,
            &right,
            &left_kappa,
            &right_kappa,
            &thickness,
        );

        let right_left = integrate_bilinear_cross_wave_products(
            &right,
            &left,
            &right_kappa,
            &left_kappa,
            &thickness,
        );

        assert_complex_close(
            scalar(left_right.forward_forward()),
            scalar(right_left.forward_forward()),
        );

        assert_complex_close(
            scalar(left_right.backward_backward()),
            scalar(right_left.backward_backward()),
        );

        assert_complex_close(
            scalar(left_right.forward_backward()),
            scalar(right_left.backward_forward()),
        );

        assert_complex_close(
            scalar(left_right.backward_forward()),
            scalar(right_left.forward_backward()),
        );
    }

    #[test]
    fn bilinear_self_wrapper_matches_cross_function_with_identical_operands() {
        let waves = waves(Complex64::new(2.0, 1.0), Complex64::new(3.0, -2.0));

        let kappa = jet(Complex64::new(0.7, 0.2));
        let thickness = jet(Complex64::new(0.8, 0.0));

        let self_products = integrate_bilinear_wave_products(&waves, &kappa, &thickness);

        let cross_products =
            integrate_bilinear_cross_wave_products(&waves, &waves, &kappa, &kappa, &thickness);

        assert_eq!(self_products, cross_products);
    }
}
