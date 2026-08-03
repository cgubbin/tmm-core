//! Analytically integrated quadratic products of the canonical isotropic
//! layer state.
//!
//! The canonical state is
//!
//! ```text
//! field     = forward + backward
//! secondary = ξ (backward - forward)
//! ξ         = -i Y
//! ```
//!
//! This module transforms integrated directional-wave products into
//! integrated state products. It does not assign electromagnetic meaning to
//! either state component.

use ndarray::Dimension;

use crate::{
    ComplexScalar,
    algebra::{RealScalarAlgebra, ScalarAlgebra},
};

use super::IntegratedWaveProducts;

/// Spatially integrated Hermitian products of the canonical layer state.
///
/// The entries are:
///
/// ```text
/// field_field
///     = ∫ field* field dz
///
/// secondary_secondary
///     = ∫ secondary* secondary dz
///
/// field_secondary
///     = ∫ field* secondary dz
///
/// secondary_field
///     = ∫ secondary* field dz
/// ```
///
/// The diagonal terms are real-valued mathematically, but remain represented
/// by the complex algebra so derivative and storage handling stays uniform.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IntegratedStateProducts<A> {
    field_field: A,
    secondary_secondary: A,
    field_secondary: A,
    secondary_field: A,
}

impl<A> IntegratedStateProducts<A> {
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

    pub(crate) fn map<B>(self, mut map: impl FnMut(A) -> B) -> IntegratedStateProducts<B> {
        IntegratedStateProducts {
            field_field: map(self.field_field),
            secondary_secondary: map(self.secondary_secondary),
            field_secondary: map(self.field_secondary),
            secondary_field: map(self.secondary_field),
        }
    }
}

/// Transform integrated directional-wave products into integrated canonical
/// state products.
///
/// The characteristic slope is:
///
/// ```text
/// ξ = -i Y.
/// ```
pub(crate) fn project_integrated_state_products<A>(
    products: &IntegratedWaveProducts<A>,
    admittance: &A,
) -> IntegratedStateProducts<A>
where
    A: RealScalarAlgebra,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    let i = <A::Scalar as ComplexScalar>::i();

    let slope = admittance.scale(-i);
    let slope_conjugate = slope.conjugated();

    /*
     * |f + b|²
     *
     * = f* f + b* b + f* b + b* f.
     */
    let field_field = products
        .forward_forward()
        .add(products.backward_backward())
        .add(products.forward_backward())
        .add(products.backward_forward());

    /*
     * |ξ(b - f)|²
     *
     * = ξ*ξ [f*f + b*b - f*b - b*f].
     */
    let difference_difference = products
        .forward_forward()
        .add(products.backward_backward())
        .subtract(products.forward_backward())
        .subtract(products.backward_forward());

    let slope_norm_squared = slope_conjugate.multiply(&slope);

    let secondary_secondary = difference_difference.multiply(&slope_norm_squared);

    /*
     * (f + b)* ξ(b - f)
     *
     * = ξ[-f*f + f*b - b*f + b*b].
     */
    let field_difference = products
        .forward_forward()
        .scale(-<A::Scalar as num_traits::One>::one())
        .add(products.forward_backward())
        .subtract(products.backward_forward())
        .add(products.backward_backward());

    let field_secondary = field_difference.multiply(&slope);

    /*
     * Hermitian reverse product. Construct independently rather than simply
     * conjugating the previous expression so derivative semantics remain
     * explicit.
     *
     * [ξ(b-f)]* (f+b)
     *
     * = ξ*[-f*f - f*b + b*f + b*b].
     */
    let difference_field = products
        .forward_forward()
        .scale(-<A::Scalar as num_traits::One>::one())
        .subtract(products.forward_backward())
        .add(products.backward_forward())
        .add(products.backward_backward());

    let secondary_field = difference_field.multiply(&slope_conjugate);

    IntegratedStateProducts::new(
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

    use super::*;

    use crate::{
        algebra::{ArrayJet0, Jet0, RealParameter},
        observable::{BoundaryWaves, layer::integrate_hermitian_wave_products},
        test_support::{C, TOLERANCE, assertions::assert_complex_close},
    };

    type A = ArrayJet0<C, Ix0, RealParameter>;

    const QUADRATURE_TOLERANCE: f64 = 2.0e-9;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn jet(value: C) -> A {
        Jet0::new(arr0(value))
    }

    fn value(value: &A) -> C {
        value.value()[()]
    }

    fn integrate_simpson(function: impl Fn(f64) -> C, start: f64, end: f64, intervals: usize) -> C {
        assert_eq!(intervals % 2, 0);

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
    fn integrated_state_products_store_all_components() {
        let products = IntegratedStateProducts::new(1, 2, 3, 4);

        assert_eq!(products.field_field(), &1);
        assert_eq!(products.secondary_secondary(), &2,);
        assert_eq!(products.field_secondary(), &3);
        assert_eq!(products.secondary_field(), &4);
    }

    #[test]
    fn into_parts_preserves_component_order() {
        let products = IntegratedStateProducts::new(1, 2, 3, 4);

        assert_eq!(products.into_parts(), (1, 2, 3, 4),);
    }

    #[test]
    fn map_transforms_every_component() {
        let products = IntegratedStateProducts::new(1, 2, 3, 4);

        let mapped = products.map(|value| value * 10);

        assert_eq!(mapped.field_field(), &10);
        assert_eq!(mapped.secondary_secondary(), &20,);
        assert_eq!(mapped.field_secondary(), &30,);
        assert_eq!(mapped.secondary_field(), &40,);
    }

    #[test]
    fn projected_cross_products_are_mutual_conjugates() {
        let waves = BoundaryWaves::new(jet(c(0.8, 0.3)), jet(c(-0.2, 0.5)));

        let wavevector = jet(c(2.4, 0.35));
        let thickness = jet(c(1.7, 0.0));
        let admittance = jet(c(1.8, 0.25));

        let wave_products = integrate_hermitian_wave_products(&waves, &wavevector, &thickness);

        let state_products = project_integrated_state_products(&wave_products, &admittance);

        assert_complex_close(
            value(state_products.secondary_field()),
            value(state_products.field_secondary()).conj(),
            TOLERANCE,
        );
    }

    #[test]
    fn diagonal_state_products_are_real() {
        let waves = BoundaryWaves::new(jet(c(0.8, 0.3)), jet(c(-0.2, 0.5)));

        let wave_products =
            integrate_hermitian_wave_products(&waves, &jet(c(2.4, 0.35)), &jet(c(1.7, 0.0)));

        let state_products = project_integrated_state_products(&wave_products, &jet(c(1.8, 0.25)));

        assert_relative_eq!(
            value(state_products.field_field()).im,
            0.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            value(state_products.secondary_secondary(),).im,
            0.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn integrated_state_products_match_numerical_quadrature() {
        let forward = c(0.8, 0.3);
        let backward = c(-0.2, 0.5);
        let wavevector = c(2.4, 0.35);
        let admittance = c(1.8, 0.25);
        let thickness = 1.7;

        let wave_products = integrate_hermitian_wave_products(
            &BoundaryWaves::new(jet(forward), jet(backward)),
            &jet(wavevector),
            &jet(c(thickness, 0.0)),
        );

        let actual = project_integrated_state_products(&wave_products, &jet(admittance));

        let slope = -C::i() * admittance;

        let forward_at = |position: f64| forward * (C::i() * wavevector * position).exp();

        let backward_at = |position: f64| backward * (-C::i() * wavevector * position).exp();

        let field_at = |position: f64| forward_at(position) + backward_at(position);

        let secondary_at = |position: f64| slope * (backward_at(position) - forward_at(position));

        let expected_field_field = integrate_simpson(
            |position| field_at(position).conj() * field_at(position),
            0.0,
            thickness,
            10_000,
        );

        let expected_secondary_secondary = integrate_simpson(
            |position| secondary_at(position).conj() * secondary_at(position),
            0.0,
            thickness,
            10_000,
        );

        let expected_field_secondary = integrate_simpson(
            |position| field_at(position).conj() * secondary_at(position),
            0.0,
            thickness,
            10_000,
        );

        let expected_secondary_field = integrate_simpson(
            |position| secondary_at(position).conj() * field_at(position),
            0.0,
            thickness,
            10_000,
        );

        assert_complex_close(
            value(actual.field_field()),
            expected_field_field,
            QUADRATURE_TOLERANCE,
        );

        assert_complex_close(
            value(actual.secondary_secondary()),
            expected_secondary_secondary,
            QUADRATURE_TOLERANCE,
        );

        assert_complex_close(
            value(actual.field_secondary()),
            expected_field_secondary,
            QUADRATURE_TOLERANCE,
        );

        assert_complex_close(
            value(actual.secondary_field()),
            expected_secondary_field,
            QUADRATURE_TOLERANCE,
        );
    }
}
