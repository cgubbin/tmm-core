//! Derivatives of isotropic layer propagation quantities.
//!
//! This module evaluates first and second derivatives of the normal
//! wavenumber and polarisation-dependent characteristic factor used by the
//! isotropic 2×2 backends.
//!
//! The primitive spectral coordinates are:
//!
//! ```text
//! k₀²
//! k∥²
//! ```
//!
//! where `k₀` is the vacuum wavenumber and `k∥` is the conserved parallel
//! wavenumber.
//!
//! For an isotropic medium,
//!
//! ```text
//! κ² = ε μ k₀² - k∥²
//! ```
//!
//! and:
//!
//! ```text
//! factor = μ    for TE
//! factor = ε    for TM
//! ```
//!
//! Derivatives with respect to linear `k₀` and `k∥` are obtained later using
//! the shared jet chain-rule transformation.

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::{
        IsotropicLayerQuantities, Polarisation,
        evaluator::{ComplexPlane, ConstitutiveDerivativeEvaluator, RealAxis},
    },
    material::{
        DerivativeOrder, EvaluateDifferentiableMaterial, EvaluateDifferentiableMeromorphicMaterial,
        SpectralVariable,
    },
};

/// First derivatives of isotropic layer quantities.
///
/// These derivatives refer to one primitive spectral coordinate. The normal
/// wavenumber derivative and factor derivative are always evaluated with
/// respect to the same coordinate.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IsotropicLayerFirstDerivatives<C, D>
where
    D: Dimension,
{
    dkappa: ArrayBase<OwnedRepr<C>, D>,
    dfactor: ArrayBase<OwnedRepr<C>, D>,
}

impl<C, D> IsotropicLayerFirstDerivatives<C, D>
where
    D: Dimension,
{
    /// Return the first derivative of the normal wavenumber.
    pub(crate) fn dkappa(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.dkappa
    }

    /// Return the first derivative of the polarisation-dependent factor.
    pub(crate) fn dfactor(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.dfactor
    }

    /// Consume the derivatives and return their components.
    pub(crate) fn into_parts(self) -> (ArrayBase<OwnedRepr<C>, D>, ArrayBase<OwnedRepr<C>, D>) {
        (self.dkappa, self.dfactor)
    }
}

/// First and second derivatives of isotropic layer quantities.
///
/// The embedded [`IsotropicLayerFirstDerivatives`] and the second derivatives
/// all refer to the same primitive spectral coordinate.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IsotropicLayerSecondDerivatives<C, D>
where
    D: Dimension,
{
    first: IsotropicLayerFirstDerivatives<C, D>,
    ddkappa: ArrayBase<OwnedRepr<C>, D>,
    ddfactor: ArrayBase<OwnedRepr<C>, D>,
}

impl<C, D> IsotropicLayerSecondDerivatives<C, D>
where
    D: Dimension,
{
    /// Return the corresponding first derivatives.
    pub(crate) fn first(&self) -> &IsotropicLayerFirstDerivatives<C, D> {
        &self.first
    }

    /// Return the second derivative of the normal wavenumber.
    pub(crate) fn ddkappa(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.ddkappa
    }

    /// Return the second derivative of the polarisation-dependent factor.
    pub(crate) fn ddfactor(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.ddfactor
    }

    /// Consume the derivatives and return their components.
    #[allow(clippy::type_complexity)]
    pub(crate) fn into_parts(
        self,
    ) -> (
        IsotropicLayerFirstDerivatives<C, D>,
        ArrayBase<OwnedRepr<C>, D>,
        ArrayBase<OwnedRepr<C>, D>,
    ) {
        (self.first, self.ddkappa, self.ddfactor)
    }
}

impl<C, D> IsotropicLayerFirstDerivatives<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn vacuum_wavenumber_squared_real_axis<M>(
        material: &M,
        quantities: &IsotropicLayerQuantities<C, D>,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
        polarisation: Polarisation,
    ) -> Self
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::vacuum_wavenumber_squared::<RealAxis, M>(
            material,
            quantities,
            vacuum_wavenumber,
            polarisation,
        )
    }

    pub(crate) fn vacuum_wavenumber_squared_complex_plane<M>(
        material: &M,
        quantities: &IsotropicLayerQuantities<C, D>,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
        polarisation: Polarisation,
    ) -> Self
    where
        M: EvaluateDifferentiableMeromorphicMaterial<C, Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::vacuum_wavenumber_squared::<ComplexPlane, M>(
            material,
            quantities,
            vacuum_wavenumber,
            polarisation,
        )
    }

    /// Evaluate derivatives with respect to squared vacuum wavenumber `k₀²`.
    ///
    /// For
    ///
    /// ```text
    /// Q = ε μ k₀² - k∥²
    /// κ = sqrt(Q)
    /// ```
    ///
    /// the derivative is:
    ///
    /// ```text
    /// dQ/d(k₀²)
    ///     = (ε′ μ + ε μ′) k₀² + ε μ
    ///
    /// dκ/d(k₀²)
    ///     = Q′ / (2κ)
    /// ```
    pub(crate) fn vacuum_wavenumber_squared<E, M>(
        material: &M,
        quantities: &IsotropicLayerQuantities<C, D>,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
        polarisation: Polarisation,
    ) -> Self
    where
        E: ConstitutiveDerivativeEvaluator<C, D, M>,
        C::RealField: Copy,
    {
        let epsilon = quantities.epsilon();
        let mu = quantities.mu();
        let kappa = quantities.kappa();

        let deps = E::relative_permittivity_derivative(
            material,
            vacuum_wavenumber,
            DerivativeOrder::First,
            SpectralVariable::VacuumWavenumberSquared,
        );

        let dmu = E::relative_permeability_derivative(
            material,
            vacuum_wavenumber,
            DerivativeOrder::First,
            SpectralVariable::VacuumWavenumberSquared,
        );

        let k0_squared = vacuum_wavenumber.mapv(|k0| k0 * k0);

        let dq = (deps.clone() * mu.view() + epsilon.clone() * dmu.view()) * k0_squared
            + epsilon.clone() * mu.view();

        let two = C::one() + C::one();

        let dkappa = dq / kappa.mapv(|value| two * value);

        let dfactor = match polarisation {
            Polarisation::TransverseElectric => dmu,
            Polarisation::TransverseMagnetic => deps,
        };

        Self { dkappa, dfactor }
    }

    /// Evaluate derivatives with respect to squared parallel wavenumber `k∥²`.
    ///
    /// Since:
    ///
    /// ```text
    /// Q = ε μ k₀² - k∥²
    /// ```
    ///
    /// material quantities are constant with respect to this coordinate and:
    ///
    /// ```text
    /// dκ/d(k∥²) = -1 / (2κ)
    /// ```
    pub(crate) fn parallel_wavenumber_squared(quantities: &IsotropicLayerQuantities<C, D>) -> Self {
        let two = C::one() + C::one();

        let dkappa = quantities.kappa().mapv(|kappa| -C::one() / (two * kappa));

        let dfactor = quantities.factor().mapv(|_| C::zero());

        Self { dkappa, dfactor }
    }
}

impl<C, D> IsotropicLayerSecondDerivatives<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn vacuum_wavenumber_squared_real_axis<M>(
        material: &M,
        quantities: &IsotropicLayerQuantities<C, D>,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
        polarisation: Polarisation,
    ) -> Self
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::vacuum_wavenumber_squared::<RealAxis, M>(
            material,
            quantities,
            vacuum_wavenumber,
            polarisation,
        )
    }

    pub(crate) fn vacuum_wavenumber_squared_complex_plane<M>(
        material: &M,
        quantities: &IsotropicLayerQuantities<C, D>,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
        polarisation: Polarisation,
    ) -> Self
    where
        M: EvaluateDifferentiableMeromorphicMaterial<C, Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::vacuum_wavenumber_squared::<ComplexPlane, M>(
            material,
            quantities,
            vacuum_wavenumber,
            polarisation,
        )
    }

    /// Evaluate first and second derivatives with respect to squared vacuum
    /// wavenumber `k₀²`.
    pub(crate) fn vacuum_wavenumber_squared<E, M>(
        material: &M,
        quantities: &IsotropicLayerQuantities<C, D>,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
        polarisation: Polarisation,
    ) -> Self
    where
        E: ConstitutiveDerivativeEvaluator<C, D, M>,
    {
        let epsilon = quantities.epsilon();
        let mu = quantities.mu();
        let kappa = quantities.kappa();

        let k0_squared = vacuum_wavenumber.mapv(|k0| k0 * k0);

        let deps: ArrayBase<OwnedRepr<C>, D> = E::relative_permittivity_derivative(
            material,
            vacuum_wavenumber,
            DerivativeOrder::First,
            SpectralVariable::VacuumWavenumberSquared,
        );

        let ddeps = E::relative_permittivity_derivative(
            material,
            vacuum_wavenumber,
            DerivativeOrder::Second,
            SpectralVariable::VacuumWavenumberSquared,
        );

        let dmu = E::relative_permeability_derivative(
            material,
            vacuum_wavenumber,
            DerivativeOrder::First,
            SpectralVariable::VacuumWavenumberSquared,
        );

        let ddmu = E::relative_permeability_derivative(
            material,
            vacuum_wavenumber,
            DerivativeOrder::Second,
            SpectralVariable::VacuumWavenumberSquared,
        );

        // A = d(εμ)/d(k₀²)
        let a = deps.clone() * mu.view() + epsilon.clone() * dmu.view();

        // A′ = d²(εμ)/d(k₀²)²
        let da = ddeps.clone() * mu.view()
            + (deps.clone() * dmu.view()).mapv(|value| value + value)
            + epsilon.clone() * ddmu.view();

        // Q = ε μ k₀² - k∥²
        //
        // Q′  = A k₀² + ε μ
        // Q″  = A′ k₀² + 2A
        let dq = a.clone() * k0_squared.view() + epsilon.clone() * mu.view();

        let ddq = da * k0_squared + a.mapv(|value| value + value);

        let two = C::one() + C::one();
        let four = two + two;

        let dkappa = dq.clone() / kappa.mapv(|value| two * value);

        // κ = sqrt(Q)
        //
        // κ′  = Q′/(2κ)
        // κ″  = Q″/(2κ) - (Q′)²/(4κ³)
        let ddkappa = ddq / kappa.mapv(|value| two * value)
            - dq.mapv(|value| value * value) / kappa.mapv(|value| four * value * value * value);

        let (dfactor, ddfactor) = match polarisation {
            Polarisation::TransverseElectric => (dmu, ddmu),
            Polarisation::TransverseMagnetic => (deps, ddeps),
        };

        Self {
            first: IsotropicLayerFirstDerivatives { dkappa, dfactor },
            ddkappa,
            ddfactor,
        }
    }

    /// Evaluate first and second derivatives with respect to squared parallel
    /// wavenumber `k∥²`.
    ///
    /// ```text
    /// dκ/d(k∥²)     = -1/(2κ)
    /// d²κ/d(k∥²)²   = -1/(4κ³)
    /// ```
    pub(crate) fn parallel_wavenumber_squared(quantities: &IsotropicLayerQuantities<C, D>) -> Self {
        let two = C::one() + C::one();
        let four = two + two;

        let dkappa = quantities.kappa().mapv(|kappa| -C::one() / (two * kappa));

        let ddkappa = quantities
            .kappa()
            .mapv(|kappa| -C::one() / (four * kappa * kappa * kappa));

        let zero = quantities.factor().mapv(|_| C::zero());

        Self {
            first: IsotropicLayerFirstDerivatives {
                dkappa,
                dfactor: zero.clone(),
            },
            ddkappa,
            ddfactor: zero,
        }
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array0, Array1, arr0, array};
    use num_complex::Complex64;

    use super::*;
    use crate::{
        backend::{PlanarInput, Polarisation},
        material::Constant,
    };

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn constant_material(epsilon: f64, mu: f64) -> Constant<f64> {
        Constant::new(epsilon, mu)
    }

    fn scalar_input(
        vacuum_wavenumber: f64,
        parallel_wavenumber: f64,
        polarisation: Polarisation,
    ) -> PlanarInput<Array0<C>> {
        PlanarInput::new(
            arr0(c(vacuum_wavenumber)),
            arr0(c(parallel_wavenumber)),
            polarisation,
        )
    }

    fn real_scalar_input(
        vacuum_wavenumber: f64,
        parallel_wavenumber: f64,
        polarisation: Polarisation,
    ) -> PlanarInput<Array0<f64>> {
        PlanarInput::new(
            arr0(vacuum_wavenumber),
            arr0(parallel_wavenumber),
            polarisation,
        )
    }

    fn assert_complex_close(actual: C, expected: C, tolerance: f64) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = tolerance,
            max_relative = tolerance
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = tolerance,
            max_relative = tolerance
        );
    }

    fn assert_array1_close(actual: &Array1<C>, expected: &Array1<C>, tolerance: f64) {
        assert_eq!(actual.raw_dim(), expected.raw_dim());

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_complex_close(*actual, *expected, tolerance);
        }
    }

    fn quantities_at_vacuum_wavenumber_squared(
        material: &Constant<f64>,
        vacuum_wavenumber_squared: f64,
        parallel_wavenumber: f64,
        polarisation: Polarisation,
    ) -> IsotropicLayerQuantities<C, ndarray::Ix0> {
        let input = scalar_input(
            vacuum_wavenumber_squared.sqrt(),
            parallel_wavenumber,
            polarisation,
        );

        IsotropicLayerQuantities::real_axis(material, &input)
    }

    fn quantities_at_parallel_wavenumber_squared(
        material: &Constant<f64>,
        vacuum_wavenumber: f64,
        parallel_wavenumber_squared: f64,
        polarisation: Polarisation,
    ) -> IsotropicLayerQuantities<C, ndarray::Ix0> {
        let input = scalar_input(
            vacuum_wavenumber,
            parallel_wavenumber_squared.sqrt(),
            polarisation,
        );

        IsotropicLayerQuantities::real_axis(material, &input)
    }

    #[test]
    fn vacuum_wavenumber_squared_real_axis_first_derivative_matches_finite_difference() {
        let material = constant_material(2.25, 1.4);

        let k0_squared: f64 = 9.0;
        let k_parallel = 0.7;
        let h = 1e-5;

        let input = scalar_input(
            k0_squared.sqrt(),
            k_parallel,
            Polarisation::TransverseElectric,
        );

        let q = IsotropicLayerQuantities::real_axis(&material, &input);

        let derivatives = IsotropicLayerFirstDerivatives::vacuum_wavenumber_squared_real_axis(
            &material,
            &q,
            input.vacuum_wavenumber(),
            input.polarisation(),
        );

        let plus = quantities_at_vacuum_wavenumber_squared(
            &material,
            k0_squared + h,
            k_parallel,
            Polarisation::TransverseElectric,
        );

        let minus = quantities_at_vacuum_wavenumber_squared(
            &material,
            k0_squared - h,
            k_parallel,
            Polarisation::TransverseElectric,
        );

        let expected_dkappa = (plus.kappa()[()] - minus.kappa()[()]) / (2.0 * h);

        let expected_dfactor = (plus.factor()[()] - minus.factor()[()]) / (2.0 * h);

        assert_complex_close(derivatives.dkappa()[()], expected_dkappa, 1e-8);

        assert_complex_close(derivatives.dfactor()[()], expected_dfactor, 1e-10);
    }

    #[test]
    fn vacuum_wavenumber_squared_real_axis_second_derivative_matches_finite_difference() {
        let material = constant_material(2.25, 1.4);

        let k0_squared: f64 = 9.0;
        let k_parallel = 0.7;
        let h = 2e-3;

        let input = scalar_input(
            k0_squared.sqrt(),
            k_parallel,
            Polarisation::TransverseElectric,
        );

        let q = IsotropicLayerQuantities::real_axis(&material, &input);

        let derivatives = IsotropicLayerSecondDerivatives::vacuum_wavenumber_squared_real_axis(
            &material,
            &q,
            input.vacuum_wavenumber(),
            input.polarisation(),
        );

        let plus = quantities_at_vacuum_wavenumber_squared(
            &material,
            k0_squared + h,
            k_parallel,
            Polarisation::TransverseElectric,
        );

        let zero = quantities_at_vacuum_wavenumber_squared(
            &material,
            k0_squared,
            k_parallel,
            Polarisation::TransverseElectric,
        );

        let minus = quantities_at_vacuum_wavenumber_squared(
            &material,
            k0_squared - h,
            k_parallel,
            Polarisation::TransverseElectric,
        );

        let expected_ddkappa =
            (plus.kappa()[()] - c(2.0) * zero.kappa()[()] + minus.kappa()[()]) / (h * h);

        let expected_ddfactor =
            (plus.factor()[()] - c(2.0) * zero.factor()[()] + minus.factor()[()]) / (h * h);

        assert_complex_close(derivatives.ddkappa()[()], expected_ddkappa, 2e-6);

        assert_complex_close(derivatives.ddfactor()[()], expected_ddfactor, 1e-8);
    }

    #[test]
    fn parallel_wavenumber_squared_first_derivative_matches_finite_difference() {
        let material = constant_material(2.25, 1.4);

        let vacuum_wavenumber = 3.0;
        let parallel_squared: f64 = 0.49;
        let h = 1e-5;

        let input = scalar_input(
            vacuum_wavenumber,
            parallel_squared.sqrt(),
            Polarisation::TransverseMagnetic,
        );

        let q = IsotropicLayerQuantities::real_axis(&material, &input);

        let derivatives = IsotropicLayerFirstDerivatives::parallel_wavenumber_squared(&q);

        let plus = quantities_at_parallel_wavenumber_squared(
            &material,
            vacuum_wavenumber,
            parallel_squared + h,
            Polarisation::TransverseMagnetic,
        );

        let minus = quantities_at_parallel_wavenumber_squared(
            &material,
            vacuum_wavenumber,
            parallel_squared - h,
            Polarisation::TransverseMagnetic,
        );

        let expected_dkappa = (plus.kappa()[()] - minus.kappa()[()]) / (2.0 * h);

        let expected_dfactor = (plus.factor()[()] - minus.factor()[()]) / (2.0 * h);

        assert_complex_close(derivatives.dkappa()[()], expected_dkappa, 1e-8);

        assert_complex_close(derivatives.dfactor()[()], expected_dfactor, 1e-10);
    }

    #[test]
    fn parallel_wavenumber_squared_second_derivative_matches_finite_difference() {
        let material = constant_material(2.25, 1.4);

        let vacuum_wavenumber = 3.0;
        let parallel_squared: f64 = 0.49;
        let h = 2e-3;

        let input = scalar_input(
            vacuum_wavenumber,
            parallel_squared.sqrt(),
            Polarisation::TransverseMagnetic,
        );

        let q = IsotropicLayerQuantities::real_axis(&material, &input);

        let derivatives = IsotropicLayerSecondDerivatives::parallel_wavenumber_squared(&q);

        let plus = quantities_at_parallel_wavenumber_squared(
            &material,
            vacuum_wavenumber,
            parallel_squared + h,
            Polarisation::TransverseMagnetic,
        );

        let zero = quantities_at_parallel_wavenumber_squared(
            &material,
            vacuum_wavenumber,
            parallel_squared,
            Polarisation::TransverseMagnetic,
        );

        let minus = quantities_at_parallel_wavenumber_squared(
            &material,
            vacuum_wavenumber,
            parallel_squared - h,
            Polarisation::TransverseMagnetic,
        );

        let expected_ddkappa =
            (plus.kappa()[()] - c(2.0) * zero.kappa()[()] + minus.kappa()[()]) / (h * h);

        let expected_ddfactor =
            (plus.factor()[()] - c(2.0) * zero.factor()[()] + minus.factor()[()]) / (h * h);

        assert_complex_close(derivatives.ddkappa()[()], expected_ddkappa, 2e-6);

        assert_complex_close(derivatives.ddfactor()[()], expected_ddfactor, 1e-8);
    }

    #[test]
    fn complex_te_factor_derivatives_follow_permeability() {
        let material = constant_material(2.25, 1.4);

        let input = scalar_input(3.0, 0.7, Polarisation::TransverseElectric);

        let q = IsotropicLayerQuantities::real_axis(&material, &input);

        let first = IsotropicLayerFirstDerivatives::vacuum_wavenumber_squared_real_axis(
            &material,
            &q,
            input.vacuum_wavenumber(),
            input.polarisation(),
        );

        let second = IsotropicLayerSecondDerivatives::vacuum_wavenumber_squared_real_axis(
            &material,
            &q,
            input.vacuum_wavenumber(),
            input.polarisation(),
        );

        assert_complex_close(first.dfactor()[()], c(0.0), 1e-12);

        assert_complex_close(second.ddfactor()[()], c(0.0), 1e-12);
    }

    #[test]
    fn complex_tm_factor_derivatives_follow_permittivity() {
        let material = constant_material(2.25, 1.4);

        let input = scalar_input(3.0, 0.7, Polarisation::TransverseMagnetic);

        let q = IsotropicLayerQuantities::real_axis(&material, &input);

        let first = IsotropicLayerFirstDerivatives::vacuum_wavenumber_squared_real_axis(
            &material,
            &q,
            input.vacuum_wavenumber(),
            input.polarisation(),
        );

        let second = IsotropicLayerSecondDerivatives::vacuum_wavenumber_squared_real_axis(
            &material,
            &q,
            input.vacuum_wavenumber(),
            input.polarisation(),
        );

        assert_complex_close(first.dfactor()[()], c(0.0), 1e-12);

        assert_complex_close(second.ddfactor()[()], c(0.0), 1e-12);
    }

    #[test]
    fn complex_nondispersive_material_has_zero_factor_derivatives() {
        let material = constant_material(3.1, 1.2);

        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            let input = scalar_input(2.4, 0.5, polarisation);

            let q = IsotropicLayerQuantities::real_axis(&material, &input);

            let first = IsotropicLayerFirstDerivatives::vacuum_wavenumber_squared_real_axis(
                &material,
                &q,
                input.vacuum_wavenumber(),
                input.polarisation(),
            );

            let second = IsotropicLayerSecondDerivatives::vacuum_wavenumber_squared_real_axis(
                &material,
                &q,
                input.vacuum_wavenumber(),
                input.polarisation(),
            );

            assert_complex_close(first.dfactor()[()], c(0.0), 1e-12);

            assert_complex_close(second.ddfactor()[()], c(0.0), 1e-12);
        }
    }

    #[test]
    fn complex_parallel_wavenumber_derivatives_have_zero_factor_terms() {
        let material = constant_material(3.1, 1.2);

        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            let input = scalar_input(2.4, 0.5, polarisation);

            let q = IsotropicLayerQuantities::real_axis(&material, &input);

            let first = IsotropicLayerFirstDerivatives::parallel_wavenumber_squared(&q);

            let second = IsotropicLayerSecondDerivatives::parallel_wavenumber_squared(&q);

            assert_complex_close(first.dfactor()[()], c(0.0), 1e-12);

            assert_complex_close(second.first().dfactor()[()], c(0.0), 1e-12);

            assert_complex_close(second.ddfactor()[()], c(0.0), 1e-12);
        }
    }

    #[test]
    fn complex_second_derivative_contains_same_first_derivative_as_first_order_path() {
        let material = constant_material(2.25, 1.4);

        let input = scalar_input(3.0, 0.7, Polarisation::TransverseMagnetic);

        let q = IsotropicLayerQuantities::real_axis(&material, &input);

        let first = IsotropicLayerFirstDerivatives::vacuum_wavenumber_squared_real_axis(
            &material,
            &q,
            input.vacuum_wavenumber(),
            input.polarisation(),
        );

        let second = IsotropicLayerSecondDerivatives::vacuum_wavenumber_squared_real_axis(
            &material,
            &q,
            input.vacuum_wavenumber(),
            input.polarisation(),
        );

        assert_complex_close(second.first().dkappa()[()], first.dkappa()[()], 1e-12);

        assert_complex_close(second.first().dfactor()[()], first.dfactor()[()], 1e-12);
    }

    #[test]
    fn complex_evanescent_quantities_have_finite_derivatives() {
        let material = constant_material(1.0, 1.0);

        let input = PlanarInput::new(
            arr0(C::new(1.0, 0.05)),
            arr0(C::new(2.0, 0.1)),
            Polarisation::TransverseElectric,
        );

        let q = IsotropicLayerQuantities::real_axis(&material, &input);

        let first = IsotropicLayerFirstDerivatives::parallel_wavenumber_squared(&q);

        let second = IsotropicLayerSecondDerivatives::parallel_wavenumber_squared(&q);

        for value in [
            first.dkappa()[()],
            second.first().dkappa()[()],
            second.ddkappa()[()],
        ] {
            assert!(value.re.is_finite());
            assert!(value.im.is_finite());
        }
    }

    #[test]
    fn complex_array1_vacuum_wavenumber_derivatives_match_scalar_evaluations() {
        let material = constant_material(2.25, 1.4);

        let vacuum_wavenumbers = array![c(2.0), c(2.5), c(3.0),];

        let parallel_wavenumbers = array![c(0.3), c(0.4), c(0.5),];

        let input = PlanarInput::new(
            vacuum_wavenumbers.clone(),
            parallel_wavenumbers.clone(),
            Polarisation::TransverseMagnetic,
        );

        let q = IsotropicLayerQuantities::real_axis(&material, &input);

        let array_derivatives =
            IsotropicLayerSecondDerivatives::vacuum_wavenumber_squared_real_axis(
                &material,
                &q,
                input.vacuum_wavenumber(),
                input.polarisation(),
            );

        let mut expected_dkappa = Vec::new();
        let mut expected_ddkappa = Vec::new();
        let mut expected_dfactor = Vec::new();
        let mut expected_ddfactor = Vec::new();

        for (&k0, &k_parallel) in vacuum_wavenumbers.iter().zip(parallel_wavenumbers.iter()) {
            let scalar =
                PlanarInput::new(arr0(k0), arr0(k_parallel), Polarisation::TransverseMagnetic);

            let scalar_q = IsotropicLayerQuantities::real_axis(&material, &scalar);

            let scalar_derivatives =
                IsotropicLayerSecondDerivatives::vacuum_wavenumber_squared_real_axis(
                    &material,
                    &scalar_q,
                    scalar.vacuum_wavenumber(),
                    scalar.polarisation(),
                );

            expected_dkappa.push(scalar_derivatives.first().dkappa()[()]);

            expected_ddkappa.push(scalar_derivatives.ddkappa()[()]);

            expected_dfactor.push(scalar_derivatives.first().dfactor()[()]);

            expected_ddfactor.push(scalar_derivatives.ddfactor()[()]);
        }

        assert_array1_close(
            array_derivatives.first().dkappa(),
            &Array1::from_vec(expected_dkappa),
            1e-12,
        );

        assert_array1_close(
            array_derivatives.ddkappa(),
            &Array1::from_vec(expected_ddkappa),
            1e-12,
        );

        assert_array1_close(
            array_derivatives.first().dfactor(),
            &Array1::from_vec(expected_dfactor),
            1e-12,
        );

        assert_array1_close(
            array_derivatives.ddfactor(),
            &Array1::from_vec(expected_ddfactor),
            1e-12,
        );
    }

    #[test]
    fn array1_parallel_wavenumber_derivatives_match_scalar_evaluations() {
        let material = constant_material(2.25, 1.4);

        let vacuum_wavenumbers = array![c(2.0), c(2.5), c(3.0),];

        let parallel_wavenumbers = array![c(0.3), c(0.4), c(0.5),];

        let input = PlanarInput::new(
            vacuum_wavenumbers.clone(),
            parallel_wavenumbers.clone(),
            Polarisation::TransverseElectric,
        );

        let q = IsotropicLayerQuantities::real_axis(&material, &input);

        let array_derivatives = IsotropicLayerSecondDerivatives::parallel_wavenumber_squared(&q);

        let mut expected_dkappa = Vec::new();
        let mut expected_ddkappa = Vec::new();

        for (&k0, &k_parallel) in vacuum_wavenumbers.iter().zip(parallel_wavenumbers.iter()) {
            let scalar =
                PlanarInput::new(arr0(k0), arr0(k_parallel), Polarisation::TransverseElectric);

            let scalar_q = IsotropicLayerQuantities::real_axis(&material, &scalar);

            let scalar_derivatives =
                IsotropicLayerSecondDerivatives::parallel_wavenumber_squared(&scalar_q);

            expected_dkappa.push(scalar_derivatives.first().dkappa()[()]);

            expected_ddkappa.push(scalar_derivatives.ddkappa()[()]);
        }

        assert_array1_close(
            array_derivatives.first().dkappa(),
            &Array1::from_vec(expected_dkappa),
            1e-12,
        );

        assert_array1_close(
            array_derivatives.ddkappa(),
            &Array1::from_vec(expected_ddkappa),
            1e-12,
        );
    }
}
