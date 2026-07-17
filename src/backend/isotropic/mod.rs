//! Shared quantities for isotropic planar backends.
//!
//! This module evaluates the material and propagation quantities reused by the
//! isotropic 2×2 transfer- and scattering-matrix backends.
//!
//! For each sampled planar input, the normal wavenumber is defined by
//!
//! ```text
//! κ² = ε μ k₀² - k∥²
//! ```
//!
//! and the polarisation-dependent factor is
//!
//! ```text
//! factor = μ    for TE
//! factor = ε    for TM
//! ```
//!
//! The corresponding characteristic admittance is
//!
//! ```text
//! Y = κ / factor
//! ```
//!
//! Material quantities are evaluated once per medium and reused when
//! constructing matrices and derivatives.
//!
//! # Normal-wavenumber branch
//!
//! For each isotropic medium, the normal wavenumber is evaluated as
//!
//! ```text
//! κ = sqrt(ε μ k₀² - k∥²)
//! ```
//!
//! using the principal complex square root supplied by [`ComplexScalar`].
//! No additional pointwise sign correction is applied.
//!
//! The principal square root is analytic away from its branch cut and branch
//! point. Consequently, derivatives returned by this module are local
//! derivatives on that selected branch.
//!
//! For real passive scattering problems, this convention gives:
//!
//! - `κ >= 0` for propagating modes with positive real `κ²`;
//! - `Im(κ) >= 0` for evanescent modes with negative real `κ²`.
//!
//! For complex continuation and contour-based mode finding, callers must choose
//! a search domain over which
//!
//! ```text
//! ε_j μ_j k₀² - k∥²
//! ```
//!
//! avoids the principal square-root branch cut and zero for every medium `j`
//! whose normal wavenumber enters the residual. A contour crossing such a
//! branch cut does not define a single analytic residual and is therefore not
//! suitable for argument-principle integration.
//!
//! The caller does not supply `κ` directly. Branch selection is part of the
//! backend's mathematical convention and is applied consistently to finite
//! layers and both exterior media.

mod admittance;
mod derivatives;

use ndarray::{ArrayBase, Dimension, OwnedRepr};

pub(crate) use admittance::IsotropicLayerAdmittance;
pub(crate) use derivatives::{IsotropicLayerFirstDerivatives, IsotropicLayerSecondDerivatives};

use crate::{
    ComplexScalar,
    backend::{
        PlanarInput, Polarisation,
        evaluator::{ComplexPlane, ConstitutiveEvaluator, RealAxis},
    },
    material::{Material, MeromorphicMaterial, Scalar},
};

/// Material and propagation quantities for one isotropic medium.
///
/// Every array has the same sampled dimension as the corresponding
/// [`PlanarInput`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IsotropicLayerQuantities<C, D>
where
    D: Dimension,
{
    epsilon: ArrayBase<OwnedRepr<C>, D>,
    mu: ArrayBase<OwnedRepr<C>, D>,
    kappa: ArrayBase<OwnedRepr<C>, D>,
    factor: ArrayBase<OwnedRepr<C>, D>,
}

impl<C, D> IsotropicLayerQuantities<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn real_axis<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self
    where
        M: Material<Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::new::<RealAxis, M>(material, planar)
    }

    pub(crate) fn complex_plane<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self
    where
        M: MeromorphicMaterial<Real = C::RealField>,
    {
        Self::new::<ComplexPlane, M>(material, planar)
    }

    /// Evaluate material and propagation quantities for one isotropic medium.
    ///
    /// The normal wavenumber is computed using the principal complex square root:
    ///
    /// ```text
    /// κ = sqrt(ε μ k₀² - k∥²).
    /// ```
    ///
    /// This operation selects a locally analytic branch away from the principal
    /// square-root cut and branch point. The backend performs no sample-by-sample
    /// sign correction.
    pub(crate) fn new<E, M>(material: &M, planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>) -> Self
    where
        E: ConstitutiveEvaluator<C, D, M>,
    {
        let epsilon = E::relative_permittivity(material, planar.vacuum_wavenumber());
        let mu = E::relative_permeability(material, planar.vacuum_wavenumber());

        let vacuum_wavenumber_squared = planar.vacuum_wavenumber().mapv(|k0| k0 * k0);

        let parallel_wavenumber_squared = planar
            .parallel_wavenumber()
            .mapv(|k_parallel| k_parallel * k_parallel);

        let kappa_squared =
            epsilon.clone() * mu.clone() * vacuum_wavenumber_squared - parallel_wavenumber_squared;

        let kappa = kappa_squared.mapv(principal_normal_wavenumber);

        let factor = match planar.polarisation() {
            Polarisation::TransverseElectric => mu.clone(),
            Polarisation::TransverseMagnetic => epsilon.clone(),
        };

        Self {
            epsilon,
            mu,
            kappa,
            factor,
        }
    }

    /// Return the relative permittivity.
    pub(crate) fn epsilon(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.epsilon
    }

    /// Return the relative permeability.
    pub(crate) fn mu(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.mu
    }

    /// Return the selected normal wavenumber `κ`.
    pub(crate) fn kappa(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.kappa
    }

    /// Return the TE/TM characteristic factor.
    pub(crate) fn factor(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.factor
    }

    /// Construct the characteristic admittance `Y = κ / factor`.
    pub(crate) fn admittance(&self) -> IsotropicLayerAdmittance<C, D> {
        IsotropicLayerAdmittance::from_quantities(self)
    }
}

// /// Select the outgoing or decaying normal-wavenumber branch.
// ///
// /// The principal square root already has a nonnegative imaginary part for
// /// values away from its branch cut. The explicit correction below documents
// /// and enforces the backend convention:
// ///
// /// - evanescent/passive waves satisfy `Im(κ) >= 0`;
// /// - when `Im(κ) == 0`, propagating waves satisfy `Re(κ) >= 0`.
// fn outgoing_normal_wavenumber<C>(kappa_squared: C) -> C
// where
//     C: ComplexScalar,
// {
//     let kappa = kappa_squared.sqrt();

//     let imaginary = kappa.imaginary();
//     let real = kappa.real();

//     if imaginary < C::zero().real() || (imaginary == C::zero().real() && real < C::zero().real()) {
//         -kappa
//     } else {
//         kappa
//     }
// }

/// Evaluate the principal complex square root used for the normal wavenumber.
///
/// No pointwise sign correction is applied.
fn principal_normal_wavenumber<C>(squared: C) -> C
where
    C: ComplexScalar,
{
    squared.sqrt()
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{arr0, array};
    use num_complex::Complex64;

    use super::*;
    use crate::{backend::Polarisation, material::Constant};

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn material(epsilon: f64, mu: f64) -> Constant<f64> {
        Constant::new(epsilon, mu)
    }

    fn scalar_input(
        vacuum_wavenumber: f64,
        parallel_wavenumber: f64,
        polarisation: Polarisation,
    ) -> PlanarInput<ndarray::Array0<C>> {
        PlanarInput::new(
            arr0(c(vacuum_wavenumber)),
            arr0(c(parallel_wavenumber)),
            polarisation,
        )
    }

    fn assert_close(actual: C, expected: C) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = 1e-12,
            max_relative = 1e-12
        );
    }

    #[test]
    fn normal_wavenumber_matches_dispersion_relation() {
        let material = material(4.0, 1.0);
        let input = scalar_input(3.0, 2.0, Polarisation::TransverseElectric);

        let q = IsotropicLayerQuantities::real_axis(&material, &input);

        let expected = c((4.0_f64 * 9.0 - 4.0).sqrt());

        assert_close(q.kappa()[()], expected);
    }

    #[test]
    fn te_factor_is_permeability() {
        let material = material(4.0, 2.0);
        let input = scalar_input(3.0, 1.0, Polarisation::TransverseElectric);

        let q = IsotropicLayerQuantities::real_axis(&material, &input);

        assert_close(q.factor()[()], c(2.0));
    }

    #[test]
    fn tm_factor_is_permittivity() {
        let material = material(4.0, 2.0);
        let input = scalar_input(3.0, 1.0, Polarisation::TransverseMagnetic);

        let q = IsotropicLayerQuantities::real_axis(&material, &input);

        assert_close(q.factor()[()], c(4.0));
    }

    #[test]
    fn evanescent_normal_wavenumber_is_decaying() {
        let material = material(1.0, 1.0);
        let input = scalar_input(1.0, 2.0, Polarisation::TransverseElectric);

        let q = IsotropicLayerQuantities::real_axis(&material, &input);

        assert_relative_eq!(q.kappa()[()].re, 0.0, epsilon = 1e-12);
        assert!(q.kappa()[()].im > 0.0);
    }

    #[test]
    fn admittance_matches_kappa_over_factor() {
        let material = material(4.0, 2.0);
        let input = scalar_input(3.0, 1.0, Polarisation::TransverseMagnetic);

        let q = IsotropicLayerQuantities::real_axis(&material, &input);
        let admittance = q.admittance();

        assert_close(admittance.value()[()], q.kappa()[()] / q.factor()[()]);
    }

    #[test]
    fn sampled_shape_is_preserved() {
        let material = material(4.0, 1.0);

        let input = PlanarInput::new(
            array![c(1.0), c(2.0), c(3.0)],
            array![c(0.1), c(0.2), c(0.3)],
            Polarisation::TransverseElectric,
        );

        let q = IsotropicLayerQuantities::real_axis(&material, &input);

        assert_eq!(q.epsilon().raw_dim(), input.vacuum_wavenumber().raw_dim());
        assert_eq!(q.mu().raw_dim(), input.vacuum_wavenumber().raw_dim());
        assert_eq!(q.kappa().raw_dim(), input.vacuum_wavenumber().raw_dim());
        assert_eq!(q.factor().raw_dim(), input.vacuum_wavenumber().raw_dim());
    }
}

#[cfg(test)]
mod branch_tests {
    use approx::assert_relative_eq;
    use num_complex::Complex64;

    use super::*;

    type C = Complex64;

    fn assert_complex_close(actual: C, expected: C, tolerance: f64) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = tolerance,
            max_relative = tolerance,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = tolerance,
            max_relative = tolerance,
        );
    }

    #[test]
    fn positive_real_argument_selects_positive_real_root() {
        let kappa = principal_normal_wavenumber(C::new(9.0, 0.0));

        assert_complex_close(kappa, C::new(3.0, 0.0), 1e-12);
    }

    #[test]
    fn negative_real_argument_selects_positive_imaginary_root() {
        let kappa = principal_normal_wavenumber(C::new(-9.0, 0.0));

        assert_complex_close(kappa, C::new(0.0, 3.0), 1e-12);
    }

    #[test]
    fn selected_root_squares_to_original_argument() {
        let values = [
            C::new(9.0, 0.0),
            C::new(-9.0, 0.0),
            C::new(2.0, 3.0),
            C::new(-2.0, 3.0),
            C::new(-2.0, -3.0),
            C::new(2.0, -3.0),
        ];

        for value in values {
            let kappa = principal_normal_wavenumber(value);

            assert_complex_close(kappa * kappa, value, 1e-12);
        }
    }

    #[test]
    fn selected_root_is_exactly_complex_scalar_principal_sqrt() {
        let values = [
            C::new(1.3, 0.7),
            C::new(-1.3, 0.7),
            C::new(-1.3, -0.7),
            C::new(1.3, -0.7),
        ];

        for value in values {
            assert_eq!(principal_normal_wavenumber(value), value.sqrt(),);
        }
    }

    #[test]
    fn local_derivative_matches_principal_sqrt_derivative_away_from_cut() {
        let value = C::new(2.0, 1.5);
        let direction = C::new(0.3, -0.2);
        let h = 1e-6;

        let plus = principal_normal_wavenumber(value + direction * h);

        let minus = principal_normal_wavenumber(value - direction * h);

        let finite_difference = (plus - minus) / (2.0 * h);

        let kappa = principal_normal_wavenumber(value);

        let expected = direction / (C::new(2.0, 0.0) * kappa);

        assert_complex_close(finite_difference, expected, 1e-9);
    }

    #[test]
    fn values_on_opposite_sides_of_cut_select_opposite_real_parts() {
        let epsilon = 1e-8;

        let above = principal_normal_wavenumber(C::new(-1.0, epsilon));

        let below = principal_normal_wavenumber(C::new(-1.0, -epsilon));

        assert!(above.re > 0.0);
        assert!(above.im > 0.0);

        assert!(below.re > 0.0);
        assert!(below.im < 0.0);

        assert_complex_close(below, above.conj(), 1e-10);
    }
}
