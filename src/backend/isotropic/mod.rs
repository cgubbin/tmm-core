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

mod admittance;
mod derivatives;

use ndarray::{ArrayBase, Dimension, OwnedRepr};

pub(crate) use admittance::IsotropicLayerAdmittance;
pub(crate) use derivatives::{IsotropicLayerFirstDerivatives, IsotropicLayerSecondDerivatives};

use crate::{
    ComplexScalar,
    backend::{PlanarInput, Polarisation},
    material::{Material, Scalar},
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
    /// Evaluate the isotropic material and propagation quantities.
    ///
    /// The material model is sampled at the input vacuum wavenumber `k₀`.
    /// Both input coordinates must use the same inverse-length unit.
    ///
    /// The normal-wavenumber branch is selected by
    /// [`outgoing_normal_wavenumber`].
    pub(crate) fn new<M>(material: &M, planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>) -> Self
    where
        M: Material<Real = C::RealField>,
    {
        let epsilon = planar
            .vacuum_wavenumber()
            .mapv(|k0| material.relative_permittivity(Scalar(k0)));

        let mu = planar
            .vacuum_wavenumber()
            .mapv(|k0| material.relative_permeability(Scalar(k0)));

        let vacuum_wavenumber_squared = planar.vacuum_wavenumber().mapv(|k0| k0 * k0);

        let parallel_wavenumber_squared = planar
            .parallel_wavenumber()
            .mapv(|k_parallel| k_parallel * k_parallel);

        let kappa_squared =
            epsilon.clone() * mu.clone() * vacuum_wavenumber_squared - parallel_wavenumber_squared;

        let kappa = kappa_squared.mapv(outgoing_normal_wavenumber);

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

/// Select the outgoing or decaying normal-wavenumber branch.
///
/// The principal square root already has a nonnegative imaginary part for
/// values away from its branch cut. The explicit correction below documents
/// and enforces the backend convention:
///
/// - evanescent/passive waves satisfy `Im(κ) >= 0`;
/// - when `Im(κ) == 0`, propagating waves satisfy `Re(κ) >= 0`.
fn outgoing_normal_wavenumber<C>(kappa_squared: C) -> C
where
    C: ComplexScalar,
{
    let kappa = kappa_squared.sqrt();

    let imaginary = kappa.imaginary();
    let real = kappa.real();

    if imaginary < C::zero().real() || (imaginary == C::zero().real() && real < C::zero().real()) {
        -kappa
    } else {
        kappa
    }
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

        let q = IsotropicLayerQuantities::new(&material, &input);

        let expected = c((4.0_f64 * 9.0 - 4.0).sqrt());

        assert_close(q.kappa()[()], expected);
    }

    #[test]
    fn te_factor_is_permeability() {
        let material = material(4.0, 2.0);
        let input = scalar_input(3.0, 1.0, Polarisation::TransverseElectric);

        let q = IsotropicLayerQuantities::new(&material, &input);

        assert_close(q.factor()[()], c(2.0));
    }

    #[test]
    fn tm_factor_is_permittivity() {
        let material = material(4.0, 2.0);
        let input = scalar_input(3.0, 1.0, Polarisation::TransverseMagnetic);

        let q = IsotropicLayerQuantities::new(&material, &input);

        assert_close(q.factor()[()], c(4.0));
    }

    #[test]
    fn evanescent_normal_wavenumber_is_decaying() {
        let material = material(1.0, 1.0);
        let input = scalar_input(1.0, 2.0, Polarisation::TransverseElectric);

        let q = IsotropicLayerQuantities::new(&material, &input);

        assert_relative_eq!(q.kappa()[()].re, 0.0, epsilon = 1e-12);
        assert!(q.kappa()[()].im > 0.0);
    }

    #[test]
    fn admittance_matches_kappa_over_factor() {
        let material = material(4.0, 2.0);
        let input = scalar_input(3.0, 1.0, Polarisation::TransverseMagnetic);

        let q = IsotropicLayerQuantities::new(&material, &input);
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

        let q = IsotropicLayerQuantities::new(&material, &input);

        assert_eq!(q.epsilon().raw_dim(), input.vacuum_wavenumber().raw_dim());
        assert_eq!(q.mu().raw_dim(), input.vacuum_wavenumber().raw_dim());
        assert_eq!(q.kappa().raw_dim(), input.vacuum_wavenumber().raw_dim());
        assert_eq!(q.factor().raw_dim(), input.vacuum_wavenumber().raw_dim());
    }
}
