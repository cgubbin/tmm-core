//! Shared quantities for isotropic planar backends.
//!
//! This module evaluates the material and propagation quantities reused by the
//! isotropic 2×2 transfer- and scattering-matrix backends.
//!
//! Evaluation occurs in two stages.
//!
//! First, [`IsotropicMediumQuantities`] evaluates the polarization-independent
//! quantities
//!
//! ```text
//! ε, μ, κ
//! ```
//!
//! where the normal angular wavenumber is
//!
//! ```text
//! κ² = ε μ k₀² - k∥².
//! ```
//!
//! Second, selecting a TE or TM polarization produces
//! [`IsotropicLayerQuantities`], which additionally stores the corresponding
//! characteristic admittance
//!
//! ```text
//! Y = κ / factor,
//! ```
//!
//! with
//!
//! ```text
//! factor = μ    for TE
//! factor = ε    for TM.
//! ```
//!
//! Separating these stages keeps normal-wavevector evaluation independent of
//! polarization while allowing backends to cache the admittance rather than
//! recomputing it throughout matrix construction.
//!
//! # Normal-wavenumber branch
//!
//! For finite isotropic media, the normal angular wavenumber is evaluated as
//!
//! ```text
//! κ = sqrt(ε μ k₀² - k∥²)
//! ```
//!
//! using the principal complex square root supplied by
//! [`nalgebra::ComplexField`].
//!
//! No additional pointwise sign correction is applied. The principal square
//! root is analytic away from its branch cut and branch point, so derivatives
//! propagated through this module are local derivatives on that selected
//! branch.
//!
//! For real passive scattering problems, this convention gives:
//!
//! - `κ >= 0` for propagating waves with positive real `κ²`;
//! - `Im(κ) >= 0` for evanescent waves with negative real `κ²`.
//!
//! Finite-layer normal wavenumbers always use this convention.
//!
//! Exterior normal wavenumbers passed to a backend are represented separately
//! by [`ExteriorWavevectors`](crate::backend::ExteriorWavevectors). Ordinary
//! plane-wave evaluation constructs those values using the same principal-root
//! convention, while advanced complex-plane callers may supply externally
//! branch-selected exterior values.

use ndarray::Dimension;

use crate::{
    ComplexScalar,
    algebra::ScalarAlgebra,
    input::{CanonicalCoordinates, Polarisation},
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};

/// Polarization-independent material and propagation quantities for one
/// isotropic medium.
///
/// The contained algebraic values have the same sampled dimension and
/// derivative structure as the canonical coordinates used for evaluation.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct IsotropicMediumQuantities<A> {
    epsilon: A,
    mu: A,
    kappa: A,
}

impl<A> IsotropicMediumQuantities<A> {
    /// Construct already-evaluated medium quantities.
    pub(crate) fn from_parts(epsilon: A, mu: A, kappa: A) -> Self {
        Self { epsilon, mu, kappa }
    }

    /// Return the relative permittivity.
    pub(crate) fn epsilon(&self) -> &A {
        &self.epsilon
    }

    /// Return the relative permeability.
    pub(crate) fn mu(&self) -> &A {
        &self.mu
    }

    /// Return the selected normal angular wavenumber `κ`.
    pub(crate) fn kappa(&self) -> &A {
        &self.kappa
    }

    /// Consume the quantities and return `(epsilon, mu, kappa)`.
    pub(crate) fn into_parts(self) -> (A, A, A) {
        (self.epsilon, self.mu, self.kappa)
    }

    /// Consume the quantities and return the selected normal angular
    /// wavenumber.
    pub(crate) fn into_kappa(self) -> A {
        self.kappa
    }

    /// Select a polarization and construct the corresponding layer quantities.
    ///
    /// The characteristic admittance is evaluated once and retained in the
    /// returned object.
    pub(crate) fn with_polarisation(self, polarisation: Polarisation) -> IsotropicLayerQuantities<A>
    where
        A: ScalarAlgebra,
    {
        let admittance = match polarisation {
            Polarisation::TransverseElectric => self.kappa.divide(&self.mu),

            Polarisation::TransverseMagnetic => self.kappa.divide(&self.epsilon),
        };

        IsotropicLayerQuantities {
            medium: self,
            admittance,
            polarisation,
        }
    }

    pub(crate) fn admittance_with_kappa(&self, kappa: &A, polarisation: Polarisation) -> A
    where
        A: ScalarAlgebra,
    {
        match polarisation {
            Polarisation::TransverseElectric => kappa.divide(self.mu()),

            Polarisation::TransverseMagnetic => kappa.divide(self.epsilon()),
        }
    }
}

impl<A> IsotropicMediumQuantities<A> {
    /// Evaluate polarization-independent quantities for one isotropic medium.
    pub(crate) fn evaluate<E, M>(material: &M, coordinates: &CanonicalCoordinates<A>) -> Self
    where
        A: ScalarAlgebra + ConstitutiveLift<E, M>,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
        E: ConstitutiveEvaluator<A::Scalar, A::Dimension, M>,
    {
        let epsilon = A::relative_permittivity(material, coordinates.vacuum_angular_wavenumber());

        let mu = A::relative_permeability(material, coordinates.vacuum_angular_wavenumber());

        let k0_squared = coordinates
            .vacuum_angular_wavenumber()
            .multiply(coordinates.vacuum_angular_wavenumber());

        let k_parallel_squared = coordinates
            .parallel_angular_wavenumber()
            .multiply(coordinates.parallel_angular_wavenumber());

        let kappa = epsilon
            .multiply(&mu)
            .multiply(&k0_squared)
            .subtract(&k_parallel_squared)
            .sqrt();

        Self::from_parts(epsilon, mu, kappa)
    }
}

/// Polarization-specialized quantities for one isotropic layer.
///
/// This combines the polarization-independent medium quantities with the
/// characteristic admittance appropriate to the selected TE or TM problem.
///
/// The admittance is cached when the polarization is selected and reused by
/// subsequent backend calculations.
#[doc(hidden)]
#[derive(Clone, Debug, PartialEq)]
pub struct IsotropicLayerQuantities<A> {
    medium: IsotropicMediumQuantities<A>,
    admittance: A,
    polarisation: Polarisation,
}

impl<A> IsotropicLayerQuantities<A> {
    /// Return the polarization-independent medium quantities.
    pub(crate) fn medium(&self) -> &IsotropicMediumQuantities<A> {
        &self.medium
    }

    /// Return the relative permittivity.
    pub(crate) fn epsilon(&self) -> &A {
        self.medium.epsilon()
    }

    /// Return the relative permeability.
    pub(crate) fn mu(&self) -> &A {
        self.medium.mu()
    }

    /// Return the selected normal angular wavenumber `κ`.
    pub(crate) fn kappa(&self) -> &A {
        self.medium.kappa()
    }

    /// Return the selected polarization.
    pub(crate) fn polarisation(&self) -> Polarisation {
        self.polarisation
    }

    /// Return the polarization-dependent characteristic factor.
    ///
    /// The factor is `μ` for TE and `ε` for TM.
    pub(crate) fn factor(&self) -> &A {
        match self.polarisation {
            Polarisation::TransverseElectric => self.mu(),
            Polarisation::TransverseMagnetic => self.epsilon(),
        }
    }

    /// Return the cached characteristic admittance.
    pub(crate) fn admittance(&self) -> &A {
        &self.admittance
    }

    /// Consume the quantities and return the cached characteristic admittance.
    pub(crate) fn into_admittance(self) -> A {
        self.admittance
    }
}

impl<A> IsotropicLayerQuantities<A> {
    /// Evaluate medium quantities and specialize them to `polarisation`.
    pub(crate) fn evaluate<E, M>(
        material: &M,
        coordinates: &CanonicalCoordinates<A>,
        polarisation: Polarisation,
    ) -> Self
    where
        A: ScalarAlgebra + ConstitutiveLift<E, M>,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
        E: ConstitutiveEvaluator<A::Scalar, A::Dimension, M>,
    {
        IsotropicMediumQuantities::evaluate::<E, M>(material, coordinates)
            .with_polarisation(polarisation)
    }
}

#[cfg(test)]
impl<A> IsotropicLayerQuantities<A>
where
    A: ScalarAlgebra,
{
    pub(crate) fn test_fixture(kappa: A, epsilon: A, mu: A, polarisation: Polarisation) -> Self {
        IsotropicMediumQuantities::from_parts(epsilon, mu, kappa).with_polarisation(polarisation)
    }

    pub(crate) fn from_parts(kappa: A, epsilon: A, mu: A, polarisation: Polarisation) -> Self
    where
        A: ScalarAlgebra,
    {
        IsotropicMediumQuantities::from_parts(epsilon, mu, kappa).with_polarisation(polarisation)
    }

    /// Evaluate polarization-specialized quantities on the real spectral axis.
    pub(crate) fn real_axis<M>(
        material: &M,
        coordinates: &CanonicalCoordinates<A>,
        polarisation: Polarisation,
    ) -> Self
    where
        A: ScalarAlgebra + ConstitutiveLift<crate::RealAxis, M>,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
        crate::RealAxis: ConstitutiveEvaluator<A::Scalar, A::Dimension, M>,
    {
        IsotropicMediumQuantities::real_axis(material, coordinates).with_polarisation(polarisation)
    }

    /// Evaluate polarization-specialized quantities in the complex spectral
    /// plane.
    pub(crate) fn complex_plane<M>(
        material: &M,
        coordinates: &CanonicalCoordinates<A>,
        polarisation: Polarisation,
    ) -> Self
    where
        A: ScalarAlgebra + ConstitutiveLift<crate::ComplexPlane, M>,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
        crate::ComplexPlane: ConstitutiveEvaluator<A::Scalar, A::Dimension, M>,
    {
        IsotropicMediumQuantities::complex_plane(material, coordinates)
            .with_polarisation(polarisation)
    }
}

#[cfg(test)]
impl<A> IsotropicMediumQuantities<A> {
    /// Evaluate medium quantities on the real spectral axis.
    pub(crate) fn real_axis<M>(material: &M, coordinates: &CanonicalCoordinates<A>) -> Self
    where
        A: ScalarAlgebra + ConstitutiveLift<crate::RealAxis, M>,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
        crate::RealAxis: ConstitutiveEvaluator<A::Scalar, A::Dimension, M>,
    {
        Self::evaluate::<crate::RealAxis, M>(material, coordinates)
    }

    /// Evaluate medium quantities in the complex spectral plane.
    pub(crate) fn complex_plane<M>(material: &M, coordinates: &CanonicalCoordinates<A>) -> Self
    where
        A: ScalarAlgebra + ConstitutiveLift<crate::ComplexPlane, M>,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
        crate::ComplexPlane: ConstitutiveEvaluator<A::Scalar, A::Dimension, M>,
    {
        Self::evaluate::<crate::ComplexPlane, M>(material, coordinates)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array0, Array1, arr0, array};

    use crate::{
        algebra::Jet0,
        backend::isotropic::{IsotropicLayerQuantities, IsotropicMediumQuantities},
        input::{CanonicalCoordinates, Polarisation},
        test_support::{
            C,
            assertions::{assert_complex_close, assert_dispersion_relation},
            c,
            expected::{
                linear_admittance, linear_epsilon, linear_kappa, linear_mu, quadratic_admittance,
                quadratic_epsilon, quadratic_kappa, quadratic_mu,
            },
            materials::{constant, linear, quadratic, vacuum},
        },
    };

    fn scalar_coordinates(
        vacuum_angular_wavenumber: f64,
        parallel_angular_wavenumber: f64,
    ) -> CanonicalCoordinates<Jet0<Array0<C>>> {
        CanonicalCoordinates::new(
            Jet0::new(arr0(c(vacuum_angular_wavenumber))),
            Jet0::new(arr0(c(parallel_angular_wavenumber))),
        )
    }

    fn sampled_coordinates(
        vacuum_angular_wavenumber: Array1<C>,
        parallel_angular_wavenumber: Array1<C>,
    ) -> CanonicalCoordinates<Jet0<Array1<C>>> {
        CanonicalCoordinates::new(
            Jet0::new(vacuum_angular_wavenumber),
            Jet0::new(parallel_angular_wavenumber),
        )
    }

    // ---------------------------------------------------------------------
    // Polarization-independent medium quantities
    // ---------------------------------------------------------------------

    #[test]
    fn medium_quantities_store_evaluated_material_properties() {
        let material = constant(4.0, 2.0);
        let coordinates = scalar_coordinates(3.0, 1.0);

        let quantities = IsotropicMediumQuantities::real_axis(&material, &coordinates);

        assert_complex_close(quantities.epsilon()[()], c(4.0), 1e-12);

        assert_complex_close(quantities.mu()[()], c(2.0), 1e-12);
    }

    #[test]
    fn medium_into_parts_preserves_component_order() {
        let material = constant(4.0, 2.0);
        let coordinates = scalar_coordinates(3.0, 1.0);

        let quantities = IsotropicMediumQuantities::real_axis(&material, &coordinates);

        let expected_epsilon = quantities.epsilon().clone();
        let expected_mu = quantities.mu().clone();
        let expected_kappa = quantities.kappa().clone();

        let (epsilon, mu, kappa) = quantities.into_parts();

        assert_eq!(epsilon, expected_epsilon);
        assert_eq!(mu, expected_mu);
        assert_eq!(kappa, expected_kappa);
    }

    #[test]
    fn computes_normal_wavenumber() {
        let material = constant(4.0, 2.0);
        let coordinates = scalar_coordinates(3.0, 1.0);

        let quantities = IsotropicMediumQuantities::real_axis(&material, &coordinates);

        let expected = c((4.0_f64 * 2.0 * 3.0_f64.powi(2) - 1.0_f64.powi(2)).sqrt());

        assert_complex_close(quantities.kappa()[()], expected, 1e-12);
    }

    #[test]
    fn normal_wavenumber_satisfies_dispersion_relation() {
        let material = constant(3.5, 1.4);

        let k0 = 2.3;
        let k_parallel = 0.7;

        let coordinates = scalar_coordinates(k0, k_parallel);

        let quantities = IsotropicMediumQuantities::real_axis(&material, &coordinates);

        assert_dispersion_relation(
            quantities.epsilon()[()],
            quantities.mu()[()],
            quantities.kappa()[()],
            k0,
            k_parallel,
            1e-12,
        );
    }

    #[test]
    fn grazing_incidence_has_zero_normal_wavenumber() {
        let material = vacuum();
        let coordinates = scalar_coordinates(2.0, 2.0);

        let quantities = IsotropicMediumQuantities::real_axis(&material, &coordinates);

        assert_complex_close(quantities.kappa()[()], C::new(0.0, 0.0), 1e-12);
    }

    #[test]
    fn evanescent_wave_uses_positive_imaginary_branch() {
        let material = vacuum();
        let coordinates = scalar_coordinates(1.0, 2.0);

        let quantities = IsotropicMediumQuantities::real_axis(&material, &coordinates);

        assert_relative_eq!(quantities.kappa()[()].re, 0.0, epsilon = 1e-12,);

        assert!(quantities.kappa()[()].im > 0.0);

        assert_complex_close(quantities.kappa()[()], C::new(0.0, 3.0_f64.sqrt()), 1e-12);
    }

    #[test]
    fn sampled_coordinates_are_evaluated_pointwise() {
        let material = constant(4.0, 1.0);

        let coordinates = sampled_coordinates(
            array![c(1.0), c(2.0), c(3.0)],
            array![c(0.0), c(1.0), c(2.0)],
        );

        let quantities = IsotropicMediumQuantities::real_axis(&material, &coordinates);

        let expected = [
            (4.0_f64 * 1.0_f64.powi(2) - 0.0_f64.powi(2)).sqrt(),
            (4.0_f64 * 2.0_f64.powi(2) - 1.0_f64.powi(2)).sqrt(),
            (4.0_f64 * 3.0_f64.powi(2) - 2.0_f64.powi(2)).sqrt(),
        ];

        for (actual, expected) in quantities.kappa().iter().zip(expected) {
            assert_complex_close(*actual, c(expected), 1e-12);
        }
    }

    // ---------------------------------------------------------------------
    // Dispersive material evaluation
    // ---------------------------------------------------------------------

    #[test]
    fn linear_dispersion_is_evaluated_at_vacuum_wavenumber() {
        let material = linear(2.0, 0.5, 3.0, -0.2);

        let k0 = 4.0;

        let coordinates = scalar_coordinates(k0, 0.7);

        let quantities = IsotropicMediumQuantities::real_axis(&material, &coordinates);

        assert_complex_close(
            quantities.epsilon()[()],
            linear_epsilon(&material, k0),
            1e-12,
        );

        assert_complex_close(quantities.mu()[()], linear_mu(&material, k0), 1e-12);
    }

    #[test]
    fn linear_dispersion_changes_with_vacuum_wavenumber() {
        let material = linear(2.0, 0.5, 3.0, -0.2);

        let first_coordinates = scalar_coordinates(1.0, 0.25);

        let second_coordinates = scalar_coordinates(3.0, 0.25);

        let first = IsotropicMediumQuantities::real_axis(&material, &first_coordinates);

        let second = IsotropicMediumQuantities::real_axis(&material, &second_coordinates);

        assert_ne!(first.epsilon()[()], second.epsilon()[()],);

        assert_ne!(first.mu()[()], second.mu()[()],);
    }

    #[test]
    fn linear_dispersion_is_used_when_computing_kappa() {
        let material = linear(2.0, 0.5, 1.5, -0.1);

        let k0 = 2.0;
        let k_parallel = 0.7;

        let coordinates = scalar_coordinates(k0, k_parallel);

        let quantities = IsotropicMediumQuantities::real_axis(&material, &coordinates);

        assert_complex_close(
            quantities.kappa()[()],
            linear_kappa(&material, k0, k_parallel),
            1e-12,
        );

        assert_dispersion_relation(
            quantities.epsilon()[()],
            quantities.mu()[()],
            quantities.kappa()[()],
            k0,
            k_parallel,
            1e-12,
        );
    }

    #[test]
    fn quadratic_dispersion_is_evaluated_at_vacuum_wavenumber() {
        let material = quadratic(1.0, 2.0, 3.0, 4.0, 5.0, 6.0);

        let k0 = 2.0;

        let coordinates = scalar_coordinates(k0, 0.5);

        let quantities = IsotropicMediumQuantities::real_axis(&material, &coordinates);

        assert_complex_close(
            quantities.epsilon()[()],
            quadratic_epsilon(&material, k0),
            1e-12,
        );

        assert_complex_close(quantities.mu()[()], quadratic_mu(&material, k0), 1e-12);

        assert_complex_close(quantities.epsilon()[()], c(17.0), 1e-12);

        assert_complex_close(quantities.mu()[()], c(38.0), 1e-12);
    }

    #[test]
    fn quadratic_dispersion_is_used_when_computing_kappa() {
        let material = quadratic(2.0, 0.4, 0.1, 1.5, -0.2, 0.05);

        let k0 = 2.5;
        let k_parallel = 0.8;

        let coordinates = scalar_coordinates(k0, k_parallel);

        let quantities = IsotropicMediumQuantities::real_axis(&material, &coordinates);

        assert_complex_close(
            quantities.kappa()[()],
            quadratic_kappa(&material, k0, k_parallel),
            1e-12,
        );

        assert_dispersion_relation(
            quantities.epsilon()[()],
            quantities.mu()[()],
            quantities.kappa()[()],
            k0,
            k_parallel,
            1e-12,
        );
    }

    #[test]
    fn medium_real_axis_and_complex_plane_agree_on_real_coordinates() {
        let material = quadratic(2.0, 0.4, 0.1, 1.5, -0.2, 0.05);

        let coordinates = scalar_coordinates(2.5, 0.8);

        let real_axis = IsotropicMediumQuantities::real_axis(&material, &coordinates);

        let complex_plane = IsotropicMediumQuantities::complex_plane(&material, &coordinates);

        assert_complex_close(real_axis.epsilon()[()], complex_plane.epsilon()[()], 1e-12);

        assert_complex_close(real_axis.mu()[()], complex_plane.mu()[()], 1e-12);

        assert_complex_close(real_axis.kappa()[()], complex_plane.kappa()[()], 1e-12);
    }

    // ---------------------------------------------------------------------
    // Polarization-specialized layer quantities
    // ---------------------------------------------------------------------

    #[test]
    fn layer_quantities_preserve_selected_polarisation() {
        let material = vacuum();
        let coordinates = scalar_coordinates(2.0, 0.5);

        let te = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
        );

        let tm = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseMagnetic,
        );

        assert_eq!(te.polarisation(), Polarisation::TransverseElectric,);

        assert_eq!(tm.polarisation(), Polarisation::TransverseMagnetic,);
    }

    #[test]
    fn te_factor_is_permeability() {
        let material = constant(4.0, 2.0);
        let coordinates = scalar_coordinates(3.0, 1.0);

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
        );

        assert_complex_close(quantities.factor()[()], quantities.mu()[()], 1e-12);
    }

    #[test]
    fn tm_factor_is_permittivity() {
        let material = constant(4.0, 2.0);
        let coordinates = scalar_coordinates(3.0, 1.0);

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseMagnetic,
        );

        assert_complex_close(quantities.factor()[()], quantities.epsilon()[()], 1e-12);
    }

    #[test]
    fn te_admittance_is_cached_kappa_over_mu() {
        let material = constant(4.0, 2.0);
        let coordinates = scalar_coordinates(3.0, 1.0);

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
        );

        let expected = quantities.kappa()[()] / quantities.mu()[()];

        assert_complex_close(quantities.admittance()[()], expected, 1e-12);
    }

    #[test]
    fn tm_admittance_is_cached_kappa_over_epsilon() {
        let material = constant(4.0, 2.0);
        let coordinates = scalar_coordinates(3.0, 1.0);

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseMagnetic,
        );

        let expected = quantities.kappa()[()] / quantities.epsilon()[()];

        assert_complex_close(quantities.admittance()[()], expected, 1e-12);
    }

    #[test]
    fn selecting_polarisation_does_not_change_medium_quantities() {
        let material = constant(4.0, 2.0);
        let coordinates = scalar_coordinates(3.0, 1.0);

        let medium = IsotropicMediumQuantities::real_axis(&material, &coordinates);

        let expected_epsilon = medium.epsilon().clone();
        let expected_mu = medium.mu().clone();
        let expected_kappa = medium.kappa().clone();

        let te = medium
            .clone()
            .with_polarisation(Polarisation::TransverseElectric);

        let tm = medium.with_polarisation(Polarisation::TransverseMagnetic);

        assert_eq!(te.epsilon(), &expected_epsilon,);

        assert_eq!(tm.epsilon(), &expected_epsilon,);

        assert_eq!(te.mu(), &expected_mu,);

        assert_eq!(tm.mu(), &expected_mu,);

        assert_eq!(te.kappa(), &expected_kappa,);

        assert_eq!(tm.kappa(), &expected_kappa,);
    }

    #[test]
    fn cached_admittance_matches_expected_dispersive_te_value() {
        let material = linear(2.0, 0.5, 1.5, -0.1);

        let k0 = 2.0;
        let k_parallel = 0.7;

        let coordinates = scalar_coordinates(k0, k_parallel);

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
        );

        assert_complex_close(
            quantities.admittance()[()],
            linear_admittance(&material, k0, k_parallel, Polarisation::TransverseElectric),
            1e-12,
        );
    }

    #[test]
    fn cached_admittance_matches_expected_dispersive_tm_value() {
        let material = quadratic(2.0, 0.4, 0.1, 1.5, -0.2, 0.05);

        let k0 = 2.5;
        let k_parallel = 0.8;

        let coordinates = scalar_coordinates(k0, k_parallel);

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseMagnetic,
        );

        assert_complex_close(
            quantities.admittance()[()],
            quadratic_admittance(&material, k0, k_parallel, Polarisation::TransverseMagnetic),
            1e-12,
        );
    }

    #[test]
    fn layer_real_axis_and_complex_plane_agree_on_real_coordinates() {
        let material = quadratic(2.0, 0.4, 0.1, 1.5, -0.2, 0.05);

        let coordinates = scalar_coordinates(2.5, 0.8);

        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            let real_axis =
                IsotropicLayerQuantities::real_axis(&material, &coordinates, polarisation);

            let complex_plane =
                IsotropicLayerQuantities::complex_plane(&material, &coordinates, polarisation);

            assert_complex_close(real_axis.epsilon()[()], complex_plane.epsilon()[()], 1e-12);

            assert_complex_close(real_axis.mu()[()], complex_plane.mu()[()], 1e-12);

            assert_complex_close(real_axis.kappa()[()], complex_plane.kappa()[()], 1e-12);

            assert_complex_close(
                real_axis.admittance()[()],
                complex_plane.admittance()[()],
                1e-12,
            );
        }
    }
}
