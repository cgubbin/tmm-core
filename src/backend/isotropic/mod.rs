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
//! using the principal complex square root supplied by [`ComplexField`].
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

use nalgebra::ComplexField;
use ndarray::Dimension;

pub(crate) use admittance::IsotropicLayerAdmittance;

use crate::{
    ComplexScalar,
    algebra::ScalarAlgebra,
    domain::{ComplexPlane, RealAxis},
    input::{CanonicalCoordinates, CanonicalSolverInput, Polarisation},
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};

/// Material and propagation quantities for one isotropic medium.
///
/// Every array has the same sampled dimension as the corresponding
/// [`PlanarInput`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IsotropicLayerQuantities<A> {
    epsilon: A,
    mu: A,
    kappa: A,
    polarisation: Polarisation,
}

impl<A> IsotropicLayerQuantities<A> {
    pub(crate) fn from_parts(epsilon: A, mu: A, kappa: A, polarisation: Polarisation) -> Self {
        Self {
            epsilon,
            mu,
            kappa,
            polarisation,
        }
    }

    /// Consume the derivatives and return their components.
    pub(crate) fn into_parts(self) -> (A, A, A, Polarisation) {
        (self.epsilon, self.mu, self.kappa, self.polarisation)
    }

    /// Return the relative permittivity.
    pub(crate) fn epsilon(&self) -> &A {
        &self.epsilon
    }

    /// Return the relative permeability.
    pub(crate) fn mu(&self) -> &A {
        &self.mu
    }

    /// Return the selected normal wavenumber `κ`.
    pub(crate) fn kappa(&self) -> &A {
        &self.kappa
    }

    /// Return the polarisation used
    pub(crate) fn polarisation(&self) -> Polarisation {
        self.polarisation
    }

    /// Return the TE/TM characteristic factor.
    pub(crate) fn factor(&self) -> &A {
        match self.polarisation {
            Polarisation::TransverseElectric => &self.mu,
            Polarisation::TransverseMagnetic => &self.epsilon,
        }
    }

    pub(crate) fn into_admittance(self) -> IsotropicLayerAdmittance<A>
    where
        A: ScalarAlgebra,
    {
        IsotropicLayerAdmittance::new(self.kappa.divide(self.factor()))
    }
}

impl<A> IsotropicLayerQuantities<A> {
    pub(crate) fn real_axis<M>(
        material: &M,
        coordinates: &CanonicalCoordinates<A>,
        polarisation: Polarisation,
    ) -> Self
    where
        A: ScalarAlgebra + ConstitutiveLift<RealAxis, M> + Clone,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
        RealAxis: ConstitutiveEvaluator<A::Scalar, A::Dimension, M>,
    {
        Self::evaluate::<RealAxis, M>(material, coordinates, polarisation)
    }

    pub(crate) fn complex_plane<M>(
        material: &M,
        coordinates: &CanonicalCoordinates<A>,
        polarisation: Polarisation,
    ) -> Self
    where
        A: ScalarAlgebra + ConstitutiveLift<ComplexPlane, M> + Clone,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
        ComplexPlane: ConstitutiveEvaluator<A::Scalar, A::Dimension, M>,
    {
        Self::evaluate::<ComplexPlane, M>(material, coordinates, polarisation)
    }

    pub(crate) fn evaluate<E, M>(
        material: &M,
        coordinates: &CanonicalCoordinates<A>,
        polarisation: Polarisation,
    ) -> Self
    where
        A: ScalarAlgebra + ConstitutiveLift<E, M> + Clone,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
        E: ConstitutiveEvaluator<A::Scalar, A::Dimension, M>,
    {
        let epsilon = A::relative_permittivity(material, coordinates.vacuum_angular_wavenumber());

        let mu = A::relative_permeability(material, coordinates.vacuum_angular_wavenumber());

        let k0_squared = coordinates
            .vacuum_angular_wavenumber()
            .multiply(coordinates.vacuum_angular_wavenumber());

        let kx_squared = coordinates
            .parallel_angular_wavenumber()
            .multiply(coordinates.parallel_angular_wavenumber());

        let kappa = epsilon
            .multiply(&mu)
            .multiply(&k0_squared)
            .subtract(&kx_squared)
            .sqrt();

        Self::from_parts(epsilon, mu, kappa, polarisation)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array0, Array1, arr0, array};
    use num_complex::Complex64;

    use crate::{
        algebra::Jet0,
        backend::isotropic::IsotropicLayerQuantities,
        input::{CanonicalCoordinates, CanonicalSolverInput, Polarisation},
        test_support::{
            C,
            assertions::{assert_complex_close, assert_dispersion_relation},
            c,
            expected::{
                linear_epsilon, linear_kappa, linear_mu, quadratic_epsilon, quadratic_kappa,
                quadratic_mu,
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

    #[test]
    fn stores_evaluated_material_quantities() {
        let material = constant(4.0, 2.0);
        let coordinates = scalar_coordinates(3.0, 1.0);

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
        );

        assert_complex_close(quantities.epsilon()[()], c(4.0), 1e-12);
        assert_complex_close(quantities.mu()[()], c(2.0), 1e-12);
    }

    #[test]
    fn stores_polarisation() {
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

        assert_eq!(te.polarisation(), Polarisation::TransverseElectric);
        assert_eq!(tm.polarisation(), Polarisation::TransverseMagnetic);
    }

    #[test]
    fn into_parts_returns_all_components() {
        let material = constant(4.0, 2.0);
        let coordinates = scalar_coordinates(3.0, 1.0);

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseMagnetic,
        );

        let expected_epsilon = quantities.epsilon().clone();
        let expected_mu = quantities.mu().clone();
        let expected_kappa = quantities.kappa().clone();

        let (epsilon, mu, kappa, polarisation) = quantities.into_parts();

        assert_eq!(epsilon, expected_epsilon);
        assert_eq!(mu, expected_mu);
        assert_eq!(kappa, expected_kappa);
        assert_eq!(polarisation, Polarisation::TransverseMagnetic);
    }

    #[test]
    fn computes_normal_wavenumber() {
        let material = constant(4.0, 2.0);
        let coordinates = scalar_coordinates(3.0, 1.0);

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
        );

        let expected = c((4.0_f64 * 2.0 * 3.0_f64.powi(2) - 1.0_f64.powi(2)).sqrt());

        assert_complex_close(quantities.kappa()[()], expected, 1e-12);
    }

    #[test]
    fn normal_wavenumber_satisfies_dispersion_relation() {
        let material = constant(3.5, 1.4);
        let k0 = 2.3;
        let k_parallel = 0.7;
        let coordinates = scalar_coordinates(k0, k_parallel);

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
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
    fn transverse_electric_factor_is_permeability() {
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
    fn transverse_magnetic_factor_is_permittivity() {
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
    fn grazing_incidence_has_zero_normal_wavenumber() {
        let material = vacuum();
        let coordinates = scalar_coordinates(2.0, 2.0);

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
        );

        assert_complex_close(quantities.kappa()[()], C::new(0.0, 0.0), 1e-12);
    }

    #[test]
    fn evanescent_wave_uses_positive_imaginary_branch() {
        let material = vacuum();
        let coordinates = scalar_coordinates(1.0, 2.0);

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
        );

        assert_relative_eq!(quantities.kappa()[()].re, 0.0, epsilon = 1e-12,);

        assert!(quantities.kappa()[()].im > 0.0);

        assert_complex_close(quantities.kappa()[()], C::new(0.0, 3.0_f64.sqrt()), 1e-12);
    }

    #[test]
    fn sampled_coordinates_are_evaluated_pointwise() {
        let material = constant(4.0, 1.0);

        let k0 = array![c(1.0), c(2.0), c(3.0)];
        let k_parallel = array![c(0.0), c(1.0), c(2.0)];

        let coordinates = sampled_coordinates(k0, k_parallel);

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
        );

        let expected = [
            (4.0_f64 * 1.0_f64.powi(2) - 0.0_f64.powi(2)).sqrt(),
            (4.0_f64 * 2.0_f64.powi(2) - 1.0_f64.powi(2)).sqrt(),
            (4.0_f64 * 3.0_f64.powi(2) - 2.0_f64.powi(2)).sqrt(),
        ];

        for (actual, expected) in quantities.kappa().iter().zip(expected) {
            assert_complex_close(*actual, c(expected), 1e-12);
        }
    }

    #[test]
    fn linear_dispersion_is_evaluated_at_vacuum_wavenumber() {
        let material = linear(2.0, 0.5, 3.0, -0.2);
        let k0 = 4.0;
        let coordinates = scalar_coordinates(k0, 0.7);

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
        );

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

        let first = IsotropicLayerQuantities::real_axis(
            &material,
            &first_coordinates,
            Polarisation::TransverseElectric,
        );

        let second = IsotropicLayerQuantities::real_axis(
            &material,
            &second_coordinates,
            Polarisation::TransverseElectric,
        );

        assert_ne!(first.epsilon()[()], second.epsilon()[()]);
        assert_ne!(first.mu()[()], second.mu()[()]);
    }

    #[test]
    fn linear_dispersion_is_used_when_computing_kappa() {
        let material = linear(2.0, 0.5, 1.5, -0.1);
        let k0 = 2.0;
        let k_parallel = 0.7;
        let coordinates = scalar_coordinates(k0, k_parallel);

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseMagnetic,
        );

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
        let material = quadratic(
            1.0, // ε₀
            2.0, // ε slope
            3.0, // ε curvature
            4.0, // μ₀
            5.0, // μ slope
            6.0, // μ curvature
        );

        let k0 = 2.0;
        let coordinates = scalar_coordinates(k0, 0.5);

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
        );

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

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseMagnetic,
        );

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
    fn real_axis_and_complex_plane_agree_on_real_coordinates() {
        let material = quadratic(2.0, 0.4, 0.1, 1.5, -0.2, 0.05);

        let coordinates = scalar_coordinates(2.5, 0.8);

        let real_axis = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseMagnetic,
        );

        let complex_plane = IsotropicLayerQuantities::complex_plane(
            &material,
            &coordinates,
            Polarisation::TransverseMagnetic,
        );

        assert_complex_close(real_axis.epsilon()[()], complex_plane.epsilon()[()], 1e-12);

        assert_complex_close(real_axis.mu()[()], complex_plane.mu()[()], 1e-12);

        assert_complex_close(real_axis.kappa()[()], complex_plane.kappa()[()], 1e-12);

        assert_complex_close(real_axis.factor()[()], complex_plane.factor()[()], 1e-12);

        assert_eq!(real_axis.polarisation(), complex_plane.polarisation(),);
    }
}
