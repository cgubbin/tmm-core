//! Characteristic admittance for isotropic planar media.
//!
//! For an isotropic medium, the characteristic admittance is
//!
//! ```text
//! Y = κ / factor
//! ```
//!
//! where
//!
//! ```text
//! factor = μ    for TE
//! factor = ε    for TM
//! ```
//!
//! The admittance is constructed from previously evaluated
//! [`IsotropicLayerQuantities`]. The underlying representation may contain
//! sampled values, first-order jets, or second-order jets; derivatives propagate
//! through the scalar algebra automatically.

/// Characteristic admittance of one isotropic medium.
///
/// The contained representation has the same sampled shape and derivative order
/// as the [`IsotropicLayerQuantities`] from which it was constructed.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IsotropicLayerAdmittance<A>(A);

impl<A> IsotropicLayerAdmittance<A> {
    /// Wrap an already-computed characteristic admittance.
    pub(crate) fn new(value: A) -> Self {
        Self(value)
    }

    /// Consume the wrapper and return the characteristic admittance.
    pub(crate) fn into_inner(self) -> A {
        self.0
    }
}

impl<A> AsRef<A> for IsotropicLayerAdmittance<A> {
    fn as_ref(&self) -> &A {
        &self.0
    }
}

impl<A> std::ops::Deref for IsotropicLayerAdmittance<A> {
    type Target = A;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array0, Array1, arr0, array};
    use num_complex::Complex64;

    use crate::{
        algebra::Jet0,
        backend::isotropic::IsotropicLayerQuantities,
        input::{CanonicalCoordinates, CanonicalSolverInput, Polarisation},
        test_support::{
            assertions::assert_complex_close,
            expected::{linear_admittance, quadratic_admittance},
            materials::{constant, linear, quadratic, vacuum},
        },
    };

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

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
    fn value_admittance_is_kappa_over_factor() {
        let material = constant(4.0, 2.0);
        let coordinates = scalar_coordinates(3.0, 1.0);

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseMagnetic,
        );

        let expected = quantities.kappa()[()] / quantities.factor()[()];
        let admittance = quantities.into_admittance();

        assert_complex_close(admittance[()], expected, 1e-12);
    }

    #[test]
    fn transverse_electric_and_magnetic_admittances_use_different_factors() {
        let material = constant(4.0, 2.0);
        let coordinates = scalar_coordinates(3.0, 1.0);

        let te = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
        )
        .into_admittance()
        .into_inner();

        let tm = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseMagnetic,
        )
        .into_admittance()
        .into_inner();

        let kappa = c((4.0_f64 * 2.0 * 9.0 - 1.0).sqrt());

        assert_complex_close(te[()], kappa / c(2.0), 1e-12);
        assert_complex_close(tm[()], kappa / c(4.0), 1e-12);
    }

    #[test]
    fn vacuum_te_and_tm_admittances_are_equal() {
        let material = vacuum();
        let coordinates = scalar_coordinates(3.0, 1.0);

        let te = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
        )
        .into_admittance()
        .into_inner();

        let tm = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseMagnetic,
        )
        .into_admittance()
        .into_inner();

        assert_complex_close(te[()], tm[()], 1e-12);
        assert_complex_close(te[()], c(8.0_f64.sqrt()), 1e-12);
    }

    #[test]
    fn grazing_incidence_has_zero_admittance() {
        let material = vacuum();
        let coordinates = scalar_coordinates(2.0, 2.0);

        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            let admittance =
                IsotropicLayerQuantities::real_axis(&material, &coordinates, polarisation)
                    .into_admittance()
                    .into_inner();

            assert_complex_close(admittance[()], C::new(0.0, 0.0), 1e-12);
        }
    }

    #[test]
    fn evanescent_admittance_uses_positive_imaginary_branch() {
        let material = vacuum();
        let coordinates = scalar_coordinates(1.0, 2.0);

        let admittance = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
        )
        .into_admittance()
        .into_inner();

        assert_complex_close(admittance[()], C::new(0.0, 3.0_f64.sqrt()), 1e-12);

        assert!(admittance[()].im > 0.0);
    }

    #[test]
    fn sampled_admittance_is_evaluated_pointwise() {
        let material = constant(4.0, 2.0);

        let coordinates = sampled_coordinates(
            array![c(1.0), c(2.0), c(3.0)],
            array![c(0.0), c(1.0), c(2.0)],
        );

        let admittance = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
        )
        .into_admittance()
        .into_inner();

        let expected = [
            (4.0_f64 * 2.0 * 1.0_f64.powi(2) - 0.0_f64.powi(2)).sqrt() / 2.0,
            (4.0_f64 * 2.0 * 2.0_f64.powi(2) - 1.0_f64.powi(2)).sqrt() / 2.0,
            (4.0_f64 * 2.0 * 3.0_f64.powi(2) - 2.0_f64.powi(2)).sqrt() / 2.0,
        ];

        for (actual, expected) in admittance.iter().zip(expected) {
            assert_complex_close(*actual, c(expected), 1e-12);
        }
    }

    #[test]
    fn linear_dispersion_te_admittance_uses_lifted_material() {
        let material = linear(2.0, 0.5, 1.5, -0.1);

        let k0 = 2.0;
        let k_parallel = 0.7;
        let coordinates = scalar_coordinates(k0, k_parallel);

        let admittance = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
        )
        .into_admittance()
        .into_inner();

        assert_complex_close(
            admittance[()],
            linear_admittance(&material, k0, k_parallel, Polarisation::TransverseElectric),
            1e-12,
        );
    }

    #[test]
    fn linear_dispersion_tm_admittance_uses_lifted_material() {
        let material = linear(2.0, 0.5, 1.5, -0.1);

        let k0 = 2.0;
        let k_parallel = 0.7;
        let coordinates = scalar_coordinates(k0, k_parallel);

        let admittance = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseMagnetic,
        )
        .into_admittance()
        .into_inner();

        assert_complex_close(
            admittance[()],
            linear_admittance(&material, k0, k_parallel, Polarisation::TransverseMagnetic),
            1e-12,
        );
    }

    #[test]
    fn linear_admittance_changes_with_vacuum_wavenumber() {
        let material = linear(2.0, 0.5, 1.5, -0.1);

        let first_coordinates = scalar_coordinates(1.0, 0.25);
        let second_coordinates = scalar_coordinates(3.0, 0.25);

        let first = IsotropicLayerQuantities::real_axis(
            &material,
            &first_coordinates,
            Polarisation::TransverseMagnetic,
        )
        .into_admittance()
        .into_inner();

        let second = IsotropicLayerQuantities::real_axis(
            &material,
            &second_coordinates,
            Polarisation::TransverseMagnetic,
        )
        .into_admittance()
        .into_inner();

        assert_ne!(first[()], second[()]);
    }

    #[test]
    fn quadratic_dispersion_te_admittance_is_correct() {
        let material = quadratic(2.0, 0.4, 0.1, 1.5, -0.2, 0.05);

        let k0 = 2.5;
        let k_parallel = 0.8;
        let coordinates = scalar_coordinates(k0, k_parallel);

        let admittance = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
        )
        .into_admittance()
        .into_inner();

        assert_complex_close(
            admittance[()],
            quadratic_admittance(&material, k0, k_parallel, Polarisation::TransverseElectric),
            1e-12,
        );
    }

    #[test]
    fn quadratic_dispersion_tm_admittance_is_correct() {
        let material = quadratic(2.0, 0.4, 0.1, 1.5, -0.2, 0.05);

        let k0 = 2.5;
        let k_parallel = 0.8;
        let coordinates = scalar_coordinates(k0, k_parallel);

        let admittance = IsotropicLayerQuantities::real_axis(
            &material,
            &coordinates,
            Polarisation::TransverseMagnetic,
        )
        .into_admittance()
        .into_inner();

        assert_complex_close(
            admittance[()],
            quadratic_admittance(&material, k0, k_parallel, Polarisation::TransverseMagnetic),
            1e-12,
        );
    }

    #[test]
    fn real_axis_and_complex_plane_give_same_admittance_on_real_coordinates() {
        let material = quadratic(2.0, 0.4, 0.1, 1.5, -0.2, 0.05);

        let coordinates = scalar_coordinates(2.5, 0.8);

        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            let real_axis =
                IsotropicLayerQuantities::real_axis(&material, &coordinates, polarisation)
                    .into_admittance()
                    .into_inner();

            let complex_plane =
                IsotropicLayerQuantities::complex_plane(&material, &coordinates, polarisation)
                    .into_admittance()
                    .into_inner();

            assert_complex_close(real_axis[()], complex_plane[()], 1e-12);
        }
    }
}
