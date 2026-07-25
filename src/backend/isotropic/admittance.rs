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
    use approx::assert_relative_eq;
    use ndarray::{Array0, Ix0, arr0, array};
    use num_complex::Complex64;

    use crate::{
        algebra::{ArrayJet0, ArrayJet1, ArrayJet2},
        backend::isotropic::IsotropicLayerQuantities,
        input::{CanonicalInput, Polarisation},
        material::Constant,
    };

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
    ) -> CanonicalInput<C, Ix0> {
        CanonicalInput::new(
            arr0(c(vacuum_wavenumber)),
            arr0(c(parallel_wavenumber)),
            polarisation,
        )
    }

    fn assert_close(actual: C, expected: C, tolerance: f64) {
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

    fn value_admittance_at_k0_squared(
        material: &Constant<f64>,
        k0_squared: f64,
        parallel_wavenumber: f64,
        polarisation: Polarisation,
    ) -> C {
        let input = scalar_input(k0_squared.sqrt(), parallel_wavenumber, polarisation);

        let quantities = IsotropicLayerQuantities::real_axis(material, &input);

        quantities.into_admittance::<C, _>().into_inner()[()]
    }

    fn value_admittance_at_parallel_squared(
        material: &Constant<f64>,
        vacuum_wavenumber: f64,
        parallel_squared: f64,
        polarisation: Polarisation,
    ) -> C {
        let input = scalar_input(vacuum_wavenumber, parallel_squared.sqrt(), polarisation);

        let quantities = IsotropicLayerQuantities::real_axis(material, &input);

        quantities.into_admittance::<C, _>().into_inner()[()]
    }

    #[test]
    fn value_admittance_is_kappa_over_factor() {
        let material = material(4.0, 2.0);

        let input = scalar_input(3.0, 1.0, Polarisation::TransverseMagnetic);

        let quantities = IsotropicLayerQuantities::real_axis(&material, &input);

        let admittance = quantities.clone().into_admittance::<C, _>();

        assert_close(
            admittance[()],
            quantities.kappa()[()] / quantities.factor()[()],
            1e-12,
        );
    }

    #[test]
    fn te_and_tm_use_different_factors() {
        let material = material(4.0, 2.0);

        let te_input = scalar_input(3.0, 1.0, Polarisation::TransverseElectric);

        let tm_input = scalar_input(3.0, 1.0, Polarisation::TransverseMagnetic);

        let te =
            IsotropicLayerQuantities::real_axis(&material, &te_input).into_admittance::<C, _>();

        let tm =
            IsotropicLayerQuantities::real_axis(&material, &tm_input).into_admittance::<C, _>();

        let kappa = c((4.0_f64 * 2.0 * 9.0 - 1.0).sqrt());

        assert_close(te[()], kappa / c(2.0), 1e-12);

        assert_close(tm[()], kappa / c(4.0), 1e-12);
    }

    #[test]
    fn first_order_admittance_value_matches_value_path() {
        let material = material(2.25, 1.4);

        let input = scalar_input(3.0, 0.7, Polarisation::TransverseMagnetic);

        let values =
            IsotropicLayerQuantities::real_axis(&material, &input).into_admittance::<C, _>();

        let differentiated =
            IsotropicLayerQuantities::<ArrayJet1<C, _>>::vacuum_wavenumber_squared_real_axis(
                &material, &input,
            )
            .into_admittance::<C, _>();

        assert_close(differentiated.value()[()], values[()], 1e-12);
    }

    #[test]
    fn first_k0_squared_derivative_matches_finite_difference() {
        let material = material(2.25, 1.4);

        let k0_squared: f64 = 9.0;
        let parallel = 0.7;
        let h = 1e-5;

        let input = scalar_input(
            k0_squared.sqrt(),
            parallel,
            Polarisation::TransverseMagnetic,
        );

        let admittance =
            IsotropicLayerQuantities::<ArrayJet1<C, _>>::vacuum_wavenumber_squared_real_axis(
                &material, &input,
            )
            .into_admittance::<C, _>();

        let plus = value_admittance_at_k0_squared(
            &material,
            k0_squared + h,
            parallel,
            Polarisation::TransverseMagnetic,
        );

        let minus = value_admittance_at_k0_squared(
            &material,
            k0_squared - h,
            parallel,
            Polarisation::TransverseMagnetic,
        );

        let expected = (plus - minus) / (2.0 * h);

        assert_close(admittance.first()[()], expected, 1e-8);
    }

    #[test]
    fn second_k0_squared_derivative_matches_finite_difference() {
        let material = material(2.25, 1.4);

        let k0_squared: f64 = 9.0;
        let parallel = 0.7;
        let h = 2e-3;

        let input = scalar_input(
            k0_squared.sqrt(),
            parallel,
            Polarisation::TransverseElectric,
        );

        let admittance =
            IsotropicLayerQuantities::<ArrayJet2<C, _>>::vacuum_wavenumber_squared_real_axis(
                &material, &input,
            )
            .into_admittance::<C, _>();

        let plus = value_admittance_at_k0_squared(
            &material,
            k0_squared + h,
            parallel,
            Polarisation::TransverseElectric,
        );

        let centre = value_admittance_at_k0_squared(
            &material,
            k0_squared,
            parallel,
            Polarisation::TransverseElectric,
        );

        let minus = value_admittance_at_k0_squared(
            &material,
            k0_squared - h,
            parallel,
            Polarisation::TransverseElectric,
        );

        let expected = (plus - c(2.0) * centre + minus) / (h * h);

        assert_close(admittance.second()[()], expected, 2e-6);
    }

    #[test]
    fn first_parallel_squared_derivative_matches_finite_difference() {
        let material = material(2.25, 1.4);

        let vacuum_wavenumber = 3.0;
        let parallel_squared: f64 = 0.49;
        let h = 1e-5;

        let input = scalar_input(
            vacuum_wavenumber,
            parallel_squared.sqrt(),
            Polarisation::TransverseMagnetic,
        );

        let admittance =
            IsotropicLayerQuantities::<ArrayJet1<C, _>>::parallel_wavenumber_squared_real_axis(
                &material, &input,
            )
            .into_admittance::<C, _>();

        let plus = value_admittance_at_parallel_squared(
            &material,
            vacuum_wavenumber,
            parallel_squared + h,
            Polarisation::TransverseMagnetic,
        );

        let minus = value_admittance_at_parallel_squared(
            &material,
            vacuum_wavenumber,
            parallel_squared - h,
            Polarisation::TransverseMagnetic,
        );

        let expected = (plus - minus) / (2.0 * h);

        assert_close(admittance.first()[()], expected, 1e-8);
    }

    #[test]
    fn second_parallel_squared_derivative_matches_finite_difference() {
        let material = material(2.25, 1.4);

        let vacuum_wavenumber = 3.0;
        let parallel_squared: f64 = 0.49;
        let h = 2e-3;

        let input = scalar_input(
            vacuum_wavenumber,
            parallel_squared.sqrt(),
            Polarisation::TransverseElectric,
        );

        let admittance =
            IsotropicLayerQuantities::<ArrayJet2<C, _>>::parallel_wavenumber_squared_real_axis(
                &material, &input,
            )
            .into_admittance::<C, _>();

        let plus = value_admittance_at_parallel_squared(
            &material,
            vacuum_wavenumber,
            parallel_squared + h,
            Polarisation::TransverseElectric,
        );

        let centre = value_admittance_at_parallel_squared(
            &material,
            vacuum_wavenumber,
            parallel_squared,
            Polarisation::TransverseElectric,
        );

        let minus = value_admittance_at_parallel_squared(
            &material,
            vacuum_wavenumber,
            parallel_squared - h,
            Polarisation::TransverseElectric,
        );

        let expected = (plus - c(2.0) * centre + minus) / (h * h);

        assert_close(admittance.second()[()], expected, 2e-6);
    }

    #[test]
    fn second_order_admittance_contains_first_order_result() {
        let material = material(2.25, 1.4);

        let input = scalar_input(3.0, 0.7, Polarisation::TransverseMagnetic);

        let first =
            IsotropicLayerQuantities::<ArrayJet1<C, _>>::vacuum_wavenumber_squared_real_axis(
                &material, &input,
            )
            .into_admittance::<C, _>();

        let second =
            IsotropicLayerQuantities::<ArrayJet2<C, _>>::vacuum_wavenumber_squared_real_axis(
                &material, &input,
            )
            .into_admittance::<C, _>();

        assert_close(second.value()[()], first.value()[()], 1e-12);

        assert_close(second.first()[()], first.first()[()], 1e-12);
    }

    #[test]
    fn sampled_shape_is_preserved() {
        let material = material(2.25, 1.4);

        let input = CanonicalInput::new(
            array![c(2.0), c(2.5), c(3.0)],
            array![c(0.3), c(0.4), c(0.5)],
            Polarisation::TransverseMagnetic,
        );

        let admittance =
            IsotropicLayerQuantities::<ArrayJet2<C, _>>::vacuum_wavenumber_squared_real_axis(
                &material, &input,
            )
            .into_admittance::<C, _>();

        let expected = input.vacuum_wavenumber().raw_dim();

        assert_eq!(admittance.value().raw_dim(), expected,);

        assert_eq!(admittance.first().raw_dim(), expected,);

        assert_eq!(admittance.second().raw_dim(), expected,);
    }

    #[test]
    fn into_inner_returns_wrapped_representation() {
        let quantities = IsotropicLayerQuantities::from_parts(
            arr0(c(4.0)),
            arr0(c(2.0)),
            arr0(c(6.0)),
            Polarisation::TransverseElectric,
        );

        let admittance = quantities.into_admittance::<C, _>();

        let inner = admittance.into_inner();

        assert_close(inner[()], c(3.0), 1e-12);
    }
}
