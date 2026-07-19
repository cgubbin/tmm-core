//! Isotropic canonical and Cartesian electromagnetic field reconstruction.
//!
//! The isotropic 2×2 backends naturally reconstruct a canonical tangential
//! pair:
//!
//! ```text
//! primary = a⁺ + a⁻
//! dual    = Y(a⁺ - a⁻)
//! ```
//!
//! where `Y` is the local characteristic admittance.
//!
//! The physical interpretation depends on polarisation:
//!
//! | Polarisation | `primary` | `dual` |
//! |---|---|---|
//! | TE | `E_y` | `-H_x` |
//! | TM | `H_y` | `E_x` |
//!
//! This module contains the single convention-sensitive mapping from that
//! canonical representation to Cartesian electric and magnetic fields.
//!
//! Coordinates are defined as:
//!
//! - `x`: signed in-plane propagation direction;
//! - `y`: transverse invariant direction;
//! - `z`: layer-normal direction, positive from left to right.
//!
//! Field amplitudes use the solver's normalized electromagnetic convention.
//! Consequently the Cartesian Poynting vector is normalized consistently with
//! the plane-wave power coefficients, but is not necessarily expressed in SI
//! power-density units.

use ndarray::Dimension;

use crate::{
    ComplexScalar, Polarisation,
    backend::{
        algebra::ScalarAlgebra,
        field::{BidirectionalWaves, BidirectionalWavesGeneric, CartesianElectromagneticField},
    },
};

/// Canonical isotropic tangential field pair at one spatial position.
///
/// The state is reconstructed from local forward and backward wave amplitudes:
///
/// ```text
/// primary = a⁺ + a⁻
/// dual    = Y(a⁺ - a⁻)
/// ```
///
/// Its physical meaning depends on polarisation:
///
/// - TE: `primary = E_y`, `dual = -H_x`;
/// - TM: `primary = H_y`, `dual = E_x`.
///
/// In either case the signed normal time-averaged flux is:
///
/// ```text
/// Pz = 1/2 Re(primary · dual*)
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct IsotropicFieldState<A> {
    primary: A,
    dual: A,
}

impl<A> IsotropicFieldState<A> {
    pub(crate) fn new(primary: A, dual: A) -> Self {
        Self { primary, dual }
    }

    /// Return the primary tangential field.
    ///
    /// This is the tangential electric field for TE and the tangential magnetic
    /// field for TM.
    pub fn primary(&self) -> &A {
        &self.primary
    }

    /// Return the signed dual tangential field.
    ///
    /// This is `-H_x` for TE polarisation and `E_x` for TM polarisation.
    pub fn dual(&self) -> &A {
        &self.dual
    }

    pub fn into_parts(self) -> (A, A) {
        (self.primary, self.dual)
    }

    pub(crate) fn from_waves<C, D>(waves: &BidirectionalWavesGeneric<A>, admittance: &A) -> Self
    where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D>,
    {
        let primary = waves.forward().add(waves.backward());

        let difference = waves.forward().subtract(waves.backward());

        let dual = admittance.multiply(&difference);

        Self::new(primary, dual)
    }

    pub fn primary_magnitude_squared<C, D>(&self) -> A::RealField
    where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D>,
    {
        self.primary.magnitude_squared()
    }

    pub fn normal_flux<C, D>(&self) -> A::RealField
    where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D>,
    {
        let half = C::one() / (C::one() + C::one());

        self.primary
            .multiply(&self.dual.conjugate())
            .scale(half)
            .real_part()
    }

    /// Reconstruct the Cartesian electric and magnetic fields.
    ///
    /// This is the single convention-sensitive conversion from the isotropic
    /// canonical state to Cartesian components. The signed linear parallel
    /// wavenumber from `input` determines the longitudinal field components.
    pub(crate) fn cartesian_fields<C, D>(
        &self,
        polarisation: Polarisation,
        parallel_wavenumber: &A,
        epsilon: &A,
        mu: &A,
    ) -> CartesianElectromagneticField<A::Vector>
    where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D> + Clone,
    {
        reconstruct_isotropic_cartesian_fields(self, polarisation, parallel_wavenumber, epsilon, mu)
    }
}

pub(crate) fn reconstruct_isotropic_cartesian_fields<C, D, A>(
    state: &IsotropicFieldState<A>,
    polarisation: Polarisation,
    parallel_wavenumber: &A,
    epsilon: &A,
    mu: &A,
) -> CartesianElectromagneticField<A::Vector>
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Clone,
{
    match polarisation {
        Polarisation::TransverseElectric => {
            reconstruct_te::<C, D, A>(state, parallel_wavenumber, mu)
        }

        Polarisation::TransverseMagnetic => {
            reconstruct_tm::<C, D, A>(state, parallel_wavenumber, epsilon)
        }
    }
}

fn reconstruct_te<C, D, A>(
    state: &IsotropicFieldState<A>,
    parallel_wavenumber: &A,
    mu: &A,
) -> CartesianElectromagneticField<A::Vector>
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Clone,
{
    let zero = mu.zero_like();

    let electric = A::into_cartesian_vector(zero.clone(), state.primary().clone(), zero.clone());

    let longitudinal = parallel_wavenumber.multiply(state.primary()).divide(mu);

    let magnetic = A::into_cartesian_vector(state.dual().negate(), zero.clone(), longitudinal);

    CartesianElectromagneticField::new(electric, magnetic)
}

fn reconstruct_tm<C, D, A>(
    state: &IsotropicFieldState<A>,
    parallel_wavenumber: &A,
    epsilon: &A,
) -> CartesianElectromagneticField<A::Vector>
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Clone,
{
    let zero = epsilon.zero_like();

    let longitudinal = parallel_wavenumber
        .multiply(state.primary())
        .divide(epsilon)
        .negate();

    let electric = A::into_cartesian_vector(state.dual().clone(), zero.clone(), longitudinal);

    let magnetic = A::into_cartesian_vector(zero.clone(), state.primary().clone(), zero);

    CartesianElectromagneticField::new(electric, magnetic)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Ix1, arr1};
    use num_complex::Complex64;
    use num_traits::Zero;

    use crate::{PlanarInput, Polarisation, backend::isotropic::IsotropicLayerQuantities};

    use super::*;

    type C = Complex64;
    type D = Ix1;

    const TOLERANCE: f64 = 1.0e-12;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn assert_real_close(actual: f64, expected: f64) {
        let error = (actual - expected).abs();

        assert!(
            error <= TOLERANCE,
            "expected {expected:e}, got {actual:e}; \
             absolute error = {error:e}",
        );
    }

    fn assert_complex_close(actual: C, expected: C) {
        let error = (actual - expected).norm();

        assert!(
            error <= TOLERANCE,
            "expected {expected:?}, got {actual:?}; \
             absolute error = {error:e}",
        );
    }

    fn assert_real_array_close(actual: &Array1<f64>, expected: &Array1<f64>) {
        assert_eq!(actual.raw_dim(), expected.raw_dim());

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_real_close(actual, expected);
        }
    }

    fn assert_complex_array_close(actual: &Array1<C>, expected: &Array1<C>) {
        assert_eq!(actual.raw_dim(), expected.raw_dim());

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_complex_close(actual, expected);
        }
    }

    fn state() -> IsotropicFieldState<C, D> {
        IsotropicFieldState::new(
            arr1(&[c(2.0, 1.0), c(-1.0, 3.0)]),
            arr1(&[c(4.0, -2.0), c(0.5, 1.5)]),
        )
    }

    /*
     * Adapt this helper to the exact public constructor of PlanarInput.
     */
    fn input(polarisation: Polarisation) -> PlanarInput<Array1<C>> {
        PlanarInput::new(
            arr1(&[c(1000.0, 0.0), c(1200.0, 0.0)]),
            arr1(&[c(300.0, 0.0), c(-400.0, 0.0)]),
            polarisation,
        )
    }

    /*
     * Adapt this helper to the actual IsotropicLayerQuantities constructor or
     * crate-private field constructors.
     *
     * Only epsilon and mu affect Cartesian reconstruction.
     */
    fn quantities() -> IsotropicLayerQuantities<C, D> {
        IsotropicLayerQuantities::from_parts(
            arr1(&[c(4.0, 0.5), c(2.5, 0.25)]),
            arr1(&[c(1.2, 0.1), c(0.9, 0.05)]),
            arr1(&[c(800.0, 10.0), c(900.0, 20.0)]),
            Polarisation::TransverseElectric,
        )
    }

    #[test]
    fn state_from_waves_reconstructs_canonical_pair() {
        let waves = BidirectionalWaves::new(
            arr1(&[c(2.0, 1.0), c(1.0, -2.0)]),
            arr1(&[c(0.5, -1.0), c(-1.0, 0.5)]),
        );

        let admittance = arr1(&[c(3.0, 0.0), c(2.0, 0.0)]);

        let state = IsotropicFieldState::from_waves(&waves, &admittance);

        let expected_primary = arr1(&[c(2.5, 0.0), c(0.0, -1.5)]);

        let expected_dual = arr1(&[c(4.5, 6.0), c(4.0, -5.0)]);

        assert_complex_array_close(state.primary(), &expected_primary);

        assert_complex_array_close(state.dual(), &expected_dual);
    }

    #[test]
    fn te_reconstruction_sets_expected_cartesian_components() {
        let state = state();
        let input = input(Polarisation::TransverseElectric);
        let quantities = quantities();

        let fields = reconstruct_isotropic_cartesian_fields(&state, &input, &quantities);

        let zero = arr1(&[C::zero(), C::zero()]);

        let expected_hz =
            input.parallel_wavenumber().clone() * state.primary().view() / quantities.mu().view();

        assert_eq!(fields.electric().x(), &zero);
        assert_eq!(fields.electric().y(), state.primary(),);
        assert_eq!(fields.electric().z(), &zero);

        assert_complex_array_close(fields.magnetic().x(), &state.dual().mapv(|value| -value));
        assert_eq!(fields.magnetic().y(), &zero);
        assert_complex_array_close(fields.magnetic().z(), &expected_hz);
    }

    #[test]
    fn tm_reconstruction_sets_expected_cartesian_components() {
        let state = state();
        let input = input(Polarisation::TransverseMagnetic);
        let quantities = quantities();

        let fields = reconstruct_isotropic_cartesian_fields(&state, &input, &quantities);

        let zero = arr1(&[C::zero(), C::zero()]);

        let expected_ez = -(input.parallel_wavenumber().clone() * state.primary().view()
            / quantities.epsilon().view());

        assert_complex_array_close(fields.electric().x(), state.dual());
        assert_eq!(fields.electric().y(), &zero);
        assert_complex_array_close(fields.electric().z(), &expected_ez);

        assert_eq!(fields.magnetic().x(), &zero);
        assert_eq!(fields.magnetic().y(), state.primary(),);
        assert_eq!(fields.magnetic().z(), &zero);
    }

    #[test]
    fn te_cartesian_poynting_z_matches_canonical_flux() {
        let state = state();
        let input = input(Polarisation::TransverseElectric);
        let quantities = quantities();

        let fields = state.cartesian_fields(&input, &quantities);

        let canonical = state.normal_flux();

        let cartesian = fields.time_averaged_poynting_vector().z().clone();

        assert_real_array_close(&cartesian, &canonical);
    }

    #[test]
    fn tm_cartesian_poynting_z_matches_canonical_flux() {
        let state = state();
        let input = input(Polarisation::TransverseMagnetic);
        let quantities = quantities();

        let fields = state.cartesian_fields(&input, &quantities);

        let canonical = state.normal_flux();

        let cartesian = fields.time_averaged_poynting_vector().z().clone();

        assert_real_array_close(&cartesian, &canonical);
    }

    #[test]
    fn longitudinal_components_vanish_at_normal_incidence() {
        let state = state();

        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            let input = PlanarInput::new(
                arr1(&[c(1000.0, 0.0), c(1200.0, 0.0)]),
                arr1(&[C::zero(), C::zero()]),
                polarisation,
            );

            let quantities = quantities();

            let fields = state.cartesian_fields(&input, &quantities);

            assert_eq!(fields.electric().z(), &arr1(&[C::zero(), C::zero()]),);

            assert_eq!(fields.magnetic().z(), &arr1(&[C::zero(), C::zero()]),);
        }
    }

    #[test]
    fn reversing_parallel_wavenumber_reverses_only_longitudinal_component() {
        let state = state();
        let quantities = quantities();

        let positive = PlanarInput::new(
            arr1(&[c(1000.0, 0.0), c(1200.0, 0.0)]),
            arr1(&[c(300.0, 0.0), c(400.0, 0.0)]),
            Polarisation::TransverseElectric,
        );

        let negative = PlanarInput::new(
            positive.vacuum_wavenumber().clone(),
            positive.parallel_wavenumber().mapv(|value| -value),
            Polarisation::TransverseElectric,
        );

        let positive_fields = state.cartesian_fields(&positive, &quantities);

        let negative_fields = state.cartesian_fields(&negative, &quantities);

        assert_eq!(positive_fields.electric(), negative_fields.electric(),);

        assert_eq!(
            positive_fields.magnetic().x(),
            negative_fields.magnetic().x(),
        );

        assert_complex_array_close(
            positive_fields.magnetic().z(),
            &negative_fields.magnetic().z().mapv(|value| -value),
        );
    }
}
