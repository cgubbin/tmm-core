//! Physical field and power post-processing for isotropic plane-wave solutions.
//!
//! This module converts [`BoundaryWaves`] into:
//!
//! - canonical tangential field states;
//! - signed normal power flux;
//! - fields sampled at arbitrary positions;
//! - per-layer absorptance;
//! - a complete plane-wave power balance.
//!
//! The calculations are backend-neutral once the boundary-wave solution has
//! been obtained.
//!
//! # Wave convention
//!
//! Geometric directions are fixed:
//!
//! - `forward` propagates from left to right;
//! - `backward` propagates from right to left.
//!
//! The finite-layer propagation convention is:
//!
//! ```text
//! p(d) = exp(-i κ d).
//! ```
//!
//! A forward wave referenced to a layer's left boundary therefore evolves as:
//!
//! ```text
//! a⁺(z) = a⁺(0) exp(-i κ z),
//! ```
//!
//! while a backward wave referenced to the layer's right boundary evolves as:
//!
//! ```text
//! a⁻(z) = a⁻(d) exp(-i κ (d - z)).
//! ```
//!
//! # Canonical field state
//!
//! For characteristic admittance `Y`, the canonical tangential pair is:
//!
//! ```text
//! primary = a⁺ + a⁻
//! dual    = Y (a⁺ - a⁻).
//! ```
//!
//! For TE polarisation, `primary` is the tangential electric-field amplitude.
//! For TM polarisation, `primary` is the tangential magnetic-field amplitude.
//! `dual` is the corresponding signed conjugate tangential field required for
//! the normal Poynting flux.
//!
//! The signed normal flux is:
//!
//! ```text
//! Pz = 1/2 Re(primary * dual*).
//! ```
//!
//! Positive flux is directed from left to right.

mod context;
mod fields;
mod power;
mod sample;

pub use fields::{
    PlaneWaveFieldDerivatives, PlaneWaveFieldDifferential, PlaneWaveFieldSample,
    PlaneWaveFieldSampleOwned, PlaneWaveFieldSampleView, PlaneWaveFields,
};

pub use power::{
    PlaneWavePowerBalance, PlaneWavePowerBalanceDerivative, PlaneWavePowerBalanceDerivatives,
};

pub(crate) use power::{
    plane_wave_power_balance_full_spectral_hessian, plane_wave_power_balance_k0_first,
    plane_wave_power_balance_k0_second, plane_wave_power_balance_kx_first,
    plane_wave_power_balance_kx_second, plane_wave_power_balance_thickness_first,
    plane_wave_power_balance_thickness_second, plane_wave_power_balance_values,
};
pub(crate) use sample::{
    sample_first_order_fields_k0, sample_first_order_fields_kx,
    sample_first_order_fields_thickness, sample_plane_wave_field_profile,
    sample_second_order_fields_full_spectral_hessian, sample_second_order_fields_k0,
    sample_second_order_fields_kx, sample_second_order_fields_thickness, sample_value_fields,
};

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array0, Ix0};
    use num_complex::Complex64;

    use crate::backend::{
        IsotropicFieldState,
        field::{BidirectionalWaves, BidirectionalWavesGeneric, LayerBoundaryWavesGeneric},
    };

    use super::*;

    type C = Complex64;
    type D = Ix0;
    type A = Array0<C>;

    const TOLERANCE: f64 = 1.0e-12;

    fn c(re: f64, im: f64) -> C {
        C::new(re, im)
    }

    fn scalar(re: f64, im: f64) -> A {
        Array0::from_elem((), c(re, im))
    }

    fn real_scalar(value: f64) -> Array0<f64> {
        Array0::from_elem((), value)
    }

    fn assert_complex_close(actual: &A, expected: C) {
        assert_relative_eq!(actual[()].re, expected.re, epsilon = TOLERANCE);

        assert_relative_eq!(actual[()].im, expected.im, epsilon = TOLERANCE);
    }

    fn assert_real_close(actual: &Array0<f64>, expected: f64) {
        assert_relative_eq!(actual[()], expected, epsilon = TOLERANCE);
    }

    fn waves(forward: C, backward: C) -> BidirectionalWavesGeneric<Array0<C>> {
        BidirectionalWavesGeneric::new(
            scalar(forward.re, forward.im),
            scalar(backward.re, backward.im),
        )
    }

    fn layer_waves(
        left_forward: C,
        left_backward: C,
        right_forward: C,
        right_backward: C,
    ) -> LayerBoundaryWavesGeneric<Array0<C>> {
        LayerBoundaryWavesGeneric::new(
            waves(left_forward, left_backward),
            waves(right_forward, right_backward),
        )
    }

    #[test]
    fn canonical_state_is_reconstructed_from_forward_and_backward_waves() {
        let forward = c(0.7, -0.2);
        let backward = c(-0.1, 0.4);
        let admittance = scalar(2.0, 0.5);

        let waves = waves(forward, backward);

        let state = IsotropicFieldState::from_waves::<C, D>(&waves, &admittance);

        assert_complex_close(state.primary(), forward + backward);

        assert_complex_close(state.dual(), c(2.0, 0.5) * (forward - backward));
    }
}
