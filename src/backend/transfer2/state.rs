//! Transfer-state representation and travelling-wave conversion.
//!
//! The isotropic transfer matrix acts on a two-component state consisting of
//! a field-like component and its slope-like conjugate:
//!
//! ```text
//! state = [field, slope]ᵀ.
//! ```
//!
//! For physical characteristic admittance `Y`, define
//!
//! ```text
//! ξ = -iY.
//! ```
//!
//! The directional basis states are then:
//!
//! ```text
//! forward:  [1, -ξ]ᵀ
//! backward: [1, +ξ]ᵀ.
//! ```
//!
//! Therefore, for forward and backward amplitudes `a⁺` and `a⁻`,
//!
//! ```text
//! field = a⁺ + a⁻
//! slope = ξ(a⁻ - a⁺).
//! ```

use nalgebra::ComplexField;
use ndarray::Dimension;

use crate::{ComplexScalar, algebra::ScalarAlgebra, backend::BidirectionalWaves};

/// Transfer state at one spatial boundary.
///
/// A transfer matrix maps the state at its right boundary to the state at its
/// left boundary.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct TransferState<A> {
    field: A,
    slope: A,
}

impl<A> TransferState<A> {
    pub(crate) const fn new(field: A, slope: A) -> Self {
        Self { field, slope }
    }

    pub(crate) fn field(&self) -> &A {
        &self.field
    }

    pub(crate) fn slope(&self) -> &A {
        &self.slope
    }

    pub(crate) fn into_parts(self) -> (A, A) {
        (self.field, self.slope)
    }
}

/// Convert physical characteristic admittance into transfer-state slope.
///
/// The transfer backend uses:
///
/// ```text
/// ξ = -iY.
/// ```
pub(crate) fn transfer_state_slope<A>(admittance: &A) -> A
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    admittance.scale(-<A::Scalar as ComplexScalar>::i())
}

/// Construct a transfer state from forward- and backward-wave amplitudes.
///
/// With `ξ = -iY`, this computes:
///
/// ```text
/// field = forward + backward
/// slope = ξ(backward - forward).
/// ```
pub(crate) fn transfer_state_from_waves<A>(
    waves: &BidirectionalWaves<A>,
    admittance: &A,
) -> TransferState<A>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    let characteristic_slope = transfer_state_slope(admittance);

    let field = waves.forward().add(waves.backward());

    let slope = characteristic_slope.multiply(&waves.backward().subtract(waves.forward()));

    TransferState::new(field, slope)
}

/// Decompose a transfer state into forward- and backward-wave amplitudes.
///
/// This is the inverse of [`transfer_state_from_waves`]:
///
/// ```text
/// forward  = ½(field - slope / ξ)
/// backward = ½(field + slope / ξ).
/// ```
pub(crate) fn bidirectional_waves_from_state<A>(
    state: &TransferState<A>,
    admittance: &A,
) -> BidirectionalWaves<A>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar + Copy,
    A::Dimension: Dimension,
{
    let characteristic_slope = transfer_state_slope(admittance);

    let slope_ratio = state.slope().divide(&characteristic_slope);

    let half =
        (<A::Scalar as num_traits::One>::one() + <A::Scalar as num_traits::One>::one()).recip();

    let forward = state.field().subtract(&slope_ratio).scale(half);

    let backward = state.field().add(&slope_ratio).scale(half);

    BidirectionalWaves::new(forward, backward)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use super::*;
    use crate::{
        algebra::{ArrayJet0, Jet0, RealParameter},
        backend::BidirectionalWaves,
        test_support::{C, TOLERANCE, c},
    };

    type A = ArrayJet0<C, Ix0, RealParameter>;

    fn jet(value: C) -> A {
        Jet0::new(arr0(value))
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    fn round_trip(forward: C, backward: C) {
        let admittance = jet(c(2.5));

        let waves = BidirectionalWaves::new(jet(forward), jet(backward));

        let state = transfer_state_from_waves(&waves, &admittance);

        let recovered = bidirectional_waves_from_state(&state, &admittance);

        assert_wave_close(&recovered, &waves);
    }

    fn assert_wave_close(actual: &BidirectionalWaves<A>, expected: &BidirectionalWaves<A>) {
        assert_complex_close(actual.forward()[()], expected.forward()[()]);

        assert_complex_close(actual.backward()[()], expected.backward()[()]);
    }

    #[test]
    fn pure_forward_wave_round_trips() {
        round_trip(c(1.5), c(0.0));
    }

    #[test]
    fn pure_backward_wave_round_trips() {
        round_trip(c(0.0), c(-0.7));
    }

    #[test]
    fn mixed_waves_round_trip() {
        round_trip(C::new(1.2, 0.4), C::new(-0.3, 0.8));
    }

    #[test]
    fn forward_wave_has_negative_characteristic_slope() {
        let admittance = jet(c(3.0));

        let waves = BidirectionalWaves::new(jet(c(2.0)), jet(c(0.0)));

        let state = transfer_state_from_waves(&waves, &admittance);

        assert_eq!(state.field()[()], c(2.0));
        assert_eq!(state.slope()[()], C::new(0.0, 6.0),);
    }

    #[test]
    fn backward_wave_has_positive_characteristic_slope() {
        let admittance = jet(c(3.0));

        let waves = BidirectionalWaves::new(jet(c(0.0)), jet(c(2.0)));

        let state = transfer_state_from_waves(&waves, &admittance);

        assert_eq!(state.field()[()], c(2.0));
        assert_eq!(state.slope()[()], C::new(0.0, -6.0),);
    }
}
