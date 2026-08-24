//! Lorentz oscillator model.

use crate::ComplexScalar;
use num_traits::Float;
use std::fmt::Debug;

/// One Lorentz oscillator.
///
/// `Δε(k₀) = S / (Ω² - k₀² - i Γ k₀)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LorentzOscillator<R> {
    strength: R,
    resonance: R,
    damping: R,
}

impl<R> LorentzOscillator<R>
where
    R: Float + Debug,
{
    pub(crate) fn value_at<C>(&self, k0: C) -> C
    where
        C: ComplexScalar<RealField = R> + Copy,
    {
        let denominator = C::from_real(self.resonance * self.resonance)
            - k0 * k0
            - C::i() * C::from_real(self.damping) * k0;

        C::from_real(self.strength) / denominator
    }

    pub(crate) fn derivatives_at<C>(&self, k0: C) -> (C, C, C)
    where
        C: ComplexScalar<RealField = R> + Copy,
    {
        let one = C::one();
        let two = one + one;
        let three = two + one;
        let six = two * three;

        let imaginary_damping = C::i() * C::from_real(self.damping);

        let denominator =
            C::from_real(self.resonance * self.resonance) - k0 * k0 - imaginary_damping * k0;

        let denominator_first = -two * k0 - imaginary_damping;

        let denominator_second = -two;

        let strength = C::from_real(self.strength);

        let denominator_squared = denominator * denominator;

        let denominator_cubed = denominator_squared * denominator;

        let denominator_fourth = denominator_cubed * denominator;

        let first = -strength * denominator_first / denominator_squared;

        let second = strength
            * (two * denominator_first * denominator_first - denominator_second * denominator)
            / denominator_cubed;

        let third = six
            * strength
            * denominator_first
            * (denominator * denominator_second - denominator_first * denominator_first)
            / denominator_fourth;

        (first, second, third)
    }
}
