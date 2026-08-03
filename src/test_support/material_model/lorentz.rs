//! Lorentz oscillator model.

use super::{
    MaterialModelError, delegate::delegate_analytical_material, drude_lorentz::DrudeLorentz,
};
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
    /// Construct from primitive oscillator strength `S`.
    pub fn new(strength: R, resonance: R, damping: R) -> Result<Self, MaterialModelError<R>> {
        Self::from_strength(strength, resonance, damping)
    }

    /// Construct from primitive oscillator strength `S`.
    pub fn from_strength(
        strength: R,
        resonance: R,
        damping: R,
    ) -> Result<Self, MaterialModelError<R>> {
        validate_nonnegative("strength", strength)?;
        validate_positive("resonance", resonance)?;
        validate_nonnegative("damping", damping)?;

        Ok(Self {
            strength,
            resonance,
            damping,
        })
    }

    /// Construct from the zero-frequency contribution `Δε`, using `S = Δε Ω²`.
    pub fn from_delta_epsilon(
        delta_epsilon: R,
        resonance: R,
        damping: R,
    ) -> Result<Self, MaterialModelError<R>> {
        validate_nonnegative("delta_epsilon", delta_epsilon)?;
        validate_positive("resonance", resonance)?;

        Self::from_strength(delta_epsilon * resonance * resonance, resonance, damping)
    }

    /// Return primitive strength `S`.
    pub fn strength(&self) -> R {
        self.strength
    }

    /// Return the static contribution `S / Ω²`.
    pub fn delta_epsilon(&self) -> R {
        self.strength / (self.resonance * self.resonance)
    }

    /// Return resonance wavenumber.
    pub fn resonance(&self) -> R {
        self.resonance
    }

    /// Return damping wavenumber.
    pub fn damping(&self) -> R {
        self.damping
    }

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

/// Pure Lorentz material model.
#[derive(Clone, Debug, PartialEq)]
pub struct Lorentz<R> {
    pub(crate) inner: DrudeLorentz<R>,
}

impl<R> Lorentz<R>
where
    R: Float + Debug,
{
    /// Construct a Lorentz model.
    pub fn new(
        epsilon_infinity: R,
        oscillators: Vec<LorentzOscillator<R>>,
    ) -> Result<Self, MaterialModelError<R>> {
        Ok(Self {
            inner: DrudeLorentz::lorentz_only(epsilon_infinity, oscillators)?,
        })
    }

    /// Return the high-frequency permittivity.
    pub fn epsilon_infinity(&self) -> R {
        self.inner.epsilon_infinity()
    }

    /// Return oscillator terms.
    pub fn oscillators(&self) -> &[LorentzOscillator<R>] {
        self.inner.oscillators()
    }
}

delegate_analytical_material!(Lorentz);

fn validate_finite<R>(name: &'static str, value: R) -> Result<(), MaterialModelError<R>>
where
    R: Float + Debug,
{
    if !value.is_finite() {
        return Err(MaterialModelError::NonFiniteParameter { name, value });
    }
    Ok(())
}

fn validate_nonnegative<R>(name: &'static str, value: R) -> Result<(), MaterialModelError<R>>
where
    R: Float + Debug,
{
    validate_finite(name, value)?;
    if value < R::zero() {
        return Err(MaterialModelError::NegativeParameter { name, value });
    }
    Ok(())
}

fn validate_positive<R>(name: &'static str, value: R) -> Result<(), MaterialModelError<R>>
where
    R: Float + Debug,
{
    validate_finite(name, value)?;
    if value <= R::zero() {
        return Err(MaterialModelError::NonPositiveParameter { name, value });
    }
    Ok(())
}
