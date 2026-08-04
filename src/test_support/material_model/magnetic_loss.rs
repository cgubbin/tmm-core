//! Combined Drude–Lorentz model.

use super::{MaterialModelError, lorentz::LorentzOscillator};
use crate::{
    ComplexScalar, DerivativeOrder, DifferentiableMaterial, DifferentiableMeromorphicMaterial,
    Material, MeromorphicMaterial, Sampled,
};
use num_traits::Float;
use std::fmt::Debug;

/// Combined Drude–Lorentz relative-permittivity model.
///
/// `ε(k₀) = ε∞ - Ωᴅ²/(k₀²+iΓᴅk₀) + Σ Sⱼ/(Ωⱼ²-k₀²-iΓⱼk₀)`.
#[derive(Clone, Debug, PartialEq)]
pub struct MagneticDrudeLorentz<R> {
    mu_infinity: R,
    drude_strength: R,
    drude_damping: R,
    oscillators: Vec<LorentzOscillator<R>>,
}

impl<R> MagneticDrudeLorentz<R>
where
    R: Float + Debug,
{
    /// Construct from primitive Drude strength `Ωᴅ²`.
    pub fn new(
        mu_infinity: R,
        drude_strength: R,
        drude_damping: R,
        oscillators: Vec<LorentzOscillator<R>>,
    ) -> Result<Self, MaterialModelError<R>> {
        validate_finite("mu_infinity", mu_infinity)?;
        validate_nonnegative("drude_strength", drude_strength)?;
        validate_nonnegative("drude_damping", drude_damping)?;

        Ok(Self {
            mu_infinity,
            drude_strength,
            drude_damping,
            oscillators,
        })
    }

    /// Construct from plasma wavenumber `Ωᴅ`.
    pub fn from_plasma_wavenumber(
        mu_infinity: R,
        plasma_wavenumber: R,
        drude_damping: R,
        oscillators: Vec<LorentzOscillator<R>>,
    ) -> Result<Self, MaterialModelError<R>> {
        validate_nonnegative("plasma_wavenumber", plasma_wavenumber)?;
        Self::new(
            mu_infinity,
            plasma_wavenumber * plasma_wavenumber,
            drude_damping,
            oscillators,
        )
    }

    /// Construct a Lorentz-only model.
    pub fn lorentz_only(
        mu_infinity: R,
        oscillators: Vec<LorentzOscillator<R>>,
    ) -> Result<Self, MaterialModelError<R>> {
        Self::new(mu_infinity, R::zero(), R::zero(), oscillators)
    }

    /// Return high-frequency permeability.
    pub fn mu_infinity(&self) -> R {
        self.mu_infinity
    }

    /// Return primitive Drude strength `Ωᴅ²`.
    pub fn drude_strength(&self) -> R {
        self.drude_strength
    }

    /// Return plasma wavenumber.
    pub fn plasma_wavenumber(&self) -> R {
        self.drude_strength.sqrt()
    }

    /// Return Drude damping.
    pub fn drude_damping(&self) -> R {
        self.drude_damping
    }

    /// Return Lorentz oscillators.
    pub fn oscillators(&self) -> &[LorentzOscillator<R>] {
        &self.oscillators
    }

    fn relative_permeability_at<C>(&self, k0: C) -> C
    where
        C: ComplexScalar<RealField = R> + Copy,
    {
        let mut epsilon = C::from_real(self.mu_infinity);

        if self.drude_strength > R::zero() {
            epsilon = epsilon + self.drude_value(k0);
        }

        for oscillator in &self.oscillators {
            epsilon = epsilon + oscillator.value_at(k0);
        }

        epsilon
    }

    fn derivative_at<C>(&self, k0: C, order: DerivativeOrder) -> C
    where
        C: ComplexScalar<RealField = R> + Copy,
    {
        let mut first = C::zero();
        let mut second = C::zero();
        let mut third = C::zero();

        if self.drude_strength > R::zero() {
            let (a, b, c) = self.drude_derivatives(k0);
            first += a;
            second += b;
            third += c;
        }

        for oscillator in &self.oscillators {
            let (a, b, c) = oscillator.derivatives_at(k0);
            first += a;
            second += b;
            third += c;
        }

        convert_derivative_variable(first, second, third, order)
    }

    fn drude_value<C>(&self, k0: C) -> C
    where
        C: ComplexScalar<RealField = R> + Copy,
    {
        let denominator = k0 * k0 + C::i() * C::from_real(self.drude_damping) * k0;

        -C::from_real(self.drude_strength) / denominator
    }

    fn drude_derivatives<C>(&self, k0: C) -> (C, C, C)
    where
        C: ComplexScalar<RealField = R> + Copy,
    {
        let one = C::one();
        let two = one + one;
        let three = two + one;
        let six = two * three;

        let imaginary_damping = C::i() * C::from_real(self.drude_damping);

        let denominator = k0 * k0 + imaginary_damping * k0;

        let denominator_first = two * k0 + imaginary_damping;

        let denominator_second = two;

        let strength = C::from_real(self.drude_strength);

        let denominator_squared = denominator * denominator;

        let denominator_cubed = denominator_squared * denominator;

        let denominator_fourth = denominator_cubed * denominator;

        let first = strength * denominator_first / denominator_squared;

        let second = strength
            * (denominator_second * denominator - two * denominator_first * denominator_first)
            / denominator_cubed;

        let third = six
            * strength
            * denominator_first
            * (denominator_first * denominator_first - denominator * denominator_second)
            / denominator_fourth;

        (first, second, third)
    }
}

impl<R> Material for MagneticDrudeLorentz<R>
where
    R: Float + Debug,
{
    type Real = R;

    fn relative_permeability<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = R> + Copy,
        I: Sampled<Elem = R>,
    {
        vacuum_wavenumber.map(|k0| self.relative_permeability_at::<C>(C::from_real(k0)))
    }

    fn relative_permittivity<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = R> + Copy,
        I: Sampled<Elem = R>,
    {
        vacuum_wavenumber.map(|_| C::one())
    }
}

impl<R> DifferentiableMaterial for MagneticDrudeLorentz<R>
where
    R: Float + Debug,
{
    fn relative_permittivity_derivative<I, C>(
        &self,
        vacuum_wavenumber: I,
        _order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = R> + Copy,
        I: Sampled<Elem = R>,
    {
        vacuum_wavenumber.map(|_| C::zero())
    }

    fn relative_permeability_derivative<I, C>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = R> + Copy,
        I: Sampled<Elem = R>,
    {
        vacuum_wavenumber.map(|k0| self.derivative_at::<C>(C::from_real(k0), order))
    }
}

impl<R> MeromorphicMaterial for MagneticDrudeLorentz<R>
where
    R: Float + Debug,
{
    fn relative_permeability_complex<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = R> + Copy,
        I: Sampled<Elem = C>,
    {
        vacuum_wavenumber.map(|k0| self.relative_permeability_at(k0))
    }

    fn relative_permittivity_complex<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = R> + Copy,
        I: Sampled<Elem = C>,
    {
        vacuum_wavenumber.map(|_| C::one())
    }
}

impl<R> DifferentiableMeromorphicMaterial for MagneticDrudeLorentz<R>
where
    R: Float + Debug,
{
    fn relative_permittivity_complex_derivative<I, C>(
        &self,
        vacuum_wavenumber: I,
        _order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = R> + Copy,
        I: Sampled<Elem = C>,
    {
        vacuum_wavenumber.map(|_| C::zero())
    }

    fn relative_permeability_complex_derivative<I, C>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = R> + Copy,
        I: Sampled<Elem = C>,
    {
        vacuum_wavenumber.map(|k0| self.derivative_at(k0, order))
    }
}

pub(crate) fn convert_derivative_variable<C>(
    first_k0: C,
    second_k0: C,
    third_k0: C,
    order: DerivativeOrder,
) -> C
where
    C: ComplexScalar + Copy,
{
    match order {
        DerivativeOrder::First => first_k0,
        DerivativeOrder::Second => second_k0,
        DerivativeOrder::Third => third_k0,
    }
}

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
