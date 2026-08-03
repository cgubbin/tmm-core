//! Drude free-carrier model.

use super::{DrudeLorentz, MaterialModelError, delegate::delegate_analytical_material};
use num_traits::Float;
use std::fmt::Debug;

/// Drude relative-permittivity model.
///
/// `ε(k₀) = ε∞ - Ωᴅ² / (k₀² + i Γᴅ k₀)`.
#[derive(Clone, Debug, PartialEq)]
pub struct Drude<R> {
    pub(crate) inner: DrudeLorentz<R>,
}

impl<R> Drude<R>
where
    R: Float + Debug,
{
    /// Construct from high-frequency permittivity, plasma wavenumber, and damping.
    pub fn new(
        epsilon_infinity: R,
        plasma_wavenumber: R,
        damping: R,
    ) -> Result<Self, MaterialModelError<R>> {
        Ok(Self {
            inner: DrudeLorentz::from_plasma_wavenumber(
                epsilon_infinity,
                plasma_wavenumber,
                damping,
                Vec::new(),
            )?,
        })
    }

    /// Return the high-frequency permittivity.
    pub fn epsilon_infinity(&self) -> R {
        self.inner.epsilon_infinity()
    }

    /// Return the plasma wavenumber.
    pub fn plasma_wavenumber(&self) -> R {
        self.inner.plasma_wavenumber()
    }

    /// Return the damping wavenumber.
    pub fn damping(&self) -> R {
        self.inner.drude_damping()
    }
}

delegate_analytical_material!(Drude);
