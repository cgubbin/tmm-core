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
}

delegate_analytical_material!(Drude);
