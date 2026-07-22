//! Shared quantities for isotropic planar backends.
//!
//! This module evaluates the material and propagation quantities reused by the
//! isotropic 2×2 transfer- and scattering-matrix backends.
//!
//! For each sampled planar input, the normal wavenumber is defined by
//!
//! ```text
//! κ² = ε μ k₀² - k∥²
//! ```
//!
//! and the polarisation-dependent factor is
//!
//! ```text
//! factor = μ    for TE
//! factor = ε    for TM
//! ```
//!
//! The corresponding characteristic admittance is
//!
//! ```text
//! Y = κ / factor
//! ```
//!
//! Material quantities are evaluated once per medium and reused when
//! constructing matrices and derivatives.
//!
//! # Normal-wavenumber branch
//!
//! For each isotropic medium, the normal wavenumber is evaluated as
//!
//! ```text
//! κ = sqrt(ε μ k₀² - k∥²)
//! ```
//!
//! using the principal complex square root supplied by [`ComplexField`].
//! No additional pointwise sign correction is applied.
//!
//! The principal square root is analytic away from its branch cut and branch
//! point. Consequently, derivatives returned by this module are local
//! derivatives on that selected branch.
//!
//! For real passive scattering problems, this convention gives:
//!
//! - `κ >= 0` for propagating modes with positive real `κ²`;
//! - `Im(κ) >= 0` for evanescent modes with negative real `κ²`.
//!
//! For complex continuation and contour-based mode finding, callers must choose
//! a search domain over which
//!
//! ```text
//! ε_j μ_j k₀² - k∥²
//! ```
//!
//! avoids the principal square-root branch cut and zero for every medium `j`
//! whose normal wavenumber enters the residual. A contour crossing such a
//! branch cut does not define a single analytic residual and is therefore not
//! suitable for argument-principle integration.
//!
//! The caller does not supply `κ` directly. Branch selection is part of the
//! backend's mathematical convention and is applied consistently to finite
//! layers and both exterior media.

mod admittance;
mod constitutive;

pub(crate) use constitutive::{
    ConstitutiveDerivativeEvaluator, ConstitutiveEvaluator, ConstitutiveLift,
};

use nalgebra::ComplexField;
use ndarray::Dimension;

pub(crate) use admittance::IsotropicLayerAdmittance;

use crate::{
    ComplexScalar,
    backend::{
        ComplexPlane, Polarisation, RealAxis, algebra::ScalarAlgebra, input::AlgebraicPlanarInput,
    },
};

/// Material and propagation quantities for one isotropic medium.
///
/// Every array has the same sampled dimension as the corresponding
/// [`PlanarInput`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IsotropicLayerQuantities<A> {
    epsilon: A,
    mu: A,
    kappa: A,
    polarisation: Polarisation,
}

impl<A> IsotropicLayerQuantities<A> {
    pub(crate) fn from_parts(epsilon: A, mu: A, kappa: A, polarisation: Polarisation) -> Self {
        Self {
            epsilon,
            mu,
            kappa,
            polarisation,
        }
    }

    /// Consume the derivatives and return their components.
    pub(crate) fn into_parts(self) -> (A, A, A, Polarisation) {
        (self.epsilon, self.mu, self.kappa, self.polarisation)
    }

    /// Return the relative permittivity.
    pub(crate) fn epsilon(&self) -> &A {
        &self.epsilon
    }

    /// Return the relative permeability.
    pub(crate) fn mu(&self) -> &A {
        &self.mu
    }

    /// Return the selected normal wavenumber `κ`.
    pub(crate) fn kappa(&self) -> &A {
        &self.kappa
    }

    /// Return the polarisation used
    pub(crate) fn polarisation(&self) -> Polarisation {
        self.polarisation
    }

    /// Return the TE/TM characteristic factor.
    pub(crate) fn factor(&self) -> &A {
        match self.polarisation {
            Polarisation::TransverseElectric => &self.mu,
            Polarisation::TransverseMagnetic => &self.epsilon,
        }
    }

    pub(crate) fn into_admittance<C, D>(self) -> IsotropicLayerAdmittance<A>
    where
        C: ComplexField,
        D: Dimension,
        A: ScalarAlgebra<C, D>,
    {
        IsotropicLayerAdmittance::new(self.kappa.divide(self.factor()))
    }
}

impl<A> IsotropicLayerQuantities<A> {
    pub(crate) fn real_axis<C, D, M>(material: &M, planar: &AlgebraicPlanarInput<A>) -> Self
    where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D> + ConstitutiveLift<C, D, RealAxis, M> + Clone,
        RealAxis: ConstitutiveEvaluator<C, D, M>,
    {
        Self::evaluate::<C, D, RealAxis, M>(material, planar)
    }

    pub(crate) fn complex_plane<C, D, M>(material: &M, planar: &AlgebraicPlanarInput<A>) -> Self
    where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D> + ConstitutiveLift<C, D, ComplexPlane, M> + Clone,
        ComplexPlane: ConstitutiveEvaluator<C, D, M>,
    {
        Self::evaluate::<C, D, ComplexPlane, M>(material, planar)
    }

    pub(crate) fn evaluate<C, D, E, M>(material: &M, planar: &AlgebraicPlanarInput<A>) -> Self
    where
        C: ComplexScalar,
        D: Dimension,
        E: ConstitutiveEvaluator<C, D, M>,
        A: ScalarAlgebra<C, D> + ConstitutiveLift<C, D, E, M> + Clone,
    {
        let epsilon = A::relative_permittivity(material, planar.vacuum_wavenumber());

        let mu = A::relative_permeability(material, planar.vacuum_wavenumber());

        let k0_squared = planar
            .vacuum_wavenumber()
            .multiply(planar.vacuum_wavenumber());

        let kx_squared = planar
            .parallel_wavenumber()
            .multiply(planar.parallel_wavenumber());

        let kappa = epsilon
            .multiply(&mu)
            .multiply(&k0_squared)
            .subtract(&kx_squared)
            .sqrt();

        Self::from_parts(epsilon, mu, kappa, planar.polarisation())
    }
}
