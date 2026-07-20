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

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar, Polarisation,
    backend::{
        algebra::ScalarAlgebra,
        field::{BidirectionalWavesGeneric, CartesianElectromagneticField},
        jet::{ArrayJet, ArrayJetFirst},
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
        parallel_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
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

impl<C, D> IsotropicFieldState<ArrayJetFirst<C, D>> {
    pub(super) fn split(
        self,
    ) -> (
        IsotropicFieldState<ArrayBase<OwnedRepr<C>, D>>,
        IsotropicFieldState<ArrayBase<OwnedRepr<C>, D>>,
    )
    where
        C: ComplexScalar,
        D: Dimension,
    {
        let (primary, dual) = self.into_parts();

        let (primary, primary_first) = primary.into_parts();
        let (dual, dual_first) = dual.into_parts();

        (
            IsotropicFieldState::new(primary, dual),
            IsotropicFieldState::new(primary_first, dual_first),
        )
    }
}

impl<C, D> IsotropicFieldState<ArrayJet<C, D>> {
    pub(super) fn split(
        self,
    ) -> (
        IsotropicFieldState<ArrayBase<OwnedRepr<C>, D>>,
        IsotropicFieldState<ArrayBase<OwnedRepr<C>, D>>,
        IsotropicFieldState<ArrayBase<OwnedRepr<C>, D>>,
    )
    where
        C: ComplexScalar,
        D: Dimension,
    {
        let (primary, dual) = self.into_parts();

        let (primary, primary_first, primary_second) = primary.into_parts();
        let (dual, dual_first, dual_second) = dual.into_parts();

        (
            IsotropicFieldState::new(primary, dual),
            IsotropicFieldState::new(primary_first, dual_first),
            IsotropicFieldState::new(primary_second, dual_second),
        )
    }
}

pub(crate) fn reconstruct_isotropic_cartesian_fields<C, D, A>(
    state: &IsotropicFieldState<A>,
    polarisation: Polarisation,
    parallel_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
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
    parallel_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    mu: &A,
) -> CartesianElectromagneticField<A::Vector>
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Clone,
{
    let zero = mu.zero_like();

    let electric = A::into_cartesian_vector(zero.clone(), state.primary().clone(), zero.clone());

    let longitudinal = A::from_value(parallel_wavenumber.clone())
        .multiply(state.primary())
        .divide(mu);

    let magnetic = A::into_cartesian_vector(state.dual().negate(), zero.clone(), longitudinal);

    CartesianElectromagneticField::new(electric, magnetic)
}

fn reconstruct_tm<C, D, A>(
    state: &IsotropicFieldState<A>,
    parallel_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    epsilon: &A,
) -> CartesianElectromagneticField<A::Vector>
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Clone,
{
    let zero = epsilon.zero_like();

    let longitudinal = A::from_value(parallel_wavenumber.clone())
        .multiply(state.primary())
        .divide(epsilon)
        .negate();

    let electric = A::into_cartesian_vector(state.dual().clone(), zero.clone(), longitudinal);

    let magnetic = A::into_cartesian_vector(zero.clone(), state.primary().clone(), zero);

    CartesianElectromagneticField::new(electric, magnetic)
}
