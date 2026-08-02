//! Constitutive-function lifting.
//!
//! Material models naturally evaluate constitutive functions such as
//!
//! ```text
//! ε = ε(s)
//! μ = μ(s)
//! ```
//!
//! where `s` is the spectral variable used internally by the backend
//! (currently `k₀²`).
//!
//! Rather than constructing derivative-aware quantities directly, material
//! models return the Taylor coefficients of the constitutive function.
//!
//! These coefficients can then be composed with any algebraic coordinate
//! representation (`Array`, `ArrayJet1`, `ArrayJet`, `SpectralJet`, …)
//! using the chain rule.
//!
//! This keeps the material layer completely independent of the automatic
//! differentiation implementation.

use ndarray::{Array, Dimension};
use std::fmt::Debug;

use crate::{
    algebra::{
        ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2,
        FirstOrderExpansion, ScalarAlgebra, SecondOrderExpansion,
    },
    domain::{ComplexPlane, RealAxis},
    material::{
        DerivativeOrder, EvaluateDifferentiableMaterial, EvaluateDifferentiableMeromorphicMaterial,
        EvaluateMaterial, EvaluateMeromorphicMaterial,
    },
    scalar::ComplexScalar,
};

/// Evaluates isotropic constitutive quantities for a selected spectral domain.
///
/// This trait adapts the public material handles used by the backend:
///
/// - [`RealAxis`] delegates to `EvaluateMaterial` and
///   `EvaluateDifferentiableMaterial`;
/// - [`ComplexPlane`] delegates to the corresponding meromorphic handles.
///
/// It operates only on ordinary sampled values. Algebraic lifting into jets is
/// handled separately by `EvaluateConstitutive`.
#[doc(hidden)]
pub trait ConstitutiveEvaluator<C, D, M>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Evaluate relative permittivity.
    fn relative_permittivity(material: &M, vacuum_wavenumber: &Array<C, D>) -> Array<C, D>;

    /// Evaluate relative permeability.
    fn relative_permeability(material: &M, vacuum_wavenumber: &Array<C, D>) -> Array<C, D>;
}

impl<C, D, M> ConstitutiveEvaluator<C, D, M> for RealAxis
where
    C: ComplexScalar,
    D: Dimension,
    M: EvaluateMaterial<C, Real = C::RealField>,
    C::RealField: Copy,
{
    fn relative_permittivity(material: &M, vacuum_wavenumber: &Array<C, D>) -> Array<C, D> {
        let spectral = vacuum_wavenumber.mapv(|value| value.real());

        material.evaluate_relative_permittivity(spectral)
    }

    fn relative_permeability(material: &M, vacuum_wavenumber: &Array<C, D>) -> Array<C, D> {
        let spectral = vacuum_wavenumber.mapv(|value| value.real());

        material.evaluate_relative_permeability(spectral)
    }
}

impl<C, D, M> ConstitutiveEvaluator<C, D, M> for ComplexPlane
where
    C: ComplexScalar,
    D: Dimension,
    M: EvaluateMeromorphicMaterial<C, Real = C::RealField>,
    C::RealField: Copy,
{
    fn relative_permittivity(material: &M, vacuum_wavenumber: &Array<C, D>) -> Array<C, D> {
        material.evaluate_relative_permittivity_complex(vacuum_wavenumber.clone())
    }

    fn relative_permeability(material: &M, vacuum_wavenumber: &Array<C, D>) -> Array<C, D> {
        material.evaluate_relative_permeability_complex(vacuum_wavenumber.clone())
    }
}

pub(crate) trait ConstitutiveDerivativeEvaluator<C, D, M>:
    ConstitutiveEvaluator<C, D, M>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Evaluate relative permittivity.
    fn relative_permittivity_derivative(
        material: &M,
        vacuum_wavenumber: &Array<C, D>,
        order: DerivativeOrder,
    ) -> Array<C, D>;

    /// Evaluate relative permeability.
    fn relative_permeability_derivative(
        material: &M,
        vacuum_wavenumber: &Array<C, D>,
        order: DerivativeOrder,
    ) -> Array<C, D>;
}

impl<C, D, M> ConstitutiveDerivativeEvaluator<C, D, M> for RealAxis
where
    C: ComplexScalar,
    D: Dimension,
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
    C::RealField: Copy,
{
    fn relative_permittivity_derivative(
        material: &M,
        vacuum_wavenumber: &Array<C, D>,
        order: DerivativeOrder,
    ) -> Array<C, D> {
        debug_assert!(
            vacuum_wavenumber
                .iter()
                .all(|value| value.imaginary() == C::zero().real())
        );
        let spectral = vacuum_wavenumber.mapv(|value| value.real());

        material.evaluate_relative_permittivity_derivative(spectral, order)
    }

    fn relative_permeability_derivative(
        material: &M,
        vacuum_wavenumber: &Array<C, D>,
        order: DerivativeOrder,
    ) -> Array<C, D> {
        debug_assert!(
            vacuum_wavenumber
                .iter()
                .all(|value| value.imaginary() == C::zero().real())
        );
        let spectral = vacuum_wavenumber.mapv(|value| value.real());

        material.evaluate_relative_permeability_derivative(spectral, order)
    }
}

impl<C, D, M> ConstitutiveDerivativeEvaluator<C, D, M> for ComplexPlane
where
    C: ComplexScalar,
    D: Dimension,
    M: EvaluateDifferentiableMeromorphicMaterial<C, Real = C::RealField>,
    C::RealField: Copy,
{
    fn relative_permittivity_derivative(
        material: &M,
        vacuum_wavenumber: &Array<C, D>,
        order: DerivativeOrder,
    ) -> Array<C, D> {
        material.evaluate_relative_permittivity_complex_derivative(vacuum_wavenumber.clone(), order)
    }

    fn relative_permeability_derivative(
        material: &M,
        vacuum_wavenumber: &Array<C, D>,
        order: DerivativeOrder,
    ) -> Array<C, D> {
        material.evaluate_relative_permeability_complex_derivative(vacuum_wavenumber.clone(), order)
    }
}

#[doc(hidden)]
pub trait ConstitutiveLift<E, M>: ScalarAlgebra
where
    Self::Scalar: ComplexScalar,
    Self::Dimension: Dimension,
    E: ConstitutiveEvaluator<Self::Scalar, Self::Dimension, M>,
{
    fn refractive_index(material: &M, vacuum_wavenumber: &Self) -> Self {
        ScalarAlgebra::sqrt(&Self::relative_permittivity(material, vacuum_wavenumber))
    }

    fn relative_permittivity(material: &M, vacuum_wavenumber: &Self) -> Self;

    fn relative_permeability(material: &M, vacuum_wavenumber: &Self) -> Self;
}

impl<C, D, E, M, P> ConstitutiveLift<E, M> for ArrayJet0<C, D, P>
where
    C: ComplexScalar + Copy,
    D: Dimension,
    E: ConstitutiveEvaluator<C, D, M>,
    P: Clone + Debug,
{
    fn relative_permittivity(material: &M, vacuum_wavenumber: &Self) -> Self {
        Self::new(E::relative_permittivity(material, vacuum_wavenumber))
    }

    fn relative_permeability(material: &M, vacuum_wavenumber: &Self) -> Self {
        Self::new(E::relative_permeability(material, vacuum_wavenumber))
    }
}

impl<C, D, E, M, P> ConstitutiveLift<E, M> for ArrayJet1<C, D, P>
where
    C: ComplexScalar + Copy,
    D: Dimension,
    E: ConstitutiveDerivativeEvaluator<C, D, M>,
    P: Clone + Debug,
{
    fn relative_permittivity(material: &M, vacuum_wavenumber: &Self) -> Self {
        let value = E::relative_permittivity(material, vacuum_wavenumber.value());
        let first = E::relative_permittivity_derivative(
            material,
            vacuum_wavenumber.value(),
            DerivativeOrder::First,
        );

        Self::compose_sampled_function(vacuum_wavenumber, FirstOrderExpansion::new(value, first))
    }

    fn relative_permeability(material: &M, vacuum_wavenumber: &Self) -> Self {
        let value = E::relative_permeability(material, vacuum_wavenumber.value());
        let first = E::relative_permeability_derivative(
            material,
            vacuum_wavenumber.value(),
            DerivativeOrder::First,
        );

        Self::compose_sampled_function(vacuum_wavenumber, FirstOrderExpansion::new(value, first))
    }
}

impl<C, D, E, M, P> ConstitutiveLift<E, M> for ArrayJet2<C, D, P>
where
    C: ComplexScalar + Copy,
    D: Dimension,
    E: ConstitutiveDerivativeEvaluator<C, D, M>,
    P: Clone + Debug,
{
    fn relative_permittivity(material: &M, vacuum_wavenumber: &Self) -> Self {
        let value = E::relative_permittivity(material, vacuum_wavenumber.value());
        let first = E::relative_permittivity_derivative(
            material,
            vacuum_wavenumber.value(),
            DerivativeOrder::First,
        );
        let second = E::relative_permittivity_derivative(
            material,
            vacuum_wavenumber.value(),
            DerivativeOrder::Second,
        );

        Self::compose_sampled_function(
            vacuum_wavenumber,
            SecondOrderExpansion::new(value, first, second),
        )
    }

    fn relative_permeability(material: &M, vacuum_wavenumber: &Self) -> Self {
        let value = E::relative_permeability(material, vacuum_wavenumber.value());
        let first = E::relative_permeability_derivative(
            material,
            vacuum_wavenumber.value(),
            DerivativeOrder::First,
        );
        let second = E::relative_permeability_derivative(
            material,
            vacuum_wavenumber.value(),
            DerivativeOrder::Second,
        );

        Self::compose_sampled_function(
            vacuum_wavenumber,
            SecondOrderExpansion::new(value, first, second),
        )
    }
}

impl<C, D, E, M, P> ConstitutiveLift<E, M> for ArrayJetBivariate1<C, D, P>
where
    C: ComplexScalar + Copy,
    D: Dimension,
    E: ConstitutiveDerivativeEvaluator<C, D, M>,
    P: Clone + Debug,
{
    fn relative_permittivity(material: &M, vacuum_wavenumber: &Self) -> Self {
        let value = E::relative_permittivity(material, vacuum_wavenumber.value());
        let first = E::relative_permittivity_derivative(
            material,
            vacuum_wavenumber.value(),
            DerivativeOrder::First,
        );
        Self::compose_sampled_function(vacuum_wavenumber, FirstOrderExpansion::new(value, first))
    }

    fn relative_permeability(material: &M, vacuum_wavenumber: &Self) -> Self {
        let value = E::relative_permeability(material, vacuum_wavenumber.value());
        let first = E::relative_permeability_derivative(
            material,
            vacuum_wavenumber.value(),
            DerivativeOrder::First,
        );

        Self::compose_sampled_function(vacuum_wavenumber, FirstOrderExpansion::new(value, first))
    }
}

impl<C, D, E, M, P> ConstitutiveLift<E, M> for ArrayJetBivariate2<C, D, P>
where
    C: ComplexScalar + Copy,
    D: Dimension,
    E: ConstitutiveDerivativeEvaluator<C, D, M>,
    P: Clone + Debug,
{
    fn relative_permittivity(material: &M, vacuum_wavenumber: &Self) -> Self {
        let value = E::relative_permittivity(material, vacuum_wavenumber.value());
        let first = E::relative_permittivity_derivative(
            material,
            vacuum_wavenumber.value(),
            DerivativeOrder::First,
        );
        let second = E::relative_permittivity_derivative(
            material,
            vacuum_wavenumber.value(),
            DerivativeOrder::Second,
        );

        Self::compose_sampled_function(
            vacuum_wavenumber,
            SecondOrderExpansion::new(value, first, second),
        )
    }

    fn relative_permeability(material: &M, vacuum_wavenumber: &Self) -> Self {
        let value = E::relative_permeability(material, vacuum_wavenumber.value());
        let first = E::relative_permeability_derivative(
            material,
            vacuum_wavenumber.value(),
            DerivativeOrder::First,
        );
        let second = E::relative_permeability_derivative(
            material,
            vacuum_wavenumber.value(),
            DerivativeOrder::Second,
        );

        Self::compose_sampled_function(
            vacuum_wavenumber,
            SecondOrderExpansion::new(value, first, second),
        )
    }
}
