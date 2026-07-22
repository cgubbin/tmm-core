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
//! representation (`Array`, `ArrayJetFirst`, `ArrayJet`, `SpectralJet`, …)
//! using the chain rule.
//!
//! This keeps the material layer completely independent of the automatic
//! differentiation implementation.

use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ArrayJet, ArrayJetFirst, ComplexScalar, EvaluateMaterial, EvaluateMeromorphicMaterial,
    backend::{
        ComplexPlane, RealAxis,
        algebra::ScalarAlgebra,
        jet::{ArraySpectralJet, SecondOrderExpansion},
    },
    material::{
        DerivativeOrder, EvaluateDifferentiableMaterial, EvaluateDifferentiableMeromorphicMaterial,
    },
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
pub(crate) trait ConstitutiveEvaluator<C, D, M>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Evaluate relative permittivity.
    fn relative_permittivity(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    ) -> ArrayBase<OwnedRepr<C>, D>;

    /// Evaluate relative permeability.
    fn relative_permeability(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    ) -> ArrayBase<OwnedRepr<C>, D>;
}

impl<C, D, M> ConstitutiveEvaluator<C, D, M> for RealAxis
where
    C: ComplexScalar,
    D: Dimension,
    M: EvaluateMaterial<C, Real = C::RealField>,
    C::RealField: Copy,
{
    fn relative_permittivity(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    ) -> ArrayBase<OwnedRepr<C>, D> {
        let spectral = vacuum_wavenumber.mapv(|value| value.real());

        material.evaluate_relative_permittivity(spectral)
    }

    fn relative_permeability(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    ) -> ArrayBase<OwnedRepr<C>, D> {
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
    fn relative_permittivity(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    ) -> ArrayBase<OwnedRepr<C>, D> {
        material.evaluate_relative_permittivity_complex(vacuum_wavenumber.clone())
    }

    fn relative_permeability(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    ) -> ArrayBase<OwnedRepr<C>, D> {
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
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
        order: DerivativeOrder,
    ) -> ArrayBase<OwnedRepr<C>, D>;

    /// Evaluate relative permeability.
    fn relative_permeability_derivative(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
        order: DerivativeOrder,
    ) -> ArrayBase<OwnedRepr<C>, D>;
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
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
        order: DerivativeOrder,
    ) -> ArrayBase<OwnedRepr<C>, D> {
        let spectral = vacuum_wavenumber.mapv(|value| value.real());

        material.evaluate_relative_permittivity_derivative(spectral, order)
    }

    fn relative_permeability_derivative(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
        order: DerivativeOrder,
    ) -> ArrayBase<OwnedRepr<C>, D> {
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
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
        order: DerivativeOrder,
    ) -> ArrayBase<OwnedRepr<C>, D> {
        material.evaluate_relative_permittivity_complex_derivative(vacuum_wavenumber.clone(), order)
    }

    fn relative_permeability_derivative(
        material: &M,
        vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
        order: DerivativeOrder,
    ) -> ArrayBase<OwnedRepr<C>, D> {
        material.evaluate_relative_permeability_complex_derivative(vacuum_wavenumber.clone(), order)
    }
}

pub(crate) trait ConstitutiveLift<C, D, E, M>: ScalarAlgebra<C, D>
where
    C: ComplexScalar,
    D: Dimension,
    E: ConstitutiveEvaluator<C, D, M>,
{
    fn relative_permittivity(material: &M, vacuum_wavenumber: &Self) -> Self;

    fn relative_permeability(material: &M, vacuum_wavenumber: &Self) -> Self;
}

impl<C, D, E, M> ConstitutiveLift<C, D, E, M> for ArrayBase<OwnedRepr<C>, D>
where
    C: ComplexScalar + Copy,
    D: Dimension,
    E: ConstitutiveEvaluator<C, D, M>,
{
    fn relative_permittivity(material: &M, vacuum_wavenumber: &Self) -> Self {
        E::relative_permittivity(material, vacuum_wavenumber)
    }

    fn relative_permeability(material: &M, vacuum_wavenumber: &Self) -> Self {
        E::relative_permeability(material, vacuum_wavenumber)
    }
}

impl<C, D, E, M> ConstitutiveLift<C, D, E, M> for ArrayJetFirst<C, D>
where
    C: ComplexScalar + Copy,
    D: Dimension,
    E: ConstitutiveDerivativeEvaluator<C, D, M>,
{
    fn relative_permittivity(material: &M, vacuum_wavenumber: &Self) -> Self {
        let value = E::relative_permittivity(material, vacuum_wavenumber.value());
        let first = E::relative_permittivity_derivative(
            material,
            vacuum_wavenumber.value(),
            DerivativeOrder::First,
        );

        Self::compose_sampled_function(vacuum_wavenumber, value, first)
    }

    fn relative_permeability(material: &M, vacuum_wavenumber: &Self) -> Self {
        let value = E::relative_permeability(material, vacuum_wavenumber.value());
        let first = E::relative_permeability_derivative(
            material,
            vacuum_wavenumber.value(),
            DerivativeOrder::First,
        );

        Self::compose_sampled_function(vacuum_wavenumber, value, first)
    }
}

impl<C, D, E, M> ConstitutiveLift<C, D, E, M> for ArrayJet<C, D>
where
    C: ComplexScalar + Copy,
    D: Dimension,
    E: ConstitutiveDerivativeEvaluator<C, D, M>,
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

impl<C, D, E, M> ConstitutiveLift<C, D, E, M> for ArraySpectralJet<C, D>
where
    C: ComplexScalar + Copy,
    D: Dimension,
    E: ConstitutiveDerivativeEvaluator<C, D, M>,
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
