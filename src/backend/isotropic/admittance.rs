//! Characteristic admittance for isotropic planar media.
//!
//! This module evaluates the characteristic admittance used by the isotropic
//! transfer- and scattering-matrix backends.
//!
//! For an isotropic medium,
//!
//! ```text
//! Y = κ / factor
//! ```
//!
//! where:
//!
//! ```text
//! factor = μ    for TE
//! factor = ε    for TM
//! ```
//!
//! Value-only evaluation returns [`IsotropicLayerAdmittance`].
//! First- and second-order evaluations return [`ArrayJetFirst`] and
//! [`ArrayJet`] directly.
//!
//! Spectral derivatives are first evaluated with respect to the primitive
//! squared coordinate and then transformed to the caller-requested coordinate
//! using the shared jet chain rule.

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::{
        DerivativeVariable, PlanarInput,
        derivative::{SpectralDerivativeVariable, StructuralDerivativeVariable},
        evaluator::{
            ComplexPlane, ConstitutiveDerivativeEvaluator, ConstitutiveEvaluator, RealAxis,
        },
        isotropic::{
            IsotropicLayerFirstDerivatives, IsotropicLayerQuantities,
            IsotropicLayerSecondDerivatives,
        },
        jet::{ArrayJet, ArrayJetFirst},
    },
    material::{
        DifferentiableMaterial, DifferentiableMeromorphicMaterial, Material, MeromorphicMaterial,
    },
};

/// Characteristic admittance of one isotropic medium.
///
/// The admittance is:
///
/// ```text
/// Y = κ / factor
/// ```
///
/// and has the same sampled shape as the corresponding [`PlanarInput`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct IsotropicLayerAdmittance<C, D>
where
    D: Dimension,
{
    value: ArrayBase<OwnedRepr<C>, D>,
}

impl<C, D> IsotropicLayerAdmittance<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Construct an admittance from already evaluated isotropic quantities.
    pub(crate) fn from_quantities(quantities: &IsotropicLayerQuantities<C, D>) -> Self {
        Self {
            value: quantities.kappa().clone() / quantities.factor().view(),
        }
    }

    /// Evaluate the admittance of a material at a sampled planar input.
    ///
    /// Prefer [`Self::from_quantities`] when the isotropic quantities have
    /// already been evaluated.
    pub(crate) fn evaluate_real_axis<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self
    where
        M: Material<Real = C::RealField>,
        C::RealField: Copy,
    {
        let quantities = IsotropicLayerQuantities::real_axis(material, planar);

        Self::from_quantities(&quantities)
    }

    /// Evaluate the admittance of a material at a sampled planar input.
    ///
    /// Prefer [`Self::from_quantities`] when the isotropic quantities have
    /// already been evaluated.
    pub(crate) fn evaluate_complex_plane<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self
    where
        M: MeromorphicMaterial<Real = C::RealField>,
    {
        let quantities = IsotropicLayerQuantities::complex_plane(material, planar);

        Self::from_quantities(&quantities)
    }

    /// Return the sampled admittance.
    pub(crate) fn value(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.value
    }

    /// Consume the wrapper and return the sampled admittance.
    pub(crate) fn into_inner(self) -> ArrayBase<OwnedRepr<C>, D> {
        self.value
    }
}

/// Compute the first admittance derivative from derivatives of `κ` and
/// `factor`.
///
/// For:
///
/// ```text
/// Y = κ / f
/// ```
///
/// ```text
/// Y′ = κ′ / f - κ f′ / f²
/// ```
fn first_derivative<C, D>(
    quantities: &IsotropicLayerQuantities<C, D>,
    derivatives: &IsotropicLayerFirstDerivatives<C, D>,
) -> ArrayBase<OwnedRepr<C>, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    let factor_squared = quantities.factor().mapv(|value| value * value);

    derivatives.dkappa().clone() / quantities.factor().view()
        - quantities.kappa().clone() * derivatives.dfactor().view() / factor_squared
}

/// Compute the second admittance derivative from derivatives of `κ` and
/// `factor`.
///
/// For:
///
/// ```text
/// Y = κ / f
/// ```
///
/// ```text
/// Y″ = κ″/f
///      - 2κ′f′/f²
///      - κf″/f²
///      + 2κ(f′)²/f³
/// ```
fn second_derivative<C, D>(
    quantities: &IsotropicLayerQuantities<C, D>,
    derivatives: &IsotropicLayerSecondDerivatives<C, D>,
) -> ArrayBase<OwnedRepr<C>, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    let two = C::one() + C::one();

    let factor_squared = quantities.factor().mapv(|value| value * value);

    let factor_cubed = quantities.factor().mapv(|value| value * value * value);

    derivatives.ddkappa().clone() / quantities.factor().view()
        - (derivatives.first().dkappa().clone() * derivatives.first().dfactor().view())
            .mapv(|value| two * value)
            / factor_squared.view()
        - quantities.kappa().clone() * derivatives.ddfactor().view() / factor_squared
        + (quantities.kappa().clone() * derivatives.first().dfactor().mapv(|value| value * value))
            .mapv(|value| two * value)
            / factor_cubed
}

impl<C, D> IsotropicLayerAdmittance<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn evaluate_first_structural_real_axis<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> ArrayJetFirst<C, D>
    where
        M: Material<Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::evaluate_first_structural::<RealAxis, _>(material, planar, variable)
    }

    pub(crate) fn evaluate_first_structural_complex_plane<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> ArrayJetFirst<C, D>
    where
        M: MeromorphicMaterial<Real = C::RealField>,
    {
        Self::evaluate_first_structural::<ComplexPlane, _>(material, planar, variable)
    }

    /// Evaluate the admittance and its first derivative.
    ///
    /// Primitive squared-coordinate derivatives are transformed to the
    /// requested linear coordinate when necessary.
    fn evaluate_first_structural<E, M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> ArrayJetFirst<C, D>
    where
        E: ConstitutiveEvaluator<C, D, M>,
    {
        let quantities = IsotropicLayerQuantities::new::<E, _>(material, planar);

        let value = Self::from_quantities(&quantities).into_inner();

        let first = match variable {
            StructuralDerivativeVariable::ParallelWavenumberSquared
            | StructuralDerivativeVariable::ParallelWavenumber => {
                let derivatives =
                    IsotropicLayerFirstDerivatives::parallel_wavenumber_squared(&quantities);

                first_derivative(&quantities, &derivatives)
            }

            StructuralDerivativeVariable::Thickness(_) => value.mapv(|_| C::zero()),
        };

        let jet = ArrayJetFirst::from_parts(value, first);

        match variable.chain_rule(&planar) {
            Some(rule) => jet.chain_rule(&rule),
            None => jet,
        }
    }

    pub(crate) fn evaluate_first_spectral_real_axis<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> ArrayJetFirst<C, D>
    where
        M: DifferentiableMaterial<Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::evaluate_first_spectral::<RealAxis, _>(material, planar, variable)
    }

    pub(crate) fn evaluate_first_spectral_complex_plane<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> ArrayJetFirst<C, D>
    where
        M: DifferentiableMeromorphicMaterial<Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::evaluate_first_spectral::<ComplexPlane, _>(material, planar, variable)
    }

    fn evaluate_first_spectral<E, M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> ArrayJetFirst<C, D>
    where
        E: ConstitutiveEvaluator<C, D, M> + ConstitutiveDerivativeEvaluator<C, D, M>,
        C::RealField: Copy,
    {
        let quantities = IsotropicLayerQuantities::new::<E, _>(material, planar);

        let value = Self::from_quantities(&quantities).into_inner();

        let derivatives = IsotropicLayerFirstDerivatives::vacuum_wavenumber_squared::<E, _>(
            material,
            &quantities,
            planar.vacuum_wavenumber(),
            planar.polarisation(),
        );

        let first = first_derivative(&quantities, &derivatives);

        let jet = ArrayJetFirst::from_parts(value, first);

        match variable.chain_rule(&planar) {
            Some(rule) => jet.chain_rule(&rule),
            None => jet,
        }
    }
}

impl<C, D> IsotropicLayerAdmittance<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn evaluate_second_structural_real_axis<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> ArrayJet<C, D>
    where
        M: Material<Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::evaluate_second_structural::<RealAxis, _>(material, planar, variable)
    }

    pub(crate) fn evaluate_second_structural_complex_plane<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> ArrayJet<C, D>
    where
        M: MeromorphicMaterial<Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::evaluate_second_structural::<ComplexPlane, _>(material, planar, variable)
    }

    /// Evaluate the admittance and its first two derivatives.
    ///
    /// Primitive squared-coordinate derivatives are transformed to the
    /// requested linear coordinate when necessary.
    pub(crate) fn evaluate_second_structural<E, M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> ArrayJet<C, D>
    where
        E: ConstitutiveEvaluator<C, D, M>,
        C::RealField: Copy,
    {
        let quantities = IsotropicLayerQuantities::new::<E, _>(material, planar);

        let value = Self::from_quantities(&quantities).into_inner();

        let primitive = variable.primitive();

        let (first, second) = match primitive {
            StructuralDerivativeVariable::ParallelWavenumberSquared
            | StructuralDerivativeVariable::ParallelWavenumber => {
                let derivatives =
                    IsotropicLayerSecondDerivatives::parallel_wavenumber_squared(&quantities);

                (
                    first_derivative(&quantities, derivatives.first()),
                    second_derivative(&quantities, &derivatives),
                )
            }
            StructuralDerivativeVariable::Thickness(_) => {
                let zero = value.mapv(|_| C::zero());

                (zero.clone(), zero)
            }
        };

        let jet = ArrayJet::from_parts(value, first, second);

        match variable.chain_rule(&planar) {
            Some(rule) => jet.chain_rule(&rule),
            None => jet,
        }
    }

    pub(crate) fn evaluate_second_spectral_real_axis<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> ArrayJet<C, D>
    where
        M: DifferentiableMaterial<Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::evaluate_second_spectral::<RealAxis, _>(material, planar, variable)
    }

    pub(crate) fn evaluate_second_spectral_complex_plane<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> ArrayJet<C, D>
    where
        M: DifferentiableMeromorphicMaterial<Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::evaluate_second_spectral::<ComplexPlane, _>(material, planar, variable)
    }

    /// Evaluate the admittance and its first two derivatives.
    ///
    /// Primitive squared-coordinate derivatives are transformed to the
    /// requested linear coordinate when necessary.
    pub(crate) fn evaluate_second_spectral<E, M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> ArrayJet<C, D>
    where
        E: ConstitutiveEvaluator<C, D, M> + ConstitutiveDerivativeEvaluator<C, D, M>,
        C::RealField: Copy,
    {
        let quantities = IsotropicLayerQuantities::new::<E, _>(material, planar);

        let value = Self::from_quantities(&quantities).into_inner();

        let derivatives = IsotropicLayerSecondDerivatives::vacuum_wavenumber_squared::<E, _>(
            material,
            &quantities,
            planar.vacuum_wavenumber(),
            planar.polarisation(),
        );

        let (first, second) = (
            first_derivative(&quantities, derivatives.first()),
            second_derivative(&quantities, &derivatives),
        );

        let jet = ArrayJet::from_parts(value, first, second);

        match variable.chain_rule(&planar) {
            Some(rule) => jet.chain_rule(&rule),
            None => jet,
        }
    }
}

impl<C, D> IsotropicLayerAdmittance<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Construct a first-order admittance jet from cached quantities.
    pub(crate) fn first_jet_from_quantities(
        quantities: &IsotropicLayerQuantities<C, D>,
        derivatives: &IsotropicLayerFirstDerivatives<C, D>,
    ) -> ArrayJetFirst<C, D> {
        ArrayJetFirst::from_parts(
            Self::from_quantities(quantities).into_inner(),
            first_derivative(quantities, derivatives),
        )
    }

    /// Construct a second-order admittance jet from cached quantities.
    pub(crate) fn second_jet_from_quantities(
        quantities: &IsotropicLayerQuantities<C, D>,
        derivatives: &IsotropicLayerSecondDerivatives<C, D>,
    ) -> ArrayJet<C, D> {
        ArrayJet::from_parts(
            Self::from_quantities(quantities).into_inner(),
            first_derivative(quantities, derivatives.first()),
            second_derivative(quantities, derivatives),
        )
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array0, arr0};
    use num_complex::Complex64;

    use super::*;
    use crate::{backend::Polarisation, material::Constant};

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn material(epsilon: f64, mu: f64) -> Constant<f64> {
        Constant::new(epsilon, mu)
    }

    fn make_input(
        vacuum_wavenumber: f64,
        parallel_wavenumber: f64,
        polarisation: Polarisation,
    ) -> PlanarInput<Array0<C>> {
        PlanarInput::new(
            arr0(c(vacuum_wavenumber)),
            arr0(c(parallel_wavenumber)),
            polarisation,
        )
    }

    fn assert_close(actual: C, expected: C, tolerance: f64) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = tolerance,
            max_relative = tolerance,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = tolerance,
            max_relative = tolerance,
        );
    }

    #[test]
    fn value_matches_kappa_over_factor() {
        let material = material(2.25, 1.4);

        let input = make_input(3.0, 0.7, Polarisation::TransverseElectric);

        let quantities = IsotropicLayerQuantities::real_axis(&material, &input);

        let admittance = IsotropicLayerAdmittance::from_quantities(&quantities);

        assert_close(
            admittance.value()[()],
            quantities.kappa()[()] / quantities.factor()[()],
            1e-12,
        );
    }

    #[test]
    fn first_parallel_squared_derivative_matches_finite_difference() {
        let material = material(2.25, 1.4);

        let vacuum_wavenumber = 3.0;
        let parallel_squared: f64 = 0.49;
        let h = 1e-5;

        let input = make_input(
            vacuum_wavenumber,
            parallel_squared.sqrt(),
            Polarisation::TransverseMagnetic,
        );

        let jet = IsotropicLayerAdmittance::evaluate_first_structural_real_axis(
            &material,
            &input,
            StructuralDerivativeVariable::ParallelWavenumberSquared,
        );

        let plus = make_input(
            vacuum_wavenumber,
            (parallel_squared + h).sqrt(),
            Polarisation::TransverseMagnetic,
        );

        let minus = make_input(
            vacuum_wavenumber,
            (parallel_squared - h).sqrt(),
            Polarisation::TransverseMagnetic,
        );

        let plus = IsotropicLayerAdmittance::evaluate_real_axis(&material, &plus);

        let minus = IsotropicLayerAdmittance::evaluate_real_axis(&material, &minus);

        let expected = (plus.value()[()] - minus.value()[()]) / (2.0 * h);

        assert_close(jet.first()[()], expected, 1e-8);
    }

    #[test]
    fn second_parallel_squared_derivative_matches_finite_difference() {
        let material = material(2.25, 1.4);

        let vacuum_wavenumber = 3.0;
        let parallel_squared: f64 = 0.49;
        let h = 2e-3;

        let input = make_input(
            vacuum_wavenumber,
            parallel_squared.sqrt(),
            Polarisation::TransverseMagnetic,
        );

        let jet = IsotropicLayerAdmittance::evaluate_second_structural_real_axis(
            &material,
            &input,
            StructuralDerivativeVariable::ParallelWavenumberSquared,
        );

        let plus_input = make_input(
            vacuum_wavenumber,
            (parallel_squared + h).sqrt(),
            Polarisation::TransverseMagnetic,
        );

        let zero_input = make_input(
            vacuum_wavenumber,
            parallel_squared.sqrt(),
            Polarisation::TransverseMagnetic,
        );

        let minus_input = make_input(
            vacuum_wavenumber,
            (parallel_squared - h).sqrt(),
            Polarisation::TransverseMagnetic,
        );

        let plus = IsotropicLayerAdmittance::evaluate_real_axis(&material, &plus_input);

        let zero = IsotropicLayerAdmittance::evaluate_real_axis(&material, &zero_input);

        let minus = IsotropicLayerAdmittance::evaluate_real_axis(&material, &minus_input);

        let expected = (plus.value()[()] - c(2.0) * zero.value()[()] + minus.value()[()]) / (h * h);

        assert_close(jet.second()[()], expected, 2e-6);
    }

    #[test]
    fn linear_parallel_derivative_applies_chain_rule() {
        let material = material(2.25, 1.4);

        let input = make_input(3.0, 0.7, Polarisation::TransverseElectric);

        let squared = IsotropicLayerAdmittance::evaluate_first_structural_real_axis(
            &material,
            &input,
            StructuralDerivativeVariable::ParallelWavenumberSquared,
        );

        let linear = IsotropicLayerAdmittance::evaluate_first_structural_real_axis(
            &material,
            &input,
            StructuralDerivativeVariable::ParallelWavenumber,
        );

        assert_close(
            linear.first()[()],
            c(2.0 * 0.7) * squared.first()[()],
            1e-12,
        );
    }

    #[test]
    fn thickness_derivatives_are_zero() {
        let material = material(2.25, 1.4);

        let input = make_input(3.0, 0.7, Polarisation::TransverseElectric);

        let first = IsotropicLayerAdmittance::evaluate_first_structural_real_axis(
            &material,
            &input,
            StructuralDerivativeVariable::Thickness(0),
        );

        let second = IsotropicLayerAdmittance::evaluate_second_structural_real_axis(
            &material,
            &input,
            StructuralDerivativeVariable::Thickness(0),
        );

        assert_close(first.first()[()], c(0.0), 1e-12);
        assert_close(second.first()[()], c(0.0), 1e-12);
        assert_close(second.second()[()], c(0.0), 1e-12);
    }
}
