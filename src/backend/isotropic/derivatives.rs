//! Derivatives of isotropic layer propagation quantities.
//!
//! This module evaluates first and second derivatives of the normal
//! wavenumber and polarisation-dependent characteristic factor used by the
//! isotropic 2×2 backends.
//!
//! The primitive spectral coordinates are:
//!
//! ```text
//! k₀²
//! k∥²
//! ```
//!
//! where `k₀` is the vacuum wavenumber and `k∥` is the conserved parallel
//! wavenumber.
//!
//! For an isotropic medium,
//!
//! ```text
//! κ² = ε μ k₀² - k∥²
//! ```
//!
//! and:
//!
//! ```text
//! factor = μ    for TE
//! factor = ε    for TM
//! ```
//!
//! Derivatives with respect to linear `k₀` and `k∥` are obtained later using
//! the shared jet chain-rule transformation.

use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar, PlanarInput, SpectralDerivativeVariable, StructuralDerivativeVariable,
    backend::{
        IsotropicLayerQuantities, Polarisation,
        derivative::ChainRule,
        evaluator::{
            ComplexPlane, ConstitutiveDerivativeEvaluator, ConstitutiveEvaluator, RealAxis,
        },
        jet::{ArrayJet, ArrayJetFirst},
    },
    material::{
        DerivativeOrder, EvaluateDifferentiableMaterial, EvaluateDifferentiableMeromorphicMaterial,
        EvaluateMaterial, EvaluateMeromorphicMaterial, SpectralVariable,
    },
};

impl<C, D> IsotropicLayerQuantities<ArrayJetFirst<C, D>>
where
    D: Dimension,
{
    /// Return the first derivative of the normal wavenumber.
    pub(crate) fn dkappa(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        self.kappa.first()
    }

    /// Return the first derivative of epsilon
    pub(crate) fn depsilon(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        self.epsilon.first()
    }

    /// Return the first derivative of mu
    pub(crate) fn dmu(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        self.mu.first()
    }

    /// Return the first derivative of the TE/TM characteristic factor.
    pub(crate) fn dfactor(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        match self.polarisation {
            Polarisation::TransverseElectric => self.mu.first(),
            Polarisation::TransverseMagnetic => self.epsilon.first(),
        }
    }
}

impl<C, D> IsotropicLayerQuantities<ArrayJet<C, D>>
where
    D: Dimension,
{
    /// Return the second derivative of the normal wavenumber.
    pub(crate) fn ddkappa(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        self.kappa.second()
    }

    /// Return the second derivative of epsilon
    pub(crate) fn ddepsilon(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        self.epsilon.second()
    }

    /// Return the second derivative of mu
    pub(crate) fn ddmu(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        self.mu.second()
    }

    /// Return the first derivative of the normal wavenumber.
    pub(crate) fn dkappa(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        self.kappa.first()
    }

    /// Return the first derivative of epsilon
    pub(crate) fn depsilon(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        self.epsilon.first()
    }

    /// Return the first derivative of mu
    pub(crate) fn dmu(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        self.mu.first()
    }

    /// Return the first derivative of the TE/TM characteristic factor.
    pub(crate) fn dfactor(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        match self.polarisation {
            Polarisation::TransverseElectric => self.mu.first(),
            Polarisation::TransverseMagnetic => self.epsilon.first(),
        }
    }

    /// Return the second derivative of the TE/TM characteristic factor.
    pub(crate) fn ddfactor(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        match self.polarisation {
            Polarisation::TransverseElectric => self.mu.second(),
            Polarisation::TransverseMagnetic => self.epsilon.second(),
        }
    }
}

impl<C, D> IsotropicLayerQuantities<ArrayJetFirst<C, D>>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn constant<E, M>(material: &M, planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>) -> Self
    where
        E: ConstitutiveEvaluator<C, D, M>,
    {
        let values =
            IsotropicLayerQuantities::<ArrayBase<OwnedRepr<C>, D>>::new::<E, M>(material, planar);

        let (epsilon, mu, kappa, polarisation) = values.into_parts();

        Self::from_parts(
            ArrayJetFirst::constant(epsilon),
            ArrayJetFirst::constant(mu),
            ArrayJetFirst::constant(kappa),
            polarisation,
        )
    }

    pub(crate) fn vacuum_wavenumber_squared_real_axis<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::vacuum_wavenumber_squared::<RealAxis, M>(material, planar)
    }

    pub(crate) fn vacuum_wavenumber_squared_complex_plane<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self
    where
        M: EvaluateDifferentiableMeromorphicMaterial<C, Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::vacuum_wavenumber_squared::<ComplexPlane, M>(material, planar)
    }

    /// Evaluate derivatives with respect to squared vacuum wavenumber `k₀²`.
    ///
    /// For
    ///
    /// ```text
    /// Q = ε μ k₀² - k∥²
    /// κ = sqrt(Q)
    /// ```
    ///
    /// the derivative is:
    ///
    /// ```text
    /// dQ/d(k₀²)
    ///     = (ε′ μ + ε μ′) k₀² + ε μ
    ///
    /// dκ/d(k₀²)
    ///     = Q′ / (2κ)
    /// ```
    pub(crate) fn vacuum_wavenumber_squared<E, M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self
    where
        E: ConstitutiveDerivativeEvaluator<C, D, M>,
    {
        let epsilon_value = E::relative_permittivity(material, planar.vacuum_wavenumber());

        let epsilon_first = E::relative_permittivity_derivative(
            material,
            planar.vacuum_wavenumber(),
            DerivativeOrder::First,
            SpectralVariable::VacuumWavenumberSquared,
        );

        let mu_value = E::relative_permeability(material, planar.vacuum_wavenumber());

        let mu_first = E::relative_permeability_derivative(
            material,
            planar.vacuum_wavenumber(),
            DerivativeOrder::First,
            SpectralVariable::VacuumWavenumberSquared,
        );

        let epsilon = ArrayJetFirst::from_parts(epsilon_value, epsilon_first);

        let mu = ArrayJetFirst::from_parts(mu_value, mu_first);

        let k0_squared_value = planar.vacuum_wavenumber().mapv(|k0| k0 * k0);

        let k_parallel_squared_value = planar
            .parallel_wavenumber()
            .mapv(|k_parallel| k_parallel * k_parallel);

        let zero = k0_squared_value.mapv(|_| C::zero());

        let one = k0_squared_value.mapv(|_| C::one());

        let k0_squared = ArrayJetFirst::from_parts(k0_squared_value, one);

        let k_parallel_squared = ArrayJetFirst::from_parts(k_parallel_squared_value, zero.clone());

        let kappa = epsilon
            .multiply(&mu)
            .multiply(&k0_squared)
            .subtract(&k_parallel_squared)
            .sqrt();

        Self::from_parts(epsilon, mu, kappa, planar.polarisation())
    }

    pub(crate) fn parallel_wavenumber_squared_real_axis<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self
    where
        M: EvaluateMaterial<C, Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::parallel_wavenumber_squared::<RealAxis, M>(material, planar)
    }

    pub(crate) fn parallel_wavenumber_squared_complex_plane<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self
    where
        M: EvaluateMeromorphicMaterial<C, Real = C::RealField>,
    {
        Self::parallel_wavenumber_squared::<ComplexPlane, M>(material, planar)
    }

    /// Evaluate derivatives with respect to squared parallel wavenumber `k∥²`.
    ///
    /// Since:
    ///
    /// ```text
    /// Q = ε μ k₀² - k∥²
    /// ```
    ///
    /// material quantities are constant with respect to this coordinate and:
    ///
    /// ```text
    /// dκ/d(k∥²) = -1 / (2κ)
    /// ```
    pub(crate) fn parallel_wavenumber_squared<E, M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self
    where
        E: ConstitutiveEvaluator<C, D, M>,
    {
        let epsilon_value = E::relative_permittivity(material, planar.vacuum_wavenumber());
        let epsilon_first = epsilon_value.mapv(|_| C::zero());

        let mu_value = E::relative_permeability(material, planar.vacuum_wavenumber());
        let mu_first = mu_value.mapv(|_| C::zero());

        let epsilon = ArrayJetFirst::from_parts(epsilon_value, epsilon_first);
        let mu = ArrayJetFirst::from_parts(mu_value, mu_first);

        let k0_squared_value = planar.vacuum_wavenumber().mapv(|k0| k0 * k0);

        let k_parallel_squared_value = planar
            .parallel_wavenumber()
            .mapv(|k_parallel| k_parallel * k_parallel);

        let zero = k0_squared_value.mapv(|_| C::zero());

        let one = k0_squared_value.mapv(|_| C::one());

        let k0_squared = ArrayJetFirst::from_parts(k0_squared_value, zero);

        let k_parallel_squared = ArrayJetFirst::from_parts(k_parallel_squared_value, one);

        let kappa = epsilon
            .multiply(&mu)
            .multiply(&k0_squared)
            .subtract(&k_parallel_squared)
            .sqrt();

        Self::from_parts(epsilon, mu, kappa, planar.polarisation())
    }

    pub(crate) fn evaluate_first_structural_real_axis<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Self
    where
        M: EvaluateMaterial<C, Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::evaluate_first_structural::<RealAxis, M>(material, planar, variable)
    }

    pub(crate) fn evaluate_first_structural_complex_plane<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Self
    where
        M: EvaluateMeromorphicMaterial<C, Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::evaluate_first_structural::<ComplexPlane, M>(material, planar, variable)
    }

    pub(crate) fn evaluate_first_structural<E, M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Self
    where
        E: ConstitutiveEvaluator<C, D, M>,
    {
        let primitive = variable.primitive();

        let quantities = match primitive {
            StructuralDerivativeVariable::ParallelWavenumberSquared
            | StructuralDerivativeVariable::ParallelWavenumber => {
                Self::parallel_wavenumber_squared::<E, M>(material, planar)
            }

            StructuralDerivativeVariable::Thickness(_) => Self::constant::<E, M>(material, planar),
        };

        match variable.chain_rule(planar) {
            Some(rule) => quantities.chain_rule(&rule),
            None => quantities,
        }
    }

    pub(crate) fn evaluate_first_spectral_real_axis<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Self
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::evaluate_first_spectral::<RealAxis, M>(material, planar, variable)
    }

    pub(crate) fn evaluate_first_spectral_complex_plane<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Self
    where
        M: EvaluateDifferentiableMeromorphicMaterial<C, Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::evaluate_first_spectral::<ComplexPlane, M>(material, planar, variable)
    }

    pub(crate) fn evaluate_first_spectral<E, M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Self
    where
        E: ConstitutiveEvaluator<C, D, M> + ConstitutiveDerivativeEvaluator<C, D, M>,
    {
        let quantities = Self::vacuum_wavenumber_squared::<E, M>(material, planar);

        match variable.chain_rule(planar) {
            Some(rule) => quantities.chain_rule(&rule),
            None => quantities,
        }
    }

    pub(crate) fn chain_rule(self, rule: &ChainRule<ArrayBase<OwnedRepr<C>, D>>) -> Self {
        let (epsilon, mu, kappa, polarisation) = self.into_parts();

        Self::from_parts(
            epsilon.chain_rule(rule),
            mu.chain_rule(rule),
            kappa.chain_rule(rule),
            polarisation,
        )
    }
}

impl<C, D> IsotropicLayerQuantities<ArrayJet<C, D>>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn constant<E, M>(material: &M, planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>) -> Self
    where
        E: ConstitutiveEvaluator<C, D, M>,
    {
        let values =
            IsotropicLayerQuantities::<ArrayBase<OwnedRepr<C>, D>>::new::<E, M>(material, planar);

        let (epsilon, mu, kappa, polarisation) = values.into_parts();

        Self::from_parts(
            ArrayJet::constant(epsilon),
            ArrayJet::constant(mu),
            ArrayJet::constant(kappa),
            polarisation,
        )
    }

    pub(crate) fn vacuum_wavenumber_squared_real_axis<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::vacuum_wavenumber_squared::<RealAxis, M>(material, planar)
    }

    pub(crate) fn vacuum_wavenumber_squared_complex_plane<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self
    where
        M: EvaluateDifferentiableMeromorphicMaterial<C, Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::vacuum_wavenumber_squared::<ComplexPlane, M>(material, planar)
    }

    /// Evaluate first and second derivatives with respect to squared vacuum
    /// wavenumber `k₀²`.
    pub(crate) fn vacuum_wavenumber_squared<E, M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self
    where
        E: ConstitutiveDerivativeEvaluator<C, D, M>,
    {
        let epsilon_value = E::relative_permittivity(material, planar.vacuum_wavenumber());

        let epsilon_first = E::relative_permittivity_derivative(
            material,
            planar.vacuum_wavenumber(),
            DerivativeOrder::First,
            SpectralVariable::VacuumWavenumberSquared,
        );

        let epsilon_second = E::relative_permittivity_derivative(
            material,
            planar.vacuum_wavenumber(),
            DerivativeOrder::Second,
            SpectralVariable::VacuumWavenumberSquared,
        );

        let mu_value = E::relative_permeability(material, planar.vacuum_wavenumber());

        let mu_first = E::relative_permeability_derivative(
            material,
            planar.vacuum_wavenumber(),
            DerivativeOrder::First,
            SpectralVariable::VacuumWavenumberSquared,
        );

        let mu_second = E::relative_permeability_derivative(
            material,
            planar.vacuum_wavenumber(),
            DerivativeOrder::Second,
            SpectralVariable::VacuumWavenumberSquared,
        );

        let epsilon = ArrayJet::from_parts(epsilon_value, epsilon_first, epsilon_second);

        let mu = ArrayJet::from_parts(mu_value, mu_first, mu_second);

        let k0_squared_value = planar.vacuum_wavenumber().mapv(|k0| k0 * k0);

        let k_parallel_squared_value = planar
            .parallel_wavenumber()
            .mapv(|k_parallel| k_parallel * k_parallel);

        let zero = k0_squared_value.mapv(|_| C::zero());

        let one = k0_squared_value.mapv(|_| C::one());

        let k0_squared = ArrayJet::from_parts(k0_squared_value, one, zero.clone());

        let k_parallel_squared = ArrayJet::from_parts(k_parallel_squared_value, zero.clone(), zero);

        let kappa = epsilon
            .multiply(&mu)
            .multiply(&k0_squared)
            .subtract(&k_parallel_squared)
            .sqrt();

        Self::from_parts(epsilon, mu, kappa, planar.polarisation())
    }

    pub(crate) fn parallel_wavenumber_squared_real_axis<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::parallel_wavenumber_squared::<RealAxis, M>(material, planar)
    }

    pub(crate) fn parallel_wavenumber_squared_complex_plane<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self
    where
        M: EvaluateDifferentiableMeromorphicMaterial<C, Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::parallel_wavenumber_squared::<ComplexPlane, M>(material, planar)
    }

    /// Evaluate first and second derivatives with respect to squared parallel
    /// wavenumber `k∥²`.
    ///
    /// ```text
    /// dκ/d(k∥²)     = -1/(2κ)
    /// d²κ/d(k∥²)²   = -1/(4κ³)
    /// ```
    pub(crate) fn parallel_wavenumber_squared<E, M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self
    where
        E: ConstitutiveEvaluator<C, D, M>,
    {
        let epsilon_value = E::relative_permittivity(material, planar.vacuum_wavenumber());
        let epsilon_first = epsilon_value.mapv(|_| C::zero());
        let epsilon_second = epsilon_value.mapv(|_| C::zero());

        let mu_value = E::relative_permeability(material, planar.vacuum_wavenumber());
        let mu_first = mu_value.mapv(|_| C::zero());
        let mu_second = mu_value.mapv(|_| C::zero());

        let epsilon = ArrayJet::from_parts(epsilon_value, epsilon_first, epsilon_second);
        let mu = ArrayJet::from_parts(mu_value, mu_first, mu_second);

        let k0_squared_value = planar.vacuum_wavenumber().mapv(|k0| k0 * k0);

        let k_parallel_squared_value = planar
            .parallel_wavenumber()
            .mapv(|k_parallel| k_parallel * k_parallel);

        let zero = k0_squared_value.mapv(|_| C::zero());

        let one = k0_squared_value.mapv(|_| C::one());

        let k0_squared = ArrayJet::from_parts(k0_squared_value, zero.clone(), zero.clone());

        let k_parallel_squared = ArrayJet::from_parts(k_parallel_squared_value, one, zero);

        let kappa = epsilon
            .multiply(&mu)
            .multiply(&k0_squared)
            .subtract(&k_parallel_squared)
            .sqrt();

        Self::from_parts(epsilon, mu, kappa, planar.polarisation())
    }

    pub(crate) fn evaluate_second_structural_real_axis<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Self
    where
        M: EvaluateMaterial<C, Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::evaluate_second_structural::<RealAxis, M>(material, planar, variable)
    }

    pub(crate) fn evaluate_second_structural_complex_plane<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Self
    where
        M: EvaluateMeromorphicMaterial<C, Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::evaluate_second_structural::<ComplexPlane, M>(material, planar, variable)
    }

    pub(crate) fn evaluate_second_structural<E, M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Self
    where
        E: ConstitutiveEvaluator<C, D, M>,
    {
        let primitive = variable.primitive();

        let quantities = match primitive {
            StructuralDerivativeVariable::ParallelWavenumberSquared
            | StructuralDerivativeVariable::ParallelWavenumber => {
                Self::parallel_wavenumber_squared::<E, M>(material, planar)
            }

            StructuralDerivativeVariable::Thickness(_) => Self::constant::<E, M>(material, planar),
        };

        match variable.chain_rule(planar) {
            Some(rule) => quantities.chain_rule(&rule),
            None => quantities,
        }
    }

    pub(crate) fn evaluate_second_spectral_real_axis<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Self
    where
        M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::evaluate_second_spectral::<RealAxis, M>(material, planar, variable)
    }

    pub(crate) fn evaluate_second_spectral_complex_plane<M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Self
    where
        M: EvaluateDifferentiableMeromorphicMaterial<C, Real = C::RealField>,
        C::RealField: Copy,
    {
        Self::evaluate_second_spectral::<ComplexPlane, M>(material, planar, variable)
    }

    pub(crate) fn evaluate_second_spectral<E, M>(
        material: &M,
        planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Self
    where
        E: ConstitutiveEvaluator<C, D, M> + ConstitutiveDerivativeEvaluator<C, D, M>,
    {
        let quantities = Self::vacuum_wavenumber_squared::<E, M>(material, planar);

        match variable.chain_rule(planar) {
            Some(rule) => quantities.chain_rule(&rule),
            None => quantities,
        }
    }

    pub(crate) fn chain_rule(self, rule: &ChainRule<ArrayBase<OwnedRepr<C>, D>>) -> Self {
        let (epsilon, mu, kappa, polarisation) = self.into_parts();

        Self::from_parts(
            epsilon.chain_rule(rule),
            mu.chain_rule(rule),
            kappa.chain_rule(rule),
            polarisation,
        )
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array0, Array1, arr0, array};
    use num_complex::Complex64;

    use super::*;
    use crate::{
        backend::{
            PlanarInput, Polarisation,
            jet::{ArrayJet, ArrayJetFirst},
        },
        material::Constant,
    };

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn material(epsilon: f64, mu: f64) -> Constant<f64> {
        Constant::new(epsilon, mu)
    }

    fn scalar_input(
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

    fn assert_array1_close(actual: &Array1<C>, expected: &Array1<C>, tolerance: f64) {
        assert_eq!(actual.raw_dim(), expected.raw_dim(),);

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_close(actual, expected, tolerance);
        }
    }

    fn value_at_k0_squared(
        material: &Constant<f64>,
        k0_squared: f64,
        parallel_wavenumber: f64,
        polarisation: Polarisation,
    ) -> IsotropicLayerQuantities<Array0<C>> {
        let input = scalar_input(k0_squared.sqrt(), parallel_wavenumber, polarisation);

        IsotropicLayerQuantities::real_axis(material, &input)
    }

    fn value_at_parallel_squared(
        material: &Constant<f64>,
        vacuum_wavenumber: f64,
        parallel_squared: f64,
        polarisation: Polarisation,
    ) -> IsotropicLayerQuantities<Array0<C>> {
        let input = scalar_input(vacuum_wavenumber, parallel_squared.sqrt(), polarisation);

        IsotropicLayerQuantities::real_axis(material, &input)
    }

    #[test]
    fn first_spectral_jet_value_matches_value_evaluation() {
        let material = material(2.25, 1.4);

        let input = scalar_input(3.0, 0.7, Polarisation::TransverseMagnetic);

        let values = IsotropicLayerQuantities::real_axis(&material, &input);

        let differentiated =
            IsotropicLayerQuantities::<ArrayJetFirst<C, _>>::vacuum_wavenumber_squared_real_axis(
                &material, &input,
            );

        assert_close(
            differentiated.epsilon().value()[()],
            values.epsilon()[()],
            1e-12,
        );

        assert_close(differentiated.mu().value()[()], values.mu()[()], 1e-12);

        assert_close(
            differentiated.kappa().value()[()],
            values.kappa()[()],
            1e-12,
        );

        assert_close(
            differentiated.factor().value()[()],
            values.factor()[()],
            1e-12,
        );
    }

    #[test]
    fn second_spectral_jet_value_matches_value_evaluation() {
        let material = material(2.25, 1.4);

        let input = scalar_input(3.0, 0.7, Polarisation::TransverseElectric);

        let values = IsotropicLayerQuantities::real_axis(&material, &input);

        let differentiated =
            IsotropicLayerQuantities::<ArrayJet<C, _>>::vacuum_wavenumber_squared_real_axis(
                &material, &input,
            );

        assert_close(
            differentiated.kappa().value()[()],
            values.kappa()[()],
            1e-12,
        );

        assert_close(
            differentiated.factor().value()[()],
            values.factor()[()],
            1e-12,
        );
    }

    #[test]
    fn first_k0_squared_derivative_matches_finite_difference() {
        let material = material(2.25, 1.4);

        let k0_squared: f64 = 9.0;
        let parallel = 0.7;
        let h = 1e-5;

        let input = scalar_input(
            k0_squared.sqrt(),
            parallel,
            Polarisation::TransverseElectric,
        );

        let differentiated =
            IsotropicLayerQuantities::<ArrayJetFirst<C, _>>::vacuum_wavenumber_squared_real_axis(
                &material, &input,
            );

        let plus = value_at_k0_squared(
            &material,
            k0_squared + h,
            parallel,
            Polarisation::TransverseElectric,
        );

        let minus = value_at_k0_squared(
            &material,
            k0_squared - h,
            parallel,
            Polarisation::TransverseElectric,
        );

        let expected = (plus.kappa()[()] - minus.kappa()[()]) / (2.0 * h);

        assert_close(differentiated.kappa().first()[()], expected, 1e-8);
    }

    #[test]
    fn second_k0_squared_derivative_matches_finite_difference() {
        let material = material(2.25, 1.4);

        let k0_squared: f64 = 9.0;
        let parallel = 0.7;
        let h = 2e-3;

        let input = scalar_input(
            k0_squared.sqrt(),
            parallel,
            Polarisation::TransverseElectric,
        );

        let differentiated =
            IsotropicLayerQuantities::<ArrayJet<C, _>>::vacuum_wavenumber_squared_real_axis(
                &material, &input,
            );

        let plus = value_at_k0_squared(
            &material,
            k0_squared + h,
            parallel,
            Polarisation::TransverseElectric,
        );

        let centre = value_at_k0_squared(
            &material,
            k0_squared,
            parallel,
            Polarisation::TransverseElectric,
        );

        let minus = value_at_k0_squared(
            &material,
            k0_squared - h,
            parallel,
            Polarisation::TransverseElectric,
        );

        let expected =
            (plus.kappa()[()] - c(2.0) * centre.kappa()[()] + minus.kappa()[()]) / (h * h);

        assert_close(differentiated.kappa().second()[()], expected, 2e-6);
    }

    #[test]
    fn first_parallel_squared_derivative_matches_finite_difference() {
        let material = material(2.25, 1.4);

        let vacuum_wavenumber = 3.0;
        let parallel_squared: f64 = 0.49;
        let h = 1e-5;

        let input = scalar_input(
            vacuum_wavenumber,
            parallel_squared.sqrt(),
            Polarisation::TransverseMagnetic,
        );

        let differentiated =
            IsotropicLayerQuantities::<ArrayJetFirst<C, _>>::parallel_wavenumber_squared_real_axis(
                &material, &input,
            );

        let plus = value_at_parallel_squared(
            &material,
            vacuum_wavenumber,
            parallel_squared + h,
            Polarisation::TransverseMagnetic,
        );

        let minus = value_at_parallel_squared(
            &material,
            vacuum_wavenumber,
            parallel_squared - h,
            Polarisation::TransverseMagnetic,
        );

        let expected = (plus.kappa()[()] - minus.kappa()[()]) / (2.0 * h);

        assert_close(differentiated.kappa().first()[()], expected, 1e-8);
    }

    #[test]
    fn second_parallel_squared_derivative_matches_finite_difference() {
        let material = material(2.25, 1.4);

        let vacuum_wavenumber = 3.0;
        let parallel_squared: f64 = 0.49;
        let h = 2e-3;

        let input = scalar_input(
            vacuum_wavenumber,
            parallel_squared.sqrt(),
            Polarisation::TransverseMagnetic,
        );

        let differentiated =
            IsotropicLayerQuantities::<ArrayJet<C, _>>::parallel_wavenumber_squared_real_axis(
                &material, &input,
            );

        let plus = value_at_parallel_squared(
            &material,
            vacuum_wavenumber,
            parallel_squared + h,
            Polarisation::TransverseMagnetic,
        );

        let centre = value_at_parallel_squared(
            &material,
            vacuum_wavenumber,
            parallel_squared,
            Polarisation::TransverseMagnetic,
        );

        let minus = value_at_parallel_squared(
            &material,
            vacuum_wavenumber,
            parallel_squared - h,
            Polarisation::TransverseMagnetic,
        );

        let expected =
            (plus.kappa()[()] - c(2.0) * centre.kappa()[()] + minus.kappa()[()]) / (h * h);

        assert_close(differentiated.kappa().second()[()], expected, 2e-6);
    }

    #[test]
    fn parallel_squared_material_derivatives_are_zero() {
        let material = material(3.1, 1.2);

        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            let input = scalar_input(2.4, 0.5, polarisation);

            let differentiated =
                IsotropicLayerQuantities::<ArrayJet<C, _>>::parallel_wavenumber_squared_real_axis(
                    &material, &input,
                );

            assert_close(
                differentiated.epsilon().first()[()],
                C::new(0.0, 0.0),
                1e-12,
            );

            assert_close(
                differentiated.epsilon().second()[()],
                C::new(0.0, 0.0),
                1e-12,
            );

            assert_close(differentiated.mu().first()[()], C::new(0.0, 0.0), 1e-12);

            assert_close(differentiated.mu().second()[()], C::new(0.0, 0.0), 1e-12);

            assert_close(differentiated.factor().first()[()], C::new(0.0, 0.0), 1e-12);

            assert_close(
                differentiated.factor().second()[()],
                C::new(0.0, 0.0),
                1e-12,
            );
        }
    }

    #[test]
    fn second_order_path_contains_same_first_derivative_as_first_order_path() {
        let material = material(2.25, 1.4);

        let input = scalar_input(3.0, 0.7, Polarisation::TransverseMagnetic);

        let first =
            IsotropicLayerQuantities::<ArrayJetFirst<C, _>>::vacuum_wavenumber_squared_real_axis(
                &material, &input,
            );

        let second =
            IsotropicLayerQuantities::<ArrayJet<C, _>>::vacuum_wavenumber_squared_real_axis(
                &material, &input,
            );

        assert_close(second.kappa().first()[()], first.kappa().first()[()], 1e-12);

        assert_close(
            second.epsilon().first()[()],
            first.epsilon().first()[()],
            1e-12,
        );

        assert_close(second.mu().first()[()], first.mu().first()[()], 1e-12);

        assert_close(
            second.factor().first()[()],
            first.factor().first()[()],
            1e-12,
        );
    }

    #[test]
    fn array_shape_is_preserved_by_all_jet_components() {
        let material = material(2.25, 1.4);

        let input = PlanarInput::new(
            array![c(2.0), c(2.5), c(3.0)],
            array![c(0.3), c(0.4), c(0.5)],
            Polarisation::TransverseMagnetic,
        );

        let differentiated =
            IsotropicLayerQuantities::<ArrayJet<C, _>>::vacuum_wavenumber_squared_real_axis(
                &material, &input,
            );

        let expected = input.vacuum_wavenumber().raw_dim();

        for component in [
            differentiated.epsilon(),
            differentiated.mu(),
            differentiated.kappa(),
            differentiated.factor(),
        ] {
            assert_eq!(component.value().raw_dim(), expected,);

            assert_eq!(component.first().raw_dim(), expected,);

            assert_eq!(component.second().raw_dim(), expected,);
        }
    }

    #[test]
    fn array_evaluation_matches_scalar_evaluation() {
        let material = material(2.25, 1.4);

        let vacuum_wavenumbers = array![c(2.0), c(2.5), c(3.0)];

        let parallel_wavenumbers = array![c(0.3), c(0.4), c(0.5)];

        let input = PlanarInput::new(
            vacuum_wavenumbers.clone(),
            parallel_wavenumbers.clone(),
            Polarisation::TransverseMagnetic,
        );

        let array_quantities =
            IsotropicLayerQuantities::<ArrayJet<C, _>>::vacuum_wavenumber_squared_real_axis(
                &material, &input,
            );

        let mut expected_first = Vec::new();
        let mut expected_second = Vec::new();

        for (&k0, &k_parallel) in vacuum_wavenumbers.iter().zip(parallel_wavenumbers.iter()) {
            let scalar_input =
                PlanarInput::new(arr0(k0), arr0(k_parallel), Polarisation::TransverseMagnetic);

            let scalar =
                IsotropicLayerQuantities::<ArrayJet<C, _>>::vacuum_wavenumber_squared_real_axis(
                    &material,
                    &scalar_input,
                );

            expected_first.push(scalar.kappa().first()[()]);

            expected_second.push(scalar.kappa().second()[()]);
        }

        assert_array1_close(
            array_quantities.kappa().first(),
            &Array1::from_vec(expected_first),
            1e-12,
        );

        assert_array1_close(
            array_quantities.kappa().second(),
            &Array1::from_vec(expected_second),
            1e-12,
        );
    }

    #[test]
    fn complex_continuation_produces_finite_local_derivatives() {
        let material = material(1.0, 1.0);

        let input = PlanarInput::new(
            arr0(C::new(1.0, 0.05)),
            arr0(C::new(2.0, 0.1)),
            Polarisation::TransverseElectric,
        );

        let differentiated =
            IsotropicLayerQuantities::<ArrayJet<C, _>>::parallel_wavenumber_squared_real_axis(
                &material, &input,
            );

        for value in [
            differentiated.kappa().value()[()],
            differentiated.kappa().first()[()],
            differentiated.kappa().second()[()],
        ] {
            assert!(value.re.is_finite());
            assert!(value.im.is_finite());
        }
    }

    #[test]
    fn first_spectral_helper_matches_primitive_squared_constructor() {
        let material = material(2.25, 1.4);

        let input = scalar_input(3.0, 0.7, Polarisation::TransverseMagnetic);

        let helper =
            IsotropicLayerQuantities::<ArrayJetFirst<C, _>>::evaluate_first_spectral_real_axis(
                &material,
                &input,
                SpectralDerivativeVariable::VacuumWavenumberSquared,
            );

        let primitive =
            IsotropicLayerQuantities::<ArrayJetFirst<C, _>>::vacuum_wavenumber_squared_real_axis(
                &material, &input,
            );

        assert_eq!(helper, primitive);
    }

    #[test]
    fn second_spectral_helper_matches_primitive_squared_constructor() {
        let material = material(2.25, 1.4);

        let input = scalar_input(3.0, 0.7, Polarisation::TransverseElectric);

        let helper = IsotropicLayerQuantities::<ArrayJet<C, _>>::evaluate_second_spectral_real_axis(
            &material,
            &input,
            SpectralDerivativeVariable::VacuumWavenumberSquared,
        );

        let primitive =
            IsotropicLayerQuantities::<ArrayJet<C, _>>::vacuum_wavenumber_squared_real_axis(
                &material, &input,
            );

        assert_eq!(helper, primitive);
    }

    #[test]
    fn first_spectral_helper_applies_linear_wavenumber_chain_rule() {
        let material = material(2.25, 1.4);

        let input = scalar_input(3.0, 0.7, Polarisation::TransverseMagnetic);

        let squared =
            IsotropicLayerQuantities::<ArrayJetFirst<C, _>>::evaluate_first_spectral_real_axis(
                &material,
                &input,
                SpectralDerivativeVariable::VacuumWavenumberSquared,
            );

        let linear =
            IsotropicLayerQuantities::<ArrayJetFirst<C, _>>::evaluate_first_spectral_real_axis(
                &material,
                &input,
                SpectralDerivativeVariable::VacuumWavenumber,
            );

        let expected = squared.kappa().first()[()] * c(2.0 * 3.0);

        assert_close(linear.kappa().first()[()], expected, 1e-12);
    }

    #[test]
    fn second_spectral_helper_applies_second_order_chain_rule() {
        let material = material(2.25, 1.4);

        let k0 = 3.0;

        let input = scalar_input(k0, 0.7, Polarisation::TransverseMagnetic);

        let squared =
            IsotropicLayerQuantities::<ArrayJet<C, _>>::evaluate_second_spectral_real_axis(
                &material,
                &input,
                SpectralDerivativeVariable::VacuumWavenumberSquared,
            );

        let linear = IsotropicLayerQuantities::<ArrayJet<C, _>>::evaluate_second_spectral_real_axis(
            &material,
            &input,
            SpectralDerivativeVariable::VacuumWavenumber,
        );

        let first_expected = squared.kappa().first()[()] * c(2.0 * k0);

        let second_expected =
            squared.kappa().second()[()] * c(4.0 * k0 * k0) + squared.kappa().first()[()] * c(2.0);

        assert_close(linear.kappa().first()[()], first_expected, 1e-12);

        assert_close(linear.kappa().second()[()], second_expected, 1e-12);
    }

    #[test]
    fn first_structural_parallel_squared_helper_matches_primitive_constructor() {
        let material = material(2.25, 1.4);

        let input = scalar_input(3.0, 0.7, Polarisation::TransverseElectric);

        let helper =
            IsotropicLayerQuantities::<ArrayJetFirst<C, _>>::evaluate_first_structural_real_axis(
                &material,
                &input,
                StructuralDerivativeVariable::ParallelWavenumberSquared,
            );

        let primitive =
            IsotropicLayerQuantities::<ArrayJetFirst<C, _>>::parallel_wavenumber_squared_real_axis(
                &material, &input,
            );

        assert_eq!(helper, primitive);
    }

    #[test]
    fn second_structural_parallel_helper_applies_chain_rule() {
        let material = material(2.25, 1.4);

        let parallel = 0.7;

        let input = scalar_input(3.0, parallel, Polarisation::TransverseElectric);

        let squared =
            IsotropicLayerQuantities::<ArrayJet<C, _>>::evaluate_second_structural_real_axis(
                &material,
                &input,
                StructuralDerivativeVariable::ParallelWavenumberSquared,
            );

        let linear =
            IsotropicLayerQuantities::<ArrayJet<C, _>>::evaluate_second_structural_real_axis(
                &material,
                &input,
                StructuralDerivativeVariable::ParallelWavenumber,
            );

        let first_expected = squared.kappa().first()[()] * c(2.0 * parallel);

        let second_expected = squared.kappa().second()[()] * c(4.0 * parallel * parallel)
            + squared.kappa().first()[()] * c(2.0);

        assert_close(linear.kappa().first()[()], first_expected, 1e-12);

        assert_close(linear.kappa().second()[()], second_expected, 1e-12);
    }

    #[test]
    fn first_structural_thickness_helper_returns_constant_quantities() {
        let material = material(2.25, 1.4);

        let input = scalar_input(3.0, 0.7, Polarisation::TransverseMagnetic);

        let quantities =
            IsotropicLayerQuantities::<ArrayJetFirst<C, _>>::evaluate_first_structural_real_axis(
                &material,
                &input,
                StructuralDerivativeVariable::Thickness(0),
            );

        for component in [
            quantities.epsilon(),
            quantities.mu(),
            quantities.kappa(),
            quantities.factor(),
        ] {
            assert_close(component.first()[()], C::new(0.0, 0.0), 1e-12);
        }
    }

    #[test]
    fn second_structural_thickness_helper_returns_constant_quantities() {
        let material = material(2.25, 1.4);

        let input = scalar_input(3.0, 0.7, Polarisation::TransverseMagnetic);

        let quantities =
            IsotropicLayerQuantities::<ArrayJet<C, _>>::evaluate_second_structural_real_axis(
                &material,
                &input,
                StructuralDerivativeVariable::Thickness(0),
            );

        for component in [
            quantities.epsilon(),
            quantities.mu(),
            quantities.kappa(),
            quantities.factor(),
        ] {
            assert_close(component.first()[()], C::new(0.0, 0.0), 1e-12);

            assert_close(component.second()[()], C::new(0.0, 0.0), 1e-12);
        }
    }

    #[test]
    fn complex_plane_spectral_helper_matches_direct_complex_plane_constructor() {
        let material = material(2.25, 1.4);

        let input = PlanarInput::new(
            arr0(C::new(3.0, 0.1)),
            arr0(C::new(0.7, 0.05)),
            Polarisation::TransverseElectric,
        );

        let helper =
            IsotropicLayerQuantities::<ArrayJet<C, _>>::evaluate_second_spectral_complex_plane(
                &material,
                &input,
                SpectralDerivativeVariable::VacuumWavenumberSquared,
            );

        let direct =
            IsotropicLayerQuantities::<ArrayJet<C, _>>::vacuum_wavenumber_squared_complex_plane(
                &material, &input,
            );

        assert_eq!(helper, direct);
    }
}
