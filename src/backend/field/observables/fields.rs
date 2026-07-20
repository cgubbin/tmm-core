use crate::{
    DerivativeVariable,
    backend::{
        FieldPosition, IsotropicFieldState,
        algebra::ScalarAlgebra,
        field::{CartesianElectromagneticField, CartesianVector3, cartesian::CartesianField},
        jet::{ArrayJet, ArrayJetFirst},
    },
};

use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, OwnedRepr};

/// Fields sampled at a sequence of requested positions.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveFields<C, D>
where
    C: ComplexField,
    D: Dimension,
{
    samples: Vec<PlaneWaveFieldSample<C, D>>,
    derivatives: Option<PlaneWaveFieldDerivatives<C, D>>,
}

impl<C, D> PlaneWaveFields<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    pub(crate) fn from_values(
        samples: Vec<AlgebraicFieldSample<C, D, ArrayBase<OwnedRepr<C>, D>>>,
    ) -> Self {
        Self {
            samples: samples
                .into_iter()
                .map(PlaneWaveFieldSample::from_algebraic)
                .collect(),
            derivatives: None,
        }
    }

    pub(crate) fn from_first_order(
        variable: DerivativeVariable,
        samples: Vec<AlgebraicFieldSample<C, D, ArrayJetFirst<C, D>>>,
    ) -> Self {
        let (samples, first): (Vec<_>, Vec<_>) = samples
            .into_iter()
            .map(AlgebraicFieldSample::into_value_and_first)
            .unzip();

        Self {
            samples,
            derivatives: Some(PlaneWaveFieldDerivatives {
                variable,
                first,
                second: None,
            }),
        }
    }

    pub(crate) fn from_second_order(
        variable: DerivativeVariable,
        samples: Vec<AlgebraicFieldSample<C, D, ArrayJet<C, D>>>,
    ) -> Self {
        let mut values = Vec::with_capacity(samples.len());
        let mut first = Vec::with_capacity(samples.len());
        let mut second = Vec::with_capacity(samples.len());

        for sample in samples {
            let (value, first_derivative, second_derivative) = sample.into_value_first_and_second();

            values.push(value);
            first.push(first_derivative);
            second.push(second_derivative);
        }

        Self {
            samples: values,
            derivatives: Some(PlaneWaveFieldDerivatives {
                variable,
                first,
                second: Some(second),
            }),
        }
    }

    pub fn samples(&self) -> &[PlaneWaveFieldSample<C, D>] {
        &self.samples
    }

    pub fn sample(&self, index: usize) -> Option<&PlaneWaveFieldSample<C, D>> {
        self.samples.get(index)
    }

    pub fn derivatives(&self) -> Option<&PlaneWaveFieldDerivatives<C, D>> {
        self.derivatives.as_ref()
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn into_samples(self) -> Vec<PlaneWaveFieldSample<C, D>> {
        self.samples
    }
}

/// Electromagnetic fields sampled at one spatial position.
///
/// The sample retains both:
///
/// - the compact canonical tangential state used by the isotropic 2×2
///   formulation;
/// - the corresponding Cartesian electric and magnetic fields.
///
/// The Cartesian fields use the same normalization as the canonical solver
/// state.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveFieldSample<C, D>
where
    C: ComplexField,
    D: Dimension,
{
    position: FieldPosition<C::RealField>,

    /// Global coordinate measured rightward from the stack's left boundary.
    ///
    /// Left-exterior coordinates are negative.
    coordinate: C::RealField,

    canonical: IsotropicFieldState<ArrayBase<OwnedRepr<C>, D>>,

    cartesian: CartesianElectromagneticField<CartesianVector3<C, D>>,
}

impl<C, D> PlaneWaveFieldSample<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn from_algebraic(sample: AlgebraicFieldSample<C, D, ArrayBase<OwnedRepr<C>, D>>) -> Self {
        Self::new(
            sample.position,
            sample.coordinate,
            sample.canonical,
            sample.cartesian,
        )
    }

    pub(crate) fn new(
        position: FieldPosition<C::RealField>,
        coordinate: C::RealField,
        canonical: IsotropicFieldState<ArrayBase<OwnedRepr<C>, D>>,
        cartesian: CartesianElectromagneticField<CartesianVector3<C, D>>,
    ) -> Self {
        debug_assert_eq!(
            canonical.primary().raw_dim(),
            cartesian.electric().x().raw_dim(),
        );

        Self {
            position,
            coordinate,
            canonical,
            cartesian,
        }
    }

    /// Return the requested stack-relative position.
    pub fn position(&self) -> FieldPosition<C::RealField>
    where
        C::RealField: Copy,
    {
        self.position
    }

    /// Return the global coordinate measured along the layer-normal axis.
    pub fn coordinate(&self) -> C::RealField
    where
        C::RealField: Copy,
    {
        self.coordinate
    }

    /// Return the canonical isotropic tangential field state.
    pub fn canonical_state(&self) -> &IsotropicFieldState<ArrayBase<OwnedRepr<C>, D>> {
        &self.canonical
    }

    /// Return the Cartesian electric and magnetic fields.
    pub fn cartesian_fields(&self) -> &CartesianElectromagneticField<CartesianVector3<C, D>> {
        &self.cartesian
    }

    /// Return the Cartesian electric field.
    pub fn electric(&self) -> &CartesianVector3<C, D> {
        self.cartesian.electric()
    }

    /// Return the Cartesian magnetic field.
    pub fn magnetic(&self) -> &CartesianVector3<C, D> {
        self.cartesian.magnetic()
    }

    /// Return the signed normal time-averaged power flux.
    ///
    /// Positive values represent power flow from left to right.
    pub fn normal_flux(&self) -> ArrayBase<OwnedRepr<C::RealField>, D> {
        self.cartesian.time_averaged_poynting_vector().z().clone()
    }

    /// Consume the sample and return its constituent values.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        FieldPosition<C::RealField>,
        C::RealField,
        IsotropicFieldState<ArrayBase<OwnedRepr<C>, D>>,
        CartesianField<C, D>,
    ) {
        (
            self.position,
            self.coordinate,
            self.canonical,
            self.cartesian,
        )
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveFieldDerivatives<C, D>
where
    C: ComplexField,
    D: Dimension,
{
    variable: DerivativeVariable,
    first: Vec<PlaneWaveFieldDifferential<C, D>>,
    second: Option<Vec<PlaneWaveFieldDifferential<C, D>>>,
}

impl<C, D> PlaneWaveFieldDerivatives<C, D>
where
    C: ComplexField,
    D: Dimension,
{
    pub fn variable(&self) -> DerivativeVariable {
        self.variable
    }

    pub fn first(&self) -> &[PlaneWaveFieldDifferential<C, D>] {
        &self.first
    }

    pub fn first_sample(&self, index: usize) -> Option<&PlaneWaveFieldDifferential<C, D>> {
        self.first.get(index)
    }

    pub fn second(&self) -> Option<&[PlaneWaveFieldDifferential<C, D>]> {
        self.second.as_deref()
    }

    pub fn second_sample(&self, index: usize) -> Option<&PlaneWaveFieldDifferential<C, D>> {
        self.second.as_ref()?.get(index)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveFieldDifferential<C, D>
where
    C: ComplexField,
    D: Dimension,
{
    canonical: IsotropicFieldState<ArrayBase<OwnedRepr<C>, D>>,
    cartesian: CartesianElectromagneticField<CartesianVector3<C, D>>,
}

impl<C, D> PlaneWaveFieldDifferential<C, D>
where
    C: ComplexField,
    D: Dimension,
{
    pub(crate) fn new(
        canonical: IsotropicFieldState<ArrayBase<OwnedRepr<C>, D>>,
        cartesian: CartesianElectromagneticField<CartesianVector3<C, D>>,
    ) -> Self {
        Self {
            canonical,
            cartesian,
        }
    }

    pub fn canonical_state(&self) -> &IsotropicFieldState<ArrayBase<OwnedRepr<C>, D>> {
        &self.canonical
    }

    pub fn cartesian_fields(&self) -> &CartesianElectromagneticField<CartesianVector3<C, D>> {
        &self.cartesian
    }

    pub fn electric(&self) -> &CartesianVector3<C, D> {
        self.cartesian.electric()
    }

    pub fn magnetic(&self) -> &CartesianVector3<C, D> {
        self.cartesian.magnetic()
    }
}

/// Per-layer and whole-stack physical power balance.

#[derive(Clone, Debug, PartialEq)]
pub(super) struct AlgebraicFieldSample<C, D, A>
where
    C: ComplexField + Copy,
    D: Dimension,
    A: ScalarAlgebra<C, D>,
{
    pub(super) position: FieldPosition<C::RealField>,
    pub(super) coordinate: C::RealField,
    pub(super) canonical: IsotropicFieldState<A>,
    pub(super) cartesian: CartesianElectromagneticField<A::Vector>,
}

impl<C, D> AlgebraicFieldSample<C, D, ArrayJetFirst<C, D>>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn into_value_and_first(
        self,
    ) -> (PlaneWaveFieldSample<C, D>, PlaneWaveFieldDifferential<C, D>) {
        let (canonical, canonical_first) = self.canonical.split();

        let (cartesian, cartesian_first) = self.cartesian.split();

        (
            PlaneWaveFieldSample::new(self.position, self.coordinate, canonical, cartesian),
            PlaneWaveFieldDifferential::new(canonical_first, cartesian_first),
        )
    }
}

impl<C, D> AlgebraicFieldSample<C, D, ArrayJet<C, D>>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    fn into_value_first_and_second(
        self,
    ) -> (
        PlaneWaveFieldSample<C, D>,
        PlaneWaveFieldDifferential<C, D>,
        PlaneWaveFieldDifferential<C, D>,
    ) {
        let (canonical, canonical_first, canonical_second) = self.canonical.split();

        let (cartesian, cartesian_first, cartesian_second) = self.cartesian.split();

        (
            PlaneWaveFieldSample::new(self.position, self.coordinate, canonical, cartesian),
            PlaneWaveFieldDifferential::new(canonical_first, cartesian_first),
            PlaneWaveFieldDifferential::new(canonical_second, cartesian_second),
        )
    }
}
