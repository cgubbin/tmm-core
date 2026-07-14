//! Backend-neutral plane-wave scattering responses.
//!
//! This module defines the interface between planar electromagnetic backends
//! and downstream plane-wave observables.
//!
//! Backends may use transfer matrices, scattering matrices, or another native
//! representation. Implementations of [`PlaneWaveBackend`] translate those
//! representations into canonical complex reflection and transmission
//! amplitude coefficients.
//!
//! The response contains field-amplitude coefficients only. Power reflectance,
//! transmittance, absorption, spectral sweeps, plotting, and optimisation belong
//! in downstream observables crates.

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::{
        DerivativeVariable, PlaneWaveInput,
        jet::{ArrayJet, ArrayJetFirst},
    },
};

/// Backend capable of solving a physical plane-wave scattering problem.
///
/// Implementations translate their native backend representation into complex
/// reflection and transmission amplitude coefficients for a unit-amplitude
/// incident wave.
///
/// Consequently, callers do not need to know whether the backend uses a
/// transfer matrix, scattering matrix, or another internal representation.
///
/// Input coordinates and returned amplitudes have the same sampled dimension
/// `D`. No implicit broadcasting is performed.
pub trait PlaneWaveBackend<C, D, S>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Error produced during the plane-wave calculation.
    type Error;

    /// Solve for the complex reflection and transmission amplitude
    /// coefficients.
    fn solve_plane_wave(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error>;

    /// Solve for the amplitudes and their first derivatives with respect to
    /// `variable`.
    fn solve_plane_wave_first_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error>;

    /// Solve for the amplitudes and their first and second derivatives with
    /// respect to `variable`.
    fn solve_plane_wave_second_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error>;
}

/// Backend-neutral plane-wave scattering response.
///
/// This type contains complex reflection and transmission amplitude
/// coefficients for a unit-amplitude incident wave, together with any
/// derivatives requested during the same evaluation.
///
/// It does not expose the backend's raw transfer or scattering matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveResponse<C, D>
where
    D: Dimension,
{
    amplitudes: PlaneWaveAmplitudes<C, D>,
    derivatives: Option<PlaneWaveResponseDerivatives<C, D>>,
}

impl<C, D> PlaneWaveResponse<C, D>
where
    D: Dimension,
{
    /// Construct a response without derivatives.
    pub fn new(amplitudes: PlaneWaveAmplitudes<C, D>) -> Self {
        Self {
            amplitudes,
            derivatives: None,
        }
    }

    /// Construct a response containing amplitude derivatives.
    pub fn with_derivatives(
        amplitudes: PlaneWaveAmplitudes<C, D>,
        derivatives: PlaneWaveResponseDerivatives<C, D>,
    ) -> Self {
        Self {
            amplitudes,
            derivatives: Some(derivatives),
        }
    }

    /// Construct a response by consuming first-order reflection and
    /// transmission jets.
    pub(crate) fn from_first_jets(
        reflection: ArrayJetFirst<C, D>,
        transmission: ArrayJetFirst<C, D>,
        variable: DerivativeVariable,
    ) -> Self {
        let (reflection, reflection_first) = reflection.into_parts();
        let (transmission, transmission_first) = transmission.into_parts();

        let amplitudes = PlaneWaveAmplitudes::new(reflection, transmission);

        let first = PlaneWaveAmplitudes::new(reflection_first, transmission_first);

        Self::with_derivatives(
            amplitudes,
            PlaneWaveResponseDerivatives::new(variable, first),
        )
    }

    /// Construct a response by consuming second-order reflection and
    /// transmission jets.
    pub(crate) fn from_second_jets(
        reflection: ArrayJet<C, D>,
        transmission: ArrayJet<C, D>,
        variable: DerivativeVariable,
    ) -> Self {
        let (reflection, reflection_first, reflection_second) = reflection.into_parts();

        let (transmission, transmission_first, transmission_second) = transmission.into_parts();

        let amplitudes = PlaneWaveAmplitudes::new(reflection, transmission);

        let first = PlaneWaveAmplitudes::new(reflection_first, transmission_first);

        let second = PlaneWaveAmplitudes::new(reflection_second, transmission_second);

        Self::with_derivatives(
            amplitudes,
            PlaneWaveResponseDerivatives::new(variable, first).with_second(second),
        )
    }

    /// Return the reflection and transmission amplitude coefficients.
    pub fn amplitudes(&self) -> &PlaneWaveAmplitudes<C, D> {
        &self.amplitudes
    }

    /// Return the complex reflection amplitude coefficient.
    pub fn reflection(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        self.amplitudes.reflection()
    }

    /// Return the complex transmission amplitude coefficient.
    pub fn transmission(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        self.amplitudes.transmission()
    }

    /// Return amplitude derivatives, when available.
    pub fn derivatives(&self) -> Option<&PlaneWaveResponseDerivatives<C, D>> {
        self.derivatives.as_ref()
    }

    /// Consume the response and return its amplitude coefficients.
    ///
    /// Any stored derivatives are discarded.
    pub fn into_amplitudes(self) -> PlaneWaveAmplitudes<C, D> {
        self.amplitudes
    }

    /// Consume the response and return its amplitudes and optional
    /// derivatives.
    pub fn into_parts(
        self,
    ) -> (
        PlaneWaveAmplitudes<C, D>,
        Option<PlaneWaveResponseDerivatives<C, D>>,
    ) {
        (self.amplitudes, self.derivatives)
    }
}

/// Complex reflection and transmission amplitude coefficients.
///
/// For a unit-amplitude incident field:
///
/// ```text
/// reflected field   = r
/// transmitted field = t
/// ```
///
/// For an arbitrary incident amplitude `a`:
///
/// ```text
/// reflected field   = r a
/// transmitted field = t a
/// ```
///
/// These are field-amplitude coefficients, not power coefficients.
/// Transmittance generally requires exterior-medium flux normalisation.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveAmplitudes<C, D>
where
    D: Dimension,
{
    reflection: ArrayBase<OwnedRepr<C>, D>,
    transmission: ArrayBase<OwnedRepr<C>, D>,
}

impl<C, D> PlaneWaveAmplitudes<C, D>
where
    D: Dimension,
{
    /// Construct a pair of complex amplitude coefficients.
    pub fn new(
        reflection: ArrayBase<OwnedRepr<C>, D>,
        transmission: ArrayBase<OwnedRepr<C>, D>,
    ) -> Self {
        Self {
            reflection,
            transmission,
        }
    }

    /// Return the complex reflection amplitude coefficient.
    pub fn reflection(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.reflection
    }

    /// Return the complex transmission amplitude coefficient.
    pub fn transmission(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.transmission
    }

    /// Consume the pair and return its amplitude arrays.
    pub fn into_parts(self) -> (ArrayBase<OwnedRepr<C>, D>, ArrayBase<OwnedRepr<C>, D>) {
        (self.reflection, self.transmission)
    }
}

/// Derivatives of plane-wave amplitude coefficients.
///
/// A first derivative of both reflection and transmission is always present.
/// A second derivative pair is present only when the response was produced by
/// a second-derivative backend method.
///
/// These are derivatives of complex field amplitudes, not derivatives of
/// reflected or transmitted power.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveResponseDerivatives<C, D>
where
    D: Dimension,
{
    variable: DerivativeVariable,
    first: PlaneWaveAmplitudes<C, D>,
    second: Option<PlaneWaveAmplitudes<C, D>>,
}

impl<C, D> PlaneWaveResponseDerivatives<C, D>
where
    D: Dimension,
{
    /// Construct first-order amplitude derivatives.
    pub fn new(variable: DerivativeVariable, first: PlaneWaveAmplitudes<C, D>) -> Self {
        Self {
            variable,
            first,
            second: None,
        }
    }

    /// Attach the corresponding second-order amplitude derivatives.
    pub fn with_second(mut self, second: PlaneWaveAmplitudes<C, D>) -> Self {
        self.second = Some(second);
        self
    }

    /// Return the independent derivative variable.
    pub fn variable(&self) -> DerivativeVariable {
        self.variable
    }

    /// Return first derivatives of the reflection and transmission
    /// amplitudes.
    pub fn first(&self) -> &PlaneWaveAmplitudes<C, D> {
        &self.first
    }

    /// Return second derivatives of the reflection and transmission
    /// amplitudes, when available.
    pub fn second(&self) -> Option<&PlaneWaveAmplitudes<C, D>> {
        self.second.as_ref()
    }

    /// Consume the derivative result and return all components.
    pub fn into_parts(
        self,
    ) -> (
        DerivativeVariable,
        PlaneWaveAmplitudes<C, D>,
        Option<PlaneWaveAmplitudes<C, D>>,
    ) {
        (self.variable, self.first, self.second)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::arr0;
    use num_complex::Complex64;

    use super::*;

    fn c(value: f64) -> Complex64 {
        Complex64::new(value, 0.0)
    }

    #[test]
    fn value_only_response_contains_amplitudes() {
        let response =
            PlaneWaveResponse::new(PlaneWaveAmplitudes::new(arr0(c(0.25)), arr0(c(0.75))));

        assert_eq!(response.reflection()[()], c(0.25));
        assert_eq!(response.transmission()[()], c(0.75));
        assert!(response.derivatives().is_none());
    }

    #[test]
    fn first_jet_conversion_preserves_all_components() {
        let reflection = ArrayJetFirst::from_parts(arr0(c(0.25)), arr0(c(0.1)));

        let transmission = ArrayJetFirst::from_parts(arr0(c(0.75)), arr0(c(-0.2)));

        let response = PlaneWaveResponse::from_first_jets(
            reflection,
            transmission,
            DerivativeVariable::VacuumWavenumber,
        );

        assert_eq!(response.reflection()[()], c(0.25));
        assert_eq!(response.transmission()[()], c(0.75));

        let derivatives = response.derivatives().unwrap();

        assert_eq!(derivatives.variable(), DerivativeVariable::VacuumWavenumber);
        assert_eq!(derivatives.first().reflection()[()], c(0.1));
        assert_eq!(derivatives.first().transmission()[()], c(-0.2));
        assert!(derivatives.second().is_none());
    }

    #[test]
    fn second_jet_conversion_preserves_all_components() {
        let reflection = ArrayJet::from_parts(arr0(c(0.25)), arr0(c(0.1)), arr0(c(0.05)));

        let transmission = ArrayJet::from_parts(arr0(c(0.75)), arr0(c(-0.2)), arr0(c(-0.1)));

        let response = PlaneWaveResponse::from_second_jets(
            reflection,
            transmission,
            DerivativeVariable::ParallelWavenumberSquared,
        );

        let derivatives = response.derivatives().unwrap();
        let second = derivatives.second().unwrap();

        assert_eq!(second.reflection()[()], c(0.05));
        assert_eq!(second.transmission()[()], c(-0.1));
    }

    #[test]
    fn amplitude_pair_into_parts_preserves_values() {
        let amplitudes = PlaneWaveAmplitudes::new(arr0(c(0.25)), arr0(c(0.75)));

        let (reflection, transmission) = amplitudes.into_parts();

        assert_eq!(reflection[()], c(0.25));
        assert_eq!(transmission[()], c(0.75));
    }

    #[test]
    fn response_derivatives_into_parts_preserves_values() {
        let derivatives = PlaneWaveResponseDerivatives::new(
            DerivativeVariable::Thickness(2),
            PlaneWaveAmplitudes::new(arr0(c(0.1)), arr0(c(-0.2))),
        )
        .with_second(PlaneWaveAmplitudes::new(arr0(c(0.05)), arr0(c(-0.1))));

        let (variable, first, second) = derivatives.into_parts();

        assert_eq!(variable, DerivativeVariable::Thickness(2));
        assert_eq!(first.reflection()[()], c(0.1));
        assert_eq!(second.unwrap().transmission()[()], c(-0.1));
    }
}
