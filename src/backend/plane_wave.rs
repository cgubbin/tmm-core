//! Backend-neutral physical plane-wave scattering responses.
//!
//! This module defines the interface between planar electromagnetic backends
//! and downstream physical plane-wave observables.
//!
//! Backends may use transfer matrices, scattering matrices, or another native
//! representation. Implementations of [`PlaneWaveBackend`] translate those
//! representations into:
//!
//! - complex reflection and transmission amplitude coefficients;
//! - real power reflectance and transmittance;
//! - optional first and second derivatives of both sets of quantities.
//!
//! Plane-wave inputs are real-valued. This ensures that power observables and
//! their derivatives are derivatives along real physical coordinates. Complex
//! spectral evaluation remains available through the raw-matrix and
//! outgoing-mode backend capabilities.

use ndarray::{ArrayBase, Dimension, OwnedRepr, Zip};
use num_traits::Float;

use crate::{
    ComplexScalar,
    backend::{
        DerivativeVariable, PlaneWaveInput,
        derivative::{SpectralDerivativeVariable, StructuralDerivativeVariable},
        jet::{ArrayJet, ArrayJetFirst},
    },
};

/// Backend capable of solving a physical plane-wave scattering problem.
///
/// Implementations translate their native backend representation into complex
/// reflection and transmission amplitudes and the corresponding real power
/// coefficients for a unit-amplitude incident wave.
///
/// Input coordinates are real-valued. Returned amplitudes remain complex so
/// that lossy materials and phase information are represented correctly.
/// Derivatives are taken along the requested real coordinate.
///
/// Input coordinates and response arrays have the same sampled dimension `D`.
/// No implicit broadcasting is performed.
pub trait PlaneWaveBackend<C, D, S>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Error produced during the plane-wave calculation.
    type Error;

    /// Solve for plane-wave amplitudes and power coefficients.
    fn solve_plane_wave(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error>;

    /// Solve for the response and its first derivative with respect to
    /// `variable`.
    fn solve_plane_wave_structural_first_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error>;

    /// Solve for the response and its first and second derivatives with
    /// respect to `variable`.
    fn solve_plane_wave_structural_second_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error>;
}

pub trait DifferentiablePlaneWaveBackend<C, D, S>: PlaneWaveBackend<C, D, S>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Solve for the response and its first derivative with respect to
    /// `variable`.
    fn solve_plane_wave_spectral_first_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error>;

    /// Solve for the response and its first and second derivatives with
    /// respect to `variable`.
    fn solve_plane_wave_spectral_second_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error>;
}

/// Backend-neutral physical plane-wave scattering response.
///
/// The response contains complex field-amplitude coefficients and real power
/// coefficients for a unit-amplitude incident wave. Optional derivatives are
/// derivatives along one real [`DerivativeVariable`].
///
/// It does not expose the backend's raw transfer or scattering matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveResponse<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    amplitudes: PlaneWaveAmplitudes<C, D>,
    power: PlaneWavePower<C::RealField, D>,
    derivatives: Option<PlaneWaveResponseDerivatives<C, D>>,
}

impl<C, D> PlaneWaveResponse<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Construct a value-only physical plane-wave response.
    pub fn new(
        amplitudes: PlaneWaveAmplitudes<C, D>,
        power: PlaneWavePower<C::RealField, D>,
    ) -> Self {
        Self {
            amplitudes,
            power,
            derivatives: None,
        }
    }

    /// Construct a physical response containing derivatives.
    pub fn with_derivatives(
        amplitudes: PlaneWaveAmplitudes<C, D>,
        power: PlaneWavePower<C::RealField, D>,
        derivatives: PlaneWaveResponseDerivatives<C, D>,
    ) -> Self {
        Self {
            amplitudes,
            power,
            derivatives: Some(derivatives),
        }
    }

    /// Return the complex reflection and transmission amplitudes.
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

    /// Return the real power coefficients.
    pub fn power(&self) -> &PlaneWavePower<C::RealField, D> {
        &self.power
    }

    /// Return the power absorptance.
    pub fn absorptance(&self) -> &ArrayBase<OwnedRepr<C::RealField>, D> {
        self.power.absorptance()
    }

    /// Return the power reflectance.
    pub fn reflectance(&self) -> &ArrayBase<OwnedRepr<C::RealField>, D> {
        self.power.reflectance()
    }

    /// Return the power transmittance.
    pub fn transmittance(&self) -> &ArrayBase<OwnedRepr<C::RealField>, D> {
        self.power.transmittance()
    }

    /// Return response derivatives, when available.
    pub fn derivatives(&self) -> Option<&PlaneWaveResponseDerivatives<C, D>> {
        self.derivatives.as_ref()
    }

    /// Return first derivatives, when available.
    pub fn first_derivatives(&self) -> Option<&PlaneWaveResponseDifferential<C, D>> {
        self.derivatives
            .as_ref()
            .map(PlaneWaveResponseDerivatives::first)
    }

    /// Return second derivatives, when available.
    pub fn second_derivatives(&self) -> Option<&PlaneWaveResponseDifferential<C, D>> {
        self.derivatives
            .as_ref()
            .and_then(PlaneWaveResponseDerivatives::second)
    }

    /// Consume the response and return all components.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        PlaneWaveAmplitudes<C, D>,
        PlaneWavePower<C::RealField, D>,
        Option<PlaneWaveResponseDerivatives<C, D>>,
    ) {
        (self.amplitudes, self.power, self.derivatives)
    }
}

impl<C, D> PlaneWaveResponse<C, D>
where
    C: ComplexScalar,
    C::RealField: Float,
    D: Dimension,
{
    pub(crate) fn from_values(
        reflection: ArrayBase<OwnedRepr<C>, D>,
        transmission: ArrayBase<OwnedRepr<C>, D>,
        incident_normalisation: ArrayBase<OwnedRepr<C>, D>,
        transmitted_normalisation: ArrayBase<OwnedRepr<C>, D>,
    ) -> Self {
        let amplitudes = PlaneWaveAmplitudes::new(reflection, transmission);

        let power = PlaneWavePower::from_amplitudes(
            &amplitudes,
            &incident_normalisation,
            &transmitted_normalisation,
        );

        Self {
            amplitudes,
            power,
            derivatives: None,
        }
    }

    pub(crate) fn from_first_jets(
        reflection: ArrayJetFirst<C, D>,
        transmission: ArrayJetFirst<C, D>,
        incident_normalisation: ArrayJetFirst<C, D>,
        transmitted_normalisation: ArrayJetFirst<C, D>,
        variable: DerivativeVariable,
    ) -> Self {
        let (reflection, d_reflection) = reflection.into_parts();

        let (transmission, d_transmission) = transmission.into_parts();

        let (incident_normalisation, d_incident_normalisation) =
            incident_normalisation.into_parts();

        let (transmitted_normalisation, d_transmitted_normalisation) =
            transmitted_normalisation.into_parts();

        let amplitudes = PlaneWaveAmplitudes::new(reflection, transmission);

        let amplitude_first = PlaneWaveAmplitudeDifferential::new(d_reflection, d_transmission);

        let power = PlaneWavePower::from_amplitudes(
            &amplitudes,
            &incident_normalisation,
            &transmitted_normalisation,
        );

        let power_first = PlaneWavePowerDifferential::first_from_parts(
            &amplitudes,
            &amplitude_first,
            &incident_normalisation,
            &d_incident_normalisation,
            &transmitted_normalisation,
            &d_transmitted_normalisation,
        );

        let first = PlaneWaveResponseDifferential::new(amplitude_first, power_first);

        Self {
            amplitudes,
            power,
            derivatives: Some(PlaneWaveResponseDerivatives::new(variable, first)),
        }
    }

    pub(crate) fn from_second_jets(
        reflection: ArrayJet<C, D>,
        transmission: ArrayJet<C, D>,
        incident_normalisation: ArrayJet<C, D>,
        transmitted_normalisation: ArrayJet<C, D>,
        variable: DerivativeVariable,
    ) -> Self {
        let (reflection, d_reflection, dd_reflection) = reflection.into_parts();

        let (transmission, d_transmission, dd_transmission) = transmission.into_parts();

        let (incident_normalisation, d_incident_normalisation, dd_incident_normalisation) =
            incident_normalisation.into_parts();

        let (transmitted_normalisation, d_transmitted_normalisation, dd_transmitted_normalisation) =
            transmitted_normalisation.into_parts();

        let amplitudes = PlaneWaveAmplitudes::new(reflection, transmission);

        let amplitude_first = PlaneWaveAmplitudeDifferential::new(d_reflection, d_transmission);

        let amplitude_second = PlaneWaveAmplitudeDifferential::new(dd_reflection, dd_transmission);

        let power = PlaneWavePower::from_amplitudes(
            &amplitudes,
            &incident_normalisation,
            &transmitted_normalisation,
        );

        let power_first = PlaneWavePowerDifferential::first_from_parts(
            &amplitudes,
            &amplitude_first,
            &incident_normalisation,
            &d_incident_normalisation,
            &transmitted_normalisation,
            &d_transmitted_normalisation,
        );

        let power_second = PlaneWavePowerDifferential::second_from_parts(
            &amplitudes,
            &amplitude_first,
            &amplitude_second,
            &incident_normalisation,
            &d_incident_normalisation,
            &dd_incident_normalisation,
            &transmitted_normalisation,
            &d_transmitted_normalisation,
            &dd_transmitted_normalisation,
        );

        let first = PlaneWaveResponseDifferential::new(amplitude_first, power_first);

        let second = PlaneWaveResponseDifferential::new(amplitude_second, power_second);

        Self {
            amplitudes,
            power,
            derivatives: Some(
                PlaneWaveResponseDerivatives::new(variable, first).with_second(second),
            ),
        }
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
/// These are field-amplitude coefficients rather than power coefficients.
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
    /// Construct complex reflection and transmission amplitudes.
    pub fn new(
        reflection: ArrayBase<OwnedRepr<C>, D>,
        transmission: ArrayBase<OwnedRepr<C>, D>,
    ) -> Self {
        debug_assert_eq!(reflection.raw_dim(), transmission.raw_dim());

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

/// Real power reflectance and transmittance.
///
/// Reflectance and transmittance are defined relative to the incident power
/// flux. The backend is responsible for applying the appropriate transmitted-
/// to-incident port flux ratio when constructing `transmittance`.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWavePower<R, D>
where
    D: Dimension,
{
    reflectance: ArrayBase<OwnedRepr<R>, D>,
    transmittance: ArrayBase<OwnedRepr<R>, D>,
    absorptance: ArrayBase<OwnedRepr<R>, D>,
}

impl<R, D> PlaneWavePower<R, D>
where
    D: Dimension,
{
    /// Construct real power reflectance and transmittance arrays.
    pub fn new(
        reflectance: ArrayBase<OwnedRepr<R>, D>,
        transmittance: ArrayBase<OwnedRepr<R>, D>,
    ) -> Self
    where
        R: Float,
    {
        debug_assert_eq!(reflectance.raw_dim(), transmittance.raw_dim());

        let absorptance = reflectance.mapv(|value| R::one() - value) - transmittance.view();

        Self {
            reflectance,
            transmittance,
            absorptance,
        }
    }

    fn from_amplitudes<C>(
        amplitudes: &PlaneWaveAmplitudes<C, D>,
        incident_normalisation: &ArrayBase<OwnedRepr<C>, D>,
        transmitted_normalisation: &ArrayBase<OwnedRepr<C>, D>,
    ) -> Self
    where
        C: ComplexScalar<RealField = R>,
        R: Float,
    {
        let incident_flux = incident_normalisation.mapv(|value| value.real());

        let transmitted_flux = transmitted_normalisation.mapv(|value| value.real());

        let transmission_factor = transmitted_flux / incident_flux.view();

        let reflectance = amplitudes
            .reflection()
            .mapv(|value| value.modulus_squared());

        let transmittance = amplitudes
            .transmission()
            .mapv(|value| value.modulus_squared())
            * transmission_factor;

        let absorptance = reflectance.mapv(|value| R::one() - value) - transmittance.view();

        Self {
            reflectance,
            transmittance,
            absorptance,
        }
    }

    /// Return the power reflectance.
    pub fn reflectance(&self) -> &ArrayBase<OwnedRepr<R>, D> {
        &self.reflectance
    }

    /// Return the power transmittance.
    pub fn transmittance(&self) -> &ArrayBase<OwnedRepr<R>, D> {
        &self.transmittance
    }

    /// Return total absorptance as `1 - R - T`.
    pub fn absorptance(&self) -> &ArrayBase<OwnedRepr<R>, D> {
        &self.absorptance
    }

    /// Consume the value and return its power arrays.
    pub fn into_parts(
        self,
    ) -> (
        ArrayBase<OwnedRepr<R>, D>,
        ArrayBase<OwnedRepr<R>, D>,
        ArrayBase<OwnedRepr<R>, D>,
    ) {
        (self.reflectance, self.transmittance, self.absorptance)
    }
}

/// First and optional second derivatives of a physical plane-wave response.
///
/// Each differential contains complex amplitude derivatives and real power
/// derivatives of the same order with respect to `variable`.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveResponseDerivatives<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    variable: DerivativeVariable,
    first: PlaneWaveResponseDifferential<C, D>,
    second: Option<PlaneWaveResponseDifferential<C, D>>,
}

impl<C, D> PlaneWaveResponseDerivatives<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Construct first-order response derivatives.
    pub fn new(variable: DerivativeVariable, first: PlaneWaveResponseDifferential<C, D>) -> Self {
        Self {
            variable,
            first,
            second: None,
        }
    }

    /// Attach the corresponding second-order response derivatives.
    pub fn with_second(mut self, second: PlaneWaveResponseDifferential<C, D>) -> Self {
        self.second = Some(second);
        self
    }

    /// Return the independent derivative variable.
    pub fn variable(&self) -> DerivativeVariable {
        self.variable
    }

    /// Return the first response derivative.
    pub fn first(&self) -> &PlaneWaveResponseDifferential<C, D> {
        &self.first
    }

    /// Return the second response derivative, when available.
    pub fn second(&self) -> Option<&PlaneWaveResponseDifferential<C, D>> {
        self.second.as_ref()
    }

    /// Consume the derivative result and return all components.
    pub fn into_parts(
        self,
    ) -> (
        DerivativeVariable,
        PlaneWaveResponseDifferential<C, D>,
        Option<PlaneWaveResponseDifferential<C, D>>,
    ) {
        (self.variable, self.first, self.second)
    }
}

/// Derivative of all physical plane-wave response quantities at one order.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveResponseDifferential<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    amplitudes: PlaneWaveAmplitudeDifferential<C, D>,
    power: PlaneWavePowerDifferential<C::RealField, D>,
}

impl<C, D> PlaneWaveResponseDifferential<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Construct one derivative order of the response.
    pub fn new(
        amplitudes: PlaneWaveAmplitudeDifferential<C, D>,
        power: PlaneWavePowerDifferential<C::RealField, D>,
    ) -> Self {
        Self { amplitudes, power }
    }

    /// Return complex amplitude derivatives.
    pub fn amplitudes(&self) -> &PlaneWaveAmplitudeDifferential<C, D> {
        &self.amplitudes
    }

    /// Return the reflection-amplitude derivative.
    pub fn reflection(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        self.amplitudes.reflection()
    }

    /// Return the transmission-amplitude derivative.
    pub fn transmission(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        self.amplitudes.transmission()
    }

    /// Return real power derivatives.
    pub fn power(&self) -> &PlaneWavePowerDifferential<C::RealField, D> {
        &self.power
    }

    /// Return the reflectance derivative.
    pub fn reflectance(&self) -> &ArrayBase<OwnedRepr<C::RealField>, D> {
        self.power.reflectance()
    }

    /// Return the transmittance derivative.
    pub fn transmittance(&self) -> &ArrayBase<OwnedRepr<C::RealField>, D> {
        self.power.transmittance()
    }

    /// Consume the differential and return its components.
    pub fn into_parts(
        self,
    ) -> (
        PlaneWaveAmplitudeDifferential<C, D>,
        PlaneWavePowerDifferential<C::RealField, D>,
    ) {
        (self.amplitudes, self.power)
    }
}

/// Complex derivatives of the reflection and transmission amplitudes.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveAmplitudeDifferential<C, D>
where
    D: Dimension,
{
    reflection: ArrayBase<OwnedRepr<C>, D>,
    transmission: ArrayBase<OwnedRepr<C>, D>,
}

impl<C, D> PlaneWaveAmplitudeDifferential<C, D>
where
    D: Dimension,
{
    /// Construct complex amplitude derivatives.
    pub fn new(
        reflection: ArrayBase<OwnedRepr<C>, D>,
        transmission: ArrayBase<OwnedRepr<C>, D>,
    ) -> Self {
        debug_assert_eq!(reflection.raw_dim(), transmission.raw_dim());

        Self {
            reflection,
            transmission,
        }
    }

    /// Return the reflection-amplitude derivative.
    pub fn reflection(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.reflection
    }

    /// Return the transmission-amplitude derivative.
    pub fn transmission(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.transmission
    }

    /// Consume the value and return its arrays.
    pub fn into_parts(self) -> (ArrayBase<OwnedRepr<C>, D>, ArrayBase<OwnedRepr<C>, D>) {
        (self.reflection, self.transmission)
    }
}

/// Real derivatives of reflectance and transmittance.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWavePowerDifferential<R, D>
where
    D: Dimension,
{
    reflectance: ArrayBase<OwnedRepr<R>, D>,
    transmittance: ArrayBase<OwnedRepr<R>, D>,
}

impl<R, D> PlaneWavePowerDifferential<R, D>
where
    D: Dimension,
{
    /// Construct real power derivatives.
    pub fn new(
        reflectance: ArrayBase<OwnedRepr<R>, D>,
        transmittance: ArrayBase<OwnedRepr<R>, D>,
    ) -> Self {
        debug_assert_eq!(reflectance.raw_dim(), transmittance.raw_dim());

        Self {
            reflectance,
            transmittance,
        }
    }

    /// Return the reflectance derivative.
    pub fn reflectance(&self) -> &ArrayBase<OwnedRepr<R>, D> {
        &self.reflectance
    }

    /// Return the transmittance derivative.
    pub fn transmittance(&self) -> &ArrayBase<OwnedRepr<R>, D> {
        &self.transmittance
    }

    /// Return the absorptance derivative as `-dR - dT`.
    ///
    /// This formula is valid for both first and second derivatives because the
    /// derivative of the constant incident-power normalisation is zero.
    pub fn absorptance(&self) -> ArrayBase<OwnedRepr<R>, D>
    where
        R: Float,
    {
        -self.reflectance.clone() - self.transmittance.view()
    }

    /// Consume the value and return its arrays.
    pub fn into_parts(self) -> (ArrayBase<OwnedRepr<R>, D>, ArrayBase<OwnedRepr<R>, D>) {
        (self.reflectance, self.transmittance)
    }

    /// Construct first derivatives of reflectance and transmittance.
    ///
    /// For a real independent variable `x`,
    ///
    /// ```text
    /// R  = |r|²
    /// R' = 2 Re(r* r')
    ///
    /// T  = η |t|²
    /// T' = η' |t|² + 2 η Re(t* t')
    ///
    /// η  = F_t / F_i
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn first_from_parts<C>(
        amplitudes: &PlaneWaveAmplitudes<C, D>,
        amplitude_first: &PlaneWaveAmplitudeDifferential<C, D>,
        incident_normalisation: &ArrayBase<OwnedRepr<C>, D>,
        incident_normalisation_first: &ArrayBase<OwnedRepr<C>, D>,
        transmitted_normalisation: &ArrayBase<OwnedRepr<C>, D>,
        transmitted_normalisation_first: &ArrayBase<OwnedRepr<C>, D>,
    ) -> Self
    where
        C: ComplexScalar<RealField = R>,
        R: Float,
    {
        let two = R::one() + R::one();

        let incident_flux = incident_normalisation.mapv(|value| value.real());

        let incident_flux_first = incident_normalisation_first.mapv(|value| value.real());

        let transmitted_flux = transmitted_normalisation.mapv(|value| value.real());

        let transmitted_flux_first = transmitted_normalisation_first.mapv(|value| value.real());

        let incident_flux_squared = incident_flux.mapv(|value| value * value);

        let transmission_factor = transmitted_flux.clone() / incident_flux.view();

        let transmission_factor_first = (transmitted_flux_first * incident_flux.view()
            - transmitted_flux * incident_flux_first)
            / incident_flux_squared;

        let reflectance_first = Zip::from(amplitudes.reflection())
            .and(amplitude_first.reflection())
            .map_collect(|&reflection, &reflection_first| {
                two * (reflection.conjugate() * reflection_first).real()
            });

        let transmission_norm = amplitudes
            .transmission()
            .mapv(|value| value.modulus_squared());

        let transmission_norm_first = Zip::from(amplitudes.transmission())
            .and(amplitude_first.transmission())
            .map_collect(|&transmission, &transmission_first| {
                two * (transmission.conjugate() * transmission_first).real()
            });

        let transmittance_first = transmission_factor_first * transmission_norm.view()
            + transmission_factor * transmission_norm_first;

        Self {
            reflectance: reflectance_first,
            transmittance: transmittance_first,
        }
    }

    /// Construct second derivatives of reflectance and transmittance.
    ///
    /// For a real independent variable `x`,
    ///
    /// ```text
    /// R'' = 2(|r'|² + Re(r* r''))
    ///
    /// T'' =
    ///     η'' |t|²
    ///     + 4 η' Re(t* t')
    ///     + 2 η (|t'|² + Re(t* t''))
    /// ```
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn second_from_parts<C>(
        amplitudes: &PlaneWaveAmplitudes<C, D>,
        amplitude_first: &PlaneWaveAmplitudeDifferential<C, D>,
        amplitude_second: &PlaneWaveAmplitudeDifferential<C, D>,
        incident_normalisation: &ArrayBase<OwnedRepr<C>, D>,
        incident_normalisation_first: &ArrayBase<OwnedRepr<C>, D>,
        incident_normalisation_second: &ArrayBase<OwnedRepr<C>, D>,
        transmitted_normalisation: &ArrayBase<OwnedRepr<C>, D>,
        transmitted_normalisation_first: &ArrayBase<OwnedRepr<C>, D>,
        transmitted_normalisation_second: &ArrayBase<OwnedRepr<C>, D>,
    ) -> Self
    where
        C: ComplexScalar<RealField = R>,
        R: Float,
    {
        let two = R::one() + R::one();

        let incident_flux = incident_normalisation.mapv(|value| value.real());

        let incident_flux_first = incident_normalisation_first.mapv(|value| value.real());

        let incident_flux_second = incident_normalisation_second.mapv(|value| value.real());

        let transmitted_flux = transmitted_normalisation.mapv(|value| value.real());

        let transmitted_flux_first = transmitted_normalisation_first.mapv(|value| value.real());

        let transmitted_flux_second = transmitted_normalisation_second.mapv(|value| value.real());

        let incident_flux_squared = incident_flux.mapv(|value| value * value);

        let incident_flux_cubed = incident_flux.mapv(|value| value * value * value);

        let transmission_factor = transmitted_flux.clone() / incident_flux.view();

        let transmission_factor_first = (transmitted_flux_first.clone() * incident_flux.view()
            - transmitted_flux.clone() * incident_flux_first.view())
            / incident_flux_squared.view();

        /*
         * η = a / b
         *
         * η'' =
         *     a'' / b
         *     - a b'' / b²
         *     - 2 a' b' / b²
         *     + 2 a (b')² / b³
         */
        let transmission_factor_second = transmitted_flux_second / incident_flux.view()
            - transmitted_flux.clone() * incident_flux_second / incident_flux_squared.view()
            - transmitted_flux_first * incident_flux_first.mapv(|x| x * two)
                / incident_flux_squared
            + transmitted_flux * incident_flux_first.mapv(|value| value * value * two)
                / incident_flux_cubed;

        let reflectance_second = Zip::from(amplitudes.reflection())
            .and(amplitude_first.reflection())
            .and(amplitude_second.reflection())
            .map_collect(|&reflection, &reflection_first, &reflection_second| {
                two * (reflection_first.modulus_squared()
                    + (reflection.conjugate() * reflection_second).real())
            });

        let transmission_norm = amplitudes
            .transmission()
            .mapv(|value| value.modulus_squared());

        let transmission_norm_first = Zip::from(amplitudes.transmission())
            .and(amplitude_first.transmission())
            .map_collect(|&transmission, &transmission_first| {
                two * (transmission.conjugate() * transmission_first).real()
            });

        let transmission_norm_second = Zip::from(amplitudes.transmission())
            .and(amplitude_first.transmission())
            .and(amplitude_second.transmission())
            .map_collect(|&transmission, &transmission_first, &transmission_second| {
                two * (transmission_first.modulus_squared()
                    + (transmission.conjugate() * transmission_second).real())
            });

        let transmittance_second = transmission_factor_second * transmission_norm.view()
            + transmission_factor_first * transmission_norm_first.mapv(|x| x * two)
            + transmission_factor * transmission_norm_second;

        /*
         * Equivalently, the middle term is:
         *
         *     4 η' Re(t* t')
         *
         * because:
         *
         *     d|t|²/dx = 2 Re(t* t').
         */

        Self {
            reflectance: reflectance_second,
            transmittance: transmittance_second,
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{arr0, array};
    use num_complex::Complex64;

    use super::*;

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn amplitudes() -> PlaneWaveAmplitudes<C, ndarray::Ix0> {
        PlaneWaveAmplitudes::new(arr0(c(0.25)), arr0(c(0.75)))
    }

    fn power() -> PlaneWavePower<f64, ndarray::Ix0> {
        PlaneWavePower::new(arr0(0.0625), arr0(0.75))
    }

    #[test]
    fn value_only_response_contains_amplitudes_and_power() {
        let response = PlaneWaveResponse::new(amplitudes(), power());

        assert_eq!(response.reflection()[()], c(0.25));
        assert_eq!(response.transmission()[()], c(0.75));
        assert_eq!(response.reflectance()[()], 0.0625);
        assert_eq!(response.transmittance()[()], 0.75);
        assert_eq!(response.power().absorptance()[()], 0.1875);
        assert!(response.derivatives().is_none());
    }

    #[test]
    fn first_order_response_preserves_all_components() {
        let first = PlaneWaveResponseDifferential::new(
            PlaneWaveAmplitudeDifferential::new(arr0(c(0.1)), arr0(c(-0.2))),
            PlaneWavePowerDifferential::new(arr0(0.05), arr0(-0.03)),
        );

        let response = PlaneWaveResponse::with_derivatives(
            amplitudes(),
            power(),
            PlaneWaveResponseDerivatives::new(DerivativeVariable::VacuumWavenumber, first),
        );

        let derivatives = response.derivatives().unwrap();

        assert_eq!(derivatives.variable(), DerivativeVariable::VacuumWavenumber);
        assert_eq!(derivatives.first().reflection()[()], c(0.1));
        assert_eq!(derivatives.first().transmission()[()], c(-0.2));
        assert_eq!(derivatives.first().reflectance()[()], 0.05);
        assert_eq!(derivatives.first().transmittance()[()], -0.03);
        approx::assert_relative_eq!(
            derivatives.first().power().absorptance()[()],
            -0.02,
            epsilon = 1e-10
        );
        assert!(derivatives.second().is_none());
    }

    #[test]
    fn second_order_response_preserves_all_components() {
        let first = PlaneWaveResponseDifferential::new(
            PlaneWaveAmplitudeDifferential::new(arr0(c(0.1)), arr0(c(-0.2))),
            PlaneWavePowerDifferential::new(arr0(0.05), arr0(-0.03)),
        );

        let second = PlaneWaveResponseDifferential::new(
            PlaneWaveAmplitudeDifferential::new(arr0(c(0.04)), arr0(c(-0.08))),
            PlaneWavePowerDifferential::new(arr0(0.02), arr0(0.07)),
        );

        let response = PlaneWaveResponse::with_derivatives(
            amplitudes(),
            power(),
            PlaneWaveResponseDerivatives::new(DerivativeVariable::ParallelWavenumberSquared, first)
                .with_second(second),
        );

        let second = response.second_derivatives().unwrap();

        assert_eq!(second.reflection()[()], c(0.04));
        assert_eq!(second.transmission()[()], c(-0.08));
        assert_eq!(second.reflectance()[()], 0.02);
        assert_eq!(second.transmittance()[()], 0.07);
        approx::assert_relative_eq!(second.power().absorptance()[()], -0.09, epsilon = 1e-10);
    }

    #[test]
    fn response_into_parts_preserves_everything() {
        let response = PlaneWaveResponse::new(amplitudes(), power());

        let (amplitudes, power, derivatives) = response.into_parts();

        assert_eq!(amplitudes.reflection()[()], c(0.25));
        assert_eq!(amplitudes.transmission()[()], c(0.75));
        assert_eq!(power.reflectance()[()], 0.0625);
        assert_eq!(power.transmittance()[()], 0.75);
        assert!(derivatives.is_none());
    }

    #[test]
    fn sampled_response_preserves_shape() {
        let amplitudes = PlaneWaveAmplitudes::new(
            array![c(0.1), c(0.2), c(0.3)],
            array![c(0.7), c(0.6), c(0.5)],
        );

        let power = PlaneWavePower::new(array![0.01, 0.04, 0.09], array![0.8, 0.7, 0.6]);

        let response = PlaneWaveResponse::new(amplitudes, power);
        let expected = ndarray::Ix1(3);

        assert_eq!(response.reflection().raw_dim(), expected);
        assert_eq!(response.transmission().raw_dim(), expected);
        assert_eq!(response.reflectance().raw_dim(), expected);
        assert_eq!(response.transmittance().raw_dim(), expected);
    }

    #[test]
    fn amplitude_pair_into_parts_preserves_values() {
        let (reflection, transmission) = amplitudes().into_parts();

        assert_eq!(reflection[()], c(0.25));
        assert_eq!(transmission[()], c(0.75));
    }
}

#[cfg(test)]
mod power_tests {
    use super::*;
    use crate::{
        DerivativeVariable, PlanarInput, PlaneWaveResponse, Polarisation,
        backend::jet::{ArrayJet, ArrayJetFirst},
    };

    use approx::assert_relative_eq;
    use nalgebra::ComplexField;
    use ndarray::{ArrayBase, Dimension, Ix0, OwnedRepr, arr0, array};
    use num_complex::Complex64;

    type C = Complex64;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn assert_complex_close(actual: C, expected: C, tolerance: f64) {
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

    fn assert_real_close(actual: f64, expected: f64, tolerance: f64) {
        assert_relative_eq!(
            actual,
            expected,
            epsilon = tolerance,
            max_relative = tolerance,
        );
    }

    fn assert_real_array_close<D>(
        actual: &ArrayBase<OwnedRepr<f64>, D>,
        expected: &ArrayBase<OwnedRepr<f64>, D>,
        tolerance: f64,
    ) where
        D: Dimension,
    {
        assert_eq!(actual.raw_dim(), expected.raw_dim());

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_real_close(actual, expected, tolerance);
        }
    }

    #[test]
    fn value_response_computes_reflectance_and_transmittance() {
        let reflection = arr0(c(0.3, 0.4));
        let transmission = arr0(c(0.6, -0.2));

        let incident_normalisation = arr0(c(2.0, 0.5));
        let transmitted_normalisation = arr0(c(3.0, -0.4));

        let response = PlaneWaveResponse::from_values(
            reflection,
            transmission,
            incident_normalisation,
            transmitted_normalisation,
        );

        // |r|² = 0.3² + 0.4² = 0.25
        let expected_reflectance = 0.25;

        // |t|² = 0.6² + 0.2² = 0.40
        //
        // η = Re(Y_t) / Re(Y_i) = 3 / 2
        //
        // T = η |t|² = 0.6
        let expected_transmittance = 0.6;

        assert_real_close(
            response.power().reflectance()[()],
            expected_reflectance,
            1e-12,
        );

        assert_real_close(
            response.power().transmittance()[()],
            expected_transmittance,
            1e-12,
        );

        assert_real_close(
            response.power().absorptance()[()],
            1.0 - expected_reflectance - expected_transmittance,
            1e-12,
        );

        assert!(response.derivatives().is_none());
    }

    #[test]
    fn first_response_derivatives_match_direct_formula() {
        let reflection = c(0.3, 0.4);
        let reflection_first = c(0.2, -0.1);

        let transmission = c(0.6, -0.2);
        let transmission_first = c(-0.1, 0.3);

        let incident = c(2.0, 0.5);
        let incident_first = c(0.4, 0.2);

        let transmitted = c(3.0, -0.4);
        let transmitted_first = c(-0.2, 0.1);

        let response = PlaneWaveResponse::from_first_jets(
            ArrayJetFirst::from_parts(arr0(reflection), arr0(reflection_first)),
            ArrayJetFirst::from_parts(arr0(transmission), arr0(transmission_first)),
            ArrayJetFirst::from_parts(arr0(incident), arr0(incident_first)),
            ArrayJetFirst::from_parts(arr0(transmitted), arr0(transmitted_first)),
            DerivativeVariable::VacuumWavenumber,
        );

        let derivatives = response.derivatives().unwrap();
        let first = derivatives.first();

        assert_eq!(derivatives.variable(), DerivativeVariable::VacuumWavenumber,);

        assert_complex_close(first.amplitudes().reflection()[()], reflection_first, 1e-12);

        assert_complex_close(
            first.amplitudes().transmission()[()],
            transmission_first,
            1e-12,
        );

        let expected_reflectance_first = 2.0 * (reflection.conj() * reflection_first).re;

        let incident_flux = incident.re;
        let incident_flux_first = incident_first.re;

        let transmitted_flux = transmitted.re;
        let transmitted_flux_first = transmitted_first.re;

        let eta = transmitted_flux / incident_flux;

        let eta_first = (transmitted_flux_first * incident_flux
            - transmitted_flux * incident_flux_first)
            / incident_flux.powi(2);

        let transmission_norm = transmission.norm_sqr();

        let transmission_norm_first = 2.0 * (transmission.conj() * transmission_first).re;

        let expected_transmittance_first =
            eta_first * transmission_norm + eta * transmission_norm_first;

        assert_real_close(
            first.power().reflectance()[()],
            expected_reflectance_first,
            1e-12,
        );

        assert_real_close(
            first.power().transmittance()[()],
            expected_transmittance_first,
            1e-12,
        );

        assert_real_close(
            first.power().absorptance()[()],
            -expected_reflectance_first - expected_transmittance_first,
            1e-12,
        );

        assert!(derivatives.second().is_none());
    }

    #[test]
    fn second_response_derivatives_match_direct_formula() {
        let reflection = c(0.3, 0.4);
        let reflection_first = c(0.2, -0.1);
        let reflection_second = c(-0.05, 0.08);

        let transmission = c(0.6, -0.2);
        let transmission_first = c(-0.1, 0.3);
        let transmission_second = c(0.04, -0.06);

        let incident = c(2.0, 0.5);
        let incident_first = c(0.4, 0.2);
        let incident_second = c(-0.1, 0.3);

        let transmitted = c(3.0, -0.4);
        let transmitted_first = c(-0.2, 0.1);
        let transmitted_second = c(0.07, -0.05);

        let response = PlaneWaveResponse::from_second_jets(
            ArrayJet::from_parts(
                arr0(reflection),
                arr0(reflection_first),
                arr0(reflection_second),
            ),
            ArrayJet::from_parts(
                arr0(transmission),
                arr0(transmission_first),
                arr0(transmission_second),
            ),
            ArrayJet::from_parts(arr0(incident), arr0(incident_first), arr0(incident_second)),
            ArrayJet::from_parts(
                arr0(transmitted),
                arr0(transmitted_first),
                arr0(transmitted_second),
            ),
            DerivativeVariable::ParallelWavenumber,
        );

        let derivatives = response.derivatives().unwrap();

        let first = derivatives.first();
        let second = derivatives.second().unwrap();

        assert_complex_close(
            second.amplitudes().reflection()[()],
            reflection_second,
            1e-12,
        );

        assert_complex_close(
            second.amplitudes().transmission()[()],
            transmission_second,
            1e-12,
        );

        let expected_reflectance_first = 2.0 * (reflection.conj() * reflection_first).re;

        let expected_reflectance_second =
            2.0 * (reflection_first.norm_sqr() + (reflection.conj() * reflection_second).re);

        let incident_flux = incident.re;
        let incident_flux_first = incident_first.re;
        let incident_flux_second = incident_second.re;

        let transmitted_flux = transmitted.re;
        let transmitted_flux_first = transmitted_first.re;
        let transmitted_flux_second = transmitted_second.re;

        let eta = transmitted_flux / incident_flux;

        let eta_first = (transmitted_flux_first * incident_flux
            - transmitted_flux * incident_flux_first)
            / incident_flux.powi(2);

        let eta_second = transmitted_flux_second / incident_flux
            - transmitted_flux * incident_flux_second / incident_flux.powi(2)
            - 2.0 * transmitted_flux_first * incident_flux_first / incident_flux.powi(2)
            + 2.0 * transmitted_flux * incident_flux_first.powi(2) / incident_flux.powi(3);

        let transmission_norm = transmission.norm_sqr();

        let transmission_norm_first = 2.0 * (transmission.conj() * transmission_first).re;

        let transmission_norm_second =
            2.0 * (transmission_first.norm_sqr() + (transmission.conj() * transmission_second).re);

        let expected_transmittance_first =
            eta_first * transmission_norm + eta * transmission_norm_first;

        let expected_transmittance_second = eta_second * transmission_norm
            + 2.0 * eta_first * transmission_norm_first
            + eta * transmission_norm_second;

        assert_real_close(
            first.power().reflectance()[()],
            expected_reflectance_first,
            1e-12,
        );

        assert_real_close(
            first.power().transmittance()[()],
            expected_transmittance_first,
            1e-12,
        );

        assert_real_close(
            second.power().reflectance()[()],
            expected_reflectance_second,
            1e-12,
        );

        assert_real_close(
            second.power().transmittance()[()],
            expected_transmittance_second,
            1e-12,
        );

        assert_real_close(
            second.power().absorptance()[()],
            -expected_reflectance_second - expected_transmittance_second,
            1e-12,
        );
    }

    #[test]
    fn constant_port_normalisations_reduce_transmittance_derivatives_to_amplitude_terms() {
        let transmission = c(0.5, 0.25);
        let transmission_first = c(0.1, -0.2);
        let transmission_second = c(-0.04, 0.03);

        let incident = c(2.0, 0.0);
        let transmitted = c(4.0, 0.0);

        let response = PlaneWaveResponse::from_second_jets(
            ArrayJet::from_parts(arr0(c(0.2, 0.1)), arr0(c(0.0, 0.0)), arr0(c(0.0, 0.0))),
            ArrayJet::from_parts(
                arr0(transmission),
                arr0(transmission_first),
                arr0(transmission_second),
            ),
            ArrayJet::constant(arr0(incident)),
            ArrayJet::constant(arr0(transmitted)),
            DerivativeVariable::Thickness(0),
        );

        let eta = 2.0;

        let expected_first = eta * 2.0 * (transmission.conj() * transmission_first).re;

        let expected_second = eta
            * 2.0
            * (transmission_first.norm_sqr() + (transmission.conj() * transmission_second).re);

        let derivatives = response.derivatives().unwrap();

        assert_real_close(
            derivatives.first().power().transmittance()[()],
            expected_first,
            1e-12,
        );

        assert_real_close(
            derivatives.second().unwrap().power().transmittance()[()],
            expected_second,
            1e-12,
        );
    }

    #[test]
    fn equal_port_normalisations_make_transmittance_equal_transmission_norm() {
        let reflection = arr0(c(0.2, -0.1));
        let transmission = arr0(c(0.7, 0.3));

        let normalisation = arr0(c(2.5, 0.4));

        let response = PlaneWaveResponse::from_values(
            reflection,
            transmission.clone(),
            normalisation.clone(),
            normalisation,
        );

        assert_real_close(
            response.power().transmittance()[()],
            transmission[()].norm_sqr(),
            1e-12,
        );
    }

    #[test]
    fn response_preserves_sample_shape() {
        let reflection = array![c(0.1, 0.2), c(0.2, 0.3), c(0.3, 0.4),];

        let transmission = array![c(0.7, 0.1), c(0.6, 0.2), c(0.5, 0.3),];

        let incident = array![c(1.0, 0.0), c(1.5, 0.0), c(2.0, 0.0),];

        let transmitted = array![c(2.0, 0.0), c(2.5, 0.0), c(3.0, 0.0),];

        let response =
            PlaneWaveResponse::from_values(reflection.clone(), transmission, incident, transmitted);

        let expected = reflection.raw_dim();

        assert_eq!(response.reflection().raw_dim(), expected,);

        assert_eq!(response.transmission().raw_dim(), expected,);

        assert_eq!(response.power().reflectance().raw_dim(), expected,);

        assert_eq!(response.power().transmittance().raw_dim(), expected,);
    }

    #[test]
    fn first_order_response_preserves_sample_shape() {
        let source = array![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0),];

        let zero = source.mapv(|_| C::new(0.0, 0.0));

        let response = PlaneWaveResponse::from_first_jets(
            ArrayJetFirst::from_parts(source.clone(), zero.clone()),
            ArrayJetFirst::from_parts(source.clone(), zero.clone()),
            ArrayJetFirst::from_parts(source.clone(), zero.clone()),
            ArrayJetFirst::from_parts(source.clone(), zero),
            DerivativeVariable::VacuumWavenumber,
        );

        let expected = source.raw_dim();
        let first = response.derivatives().unwrap().first();

        assert_eq!(first.amplitudes().reflection().raw_dim(), expected,);

        assert_eq!(first.amplitudes().transmission().raw_dim(), expected,);

        assert_eq!(first.power().reflectance().raw_dim(), expected,);

        assert_eq!(first.power().transmittance().raw_dim(), expected,);
    }

    #[test]
    fn power_differential_absorptance_is_negative_sum() {
        let differential = PlaneWavePowerDifferential::new(arr0(0.2), arr0(-0.1));

        assert_real_close(differential.absorptance()[()], -0.2 - (-0.1), 1e-12);
    }

    #[test]
    fn planar_input_map_converts_real_arrays_to_complex() {
        let input = PlanarInput::new(
            array![1.0, 2.0, 3.0],
            array![0.1, 0.2, 0.3],
            Polarisation::TransverseElectric,
        );

        let complex: PlanarInput<ArrayBase<OwnedRepr<C>, ndarray::Ix1>> =
            input.map(|values| values.mapv(C::from_real));

        assert_eq!(
            complex.vacuum_wavenumber(),
            &array![c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0),],
        );

        assert_eq!(
            complex.parallel_wavenumber(),
            &array![c(0.1, 0.0), c(0.2, 0.0), c(0.3, 0.0),],
        );

        assert_eq!(complex.polarisation(), Polarisation::TransverseElectric,);
    }

    #[test]
    fn planar_input_map_preserves_scalar_shape() {
        let input = PlanarInput::new(arr0(2.0), arr0(0.5), Polarisation::TransverseMagnetic);

        let complex: PlanarInput<ArrayBase<OwnedRepr<C>, Ix0>> =
            input.map(|values| values.mapv(C::from_real));

        assert_eq!(complex.vacuum_wavenumber()[()], c(2.0, 0.0),);

        assert_eq!(complex.parallel_wavenumber()[()], c(0.5, 0.0),);

        assert_eq!(complex.polarisation(), Polarisation::TransverseMagnetic,);
    }
}
