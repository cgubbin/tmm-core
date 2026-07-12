use ndarray::{ArrayBase, Dimension, OwnedRepr};

use super::{DerivativeVariable, PlaneWaveInput};

/// Backend capable of solving a physical plane-wave scattering problem.
///
/// Implementations are responsible for translating their native matrix
/// representation into the canonical reflection and transmission amplitude
/// coefficients.
///
/// Consequently, callers of this trait do not need to know whether the backend
/// uses a transfer matrix, scattering matrix, or another internal
/// representation.
pub trait PlaneWaveBackend<C, D, S>
where
    D: Dimension,
{
    /// Error produced during the plane-wave calculation.
    type Error;

    /// Solve for reflection and transmission amplitudes.
    fn solve_plane_wave(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error>;

    /// Solve for amplitudes and their first derivatives.
    fn solve_plane_wave_first_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error>;

    /// Solve for amplitudes and their first and second derivatives.
    fn solve_plane_wave_second_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C>, D>>,
        variable: DerivativeVariable,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error>;
}

/// Backend-neutral plane-wave scattering response.
///
/// This response contains complex reflection and transmission amplitude
/// coefficients, together with optional derivatives.
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
    /// Construct a value-only plane-wave response.
    pub fn new(amplitudes: PlaneWaveAmplitudes<C, D>) -> Self {
        Self {
            amplitudes,
            derivatives: None,
        }
    }

    /// Construct a plane-wave response containing derivatives.
    pub fn with_derivatives(
        amplitudes: PlaneWaveAmplitudes<C, D>,
        derivatives: PlaneWaveResponseDerivatives<C, D>,
    ) -> Self {
        Self {
            amplitudes,
            derivatives: Some(derivatives),
        }
    }

    /// Return the reflection and transmission amplitude coefficients.
    pub fn amplitudes(&self) -> &PlaneWaveAmplitudes<C, D> {
        &self.amplitudes
    }

    /// Return derivatives of the amplitude coefficients.
    pub fn derivatives(&self) -> Option<&PlaneWaveResponseDerivatives<C, D>> {
        self.derivatives.as_ref()
    }

    /// Return the complex reflection amplitude coefficient.
    pub fn reflection(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        self.amplitudes.reflection()
    }

    /// Return the complex transmission amplitude coefficient.
    pub fn transmission(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        self.amplitudes.transmission()
    }
}

/// Reflection and transmission amplitude coefficients.
///
/// These are complex field-amplitude coefficients for a unit-amplitude
/// incident wave:
///
/// ```text
/// reflected field   = r × incident field
/// transmitted field = t × incident field
/// ```
///
/// Power reflectance and transmittance are not stored here. In particular,
/// transmittance generally requires exterior-medium flux normalisation.
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

    /// Consume the pair and return its components.
    pub fn into_parts(self) -> (ArrayBase<OwnedRepr<C>, D>, ArrayBase<OwnedRepr<C>, D>) {
        (self.reflection, self.transmission)
    }
}

/// Derivatives of plane-wave amplitude coefficients.
///
/// The first derivative is always present. The second derivative is available
/// only after a second-derivative solve.
///
/// These are derivatives of complex amplitudes, not derivatives of reflected
/// or transmitted power.
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

    /// Attach second-order amplitude derivatives.
    pub fn with_second(mut self, second: PlaneWaveAmplitudes<C, D>) -> Self {
        self.second = Some(second);
        self
    }

    /// Return the independent derivative variable.
    pub fn variable(&self) -> DerivativeVariable {
        self.variable
    }

    /// Return first derivatives of reflection and transmission amplitudes.
    pub fn first(&self) -> &PlaneWaveAmplitudes<C, D> {
        &self.first
    }

    /// Return second derivatives of reflection and transmission amplitudes.
    pub fn second(&self) -> Option<&PlaneWaveAmplitudes<C, D>> {
        self.second.as_ref()
    }
}
