//! Spatial field reconstruction and power-flow observables.
//!
//! This module reconstructs physical electromagnetic fields from the
//! boundary-wave solution returned by a plane-wave backend.
//!
//! Once a backend has computed the travelling-wave amplitudes at every
//! interface, this module provides backend-independent algorithms for
//!
//! - reconstructing canonical tangential fields;
//! - sampling fields at arbitrary positions;
//! - generating field profiles through multilayer stacks;
//! - computing signed normal power flux;
//! - evaluating per-layer absorptance;
//! - checking global energy conservation.
//!
//! # Workflow
//!
//! Typical usage is
//!
//! ```ignore
//! let solution = backend.solve_plane_wave_fields(
//!     &stack,
//!     &input,
//! )?;
//!
//! let sampling = FieldSampling::new()
//!     .layer_interfaces()
//!     .layer(
//!         0,
//!         LayerSampling::uniform(200),
//!     )
//!     .layer_centres();
//!
//! let fields = solution.sample_fields(
//!     &stack,
//!     &input,
//!     &sampling,
//! )?;
//!
//! let balance = solution.power_balance(
//!     &stack,
//!     &input,
//! )?;
//! ```
//!
//! # Coordinate system
//!
//! Distances are measured in centimetres.
//!
//! Global coordinates increase from left to right.
//!
//! Left-exterior coordinates are negative.
//!
//! Finite-layer positions are measured relative to each layer's left
//! boundary.
//!
//! Right-exterior coordinates are greater than the total stack thickness.
//!
//! # Canonical field state
//!
//! Each sampled field is represented by an [`IsotropicFieldState`].
//!
//! The stored quantities are
//!
//! ```text
//! primary = a⁺ + a⁻
//! dual = Y(a⁺ − a⁻)
//! ```
//!
//! where `Y` is the characteristic admittance.
//!
//! This representation is independent of TE/TM convention and gives the
//! normal power flux
//!
//! ```text
//! Pz = 1/2 Re(primary · dual*).
//! ```
//!
//! # Sampling
//!
//! High-level spatial sampling is described using [`FieldSampling`].
//!
//! A sampling specification is expanded into concrete [`FieldPosition`]s
//! before reconstruction.
//!
//! This separation allows sampling strategies to evolve independently of
//! the field-reconstruction algorithms.
mod error;
mod observables;
mod sampling;

pub use error::PlaneWaveFieldError;
pub use observables::{
    IsotropicFieldState, PlaneWaveFieldSample, PlaneWaveFields, PlaneWavePowerBalance,
    plane_wave_power_balance, sample_plane_wave_field_profile, sample_plane_wave_fields,
};
pub use sampling::{
    ExteriorSampling, FieldPosition, FieldSampling, FieldSamplingRegion, LayerSampling,
};

use crate::{
    ComplexScalar, DerivativeVariable, IncidentSide, PlaneWaveBackend, PlaneWaveInput,
    PlaneWaveResponse, PlaneWaveResponseDerivatives, Stack,
    backend::{
        PlaneWaveAmplitudes,
        derivative::{SpectralDerivativeVariable, StructuralDerivativeVariable},
        jet::{ArrayJet, ArrayJetFirst},
        plane_wave::PlaneWavePower,
    },
    material::EvaluateMaterial,
};

use ndarray::{ArrayBase, Dimension, OwnedRepr};
use num_traits::Float;

/// Backend capable of reconstructing internal waves for a physical plane-wave
/// scattering problem.
///
/// Implementations return the ordinary external plane-wave response together
/// with forward- and backward-propagating amplitudes immediately inside both
/// boundaries of every finite layer.
///
/// Geometric directions are fixed:
///
/// - `forward` means left to right;
/// - `backward` means right to left.
///
/// These meanings do not change with the incident side.
///
/// Input coordinates are real-valued. Internal amplitudes and their
/// derivatives are complex. Derivatives are taken along the requested real
/// coordinate.
///
/// The returned boundary amplitudes are modal wave amplitudes, not yet sampled
/// electric or magnetic fields. Physical field reconstruction additionally
/// requires the material, polarisation, normal wavenumber, and position within
/// a layer.
pub trait PlaneWaveFieldBackend<C, D, S>: PlaneWaveBackend<C, D, S>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Error produced during the plane-wave calculation.
    fn solve_plane_wave_internal_fields(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error>;

    fn solve_plane_wave_internal_fields_structural_first_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error>;

    fn solve_plane_wave_internal_fields_structural_second_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error>;
}

pub trait DifferentiablePlaneWaveFieldBackend<C, D, S>: PlaneWaveBackend<C, D, S>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn solve_plane_wave_internal_fields_spectral_first_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error>;

    fn solve_plane_wave_internal_fields_spectral_second_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error>;
}

/// Internal-field data requested from a backend solve
///
/// The derivative order is represented by the workspace entry type:
///
/// - `SampleArray<C, D>` for values;
/// - `ArrayJetFirst<C, D>` for first derivatives;
/// - `ArrayJet<C, D>` for second derivatives.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) enum InternalFieldRequest {
    /// Compute only the external scattering response.
    None,

    /// Retain sufficient component data to reconstruct waves at finite-layer
    /// boundaries.
    LayerBoundaries,
}

impl InternalFieldRequest {
    pub(crate) const fn is_requested(self) -> bool {
        matches!(self, Self::LayerBoundaries)
    }
}

/// Physical response together with internal finite-layer waves
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveFieldResponse<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    response: PlaneWaveResponse<C, D>,
    boundary_waves: PlaneWaveBoundaryWaves<C, D>,
}

impl<C, D> PlaneWaveFieldResponse<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn new(
        response: PlaneWaveResponse<C, D>,
        boundary_waves: PlaneWaveBoundaryWaves<C, D>,
    ) -> Self {
        Self {
            response,
            boundary_waves,
        }
    }

    pub fn response(&self) -> &PlaneWaveResponse<C, D> {
        &self.response
    }

    pub fn boundary_waves(&self) -> &PlaneWaveBoundaryWaves<C, D> {
        &self.boundary_waves
    }

    pub fn into_parts(self) -> (PlaneWaveResponse<C, D>, PlaneWaveBoundaryWaves<C, D>) {
        (self.response, self.boundary_waves)
    }

    pub fn amplitudes(&self) -> &PlaneWaveAmplitudes<C, D> {
        self.response.amplitudes()
    }

    pub fn power(&self) -> &PlaneWavePower<C::RealField, D> {
        self.response.power()
    }

    pub fn derivatives(&self) -> Option<&PlaneWaveResponseDerivatives<C, D>> {
        self.response.derivatives()
    }
}

impl<C, D> PlaneWaveFieldResponse<C, D>
where
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
{
    pub fn sample_fields<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        sampling: &FieldSampling<C::RealField>,
    ) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateMaterial<C, Real = C::RealField>,
    {
        sample_plane_wave_field_profile(stack, input, self.boundary_waves(), sampling)
    }

    pub fn sample_field_positions<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        positions: impl IntoIterator<Item = FieldPosition<C::RealField>>,
    ) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateMaterial<C, Real = C::RealField>,
    {
        sample_plane_wave_fields(stack, input, self.boundary_waves(), positions)
    }

    pub fn power_balance<M>(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWavePowerBalance<C::RealField, D>, PlaneWaveFieldError<C::RealField>>
    where
        M: EvaluateMaterial<C, Real = C::RealField>,
    {
        plane_wave_power_balance(stack, input, self)
    }
}

/// Internal boundary-wave amplitudes for every finite layer.
///
/// Layers follow fixed geometric left-to-right order. Exterior media are not
/// included.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveBoundaryWaves<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    exterior: ExteriorBoundaryWaves<C, D>,
    layers: Vec<LayerBoundaryWaves<C, D>>,
    derivatives: Option<PlaneWaveBoundaryWaveDerivatives<C, D>>,
}

impl<C, D> PlaneWaveBoundaryWaves<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn new(
        exterior: ExteriorBoundaryWaves<C, D>,
        layers: Vec<LayerBoundaryWaves<C, D>>,
    ) -> Self {
        Self {
            exterior,
            layers,
            derivatives: None,
        }
    }

    pub(crate) fn with_derivatives(
        exterior: ExteriorBoundaryWaves<C, D>,
        layers: Vec<LayerBoundaryWaves<C, D>>,
        derivatives: PlaneWaveBoundaryWaveDerivatives<C, D>,
    ) -> Self {
        debug_assert_eq!(layers.len(), derivatives.first.len(),);

        if let Some(second_layers) = &derivatives.second {
            debug_assert_eq!(layers.len(), second_layers.len(),);
        }

        Self {
            exterior,
            layers,
            derivatives: Some(derivatives),
        }
    }

    pub fn exterior(&self) -> &ExteriorBoundaryWaves<C, D> {
        &self.exterior
    }

    pub fn layers(&self) -> &[LayerBoundaryWaves<C, D>] {
        &self.layers
    }

    pub fn layer(&self, index: usize) -> Option<&LayerBoundaryWaves<C, D>> {
        self.layers.get(index)
    }

    pub fn len(&self) -> usize {
        self.layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub fn derivatives(&self) -> Option<&PlaneWaveBoundaryWaveDerivatives<C, D>> {
        self.derivatives.as_ref()
    }

    /// Consume the result.
    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        Vec<LayerBoundaryWaves<C, D>>,
        Option<PlaneWaveBoundaryWaveDerivatives<C, D>>,
    ) {
        (self.layers, self.derivatives)
    }
}

/// First and optional second derivatives of all finite-layer boundary waves.
///
/// `first[j]` and `second[j]` correspond to `fields.layers()[j]`.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveBoundaryWaveDerivatives<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    variable: DerivativeVariable,
    exterior_first: ExteriorBoundaryWaveDifferential<C, D>,
    first: Vec<LayerBoundaryWaveDifferential<C, D>>,
    exterior_second: Option<ExteriorBoundaryWaveDifferential<C, D>>,
    second: Option<Vec<LayerBoundaryWaveDifferential<C, D>>>,
}

impl<C, D> PlaneWaveBoundaryWaveDerivatives<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn new(
        variable: DerivativeVariable,
        exterior_first: ExteriorBoundaryWaveDifferential<C, D>,
        first: Vec<LayerBoundaryWaveDifferential<C, D>>,
    ) -> Self {
        Self {
            variable,
            exterior_first,
            first,
            exterior_second: None,
            second: None,
        }
    }

    pub(crate) fn with_second(
        mut self,
        exterior_second: ExteriorBoundaryWaveDifferential<C, D>,
        second: Vec<LayerBoundaryWaveDifferential<C, D>>,
    ) -> Self {
        self.exterior_second = Some(exterior_second);
        self.second = Some(second);
        self
    }

    /// Return the independent variable.
    pub(crate) fn variable(&self) -> DerivativeVariable {
        self.variable
    }

    /// Return first derivatives for every finite layer.
    pub(crate) fn first_layers(&self) -> &[LayerBoundaryWaveDifferential<C, D>] {
        &self.first
    }

    /// Return first derivatives for the exterior
    pub(crate) fn exterior_first(&self) -> &ExteriorBoundaryWaveDifferential<C, D> {
        &self.exterior_first
    }

    /// Return second derivatives for the exterior
    pub(crate) fn exterior_second(&self) -> Option<&ExteriorBoundaryWaveDifferential<C, D>> {
        self.exterior_second.as_ref()
    }

    /// Return second derivatives for every finite layer, when available.
    pub(crate) fn second_layers(&self) -> Option<&[LayerBoundaryWaveDifferential<C, D>]> {
        self.second.as_deref()
    }

    /// Return the first derivative for one finite layer.
    pub(crate) fn first_layer(&self, index: usize) -> Option<&LayerBoundaryWaveDifferential<C, D>> {
        self.first.get(index)
    }

    /// Return the second derivative for one finite layer.
    pub(crate) fn second_layer(
        &self,
        index: usize,
    ) -> Option<&LayerBoundaryWaveDifferential<C, D>> {
        self.second.as_ref()?.get(index)
    }

    /// Consume the derivative result.
    #[allow(clippy::type_complexity)]
    pub(crate) fn into_parts(
        self,
    ) -> (
        DerivativeVariable,
        Vec<LayerBoundaryWaveDifferential<C, D>>,
        Option<Vec<LayerBoundaryWaveDifferential<C, D>>>,
    ) {
        (self.variable, self.first, self.second)
    }
}

/// Forward- and backward-propagating wave amplitudes at one reference plane.
///
/// Directions are geometric:
///
/// - `forward` propagates from left to right;
/// - `backward` propagates from right to left.
///
/// The meaning does not change with [`crate::backend::IncidentSide`].
#[derive(Clone, Debug, PartialEq)]
pub struct BidirectionalWaves<C, D>
where
    D: Dimension,
{
    forward: ArrayBase<OwnedRepr<C>, D>,
    backward: ArrayBase<OwnedRepr<C>, D>,
}

impl<C, D> BidirectionalWaves<C, D>
where
    D: Dimension,
{
    /// Construct forward and backward wave amplitudes.
    pub(crate) fn new(
        forward: ArrayBase<OwnedRepr<C>, D>,
        backward: ArrayBase<OwnedRepr<C>, D>,
    ) -> Self {
        debug_assert_eq!(forward.raw_dim(), backward.raw_dim(),);

        Self { forward, backward }
    }

    /// Return the left-to-right wave amplitude.
    pub(crate) fn forward(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.forward
    }

    /// Return the right-to-left wave amplitude.
    pub(crate) fn backward(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.backward
    }

    /// Consume the pair and return its amplitudes.
    pub(crate) fn into_parts(self) -> (ArrayBase<OwnedRepr<C>, D>, ArrayBase<OwnedRepr<C>, D>) {
        (self.forward, self.backward)
    }
}

/// Forward and backward wave amplitudes at both boundaries of one finite layer.
///
/// The `left` and `right` fields refer to geometric layer boundaries:
///
/// ```text
/// left boundary | finite layer | right boundary
/// ```
///
/// Both boundary values are retained explicitly. This avoids reconstructing one
/// boundary by dividing by a potentially very small evanescent propagation
/// factor.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerBoundaryWaves<C, D>
where
    D: Dimension,
{
    left: BidirectionalWaves<C, D>,
    right: BidirectionalWaves<C, D>,
}

impl<C, D> LayerBoundaryWaves<C, D>
where
    D: Dimension,
{
    /// Construct layer-boundary wave amplitudes.
    pub(crate) fn new(left: BidirectionalWaves<C, D>, right: BidirectionalWaves<C, D>) -> Self {
        debug_assert_eq!(left.forward().raw_dim(), right.forward().raw_dim(),);
        debug_assert_eq!(left.backward().raw_dim(), right.backward().raw_dim(),);

        Self { left, right }
    }

    /// Return waves immediately inside the layer at its left boundary.
    pub(crate) fn left(&self) -> &BidirectionalWaves<C, D> {
        &self.left
    }

    /// Return waves immediately inside the layer at its right boundary.
    pub(crate) fn right(&self) -> &BidirectionalWaves<C, D> {
        &self.right
    }

    /// Consume the layer response and return both boundary wave pairs.
    pub(crate) fn into_parts(self) -> (BidirectionalWaves<C, D>, BidirectionalWaves<C, D>) {
        (self.left, self.right)
    }
}

/// Derivatives of forward and backward wave amplitudes at one reference plane.
///
/// The derivative variable and derivative order are recorded by the containing
/// [`PlaneWaveBoundaryWaveDerivatives`] object.
#[derive(Clone, Debug, PartialEq)]
pub struct BidirectionalWaveDifferential<C, D>
where
    D: Dimension,
{
    forward: ArrayBase<OwnedRepr<C>, D>,
    backward: ArrayBase<OwnedRepr<C>, D>,
}

impl<C, D> BidirectionalWaveDifferential<C, D>
where
    D: Dimension,
{
    /// Construct derivatives of forward and backward amplitudes.
    pub(crate) fn new(
        forward: ArrayBase<OwnedRepr<C>, D>,
        backward: ArrayBase<OwnedRepr<C>, D>,
    ) -> Self {
        debug_assert_eq!(forward.raw_dim(), backward.raw_dim(),);

        Self { forward, backward }
    }

    /// Return the derivative of the forward wave.
    pub(crate) fn forward(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.forward
    }

    /// Return the derivative of the backward wave.
    pub(crate) fn backward(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.backward
    }

    /// Consume the differential and return both arrays.
    pub(crate) fn into_parts(self) -> (ArrayBase<OwnedRepr<C>, D>, ArrayBase<OwnedRepr<C>, D>) {
        (self.forward, self.backward)
    }
}

/// Derivatives of internal wave amplitudes at both boundaries of one layer.
///
/// This type is used for either first or second derivatives. The containing
/// derivative result identifies which order it represents.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerBoundaryWaveDifferential<C, D>
where
    D: Dimension,
{
    left: BidirectionalWaveDifferential<C, D>,
    right: BidirectionalWaveDifferential<C, D>,
}

impl<C, D> LayerBoundaryWaveDifferential<C, D>
where
    D: Dimension,
{
    /// Construct derivatives at both layer boundaries.
    pub(crate) fn new(
        left: BidirectionalWaveDifferential<C, D>,
        right: BidirectionalWaveDifferential<C, D>,
    ) -> Self {
        debug_assert_eq!(left.forward().raw_dim(), right.forward().raw_dim(),);
        debug_assert_eq!(left.backward().raw_dim(), right.backward().raw_dim(),);

        Self { left, right }
    }

    /// Return derivatives at the left boundary.
    pub(crate) fn left(&self) -> &BidirectionalWaveDifferential<C, D> {
        &self.left
    }

    /// Return derivatives at the right boundary.
    pub(crate) fn right(&self) -> &BidirectionalWaveDifferential<C, D> {
        &self.right
    }

    /// Consume the result and return both boundary differentials.
    pub(crate) fn into_parts(
        self,
    ) -> (
        BidirectionalWaveDifferential<C, D>,
        BidirectionalWaveDifferential<C, D>,
    ) {
        (self.left, self.right)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BidirectionalWavesGeneric<A> {
    pub(crate) forward: A,
    pub(crate) backward: A,
}

impl<A> BidirectionalWavesGeneric<A> {
    pub(crate) fn new(forward: A, backward: A) -> Self {
        Self { forward, backward }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LayerBoundaryWavesGeneric<A> {
    pub(crate) left: BidirectionalWavesGeneric<A>,
    pub(crate) right: BidirectionalWavesGeneric<A>,
}

impl<A> LayerBoundaryWavesGeneric<A> {
    pub(crate) fn new(
        left: BidirectionalWavesGeneric<A>,
        right: BidirectionalWavesGeneric<A>,
    ) -> Self {
        Self { left, right }
    }
}

pub(crate) fn value_fields_from_generic<C, D>(
    layers: Vec<LayerBoundaryWavesGeneric<ArrayBase<OwnedRepr<C>, D>>>,
) -> Vec<LayerBoundaryWaves<C, D>>
where
    C: ComplexScalar,
    D: Dimension,
{
    layers
        .into_iter()
        .map(|layer| {
            layer_waves(
                layer.left.forward,
                layer.left.backward,
                layer.right.forward,
                layer.right.backward,
            )
        })
        .collect()
}

#[allow(clippy::type_complexity)]
pub(crate) fn first_order_fields_from_generic<C, D>(
    layers: Vec<LayerBoundaryWavesGeneric<ArrayJetFirst<C, D>>>,
) -> (
    Vec<LayerBoundaryWaves<C, D>>,
    Vec<LayerBoundaryWaveDifferential<C, D>>,
)
where
    C: ComplexScalar,
    D: Dimension,
{
    let mut values = Vec::with_capacity(layers.len());

    let mut first = Vec::with_capacity(layers.len());

    for layer in layers {
        let (left_forward, left_forward_first) = layer.left.forward.into_parts();

        let (left_backward, left_backward_first) = layer.left.backward.into_parts();

        let (right_forward, right_forward_first) = layer.right.forward.into_parts();

        let (right_backward, right_backward_first) = layer.right.backward.into_parts();

        values.push(layer_waves(
            left_forward,
            left_backward,
            right_forward,
            right_backward,
        ));

        first.push(layer_differential(
            left_forward_first,
            left_backward_first,
            right_forward_first,
            right_backward_first,
        ));
    }

    (values, first)
}

#[allow(clippy::type_complexity)]
pub(crate) fn second_order_fields_from_generic<C, D>(
    layers: Vec<LayerBoundaryWavesGeneric<ArrayJet<C, D>>>,
) -> (
    Vec<LayerBoundaryWaves<C, D>>,
    Vec<LayerBoundaryWaveDifferential<C, D>>,
    Vec<LayerBoundaryWaveDifferential<C, D>>,
)
where
    C: ComplexScalar,
    D: Dimension,
{
    let mut values = Vec::with_capacity(layers.len());

    let mut first = Vec::with_capacity(layers.len());

    let mut second = Vec::with_capacity(layers.len());

    for layer in layers {
        let (left_forward, left_forward_first, left_forward_second) =
            layer.left.forward.into_parts();

        let (left_backward, left_backward_first, left_backward_second) =
            layer.left.backward.into_parts();

        let (right_forward, right_forward_first, right_forward_second) =
            layer.right.forward.into_parts();

        let (right_backward, right_backward_first, right_backward_second) =
            layer.right.backward.into_parts();

        values.push(layer_waves(
            left_forward,
            left_backward,
            right_forward,
            right_backward,
        ));

        first.push(layer_differential(
            left_forward_first,
            left_backward_first,
            right_forward_first,
            right_backward_first,
        ));

        second.push(layer_differential(
            left_forward_second,
            left_backward_second,
            right_forward_second,
            right_backward_second,
        ));
    }

    (values, first, second)
}

fn layer_waves<C, D>(
    left_forward: ArrayBase<OwnedRepr<C>, D>,
    left_backward: ArrayBase<OwnedRepr<C>, D>,
    right_forward: ArrayBase<OwnedRepr<C>, D>,
    right_backward: ArrayBase<OwnedRepr<C>, D>,
) -> LayerBoundaryWaves<C, D>
where
    D: Dimension,
{
    LayerBoundaryWaves::new(
        BidirectionalWaves::new(left_forward, left_backward),
        BidirectionalWaves::new(right_forward, right_backward),
    )
}

fn layer_differential<C, D>(
    left_forward: ArrayBase<OwnedRepr<C>, D>,
    left_backward: ArrayBase<OwnedRepr<C>, D>,
    right_forward: ArrayBase<OwnedRepr<C>, D>,
    right_backward: ArrayBase<OwnedRepr<C>, D>,
) -> LayerBoundaryWaveDifferential<C, D>
where
    D: Dimension,
{
    LayerBoundaryWaveDifferential::new(
        BidirectionalWaveDifferential::new(left_forward, left_backward),
        BidirectionalWaveDifferential::new(right_forward, right_backward),
    )
}

/// Modal amplitudes in the two semi-infinite exterior media.
///
/// Directions are geometric:
///
/// - `forward` propagates left to right;
/// - `backward` propagates right to left.
#[derive(Clone, Debug, PartialEq)]
pub struct ExteriorBoundaryWaves<C, D>
where
    D: Dimension,
{
    left: BidirectionalWaves<C, D>,
    right: BidirectionalWaves<C, D>,
}

impl<C, D> ExteriorBoundaryWaves<C, D>
where
    D: Dimension,
{
    pub(crate) fn new(left: BidirectionalWaves<C, D>, right: BidirectionalWaves<C, D>) -> Self {
        Self { left, right }
    }

    pub(crate) fn from_values(
        reflection: ArrayBase<OwnedRepr<C>, D>,
        transmission: ArrayBase<OwnedRepr<C>, D>,
        incident_side: IncidentSide,
    ) -> ExteriorBoundaryWaves<C, D>
    where
        C: ComplexScalar,
    {
        let one = reflection.mapv(|_| C::one());
        let zero = reflection.mapv(|_| C::zero());

        match incident_side {
            IncidentSide::Left => ExteriorBoundaryWaves::new(
                BidirectionalWaves::new(one, reflection),
                BidirectionalWaves::new(transmission, zero),
            ),

            IncidentSide::Right => ExteriorBoundaryWaves::new(
                BidirectionalWaves::new(zero, transmission),
                BidirectionalWaves::new(reflection, one),
            ),
        }
    }

    pub fn left(&self) -> &BidirectionalWaves<C, D> {
        &self.left
    }

    pub fn right(&self) -> &BidirectionalWaves<C, D> {
        &self.right
    }

    pub fn into_parts(self) -> (BidirectionalWaves<C, D>, BidirectionalWaves<C, D>) {
        (self.left, self.right)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ExteriorBoundaryWaveDifferential<C, D>
where
    D: Dimension,
{
    left: BidirectionalWaveDifferential<C, D>,
    right: BidirectionalWaveDifferential<C, D>,
}

impl<C, D> ExteriorBoundaryWaveDifferential<C, D>
where
    D: Dimension,
{
    pub(crate) fn new(
        left: BidirectionalWaveDifferential<C, D>,
        right: BidirectionalWaveDifferential<C, D>,
    ) -> Self {
        Self { left, right }
    }

    pub fn left(&self) -> &BidirectionalWaveDifferential<C, D> {
        &self.left
    }

    pub fn right(&self) -> &BidirectionalWaveDifferential<C, D> {
        &self.right
    }

    pub fn into_parts(
        self,
    ) -> (
        BidirectionalWaveDifferential<C, D>,
        BidirectionalWaveDifferential<C, D>,
    ) {
        (self.left, self.right)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{ArrayBase, OwnedRepr, arr1};
    use num_complex::Complex64;

    use super::*;

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn exterior_values(
        left_forward: f64,
        left_backward: f64,
        right_forward: f64,
        right_backward: f64,
    ) -> ExteriorBoundaryWaves<C, ndarray::Ix1> {
        ExteriorBoundaryWaves::new(
            BidirectionalWaves::new(arr1(&[c(left_forward)]), arr1(&[c(left_backward)])),
            BidirectionalWaves::new(arr1(&[c(right_forward)]), arr1(&[c(right_backward)])),
        )
    }

    fn exterior_differential(
        left_forward: f64,
        left_backward: f64,
        right_forward: f64,
        right_backward: f64,
    ) -> ExteriorBoundaryWaveDifferential<C, ndarray::Ix1> {
        ExteriorBoundaryWaveDifferential::new(
            BidirectionalWaveDifferential::new(arr1(&[c(left_forward)]), arr1(&[c(left_backward)])),
            BidirectionalWaveDifferential::new(
                arr1(&[c(right_forward)]),
                arr1(&[c(right_backward)]),
            ),
        )
    }

    #[test]
    fn value_conversion_preserves_layer_boundary_waves() {
        let generic = vec![LayerBoundaryWavesGeneric::new(
            BidirectionalWavesGeneric::new(arr1(&[c(1.0), c(2.0)]), arr1(&[c(3.0), c(4.0)])),
            BidirectionalWavesGeneric::new(arr1(&[c(5.0), c(6.0)]), arr1(&[c(7.0), c(8.0)])),
        )];

        let layers = value_fields_from_generic(generic);

        assert_eq!(layers.len(), 1);

        let layer = &layers[0];

        assert_eq!(layer.left().forward(), &arr1(&[c(1.0), c(2.0)]),);

        assert_eq!(layer.left().backward(), &arr1(&[c(3.0), c(4.0)]),);

        assert_eq!(layer.right().forward(), &arr1(&[c(5.0), c(6.0)]),);

        assert_eq!(layer.right().backward(), &arr1(&[c(7.0), c(8.0)]),);
    }

    #[test]
    fn value_layers_can_be_assembled_with_exterior_waves() {
        let generic = vec![LayerBoundaryWavesGeneric::new(
            BidirectionalWavesGeneric::new(arr1(&[c(1.0)]), arr1(&[c(2.0)])),
            BidirectionalWavesGeneric::new(arr1(&[c(3.0)]), arr1(&[c(4.0)])),
        )];

        let layers = value_fields_from_generic(generic);

        let exterior = exterior_values(1.0, 0.25, 0.75, 0.0);

        let fields = PlaneWaveBoundaryWaves::new(exterior, layers);

        assert_eq!(fields.len(), 1);
        assert!(!fields.is_empty());
        assert!(fields.derivatives().is_none());

        assert_eq!(fields.exterior().left().forward(), &arr1(&[c(1.0)]),);

        assert_eq!(fields.exterior().left().backward(), &arr1(&[c(0.25)]),);

        assert_eq!(fields.exterior().right().forward(), &arr1(&[c(0.75)]),);

        assert_eq!(fields.exterior().right().backward(), &arr1(&[c(0.0)]),);

        let layer = fields.layer(0).unwrap();

        assert_eq!(layer.left().forward(), &arr1(&[c(1.0)]),);

        assert_eq!(layer.right().backward(), &arr1(&[c(4.0)]),);
    }

    #[test]
    fn first_order_conversion_separates_values_and_first_derivatives() {
        let generic = vec![LayerBoundaryWavesGeneric::new(
            BidirectionalWavesGeneric::new(
                ArrayJetFirst::from_parts(arr1(&[c(1.0)]), arr1(&[c(11.0)])),
                ArrayJetFirst::from_parts(arr1(&[c(2.0)]), arr1(&[c(12.0)])),
            ),
            BidirectionalWavesGeneric::new(
                ArrayJetFirst::from_parts(arr1(&[c(3.0)]), arr1(&[c(13.0)])),
                ArrayJetFirst::from_parts(arr1(&[c(4.0)]), arr1(&[c(14.0)])),
            ),
        )];

        let (layers, first) = first_order_fields_from_generic(generic);

        assert_eq!(layers.len(), 1);
        assert_eq!(first.len(), 1);

        let value = &layers[0];

        assert_eq!(value.left().forward(), &arr1(&[c(1.0)]),);

        assert_eq!(value.left().backward(), &arr1(&[c(2.0)]),);

        assert_eq!(value.right().forward(), &arr1(&[c(3.0)]),);

        assert_eq!(value.right().backward(), &arr1(&[c(4.0)]),);

        let derivative = &first[0];

        assert_eq!(derivative.left().forward(), &arr1(&[c(11.0)]),);

        assert_eq!(derivative.left().backward(), &arr1(&[c(12.0)]),);

        assert_eq!(derivative.right().forward(), &arr1(&[c(13.0)]),);

        assert_eq!(derivative.right().backward(), &arr1(&[c(14.0)]),);
    }

    #[test]
    fn first_order_layers_can_be_assembled_with_exterior_derivatives() {
        let generic = vec![LayerBoundaryWavesGeneric::new(
            BidirectionalWavesGeneric::new(
                ArrayJetFirst::from_parts(arr1(&[c(1.0)]), arr1(&[c(11.0)])),
                ArrayJetFirst::from_parts(arr1(&[c(2.0)]), arr1(&[c(12.0)])),
            ),
            BidirectionalWavesGeneric::new(
                ArrayJetFirst::from_parts(arr1(&[c(3.0)]), arr1(&[c(13.0)])),
                ArrayJetFirst::from_parts(arr1(&[c(4.0)]), arr1(&[c(14.0)])),
            ),
        )];

        let (layers, first_layers) = first_order_fields_from_generic(generic);

        let exterior = exterior_values(1.0, 0.25, 0.75, 0.0);

        let exterior_first = exterior_differential(0.0, 10.0, 20.0, 0.0);

        let derivatives = PlaneWaveBoundaryWaveDerivatives::new(
            DerivativeVariable::VacuumWavenumber,
            exterior_first,
            first_layers,
        );

        let fields = PlaneWaveBoundaryWaves::with_derivatives(exterior, layers, derivatives);

        let derivatives = fields.derivatives().unwrap();

        assert_eq!(derivatives.variable(), DerivativeVariable::VacuumWavenumber,);

        assert!(derivatives.second_layers().is_none());
        assert!(derivatives.exterior_second().is_none());

        assert_eq!(
            derivatives.exterior_first().left().forward(),
            &arr1(&[c(0.0)]),
        );

        assert_eq!(
            derivatives.exterior_first().left().backward(),
            &arr1(&[c(10.0)]),
        );

        assert_eq!(
            derivatives.exterior_first().right().forward(),
            &arr1(&[c(20.0)]),
        );

        assert_eq!(
            derivatives.first_layer(0).unwrap().left().forward(),
            &arr1(&[c(11.0)]),
        );

        assert_eq!(
            derivatives.first_layer(0).unwrap().right().backward(),
            &arr1(&[c(14.0)]),
        );
    }

    #[test]
    fn second_order_conversion_separates_all_orders() {
        fn jet(value: f64, first: f64, second: f64) -> ArrayJet<C, ndarray::Ix1> {
            ArrayJet::from_parts(arr1(&[c(value)]), arr1(&[c(first)]), arr1(&[c(second)]))
        }

        let generic = vec![LayerBoundaryWavesGeneric::new(
            BidirectionalWavesGeneric::new(jet(1.0, 11.0, 21.0), jet(2.0, 12.0, 22.0)),
            BidirectionalWavesGeneric::new(jet(3.0, 13.0, 23.0), jet(4.0, 14.0, 24.0)),
        )];

        let (layers, first, second) = second_order_fields_from_generic(generic);

        assert_eq!(layers.len(), 1);
        assert_eq!(first.len(), 1);
        assert_eq!(second.len(), 1);

        let value = &layers[0];

        assert_eq!(value.left().forward(), &arr1(&[c(1.0)]),);

        assert_eq!(value.right().backward(), &arr1(&[c(4.0)]),);

        let first = &first[0];

        assert_eq!(first.left().forward(), &arr1(&[c(11.0)]),);

        assert_eq!(first.left().backward(), &arr1(&[c(12.0)]),);

        assert_eq!(first.right().forward(), &arr1(&[c(13.0)]),);

        assert_eq!(first.right().backward(), &arr1(&[c(14.0)]),);

        let second = &second[0];

        assert_eq!(second.left().forward(), &arr1(&[c(21.0)]),);

        assert_eq!(second.left().backward(), &arr1(&[c(22.0)]),);

        assert_eq!(second.right().forward(), &arr1(&[c(23.0)]),);

        assert_eq!(second.right().backward(), &arr1(&[c(24.0)]),);
    }

    #[test]
    fn second_order_layers_can_be_assembled_with_all_exterior_derivatives() {
        fn jet(value: f64, first: f64, second: f64) -> ArrayJet<C, ndarray::Ix1> {
            ArrayJet::from_parts(arr1(&[c(value)]), arr1(&[c(first)]), arr1(&[c(second)]))
        }

        let generic = vec![LayerBoundaryWavesGeneric::new(
            BidirectionalWavesGeneric::new(jet(1.0, 11.0, 21.0), jet(2.0, 12.0, 22.0)),
            BidirectionalWavesGeneric::new(jet(3.0, 13.0, 23.0), jet(4.0, 14.0, 24.0)),
        )];

        let (layers, first_layers, second_layers) = second_order_fields_from_generic(generic);

        let exterior = exterior_values(1.0, 0.25, 0.75, 0.0);

        let exterior_first = exterior_differential(0.0, 10.0, 20.0, 0.0);

        let exterior_second = exterior_differential(0.0, 30.0, 40.0, 0.0);

        let derivatives = PlaneWaveBoundaryWaveDerivatives::new(
            DerivativeVariable::Thickness(0),
            exterior_first,
            first_layers,
        )
        .with_second(exterior_second, second_layers);

        let fields = PlaneWaveBoundaryWaves::with_derivatives(exterior, layers, derivatives);

        let derivatives = fields.derivatives().unwrap();

        assert_eq!(derivatives.variable(), DerivativeVariable::Thickness(0),);

        assert_eq!(
            derivatives.exterior_second().unwrap().left().backward(),
            &arr1(&[c(30.0)]),
        );

        assert_eq!(
            derivatives.exterior_second().unwrap().right().forward(),
            &arr1(&[c(40.0)]),
        );

        assert_eq!(
            derivatives.second_layer(0).unwrap().left().forward(),
            &arr1(&[c(21.0)]),
        );

        assert_eq!(
            derivatives.second_layer(0).unwrap().right().backward(),
            &arr1(&[c(24.0)]),
        );
    }

    #[test]
    fn empty_generic_layers_can_be_assembled_with_exterior_waves() {
        let generic: Vec<LayerBoundaryWavesGeneric<ArrayBase<OwnedRepr<C>, ndarray::Ix1>>> =
            Vec::new();

        let layers = value_fields_from_generic(generic);

        assert!(layers.is_empty());

        let fields = PlaneWaveBoundaryWaves::new(exterior_values(1.0, 0.25, 0.75, 0.0), layers);

        assert!(fields.is_empty());
        assert_eq!(fields.len(), 0);
        assert!(fields.derivatives().is_none());

        assert_eq!(fields.exterior().left().forward(), &arr1(&[c(1.0)]),);
    }

    #[test]
    fn conversion_preserves_geometric_layer_order() {
        let generic = vec![
            LayerBoundaryWavesGeneric::new(
                BidirectionalWavesGeneric::new(arr1(&[c(1.0)]), arr1(&[c(0.0)])),
                BidirectionalWavesGeneric::new(arr1(&[c(2.0)]), arr1(&[c(0.0)])),
            ),
            LayerBoundaryWavesGeneric::new(
                BidirectionalWavesGeneric::new(arr1(&[c(3.0)]), arr1(&[c(0.0)])),
                BidirectionalWavesGeneric::new(arr1(&[c(4.0)]), arr1(&[c(0.0)])),
            ),
        ];

        let layers = value_fields_from_generic(generic);

        let fields = PlaneWaveBoundaryWaves::new(exterior_values(1.0, 0.0, 1.0, 0.0), layers);

        assert_eq!(fields.layer(0).unwrap().left().forward()[0], c(1.0),);

        assert_eq!(fields.layer(1).unwrap().left().forward()[0], c(3.0),);
    }
}
