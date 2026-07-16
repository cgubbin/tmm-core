//! Internal plane-wave amplitude conventions.
//!
//! This module represents the internal solution of a scalar isotropic planar
//! scattering problem in terms of forward- and backward-propagating modal
//! amplitudes.
//!
//! # Geometry and propagation direction
//!
//! The stack has a fixed geometric orientation:
//!
//! ```text
//! left exterior | layer 0 | layer 1 | ... | layer N - 1 | right exterior
//! ```
//!
//! Wave directions are defined geometrically and do not depend on the incident
//! side:
//!
//! - `forward` propagates from left to right;
//! - `backward` propagates from right to left.
//!
//! Consequently, for right incidence the imposed incident wave is a backward
//! wave in the right exterior.
//!
//! # Normalisation
//!
//! Internal amplitudes are normalised to a unit incident modal amplitude:
//!
//! ```text
//! left incidence:
//!     left-exterior forward amplitude  = 1
//!     right-exterior backward amplitude = 0
//!
//! right incidence:
//!     left-exterior forward amplitude   = 0
//!     right-exterior backward amplitude = 1
//! ```
//!
//! The reflected, transmitted, and internal amplitudes therefore use the same
//! modal-amplitude normalisation as [`PlaneWaveResponse`].
//!
//! This is an amplitude normalisation, not a unit-power normalisation. When the
//! exterior media differ, a unit transmitted amplitude does not generally
//! carry the same normal power flux as a unit incident amplitude.
//!
//! # Boundary reference planes
//!
//! Each finite layer stores amplitudes at two reference planes:
//!
//! ```text
//! left boundary |        finite layer        | right boundary
//! ```
//!
//! `LayerBoundaryWaves::left()` contains amplitudes evaluated immediately
//! inside the finite layer on the right-hand side of its left interface.
//!
//! `LayerBoundaryWaves::right()` contains amplitudes evaluated immediately
//! inside the same finite layer on the left-hand side of its right interface.
//!
//! Thus both wave pairs belong to the finite-layer material. They are not the
//! amplitudes in the adjacent exterior or neighbouring layer.
//!
//! # Phase reference
//!
//! At each stored boundary, the local modal amplitudes are referenced to that
//! boundary plane. No additional propagation phase is included in the position
//! of the reference plane.
//!
//! For a layer extending from `z = 0` at its left boundary to `z = d` at its
//! right boundary, the local waves may be written schematically as:
//!
//! ```text
//! forward(z)  = forward_at_left  * exp(s i κ z)
//! backward(z) = backward_at_right * exp(s i κ (d - z))
//! ```
//!
//! where `s` is the spatial-phase sign used by the backend convention.
//!
//! Both boundary values are stored explicitly so callers do not need to
//! recover one boundary by dividing by a potentially very small evanescent
//! propagation factor.
//!
//! # Layer indexing
//!
//! Layers are returned in fixed geometric left-to-right order. Layer index `j`
//! refers to the same finite layer as:
//!
//! ```text
//! DerivativeVariable::Thickness(j)
//! ```
//!
//! and does not change with the incident side.
//!
//! # Derivatives
//!
//! First and second derivatives are taken along the requested real coordinate.
//! The imposed unit incident amplitude is constant with respect to all
//! derivative variables.
//!
//! For example, a thickness derivative differentiates the internal amplitudes
//! while keeping the incident modal amplitude equal to one.
//!
//! The stored differential arrays are derivatives of modal amplitudes. They are
//! not derivatives of intensity, energy density, or power flux. Those real
//! observables must be formed from the complex amplitudes and their
//! derivatives.

use crate::{
    ComplexScalar, DerivativeVariable, IncidentSide, PlaneWaveBackend, PlaneWaveInput,
    PlaneWaveResponse,
    backend::jet::{ArrayJet, ArrayJetFirst},
};

use ndarray::{ArrayBase, Dimension, OwnedRepr};

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
}

pub trait DifferentiablePlaneWaveFieldBackend<C, D, S>: PlaneWaveBackend<C, D, S>
where
    C: ComplexScalar,
    D: Dimension,
{
    fn solve_plane_wave_internal_fields_first_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: DerivativeVariable,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error>;

    fn solve_plane_wave_internal_fields_second_derivative(
        &self,
        stack: &S,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: DerivativeVariable,
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

    pub(crate) fn response(&self) -> &PlaneWaveResponse<C, D> {
        &self.response
    }

    pub fn boundary_waves(&self) -> &PlaneWaveBoundaryWaves<C, D> {
        &self.boundary_waves
    }

    pub(crate) fn into_parts(self) -> (PlaneWaveResponse<C, D>, PlaneWaveBoundaryWaves<C, D>) {
        (self.response, self.boundary_waves)
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
    pub(crate) fn first(&self) -> &[LayerBoundaryWaveDifferential<C, D>] {
        &self.first
    }

    /// Return second derivatives for every finite layer, when available.
    pub(crate) fn second(&self) -> Option<&[LayerBoundaryWaveDifferential<C, D>]> {
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
}

pub enum FieldPosition<R> {
    /// Position measured from the left boundary of a finite layer.
    Layer { layer: usize, offset: R },

    /// Position in the left exterior, measured away from the stack.
    LeftExterior { distance: R },

    /// Position in the right exterior, measured away from the stack.
    RightExterior { distance: R },
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWavePowerBalance<R, D>
where
    D: Dimension,
{
    incident: ArrayBase<OwnedRepr<R>, D>,
    reflected: ArrayBase<OwnedRepr<R>, D>,
    transmitted: ArrayBase<OwnedRepr<R>, D>,
    layer_absorptance: Vec<ArrayBase<OwnedRepr<R>, D>>,
    balance_residual: ArrayBase<OwnedRepr<R>, D>,
}

pub struct PlaneWaveFields<C, D>
where
    D: Dimension,
{
    electric_tangential: ArrayBase<OwnedRepr<C>, D>,
    magnetic_tangential: ArrayBase<OwnedRepr<C>, D>,
}

#[cfg(test)]
mod tests {
    use ndarray::arr1;
    use num_complex::Complex64;

    use super::*;

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    #[test]
    fn value_conversion_preserves_layer_boundary_waves() {
        let generic = vec![LayerBoundaryWavesGeneric::new(
            BidirectionalWavesGeneric::new(arr1(&[c(1.0), c(2.0)]), arr1(&[c(3.0), c(4.0)])),
            BidirectionalWavesGeneric::new(arr1(&[c(5.0), c(6.0)]), arr1(&[c(7.0), c(8.0)])),
        )];

        let fields = value_fields_from_generic(generic);

        assert_eq!(fields.len(), 1);
        assert!(fields.derivatives().is_none());

        let layer = fields.layer(0).unwrap();

        assert_eq!(layer.left().forward(), &arr1(&[c(1.0), c(2.0)]),);
        assert_eq!(layer.left().backward(), &arr1(&[c(3.0), c(4.0)]),);
        assert_eq!(layer.right().forward(), &arr1(&[c(5.0), c(6.0)]),);
        assert_eq!(layer.right().backward(), &arr1(&[c(7.0), c(8.0)]),);
    }

    #[test]
    fn first_order_conversion_separates_value_and_first_derivative() {
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

        let fields = first_order_fields_from_generic(generic, DerivativeVariable::VacuumWavenumber);

        let value = fields.layer(0).unwrap();
        assert_eq!(value.left().forward(), &arr1(&[c(1.0)]));
        assert_eq!(value.left().backward(), &arr1(&[c(2.0)]));
        assert_eq!(value.right().forward(), &arr1(&[c(3.0)]));
        assert_eq!(value.right().backward(), &arr1(&[c(4.0)]));

        let derivatives = fields.derivatives().unwrap();

        assert_eq!(derivatives.variable(), DerivativeVariable::VacuumWavenumber,);
        assert!(derivatives.second().is_none());

        let first = derivatives.first_layer(0).unwrap();

        assert_eq!(first.left().forward(), &arr1(&[c(11.0)]),);
        assert_eq!(first.left().backward(), &arr1(&[c(12.0)]),);
        assert_eq!(first.right().forward(), &arr1(&[c(13.0)]),);
        assert_eq!(first.right().backward(), &arr1(&[c(14.0)]),);
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

        let fields = second_order_fields_from_generic(generic, DerivativeVariable::Thickness(0));

        let derivatives = fields.derivatives().unwrap();

        let first = derivatives.first_layer(0).unwrap();
        let second = derivatives.second_layer(0).unwrap();

        assert_eq!(first.left().forward(), &arr1(&[c(11.0)]),);
        assert_eq!(first.right().backward(), &arr1(&[c(14.0)]),);

        assert_eq!(second.left().forward(), &arr1(&[c(21.0)]),);
        assert_eq!(second.left().backward(), &arr1(&[c(22.0)]),);
        assert_eq!(second.right().forward(), &arr1(&[c(23.0)]),);
        assert_eq!(second.right().backward(), &arr1(&[c(24.0)]),);
    }

    #[test]
    fn empty_generic_fields_produce_empty_public_fields() {
        let generic: Vec<LayerBoundaryWavesGeneric<ArrayBase<OwnedRepr<C>, ndarray::Ix1>>> =
            Vec::new();

        let fields = value_fields_from_generic(generic);

        assert!(fields.is_empty());
        assert_eq!(fields.len(), 0);
        assert!(fields.derivatives().is_none());
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

        let fields = value_fields_from_generic(generic);

        assert_eq!(fields.layer(0).unwrap().left().forward()[0], c(1.0),);
        assert_eq!(fields.layer(1).unwrap().left().forward()[0], c(3.0),);
    }
}
