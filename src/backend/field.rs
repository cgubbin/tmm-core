use crate::{ComplexScalar, DerivativeVariable, PlaneWaveResponse};

use ndarray::{ArrayBase, Dimension, OwnedRepr};

/// Internal-field data requested from the scattering solve.
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

pub struct PlaneWaveFieldResponse<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    response: PlaneWaveResponse<C, D>,
    fields: PlaneWaveInternalFields<C, D>,
}

impl<C, D> PlaneWaveFieldResponse<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub fn new(response: PlaneWaveResponse<C, D>, fields: PlaneWaveInternalFields<C, D>) -> Self {
        Self { response, fields }
    }

    pub fn response(&self) -> &PlaneWaveResponse<C, D> {
        &self.response
    }

    pub fn fields(&self) -> &PlaneWaveInternalFields<C, D> {
        &self.fields
    }

    pub fn into_parts(self) -> (PlaneWaveResponse<C, D>, PlaneWaveInternalFields<C, D>) {
        (self.response, self.fields)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveInternalFields<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    layers: Vec<LayerBoundaryWaves<C, D>>,
    derivatives: Option<PlaneWaveInternalFieldDerivatives<C, D>>,
}

impl<C, D> PlaneWaveInternalFields<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub fn new(layers: Vec<LayerBoundaryWaves<C, D>>) -> Self {
        Self {
            layers,
            derivatives: None,
        }
    }

    pub fn with_derivatives(
        layers: Vec<LayerBoundaryWaves<C, D>>,
        derivatives: PlaneWaveInternalFieldDerivatives<C, D>,
    ) -> Self {
        Self {
            layers,
            derivatives: Some(derivatives),
        }
    }

    pub fn layers(&self) -> &[LayerBoundaryWaves<C, D>] {
        &self.layers
    }

    pub fn derivatives(&self) -> Option<&PlaneWaveInternalFieldDerivatives<C, D>> {
        self.derivatives.as_ref()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveInternalFieldDerivatives<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    variable: DerivativeVariable,
    first: Vec<LayerBoundaryWaveDifferential<C, D>>,
    second: Option<Vec<LayerBoundaryWaveDifferential<C, D>>>,
}

/// Forward- and backward-propagating wave amplitudes at one reference plane.
///
/// Directions are geometric:
///
/// - `forward` propagates from left to right;
/// - `backward` propagates from right to left.
///
/// The meaning does not change with [`IncidentSide`].
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
    pub fn new(forward: ArrayBase<OwnedRepr<C>, D>, backward: ArrayBase<OwnedRepr<C>, D>) -> Self {
        debug_assert_eq!(forward.raw_dim(), backward.raw_dim(),);

        Self { forward, backward }
    }

    /// Return the left-to-right wave amplitude.
    pub fn forward(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.forward
    }

    /// Return the right-to-left wave amplitude.
    pub fn backward(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.backward
    }

    /// Consume the pair and return its amplitudes.
    pub fn into_parts(self) -> (ArrayBase<OwnedRepr<C>, D>, ArrayBase<OwnedRepr<C>, D>) {
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
    pub fn new(left: BidirectionalWaves<C, D>, right: BidirectionalWaves<C, D>) -> Self {
        debug_assert_eq!(left.forward().raw_dim(), right.forward().raw_dim(),);

        Self { left, right }
    }

    /// Return waves immediately inside the layer at its left boundary.
    pub fn left(&self) -> &BidirectionalWaves<C, D> {
        &self.left
    }

    /// Return waves immediately inside the layer at its right boundary.
    pub fn right(&self) -> &BidirectionalWaves<C, D> {
        &self.right
    }

    /// Consume the layer response and return both boundary wave pairs.
    pub fn into_parts(self) -> (BidirectionalWaves<C, D>, BidirectionalWaves<C, D>) {
        (self.left, self.right)
    }
}

/// Derivatives of forward and backward wave amplitudes at one reference plane.
///
/// The derivative variable and derivative order are recorded by the containing
/// [`PlaneWaveInternalFieldDerivatives`] object.
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
    pub fn new(forward: ArrayBase<OwnedRepr<C>, D>, backward: ArrayBase<OwnedRepr<C>, D>) -> Self {
        debug_assert_eq!(forward.raw_dim(), backward.raw_dim(),);

        Self { forward, backward }
    }

    /// Return the derivative of the forward wave.
    pub fn forward(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.forward
    }

    /// Return the derivative of the backward wave.
    pub fn backward(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.backward
    }

    /// Consume the differential and return both arrays.
    pub fn into_parts(self) -> (ArrayBase<OwnedRepr<C>, D>, ArrayBase<OwnedRepr<C>, D>) {
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
    pub fn new(
        left: BidirectionalWaveDifferential<C, D>,
        right: BidirectionalWaveDifferential<C, D>,
    ) -> Self {
        debug_assert_eq!(left.forward().raw_dim(), right.forward().raw_dim(),);

        Self { left, right }
    }

    /// Return derivatives at the left boundary.
    pub fn left(&self) -> &BidirectionalWaveDifferential<C, D> {
        &self.left
    }

    /// Return derivatives at the right boundary.
    pub fn right(&self) -> &BidirectionalWaveDifferential<C, D> {
        &self.right
    }

    /// Consume the result and return both boundary differentials.
    pub fn into_parts(
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

fn value_fields_from_generic<C, D>(
    layers: Vec<LayerBoundaryWavesGeneric<ArrayBase<OwnedRepr<C>, D>>>,
) -> Vec<LayerBoundaryWaves<C, D>>
where
    D: Dimension,
{
    layers
        .into_iter()
        .map(|layer| {
            LayerBoundaryWaves::new(
                BidirectionalWaves::new(layer.left.forward, layer.left.backward),
                BidirectionalWaves::new(layer.right.forward, layer.right.backward),
            )
        })
        .collect()
}
