use crate::{
    ComplexScalar, DerivativeVariable, IncidentSide,
    backend::jet::{ArrayJet, ArrayJetFirst},
};

use ndarray::{ArrayBase, Dimension, OwnedRepr};

#[derive(Clone, Debug, PartialEq)]
pub enum BoundaryWaveSolution<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    Values(BoundaryWaves<C, D>),
    Differentiated(DifferentiatedBoundaryWaves<C, D>),
}

impl<C, D> BoundaryWaveSolution<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn new(
        exterior: ExteriorBoundaryWaves<C, D>,
        layers: Vec<LayerBoundaryWaves<C, D>>,
    ) -> Self {
        Self::Values(BoundaryWaves::new(exterior, layers))
    }

    pub(crate) fn new_with_derivative(
        exterior: ExteriorBoundaryWaves<C, D>,
        layers: Vec<LayerBoundaryWaves<C, D>>,
        derivatives: BoundaryWaveDerivatives<C, D>,
    ) -> Self {
        debug_assert_eq!(layers.len(), derivatives.first_layers().len());

        if let Some(second) = derivatives.second_layers() {
            debug_assert_eq!(layers.len(), second.len());
        }
        Self::Differentiated(DifferentiatedBoundaryWaves::new(
            BoundaryWaves::new(exterior, layers),
            derivatives,
        ))
    }

    pub fn exterior(&self) -> &ExteriorBoundaryWaves<C, D> {
        match self {
            Self::Values(boundary_waves) => boundary_waves.exterior(),
            Self::Differentiated(differentiated) => differentiated.exterior(),
        }
    }

    pub fn layers(&self) -> &[LayerBoundaryWaves<C, D>] {
        match self {
            Self::Values(boundary_waves) => boundary_waves.layers(),
            Self::Differentiated(differentiated) => differentiated.layers(),
        }
    }

    pub fn layer(&self, index: usize) -> Option<&LayerBoundaryWaves<C, D>> {
        self.layers().get(index)
    }

    pub fn len(&self) -> usize {
        self.layers().len()
    }

    pub fn is_empty(&self) -> bool {
        self.layers().is_empty()
    }

    pub fn values(&self) -> &BoundaryWaves<C, D> {
        match self {
            Self::Values(boundary_waves) => &boundary_waves,
            Self::Differentiated(differentiated) => &differentiated.values,
        }
    }

    pub fn derivatives(&self) -> Option<&BoundaryWaveDerivatives<C, D>> {
        match self {
            Self::Values(_) => None,
            Self::Differentiated(differentiated) => Some(&differentiated.derivatives),
        }
    }

    #[allow(clippy::type_complexity)]
    pub fn into_parts(
        self,
    ) -> (
        ExteriorBoundaryWaves<C, D>,
        Vec<LayerBoundaryWaves<C, D>>,
        Option<BoundaryWaveDerivatives<C, D>>,
    ) {
        match self {
            Self::Values(boundary_waves) => (boundary_waves.exterior, boundary_waves.layers, None),
            Self::Differentiated(differentiated) => (
                differentiated.values.exterior,
                differentiated.values.layers,
                Some(differentiated.derivatives),
            ),
        }
    }
}

/// Boundary-wave amplitudes in the exterior media and every finite layer.
///
/// Geometric directions are fixed:
///
/// - `forward` means left to right;
/// - `backward` means right to left.
///
/// These meanings apply equally to driven scattering solutions and
/// source-free outgoing modes.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryWaves<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    exterior: ExteriorBoundaryWaves<C, D>,
    layers: Vec<LayerBoundaryWaves<C, D>>,
}

impl<C, D> BoundaryWaves<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn new(
        exterior: ExteriorBoundaryWaves<C, D>,
        layers: Vec<LayerBoundaryWaves<C, D>>,
    ) -> Self {
        Self { exterior, layers }
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

    #[allow(clippy::type_complexity)]
    pub fn into_parts(self) -> (ExteriorBoundaryWaves<C, D>, Vec<LayerBoundaryWaves<C, D>>) {
        (self.exterior, self.layers)
    }
}

/// Modal amplitudes in the two semi-infinite exterior media.
/// Directions are geometric:
/// - forward propagates left to right;
/// - backward propagates right to left.
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

    pub(crate) fn from_outgoing_values(
        left_outgoing: ArrayBase<OwnedRepr<C>, D>,
        right_outgoing: ArrayBase<OwnedRepr<C>, D>,
    ) -> Self
    where
        C: ComplexScalar,
    {
        let zero = left_outgoing.mapv(|_| C::zero());

        Self::new(
            BidirectionalWaves::new(zero.clone(), left_outgoing),
            BidirectionalWaves::new(right_outgoing, zero),
        )
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

/// Forward and backward wave amplitudes at both boundaries of one finite layer.
/// The left and right fields refer to geometric layer boundaries:
/// left boundary | finite layer | right boundary
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

/// Forward- and backward-propagating wave amplitudes at one reference plane.
/// Directions are geometric:
/// - forward propagates from left to right;
/// - backward propagates from right to left.
/// The meaning does not change with [crate::backend::IncidentSide].
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

#[derive(Clone, Debug, PartialEq)]
pub struct DifferentiatedBoundaryWaves<C, D>
where
    D: Dimension,
    C: ComplexScalar,
{
    values: BoundaryWaves<C, D>,
    derivatives: BoundaryWaveDerivatives<C, D>,
}

impl<C, D> DifferentiatedBoundaryWaves<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn new(
        values: BoundaryWaves<C, D>,
        derivatives: BoundaryWaveDerivatives<C, D>,
    ) -> Self {
        Self {
            values,
            derivatives,
        }
    }

    pub fn values(&self) -> &BoundaryWaves<C, D> {
        &self.values
    }

    pub fn derivatives(&self) -> &BoundaryWaveDerivatives<C, D> {
        &self.derivatives
    }

    pub fn exterior(&self) -> &ExteriorBoundaryWaves<C, D> {
        &self.values.exterior
    }

    pub fn layers(&self) -> &[LayerBoundaryWaves<C, D>] {
        &self.values.layers
    }

    #[allow(clippy::type_complexity)]
    pub fn into_parts(self) -> (BoundaryWaves<C, D>, BoundaryWaveDerivatives<C, D>) {
        (self.values, self.derivatives)
    }
}

/// First and optional second derivatives of all boundary waves.
#[derive(Clone, Debug, PartialEq)]
pub struct BoundaryWaveDerivatives<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    variable: DerivativeVariable,

    exterior_first: ExteriorBoundaryWaveDifferential<C, D>,
    first_layers: Vec<LayerBoundaryWaveDifferential<C, D>>,

    exterior_second: Option<ExteriorBoundaryWaveDifferential<C, D>>,
    second_layers: Option<Vec<LayerBoundaryWaveDifferential<C, D>>>,
}

impl<C, D> BoundaryWaveDerivatives<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn new(
        variable: DerivativeVariable,
        exterior_first: ExteriorBoundaryWaveDifferential<C, D>,
        first_layers: Vec<LayerBoundaryWaveDifferential<C, D>>,
    ) -> Self {
        Self {
            variable,
            exterior_first,
            first_layers,
            exterior_second: None,
            second_layers: None,
        }
    }

    pub(crate) fn with_second(
        mut self,
        exterior_second: ExteriorBoundaryWaveDifferential<C, D>,
        second_layers: Vec<LayerBoundaryWaveDifferential<C, D>>,
    ) -> Self {
        debug_assert_eq!(self.first_layers.len(), second_layers.len());

        self.exterior_second = Some(exterior_second);
        self.second_layers = Some(second_layers);
        self
    }

    pub fn variable(&self) -> DerivativeVariable {
        self.variable
    }

    pub fn exterior_first(&self) -> &ExteriorBoundaryWaveDifferential<C, D> {
        &self.exterior_first
    }

    pub fn first_layers(&self) -> &[LayerBoundaryWaveDifferential<C, D>] {
        &self.first_layers
    }

    pub fn first_layer(&self, index: usize) -> Option<&LayerBoundaryWaveDifferential<C, D>> {
        self.first_layers.get(index)
    }

    pub fn exterior_second(&self) -> Option<&ExteriorBoundaryWaveDifferential<C, D>> {
        self.exterior_second.as_ref()
    }

    pub fn second_layers(&self) -> Option<&[LayerBoundaryWaveDifferential<C, D>]> {
        self.second_layers.as_deref()
    }

    pub fn second_layer(&self, index: usize) -> Option<&LayerBoundaryWaveDifferential<C, D>> {
        self.second_layers.as_ref()?.get(index)
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

/// Derivatives of internal wave amplitudes at both boundaries of one layer.
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

/// Derivatives of forward and backward wave amplitudes at one reference plane.
/// The derivative variable and derivative order are recorded by the containing
/// [PlaneWaveBoundaryWaveDerivatives] object.
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

impl<C, D> LayerBoundaryWavesGeneric<ArrayBase<OwnedRepr<C>, D>>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn scale(self, factor: ArrayBase<OwnedRepr<C>, D>) -> Self {
        Self {
            left: self.left.scale(factor.clone()),
            right: self.right.scale(factor),
        }
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

impl<C, D> BidirectionalWavesGeneric<ArrayBase<OwnedRepr<C>, D>>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn scale(self, factor: ArrayBase<OwnedRepr<C>, D>) -> Self {
        Self {
            forward: self.forward * factor.clone(),
            backward: self.backward * factor,
        }
    }
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
