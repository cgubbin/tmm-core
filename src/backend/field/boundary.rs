use crate::{
    DerivativeVariable, IncidentSide,
    backend::jet::{ArrayJet, ArrayJetFirst},
};

use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, OwnedRepr};

/// Boundary-wave values returned by an internal-field solve.
///
/// Every solution contains the reconstructed boundary-wave values. A
/// differentiated solve additionally contains first and optionally second
/// derivatives of those amplitudes.
///
/// Accessing [`Self::values`] is therefore valid for every variant, while
/// [`Self::derivatives`] returns `None` for an undifferentiated solve.
#[derive(Clone, Debug, PartialEq)]
pub enum BoundaryWaveSolution<C, D>
where
    C: ComplexField,
    D: Dimension,
{
    /// Boundary-wave values without amplitude derivatives.
    Values(BoundaryWaves<C, D>),

    /// Boundary-wave values and their derivatives.
    Differentiated(DifferentiatedBoundaryWaves<C, D>),
}

impl<C, D> BoundaryWaveSolution<C, D>
where
    C: ComplexField,
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

    pub fn spectral(&self) -> Option<&BoundaryWaveDerivatives<C, D>> {
        let derivatives = self.derivatives()?;

        if derivatives.variable.is_spectral() {
            return Some(derivatives);
        }

        None
    }

    pub fn structural(&self) -> Option<&BoundaryWaveDerivatives<C, D>> {
        let derivatives = self.derivatives()?;

        if derivatives.variable.is_structural() {
            return Some(derivatives);
        }

        None
    }

    pub fn derivatives(&self) -> Option<&BoundaryWaveDerivatives<C, D>> {
        match self {
            Self::Values(_) => None,
            Self::Differentiated(differentiated) => Some(&differentiated.derivatives),
        }
    }

    pub fn into_values_and_derivatives(
        self,
    ) -> (BoundaryWaves<C, D>, Option<BoundaryWaveDerivatives<C, D>>) {
        match self {
            Self::Values(values) => (values, None),
            Self::Differentiated(differentiated) => {
                let (values, derivatives) = differentiated.into_parts();
                (values, Some(derivatives))
            }
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
    C: ComplexField,
    D: Dimension,
{
    exterior: ExteriorBoundaryWaves<C, D>,
    layers: Vec<LayerBoundaryWaves<C, D>>,
}

impl<C, D> BoundaryWaves<C, D>
where
    C: ComplexField,
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

/// Forward and backward wave amplitudes in the two semi-infinite exterior
/// media.
///
/// Directions are geometric and independent of the physical problem:
///
/// - `forward` propagates from left to right;
/// - `backward` propagates from right to left.
///
/// For a driven solution, one exterior component represents the unit incident
/// wave. For a source-free outgoing mode, both incoming exterior components
/// are zero.
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
        C: ComplexField,
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
        C: ComplexField,
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

/// Wave amplitudes immediately inside both boundaries of one finite layer.
///
/// The stored reference planes are:
///
/// ```text
/// left boundary | finite layer | right boundary
/// ```
///
/// Both forward and backward amplitudes are retained at both boundaries.
/// Keeping both reference planes avoids reconstructing one boundary by
/// dividing by a potentially very small evanescent propagation factor.
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

    /// Return the waves immediately inside the layer at its left boundary.
    pub fn left(&self) -> &BidirectionalWaves<C, D> {
        &self.left
    }

    /// Return the waves immediately inside the layer at its right boundary.
    pub fn right(&self) -> &BidirectionalWaves<C, D> {
        &self.right
    }

    /// Consume the value and return `(left, right)`.
    pub fn into_parts(self) -> (BidirectionalWaves<C, D>, BidirectionalWaves<C, D>) {
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

    /// Return the wave amplitude propagating geometrically from left to right.
    pub fn forward(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.forward
    }

    /// Return the wave amplitude propagating geometrically from right to left.
    pub fn backward(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.backward
    }

    /// Consume the pair and return `(forward, backward)`.
    pub fn into_parts(self) -> (ArrayBase<OwnedRepr<C>, D>, ArrayBase<OwnedRepr<C>, D>) {
        (self.forward, self.backward)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct DifferentiatedBoundaryWaves<C, D>
where
    D: Dimension,
    C: ComplexField,
{
    values: BoundaryWaves<C, D>,
    derivatives: BoundaryWaveDerivatives<C, D>,
}

impl<C, D> DifferentiatedBoundaryWaves<C, D>
where
    C: ComplexField,
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
    C: ComplexField,
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
    C: ComplexField,
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

/// Derivatives of forward and backward wave amplitudes at one reference plane.
/// The derivative variable and derivative order are recorded by the containing
/// [BoundaryWaveDerivatives] object.
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

#[derive(Clone, Debug)]
pub(crate) struct BoundaryWavesGeneric<A> {
    exterior: ExteriorBoundaryWavesGeneric<A>,
    layers: Vec<LayerBoundaryWavesGeneric<A>>,
}

impl<A> BoundaryWavesGeneric<A> {
    pub(crate) fn new(
        exterior: ExteriorBoundaryWavesGeneric<A>,
        layers: Vec<LayerBoundaryWavesGeneric<A>>,
    ) -> Self {
        Self { exterior, layers }
    }

    pub(crate) fn exterior(&self) -> &ExteriorBoundaryWavesGeneric<A> {
        &self.exterior
    }

    pub(crate) fn layers(&self) -> &[LayerBoundaryWavesGeneric<A>] {
        &self.layers
    }

    pub(crate) fn layer(&self, index: usize) -> Option<&LayerBoundaryWavesGeneric<A>> {
        self.layers.get(index)
    }

    pub(crate) fn len(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        ExteriorBoundaryWavesGeneric<A>,
        Vec<LayerBoundaryWavesGeneric<A>>,
    ) {
        (self.exterior, self.layers)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LayerBoundaryWavesGeneric<A> {
    left: BidirectionalWavesGeneric<A>,
    right: BidirectionalWavesGeneric<A>,
}

impl<A> LayerBoundaryWavesGeneric<A> {
    pub(crate) fn new(
        left: BidirectionalWavesGeneric<A>,
        right: BidirectionalWavesGeneric<A>,
    ) -> Self {
        Self { left, right }
    }

    pub(crate) fn left(&self) -> &BidirectionalWavesGeneric<A> {
        &self.left
    }

    pub(crate) fn right(&self) -> &BidirectionalWavesGeneric<A> {
        &self.right
    }

    pub(crate) fn into_parts(self) -> (BidirectionalWavesGeneric<A>, BidirectionalWavesGeneric<A>) {
        (self.left, self.right)
    }
}

impl<C, D> LayerBoundaryWavesGeneric<ArrayBase<OwnedRepr<C>, D>>
where
    C: ComplexField,
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
    C: ComplexField,
    D: Dimension,
{
    layers
        .into_iter()
        .map(|layer| {
            let (left, right) = layer.into_parts();
            let (lf, lb) = left.into_parts();
            let (rf, rb) = right.into_parts();
            (lf, lb, rf, rb)
        })
        .map(|(lf, lb, rf, rb)| layer_waves(lf, lb, rf, rb))
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
    C: ComplexField,
    D: Dimension,
{
    let mut values = Vec::with_capacity(layers.len());

    let mut first = Vec::with_capacity(layers.len());

    for layer in layers {
        let (left, right) = layer.into_parts();
        let (lf, lb) = left.into_parts();
        let (rf, rb) = right.into_parts();

        let (left_forward, left_forward_first) = lf.into_parts();
        let (left_backward, left_backward_first) = lb.into_parts();

        let (right_forward, right_forward_first) = rf.into_parts();
        let (right_backward, right_backward_first) = rb.into_parts();

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
    C: ComplexField,
    D: Dimension,
{
    let mut values = Vec::with_capacity(layers.len());

    let mut first = Vec::with_capacity(layers.len());

    let mut second = Vec::with_capacity(layers.len());

    for layer in layers {
        let (left, right) = layer.into_parts();
        let (lf, lb) = left.into_parts();
        let (rf, rb) = right.into_parts();

        let (left_forward, left_forward_first, left_forward_second) = lf.into_parts();

        let (left_backward, left_backward_first, left_backward_second) = lb.into_parts();

        let (right_forward, right_forward_first, right_forward_second) = rf.into_parts();

        let (right_backward, right_backward_first, right_backward_second) = rb.into_parts();

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
pub(crate) struct ExteriorBoundaryWavesGeneric<A> {
    left: BidirectionalWavesGeneric<A>,
    right: BidirectionalWavesGeneric<A>,
}

impl<A> ExteriorBoundaryWavesGeneric<A> {
    pub(crate) fn new(
        left: BidirectionalWavesGeneric<A>,
        right: BidirectionalWavesGeneric<A>,
    ) -> Self {
        Self { left, right }
    }

    pub(crate) fn left(&self) -> &BidirectionalWavesGeneric<A> {
        &self.left
    }

    pub(crate) fn right(&self) -> &BidirectionalWavesGeneric<A> {
        &self.right
    }

    pub(crate) fn into_parts(self) -> (BidirectionalWavesGeneric<A>, BidirectionalWavesGeneric<A>) {
        (self.left, self.right)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct BidirectionalWavesGeneric<A> {
    forward: A,
    backward: A,
}

impl<A> BidirectionalWavesGeneric<A> {
    pub(crate) fn new(forward: A, backward: A) -> Self {
        Self { forward, backward }
    }

    pub(crate) fn forward(&self) -> &A {
        &self.forward
    }

    pub(crate) fn backward(&self) -> &A {
        &self.backward
    }

    pub(crate) fn into_parts(self) -> (A, A) {
        (self.forward, self.backward)
    }
}

impl<C, D> BidirectionalWavesGeneric<ArrayBase<OwnedRepr<C>, D>>
where
    C: ComplexField,
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

pub(crate) fn generic_boundary_values<C, D>(
    waves: &BoundaryWaves<C, D>,
) -> BoundaryWavesGeneric<ArrayBase<OwnedRepr<C>, D>>
where
    C: ComplexField,
    D: Dimension,
{
    BoundaryWavesGeneric::new(
        generic_exterior_values(waves.exterior()),
        waves.layers().iter().map(generic_layer_values).collect(),
    )
}

fn generic_bidirectional_values<C, D>(
    waves: &BidirectionalWaves<C, D>,
) -> BidirectionalWavesGeneric<ArrayBase<OwnedRepr<C>, D>>
where
    C: Clone,
    D: Dimension,
{
    BidirectionalWavesGeneric::new(waves.forward().clone(), waves.backward().clone())
}

fn generic_exterior_values<C, D>(
    waves: &ExteriorBoundaryWaves<C, D>,
) -> ExteriorBoundaryWavesGeneric<ArrayBase<OwnedRepr<C>, D>>
where
    C: Clone,
    D: Dimension,
{
    ExteriorBoundaryWavesGeneric::new(
        generic_bidirectional_values(waves.left()),
        generic_bidirectional_values(waves.right()),
    )
}

fn generic_layer_values<C, D>(
    waves: &LayerBoundaryWaves<C, D>,
) -> LayerBoundaryWavesGeneric<ArrayBase<OwnedRepr<C>, D>>
where
    C: Clone,
    D: Dimension,
{
    LayerBoundaryWavesGeneric::new(
        generic_bidirectional_values(waves.left()),
        generic_bidirectional_values(waves.right()),
    )
}

pub(crate) fn generic_boundary_first<C, D>(
    values: &BoundaryWaves<C, D>,
    derivatives: &BoundaryWaveDerivatives<C, D>,
) -> BoundaryWavesGeneric<ArrayJetFirst<C, D>>
where
    C: ComplexField,
    D: Dimension,
{
    BoundaryWavesGeneric::new(
        generic_exterior_first(values.exterior(), derivatives.exterior_first()),
        values
            .layers()
            .iter()
            .zip(derivatives.first_layers())
            .map(|(value, first)| generic_layer_first(value, first))
            .collect(),
    )
}

fn generic_bidirectional_first<C, D>(
    values: &BidirectionalWaves<C, D>,
    first: &BidirectionalWaveDifferential<C, D>,
) -> BidirectionalWavesGeneric<ArrayJetFirst<C, D>>
where
    C: ComplexField,
    D: Dimension,
{
    BidirectionalWavesGeneric::new(
        ArrayJetFirst::from_parts(values.forward().clone(), first.forward().clone()),
        ArrayJetFirst::from_parts(values.backward().clone(), first.backward().clone()),
    )
}

fn generic_exterior_first<C, D>(
    values: &ExteriorBoundaryWaves<C, D>,
    first: &ExteriorBoundaryWaveDifferential<C, D>,
) -> ExteriorBoundaryWavesGeneric<ArrayJetFirst<C, D>>
where
    C: ComplexField,
    D: Dimension,
{
    ExteriorBoundaryWavesGeneric::new(
        generic_bidirectional_first(values.left(), first.left()),
        generic_bidirectional_first(values.right(), first.right()),
    )
}

fn generic_layer_first<C, D>(
    values: &LayerBoundaryWaves<C, D>,
    first: &LayerBoundaryWaveDifferential<C, D>,
) -> LayerBoundaryWavesGeneric<ArrayJetFirst<C, D>>
where
    C: ComplexField,
    D: Dimension,
{
    LayerBoundaryWavesGeneric::new(
        generic_bidirectional_first(values.left(), first.left()),
        generic_bidirectional_first(values.right(), first.right()),
    )
}

pub(crate) fn generic_boundary_second<C, D>(
    values: &BoundaryWaves<C, D>,
    derivatives: &BoundaryWaveDerivatives<C, D>,
) -> Option<BoundaryWavesGeneric<ArrayJet<C, D>>>
where
    C: ComplexField,
    D: Dimension,
{
    let exterior_second = derivatives.exterior_second()?;

    let second_layers = derivatives.second_layers()?;

    Some(BoundaryWavesGeneric::new(
        generic_exterior_second(
            values.exterior(),
            derivatives.exterior_first(),
            exterior_second,
        ),
        values
            .layers()
            .iter()
            .zip(derivatives.first_layers())
            .zip(second_layers)
            .map(|((value, first), second)| generic_layer_second(value, first, second))
            .collect(),
    ))
}

fn generic_bidirectional_second<C, D>(
    values: &BidirectionalWaves<C, D>,
    first: &BidirectionalWaveDifferential<C, D>,
    second: &BidirectionalWaveDifferential<C, D>,
) -> BidirectionalWavesGeneric<ArrayJet<C, D>>
where
    C: ComplexField,
    D: Dimension,
{
    BidirectionalWavesGeneric::new(
        ArrayJet::from_parts(
            values.forward().clone(),
            first.forward().clone(),
            second.forward().clone(),
        ),
        ArrayJet::from_parts(
            values.backward().clone(),
            first.backward().clone(),
            second.backward().clone(),
        ),
    )
}

fn generic_exterior_second<C, D>(
    values: &ExteriorBoundaryWaves<C, D>,
    first: &ExteriorBoundaryWaveDifferential<C, D>,
    second: &ExteriorBoundaryWaveDifferential<C, D>,
) -> ExteriorBoundaryWavesGeneric<ArrayJet<C, D>>
where
    C: ComplexField,
    D: Dimension,
{
    ExteriorBoundaryWavesGeneric::new(
        generic_bidirectional_second(values.left(), first.left(), second.left()),
        generic_bidirectional_second(values.right(), first.right(), second.right()),
    )
}

fn generic_layer_second<C, D>(
    values: &LayerBoundaryWaves<C, D>,
    first: &LayerBoundaryWaveDifferential<C, D>,
    second: &LayerBoundaryWaveDifferential<C, D>,
) -> LayerBoundaryWavesGeneric<ArrayJet<C, D>>
where
    C: ComplexField,
    D: Dimension,
{
    LayerBoundaryWavesGeneric::new(
        generic_bidirectional_second(values.left(), first.left(), second.left()),
        generic_bidirectional_second(values.right(), first.right(), second.right()),
    )
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Ix1, arr1};
    use num_complex::Complex64;

    use super::*;

    type C = Complex64;
    type D = Ix1;

    fn waves(forward: &[C], backward: &[C]) -> BidirectionalWaves<C, D> {
        BidirectionalWaves::new(
            Array1::from_vec(forward.to_vec()),
            Array1::from_vec(backward.to_vec()),
        )
    }

    #[test]
    fn bidirectional_waves_preserve_geometric_components() {
        let forward = arr1(&[C::new(1.0, 2.0)]);
        let backward = arr1(&[C::new(3.0, 4.0)]);

        let waves = BidirectionalWaves::new(forward.clone(), backward.clone());

        assert_eq!(waves.forward(), &forward);
        assert_eq!(waves.backward(), &backward);
    }

    #[test]
    fn driven_left_incidence_sets_correct_exterior_components() {
        let reflection = arr1(&[C::new(0.25, 0.5)]);
        let transmission = arr1(&[C::new(0.75, -0.25)]);

        let exterior = ExteriorBoundaryWaves::from_values(
            reflection.clone(),
            transmission.clone(),
            IncidentSide::Left,
        );

        assert_eq!(exterior.left().forward(), &arr1(&[C::new(1.0, 0.0)]));
        assert_eq!(exterior.left().backward(), &reflection);
        assert_eq!(exterior.right().forward(), &transmission);
        assert_eq!(exterior.right().backward(), &arr1(&[C::new(0.0, 0.0)]));
    }

    #[test]
    fn driven_right_incidence_sets_correct_exterior_components() {
        let reflection = arr1(&[C::new(0.25, 0.5)]);
        let transmission = arr1(&[C::new(0.75, -0.25)]);

        let exterior = ExteriorBoundaryWaves::from_values(
            reflection.clone(),
            transmission.clone(),
            IncidentSide::Right,
        );

        assert_eq!(exterior.left().forward(), &arr1(&[C::new(0.0, 0.0)]));
        assert_eq!(exterior.left().backward(), &transmission);
        assert_eq!(exterior.right().forward(), &reflection);
        assert_eq!(exterior.right().backward(), &arr1(&[C::new(1.0, 0.0)]));
    }

    #[test]
    fn outgoing_exterior_has_no_incoming_components() {
        let left_outgoing = arr1(&[C::new(2.0, 1.0)]);
        let right_outgoing = arr1(&[C::new(-1.0, 3.0)]);

        let exterior = ExteriorBoundaryWaves::from_outgoing_values(
            left_outgoing.clone(),
            right_outgoing.clone(),
        );

        assert_eq!(exterior.left().forward(), &arr1(&[C::new(0.0, 0.0)]));
        assert_eq!(exterior.left().backward(), &left_outgoing);

        assert_eq!(exterior.right().forward(), &right_outgoing);
        assert_eq!(exterior.right().backward(), &arr1(&[C::new(0.0, 0.0)]));
    }

    #[test]
    fn generic_layer_scaling_applies_to_every_component() {
        let layer = LayerBoundaryWavesGeneric::new(
            BidirectionalWavesGeneric::new(arr1(&[C::new(1.0, 0.0)]), arr1(&[C::new(2.0, 0.0)])),
            BidirectionalWavesGeneric::new(arr1(&[C::new(3.0, 0.0)]), arr1(&[C::new(4.0, 0.0)])),
        );

        let scaled = layer.scale(arr1(&[C::new(0.5, 0.0)]));

        assert_eq!(scaled.left.forward, arr1(&[C::new(0.5, 0.0)]));
        assert_eq!(scaled.left.backward, arr1(&[C::new(1.0, 0.0)]));
        assert_eq!(scaled.right.forward, arr1(&[C::new(1.5, 0.0)]));
        assert_eq!(scaled.right.backward, arr1(&[C::new(2.0, 0.0)]));
    }

    #[test]
    fn value_solution_exposes_values_without_derivatives() {
        let exterior = ExteriorBoundaryWaves::new(
            waves(&[C::new(1.0, 0.0)], &[C::new(2.0, 0.0)]),
            waves(&[C::new(3.0, 0.0)], &[C::new(4.0, 0.0)]),
        );

        let solution = BoundaryWaveSolution::new(exterior, Vec::new());

        assert!(solution.derivatives().is_none());
        assert!(solution.is_empty());
        assert_eq!(solution.values().len(), 0);
    }
}
