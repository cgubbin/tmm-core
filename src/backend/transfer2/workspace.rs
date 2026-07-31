use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, OwnedRepr};
use num_traits::{One, Zero};

use crate::{
    ComplexScalar,
    algebra::ScalarAlgebra,
    backend::{
        BidirectionalWaves, LayerBoundaryWaves, PlaneWaveSolution, PlaneWaveSolutionSource,
        RunMode, SolutionWorkspace,
        isotropic::IsotropicLayerQuantities,
        solution::PlaneWaveSolutionView,
        transfer2::{Transfer2Entries, entries::Transfer2ExteriorContext},
    },
};

/// The transfer state at a single spatial boundary.
///
/// The state is represented by the field-like component and its corresponding
/// slope-like component. A transfer matrix maps the state at its right
/// boundary to the state at its left boundary.
#[derive(Clone, Debug)]
pub(crate) struct TransferState<A> {
    field: A,
    slope: A,
}

impl<A> TransferState<A> {
    pub(crate) fn new(field: A, slope: A) -> Self {
        Self { field, slope }
    }

    pub(crate) fn field(&self) -> &A {
        &self.field
    }

    pub(crate) fn slope(&self) -> &A {
        &self.slope
    }

    pub(crate) fn into_parts(self) -> (A, A) {
        (self.field, self.slope)
    }
}

/// The transfer states at the two boundaries of one finite layer.
///
/// The layer quantities are retained alongside the states so that the transfer
/// state can subsequently be decomposed into forward- and backward-propagating
/// amplitudes using the characteristic slope of that layer.
#[derive(Clone, Debug)]
pub(crate) struct LayerBoundaryStates<A> {
    left: TransferState<A>,
    right: TransferState<A>,
    quantities: IsotropicLayerQuantities<A>,
}

impl<A> LayerBoundaryStates<A> {
    pub(crate) fn new(
        left: TransferState<A>,
        right: TransferState<A>,
        quantities: IsotropicLayerQuantities<A>,
    ) -> Self {
        Self {
            left,
            right,
            quantities,
        }
    }

    pub(crate) fn left(&self) -> &TransferState<A> {
        &self.left
    }

    pub(crate) fn right(&self) -> &TransferState<A> {
        &self.right
    }

    pub(crate) fn quantities(&self) -> &IsotropicLayerQuantities<A> {
        &self.quantities
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        TransferState<A>,
        TransferState<A>,
        IsotropicLayerQuantities<A>,
    ) {
        (self.left, self.right, self.quantities)
    }

    /// Decompose the transfer states at both boundaries into forward- and
    /// backward-propagating amplitudes.
    pub(crate) fn into_boundary_waves(self) -> LayerBoundaryWaves<A>
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        let (left_state, right_state, quantities) = self.into_parts();

        let characteristic_slope = boundary_slope(&quantities.into_admittance().into_inner());

        let left = bidirectional_waves_from_state(&left_state, &characteristic_slope);

        let right = bidirectional_waves_from_state(&right_state, &characteristic_slope);

        LayerBoundaryWaves::new(left, right)
    }
}

/// The retained representation of one finite transfer layer.
#[derive(Clone, Debug)]
pub(crate) struct RetainedTransferLayer<A> {
    matrix: Transfer2Entries<A>,
    quantities: IsotropicLayerQuantities<A>,
}

impl<A> RetainedTransferLayer<A> {
    pub(crate) fn new(
        matrix: Transfer2Entries<A>,
        quantities: IsotropicLayerQuantities<A>,
    ) -> Self {
        Self { matrix, quantities }
    }

    pub(crate) fn matrix(&self) -> &Transfer2Entries<A> {
        &self.matrix
    }

    pub(crate) fn quantities(&self) -> &IsotropicLayerQuantities<A> {
        &self.quantities
    }
}

/// Finite layers retained during transfer-matrix accumulation.
///
/// Layers are stored in physical stack order, from left to right().
#[derive(Clone, Debug, Default)]
pub(crate) struct RetainedTransferLayers<A> {
    layers: Vec<RetainedTransferLayer<A>>,
}

impl<A> RetainedTransferLayers<A> {
    pub(crate) fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            layers: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn push(
        &mut self,
        matrix: Transfer2Entries<A>,
        quantities: IsotropicLayerQuantities<A>,
    ) {
        self.layers
            .push(RetainedTransferLayer::new(matrix, quantities));
    }

    pub(crate) fn len(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    pub(crate) fn layers(&self) -> &[RetainedTransferLayer<A>] {
        &self.layers
    }

    /// Propagate a known state at the right exterior boundary through every
    /// retained finite layer.
    ///
    /// Each layer matrix obeys
    ///
    /// ```text
    /// state_left = matrix * state_right().
    /// ```
    ///
    /// Layers are stored left-to-right, so propagation proceeds through them
    /// in reverse order. The returned records are restored to physical
    /// left-to-right stack order.
    pub(crate) fn propagate_right_state(
        &self,
        right_exterior_state: TransferState<A>,
    ) -> Vec<LayerBoundaryStates<A>>
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        let mut state_at_right = right_exterior_state;

        let mut reversed = Vec::with_capacity(self.layers.len());

        for layer in self.layers.iter().rev() {
            /*
             * Before applying this layer, the current state lies at the
             * layer's right boundary.
             */
            let right = state_at_right.clone();

            /*
             * The layer transfer matrix maps its right boundary state to its
             * left boundary state.
             */
            let left = layer.matrix.apply_state(&right);

            reversed.push(LayerBoundaryStates::new(
                left.clone(),
                right,
                layer.quantities.clone(),
            ));

            state_at_right = left;
        }

        /*
         * Propagation occurred right-to-left, but field consumers expect the
         * layers in physical left-to-right order.
         */
        reversed.reverse();

        reversed
    }

    /// Reconstruct forward- and backward-propagating waves at both boundaries
    /// of every retained finite layer.
    pub(crate) fn reconstruct_layer_boundary_waves(
        &self,
        right_exterior_state: TransferState<A>,
    ) -> Vec<LayerBoundaryWaves<A>>
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        self.propagate_right_state(right_exterior_state)
            .into_iter()
            .map(|states| states.into_boundary_waves())
            .collect()
    }
}

/// State accumulated while evaluating a transfer-matrix stack.
pub(crate) struct Transfer2Workspace<A> {
    solution: PlaneWaveSolution<Transfer2Entries<A>>,
    retained: Option<RetainedTransferLayers<A>>,
}

impl<A> PlaneWaveSolutionSource for Transfer2Workspace<A> {
    type Entries = Transfer2Entries<A>;

    fn solution(&self) -> PlaneWaveSolutionView<'_, Self::Entries> {
        self.solution.as_view()
    }
}

impl<A> SolutionWorkspace for Transfer2Workspace<A> {
    fn into_solution(self) -> PlaneWaveSolution<Self::Entries> {
        let (solution, ..) = self.into_parts();
        solution
    }
}

impl<A> Transfer2Workspace<A> {
    /// Construct a workspace
    pub(crate) fn new(
        source: &ArrayBase<OwnedRepr<A::Scalar>, A::Dimension>,
        context: Transfer2ExteriorContext<A>,
        mode: RunMode,
        layer_count: usize,
    ) -> Self
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexField + One + Zero,
    {
        Self {
            solution: PlaneWaveSolution::new(Transfer2Entries::identity_like(source), context),
            retained: None,
        }
    }

    /// Construct a workspace that retains every finite layer for subsequent
    /// field reconstruction.
    pub(crate) fn retaining_layers(
        entries: Transfer2Entries<A>,
        context: Transfer2ExteriorContext<A>,
    ) -> Self {
        Self {
            solution: PlaneWaveSolution::new(entries, context),
            retained: Some(RetainedTransferLayers::new()),
        }
    }

    /// Construct a retaining workspace with storage reserved for `capacity`
    /// finite layers.
    pub(crate) fn retaining_layers_with_capacity(
        entries: Transfer2Entries<A>,
        context: Transfer2ExteriorContext<A>,
        capacity: usize,
    ) -> Self {
        Self {
            solution: PlaneWaveSolution::new(entries, context),
            retained: Some(RetainedTransferLayers::with_capacity(capacity)),
        }
    }

    pub(crate) fn entries(&self) -> &Transfer2Entries<A> {
        self.solution.entries()
    }

    pub(crate) fn retained(&self) -> Option<&RetainedTransferLayers<A>> {
        self.retained.as_ref()
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        PlaneWaveSolution<Transfer2Entries<A>>,
        Option<RetainedTransferLayers<A>>,
    ) {
        (self.solution, self.retained)
    }

    pub(crate) fn retains_layers(&self) -> bool {
        self.retained.is_some()
    }

    /// Append a finite layer to the accumulated transfer matrix.
    ///
    /// Layers must be appended in physical left-to-right stack order. The
    /// resulting total matrix is therefore
    ///
    /// ```text
    /// M_total = M_0 M_1 ... M_n,
    /// ```
    ///
    /// and maps the right exterior state to the left exterior state.
    ///
    /// When retention is enabled, the layer matrix and its evaluated
    /// quantities are moved into retained storage after the total has been
    /// updated.
    pub(crate) fn append(
        &mut self,
        next: Transfer2Entries<A>,
        quantities: IsotropicLayerQuantities<A>,
    ) where
        A: ScalarAlgebra,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        let entries = self.solution.entries().multiply(&next);

        self.solution.replace_entries(entries);

        if let Some(retained) = &mut self.retained {
            retained.push(next, quantities);
        }
    }

    /// Propagate a known right exterior state through all retained layers.
    ///
    /// # Panics
    ///
    /// Panics if the workspace was constructed without layer retention.
    pub(crate) fn propagate_right_state(
        &self,
        right_exterior_state: TransferState<A>,
    ) -> Vec<LayerBoundaryStates<A>>
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        self.retained
            .as_ref()
            .expect("transfer layers were not retained")
            .propagate_right_state(right_exterior_state)
    }

    /// Reconstruct forward- and backward-propagating waves at both boundaries
    /// of every retained finite layer.
    ///
    /// # Panics
    ///
    /// Panics if the workspace was constructed without layer retention.
    pub(crate) fn reconstruct_layer_boundary_waves(
        &self,
        right_exterior_state: TransferState<A>,
    ) -> Vec<LayerBoundaryWaves<A>>
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        self.retained
            .as_ref()
            .expect("transfer layers were not retained")
            .reconstruct_layer_boundary_waves(right_exterior_state)
    }
}

impl<A> Transfer2Entries<A> {
    /// Apply this transfer matrix to a boundary state.
    ///
    /// For
    ///
    /// ```text
    ///     [m11 m12]
    /// M = [m21 m22],
    /// ```
    ///
    /// this computes
    ///
    /// ```text
    /// field_left = m11 field_right + m12 slope_right
    /// slope_left = m21 field_right + m22 slope_right().
    /// ```
    pub(crate) fn apply_state(&self, state: &TransferState<A>) -> TransferState<A>
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        let field = self
            .m11
            .multiply(state.field())
            .add(&self.m12.multiply(state.slope()));

        let slope = self
            .m21
            .multiply(state.field())
            .add(&self.m22.multiply(state.slope()));

        TransferState::new(field, slope)
    }
}

/// Decompose a transfer state into forward- and backward-propagating waves.
///
/// The directional state convention is
///
/// ```text
/// forward:  [1, -ξ]
/// backward: [1, +ξ].
/// ```
///
/// Consequently,
///
/// ```text
/// field = forward + backward
/// slope = ξ(backward - forward),
/// ```
///
/// and therefore
///
/// ```text
/// forward  = ½(field - slope / ξ)
/// backward = ½(field + slope / ξ).
/// ```
fn bidirectional_waves_from_state<A>(
    state: &TransferState<A>,
    characteristic_slope: &A,
) -> BidirectionalWaves<A>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexField + Copy,
    A::Dimension: Dimension,
{
    let slope_ratio = state.slope().divide(characteristic_slope);

    let half = (<A::Scalar as One>::one() + <A::Scalar as One>::one()).recip();

    let forward = state.field().subtract(&slope_ratio).scale(half);

    let backward = state.field().add(&slope_ratio).scale(half);

    BidirectionalWaves::new(forward, backward)
}

/// Construct a pure outgoing state at the right exterior boundary.
///
/// For right-outgoing amplitude `a` and exterior characteristic slope `ξ`,
/// the state is
///
/// ```text
/// [field, slope] = [a, -ξa].
/// ```
///
/// This helper is useful after the outgoing amplitudes have already been
/// normalised and phase-fixed.
pub(crate) fn right_outgoing_transfer_state<A>(
    right_outgoing: &A,
    right_admittance: &A,
) -> TransferState<A>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    let right_slope = boundary_slope(right_admittance);

    let slope = right_slope.multiply(right_outgoing).negate();

    TransferState::new(right_outgoing.clone(), slope)
}

/// Convert a physical characteristic admittance into the field-state slope
/// used by the transfer matrix.
///
/// For the matrix convention
///
/// ```text
/// M = [ cos(κd)    -sin(κd)/Y ]
///     [ Y sin(κd)   cos(κd)   ],
/// ```
///
/// travelling-wave states have derivative components `±iY`.
pub(super) fn boundary_slope<A>(admittance: &A) -> A
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    // TODO: Should be minus for outgoing
    let imaginary_unit =
        A::filled_constant_like(admittance.value(), -<A::Scalar as ComplexScalar>::i());

    imaginary_unit.multiply(admittance)
}
