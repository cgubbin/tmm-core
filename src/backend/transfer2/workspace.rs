use ndarray::Dimension;

use crate::{
    ComplexScalar,
    backend::{
        algebra::ScalarAlgebra,
        derivative::ChainRule,
        field::{BidirectionalWavesGeneric, LayerBoundaryWavesGeneric},
        isotropic::IsotropicLayerQuantities,
        jet::{ArrayJet, ArrayJetFirst},
    },
};

use super::{matrix::Matrix2Entries, plane_wave::boundary_slope};

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
    pub(crate) fn into_boundary_waves<C, D>(self) -> LayerBoundaryWavesGeneric<A>
    where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D>,
    {
        let (left_state, right_state, quantities) = self.into_parts();

        let characteristic_slope = boundary_slope::<C, D, A>(&quantities.into_admittance());

        let left = bidirectional_waves_from_state::<C, D, A>(&left_state, &characteristic_slope);

        let right = bidirectional_waves_from_state::<C, D, A>(&right_state, &characteristic_slope);

        LayerBoundaryWavesGeneric::new(left, right)
    }
}

/// The retained representation of one finite transfer layer.
#[derive(Clone, Debug)]
pub(crate) struct RetainedTransferLayer<A> {
    matrix: Matrix2Entries<A>,
    quantities: IsotropicLayerQuantities<A>,
}

impl<A> RetainedTransferLayer<A> {
    pub(crate) fn new(matrix: Matrix2Entries<A>, quantities: IsotropicLayerQuantities<A>) -> Self {
        Self { matrix, quantities }
    }

    pub(crate) fn matrix(&self) -> &Matrix2Entries<A> {
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
        matrix: Matrix2Entries<A>,
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
    pub(crate) fn propagate_right_state<C, D>(
        &self,
        right_exterior_state: TransferState<A>,
    ) -> Vec<LayerBoundaryStates<A>>
    where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D> + Clone,
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
            let left = layer.matrix.apply_state::<C, D>(&right);

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
    pub(crate) fn reconstruct_layer_boundary_waves<C, D>(
        &self,
        right_exterior_state: TransferState<A>,
    ) -> Vec<LayerBoundaryWavesGeneric<A>>
    where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D> + Clone,
    {
        self.propagate_right_state::<C, D>(right_exterior_state)
            .into_iter()
            .map(|states| states.into_boundary_waves::<C, D>())
            .collect()
    }
}

/// State accumulated while evaluating a transfer-matrix stack.
#[derive(Clone, Debug)]
pub(crate) struct TransferWorkspace<A> {
    total: Matrix2Entries<A>,
    retained: Option<RetainedTransferLayers<A>>,
}

impl<A> TransferWorkspace<A> {
    /// Construct a workspace without retaining per-layer data.
    pub(crate) fn new(total: Matrix2Entries<A>) -> Self {
        Self {
            total,
            retained: None,
        }
    }

    /// Construct a workspace that retains every finite layer for subsequent
    /// field reconstruction.
    pub(crate) fn retaining_layers(total: Matrix2Entries<A>) -> Self {
        Self {
            total,
            retained: Some(RetainedTransferLayers::new()),
        }
    }

    /// Construct a retaining workspace with storage reserved for `capacity`
    /// finite layers.
    pub(crate) fn retaining_layers_with_capacity(
        total: Matrix2Entries<A>,
        capacity: usize,
    ) -> Self {
        Self {
            total,
            retained: Some(RetainedTransferLayers::with_capacity(capacity)),
        }
    }

    pub(crate) fn total(&self) -> &Matrix2Entries<A> {
        &self.total
    }

    pub(crate) fn into_total(self) -> Matrix2Entries<A> {
        self.total
    }

    pub(crate) fn retained(&self) -> Option<&RetainedTransferLayers<A>> {
        self.retained.as_ref()
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
    pub(crate) fn append<C, D>(
        &mut self,
        matrix: Matrix2Entries<A>,
        quantities: IsotropicLayerQuantities<A>,
    ) where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D> + Clone,
    {
        self.total = self.total.multiply::<C, D>(&matrix);

        if let Some(retained) = &mut self.retained {
            retained.push(matrix, quantities);
        }
    }

    /// Propagate a known right exterior state through all retained layers.
    ///
    /// # Panics
    ///
    /// Panics if the workspace was constructed without layer retention.
    pub(crate) fn propagate_right_state<C, D>(
        &self,
        right_exterior_state: TransferState<A>,
    ) -> Vec<LayerBoundaryStates<A>>
    where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D> + Clone,
    {
        self.retained
            .as_ref()
            .expect("transfer layers were not retained")
            .propagate_right_state::<C, D>(right_exterior_state)
    }

    /// Reconstruct forward- and backward-propagating waves at both boundaries
    /// of every retained finite layer.
    ///
    /// # Panics
    ///
    /// Panics if the workspace was constructed without layer retention.
    pub(crate) fn reconstruct_layer_boundary_waves<C, D>(
        &self,
        right_exterior_state: TransferState<A>,
    ) -> Vec<LayerBoundaryWavesGeneric<A>>
    where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D> + Clone,
    {
        self.retained
            .as_ref()
            .expect("transfer layers were not retained")
            .reconstruct_layer_boundary_waves::<C, D>(right_exterior_state)
    }
}

impl<C, D> TransferWorkspace<ArrayJetFirst<C, D>> {
    pub(crate) fn chain_rule(self, chain_rule: &ChainRule<ArrayBase<OwnedRepr<C>, D>>) -> Self
    where
        C: ComplexScalar,
        D: Dimension,
    {
        Self {
            total: self.total.chain_rule(chain_rule),

            retained: self
                .retained
                .map(|retained| retained.chain_rule(chain_rule)),
        }
    }
}

impl<C, D> TransferWorkspace<ArrayJet<C, D>> {
    pub(crate) fn chain_rule(self, chain_rule: &ChainRule<ArrayBase<OwnedRepr<C>, D>>) -> Self
    where
        C: ComplexScalar,
        D: Dimension,
    {
        Self {
            total: self.total.chain_rule(chain_rule),

            retained: self
                .retained
                .map(|retained| retained.chain_rule(chain_rule)),
        }
    }
}

impl<C, D> RetainedTransferLayers<ArrayJetFirst<C, D>> {
    pub(crate) fn chain_rule(self, chain_rule: &ChainRule<ArrayBase<OwnedRepr<C>, D>>) -> Self
    where
        C: ComplexScalar,
        D: Dimension,
    {
        Self {
            layers: self
                .layers
                .into_iter()
                .map(|layer| layer.chain_rule(chain_rule))
                .collect(),
        }
    }
}

impl<C, D> RetainedTransferLayers<ArrayJet<C, D>> {
    pub(crate) fn chain_rule(self, chain_rule: &ChainRule<ArrayBase<OwnedRepr<C>, D>>) -> Self
    where
        C: ComplexScalar,
        D: Dimension,
    {
        Self {
            layers: self
                .layers
                .into_iter()
                .map(|layer| layer.chain_rule(chain_rule))
                .collect(),
        }
    }
}

impl<C, D> RetainedTransferLayer<ArrayJetFirst<C, D>> {
    pub(crate) fn chain_rule(self, chain_rule: &ChainRule<ArrayBase<OwnedRepr<C>, D>>) -> Self
    where
        C: ComplexScalar,
        D: Dimension,
    {
        Self {
            matrix: self.matrix.chain_rule(chain_rule),

            quantities: self.quantities.chain_rule(chain_rule),
        }
    }
}

impl<C, D> RetainedTransferLayer<ArrayJet<C, D>> {
    pub(crate) fn chain_rule(self, chain_rule: &ChainRule<ArrayBase<OwnedRepr<C>, D>>) -> Self
    where
        C: ComplexScalar,
        D: Dimension,
    {
        Self {
            matrix: self.matrix.chain_rule(chain_rule),

            quantities: self.quantities.chain_rule(chain_rule),
        }
    }
}

use ndarray::{ArrayBase, OwnedRepr};

impl<A> Matrix2Entries<A> {
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
    pub(crate) fn apply_state<C, D>(&self, state: &TransferState<A>) -> TransferState<A>
    where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D>,
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
fn bidirectional_waves_from_state<C, D, A>(
    state: &TransferState<A>,
    characteristic_slope: &A,
) -> BidirectionalWavesGeneric<A>
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D>,
{
    let slope_ratio = state.slope().divide(characteristic_slope);

    let half = (C::one() + C::one()).recip();

    let forward = state.field().subtract(&slope_ratio).scale(half);

    let backward = state.field().add(&slope_ratio).scale(half);

    BidirectionalWavesGeneric::new(forward, backward)
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
pub(crate) fn right_outgoing_transfer_state<C, D, A>(
    right_outgoing: &A,
    right_admittance: &A,
) -> TransferState<A>
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Clone,
{
    let right_slope = boundary_slope::<C, D, A>(right_admittance);

    let slope = right_slope.multiply(right_outgoing).negate();

    TransferState::new(right_outgoing.clone(), slope)
}

#[cfg(test)]
mod tests {
    use crate::{Constant, PlanarInput, Polarisation};

    use super::*;

    use approx::assert_relative_eq;
    use ndarray::{Array0, arr0};
    use num_complex::Complex64;

    type C = Complex64;
    type A = Array0<C>;
    type D = ndarray::Ix0;

    const TOLERANCE: f64 = 1.0e-12;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn scalar(real: f64, imaginary: f64) -> A {
        arr0(c(real, imaginary))
    }

    fn assert_complex_close(actual: &A, expected: C) {
        assert_relative_eq!(actual[()].re, expected.re, epsilon = TOLERANCE,);

        assert_relative_eq!(actual[()].im, expected.im, epsilon = TOLERANCE,);
    }

    fn assert_state_close(actual: &TransferState<A>, expected_field: C, expected_slope: C) {
        assert_complex_close(actual.field(), expected_field);

        assert_complex_close(actual.slope(), expected_slope);
    }

    /*
     * Adapt this helper if Matrix2Entries uses a differently named
     * constructor.
     */
    fn matrix(m11: C, m12: C, m21: C, m22: C) -> Matrix2Entries<A> {
        Matrix2Entries::new(arr0(m11), arr0(m12), arr0(m21), arr0(m22))
    }

    fn identity_matrix() -> Matrix2Entries<A> {
        matrix(
            C::new(1.0, 0.0),
            C::new(0.0, 0.0),
            C::new(0.0, 0.0),
            C::new(1.0, 0.0),
        )
    }

    /*
     * Adapt this helper to the actual IsotropicLayerQuantities constructor.
     *
     * The important property for these tests is that:
     *
     *     boundary_slope(quantities.admittance()) == slope
     *
     * If boundary_slope performs an additional polarisation-dependent
     * transformation, construct the quantities so that the resulting slope
     * has the requested value.
     */
    fn quantities(slope: C) -> IsotropicLayerQuantities<A> {
        let material = Constant::vacuum();

        let quantities = IsotropicLayerQuantities::real_axis(
            &material,
            &PlanarInput::new(
                arr0(c(1.0, 0.0)),
                arr0(c(1.0, 0.0)),
                Polarisation::TransverseElectric,
            ),
        );

        quantities
    }

    fn state(field: C, slope: C) -> TransferState<A> {
        TransferState::new(arr0(field), arr0(slope))
    }

    #[test]
    fn append_without_retention_updates_only_total() {
        let first = matrix(c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0));

        let second = matrix(c(5.0, 0.0), c(6.0, 0.0), c(7.0, 0.0), c(8.0, 0.0));

        let expected = first.multiply::<C, D>(&second);

        let mut workspace = TransferWorkspace::new(first);

        workspace.append::<C, D>(second, quantities(c(1.0, 0.0)));

        assert!(workspace.retained().is_none());

        assert_complex_close(&workspace.total().m11, expected.m11[()]);

        assert_complex_close(&workspace.total().m12, expected.m12[()]);

        assert_complex_close(&workspace.total().m21, expected.m21[()]);

        assert_complex_close(&workspace.total().m22, expected.m22[()]);
    }

    #[test]
    fn append_with_retention_stores_matrix_and_quantities() {
        let layer_matrix = matrix(c(1.0, 0.0), c(2.0, 0.0), c(3.0, 0.0), c(4.0, 0.0));

        let layer_quantities = quantities(c(2.5, 0.0));

        let expected_matrix = layer_matrix.clone();

        let expected_admittance = layer_quantities.clone().into_admittance();

        let mut workspace = TransferWorkspace::retaining_layers(identity_matrix());

        workspace.append::<C, D>(layer_matrix, layer_quantities);

        let retained = workspace
            .retained()
            .expect("layers should have been retained");

        assert_eq!(retained.len(), 1);

        let retained_layer = &retained.layers()[0];

        assert_complex_close(&retained_layer.matrix().m11, expected_matrix.m11[()]);

        assert_complex_close(&retained_layer.matrix().m12, expected_matrix.m12[()]);

        assert_complex_close(&retained_layer.matrix().m21, expected_matrix.m21[()]);

        assert_complex_close(&retained_layer.matrix().m22, expected_matrix.m22[()]);

        let admittance = retained_layer.quantities().clone().into_admittance();
        assert_complex_close(&admittance, expected_admittance[()]);
    }

    #[test]
    fn propagate_identity_layer_preserves_state() {
        let mut workspace = TransferWorkspace::retaining_layers(identity_matrix());

        workspace.append::<C, D>(identity_matrix(), quantities(c(1.0, 0.0)));

        let right_exterior = state(c(2.0, -1.0), c(3.0, 4.0));

        let states = workspace.propagate_right_state::<C, D>(right_exterior);

        assert_eq!(states.len(), 1);

        assert_state_close(states[0].right(), c(2.0, -1.0), c(3.0, 4.0));

        assert_state_close(states[0].left(), c(2.0, -1.0), c(3.0, 4.0));
    }

    #[test]
    fn propagate_single_layer_matches_direct_matrix_application() {
        let layer_matrix = matrix(c(2.0, 0.0), c(1.0, 0.0), c(-1.0, 0.0), c(3.0, 0.0));

        let right = state(c(4.0, 1.0), c(-2.0, 3.0));

        let expected_left = layer_matrix.apply_state::<C, D>(&right);

        let mut workspace = TransferWorkspace::retaining_layers(identity_matrix());

        workspace.append::<C, D>(layer_matrix, quantities(c(1.0, 0.0)));

        let states = workspace.propagate_right_state::<C, D>(right);

        assert_eq!(states.len(), 1);

        assert_state_close(
            states[0].left(),
            expected_left.field()[()],
            expected_left.slope()[()],
        );
    }

    #[test]
    fn propagate_multiple_layers_walks_right_to_left_but_returns_stack_order() {
        /*
         * Use diagonal matrices so the expected propagation is transparent.
         *
         * Left-to-right physical order:
         *
         *     M0 = diag(2, 3)
         *     M1 = diag(5, 7)
         *     M2 = diag(11, 13)
         *
         * Reconstruction begins at the right of M2:
         *
         *     right(M2) = (1, 1)
         *     left(M2)  = (11, 13)
         *
         *     right(M1) = (11, 13)
         *     left(M1)  = (55, 91)
         *
         *     right(M0) = (55, 91)
         *     left(M0)  = (110, 273)
         */
        let m0 = matrix(c(2.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(3.0, 0.0));

        let m1 = matrix(c(5.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(7.0, 0.0));

        let m2 = matrix(c(11.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(13.0, 0.0));

        let mut workspace = TransferWorkspace::retaining_layers(identity_matrix());

        workspace.append::<C, D>(m0, quantities(c(1.0, 0.0)));

        workspace.append::<C, D>(m1, quantities(c(1.0, 0.0)));

        workspace.append::<C, D>(m2, quantities(c(1.0, 0.0)));

        let states = workspace.propagate_right_state::<C, D>(state(c(1.0, 0.0), c(1.0, 0.0)));

        assert_eq!(states.len(), 3);

        /*
         * Results must be returned in physical left-to-right order.
         */
        assert_state_close(states[0].right(), c(55.0, 0.0), c(91.0, 0.0));

        assert_state_close(states[0].left(), c(110.0, 0.0), c(273.0, 0.0));

        assert_state_close(states[1].right(), c(11.0, 0.0), c(13.0, 0.0));

        assert_state_close(states[1].left(), c(55.0, 0.0), c(91.0, 0.0));

        assert_state_close(states[2].right(), c(1.0, 0.0), c(1.0, 0.0));

        assert_state_close(states[2].left(), c(11.0, 0.0), c(13.0, 0.0));
    }

    #[test]
    fn decomposes_pure_forward_state() {
        let characteristic_slope = scalar(2.0, 0.0);

        /*
         * Pure forward wave with amplitude one:
         *
         *     field = 1
         *     slope = -ξ = -2
         */
        let transfer_state = state(c(1.0, 0.0), c(-2.0, 0.0));

        let waves =
            bidirectional_waves_from_state::<C, D, A>(&transfer_state, &characteristic_slope);

        assert_complex_close(&waves.forward(), c(1.0, 0.0));

        assert_complex_close(&waves.backward(), c(0.0, 0.0));
    }

    #[test]
    fn decomposes_pure_backward_state() {
        let characteristic_slope = scalar(2.0, 0.0);

        /*
         * Pure backward wave with amplitude one:
         *
         *     field = 1
         *     slope = +ξ = 2
         */
        let transfer_state = state(c(1.0, 0.0), c(2.0, 0.0));

        let waves =
            bidirectional_waves_from_state::<C, D, A>(&transfer_state, &characteristic_slope);

        assert_complex_close(&waves.forward(), c(0.0, 0.0));

        assert_complex_close(&waves.backward(), c(1.0, 0.0));
    }

    #[test]
    fn decomposition_recovers_arbitrary_forward_and_backward_waves() {
        let forward = c(0.3, 0.4);

        let backward = c(-0.7, 0.2);

        let characteristic_slope = c(2.0, -0.5);

        /*
         * field = forward + backward
         *
         * slope = ξ(backward - forward)
         */
        let field = forward + backward;

        let slope = characteristic_slope * (backward - forward);

        let transfer_state = state(field, slope);

        let waves =
            bidirectional_waves_from_state::<C, D, A>(&transfer_state, &arr0(characteristic_slope));

        assert_complex_close(&waves.forward(), forward);

        assert_complex_close(&waves.backward(), backward);
    }

    #[test]
    #[should_panic(expected = "transfer layers were not retained")]
    fn reconstruction_panics_when_layers_were_not_retained() {
        let workspace = TransferWorkspace::new(identity_matrix());

        let _ =
            workspace.reconstruct_layer_boundary_waves::<C, D>(state(c(1.0, 0.0), c(-1.0, 0.0)));
    }
}
