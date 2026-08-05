use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, OwnedRepr};
use num_traits::{One, Zero};

use crate::{
    ComplexScalar, IncidentSide, PlaneWaveAmplitudes,
    algebra::ScalarAlgebra,
    backend::{
        BidirectionalWaves, ExteriorAdmittanceProvider, LayerBoundaryWaves, PlaneWaveSolution,
        PlaneWaveSolutionSource, RunMode, SolutionWorkspace,
        isotropic::IsotropicLayerQuantities,
        solution::PlaneWaveSolutionView,
        workspace::{ReconstructLayerBoundaryWaves, RetainedIsotropicLayers},
    },
};

use super::{
    Transfer2Entries,
    entries::Transfer2ExteriorContext,
    state::{TransferState, bidirectional_waves_from_state, transfer_state_from_waves},
};

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

    /// Decompose the transfer states at both layer boundaries into the
    /// layer-local forward- and backward-wave basis.
    pub(crate) fn into_boundary_waves(self) -> LayerBoundaryWaves<A>
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        let (left_state, right_state, quantities) = self.into_parts();

        let admittance = quantities.into_admittance().into_inner();

        let left = bidirectional_waves_from_state(&left_state, &admittance);

        let right = bidirectional_waves_from_state(&right_state, &admittance);

        LayerBoundaryWaves::new(left, right)
    }
}

/// The retained representation of one finite transfer layer.
#[derive(Clone, Debug)]
pub(crate) struct RetainedTransferLayer<A> {
    matrix: Transfer2Entries<A>,
    quantities: IsotropicLayerQuantities<A>,
    thickness: A,
}

impl<A> RetainedTransferLayer<A> {
    pub(crate) fn new(
        matrix: Transfer2Entries<A>,
        quantities: IsotropicLayerQuantities<A>,
        thickness: A,
    ) -> Self {
        Self {
            matrix,
            quantities,
            thickness,
        }
    }

    pub(crate) fn matrix(&self) -> &Transfer2Entries<A> {
        &self.matrix
    }

    pub(crate) fn quantities(&self) -> &IsotropicLayerQuantities<A> {
        &self.quantities
    }

    pub(crate) fn thickness(&self) -> &A {
        &self.thickness
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

    pub(crate) fn from_layers(layers: Vec<RetainedTransferLayer<A>>) -> Self {
        Self { layers }
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
        thickness: A,
    ) {
        self.layers
            .push(RetainedTransferLayer::new(matrix, quantities, thickness));
    }

    pub(crate) fn len(&self) -> usize {
        self.layers.len()
    }

    pub(crate) fn get_quantities(
        &self,
        layer_index: usize,
    ) -> Option<&IsotropicLayerQuantities<A>> {
        self.layers.get(layer_index).map(|layer| layer.quantities())
    }

    pub(crate) fn get_thickness(&self, layer_index: usize) -> Option<&A> {
        self.layers.get(layer_index).map(|layer| layer.thickness())
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
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct Transfer2Workspace<A> {
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

impl<A> RetainedIsotropicLayers for Transfer2Workspace<A> {
    type Algebra = A;
    fn retained_layer_count(&self) -> Option<usize> {
        self.retained.as_ref().map(|x| x.len())
    }

    fn layer_quantities(&self, index: usize) -> Option<&IsotropicLayerQuantities<Self::Algebra>> {
        self.retained.as_ref()?.get_quantities(index)
    }

    fn layer_thickness(&self, index: usize) -> Option<&Self::Algebra> {
        self.retained.as_ref()?.get_thickness(index)
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
            retained: mode
                .is_requested()
                .then(|| RetainedTransferLayers::with_capacity(layer_count)),
        }
    }

    pub(crate) fn from_parts(
        solution: PlaneWaveSolution<Transfer2Entries<A>>,
        retained: Option<RetainedTransferLayers<A>>,
    ) -> Self {
        Self { solution, retained }
    }

    pub(crate) fn solution(&self) -> &PlaneWaveSolution<Transfer2Entries<A>> {
        &self.solution
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
        thickness: A,
    ) where
        A: ScalarAlgebra,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        let entries = self.solution.entries().multiply(&next);

        self.solution.replace_entries(entries);

        if let Some(retained) = &mut self.retained {
            retained.push(next, quantities, thickness);
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

    fn sample_source(&self) -> &ArrayBase<OwnedRepr<A::Scalar>, A::Dimension>
    where
        A: ScalarAlgebra,
    {
        self.solution.entries().sample_source()
    }
}

impl<A> ReconstructLayerBoundaryWaves for Transfer2Workspace<A>
where
    A: ScalarAlgebra + Clone,
    A::Scalar: ComplexScalar + One + Zero,
    A::Dimension: Dimension,
{
    type Algebra = A;

    fn reconstruct_layer_boundary_waves(
        &self,
        incident_side: IncidentSide,
    ) -> Option<Vec<LayerBoundaryWaves<A>>> {
        let retained = self.retained.as_ref()?;

        let solution = self.solution();
        let amplitudes = solution.amplitudes(incident_side);

        let right_admittance = solution.context().right_admittance();

        let right_exterior_waves =
            right_exterior_waves(&amplitudes, incident_side, right_admittance.value());

        let right_exterior_state =
            transfer_state_from_waves(&right_exterior_waves, right_admittance);

        Some(retained.reconstruct_layer_boundary_waves(right_exterior_state))
    }
}

pub(crate) fn right_exterior_waves<A>(
    amplitudes: &PlaneWaveAmplitudes<A>,
    incident_side: IncidentSide,
    source: &ArrayBase<OwnedRepr<A::Scalar>, A::Dimension>,
) -> BidirectionalWaves<A>
where
    A: ScalarAlgebra,
    A::Scalar: Zero + One,
{
    match incident_side {
        IncidentSide::Left => {
            let zero = A::filled_constant_like(source, <A::Scalar as Zero>::zero());
            BidirectionalWaves::new(amplitudes.transmission().clone(), zero)
        }

        IncidentSide::Right => {
            let one = A::filled_constant_like(source, <A::Scalar as One>::one());
            BidirectionalWaves::new(amplitudes.reflection().clone(), one)
        }
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

#[cfg(test)]
mod tests {
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use super::*;
    use crate::{
        Constant, Polarisation, RealAxis,
        algebra::{ArrayJet0, RealParameter},
        backend::{RunMode, Transfer2},
        input::{CanonicalCoordinates, CanonicalStack},
        test_support::{
            C, TOLERANCE,
            assertions::{
                assert_bidirectional_waves_close, assert_complex_close, assert_zero_jet_close,
            },
            c,
            jet::{J0, zero_jet_from_real_value, zero_jet_from_value},
            materials::constant,
            planar::{
                boundary_test_empty_stack, boundary_test_jet, boundary_test_single_layer_stack,
                boundary_test_two_layer_stack, boundary_test_zero_thickness_stack,
            },
        },
    };

    type A = ArrayJet0<C, Ix0, RealParameter>;

    fn matrix(m11: f64, m12: f64, m21: f64, m22: f64) -> Transfer2Entries<A> {
        Transfer2Entries::new(
            zero_jet_from_value(c(m11)),
            zero_jet_from_value(c(m12)),
            zero_jet_from_value(c(m21)),
            zero_jet_from_value(c(m22)),
        )
    }

    fn coordinates() -> CanonicalCoordinates<A> {
        CanonicalCoordinates::new(zero_jet_from_value(c(2.0)), zero_jet_from_value(c(0.0)))
    }

    fn quantities() -> IsotropicLayerQuantities<A> {
        IsotropicLayerQuantities::real_axis(
            &constant(4.0, 1.0),
            &coordinates(),
            Polarisation::TransverseElectric,
        )
    }

    fn context() -> Transfer2ExteriorContext<A> {
        Transfer2ExteriorContext::new::<crate::domain::RealAxis, _>(
            &coordinates(),
            &constant(1.0, 1.0),
            &constant(1.0, 1.0),
            Polarisation::TransverseElectric,
        )
    }

    #[test]
    fn response_only_workspace_starts_at_identity() {
        let workspace = Transfer2Workspace::new(&arr0(c(0.0)), context(), RunMode::ResponseOnly, 2);

        assert!(!workspace.retains_layers());

        assert_complex_close(workspace.entries().m11()[()], c(1.0), TOLERANCE);
        assert_complex_close(workspace.entries().m12()[()], c(0.0), TOLERANCE);
        assert_complex_close(workspace.entries().m21()[()], c(0.0), TOLERANCE);
        assert_complex_close(workspace.entries().m22()[()], c(1.0), TOLERANCE);
    }

    #[test]
    fn internal_fields_workspace_retains_layers() {
        let workspace =
            Transfer2Workspace::new(&arr0(c(0.0)), context(), RunMode::InternalFields, 3);

        assert!(workspace.retains_layers());
        assert_eq!(workspace.retained().unwrap().len(), 0);
    }

    #[test]
    fn append_multiplies_layers_in_left_to_right_order() {
        let mut workspace =
            Transfer2Workspace::new(&arr0(c(0.0)), context(), RunMode::InternalFields, 2);

        let left = matrix(1.0, 2.0, 0.0, 1.0);
        let right = matrix(1.0, 0.0, 3.0, 1.0);

        workspace.append(left.clone(), quantities(), thickness(1.0));
        workspace.append(right.clone(), quantities(), thickness(1.0));

        assert_eq!(workspace.entries(), &left.multiply(&right),);

        assert_ne!(workspace.entries(), &right.multiply(&left),);
    }

    #[test]
    fn response_only_append_does_not_retain_layer() {
        let mut workspace =
            Transfer2Workspace::new(&arr0(c(0.0)), context(), RunMode::ResponseOnly, 1);

        workspace.append(matrix(1.0, 2.0, 3.0, 4.0), quantities(), thickness(1.0));

        assert!(workspace.retained().is_none());
    }

    #[test]
    fn retained_append_preserves_physical_layer_order() {
        let mut workspace =
            Transfer2Workspace::new(&arr0(c(0.0)), context(), RunMode::InternalFields, 2);

        let first = matrix(1.0, 2.0, 0.0, 1.0);
        let second = matrix(1.0, 0.0, 3.0, 1.0);

        workspace.append(first.clone(), quantities(), thickness(1.0));
        workspace.append(second.clone(), quantities(), thickness(1.0));

        let retained = workspace.retained().unwrap();

        assert_eq!(retained.len(), 2);
        assert_eq!(retained.layers()[0].matrix(), &first);
        assert_eq!(retained.layers()[1].matrix(), &second);
    }

    #[test]
    fn apply_state_uses_transfer_matrix_convention() {
        let matrix = matrix(1.0, 2.0, 3.0, 4.0);

        let state = TransferState::new(zero_jet_from_value(c(5.0)), zero_jet_from_value(c(7.0)));

        let result = matrix.apply_state(&state);

        assert_complex_close(result.field()[()], c(1.0 * 5.0 + 2.0 * 7.0), TOLERANCE);
        assert_complex_close(result.slope()[()], c(3.0 * 5.0 + 4.0 * 7.0), TOLERANCE);
    }

    #[test]
    fn retained_layers_propagate_from_right_to_left() {
        let mut retained = RetainedTransferLayers::new();

        let left = matrix(1.0, 2.0, 0.0, 1.0);
        let right = matrix(1.0, 0.0, 3.0, 1.0);

        retained.push(left.clone(), quantities(), thickness(1.0));
        retained.push(right.clone(), quantities(), thickness(1.0));

        let right_exterior =
            TransferState::new(zero_jet_from_value(c(5.0)), zero_jet_from_value(c(7.0)));

        let states = retained.propagate_right_state(right_exterior.clone());

        assert_eq!(states.len(), 2);

        let expected_right_layer_left = right.apply_state(&right_exterior);

        assert_eq!(states[1].right(), &right_exterior,);

        assert_eq!(states[1].left(), &expected_right_layer_left,);

        let expected_left_exterior = left.apply_state(&expected_right_layer_left);

        assert_eq!(states[0].right(), &expected_right_layer_left,);

        assert_eq!(states[0].left(), &expected_left_exterior,);
    }

    #[test]
    #[should_panic(expected = "transfer layers were not retained")]
    fn propagation_panics_without_retention() {
        let workspace = Transfer2Workspace::new(&arr0(c(0.0)), context(), RunMode::ResponseOnly, 0);

        workspace.propagate_right_state(TransferState::new(
            zero_jet_from_value(c(1.0)),
            zero_jet_from_value(c(0.0)),
        ));
    }

    fn test_coordinates() -> CanonicalCoordinates<J0> {
        CanonicalCoordinates::new(
            zero_jet_from_real_value(2.3),
            zero_jet_from_real_value(0.37),
        )
    }

    fn build_workspace(
        stack: CanonicalStack<Constant<f64>, J0>,
        mode: RunMode,
    ) -> Transfer2Workspace<J0> {
        Transfer2::new()
            .accumulate::<J0, RealAxis, _>(
                &test_coordinates(),
                &stack,
                Polarisation::TransverseElectric,
                mode,
            )
            .expect("scatter workspace accumulation should succeed")
    }

    #[test]
    fn response_only_workspace_does_not_reconstruct_boundary_waves() {
        let workspace = build_workspace(boundary_test_single_layer_stack(), RunMode::ResponseOnly);

        for side in [IncidentSide::Left, IncidentSide::Right] {
            assert!(workspace.reconstruct_layer_boundary_waves(side).is_none(),);
        }
    }

    #[test]
    fn retained_empty_stack_reconstructs_no_layers() {
        let workspace = build_workspace(boundary_test_empty_stack(), RunMode::InternalFields);

        for side in [IncidentSide::Left, IncidentSide::Right] {
            let waves = workspace
                .reconstruct_layer_boundary_waves(side)
                .expect("workspace retained internal data");

            assert!(waves.is_empty());
        }
    }

    #[test]
    fn reconstruction_returns_one_record_per_finite_layer() {
        let workspace = build_workspace(boundary_test_two_layer_stack(), RunMode::InternalFields);

        let waves = workspace
            .reconstruct_layer_boundary_waves(IncidentSide::Left)
            .unwrap();

        assert_eq!(waves.len(), 2);
    }

    #[test]
    fn zero_thickness_layer_has_equal_boundary_waves() {
        let workspace = build_workspace(
            boundary_test_zero_thickness_stack(),
            RunMode::InternalFields,
        );

        for side in [IncidentSide::Left, IncidentSide::Right] {
            let waves = workspace.reconstruct_layer_boundary_waves(side).unwrap();

            assert_eq!(waves.len(), 1);

            assert_bidirectional_waves_close(waves[0].left(), waves[0].right(), 1.0e-12);
        }
    }

    #[test]
    fn reconstruction_respects_incident_side() {
        let workspace = build_workspace(boundary_test_two_layer_stack(), RunMode::InternalFields);

        let left = workspace
            .reconstruct_layer_boundary_waves(IncidentSide::Left)
            .unwrap();

        let right = workspace
            .reconstruct_layer_boundary_waves(IncidentSide::Right)
            .unwrap();

        assert_ne!(left, right);
    }

    #[test]
    fn left_incidence_right_exterior_contains_only_transmission() {
        let transmission = boundary_test_jet(Complex64::new(0.7, 0.2));

        let reflection = boundary_test_jet(Complex64::new(-0.1, 0.3));

        let amplitudes = PlaneWaveAmplitudes::new(reflection, transmission.clone());

        let waves = right_exterior_waves(&amplitudes, IncidentSide::Left, transmission.value());

        assert_complex_close(
            waves.forward().value()[()],
            transmission.value()[()],
            1.0e-14,
        );

        assert_complex_close(
            waves.backward().value()[()],
            Complex64::new(0.0, 0.0),
            1.0e-14,
        );
    }

    #[test]
    fn right_incidence_right_exterior_contains_incident_and_reflected_waves() {
        let transmission = boundary_test_jet(Complex64::new(0.7, 0.2));

        let reflection = boundary_test_jet(Complex64::new(-0.1, 0.3));

        let amplitudes = PlaneWaveAmplitudes::new(reflection.clone(), transmission);

        let waves = right_exterior_waves(&amplitudes, IncidentSide::Right, reflection.value());

        assert_complex_close(waves.forward().value()[()], reflection.value()[()], 1.0e-14);

        assert_complex_close(
            waves.backward().value()[()],
            Complex64::new(1.0, 0.0),
            1.0e-14,
        );
    }

    fn thickness(thickness: f64) -> J0 {
        J0::new(arr0(c(thickness)))
    }

    fn quantities_for_material(material: Constant<f64>) -> IsotropicLayerQuantities<J0> {
        IsotropicLayerQuantities::evaluate::<RealAxis, _>(
            &material,
            &test_coordinates(),
            Polarisation::TransverseElectric,
        )
    }

    fn entries(m11: f64, m12: f64, m21: f64, m22: f64) -> Transfer2Entries<J0> {
        Transfer2Entries::new(
            zero_jet_from_real_value(m11),
            zero_jet_from_real_value(m12),
            zero_jet_from_real_value(m21),
            zero_jet_from_real_value(m22),
        )
    }

    #[test]
    fn propagated_boundary_states_are_returned_in_physical_layer_order() {
        /*
         * Physical stack:
         *
         * left exterior -> layer 0 -> layer 1 -> right exterior
         *
         * Each matrix maps:
         *
         *     state_left = matrix * state_right
         */
        let layer0_matrix = entries(1.0, 2.0, 0.0, 1.0);

        let layer1_matrix = entries(1.0, 0.0, 3.0, 1.0);

        let layer0_quantities = quantities_for_material(constant(2.25, 1.0));

        let layer1_quantities = quantities_for_material(constant(4.0, 1.0));

        let layer0_thickness = thickness(5.0);

        let layer1_thickness = thickness(8.0);

        let mut retained = RetainedTransferLayers::new();

        retained.push(layer0_matrix.clone(), layer0_quantities, layer0_thickness);

        retained.push(layer1_matrix.clone(), layer1_quantities, layer1_thickness);

        let right_exterior_state =
            TransferState::new(zero_jet_from_value(c(5.0)), zero_jet_from_value(c(7.0)));

        /*
         * Propagation occurs right-to-left:
         *
         *     right exterior
         *         ↓ layer 1
         *     interface between layers
         *         ↓ layer 0
         *     left exterior
         */
        let layer1_left = layer1_matrix.apply_state(&right_exterior_state);

        let layer0_left = layer0_matrix.apply_state(&layer1_left);

        let states = retained.propagate_right_state(right_exterior_state.clone());

        assert_eq!(states.len(), 2);

        /*
         * The returned vector must nevertheless remain in physical
         * left-to-right layer order.
         */
        let layer0 = &states[0];
        let layer1 = &states[1];

        assert_eq!(
            layer0.right(),
            &layer1_left,
            "layer 0 right boundary should be the state at the \
         interface between layers 0 and 1",
        );

        assert_eq!(
            layer0.left(),
            &layer0_left,
            "layer 0 left boundary should be the final propagated \
         left-exterior state",
        );

        assert_eq!(
            layer1.right(),
            &right_exterior_state,
            "layer 1 right boundary should be the supplied \
         right-exterior state",
        );

        assert_eq!(
            layer1.left(),
            &layer1_left,
            "layer 1 left boundary should be the state obtained by \
         applying the physical rightmost layer first",
        );

        /*
         * Adjacent records must share the same physical interface
         * state, even though they belong to different layer records.
         */
        assert_eq!(layer0.right(), layer1.left());
    }

    fn assert_transfer_state_close(actual: &TransferState<J0>, expected: &TransferState<J0>) {
        assert_zero_jet_close(actual.field(), expected.field());
        assert_zero_jet_close(actual.slope(), expected.slope());
    }

    #[test]
    fn one_layer_right_boundary_state_is_the_supplied_exterior_state() {
        let workspace =
            build_workspace(boundary_test_single_layer_stack(), RunMode::InternalFields);

        let solution = workspace.solution();

        let amplitudes = solution.amplitudes(IncidentSide::Left);

        let right_admittance = solution.context().right_admittance();

        let right_waves =
            right_exterior_waves(&amplitudes, IncidentSide::Left, right_admittance.value());

        let expected_right_state = transfer_state_from_waves(&right_waves, &right_admittance);

        let states = workspace
            .retained()
            .expect("layers should be retained")
            .propagate_right_state(expected_right_state.clone());

        assert_eq!(states.len(), 1);

        assert_transfer_state_close(states[0].right(), &expected_right_state);
    }

    #[test]
    fn retained_right_state_decomposes_with_retained_layer_admittance() {
        let workspace =
            build_workspace(boundary_test_single_layer_stack(), RunMode::InternalFields);

        let solution = workspace.solution();

        let amplitudes = solution.amplitudes(IncidentSide::Left);

        let right_admittance = solution.context().right_admittance();

        let exterior_waves =
            right_exterior_waves(&amplitudes, IncidentSide::Left, right_admittance.value());

        let right_state = transfer_state_from_waves(&exterior_waves, &right_admittance);

        let retained = workspace.retained().expect("layers should be retained");

        let layer = &retained.layers()[0];

        let layer_admittance = layer.quantities().clone().into_admittance().into_inner();

        let expected = bidirectional_waves_from_state(&right_state, &layer_admittance);

        let actual = retained.reconstruct_layer_boundary_waves(right_state);

        assert_eq!(actual.len(), 1);

        assert_bidirectional_waves_close(actual[0].right(), &expected, 1.0e-12);
    }

    #[test]
    fn retained_layer_admittance_matches_fresh_layer_evaluation() {
        let coordinates = test_coordinates();
        let stack = boundary_test_single_layer_stack();

        let workspace = build_workspace(stack.clone(), RunMode::InternalFields);

        let retained = workspace.retained().expect("layers should be retained");

        let retained_admittance = retained.layers()[0]
            .quantities()
            .clone()
            .into_admittance()
            .into_inner();

        let fresh_quantities = IsotropicLayerQuantities::evaluate::<RealAxis, _>(
            stack.layers()[0].material(),
            &coordinates,
            Polarisation::TransverseElectric,
        );

        let fresh_admittance = fresh_quantities.into_admittance().into_inner();

        assert_zero_jet_close(&retained_admittance, &fresh_admittance);
    }
}
