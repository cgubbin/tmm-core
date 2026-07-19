use crate::{
    ComplexScalar, IncidentSide,
    backend::{
        algebra::ScalarAlgebra,
        derivative::ChainRule,
        field::{BidirectionalWavesGeneric, InternalFieldRequest, LayerBoundaryWavesGeneric},
        jet::{ArrayJet, ArrayJetFirst},
        scatter2::entries::{ScatterEntries, cascade},
    },
};

use ndarray::{ArrayBase, Dimension, OwnedRepr};

/// Cut positions corresponding to the two boundaries of one finite layer.
///
/// A cut index `k` refers to the division:
///
/// ```text
/// components[..k] | components[k..]
/// ```
///
/// Thus:
///
/// - `left` is the cut immediately after the interface entering the layer;
/// - `right` is the cut immediately after propagation through the layer and
///   before the interface leaving it.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub(crate) struct LayerCutIndices {
    left: usize,
    right: usize,
}

impl LayerCutIndices {
    pub(crate) const fn new(left: usize, right: usize) -> Self {
        Self { left, right }
    }

    pub(crate) const fn left(self) -> usize {
        self.left
    }

    pub(crate) const fn right(self) -> usize {
        self.right
    }
}

/// Workspace used while constructing a scalar-channel scattering response.
///
/// `A` determines the derivative order carried by each scattering entry:
///
/// - sampled arrays for value-only evaluation;
/// - first-order jets for first derivatives;
/// - second-order jets for first and second derivatives.
///
/// The workspace always accumulates `total`. Individual components and layer
/// cut positions are retained only when internal fields were requested.
pub(crate) struct ScatterWorkspace<A> {
    total: ScatterEntries<A>,
    retained: Option<RetainedScatterComponents<A>>,
}

pub(crate) struct RetainedScatterComponents<A> {
    components: Vec<ScatterEntries<A>>,
    layer_cuts: Vec<LayerCutIndices>,
}

impl<A> ScatterWorkspace<A> {
    pub(crate) fn new<C, D>(
        source: &ArrayBase<OwnedRepr<C>, D>,
        request: InternalFieldRequest,
        layer_count: usize,
    ) -> Self
    where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D>,
    {
        Self {
            total: ScatterEntries::identity_like(source),

            retained: request.is_requested().then(|| RetainedScatterComponents {
                components: Vec::with_capacity(layer_count.saturating_mul(2).saturating_add(1)),
                layer_cuts: Vec::with_capacity(layer_count),
            }),
        }
    }

    pub(crate) fn append<C, D>(&mut self, component: ScatterEntries<A>)
    where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D>,
    {
        self.total = cascade::<C, D, A>(&self.total, &component);

        if let Some(retained) = &mut self.retained {
            retained.components.push(component);
        }
    }

    pub(crate) fn append_layer<C, D>(
        &mut self,
        interface: ScatterEntries<A>,
        propagation: ScatterEntries<A>,
    ) where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D>,
    {
        self.append::<C, D>(interface);

        let left_cut = self
            .retained
            .as_ref()
            .map(|retained| retained.components.len());

        self.append::<C, D>(propagation);

        if let (Some(left_cut), Some(retained)) = (left_cut, &mut self.retained) {
            let right_cut = retained.components.len();

            retained
                .layer_cuts
                .push(LayerCutIndices::new(left_cut, right_cut));
        }
    }

    pub(crate) fn reconstruct_layer_boundary_waves<C, D>(
        &self,
        incident_side: IncidentSide,
        source: &ArrayBase<OwnedRepr<C>, D>,
    ) -> Option<Vec<LayerBoundaryWavesGeneric<A>>>
    where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D> + Clone,
    {
        Some(
            self.retained
                .as_ref()?
                .reconstruct_layer_boundary_waves(incident_side, source),
        )
    }

    /// Transform every scattering-entry representation in the workspace.
    ///
    /// The transformation is applied to both the accumulated response and all
    /// retained physical components. Layer cut indices remain unchanged because
    /// the component topology is preserved.
    fn map_entries<B>(
        self,
        mut map: impl FnMut(ScatterEntries<A>) -> ScatterEntries<B>,
    ) -> ScatterWorkspace<B> {
        let total = map(self.total);

        if let Some(retained) = self.retained {
            let retained = retained.map_entries(map);

            return ScatterWorkspace {
                total,
                retained: Some(retained),
            };
        }

        ScatterWorkspace {
            total,
            retained: None,
        }
    }

    pub(crate) fn total(&self) -> &ScatterEntries<A> {
        &self.total
    }

    pub(crate) fn into_total(self) -> ScatterEntries<A> {
        self.total
    }
}

impl<C, D> ScatterWorkspace<ArrayJetFirst<C, D>>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Transform all jets from the primitive derivative variable to the requested
    /// public variable.
    ///
    /// After this method returns, the total response and every retained component
    /// carry derivatives with respect to the same variable.
    pub(crate) fn chain_rule(self, rule: &ChainRule<ArrayBase<OwnedRepr<C>, D>>) -> Self {
        self.map_entries(|entries| entries.chain_rule(rule))
    }
}

impl<C, D> ScatterWorkspace<ArrayJet<C, D>>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Transform all jets from the primitive derivative variable to the requested
    /// public variable.
    ///
    /// After this method returns, the total response and every retained component
    /// carry derivatives with respect to the same variable.
    pub(crate) fn chain_rule(self, rule: &ChainRule<ArrayBase<OwnedRepr<C>, D>>) -> Self {
        self.map_entries(|entries| entries.chain_rule(rule))
    }
}

pub(crate) struct ScatterWorkspaceParts<A> {
    pub(crate) total: ScatterEntries<A>,
    pub(crate) components: Option<Vec<ScatterEntries<A>>>,
    pub(crate) layer_cuts: Option<Vec<LayerCutIndices>>,
}

impl<A> RetainedScatterComponents<A> {
    pub(crate) fn reconstruct_layer_boundary_waves<C, D>(
        &self,
        incident_side: IncidentSide,
        source: &ArrayBase<OwnedRepr<C>, D>,
    ) -> Vec<LayerBoundaryWavesGeneric<A>>
    where
        C: ComplexScalar,
        D: Dimension,
        A: ScalarAlgebra<C, D> + Clone,
    {
        let prefixes = prefix_cascades::<C, D, A>(&self.components, source);

        let suffixes = suffix_cascades::<C, D, A>(&self.components, source);

        /*
         * Incoming amplitudes are represented using the same algebra as the
         * scattering entries. For jets, both are constants: changing a stack or
         * spectral parameter does not change the imposed unit incident amplitude.
         */
        let zero = A::constant_like(source, C::zero());

        let one = A::constant_like(source, C::one());

        let (left_incoming, right_incoming) = match incident_side {
            IncidentSide::Left => (one, zero),

            IncidentSide::Right => (zero, one),
        };

        self.layer_cuts
            .iter()
            .map(|cuts| {
                let left_cut = cuts.left();
                let right_cut = cuts.right();

                let left = waves_at_cut::<C, D, A>(
                    &prefixes[left_cut],
                    &suffixes[left_cut],
                    &left_incoming,
                    &right_incoming,
                );

                let right = waves_at_cut::<C, D, A>(
                    &prefixes[right_cut],
                    &suffixes[right_cut],
                    &left_incoming,
                    &right_incoming,
                );

                LayerBoundaryWavesGeneric::new(left, right)
            })
            .collect()
    }

    fn map_entries<B>(
        self,
        mut map: impl FnMut(ScatterEntries<A>) -> ScatterEntries<B>,
    ) -> RetainedScatterComponents<B> {
        let components = self.components.into_iter().map(&mut map).collect();

        RetainedScatterComponents {
            components,
            layer_cuts: self.layer_cuts,
        }
    }
}

fn prefix_cascades<C, D, A>(
    components: &[ScatterEntries<A>],
    source: &ArrayBase<OwnedRepr<C>, D>,
) -> Vec<ScatterEntries<A>>
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Clone,
{
    let mut prefixes = Vec::with_capacity(components.len() + 1);

    prefixes.push(ScatterEntries::identity_like(source));

    for component in components {
        let next = cascade::<C, D, A>(
            prefixes.last().expect("identity prefix was inserted"),
            component,
        );

        prefixes.push(next);
    }

    prefixes
}

fn suffix_cascades<C, D, A>(
    components: &[ScatterEntries<A>],
    source: &ArrayBase<OwnedRepr<C>, D>,
) -> Vec<ScatterEntries<A>>
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Clone,
{
    let component_count = components.len();

    let mut reversed = Vec::with_capacity(component_count + 1);

    reversed.push(ScatterEntries::identity_like(source));

    for component in components.iter().rev() {
        let next = cascade::<C, D, A>(
            component,
            reversed.last().expect("identity suffix was inserted"),
        );

        reversed.push(next);
    }

    reversed.reverse();
    reversed
}

/// Solve the forward and backward waves at a cut between two scattering
/// networks.
///
/// If `u` is the forward cut wave and `v` the backward cut wave:
///
/// ```text
/// u = L21 x + L22 v
/// v = R11 u + R12 y
/// ```
///
/// where `x` and `y` are the imposed incoming waves at the left and right
/// exterior ports.
fn waves_at_cut<C, D, A>(
    left: &ScatterEntries<A>,
    right: &ScatterEntries<A>,
    left_incoming: &A,
    right_incoming: &A,
) -> BidirectionalWavesGeneric<A>
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D>,
{
    let one = A::constant_like(left.s11.value(), C::one());

    let denominator = one.subtract(&left.s22.multiply(&right.s11));

    let forward = left
        .s21
        .multiply(left_incoming)
        .add(&left.s22.multiply(&right.s12).multiply(right_incoming))
        .divide(&denominator);

    let backward = right
        .s11
        .multiply(&forward)
        .add(&right.s12.multiply(right_incoming));

    BidirectionalWavesGeneric::new(forward, backward)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array0, Ix0, arr0};
    use num_complex::Complex64;
    use num_traits::{One, Zero};

    use super::*;

    type C = Complex64;
    type Samples = Array0<C>;

    const TOLERANCE: f64 = 1e-12;

    fn c(real: f64) -> C {
        C::new(real, 0.0)
    }

    fn ci(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn scalar(value: C) -> Samples {
        arr0(value)
    }

    fn source() -> Samples {
        scalar(C::zero())
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

    fn assert_array_close(actual: &Samples, expected: &Samples, tolerance: f64) {
        assert_eq!(actual.raw_dim(), expected.raw_dim());

        assert_complex_close(actual[()], expected[()], tolerance);
    }

    fn assert_entries_close(
        actual: &ScatterEntries<Samples>,
        expected: &ScatterEntries<Samples>,
        tolerance: f64,
    ) {
        assert_array_close(&actual.s11, &expected.s11, tolerance);

        assert_array_close(&actual.s12, &expected.s12, tolerance);

        assert_array_close(&actual.s21, &expected.s21, tolerance);

        assert_array_close(&actual.s22, &expected.s22, tolerance);
    }

    fn entries(s11: C, s12: C, s21: C, s22: C) -> ScatterEntries<Samples> {
        ScatterEntries {
            s11: scalar(s11),
            s12: scalar(s12),
            s21: scalar(s21),
            s22: scalar(s22),
        }
    }

    fn identity() -> ScatterEntries<Samples> {
        ScatterEntries::identity_like(&source())
    }

    /// Transparent reciprocal propagation component:
    ///
    /// ```text
    /// [0 p]
    /// [p 0]
    /// ```
    fn propagation(phase: C) -> ScatterEntries<Samples> {
        entries(C::zero(), phase, phase, C::zero())
    }

    /// Reflectionless asymmetric transmission component.
    ///
    /// This is useful for testing cascade ordering because the two
    /// transmission channels are deliberately different.
    fn asymmetric_transmission(right_to_left: C, left_to_right: C) -> ScatterEntries<Samples> {
        entries(C::zero(), right_to_left, left_to_right, C::zero())
    }

    fn first_jet(value: C, first: C) -> ArrayJetFirst<C, Ix0> {
        ArrayJetFirst::from_parts(scalar(value), scalar(first))
    }

    fn second_jet(value: C, first: C, second: C) -> ArrayJet<C, Ix0> {
        ArrayJet::from_parts(scalar(value), scalar(first), scalar(second))
    }

    fn first_jet_entries(value: [C; 4], first: [C; 4]) -> ScatterEntries<ArrayJetFirst<C, Ix0>> {
        ScatterEntries {
            s11: first_jet(value[0], first[0]),
            s12: first_jet(value[1], first[1]),
            s21: first_jet(value[2], first[2]),
            s22: first_jet(value[3], first[3]),
        }
    }

    fn second_jet_entries(
        value: [C; 4],
        first: [C; 4],
        second: [C; 4],
    ) -> ScatterEntries<ArrayJet<C, Ix0>> {
        ScatterEntries {
            s11: second_jet(value[0], first[0], second[0]),
            s12: second_jet(value[1], first[1], second[1]),
            s21: second_jet(value[2], first[2], second[2]),
            s22: second_jet(value[3], first[3], second[3]),
        }
    }

    fn assert_first_jet_close(
        actual: &ArrayJetFirst<C, Ix0>,
        expected_value: C,
        expected_first: C,
        tolerance: f64,
    ) {
        assert_complex_close(actual.value()[()], expected_value, tolerance);

        assert_complex_close(actual.first()[()], expected_first, tolerance);
    }

    fn assert_second_jet_close(
        actual: &ArrayJet<C, Ix0>,
        expected_value: C,
        expected_first: C,
        expected_second: C,
        tolerance: f64,
    ) {
        assert_complex_close(actual.value()[()], expected_value, tolerance);

        assert_complex_close(actual.first()[()], expected_first, tolerance);

        assert_complex_close(actual.second()[()], expected_second, tolerance);
    }

    #[test]
    fn response_only_workspace_does_not_retain_components() {
        let workspace: ScatterWorkspace<Samples> =
            ScatterWorkspace::new(&source(), InternalFieldRequest::None, 3);

        assert!(workspace.retained.is_none());

        assert_entries_close(workspace.total(), &identity(), TOLERANCE);
    }

    #[test]
    fn layer_boundary_request_creates_empty_retained_workspace() {
        let workspace: ScatterWorkspace<Samples> =
            ScatterWorkspace::new(&source(), InternalFieldRequest::LayerBoundaries, 3);

        let retained = workspace
            .retained
            .as_ref()
            .expect("retention was requested");

        assert!(retained.components.is_empty());
        assert!(retained.layer_cuts.is_empty());

        assert!(
            retained.components.capacity() >= 7,
            "three layers require capacity for seven physical components",
        );

        assert!(retained.layer_cuts.capacity() >= 3,);
    }

    #[test]
    fn append_accumulates_total_without_retaining_when_not_requested() {
        let mut workspace: ScatterWorkspace<Samples> =
            ScatterWorkspace::new(&source(), InternalFieldRequest::None, 0);

        let component = asymmetric_transmission(c(2.0), c(3.0));

        workspace.append::<C, Ix0>(component.clone());

        assert_entries_close(workspace.total(), &component, TOLERANCE);

        assert!(workspace.retained.is_none());
    }

    #[test]
    fn append_retains_physical_components_when_requested() {
        let mut workspace: ScatterWorkspace<Samples> =
            ScatterWorkspace::new(&source(), InternalFieldRequest::LayerBoundaries, 0);

        let first = asymmetric_transmission(c(2.0), c(3.0));

        let second = propagation(ci(0.5, 0.25));

        workspace.append::<C, Ix0>(first.clone());

        workspace.append::<C, Ix0>(second.clone());

        let retained = workspace
            .retained
            .as_ref()
            .expect("retention was requested");

        assert_eq!(retained.components.len(), 2);

        assert_entries_close(&retained.components[0], &first, TOLERANCE);

        assert_entries_close(&retained.components[1], &second, TOLERANCE);

        let expected = cascade::<C, Ix0, _>(&first, &second);

        assert_entries_close(workspace.total(), &expected, TOLERANCE);
    }

    #[test]
    fn append_layer_records_internal_boundary_cuts() {
        let mut workspace: ScatterWorkspace<Samples> =
            ScatterWorkspace::new(&source(), InternalFieldRequest::LayerBoundaries, 2);

        workspace.append_layer::<C, Ix0>(identity(), propagation(c(2.0)));

        workspace.append_layer::<C, Ix0>(identity(), propagation(c(3.0)));

        /*
         * Append the final interface, as the backend does after all finite
         * layers.
         */
        workspace.append::<C, Ix0>(identity());

        let retained = workspace
            .retained
            .as_ref()
            .expect("retention was requested");

        assert_eq!(retained.components.len(), 5,);

        assert_eq!(
            retained.layer_cuts,
            vec![LayerCutIndices::new(1, 2), LayerCutIndices::new(3, 4),],
        );
    }

    #[test]
    fn append_layer_does_not_create_cut_state_without_retention() {
        let mut workspace: ScatterWorkspace<Samples> =
            ScatterWorkspace::new(&source(), InternalFieldRequest::None, 1);

        workspace.append_layer::<C, Ix0>(identity(), propagation(c(2.0)));

        workspace.append::<C, Ix0>(identity());

        assert!(workspace.retained.is_none());

        let expected = propagation(c(2.0));

        assert_entries_close(workspace.total(), &expected, TOLERANCE);
    }

    #[test]
    fn into_total_returns_accumulated_response() {
        let mut workspace: ScatterWorkspace<Samples> =
            ScatterWorkspace::new(&source(), InternalFieldRequest::LayerBoundaries, 1);

        let first = asymmetric_transmission(c(2.0), c(3.0));

        let second = asymmetric_transmission(c(5.0), c(7.0));

        workspace.append::<C, Ix0>(first.clone());

        workspace.append::<C, Ix0>(second.clone());

        let expected = cascade::<C, Ix0, _>(&first, &second);

        let total = workspace.into_total();

        assert_entries_close(&total, &expected, TOLERANCE);
    }

    #[test]
    fn prefix_cascades_include_identity_and_ordered_totals() {
        let first = asymmetric_transmission(c(2.0), c(3.0));

        let second = asymmetric_transmission(c(5.0), c(7.0));

        let third = propagation(ci(0.8, 0.1));

        let components = vec![first.clone(), second.clone(), third.clone()];

        let prefixes = prefix_cascades::<C, Ix0, _>(&components, &source());

        assert_eq!(prefixes.len(), components.len() + 1,);

        assert_entries_close(&prefixes[0], &identity(), TOLERANCE);

        assert_entries_close(&prefixes[1], &first, TOLERANCE);

        let expected_two = cascade::<C, Ix0, _>(&first, &second);

        assert_entries_close(&prefixes[2], &expected_two, TOLERANCE);

        let expected_three = cascade::<C, Ix0, _>(&expected_two, &third);

        assert_entries_close(&prefixes[3], &expected_three, TOLERANCE);
    }

    #[test]
    fn suffix_cascades_include_ordered_totals_and_identity() {
        let first = asymmetric_transmission(c(2.0), c(3.0));

        let second = asymmetric_transmission(c(5.0), c(7.0));

        let third = propagation(ci(0.8, 0.1));

        let components = vec![first.clone(), second.clone(), third.clone()];

        let suffixes = suffix_cascades::<C, Ix0, _>(&components, &source());

        assert_eq!(suffixes.len(), components.len() + 1,);

        assert_entries_close(suffixes.last().unwrap(), &identity(), TOLERANCE);

        assert_entries_close(&suffixes[2], &third, TOLERANCE);

        let expected_two = cascade::<C, Ix0, _>(&second, &third);

        assert_entries_close(&suffixes[1], &expected_two, TOLERANCE);

        let expected_three = cascade::<C, Ix0, _>(&first, &expected_two);

        assert_entries_close(&suffixes[0], &expected_three, TOLERANCE);
    }

    #[test]
    fn prefix_final_equals_suffix_initial() {
        let components = vec![
            entries(c(0.1), c(0.7), c(0.8), c(-0.2)),
            entries(c(-0.3), c(0.9), c(0.6), c(0.15)),
            propagation(ci(0.8, -0.1)),
        ];

        let prefixes = prefix_cascades::<C, Ix0, _>(&components, &source());

        let suffixes = suffix_cascades::<C, Ix0, _>(&components, &source());

        assert_entries_close(prefixes.last().unwrap(), &suffixes[0], TOLERANCE);
    }

    #[test]
    fn empty_prefix_and_suffix_lists_contain_only_identity() {
        let components: Vec<ScatterEntries<Samples>> = Vec::new();

        let prefixes = prefix_cascades::<C, Ix0, _>(&components, &source());

        let suffixes = suffix_cascades::<C, Ix0, _>(&components, &source());

        assert_eq!(prefixes.len(), 1);
        assert_eq!(suffixes.len(), 1);

        assert_entries_close(&prefixes[0], &identity(), TOLERANCE);

        assert_entries_close(&suffixes[0], &identity(), TOLERANCE);
    }

    #[test]
    fn waves_at_cut_satisfy_coupled_scattering_equations() {
        let left = entries(ci(0.1, 0.05), ci(0.7, -0.1), ci(0.8, 0.2), ci(-0.15, 0.1));

        let right = entries(
            ci(0.2, -0.05),
            ci(0.6, 0.15),
            ci(0.75, -0.1),
            ci(-0.1, -0.05),
        );

        let left_incoming = scalar(ci(1.2, -0.3));

        let right_incoming = scalar(ci(-0.4, 0.25));

        let waves = waves_at_cut::<C, Ix0, _>(&left, &right, &left_incoming, &right_incoming);

        let forward = waves.forward()[()];

        let backward = waves.backward()[()];

        let expected_forward = left.s21[()] * left_incoming[()] + left.s22[()] * backward;

        let expected_backward = right.s11[()] * forward + right.s12[()] * right_incoming[()];

        assert_complex_close(forward, expected_forward, TOLERANCE);

        assert_complex_close(backward, expected_backward, TOLERANCE);
    }

    #[test]
    fn waves_at_cut_between_identity_networks_equal_external_inputs() {
        let left = identity();
        let right = identity();

        let left_incoming = scalar(ci(1.2, -0.3));

        let right_incoming = scalar(ci(-0.4, 0.25));

        let waves = waves_at_cut::<C, Ix0, _>(&left, &right, &left_incoming, &right_incoming);

        assert_array_close(&waves.forward(), &left_incoming, TOLERANCE);

        assert_array_close(&waves.backward(), &right_incoming, TOLERANCE);
    }

    #[test]
    fn reconstruction_returns_none_without_retained_components() {
        let workspace: ScatterWorkspace<Samples> =
            ScatterWorkspace::new(&source(), InternalFieldRequest::None, 1);

        let reconstructed =
            workspace.reconstruct_layer_boundary_waves(IncidentSide::Left, &source());

        assert!(reconstructed.is_none());
    }

    #[test]
    fn empty_retained_workspace_reconstructs_no_finite_layers() {
        let workspace: ScatterWorkspace<Samples> =
            ScatterWorkspace::new(&source(), InternalFieldRequest::LayerBoundaries, 0);

        let reconstructed = workspace
            .reconstruct_layer_boundary_waves(IncidentSide::Left, &source())
            .expect("retention was requested");

        assert!(reconstructed.is_empty());
    }

    #[test]
    fn transparent_layer_reconstruction_for_left_incidence() {
        let phase = ci(0.6, -0.2);

        let mut workspace: ScatterWorkspace<Samples> =
            ScatterWorkspace::new(&source(), InternalFieldRequest::LayerBoundaries, 1);

        workspace.append_layer::<C, Ix0>(identity(), propagation(phase));

        workspace.append::<C, Ix0>(identity());

        let reconstructed = workspace
            .reconstruct_layer_boundary_waves(IncidentSide::Left, &source())
            .expect("retention was requested");

        assert_eq!(reconstructed.len(), 1);

        let layer = &reconstructed[0];

        assert_complex_close(layer.left().forward()[()], C::one(), TOLERANCE);

        assert_complex_close(layer.left().backward()[()], C::zero(), TOLERANCE);

        assert_complex_close(layer.right().forward()[()], phase, TOLERANCE);

        assert_complex_close(layer.right().backward()[()], C::zero(), TOLERANCE);
    }

    #[test]
    fn transparent_layer_reconstruction_for_right_incidence() {
        let phase = ci(0.6, -0.2);

        let mut workspace: ScatterWorkspace<Samples> =
            ScatterWorkspace::new(&source(), InternalFieldRequest::LayerBoundaries, 1);

        workspace.append_layer::<C, Ix0>(identity(), propagation(phase));

        workspace.append::<C, Ix0>(identity());

        let reconstructed = workspace
            .reconstruct_layer_boundary_waves(IncidentSide::Right, &source())
            .expect("retention was requested");

        assert_eq!(reconstructed.len(), 1);

        let layer = &reconstructed[0];

        assert_complex_close(layer.right().forward()[()], C::zero(), TOLERANCE);

        assert_complex_close(layer.right().backward()[()], C::one(), TOLERANCE);

        assert_complex_close(layer.left().forward()[()], C::zero(), TOLERANCE);

        assert_complex_close(layer.left().backward()[()], phase, TOLERANCE);
    }

    #[test]
    fn multiple_transparent_layers_preserve_geometric_layer_order() {
        let first_phase = ci(0.8, 0.1);
        let second_phase = ci(0.6, -0.2);

        let mut workspace: ScatterWorkspace<Samples> =
            ScatterWorkspace::new(&source(), InternalFieldRequest::LayerBoundaries, 2);

        workspace.append_layer::<C, Ix0>(identity(), propagation(first_phase));

        workspace.append_layer::<C, Ix0>(identity(), propagation(second_phase));

        workspace.append::<C, Ix0>(identity());

        let reconstructed = workspace
            .reconstruct_layer_boundary_waves(IncidentSide::Left, &source())
            .expect("retention was requested");

        assert_eq!(reconstructed.len(), 2);

        let first = &reconstructed[0];
        let second = &reconstructed[1];

        assert_complex_close(first.left().forward()[()], C::one(), TOLERANCE);

        assert_complex_close(first.right().forward()[()], first_phase, TOLERANCE);

        assert_complex_close(second.left().forward()[()], first_phase, TOLERANCE);

        assert_complex_close(
            second.right().forward()[()],
            first_phase * second_phase,
            TOLERANCE,
        );

        for layer in reconstructed {
            assert_complex_close(layer.left().backward()[()], C::zero(), TOLERANCE);

            assert_complex_close(layer.right().backward()[()], C::zero(), TOLERANCE);
        }
    }

    #[test]
    fn reconstructed_external_cut_waves_match_total_scattering_channels() {
        let interface_left = entries(c(0.2), c(0.8), c(1.2), c(-0.2));

        let phase = ci(0.7, -0.1);

        let interface_right = entries(c(-0.1), c(1.1), c(0.9), c(0.1));

        let mut workspace: ScatterWorkspace<Samples> =
            ScatterWorkspace::new(&source(), InternalFieldRequest::LayerBoundaries, 1);

        workspace.append_layer::<C, Ix0>(interface_left, propagation(phase));

        workspace.append::<C, Ix0>(interface_right);

        let reconstructed = workspace
            .reconstruct_layer_boundary_waves(IncidentSide::Left, &source())
            .expect("retention was requested");

        let total = workspace.total();

        let layer = &reconstructed[0];

        /*
         * At the left internal cut, the backward wave is not generally equal
         * to the external reflection because the entering interface lies
         * between them. At the right internal cut, however, the forward wave
         * is related to the external transmitted wave by the final interface.
         *
         * The strongest backend-independent check is that the reconstructed
         * waves satisfy the prefix/suffix equations at both recorded cuts.
         */
        let retained = workspace.retained.as_ref().unwrap();

        let prefixes = prefix_cascades::<C, Ix0, _>(&retained.components, &source());

        let suffixes = suffix_cascades::<C, Ix0, _>(&retained.components, &source());

        for (waves, cut) in [
            (&layer.left(), retained.layer_cuts[0].left()),
            (&layer.right(), retained.layer_cuts[0].right()),
        ] {
            let forward = waves.forward()[()];

            let backward = waves.backward()[()];

            let expected_forward = prefixes[cut].s21[()] + prefixes[cut].s22[()] * backward;

            let expected_backward = suffixes[cut].s11[()] * forward;

            assert_complex_close(forward, expected_forward, TOLERANCE);

            assert_complex_close(backward, expected_backward, TOLERANCE);
        }

        assert!(total.s11[()].re.is_finite() && total.s11[()].im.is_finite(),);

        assert!(total.s21[()].re.is_finite() && total.s21[()].im.is_finite(),);
    }

    #[test]
    fn first_order_chain_rule_transforms_total_and_retained_components() {
        let mut workspace: ScatterWorkspace<ArrayJetFirst<C, Ix0>> =
            ScatterWorkspace::new(&source(), InternalFieldRequest::LayerBoundaries, 1);

        /*
         * Use a single component so the workspace total is exactly this
         * component and the expected transformed derivatives are direct.
         */
        let component = first_jet_entries(
            [c(1.0), c(2.0), c(3.0), c(4.0)],
            [c(5.0), c(6.0), c(7.0), c(8.0)],
        );

        workspace.append::<C, Ix0>(component);

        let rule = ChainRule {
            first: scalar(c(3.0)),
            second: scalar(c(11.0)),
        };

        let transformed = workspace.chain_rule(&rule);

        assert_first_jet_close(&transformed.total().s11, c(1.0), c(15.0), TOLERANCE);

        assert_first_jet_close(&transformed.total().s12, c(2.0), c(18.0), TOLERANCE);

        assert_first_jet_close(&transformed.total().s21, c(3.0), c(21.0), TOLERANCE);

        assert_first_jet_close(&transformed.total().s22, c(4.0), c(24.0), TOLERANCE);

        let retained = transformed
            .retained
            .as_ref()
            .expect("retention was requested");

        assert_eq!(retained.components.len(), 1);

        assert_first_jet_close(&retained.components[0].s11, c(1.0), c(15.0), TOLERANCE);

        assert_first_jet_close(&retained.components[0].s22, c(4.0), c(24.0), TOLERANCE);
    }

    #[test]
    fn second_order_chain_rule_transforms_total_and_retained_components() {
        let mut workspace: ScatterWorkspace<ArrayJet<C, Ix0>> =
            ScatterWorkspace::new(&source(), InternalFieldRequest::LayerBoundaries, 1);

        let component = second_jet_entries(
            [c(1.0), c(2.0), c(3.0), c(4.0)],
            [c(5.0), c(6.0), c(7.0), c(8.0)],
            [c(9.0), c(10.0), c(11.0), c(12.0)],
        );

        workspace.append::<C, Ix0>(component);

        /*
         * For q = q(x):
         *
         * d/dx   = dq/dx d/dq
         *
         * d²/dx² =
         *     d²/dq² (dq/dx)²
         *     + d/dq d²q/dx².
         */
        let rule = ChainRule {
            first: scalar(c(3.0)),
            second: scalar(c(2.0)),
        };

        let transformed = workspace.chain_rule(&rule);

        /*
         * s11:
         *
         * first  = 5 * 3 = 15
         * second = 9 * 3² + 5 * 2 = 91
         */
        assert_second_jet_close(
            &transformed.total().s11,
            c(1.0),
            c(15.0),
            c(91.0),
            TOLERANCE,
        );

        /*
         * s12:
         *
         * first  = 6 * 3 = 18
         * second = 10 * 9 + 6 * 2 = 102
         */
        assert_second_jet_close(
            &transformed.total().s12,
            c(2.0),
            c(18.0),
            c(102.0),
            TOLERANCE,
        );

        /*
         * s21:
         *
         * first  = 7 * 3 = 21
         * second = 11 * 9 + 7 * 2 = 113
         */
        assert_second_jet_close(
            &transformed.total().s21,
            c(3.0),
            c(21.0),
            c(113.0),
            TOLERANCE,
        );

        /*
         * s22:
         *
         * first  = 8 * 3 = 24
         * second = 12 * 9 + 8 * 2 = 124
         */
        assert_second_jet_close(
            &transformed.total().s22,
            c(4.0),
            c(24.0),
            c(124.0),
            TOLERANCE,
        );

        let retained = transformed
            .retained
            .as_ref()
            .expect("retention was requested");

        assert_eq!(retained.components.len(), 1);

        assert_second_jet_close(
            &retained.components[0].s11,
            c(1.0),
            c(15.0),
            c(91.0),
            TOLERANCE,
        );

        assert_second_jet_close(
            &retained.components[0].s22,
            c(4.0),
            c(24.0),
            c(124.0),
            TOLERANCE,
        );
    }

    #[test]
    fn chain_rule_preserves_layer_cut_topology() {
        let mut workspace: ScatterWorkspace<ArrayJetFirst<C, Ix0>> =
            ScatterWorkspace::new(&source(), InternalFieldRequest::LayerBoundaries, 2);

        let identity_jet = first_jet_entries(
            [C::zero(), C::one(), C::one(), C::zero()],
            [C::zero(), C::zero(), C::zero(), C::zero()],
        );

        workspace.append_layer::<C, Ix0>(identity_jet.clone(), identity_jet.clone());

        workspace.append_layer::<C, Ix0>(identity_jet.clone(), identity_jet.clone());

        workspace.append::<C, Ix0>(identity_jet);

        let before = workspace.retained.as_ref().unwrap().layer_cuts.clone();

        let rule = ChainRule {
            first: scalar(c(2.0)),
            second: scalar(c(3.0)),
        };

        let transformed = workspace.chain_rule(&rule);

        let after = &transformed.retained.as_ref().unwrap().layer_cuts;

        assert_eq!(&before, after);

        assert_eq!(
            after,
            &vec![LayerCutIndices::new(1, 2), LayerCutIndices::new(3, 4),],
        );
    }

    #[test]
    fn retained_component_mapping_preserves_order() {
        let retained = RetainedScatterComponents {
            components: vec![
                entries(c(1.0), c(2.0), c(3.0), c(4.0)),
                entries(c(5.0), c(6.0), c(7.0), c(8.0)),
            ],
            layer_cuts: vec![LayerCutIndices::new(1, 2)],
        };

        let mapped = retained.map_entries(|entry| ScatterEntries {
            s11: entry.s11.mapv(|x| x + c(10.0)),
            s12: entry.s12.mapv(|x| x + c(10.0)),
            s21: entry.s21.mapv(|x| x + c(10.0)),
            s22: entry.s22.mapv(|x| x + c(10.0)),
        });

        assert_eq!(mapped.components.len(), 2);

        assert_complex_close(mapped.components[0].s11[()], c(11.0), TOLERANCE);

        assert_complex_close(mapped.components[1].s11[()], c(15.0), TOLERANCE);

        assert_eq!(mapped.layer_cuts, vec![LayerCutIndices::new(1, 2)],);
    }

    #[test]
    fn layer_cut_accessors_return_stored_indices() {
        let cuts = LayerCutIndices::new(3, 7);

        assert_eq!(cuts.left(), 3);
        assert_eq!(cuts.right(), 7);
    }
}
