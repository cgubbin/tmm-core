use crate::{
    ComplexScalar,
    algebra::{
        ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, ScalarAlgebra,
    },
    backend::{
        BidirectionalWaves, LayerBoundaryWaves, PlaneWaveSolution, PlaneWaveSolutionSource,
        PlaneWaveSolutionView, RunMode, SolutionWorkspace,
        scatter2::{
            Scatter2ExteriorContext,
            entries::{Scatter2Entries, cascade},
        },
    },
    input::IncidentSide,
};

use ndarray::{ArrayBase, Dimension, OwnedRepr};
use num_traits::{One, Zero};

pub type Scatter2WorkspaceJet0<C, D, P> = Scatter2Workspace<ArrayJet0<C, D, P>>;

pub type Scatter2WorkspaceJet1<C, D, P> = Scatter2Workspace<ArrayJet1<C, D, P>>;

pub type Scatter2WorkspaceJet2<C, D, P> = Scatter2Workspace<ArrayJet2<C, D, P>>;

pub type Scatter2WorkspaceJetBivariate1<C, D, P> = Scatter2Workspace<ArrayJetBivariate1<C, D, P>>;

pub type Scatter2WorkspaceJetBivariate2<C, D, P> = Scatter2Workspace<ArrayJetBivariate2<C, D, P>>;

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
/// The workspace always accumulates `solution`. Individual components and layer
/// cut positions are retained only when internal fields were requested.
#[doc(hidden)]
pub struct Scatter2Workspace<A> {
    solution: PlaneWaveSolution<Scatter2Entries<A>>,
    retained: Option<RetainedScatterComponents<A>>,
}

impl<A> PlaneWaveSolutionSource for Scatter2Workspace<A> {
    type Entries = Scatter2Entries<A>;

    fn solution(&self) -> PlaneWaveSolutionView<'_, Self::Entries> {
        self.solution.as_view()
    }
}

impl<A> SolutionWorkspace for Scatter2Workspace<A> {
    fn into_solution(self) -> PlaneWaveSolution<Self::Entries> {
        let (solution, ..) = self.into_parts();
        solution
    }
}

pub(crate) struct RetainedScatterComponents<A> {
    pub(super) components: Vec<Scatter2Entries<A>>,
    pub(super) layer_cuts: Vec<LayerCutIndices>,
}

impl<A> Scatter2Workspace<A> {
    pub(crate) fn new(
        source: &ArrayBase<OwnedRepr<A::Scalar>, A::Dimension>,
        context: Scatter2ExteriorContext<A>,
        mode: RunMode,
        layer_count: usize,
    ) -> Self
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        Self {
            solution: PlaneWaveSolution::new(Scatter2Entries::identity_like(source), context),

            retained: mode.is_requested().then(|| RetainedScatterComponents {
                components: Vec::with_capacity(layer_count.saturating_mul(2).saturating_add(1)),
                layer_cuts: Vec::with_capacity(layer_count),
            }),
        }
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        PlaneWaveSolution<Scatter2Entries<A>>,
        Option<RetainedScatterComponents<A>>,
    ) {
        (self.solution, self.retained)
    }

    pub(crate) fn total(&self) -> &Scatter2Entries<A> {
        self.solution.entries()
    }

    pub(crate) fn append(&mut self, component: Scatter2Entries<A>)
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        let total = cascade(self.solution.entries(), &component);

        self.solution.replace_entries(total);

        if let Some(retained) = &mut self.retained {
            retained.components.push(component);
        }
    }

    pub(crate) fn append_layer(
        &mut self,
        interface: Scatter2Entries<A>,
        propagation: Scatter2Entries<A>,
    ) where
        A: ScalarAlgebra,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        self.append(interface);

        let left_cut = self
            .retained
            .as_ref()
            .map(|retained| retained.components.len());

        self.append(propagation);

        if let (Some(left_cut), Some(retained)) = (left_cut, &mut self.retained) {
            let right_cut = retained.components.len();

            retained
                .layer_cuts
                .push(LayerCutIndices::new(left_cut, right_cut));
        }
    }

    // pub(crate) fn reconstruct_layer_boundary_waves<C, D>(
    //     &self,
    //     incident_side: IncidentSide,
    //     source: &ArrayBase<OwnedRepr<C>, D>,
    // ) -> Option<Vec<LayerBoundaryWaves<A>>>
    // where
    //     C: ComplexScalar,
    //     D: Dimension,
    //     A: ScalarAlgebra<C, D> + Clone,
    // {
    //     Some(
    //         self.retained
    //             .as_ref()?
    //             .reconstruct_layer_boundary_waves(incident_side, source),
    //     )
    // }
}

impl<A> RetainedScatterComponents<A> {
    pub(crate) fn reconstruct_layer_boundary_waves(
        &self,
        incident_side: IncidentSide,
        source: &ArrayBase<OwnedRepr<A::Scalar>, A::Dimension>,
    ) -> Vec<LayerBoundaryWaves<A>>
    where
        A: ScalarAlgebra + Clone,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        let prefixes = prefix_cascades(&self.components, source);

        let suffixes = suffix_cascades(&self.components, source);

        /*
         * Incoming amplitudes are represented using the same algebra as the
         * scattering entries. For jets, both are constants: changing a stack or
         * spectral parameter does not change the imposed unit incident amplitude.
         */
        let zero = A::filled_constant_like(source, <A::Scalar as Zero>::zero());

        let one = A::filled_constant_like(source, <A::Scalar as One>::one());

        let (left_incoming, right_incoming) = match incident_side {
            IncidentSide::Left => (one, zero),

            IncidentSide::Right => (zero, one),
        };

        self.layer_cuts
            .iter()
            .map(|cuts| {
                let left_cut = cuts.left();
                let right_cut = cuts.right();

                let left = waves_at_cut(
                    &prefixes[left_cut],
                    &suffixes[left_cut],
                    &left_incoming,
                    &right_incoming,
                );

                let right = waves_at_cut(
                    &prefixes[right_cut],
                    &suffixes[right_cut],
                    &left_incoming,
                    &right_incoming,
                );

                LayerBoundaryWaves::new(left, right)
            })
            .collect()
    }

    fn map_entries<B>(
        self,
        mut map: impl FnMut(Scatter2Entries<A>) -> Scatter2Entries<B>,
    ) -> RetainedScatterComponents<B> {
        let components = self.components.into_iter().map(&mut map).collect();

        RetainedScatterComponents {
            components,
            layer_cuts: self.layer_cuts,
        }
    }
}

fn prefix_cascades<A>(
    components: &[Scatter2Entries<A>],
    source: &ArrayBase<OwnedRepr<A::Scalar>, A::Dimension>,
) -> Vec<Scatter2Entries<A>>
where
    A: ScalarAlgebra + Clone,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    let mut prefixes = Vec::with_capacity(components.len() + 1);

    prefixes.push(Scatter2Entries::identity_like(source));

    for component in components {
        let next = cascade(
            prefixes.last().expect("identity prefix was inserted"),
            component,
        );

        prefixes.push(next);
    }

    prefixes
}

fn suffix_cascades<A>(
    components: &[Scatter2Entries<A>],
    source: &ArrayBase<OwnedRepr<A::Scalar>, A::Dimension>,
) -> Vec<Scatter2Entries<A>>
where
    A: ScalarAlgebra + Clone,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    let component_count = components.len();

    let mut reversed = Vec::with_capacity(component_count + 1);

    reversed.push(Scatter2Entries::identity_like(source));

    for component in components.iter().rev() {
        let next = cascade(
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
fn waves_at_cut<A>(
    left: &Scatter2Entries<A>,
    right: &Scatter2Entries<A>,
    left_incoming: &A,
    right_incoming: &A,
) -> BidirectionalWaves<A>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    let one = A::filled_constant_like(left.s11.value(), <A::Scalar as One>::one());

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

    BidirectionalWaves::new(forward, backward)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array0, Ix0, arr0};
    use num_complex::Complex64;

    use super::{
        LayerCutIndices, RetainedScatterComponents, Scatter2Workspace, prefix_cascades,
        suffix_cascades, waves_at_cut,
    };

    use crate::{
        Polarisation, RealAxis,
        algebra::{ArrayJet0, ArrayJet1, RealParameter, ScalarAlgebra},
        backend::{
            RunMode, SolutionWorkspace,
            scatter2::{
                Scatter2ExteriorContext,
                entries::{Scatter2Entries, cascade},
            },
        },
        input::{CanonicalCoordinates, CanonicalStack, IncidentSide},
        material::{Constant, ConstitutiveLift},
        test_support::{
            C, TOLERANCE,
            assertions::{assert_array_close, assert_complex_close},
            c,
            jet::{J0, J1, P, zero_jet_from_array, zero_jet_from_value},
        },
    };

    type Entries0 = Scatter2Entries<J0>;
    type Entries1 = Scatter2Entries<J1>;

    fn scalar_entries(s11: C, s12: C, s21: C, s22: C) -> Entries0 {
        Scatter2Entries {
            s11: zero_jet_from_value(s11),
            s12: zero_jet_from_value(s12),
            s21: zero_jet_from_value(s21),
            s22: zero_jet_from_value(s22),
        }
    }

    fn transparent_component() -> Entries0 {
        scalar_entries(c(0.0), c(1.0), c(1.0), c(0.0))
    }

    fn first_component() -> Entries0 {
        scalar_entries(c(0.10), c(0.80), c(0.70), c(-0.20))
    }

    fn second_component() -> Entries0 {
        scalar_entries(c(0.30), c(0.60), c(0.50), c(-0.10))
    }

    fn third_component() -> Entries0 {
        scalar_entries(c(-0.08), c(0.83), c(0.79), c(0.06))
    }

    fn make_context<J>(source: &J) -> Scatter2ExteriorContext<J>
    where
        J: ScalarAlgebra<Scalar = C> + ConstitutiveLift<RealAxis, Constant<f64>>,
        J::Dimension: ndarray::Dimension,
    {
        let left_exterior = Constant::vacuum();
        let right_exterior = Constant::vacuum();

        let coordinates = CanonicalCoordinates::new(source.clone(), source.clone());

        let polarisation = Polarisation::TransverseMagnetic;

        Scatter2ExteriorContext::new::<RealAxis, _>(
            &coordinates,
            &left_exterior,
            &right_exterior,
            polarisation,
        )
    }

    fn assert_entries_close(actual: &Entries0, expected: &Entries0, tolerance: f64) {
        assert_array_close(actual.s11.value(), expected.s11.value(), tolerance);

        assert_array_close(actual.s12.value(), expected.s12.value(), tolerance);

        assert_array_close(actual.s21.value(), expected.s21.value(), tolerance);

        assert_array_close(actual.s22.value(), expected.s22.value(), tolerance);
    }

    #[test]
    fn layer_cut_indices_store_both_cuts() {
        let cuts = LayerCutIndices::new(2, 3);

        assert_eq!(cuts.left(), 2);
        assert_eq!(cuts.right(), 3);
    }

    #[test]
    fn new_workspace_starts_with_redheffer_identity() {
        let source = arr0(c(4.0));

        let context = make_context(&zero_jet_from_array(source.clone()));

        let workspace: Scatter2Workspace<J0> =
            Scatter2Workspace::new(&source, context, RunMode::ResponseOnly, 3);

        assert_complex_close(workspace.total().s11.value()[()], c(0.0), TOLERANCE);

        assert_complex_close(workspace.total().s12.value()[()], c(1.0), TOLERANCE);

        assert_complex_close(workspace.total().s21.value()[()], c(1.0), TOLERANCE);

        assert_complex_close(workspace.total().s22.value()[()], c(0.0), TOLERANCE);
    }

    #[test]
    fn response_only_workspace_does_not_retain_components() {
        let source = arr0(c(0.0));

        let context = make_context(&zero_jet_from_array(source.clone()));

        let workspace: Scatter2Workspace<J0> =
            Scatter2Workspace::new(&source, context, RunMode::ResponseOnly, 2);

        let (_, retained) = workspace.into_parts();

        assert!(retained.is_none());
    }

    #[test]
    fn internal_field_workspace_creates_retained_storage() {
        let source = arr0(c(0.0));

        let context = make_context(&zero_jet_from_array(source.clone()));

        let workspace: Scatter2Workspace<J0> =
            Scatter2Workspace::new(&source, context, RunMode::InternalFields, 2);

        let (_, retained) = workspace.into_parts();

        let retained = retained.expect("internal fields should retain components");

        assert!(retained.components.is_empty());
        assert!(retained.layer_cuts.is_empty());
    }

    #[test]
    fn entries_returns_accumulated_total() {
        let source = arr0(c(0.0));

        let context = make_context(&zero_jet_from_array(source.clone()));

        let mut workspace: Scatter2Workspace<J0> =
            Scatter2Workspace::new(&source, context, RunMode::ResponseOnly, 1);

        let component = first_component();

        workspace.append(component.clone());

        assert_entries_close(workspace.total(), &component, TOLERANCE);
    }

    #[test]
    fn into_entries_returns_accumulated_total() {
        let source = arr0(c(0.0));

        let context = make_context(&zero_jet_from_array(source.clone()));

        let mut workspace: Scatter2Workspace<J0> =
            Scatter2Workspace::new(&source, context, RunMode::ResponseOnly, 1);

        let component = first_component();

        workspace.append(component.clone());

        let solution = workspace.into_solution();
        let (entries, ..) = solution.into_parts();

        assert_entries_close(&entries, &component, TOLERANCE);
    }

    #[test]
    fn append_cascades_component_onto_total() {
        let source = arr0(c(0.0));

        let context = make_context(&zero_jet_from_array(source.clone()));

        let mut workspace: Scatter2Workspace<J0> =
            Scatter2Workspace::new(&source, context, RunMode::ResponseOnly, 2);

        let first = first_component();
        let second = second_component();

        workspace.append(first.clone());
        workspace.append(second.clone());

        let expected = cascade(&first, &second);

        assert_entries_close(workspace.total(), &expected, TOLERANCE);
    }

    #[test]
    fn append_retains_component_when_requested() {
        let source = arr0(c(0.0));

        let context = make_context(&zero_jet_from_array(source.clone()));

        let mut workspace: Scatter2Workspace<J0> =
            Scatter2Workspace::new(&source, context, RunMode::InternalFields, 1);

        let component = first_component();

        workspace.append(component.clone());

        let (_, retained) = workspace.into_parts();

        let retained = retained.expect("components should be retained");

        assert_eq!(retained.components.len(), 1);

        assert_entries_close(&retained.components[0], &component, TOLERANCE);
    }

    #[test]
    fn append_does_not_create_retention_in_response_only_mode() {
        let source = arr0(c(0.0));

        let context = make_context(&zero_jet_from_array(source.clone()));

        let mut workspace: Scatter2Workspace<J0> =
            Scatter2Workspace::new(&source, context, RunMode::ResponseOnly, 1);

        workspace.append(first_component());

        let (_, retained) = workspace.into_parts();

        assert!(retained.is_none());
    }

    #[test]
    fn append_layer_cascades_interface_then_propagation() {
        let source = arr0(c(0.0));

        let context = make_context(&zero_jet_from_array(source.clone()));

        let mut workspace: Scatter2Workspace<J0> =
            Scatter2Workspace::new(&source, context, RunMode::ResponseOnly, 1);

        let interface = first_component();
        let propagation = second_component();

        workspace.append_layer(interface.clone(), propagation.clone());

        let expected = cascade(&interface, &propagation);

        assert_entries_close(workspace.total(), &expected, TOLERANCE);
    }

    #[test]
    fn append_layer_retains_both_components() {
        let source = arr0(c(0.0));

        let context = make_context(&zero_jet_from_array(source.clone()));

        let mut workspace: Scatter2Workspace<J0> =
            Scatter2Workspace::new(&source, context, RunMode::InternalFields, 1);

        let interface = first_component();
        let propagation = second_component();

        workspace.append_layer(interface.clone(), propagation.clone());

        let (_, retained) = workspace.into_parts();

        let retained = retained.expect("layer components should be retained");

        assert_eq!(retained.components.len(), 2);

        assert_entries_close(&retained.components[0], &interface, TOLERANCE);

        assert_entries_close(&retained.components[1], &propagation, TOLERANCE);
    }

    #[test]
    fn append_layer_records_cuts_around_propagation() {
        let source = arr0(c(0.0));

        let context = make_context(&zero_jet_from_array(source.clone()));

        let mut workspace: Scatter2Workspace<J0> =
            Scatter2Workspace::new(&source, context, RunMode::InternalFields, 2);

        workspace.append_layer(first_component(), second_component());

        workspace.append_layer(third_component(), transparent_component());

        let (_, retained) = workspace.into_parts();

        let retained = retained.expect("layer cuts should be retained");

        assert_eq!(
            retained.layer_cuts,
            vec![LayerCutIndices::new(1, 2), LayerCutIndices::new(3, 4),],
        );
    }

    #[test]
    fn prefix_cascades_start_with_identity() {
        let source = arr0(c(0.0));

        let components = vec![first_component(), second_component()];

        let prefixes = prefix_cascades(&components, &source);

        assert_eq!(prefixes.len(), 3);

        let identity: Entries0 = Scatter2Entries::identity_like(&source);

        assert_entries_close(&prefixes[0], &identity, TOLERANCE);

        assert_entries_close(&prefixes[1], &components[0], TOLERANCE);

        let expected_total = cascade(&components[0], &components[1]);

        assert_entries_close(&prefixes[2], &expected_total, TOLERANCE);
    }

    #[test]
    fn suffix_cascades_end_with_identity() {
        let source = arr0(c(0.0));

        let components = vec![first_component(), second_component()];

        let suffixes = suffix_cascades(&components, &source);

        assert_eq!(suffixes.len(), 3);

        let identity: Entries0 = Scatter2Entries::identity_like(&source);

        assert_entries_close(&suffixes[2], &identity, TOLERANCE);

        assert_entries_close(&suffixes[1], &components[1], TOLERANCE);

        let expected_total = cascade(&components[0], &components[1]);

        assert_entries_close(&suffixes[0], &expected_total, TOLERANCE);
    }

    #[test]
    fn waves_at_cut_satisfy_coupled_scattering_equations() {
        let left = first_component();
        let right = second_component();

        let left_incoming = zero_jet_from_value(c(0.7));
        let right_incoming = zero_jet_from_value(c(-0.2));

        let waves = waves_at_cut(&left, &right, &left_incoming, &right_incoming);

        let forward = waves.forward();
        let backward = waves.backward();

        let expected_forward = left
            .s21
            .multiply(&left_incoming)
            .add(&left.s22.multiply(backward));

        let expected_backward = right
            .s11
            .multiply(forward)
            .add(&right.s12.multiply(&right_incoming));

        assert_array_close(forward.value(), expected_forward.value(), TOLERANCE);

        assert_array_close(backward.value(), expected_backward.value(), TOLERANCE);
    }

    #[test]
    fn transparent_cut_passes_left_incident_wave_forward() {
        let identity = transparent_component();

        let left_incoming = zero_jet_from_value(c(1.0));
        let right_incoming = zero_jet_from_value(c(0.0));

        let waves = waves_at_cut(&identity, &identity, &left_incoming, &right_incoming);

        assert_complex_close(waves.forward().value()[()], c(1.0), TOLERANCE);

        assert_complex_close(waves.backward().value()[()], c(0.0), TOLERANCE);
    }

    #[test]
    fn transparent_cut_passes_right_incident_wave_backward() {
        let identity = transparent_component();

        let left_incoming = zero_jet_from_value(c(0.0));
        let right_incoming = zero_jet_from_value(c(1.0));

        let waves = waves_at_cut(&identity, &identity, &left_incoming, &right_incoming);

        assert_complex_close(waves.forward().value()[()], c(0.0), TOLERANCE);

        assert_complex_close(waves.backward().value()[()], c(1.0), TOLERANCE);
    }

    #[test]
    fn reconstruction_returns_one_result_per_layer() {
        let source = arr0(c(0.0));

        let retained = RetainedScatterComponents {
            components: vec![
                first_component(),
                second_component(),
                third_component(),
                transparent_component(),
            ],
            layer_cuts: vec![LayerCutIndices::new(1, 2), LayerCutIndices::new(3, 4)],
        };

        let waves = retained.reconstruct_layer_boundary_waves(IncidentSide::Left, &source);

        assert_eq!(waves.len(), 2);
    }

    #[test]
    fn reconstructed_left_incident_waves_match_direct_cut_solution() {
        let source = arr0(c(0.0));

        let components = vec![first_component(), second_component()];

        let retained = RetainedScatterComponents {
            components: components.clone(),
            layer_cuts: vec![LayerCutIndices::new(1, 2)],
        };

        let reconstructed = retained.reconstruct_layer_boundary_waves(IncidentSide::Left, &source);

        let prefixes = prefix_cascades(&components, &source);
        let suffixes = suffix_cascades(&components, &source);

        let one = zero_jet_from_value(c(1.0));
        let zero = zero_jet_from_value(c(0.0));

        let expected_left = waves_at_cut(&prefixes[1], &suffixes[1], &one, &zero);

        let expected_right = waves_at_cut(&prefixes[2], &suffixes[2], &one, &zero);

        assert_array_close(
            reconstructed[0].left().forward().value(),
            expected_left.forward().value(),
            TOLERANCE,
        );

        assert_array_close(
            reconstructed[0].left().backward().value(),
            expected_left.backward().value(),
            TOLERANCE,
        );

        assert_array_close(
            reconstructed[0].right().forward().value(),
            expected_right.forward().value(),
            TOLERANCE,
        );

        assert_array_close(
            reconstructed[0].right().backward().value(),
            expected_right.backward().value(),
            TOLERANCE,
        );
    }

    #[test]
    fn reconstructed_right_incident_waves_match_direct_cut_solution() {
        let source = arr0(c(0.0));

        let components = vec![first_component(), second_component()];

        let retained = RetainedScatterComponents {
            components: components.clone(),
            layer_cuts: vec![LayerCutIndices::new(1, 2)],
        };

        let reconstructed = retained.reconstruct_layer_boundary_waves(IncidentSide::Right, &source);

        let prefixes = prefix_cascades(&components, &source);
        let suffixes = suffix_cascades(&components, &source);

        let zero = zero_jet_from_value(c(0.0));
        let one = zero_jet_from_value(c(1.0));

        let expected_left = waves_at_cut(&prefixes[1], &suffixes[1], &zero, &one);

        let expected_right = waves_at_cut(&prefixes[2], &suffixes[2], &zero, &one);

        assert_array_close(
            reconstructed[0].left().forward().value(),
            expected_left.forward().value(),
            TOLERANCE,
        );

        assert_array_close(
            reconstructed[0].left().backward().value(),
            expected_left.backward().value(),
            TOLERANCE,
        );

        assert_array_close(
            reconstructed[0].right().forward().value(),
            expected_right.forward().value(),
            TOLERANCE,
        );

        assert_array_close(
            reconstructed[0].right().backward().value(),
            expected_right.backward().value(),
            TOLERANCE,
        );
    }

    #[test]
    fn imposed_incident_wave_has_zero_derivative() {
        let source = arr0(c(0.0));

        let component = Scatter2Entries {
            s11: J1::from_parts(arr0(c(0.1)), arr0(c(0.02))),
            s12: J1::from_parts(arr0(c(0.8)), arr0(c(-0.03))),
            s21: J1::from_parts(arr0(c(0.7)), arr0(c(0.04))),
            s22: J1::from_parts(arr0(c(-0.2)), arr0(c(0.01))),
        };

        let retained = RetainedScatterComponents {
            components: vec![component],
            layer_cuts: vec![LayerCutIndices::new(0, 1)],
        };

        let waves = retained.reconstruct_layer_boundary_waves(IncidentSide::Left, &source);

        /*
         * At the exterior-left identity cut, the imposed forward amplitude
         * is exactly one and is independent of the differentiation
         * parameter.
         */
        assert_complex_close(waves[0].left().forward().value()[()], c(1.0), TOLERANCE);

        assert_complex_close(waves[0].left().forward().first()[()], c(0.0), TOLERANCE);
    }
}
