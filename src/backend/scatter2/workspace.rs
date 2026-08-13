use crate::{
    ComplexScalar,
    algebra::{
        ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, ScalarAlgebra,
    },
    backend::{
        ExteriorContextProvider, IsotropicLayerQuantities, ModalSolutionSource,
        ModeReconstructionError, PlaneWaveModeCandidate, PlaneWaveSolution,
        PlaneWaveSolutionSource, PlaneWaveSolutionView, ReconstructExteriorModeWaves,
        ReconstructLayerModeWaves, RetainedIsotropicLayers, RunMode, SolutionWorkspace,
        scatter2::{
            Scatter2ExteriorContext, Scatter2ProjectiveEntries, cascade_projection,
            entries::{Scatter2Entries, cascade},
        },
    },
    input::IncidentSide,
    observable::BoundaryState,
    waves::{
        BidirectionalWaves, ExteriorBoundaryWaves, LayerBoundaryWaves,
        ReconstructLayerBoundaryWaves,
    },
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

/// Regularized projective data for the right-outgoing modal chart.
///
/// The physical scattering entries satisfy:
///
/// ```text
/// s11 = left_reflection_numerator / denominator
/// s21 = transmission_numerator    / denominator
/// ```
///
/// The modal candidate is represented projectively, so no division by the
/// transmission numerator is required.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Scatter2ModalData<A> {
    denominator: A,
    left_reflection_numerator: A,
    transmission_numerator: A,
}

impl<A> Scatter2ModalData<A> {
    pub(crate) const fn new(
        denominator: A,
        left_reflection_numerator: A,
        transmission_numerator: A,
    ) -> Self {
        Self {
            denominator,
            left_reflection_numerator,
            transmission_numerator,
        }
    }

    pub(crate) fn denominator(&self) -> &A {
        &self.denominator
    }

    pub(crate) fn left_reflection_numerator(&self) -> &A {
        &self.left_reflection_numerator
    }

    pub(crate) fn transmission_numerator(&self) -> &A {
        &self.transmission_numerator
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
#[derive(Debug)]
pub struct Scatter2Workspace<A> {
    solution: PlaneWaveSolution<Scatter2ProjectiveEntries<A>>,
    retained: Option<RetainedScatterComponents<A>>,
}

impl<A> PlaneWaveSolutionSource for Scatter2Workspace<A> {
    type Entries = Scatter2ProjectiveEntries<A>;

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

impl<A> RetainedIsotropicLayers for Scatter2Workspace<A> {
    type Algebra = A;

    fn retained_layer_count(&self) -> Option<usize> {
        self.retained.as_ref().map(|x| x.num_layers())
    }

    fn layer_quantities(&self, index: usize) -> Option<&IsotropicLayerQuantities<Self::Algebra>> {
        self.retained.as_ref()?.get_quantities(index)
    }

    fn layer_thickness(&self, index: usize) -> Option<&Self::Algebra> {
        self.retained.as_ref()?.get_thickness(index)
    }
}

#[derive(Debug)]
pub(crate) struct RetainedScatterComponents<A> {
    pub(super) components: Vec<Scatter2Entries<A>>,
    pub(super) layer_cuts: Vec<LayerCutIndices>,
    pub(super) quantities: Vec<IsotropicLayerQuantities<A>>,
    pub(super) thicknesses: Vec<A>,
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
        let entries = Scatter2ProjectiveEntries::identity_like(source);

        Self {
            solution: PlaneWaveSolution::new(entries, context),
            retained: mode.is_requested().then(|| RetainedScatterComponents {
                components: Vec::with_capacity(layer_count.saturating_mul(2).saturating_add(1)),
                layer_cuts: Vec::with_capacity(layer_count),
                quantities: Vec::with_capacity(layer_count),
                thicknesses: Vec::with_capacity(layer_count),
            }),
        }
    }

    pub(crate) fn from_parts(
        solution: PlaneWaveSolution<Scatter2ProjectiveEntries<A>>,
        retained: Option<RetainedScatterComponents<A>>,
    ) -> Self {
        Self { solution, retained }
    }

    pub(crate) fn into_parts(
        self,
    ) -> (
        PlaneWaveSolution<Scatter2ProjectiveEntries<A>>,
        Option<RetainedScatterComponents<A>>,
    ) {
        (self.solution, self.retained)
    }

    pub(crate) fn solution(&self) -> &PlaneWaveSolution<Scatter2ProjectiveEntries<A>> {
        &self.solution
    }

    pub(crate) fn entries(&self) -> &Scatter2ProjectiveEntries<A> {
        self.solution.entries()
    }

    fn total(&self) -> Scatter2Entries<A>
    where
        A: ScalarAlgebra,
    {
        self.solution.entries().entries()
    }

    pub(crate) fn total_projection(&self) -> Scatter2Entries<A>
    where
        A: ScalarAlgebra,
    {
        self.solution.entries().entries()
    }

    pub(crate) fn retained(&self) -> Option<&RetainedScatterComponents<A>> {
        self.retained.as_ref()
    }

    pub(crate) fn append(&mut self, component: Scatter2Entries<A>)
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexScalar + One,
        A::Dimension: Dimension,
    {
        let component_projection = Scatter2ProjectiveEntries::from_entries(&component);

        let total = cascade_projection(self.solution.entries(), &component_projection);

        self.solution.replace_entries(total);

        if let Some(retained) = &mut self.retained {
            retained.components.push(component);
        }
    }

    pub(crate) fn append_layer(
        &mut self,
        interface: Scatter2Entries<A>,
        propagation: Scatter2Entries<A>,
        quantities: IsotropicLayerQuantities<A>,
        thickness: A,
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

        if let Some(retained) = &mut self.retained {
            retained.quantities.push(quantities);
            retained.thicknesses.push(thickness);
        }
    }

    fn sample_source(&self) -> &ArrayBase<OwnedRepr<A::Scalar>, A::Dimension>
    where
        A: ScalarAlgebra,
    {
        self.solution.entries().sample_source()
    }
}

impl<A> ReconstructLayerBoundaryWaves for Scatter2Workspace<A>
where
    A: ScalarAlgebra + Clone,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    type Algebra = A;

    fn reconstruct_layer_boundary_waves(
        &self,
        incident_side: IncidentSide,
    ) -> Option<Vec<LayerBoundaryWaves<A>>> {
        Some(
            self.retained
                .as_ref()?
                .reconstruct_layer_boundary_waves(incident_side, self.sample_source()),
        )
    }
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
        let zero = A::filled_constant_like(source, A::Scalar::zero());

        let one = A::filled_constant_like(source, A::Scalar::one());

        let (left_incoming, right_incoming) = match incident_side {
            IncidentSide::Left => (one, zero),
            IncidentSide::Right => (zero, one),
        };

        self.reconstruct_from_incoming_waves(&left_incoming, &right_incoming, source)
    }

    fn get_quantities(&self, layer_index: usize) -> Option<&IsotropicLayerQuantities<A>> {
        self.quantities.get(layer_index)
    }

    fn get_thickness(&self, layer_index: usize) -> Option<&A> {
        self.thicknesses.get(layer_index)
    }

    fn num_layers(&self) -> usize {
        self.quantities.len()
    }

    pub(crate) fn components(&self) -> &Vec<Scatter2Entries<A>> {
        &self.components
    }

    pub(crate) fn layer_cuts(&self) -> &Vec<LayerCutIndices> {
        &self.layer_cuts
    }

    pub(crate) fn quantities(&self) -> &Vec<IsotropicLayerQuantities<A>> {
        &self.quantities
    }

    pub(crate) fn thicknesses(&self) -> &Vec<A> {
        &self.thicknesses
    }

    pub(crate) fn from_parts(
        components: Vec<Scatter2Entries<A>>,
        layer_cuts: Vec<LayerCutIndices>,
        quantities: Vec<IsotropicLayerQuantities<A>>,
        thicknesses: Vec<A>,
    ) -> Self {
        Self {
            components,
            layer_cuts,
            quantities,
            thicknesses,
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

// fn suffix_cascades<A>(
//     components: &[Scatter2Entries<A>],
//     source: &ArrayBase<OwnedRepr<A::Scalar>, A::Dimension>,
// ) -> Vec<Scatter2Entries<A>>
// where
//     A: ScalarAlgebra + Clone,
//     A::Scalar: ComplexScalar,
//     A::Dimension: Dimension,
// {
//     let component_count = components.len();

//     let mut reversed = Vec::with_capacity(component_count + 1);

//     reversed.push(Scatter2Entries::identity_like(source));

//     for component in components.iter().rev() {
//         let next = cascade(
//             component,
//             reversed.last().expect("identity suffix was inserted"),
//         );

//         reversed.push(next);
//     }

//     reversed.reverse();
//     reversed
// }
fn suffix_cascades<A>(
    components: &[Scatter2Entries<A>],
    source: &ArrayBase<OwnedRepr<A::Scalar>, A::Dimension>,
) -> Vec<Scatter2Entries<A>>
where
    A: ScalarAlgebra + Clone,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    let mut reversed = Vec::with_capacity(components.len() + 1);

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
    prefix: &Scatter2Entries<A>,
    suffix: &Scatter2Entries<A>,
    left_incoming: &A,
    right_incoming: &A,
) -> BidirectionalWaves<A>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar + One,
    A::Dimension: Dimension,
{
    let one = A::filled_constant_like(left_incoming.value(), <A::Scalar as One>::one());

    let suffix_from_right = suffix.s12().multiply(right_incoming);

    let forward_numerator = prefix
        .s21()
        .multiply(left_incoming)
        .add(&prefix.s22().multiply(&suffix_from_right));

    let denominator = one.subtract(&prefix.s22().multiply(suffix.s11()));

    let forward = forward_numerator.divide(&denominator);

    let backward = suffix.s11().multiply(&forward).add(&suffix_from_right);

    BidirectionalWaves::new(forward, backward)
}

impl<A> ModalSolutionSource for Scatter2Workspace<A>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar + One,
    A::Dimension: Dimension,
{
    type Algebra = A;

    fn modal_boundary_solution(
        &self,
    ) -> Result<PlaneWaveModeCandidate<A>, ModeReconstructionError> {
        Ok(self
            .solution()
            .entries()
            .right_gauged_mode_candidate(self.solution().context().left_admittance()))
    }
}

pub(crate) fn bidirectional_waves_from_state<A>(
    state: &BoundaryState<A>,
    admittance: &A,
) -> BidirectionalWaves<A>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    let slope = admittance.scale(-A::Scalar::i());

    let difference = state.secondary().divide(&slope);

    let half = A::Scalar::one() / (A::Scalar::one() + A::Scalar::one());

    let forward = state.field().subtract(&difference).scale(half);

    let backward = state.field().add(&difference).scale(half);

    BidirectionalWaves::new(forward, backward)
}

impl<A> ReconstructExteriorModeWaves for Scatter2Workspace<A>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    type Algebra = A;

    fn reconstruct_exterior_mode_waves(
        &self,
        seed: &PlaneWaveModeCandidate<Self::Algebra>,
    ) -> Result<crate::waves::ExteriorBoundaryWaves<Self::Algebra>, ModeReconstructionError> {
        let left = bidirectional_waves_from_state(
            seed.state(),
            self.solution().context().left_admittance(),
        );

        let zero = seed.right_outgoing().zero_like();

        let right = BidirectionalWaves::new(seed.right_outgoing().clone(), zero);

        Ok(ExteriorBoundaryWaves::new(left, right))
    }
}

impl<A> ReconstructLayerModeWaves for Scatter2Workspace<A>
where
    A: ScalarAlgebra + Clone,
    A::Scalar: ComplexScalar + Zero,
    A::Dimension: Dimension,
{
    type Algebra = A;

    fn reconstruct_layer_mode_waves(
        &self,
        candidate: &PlaneWaveModeCandidate<A>,
    ) -> Result<Vec<LayerBoundaryWaves<A>>, ModeReconstructionError> {
        let retained = self
            .retained()
            .ok_or(ModeReconstructionError::ModeDataNotRetained)?;

        let source = self.sample_source();

        let left_waves = bidirectional_waves_from_state(
            candidate.state(),
            self.solution().context().left_admittance(),
        );

        let right_incoming = A::filled_constant_like(source, A::Scalar::zero());

        let waves =
            retained.reconstruct_from_incoming_waves(left_waves.forward(), &right_incoming, source);

        debug_assert_eq!(waves.len(), retained.num_layers());

        Ok(waves)
    }
}

impl<A> RetainedScatterComponents<A> {
    pub(crate) fn reconstruct_from_incoming_waves(
        &self,
        left_incoming: &A,
        right_incoming: &A,
        source: &ArrayBase<OwnedRepr<A::Scalar>, A::Dimension>,
    ) -> Vec<LayerBoundaryWaves<A>>
    where
        A: ScalarAlgebra + Clone,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        let prefixes = prefix_cascades(&self.components, source);

        let suffixes = suffix_cascades(&self.components, source);

        self.layer_cuts
            .iter()
            .map(|cuts| {
                let left = waves_at_cut(
                    &prefixes[cuts.left()],
                    &suffixes[cuts.left()],
                    left_incoming,
                    right_incoming,
                );

                let right = waves_at_cut(
                    &prefixes[cuts.right()],
                    &suffixes[cuts.right()],
                    left_incoming,
                    right_incoming,
                );

                LayerBoundaryWaves::new(left, right)
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{ArrayBase, Ix0, OwnedRepr, arr0};

    use super::{
        LayerCutIndices, RetainedScatterComponents, Scatter2Workspace,
        bidirectional_waves_from_state, prefix_cascades, suffix_cascades, waves_at_cut,
    };

    use crate::{
        Polarisation, RealAxis,
        algebra::ScalarAlgebra,
        backend::{
            ExteriorContextProvider, ExteriorWavevectors, IsotropicLayerQuantities,
            ModalSolutionSource, ReconstructLayerModeWaves, RunMode, Scatter2, SolutionWorkspace,
            scatter2::{
                Scatter2ExteriorContext,
                entries::{Scatter2Entries, cascade},
            },
        },
        input::{CanonicalCoordinates, CanonicalStack, IncidentSide},
        material::{Constant, ConstitutiveLift},
        test_support::{
            C, TOLERANCE,
            assertions::{
                assert_array_close, assert_bidirectional_waves_close, assert_complex_close,
                assert_zero_jet_close,
            },
            c,
            jet::{J0, J1, zero_jet_from_array, zero_jet_from_real_value, zero_jet_from_value},
            planar::{
                boundary_test_empty_stack, boundary_test_single_layer_stack,
                boundary_test_two_layer_stack, boundary_test_zero_thickness_stack,
            },
        },
        waves::{LayerBoundaryWaves, ReconstructLayerBoundaryWaves},
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
            &ExteriorWavevectors::new(
                IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                    &left_exterior,
                    &coordinates,
                    polarisation,
                )
                .kappa()
                .clone(),
                IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                    &right_exterior,
                    &coordinates,
                    polarisation,
                )
                .kappa()
                .clone(),
            ),
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

        assert_entries_close(&workspace.total(), &component, TOLERANCE);
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

        assert_entries_close(&entries.into_entries(), &component, TOLERANCE);
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

        assert_entries_close(&workspace.total(), &expected, TOLERANCE);
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

        workspace.append_layer(
            interface.clone(),
            propagation.clone(),
            sample_quantities(),
            sample_thickness(),
        );

        let expected = cascade(&interface, &propagation);

        assert_entries_close(&workspace.total(), &expected, TOLERANCE);
    }

    #[test]
    fn append_layer_retains_both_components() {
        let source = arr0(c(0.0));

        let context = make_context(&zero_jet_from_array(source.clone()));

        let mut workspace: Scatter2Workspace<J0> =
            Scatter2Workspace::new(&source, context, RunMode::InternalFields, 1);

        let interface = first_component();
        let propagation = second_component();

        workspace.append_layer(
            interface.clone(),
            propagation.clone(),
            sample_quantities(),
            sample_thickness(),
        );

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

        workspace.append_layer(
            first_component(),
            second_component(),
            sample_quantities(),
            sample_thickness(),
        );

        workspace.append_layer(
            third_component(),
            transparent_component(),
            sample_quantities(),
            sample_thickness(),
        );

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
            quantities: vec![sample_quantities(), sample_quantities()],
            thicknesses: vec![sample_thickness(), sample_thickness()],
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
            quantities: vec![sample_quantities()],
            thicknesses: vec![sample_thickness()],
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
            quantities: vec![sample_quantities()],
            thicknesses: vec![sample_thickness()],
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

        let quantities = IsotropicLayerQuantities::evaluate::<RealAxis, _>(
            &Constant::vacuum(),
            &CanonicalCoordinates::new(
                J1::from_parts(arr0(c(0.1)), arr0(c(0.0))),
                J1::from_parts(arr0(c(0.1)), arr0(c(0.0))),
            ),
            Polarisation::TransverseElectric,
        );

        let thickness = J1::from_parts(arr0(c(1.0)), arr0(c(0.0)));

        let retained = RetainedScatterComponents {
            components: vec![component],
            layer_cuts: vec![LayerCutIndices::new(0, 1)],
            quantities: vec![quantities],
            thicknesses: vec![thickness],
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

    fn test_coordinates() -> CanonicalCoordinates<J0> {
        CanonicalCoordinates::new(
            zero_jet_from_real_value(2.3),
            zero_jet_from_real_value(0.37),
        )
    }

    fn build_workspace(
        stack: CanonicalStack<Constant<f64>, J0>,
        mode: RunMode,
    ) -> Scatter2Workspace<J0> {
        Scatter2::new()
            .accumulate::<J0, RealAxis, _>(
                &test_coordinates(),
                &stack,
                Polarisation::TransverseElectric,
                &ExteriorWavevectors::new(
                    IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                        stack.left_exterior(),
                        &test_coordinates(),
                        Polarisation::TransverseElectric,
                    )
                    .kappa()
                    .clone(),
                    IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                        stack.right_exterior(),
                        &test_coordinates(),
                        Polarisation::TransverseElectric,
                    )
                    .kappa()
                    .clone(),
                ),
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

    fn sample_entries(offset: f64) -> Scatter2Entries<J0> {
        Scatter2Entries::from_parts(
            zero_jet_from_real_value(offset + 0.1),
            zero_jet_from_real_value(offset + 0.2),
            zero_jet_from_real_value(offset + 0.3),
            zero_jet_from_real_value(offset + 0.4),
        )
    }

    fn sample_quantities() -> IsotropicLayerQuantities<J0> {
        let material = Constant::vacuum();

        let coordinates =
            CanonicalCoordinates::new(zero_jet_from_real_value(1.0), zero_jet_from_real_value(0.5));

        IsotropicLayerQuantities::evaluate::<RealAxis, _>(
            &material,
            &coordinates,
            Polarisation::TransverseElectric,
        )
    }

    fn sample_thickness() -> J0 {
        J0::new(arr0(c(1.0)))
    }

    fn sample_source() -> ArrayBase<OwnedRepr<C>, Ix0> {
        ndarray::arr0(c(0.0))
    }

    fn sample_context() -> Scatter2ExteriorContext<J0> {
        Scatter2ExteriorContext::new::<RealAxis, _>(
            &test_coordinates(),
            &Constant::vacuum(),
            &Constant::vacuum(),
            &ExteriorWavevectors::new(
                IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                    &Constant::vacuum(),
                    &test_coordinates(),
                    Polarisation::TransverseElectric,
                )
                .kappa()
                .clone(),
                IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                    &Constant::vacuum(),
                    &test_coordinates(),
                    Polarisation::TransverseElectric,
                )
                .kappa()
                .clone(),
            ),
            Polarisation::TransverseElectric,
        )
    }

    #[test]
    fn one_layer_cut_surrounds_the_propagation_component() {
        let source = sample_source();

        let mut workspace =
            Scatter2Workspace::new(&source, sample_context(), RunMode::InternalFields, 1);

        let left_interface = sample_entries(1.0);
        let propagation = sample_entries(2.0);
        let right_interface = sample_entries(3.0);

        workspace.append_layer(
            left_interface.clone(),
            propagation.clone(),
            sample_quantities(),
            sample_thickness(),
        );

        workspace.append(right_interface.clone());

        let retained = workspace
            .retained
            .as_ref()
            .expect("internal-field mode should retain components");

        assert_eq!(
            retained.components.len(),
            3,
            "one finite layer should retain its left interface, \
         propagation component, and final right interface",
        );

        assert_eq!(retained.components[0], left_interface,);

        assert_eq!(retained.components[1], propagation,);

        assert_eq!(retained.components[2], right_interface,);

        assert_eq!(
            retained.layer_cuts.len(),
            1,
            "one finite layer should create one pair of cut indices",
        );

        let cuts = &retained.layer_cuts[0];

        assert_eq!(
            cuts.left(),
            1,
            "the left layer boundary lies after the left interface",
        );

        assert_eq!(
            cuts.right(),
            2,
            "the right layer boundary lies after propagation",
        );
    }

    #[test]
    fn two_layer_cuts_follow_physical_layer_order() {
        let source = sample_source();

        let mut workspace =
            Scatter2Workspace::new(&source, sample_context(), RunMode::InternalFields, 2);

        let interface0 = sample_entries(1.0);
        let propagation0 = sample_entries(2.0);
        let interface1 = sample_entries(3.0);
        let propagation1 = sample_entries(4.0);
        let final_interface = sample_entries(5.0);

        workspace.append_layer(
            interface0.clone(),
            propagation0.clone(),
            sample_quantities(),
            sample_thickness(),
        );

        workspace.append_layer(
            interface1.clone(),
            propagation1.clone(),
            sample_quantities(),
            sample_thickness(),
        );

        workspace.append(final_interface.clone());

        let retained = workspace
            .retained
            .as_ref()
            .expect("internal-field mode should retain components");

        assert_eq!(retained.components.len(), 5);

        assert_eq!(
            retained.components,
            vec![
                interface0,
                propagation0,
                interface1,
                propagation1,
                final_interface,
            ],
            "components must remain in physical left-to-right order",
        );

        assert_eq!(retained.layer_cuts.len(), 2);

        let layer0 = &retained.layer_cuts[0];
        let layer1 = &retained.layer_cuts[1];

        assert_eq!(
            (layer0.left(), layer0.right()),
            (1, 2),
            "layer 0 cuts should surround the first propagation component",
        );

        assert_eq!(
            (layer1.left(), layer1.right()),
            (3, 4),
            "layer 1 cuts should surround the second propagation component",
        );
    }

    #[test]
    fn response_only_append_layer_does_not_create_cut_topology() {
        let source = sample_source();

        let mut workspace =
            Scatter2Workspace::new(&source, sample_context(), RunMode::ResponseOnly, 2);

        workspace.append_layer(
            sample_entries(1.0),
            sample_entries(2.0),
            sample_quantities(),
            sample_thickness(),
        );

        workspace.append_layer(
            sample_entries(3.0),
            sample_entries(4.0),
            sample_quantities(),
            sample_thickness(),
        );

        workspace.append(sample_entries(5.0));

        assert!(
            workspace.retained.is_none(),
            "response-only workspaces must not retain components or cuts",
        );
    }

    #[test]
    fn newly_retained_workspace_has_empty_component_topology() {
        let source = sample_source();

        let workspace =
            Scatter2Workspace::new(&source, sample_context(), RunMode::InternalFields, 3);

        let retained = workspace
            .retained
            .as_ref()
            .expect("internal-field mode should enable retention");

        assert!(retained.components.is_empty());
        assert!(retained.layer_cuts.is_empty());
    }

    #[test]
    fn waves_at_identity_cut_equal_external_incoming_waves() {
        let source = sample_source();

        let identity = Scatter2Entries::<J0>::identity_like(&source);

        let left_incoming = zero_jet_from_real_value(1.0);

        let right_incoming = zero_jet_from_real_value(0.0);

        let waves = waves_at_cut(&identity, &identity, &left_incoming, &right_incoming);

        assert_zero_jet_close(waves.forward(), &left_incoming);

        assert_zero_jet_close(waves.backward(), &right_incoming);
    }

    #[test]
    fn cut_before_network_recovers_left_external_waves() {
        let total = sample_physical_entries();

        let identity = Scatter2Entries::<J0>::identity_like(total.sample_source());

        let left_incoming = zero_jet_from_real_value(1.0);

        let right_incoming = zero_jet_from_real_value(0.0);

        let waves = waves_at_cut(&identity, &total, &left_incoming, &right_incoming);

        assert_zero_jet_close(waves.forward(), &left_incoming);

        assert_zero_jet_close(waves.backward(), total.s11());
    }

    fn sample_physical_entries() -> Scatter2Entries<J0> {
        Scatter2Entries::from_parts(
            zero_jet_from_real_value(1.0),
            zero_jet_from_real_value(1.0),
            zero_jet_from_real_value(1.0),
            zero_jet_from_real_value(1.0),
        )
    }

    #[test]
    fn cut_after_network_recovers_right_external_waves() {
        let total = sample_physical_entries();

        let identity = Scatter2Entries::<J0>::identity_like(total.sample_source());

        let left_incoming = zero_jet_from_real_value(1.0);

        let right_incoming = zero_jet_from_real_value(0.0);

        let waves = waves_at_cut(&total, &identity, &left_incoming, &right_incoming);

        assert_zero_jet_close(waves.forward(), total.s21());

        assert_zero_jet_close(waves.backward(), &right_incoming);
    }

    #[test]
    fn prefix_cascades_have_expected_endpoint_semantics() {
        let source = sample_source();

        let components = vec![
            sample_entries(1.0),
            sample_entries(2.0),
            sample_entries(3.0),
        ];

        let prefixes = prefix_cascades(&components, &source);

        assert_eq!(prefixes.len(), 4);

        assert_entries_close(
            &prefixes[0],
            &Scatter2Entries::identity_like(&source),
            TOLERANCE,
        );

        assert_entries_close(&prefixes[1], &components[0], TOLERANCE);

        assert_entries_close(
            &prefixes[2],
            &cascade(&components[0], &components[1]),
            TOLERANCE,
        );

        assert_entries_close(
            &prefixes[3],
            &cascade(&cascade(&components[0], &components[1]), &components[2]),
            TOLERANCE,
        );
    }

    #[test]
    fn suffix_cascades_have_expected_endpoint_semantics() {
        let source = sample_source();

        let components = vec![
            sample_entries(1.0),
            sample_entries(2.0),
            sample_entries(3.0),
        ];

        let suffixes = suffix_cascades(&components, &source);

        assert_eq!(suffixes.len(), 4);

        assert_entries_close(
            &suffixes[0],
            &cascade(&components[0], &cascade(&components[1], &components[2])),
            TOLERANCE,
        );

        assert_entries_close(
            &suffixes[1],
            &cascade(&components[1], &components[2]),
            TOLERANCE,
        );

        assert_entries_close(&suffixes[2], &components[2], TOLERANCE);

        assert_entries_close(
            &suffixes[3],
            &Scatter2Entries::identity_like(&source),
            TOLERANCE,
        );
    }

    fn assert_layer_boundary_waves_close(
        actual: &LayerBoundaryWaves<J0>,
        expected: &LayerBoundaryWaves<J0>,
    ) {
        assert_bidirectional_waves_close(actual.left(), expected.left(), TOLERANCE);

        assert_bidirectional_waves_close(actual.right(), expected.right(), TOLERANCE);
    }

    fn assert_layer_boundary_waves_scaled(
        actual: &LayerBoundaryWaves<J0>,
        expected: &LayerBoundaryWaves<J0>,
        scale: C,
    ) {
        for (actual, expected) in [
            (actual.left().forward(), expected.left().forward()),
            (actual.left().backward(), expected.left().backward()),
            (actual.right().forward(), expected.right().forward()),
            (actual.right().backward(), expected.right().backward()),
        ] {
            assert_complex_close(actual.value()[()], scale * expected.value()[()], TOLERANCE);
        }
    }

    fn retained_components_fixture() -> RetainedScatterComponents<J0> {
        RetainedScatterComponents {
            components: vec![
                first_component(),
                second_component(),
                third_component(),
                transparent_component(),
            ],

            layer_cuts: vec![LayerCutIndices::new(1, 2), LayerCutIndices::new(3, 4)],

            quantities: vec![sample_quantities(), sample_quantities()],

            thicknesses: vec![sample_thickness(), sample_thickness()],
        }
    }

    #[test]
    fn arbitrary_incoming_reconstruction_returns_one_record_per_layer() {
        let source = arr0(c(0.0));

        let retained = retained_components_fixture();

        let waves = retained.reconstruct_from_incoming_waves(
            &zero_jet_from_value(c(0.7)),
            &zero_jet_from_value(c(-0.2)),
            &source,
        );

        assert_eq!(waves.len(), retained.num_layers(),);
    }

    #[test]
    fn arbitrary_incoming_reconstruction_is_linear() {
        let source = arr0(c(0.0));

        let retained = retained_components_fixture();

        let left = zero_jet_from_value(c(0.7));

        let right = zero_jet_from_value(c(-0.2));

        let scale = c(1.7) + C::i() * c(-0.4);

        let base = retained.reconstruct_from_incoming_waves(&left, &right, &source);

        let scaled = retained.reconstruct_from_incoming_waves(
            &left.scale(scale),
            &right.scale(scale),
            &source,
        );

        for (actual, expected) in scaled.iter().zip(&base) {
            assert_layer_boundary_waves_scaled(actual, expected, scale);
        }
    }

    #[test]
    fn driven_reconstruction_delegates_to_general_incoming_reconstruction() {
        let source = arr0(c(0.0));

        let retained = retained_components_fixture();

        let driven = retained.reconstruct_layer_boundary_waves(IncidentSide::Left, &source);

        let general = retained.reconstruct_from_incoming_waves(
            &zero_jet_from_value(c(1.0)),
            &zero_jet_from_value(c(0.0)),
            &source,
        );

        assert_eq!(driven.len(), general.len());

        for (actual, expected) in driven.iter().zip(&general) {
            assert_layer_boundary_waves_close(actual, expected);
        }
    }

    #[test]
    fn projective_modal_candidate_reconstructs_original_projective_solution() {
        let workspace = build_workspace(boundary_test_two_layer_stack(), RunMode::InternalFields);

        let candidate = workspace.modal_boundary_solution().unwrap();

        let actual = workspace.reconstruct_layer_mode_waves(&candidate).unwrap();

        let source = workspace.sample_source();

        let left_waves = bidirectional_waves_from_state(
            candidate.state(),
            workspace.solution().context().left_admittance(),
        );

        let zero = J0::filled_constant_like(source, C::new(0.0, 0.0));

        let expected = workspace
            .retained()
            .unwrap()
            .reconstruct_from_incoming_waves(left_waves.forward(), &zero, source);

        assert_eq!(actual.len(), expected.len());

        for (actual, expected) in actual.iter().zip(&expected) {
            assert_layer_boundary_waves_close(actual, expected);
        }
    }
}
