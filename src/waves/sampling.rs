use std::marker::PhantomData;

use nalgebra::ComplexField;
use ndarray::Dimension;

use crate::{
    ComplexScalar, IncidentSide,
    algebra::{Jet, ScalarAlgebra},
    backend::{
        ExteriorContextProvider, ModeReconstructionError, PlaneWaveEntries, PlaneWaveModeCandidate,
        PlaneWaveSolutionSource, ReconstructExteriorModeWaves, ReconstructLayerModeWaves,
        RetainedIsotropicLayers,
    },
    observable::{Amplitudes, ProjectAmplitudes},
    spatial::{CanonicalFieldPosition, CompiledFieldSampling},
    waves::{
        BidirectionalWaves, ExteriorBoundaryWaves, LayerBoundaryWaves,
        boundary::BoundaryWaveSolution,
    },
};

use super::{
    PropagateLayerWaves, PropagateWaves, ReconstructExteriorBoundaryWaves,
    ReconstructLayerBoundaryWaves,
};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum WaveSamplingError {
    #[error(
        "sampling requested finite layer {requested}, \
         but propagation data contain {layer_count} layers"
    )]
    LayerOutOfBounds {
        requested: usize,
        layer_count: usize,
    },

    #[error("sampling requested but no layer data was retained")]
    LayersNotRetained,

    #[error(transparent)]
    ModeReconstruction(#[from] ModeReconstructionError),

    #[error("retained data are incomplete for finite layer {index}")]
    MissingLayerData { index: usize },

    #[error(
        "reconstructed wave count {wave_count} does not match retained \
     layer count {retained_count}"
    )]
    LayerCountMismatch {
        wave_count: usize,
        retained_count: usize,
    },
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct WaveSamplingContext<'a, W, A> {
    workspace: &'a W,
    algebra: PhantomData<A>,
}

impl<'a, W, A> WaveSamplingContext<'a, W, A> {
    pub(crate) const fn new(workspace: &'a W) -> Self {
        Self {
            workspace,
            algebra: PhantomData,
        }
    }

    pub(crate) const fn workspace(&self) -> &'a W {
        self.workspace
    }
}

impl<'a, W, A> WaveSamplingContext<'a, W, A> {
    pub(crate) fn modal_boundary_waves(
        &self,
        seed: &PlaneWaveModeCandidate<A>,
    ) -> Result<BoundaryWaveSolution<A>, WaveSamplingError>
    where
        W: ReconstructExteriorModeWaves<Algebra = A>
            + ReconstructLayerModeWaves<Algebra = A>
            + RetainedIsotropicLayers,
    {
        let exterior = self.workspace.reconstruct_exterior_mode_waves(seed)?;

        let layers = self.workspace.reconstruct_layer_mode_waves(seed)?;

        let retained_layer_count = self
            .workspace
            .retained_layer_count()
            .ok_or(WaveSamplingError::LayersNotRetained)?;

        if layers.len() != retained_layer_count {
            return Err(WaveSamplingError::LayerCountMismatch {
                wave_count: layers.len(),
                retained_count: retained_layer_count,
            });
        }

        Ok(BoundaryWaveSolution::new(exterior, layers))
    }

    pub(crate) fn driven_boundary_waves(
        &self,
        incident_side: IncidentSide,
    ) -> Result<BoundaryWaveSolution<A>, WaveSamplingError>
    where
        A: Jet,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
        W: ReconstructExteriorBoundaryWaves<Algebra = A>
            + ReconstructLayerBoundaryWaves<Algebra = A>
            + RetainedIsotropicLayers,
        W::Entries: ProjectAmplitudes,
        <W::Entries as ProjectAmplitudes>::Amplitudes: Amplitudes<Algebra = A>,
    {
        let exterior = self
            .workspace
            .reconstruct_exterior_boundary_waves(incident_side);

        let layers = self
            .workspace
            .reconstruct_layer_boundary_waves(incident_side)
            .ok_or(WaveSamplingError::LayersNotRetained)?;

        let retained_layer_count = self
            .workspace
            .retained_layer_count()
            .ok_or(WaveSamplingError::LayersNotRetained)?;

        if layers.len() != retained_layer_count {
            return Err(WaveSamplingError::LayerCountMismatch {
                wave_count: layers.len(),
                retained_count: retained_layer_count,
            });
        }

        Ok(BoundaryWaveSolution::new(exterior, layers))
    }

    pub(crate) fn propagate_sampling(
        &self,
        incident_side: IncidentSide,
        sampling: &CompiledFieldSampling<<A::Scalar as ComplexField>::RealField>,
    ) -> Result<Vec<BidirectionalWaves<A>>, WaveSamplingError>
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
        W: ReconstructExteriorBoundaryWaves<Algebra = A>
            + ReconstructLayerBoundaryWaves<Algebra = A>
            + RetainedIsotropicLayers<Algebra = A>
            + PlaneWaveSolutionSource,
        W::Entries: ProjectAmplitudes,
        <W::Entries as ProjectAmplitudes>::Amplitudes: Amplitudes<Algebra = A>,
        <A::Scalar as ComplexField>::RealField: Copy,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = A>,
    {
        let waves = self.driven_boundary_waves(incident_side)?;

        self.propagate_reconstructed(&waves, sampling)
    }

    pub(crate) fn propagate_reconstructed(
        &self,
        waves: &BoundaryWaveSolution<A>,
        sampling: &CompiledFieldSampling<<A::Scalar as ComplexField>::RealField>,
    ) -> Result<Vec<BidirectionalWaves<A>>, WaveSamplingError>
    where
        A: ScalarAlgebra,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
        <A::Scalar as ComplexField>::RealField: Copy,
        W: PlaneWaveSolutionSource + RetainedIsotropicLayers<Algebra = A>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = A>,
    {
        sampling
            .positions()
            .iter()
            .copied()
            .map(|position| {
                propagate_position_from_reconstructed(
                    self.workspace,
                    waves.exterior(),
                    waves.layers(),
                    position,
                )
            })
            .collect()
    }
}

fn propagate_position_from_reconstructed<A, W>(
    workspace: &W,
    exterior: &ExteriorBoundaryWaves<A>,
    layers: &[LayerBoundaryWaves<A>],
    position: CanonicalFieldPosition<<<A as Jet>::Scalar as ComplexField>::RealField>,
) -> Result<BidirectionalWaves<A>, WaveSamplingError>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
    <A::Scalar as ComplexField>::RealField: Copy,
    W: PlaneWaveSolutionSource + RetainedIsotropicLayers<Algebra = A>,
    <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = A>,
{
    let exterior_context = workspace.solution().context();
    match position {
        CanonicalFieldPosition::LeftExterior { distance } => Ok(exterior
            .left()
            .propagate(exterior_context.left_kappa(), -distance)),

        CanonicalFieldPosition::Layer { index, position } => {
            let layer_thickness = workspace
                .layer_thickness(index.get())
                .ok_or(WaveSamplingError::MissingLayerData { index: index.get() })?;

            let layer = layers
                .get(index.get())
                .ok_or(WaveSamplingError::LayerOutOfBounds {
                    requested: index.get(),
                    layer_count: layers.len(),
                })?;

            let longitudinal_wavevector = workspace
                .layer_quantities(index.get())
                .ok_or(WaveSamplingError::MissingLayerData { index: index.get() })?
                .kappa();

            Ok(layer.propagate_to_position(longitudinal_wavevector, layer_thickness, position))
        }

        CanonicalFieldPosition::RightExterior { distance } => Ok(exterior
            .right()
            .propagate(exterior_context.right_kappa(), distance)),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Ix0;

    use super::*;

    use crate::{
        FiniteLayerIndex, Polarisation, RealAxis,
        algebra::ArrayJet0,
        backend::{RunMode, Scatter2, evaluate_exterior_wavevectors},
        input::{CanonicalCoordinates, CanonicalStack},
        material::Constant,
        spatial::{CanonicalFieldPosition, CanonicalLayerPosition, CompiledFieldSampling},
        test_support::{
            C, TOLERANCE,
            assertions::assert_bidirectional_waves_close,
            jet::zero_jet_from_real_value,
            planar::{boundary_test_empty_stack, boundary_test_single_layer_stack},
        },
    };

    type A = ArrayJet0<C, Ix0, crate::algebra::RealParameter>;

    fn coordinates() -> CanonicalCoordinates<A> {
        CanonicalCoordinates::new(
            zero_jet_from_real_value(2.3),
            zero_jet_from_real_value(0.37),
        )
    }

    fn build_workspace(
        stack: CanonicalStack<Constant<f64>, A>,
        mode: RunMode,
    ) -> crate::backend::scatter2::Scatter2Workspace<A> {
        let coordinates = coordinates();
        let exterior = evaluate_exterior_wavevectors::<RealAxis, _, _>(
            &coordinates,
            stack.left_exterior(),
            stack.right_exterior(),
        );

        Scatter2::new().accumulate::<A, RealAxis, _>(
            &coordinates,
            &stack,
            &exterior,
            Polarisation::TransverseElectric,
            mode,
        )
    }

    #[test]
    fn sampling_requires_retained_layer_data() {
        let workspace = build_workspace(boundary_test_single_layer_stack(), RunMode::ResponseOnly);

        let context = WaveSamplingContext::new(&workspace);

        let sampling = CompiledFieldSampling::new(vec![CanonicalFieldPosition::Layer {
            index: FiniteLayerIndex::new(0),
            position: CanonicalLayerPosition::FromLeft(0.1),
        }]);

        let error = context
            .propagate_sampling(IncidentSide::Left, &sampling)
            .unwrap_err();

        assert_eq!(error, WaveSamplingError::LayersNotRetained,);
    }

    #[test]
    fn empty_stack_samples_both_exteriors() {
        let workspace = build_workspace(boundary_test_empty_stack(), RunMode::InternalFields);

        let context = WaveSamplingContext::new(&workspace);

        let sampling = CompiledFieldSampling::new(vec![
            CanonicalFieldPosition::LeftExterior { distance: 0.2 },
            CanonicalFieldPosition::RightExterior { distance: 0.3 },
        ]);

        let actual = context
            .propagate_sampling(IncidentSide::Left, &sampling)
            .unwrap();

        assert_eq!(actual.len(), 2);

        let waves = context.driven_boundary_waves(IncidentSide::Left).unwrap();

        let exterior = waves.exterior();

        let solution = workspace.solution();
        let exterior_context = solution.context();

        let expected_left = exterior
            .left()
            .propagate(exterior_context.left_kappa(), -0.2);

        let expected_right = exterior
            .right()
            .propagate(exterior_context.right_kappa(), 0.3);

        assert_bidirectional_waves_close(&actual[0], &expected_left, TOLERANCE);

        assert_bidirectional_waves_close(&actual[1], &expected_right, TOLERANCE);
    }

    #[test]
    fn samples_inside_retained_finite_layer() {
        let workspace =
            build_workspace(boundary_test_single_layer_stack(), RunMode::InternalFields);

        let context = WaveSamplingContext::new(&workspace);

        let position = CanonicalLayerPosition::FromLeft(0.1);

        let sampling = CompiledFieldSampling::new(vec![CanonicalFieldPosition::Layer {
            index: FiniteLayerIndex::new(0),
            position,
        }]);

        let actual = context
            .propagate_sampling(IncidentSide::Left, &sampling)
            .unwrap();

        assert_eq!(actual.len(), 1);

        let waves = context.driven_boundary_waves(IncidentSide::Left).unwrap();

        let boundaries = waves.layers();

        let quantities = workspace
            .layer_quantities(0)
            .expect("single layer should retain quantities");

        let thickness = workspace
            .layer_thickness(0)
            .expect("single layer should retain thickness");

        let expected = boundaries[0].propagate_to_position(quantities.kappa(), thickness, position);

        assert_bidirectional_waves_close(&actual[0], &expected, TOLERANCE);
    }

    #[test]
    fn dispatch_preserves_layer_position_semantics() {
        let workspace =
            build_workspace(boundary_test_single_layer_stack(), RunMode::InternalFields);

        let context = WaveSamplingContext::new(&workspace);

        let positions = [
            CanonicalLayerPosition::FromLeft(0.1),
            CanonicalLayerPosition::FromRight(0.1),
            CanonicalLayerPosition::Fraction(0.25),
        ];

        let sampling = CompiledFieldSampling::new(
            positions
                .iter()
                .copied()
                .map(|position| CanonicalFieldPosition::Layer {
                    index: FiniteLayerIndex::new(0),
                    position,
                })
                .collect(),
        );

        let actual = context
            .propagate_sampling(IncidentSide::Left, &sampling)
            .unwrap();

        let waves = context.driven_boundary_waves(IncidentSide::Left).unwrap();
        let boundaries = waves.layers();

        let quantities = workspace
            .layer_quantities(0)
            .expect("single layer should retain quantities");

        let thickness = workspace
            .layer_thickness(0)
            .expect("single layer should retain thickness");

        assert_eq!(actual.len(), positions.len());

        for (actual, position) in actual.iter().zip(positions) {
            let expected =
                boundaries[0].propagate_to_position(quantities.kappa(), thickness, position);

            assert_bidirectional_waves_close(actual, &expected, TOLERANCE);
        }
    }

    #[test]
    fn sampling_respects_incident_side() {
        let workspace =
            build_workspace(boundary_test_single_layer_stack(), RunMode::InternalFields);

        let context = WaveSamplingContext::new(&workspace);

        let sampling = CompiledFieldSampling::new(vec![
            CanonicalFieldPosition::LeftExterior { distance: 0.0 },
            CanonicalFieldPosition::Layer {
                index: FiniteLayerIndex::new(0),
                position: CanonicalLayerPosition::FromLeft(0.1),
            },
            CanonicalFieldPosition::RightExterior { distance: 0.0 },
        ]);

        let left = context
            .propagate_sampling(IncidentSide::Left, &sampling)
            .unwrap();

        let right = context
            .propagate_sampling(IncidentSide::Right, &sampling)
            .unwrap();

        assert_ne!(left, right);
    }

    #[test]
    fn exterior_boundary_sampling_matches_solution_amplitudes() {
        let workspace =
            build_workspace(boundary_test_single_layer_stack(), RunMode::InternalFields);

        let context = WaveSamplingContext::new(&workspace);

        for side in [IncidentSide::Left, IncidentSide::Right] {
            let waves = context.driven_boundary_waves(side).unwrap();
            let exterior = waves.exterior();

            let solution = workspace.solution();

            let amplitudes = solution
                .entries()
                .project_amplitudes(solution.context(), side);

            let zero = zero_jet_from_real_value(0.0);
            let one = zero_jet_from_real_value(1.0);

            match side {
                IncidentSide::Left => {
                    assert_bidirectional_waves_close(
                        exterior.left(),
                        &BidirectionalWaves::new(one.clone(), amplitudes.reflection().clone()),
                        TOLERANCE,
                    );

                    assert_bidirectional_waves_close(
                        exterior.right(),
                        &BidirectionalWaves::new(amplitudes.transmission().clone(), zero.clone()),
                        TOLERANCE,
                    );
                }

                IncidentSide::Right => {
                    assert_bidirectional_waves_close(
                        exterior.left(),
                        &BidirectionalWaves::new(zero.clone(), amplitudes.transmission().clone()),
                        TOLERANCE,
                    );

                    assert_bidirectional_waves_close(
                        exterior.right(),
                        &BidirectionalWaves::new(amplitudes.reflection().clone(), one.clone()),
                        TOLERANCE,
                    );
                }
            }
        }
    }
}
