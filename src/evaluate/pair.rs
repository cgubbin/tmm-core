use std::fmt::Debug;

use nalgebra::ComplexField;
use ndarray::Dimension;
use thiserror::Error;

use crate::{
    ComplexScalar, FiniteLayerIndex, IncidentSide, Polarisation,
    algebra::{Jet, RealScalarAlgebra, ScalarAlgebra, ScalarAlgebraExpRelExt},
    backend::{PlaneWaveSolutionSource, ReconstructLayerBoundaryWaves, RetainedIsotropicLayers},
    derivative_parts::DerivativePartsPolicy,
    differential::IntoDifferentialResponse,
    evaluate::query::DifferentialResponseFor,
    input::JetMapping,
    observable::{
        AggregateHermitianOverlap, BoundaryProjectionError, HermitianLayerOverlap,
        HermitianLayerOverlapInput, HermitianOverlapError, LayerOverlapOperand, Layers,
        NormalizedHermitianOverlap, PairOperand, project_layer_boundary_waves,
    },
};

use super::{PlaneWaveState, query::PlaneWaveQuery};

/// Two retained plane-wave solutions validated for pairwise Hermitian
/// operations.
///
/// The reference solution forms the conjugated, or left, operand of the
/// Hermitian product:
///
/// ```text
/// overlap = reference* comparison
/// ```
///
/// The pair represents an elementwise comparison between aligned compiled
/// samples. The canonical coordinate values may differ, but the two states
/// must have compatible sampled shapes, layer geometries, polarizations, and
/// differential mappings.
#[derive(Debug)]
pub struct HermitianPlaneWavePair<'a, JL, JR, IL, IR, ML, MR, WL, WR>
where
    JL: Jet + JetMapping,
    JR: Jet + JetMapping,
    JL::Scalar: ComplexField,
    JR::Scalar: ComplexField,
    JL::Dimension: Dimension,
    JR::Dimension: Dimension,
    IL: ComplexField,
    IR: ComplexField,
    JL::Mapping: Debug,
    JR::Mapping: Debug,
{
    reference: &'a PlaneWaveState<JL, IL, ML, WL>,
    comparison: &'a PlaneWaveState<JR, IR, MR, WR>,
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum HermitianOverlapEvaluationError {
    // Existing compatibility variants...
    #[error("failed to reconstruct reference boundary waves: {0}")]
    ReferenceBoundaryProjection(#[source] BoundaryProjectionError),

    #[error("failed to reconstruct comparison boundary waves: {0}")]
    ComparisonBoundaryProjection(#[source] BoundaryProjectionError),

    #[error(
        "reference overlap data are unavailable for finite layer {index}; \
         retained layer count is {layer_count}"
    )]
    MissingReferenceLayerData { index: usize, layer_count: usize },

    #[error(
        "comparison overlap data are unavailable for finite layer {index}; \
         retained layer count is {layer_count}"
    )]
    MissingComparisonLayerData { index: usize, layer_count: usize },

    #[error(transparent)]
    FieldProjection(#[from] HermitianOverlapError),
}

impl<'a, J, I, ML, MR, WL, WR> HermitianPlaneWavePair<'a, J, J, I, I, ML, MR, WL, WR>
where
    J: ScalarAlgebra + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    J::Mapping: Debug,
    I: ComplexField,
{
    /// Validate two retained states for aligned pairwise Hermitian operations.
    ///
    /// The initial implementation requires both states to use the same jet
    /// and input scalar types. This guarantees representation compatibility,
    /// but construction must still verify the sampled shape, physical layer
    /// partition, polarization, and semantic derivative mapping.
    pub(crate) fn new(
        reference: &'a PlaneWaveState<J, I, ML, WL>,
        comparison: &'a PlaneWaveState<J, I, MR, WR>,
    ) -> Result<Self, HermitianOverlapError>
    where
        WL: PairWorkspace<Thickness = J>,
        WR: PairWorkspace<Thickness = J>,
        J::Mapping: PairMappingCompatibility,
    {
        validate_sample_shape(reference, comparison)?;

        validate_polarisation(reference, comparison)?;

        validate_layer_geometry(reference, comparison)?;

        validate_mapping(reference, comparison)?;

        Ok(Self {
            reference,
            comparison,
        })
    }

    /// Return the conjugated, or left, operand.
    pub fn reference(&self) -> &'a PlaneWaveState<J, I, ML, WL> {
        self.reference
    }

    /// Return the unconjugated, or right, operand.
    pub fn comparison(&self) -> &'a PlaneWaveState<J, I, MR, WR> {
        self.comparison
    }

    /// Consume the pair and return `(reference, comparison)`.
    pub fn into_parts(
        self,
    ) -> (
        &'a PlaneWaveState<J, I, ML, WL>,
        &'a PlaneWaveState<J, I, MR, WR>,
    ) {
        (self.reference, self.comparison)
    }
}

impl<J, I, M, W> PlaneWaveState<J, I, M, W>
where
    J: ScalarAlgebra + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    J::Mapping: Debug,
    I: ComplexField,
{
    /// Validate this state and `other` for aligned Hermitian operations.
    ///
    /// This state becomes the reference operand and is conjugated in the
    /// Hermitian product. `other` becomes the comparison operand.
    ///
    /// Canonical coordinate values may differ elementwise, but both states
    /// must have the same compiled sample layout and physical finite-layer
    /// partition.
    pub fn pair_with<'a, M2, W2>(
        &'a self,
        other: &'a PlaneWaveState<J, I, M2, W2>,
    ) -> Result<HermitianPlaneWavePair<'a, J, J, I, I, M, M2, W, W2>, HermitianOverlapError>
    where
        W: PairWorkspace<Thickness = J>,
        W2: PairWorkspace<Thickness = J>,
        J::Mapping: PairMappingCompatibility,
    {
        HermitianPlaneWavePair::new(self, other)
    }
}

/// Workspace information required to validate pairwise layer geometry.
pub(crate) trait PairWorkspace {
    type Thickness;

    /// Return the retained finite-layer count.
    fn pair_layer_count(&self) -> Option<usize>;

    /// Return the physical thickness of one finite layer.
    fn pair_layer_thickness(&self, index: usize) -> Option<&Self::Thickness>;
}

/// Semantic compatibility of two differential mappings.
///
/// Equality of mapping types is insufficient when values of the same type can
/// describe different derivative parameters or layer indices.
pub(crate) trait PairMappingCompatibility {
    fn pair_mapping_compatible(&self, other: &Self) -> bool;
}

fn validate_sample_shape<J, I, ML, MR, WL, WR>(
    reference: &PlaneWaveState<J, I, ML, WL>,
    comparison: &PlaneWaveState<J, I, MR, WR>,
) -> Result<(), HermitianOverlapError>
where
    J: JetMapping + ScalarAlgebra,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    I: ComplexField,
{
    let reference_shape = reference
        .problem()
        .coordinates()
        .vacuum_angular_wavenumber()
        .value()
        .shape();

    let comparison_shape = comparison
        .problem()
        .coordinates()
        .vacuum_angular_wavenumber()
        .value()
        .shape();

    if reference_shape != comparison_shape {
        return Err(HermitianOverlapError::SampleShapeMismatch {
            reference: reference_shape.to_vec(),
            comparison: comparison_shape.to_vec(),
        });
    }

    Ok(())
}

fn validate_polarisation<J, I, ML, MR, WL, WR>(
    reference: &PlaneWaveState<J, I, ML, WL>,
    comparison: &PlaneWaveState<J, I, MR, WR>,
) -> Result<(), HermitianOverlapError>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    I: ComplexField,
{
    let reference_polarisation = reference.polarisation();

    let comparison_polarisation = comparison.polarisation();

    if reference_polarisation != comparison_polarisation {
        return Err(HermitianOverlapError::PolarisationMismatch {
            reference: reference_polarisation,
            comparison: comparison_polarisation,
        });
    }

    Ok(())
}

fn validate_layer_geometry<J, I, ML, MR, WL, WR>(
    reference: &PlaneWaveState<J, I, ML, WL>,
    comparison: &PlaneWaveState<J, I, MR, WR>,
) -> Result<(), HermitianOverlapError>
where
    J: ScalarAlgebra + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    I: ComplexField,
    WL: PairWorkspace<Thickness = J>,
    WR: PairWorkspace<Thickness = J>,
{
    let reference_count = reference.workspace().pair_layer_count().ok_or(
        HermitianOverlapError::LayersNotRetained {
            operand: PairOperand::Reference,
        },
    )?;

    let comparison_count = comparison.workspace().pair_layer_count().ok_or(
        HermitianOverlapError::LayersNotRetained {
            operand: PairOperand::Comparison,
        },
    )?;

    if reference_count != comparison_count {
        return Err(HermitianOverlapError::LayerCountMismatch {
            reference_count,
            comparison_count,
        });
    }

    for index in 0..reference_count {
        let reference_thickness = reference.workspace().pair_layer_thickness(index).ok_or(
            HermitianOverlapError::LayersNotRetained {
                operand: PairOperand::Reference,
            },
        )?;

        let comparison_thickness = comparison.workspace().pair_layer_thickness(index).ok_or(
            HermitianOverlapError::LayersNotRetained {
                operand: PairOperand::Comparison,
            },
        )?;

        if reference_thickness.value() != comparison_thickness.value() {
            return Err(HermitianOverlapError::LayerThicknessMismatch {
                index: FiniteLayerIndex(index),
            });
        }
    }

    Ok(())
}

fn validate_mapping<J, I, ML, MR, WL, WR>(
    reference: &PlaneWaveState<J, I, ML, WL>,
    comparison: &PlaneWaveState<J, I, MR, WR>,
) -> Result<(), HermitianOverlapError>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    I: ComplexField,
    J::Mapping: PairMappingCompatibility,
{
    if !reference
        .context()
        .mapping()
        .pair_mapping_compatible(comparison.context().mapping())
    {
        return Err(HermitianOverlapError::DifferentialMappingMismatch);
    }

    Ok(())
}

impl<'a, J, I, ML, MR, WL, WR> HermitianPlaneWavePair<'a, J, J, I, I, ML, MR, WL, WR>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    I: ComplexField,
    J::Mapping: Debug,
{
    fn raw_layer_overlap_inputs(
        &self,
        reference_side: IncidentSide,
        comparison_side: IncidentSide,
    ) -> Result<Layers<HermitianLayerOverlapInput<J>>, HermitianOverlapEvaluationError>
    where
        J: Clone,
        WL: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
        WR: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
    {
        let reference_waves =
            project_layer_boundary_waves(self.reference.workspace(), reference_side)
                .map_err(HermitianOverlapEvaluationError::ReferenceBoundaryProjection)?;

        let comparison_waves =
            project_layer_boundary_waves(self.comparison.workspace(), comparison_side)
                .map_err(HermitianOverlapEvaluationError::ComparisonBoundaryProjection)?;

        let reference_count = self.reference.workspace().retained_layer_count().ok_or(
            HermitianOverlapError::LayersNotRetained {
                operand: PairOperand::Reference,
            },
        )?;

        let comparison_count = self.comparison.workspace().retained_layer_count().ok_or(
            HermitianOverlapError::LayersNotRetained {
                operand: PairOperand::Comparison,
            },
        )?;

        /*
         * This should already have been checked by pair construction, but
         * retaining the guard here protects against inconsistent workspace
         * implementations.
         */
        if reference_count != comparison_count {
            return Err(HermitianOverlapEvaluationError::FieldProjection(
                HermitianOverlapError::LayerCountMismatch {
                    reference_count,
                    comparison_count,
                },
            ));
        }

        if reference_waves.len() != reference_count {
            return Err(HermitianOverlapEvaluationError::FieldProjection(
                HermitianOverlapError::LayerCountMismatch {
                    reference_count: reference_waves.len(),
                    comparison_count: reference_count,
                },
            ));
        }

        if comparison_waves.len() != comparison_count {
            return Err(HermitianOverlapEvaluationError::FieldProjection(
                HermitianOverlapError::LayerCountMismatch {
                    reference_count: comparison_waves.len(),
                    comparison_count,
                },
            ));
        }

        let reference_waves = reference_waves.into_inner();

        let comparison_waves = comparison_waves.into_inner();

        let mut inputs = Vec::with_capacity(reference_count);

        for index in 0..reference_count {
            let reference_quantities = self
                .reference
                .workspace()
                .layer_quantities(index)
                .ok_or(HermitianOverlapEvaluationError::MissingReferenceLayerData {
                    index,
                    layer_count: reference_count,
                })?
                .clone();

            let comparison_quantities = self
                .comparison
                .workspace()
                .layer_quantities(index)
                .ok_or(
                    HermitianOverlapEvaluationError::MissingComparisonLayerData {
                        index,
                        layer_count: comparison_count,
                    },
                )?
                .clone();

            /*
             * Pair construction established that corresponding physical
             * thicknesses match. Use the reference thickness as the common
             * analytic integration interval.
             */
            let thickness = self
                .reference
                .workspace()
                .layer_thickness(index)
                .ok_or(HermitianOverlapEvaluationError::MissingReferenceLayerData {
                    index,
                    layer_count: reference_count,
                })?
                .clone();

            let reference_boundaries = reference_waves[index].clone();

            let comparison_boundaries = comparison_waves[index].clone();

            /*
             * Directional propagation is parameterized from each finite
             * layer's left boundary.
             */
            let (reference_left, _) = reference_boundaries.into_parts();

            let (comparison_left, _) = comparison_boundaries.into_parts();

            inputs.push(HermitianLayerOverlapInput::new(
                LayerOverlapOperand::new(reference_left, reference_quantities),
                LayerOverlapOperand::new(comparison_left, comparison_quantities),
                thickness,
            ));
        }

        Ok(Layers::new(inputs))
    }
}

impl<'a, J, I, ML, MR, WL, WR> HermitianPlaneWavePair<'a, J, J, I, I, ML, MR, WL, WR>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    J::Mapping: Debug,
    I: ComplexField,
{
    pub(crate) fn raw_layer_overlap(
        &self,
        reference_side: IncidentSide,
        comparison_side: IncidentSide,
    ) -> Result<Layers<HermitianLayerOverlap<J>>, HermitianOverlapEvaluationError>
    where
        J: RealScalarAlgebra + ScalarAlgebraExpRelExt + Clone,
        J::Scalar: ComplexScalar,
        WL: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
        WR: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
    {
        let inputs = self.raw_layer_overlap_inputs(reference_side, comparison_side)?;

        let reference_coordinates = self.reference.problem().coordinates();

        let comparison_coordinates = self.comparison.problem().coordinates();

        inputs
            .integrate(
                reference_coordinates.vacuum_angular_wavenumber(),
                comparison_coordinates.vacuum_angular_wavenumber(),
                reference_coordinates.parallel_angular_wavenumber(),
                comparison_coordinates.parallel_angular_wavenumber(),
            )
            .map_err(Into::into)
    }
}

impl<'a, J, I, ML, MR, WL, WR> HermitianPlaneWavePair<'a, J, J, I, I, ML, MR, WL, WR>
where
    J: Jet + JetMapping,
    J::Scalar: ComplexField,
    J::Dimension: Dimension,
    J::Mapping: Debug,
    I: ComplexField,
{
    /// Calculate layer-resolved Hermitian field overlaps.
    ///
    /// The reference state is conjugated and the comparison state is not:
    ///
    /// ```text
    /// electric = ∫ E_reference* · E_comparison dz
    /// magnetic = ∫ H_reference* · H_comparison dz
    /// ```
    ///
    /// The operation is elementwise across the aligned compiled sample
    /// layout. The two incident sides are independent because the retained
    /// states may be queried for different excitations.
    pub fn layer_overlap(
        &self,
        reference_side: IncidentSide,
        comparison_side: IncidentSide,
    ) -> Result<
        DifferentialResponseFor<J, Layers<HermitianLayerOverlap<J>>>,
        HermitianOverlapEvaluationError,
    >
    where
        J: RealScalarAlgebra + ScalarAlgebraExpRelExt + Clone,
        J::Scalar: ComplexScalar,
        J::Policy: Default + DerivativePartsPolicy<Layers<HermitianLayerOverlap<J>>>,
        Layers<HermitianLayerOverlap<J>>: IntoDifferentialResponse<J::Policy, J::Mapping>,
        WL: ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>
            + PlaneWaveSolutionSource,
        WR: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
    {
        let raw = self.raw_layer_overlap(reference_side, comparison_side)?;

        Ok(raw.into_differential_response(&J::Policy::default(), self.reference.mapping()))
    }

    /// Calculate the Hermitian field overlap aggregated over all finite
    /// layers.
    ///
    /// The reference state forms the conjugated operand:
    ///
    /// ```text
    /// electric = sum_layers ∫ E_reference* · E_comparison dz
    /// magnetic = sum_layers ∫ H_reference* · H_comparison dz
    /// total    = electric + magnetic
    /// ```
    ///
    /// Exterior media are not included.
    ///
    /// The operation is elementwise across the aligned compiled sample
    /// layout.
    pub fn aggregate_overlap(
        &self,
        reference_side: IncidentSide,
        comparison_side: IncidentSide,
    ) -> Result<
        DifferentialResponseFor<J, AggregateHermitianOverlap<J>>,
        HermitianOverlapEvaluationError,
    >
    where
        J: RealScalarAlgebra + ScalarAlgebraExpRelExt + Clone,
        J::Scalar: ComplexScalar,
        J::Policy: Default + DerivativePartsPolicy<AggregateHermitianOverlap<J>>,
        AggregateHermitianOverlap<J>: IntoDifferentialResponse<J::Policy, J::Mapping>,
        WL: ReconstructLayerBoundaryWaves<Algebra = J>
            + RetainedIsotropicLayers<Algebra = J>
            + PlaneWaveSolutionSource,
        WR: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
    {
        let aggregate = self
            .raw_layer_overlap(reference_side, comparison_side)?
            .aggregate()
            .map_err(HermitianOverlapError::Aggregate)?;

        Ok(aggregate.into_differential_response(&J::Policy::default(), self.reference.mapping()))
    }

    pub fn normalized_overlap(
        &self,
    ) -> Result<
        DifferentialResponseFor<J, NormalizedHermitianOverlap<J>>,
        HermitianOverlapEvaluationError,
    >
    where
        J: RealScalarAlgebra + ScalarAlgebraExpRelExt + Clone,
        J::Scalar: ComplexScalar,
        WL: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
        WR: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
    {
        let cross = self.raw_layer_overlap()?.aggregate();

        let reference_self = self.reference().raw_aggregate_self_overlap()?;

        let comparison_self = self.comparison().raw_aggregate_self_overlap()?;

        cross
            .normalized(&reference_self, &comparison_self)
            .into_differential_response(&J::Policy::default(), self.reference().state().mapping())
    }
}
