use nalgebra::ComplexField;
use ndarray::Ix0;
use thiserror::Error;

use crate::{
    ComplexScalar, FiniteLayerIndex, Polarisation,
    algebra::{Jet, RealScalarAlgebra, ScalarAlgebra, ScalarAlgebraExpRelExt},
    backend::{PlaneWaveSolutionSource, RetainedIsotropicLayers},
    derivative_parts::DerivativePartsPolicy,
    differential::IntoDifferentialResponse,
    evaluate::{
        excitation::PlaneWaveExcitation,
        query::{DifferentialResponseFor, PlaneWaveQuery},
    },
    input::JetMapping,
    observable::{
        AggregateBilinearOverlap, AggregateHermitianOverlap, BilinearLayerOverlap,
        BilinearLayerOverlapInput, BoundaryProjectionError, HermitianLayerOverlap,
        HermitianLayerOverlapInput, LayerAggregateError, LayerOverlapInput, LayerOverlapOperand,
        Layers, OverlapError,
    },
    waves::ReconstructLayerBoundaryWaves,
};

/// Operand involved in a pairwise plane-wave operation.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PairOperand {
    Reference,
    Comparison,
}

impl std::fmt::Display for PairOperand {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reference => formatter.write_str("reference"),
            Self::Comparison => formatter.write_str("comparison"),
        }
    }
}

/// Failure to form a compatible pair of scalar plane-wave excitations.
#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum PlaneWavePairError {
    #[error(
        "pairing requires matching polarizations; \
     reference is {reference:?}, comparison is {comparison:?}"
    )]
    PolarisationMismatch {
        reference: Polarisation,
        comparison: Polarisation,
    },

    #[error("reference and comparison derivative mappings are incompatible")]
    DifferentialMappingMismatch,

    #[error("{operand} excitation does not retain finite-layer analysis data")]
    LayersNotRetained { operand: PairOperand },

    #[error(
        "reference finite-layer count {reference_count} does not match \
         comparison finite-layer count {comparison_count}"
    )]
    LayerCountMismatch {
        reference_count: usize,
        comparison_count: usize,
    },

    #[error(
        "finite layer {index:?} has incompatible reference and comparison \
         thickness jets"
    )]
    LayerThicknessMismatch { index: FiniteLayerIndex },

    #[error(
        "{operand} excitation is missing retained data for finite layer \
         {index:?}; retained layer count is {layer_count}"
    )]
    MissingLayerData {
        operand: PairOperand,
        index: FiniteLayerIndex,
        layer_count: usize,
    },

    #[error(transparent)]
    Projection(#[from] OverlapError),

    #[error(transparent)]
    Aggregation(#[from] LayerAggregateError),

    #[error("failed to reconstruct reference layer boundary waves: {0}")]
    ReferenceBoundaryProjection(#[source] BoundaryProjectionError),

    #[error("failed to reconstruct comparison layer boundary waves: {0}")]
    ComparisonBoundaryProjection(#[source] BoundaryProjectionError),

    #[error(
        "reference boundary-wave count {wave_count} does not match retained \
     layer count {layer_count}"
    )]
    ReferenceWaveCountMismatch {
        wave_count: usize,
        layer_count: usize,
    },

    #[error(
        "comparison boundary-wave count {wave_count} does not match retained \
     layer count {layer_count}"
    )]
    ComparisonWaveCountMismatch {
        wave_count: usize,
        layer_count: usize,
    },
}

/// Two compatible scalar plane-wave excitations.
///
/// The reference excitation forms the left operand of pairwise contractions.
/// Hermitian operations conjugate the reference; bilinear operations do not.
#[derive(Debug)]
pub struct PlaneWaveExcitationPair<'a, J, I, ML, MR, WL, WR>
where
    J: Jet<Dimension = Ix0> + JetMapping,
    J::Scalar: ComplexField,
    I: ComplexField,
{
    reference: PlaneWaveExcitation<'a, J, I, ML, WL>,

    comparison: PlaneWaveExcitation<'a, J, I, MR, WR>,
}

impl<'a, J, I, ML, MR, WL, WR> PlaneWaveExcitationPair<'a, J, I, ML, MR, WL, WR>
where
    J: Jet<Dimension = Ix0> + JetMapping + PartialEq,
    J::Scalar: ComplexField,
    I: ComplexField,
    J::Mapping: PartialEq,
    WL: RetainedIsotropicLayers<Algebra = J>,
    WR: RetainedIsotropicLayers<Algebra = J>,
{
    pub(crate) fn new(
        reference: PlaneWaveExcitation<'a, J, I, ML, WL>,

        comparison: PlaneWaveExcitation<'a, J, I, MR, WR>,
    ) -> Result<Self, PlaneWavePairError> {
        validate_scalar_excitation_pair(&reference, &comparison)?;

        Ok(Self {
            reference,
            comparison,
        })
    }

    pub fn reference(&self) -> &PlaneWaveExcitation<'a, J, I, ML, WL> {
        &self.reference
    }

    pub fn comparison(&self) -> &PlaneWaveExcitation<'a, J, I, MR, WR> {
        &self.comparison
    }

    pub fn into_parts(
        self,
    ) -> (
        PlaneWaveExcitation<'a, J, I, ML, WL>,
        PlaneWaveExcitation<'a, J, I, MR, WR>,
    ) {
        (self.reference, self.comparison)
    }
}

impl<'a, J, I, ML, MR, WL, WR> PlaneWaveExcitationPair<'a, J, I, ML, MR, WL, WR>
where
    J: Jet<Dimension = Ix0> + JetMapping + PartialEq + Clone,
    J::Scalar: ComplexField,
    I: ComplexField,
    <J::Scalar as ComplexField>::RealField: ComplexField,
    J::Mapping: PartialEq,
    WL: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
    WR: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
{
    fn layer_inputs(&self) -> Result<Layers<LayerOverlapInput<J>>, PlaneWavePairError> {
        let reference_state = self.reference.state();

        let comparison_state = self.comparison.state();

        let reference_waves = reference_state
            .raw_layer_boundary_waves_unchecked(self.reference.incident_side())
            .map_err(PlaneWavePairError::ReferenceBoundaryProjection)?;

        let comparison_waves = comparison_state
            .raw_layer_boundary_waves_unchecked(self.comparison.incident_side())
            .map_err(PlaneWavePairError::ComparisonBoundaryProjection)?;

        let reference_count = reference_state.workspace().retained_layer_count().ok_or(
            PlaneWavePairError::LayersNotRetained {
                operand: PairOperand::Reference,
            },
        )?;

        let comparison_count = comparison_state.workspace().retained_layer_count().ok_or(
            PlaneWavePairError::LayersNotRetained {
                operand: PairOperand::Comparison,
            },
        )?;

        if reference_waves.len() != reference_count {
            return Err(PlaneWavePairError::ReferenceWaveCountMismatch {
                wave_count: reference_waves.len(),
                layer_count: reference_count,
            });
        }

        if comparison_waves.len() != comparison_count {
            return Err(PlaneWavePairError::ComparisonWaveCountMismatch {
                wave_count: comparison_waves.len(),
                layer_count: comparison_count,
            });
        }

        let paired_waves = reference_waves
            .into_inner()
            .into_iter()
            .zip(comparison_waves.into_inner());

        let mut inputs = Vec::with_capacity(reference_count);

        for (index, (reference_boundaries, comparison_boundaries)) in paired_waves.enumerate() {
            let layer_index = FiniteLayerIndex(index);

            let reference_quantities = reference_state
                .workspace()
                .layer_quantities(index)
                .ok_or(PlaneWavePairError::MissingLayerData {
                    operand: PairOperand::Reference,
                    index: layer_index,
                    layer_count: reference_count,
                })?
                .clone();

            let comparison_quantities = comparison_state
                .workspace()
                .layer_quantities(index)
                .ok_or(PlaneWavePairError::MissingLayerData {
                    operand: PairOperand::Comparison,
                    index: layer_index,
                    layer_count: comparison_count,
                })?
                .clone();

            let thickness = reference_state
                .workspace()
                .layer_thickness(index)
                .ok_or(PlaneWavePairError::MissingLayerData {
                    operand: PairOperand::Reference,
                    index: layer_index,
                    layer_count: reference_count,
                })?
                .clone();

            let (reference_left, _) = reference_boundaries.into_parts();

            let (comparison_left, _) = comparison_boundaries.into_parts();

            inputs.push(LayerOverlapInput::new(
                LayerOverlapOperand::new(reference_left, reference_quantities),
                LayerOverlapOperand::new(comparison_left, comparison_quantities),
                thickness,
            ));
        }

        Ok(Layers::new(inputs))
    }
}

impl<'a, J, R, ML, MR, WL, WR> PlaneWaveExcitationPair<'a, J, R, ML, MR, WL, WR>
where
    J: Jet<Dimension = Ix0> + JetMapping + PartialEq + Clone,
    J::Scalar: ComplexField<RealField = R>,
    R: ComplexField,
    J::Mapping: PartialEq,
    WL: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
    WR: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
{
    fn hermitian_layer_inputs(
        &self,
    ) -> Result<Layers<HermitianLayerOverlapInput<J>>, PlaneWavePairError> {
        self.layer_inputs()
            .map(|layers| layers.map(|each| each.into_hermitian()))
    }

    fn raw_layer_hermitian_overlap(
        &self,
    ) -> Result<Layers<HermitianLayerOverlap<J>>, PlaneWavePairError>
    where
        J: RealScalarAlgebra + ScalarAlgebraExpRelExt,
        J::Scalar: ComplexScalar,
    {
        let reference_coordinates = self.reference.state().problem().coordinates();

        let comparison_coordinates = self.comparison.state().problem().coordinates();

        self.hermitian_layer_inputs()?
            .integrate(
                reference_coordinates.vacuum_angular_wavenumber(),
                comparison_coordinates.vacuum_angular_wavenumber(),
                reference_coordinates.parallel_angular_wavenumber(),
                comparison_coordinates.parallel_angular_wavenumber(),
            )
            .map_err(Into::into)
    }

    fn raw_aggregate_hermitian_overlap(
        &self,
    ) -> Result<AggregateHermitianOverlap<J>, PlaneWavePairError>
    where
        J: RealScalarAlgebra + ScalarAlgebraExpRelExt,
        J::Scalar: ComplexScalar,
    {
        Ok(self.raw_layer_hermitian_overlap()?.aggregate()?)
    }

    pub fn layer_hermitian_overlap(
        &self,
    ) -> Result<DifferentialResponseFor<J, Layers<HermitianLayerOverlap<J>>>, PlaneWavePairError>
    where
        J: RealScalarAlgebra + ScalarAlgebraExpRelExt,
        J::Scalar: ComplexScalar,
        J::Policy: DerivativePartsPolicy<Layers<HermitianLayerOverlap<J>>>,
        Layers<HermitianLayerOverlap<J>>: IntoDifferentialResponse<J::Policy, J::Mapping>,
        WL: PlaneWaveSolutionSource,
    {
        Ok(self
            .raw_layer_hermitian_overlap()?
            .into_differential_response(&J::Policy::default(), self.reference().state().mapping()))
    }

    pub fn aggregate_hermitian_overlap(
        &self,
    ) -> Result<DifferentialResponseFor<J, AggregateHermitianOverlap<J>>, PlaneWavePairError>
    where
        J: RealScalarAlgebra + ScalarAlgebraExpRelExt,
        J::Scalar: ComplexScalar,
        J::Policy: DerivativePartsPolicy<AggregateHermitianOverlap<J>>,
        AggregateHermitianOverlap<J>: IntoDifferentialResponse<J::Policy, J::Mapping>,
        WL: PlaneWaveSolutionSource,
    {
        Ok(self
            .raw_aggregate_hermitian_overlap()?
            .into_differential_response(&J::Policy::default(), self.reference().state().mapping()))
    }
}

impl<'a, J, ML, MR, WL, WR> PlaneWaveExcitationPair<'a, J, J::Scalar, ML, MR, WL, WR>
where
    J: Jet<Dimension = Ix0> + JetMapping + PartialEq + Clone,
    J::Scalar: ComplexField,
    <J::Scalar as ComplexField>::RealField: ComplexField,
    J::Mapping: PartialEq,
    WL: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
    WR: ReconstructLayerBoundaryWaves<Algebra = J> + RetainedIsotropicLayers<Algebra = J>,
{
    fn bilinear_layer_inputs(
        &self,
    ) -> Result<Layers<BilinearLayerOverlapInput<J>>, PlaneWavePairError> {
        self.layer_inputs()
            .map(|layers| layers.map(|each| each.into_bilinear()))
    }

    fn raw_layer_bilinear_overlap(
        &self,
    ) -> Result<Layers<BilinearLayerOverlap<J>>, PlaneWavePairError>
    where
        J: ScalarAlgebra + ScalarAlgebraExpRelExt,
        J::Scalar: ComplexScalar,
    {
        let reference_coordinates = self.reference.state().problem().coordinates();

        let comparison_coordinates = self.comparison.state().problem().coordinates();

        self.bilinear_layer_inputs()?
            .integrate(
                reference_coordinates.vacuum_angular_wavenumber(),
                comparison_coordinates.vacuum_angular_wavenumber(),
                reference_coordinates.parallel_angular_wavenumber(),
                comparison_coordinates.parallel_angular_wavenumber(),
            )
            .map_err(Into::into)
    }

    fn raw_aggregate_bilinear_overlap(
        &self,
    ) -> Result<AggregateBilinearOverlap<J>, PlaneWavePairError>
    where
        J: ScalarAlgebra + ScalarAlgebraExpRelExt,
        J::Scalar: ComplexScalar,
    {
        Ok(self.raw_layer_bilinear_overlap()?.aggregate()?)
    }

    /// Calculate the layer-resolved bilinear field overlap.
    ///
    /// Neither operand is conjugated:
    ///
    /// ```text
    /// electric = ∫ E_reference · E_comparison dz
    /// magnetic = ∫ H_reference · H_comparison dz
    /// total    = electric + magnetic
    /// ```
    ///
    /// The result contains one record per finite layer, in physical
    /// left-to-right order. Exterior media are not included.
    ///
    /// This is an unweighted field overlap. It is not by itself a complete
    /// dispersive quasinormal-mode normalization.
    pub fn layer_bilinear_overlap(
        &self,
    ) -> Result<DifferentialResponseFor<J, Layers<BilinearLayerOverlap<J>>>, PlaneWavePairError>
    where
        J: ScalarAlgebra + ScalarAlgebraExpRelExt,
        J::Scalar: ComplexScalar,
        J::Policy: DerivativePartsPolicy<Layers<BilinearLayerOverlap<J>>>,
        Layers<BilinearLayerOverlap<J>>: IntoDifferentialResponse<J::Policy, J::Mapping>,
        WL: PlaneWaveSolutionSource,
    {
        Ok(self
            .raw_layer_bilinear_overlap()?
            .into_differential_response(&J::Policy::default(), self.reference().state().mapping()))
    }

    /// Calculate the bilinear field overlap aggregated over all finite layers.
    ///
    /// Neither operand is conjugated. Aggregation is performed on the full
    /// jet-valued layer results before differential decomposition.
    pub fn aggregate_bilinear_overlap(
        &self,
    ) -> Result<DifferentialResponseFor<J, AggregateBilinearOverlap<J>>, PlaneWavePairError>
    where
        J: ScalarAlgebra + ScalarAlgebraExpRelExt,
        J::Scalar: ComplexScalar,
        J::Policy: DerivativePartsPolicy<AggregateBilinearOverlap<J>>,
        AggregateBilinearOverlap<J>: IntoDifferentialResponse<J::Policy, J::Mapping>,
        WL: PlaneWaveSolutionSource,
    {
        Ok(self
            .raw_aggregate_bilinear_overlap()?
            .into_differential_response(&J::Policy::default(), self.reference().state().mapping()))
    }
}

/// Validate two scalar plane-wave excitations for pairwise layer integration.
///
/// The excitations may have different canonical coordinates and constitutive
/// quantities. They must describe the same finite-layer partition and use
/// compatible derivative mappings.
pub(crate) fn validate_scalar_excitation_pair<J, I, ML, MR, WL, WR>(
    reference: &PlaneWaveExcitation<'_, J, I, ML, WL>,
    comparison: &PlaneWaveExcitation<'_, J, I, MR, WR>,
) -> Result<(), PlaneWavePairError>
where
    J: Jet<Dimension = Ix0> + JetMapping + PartialEq,
    J::Scalar: ComplexField,
    I: ComplexField,
    J::Mapping: PartialEq,
    WL: RetainedIsotropicLayers<Algebra = J>,
    WR: RetainedIsotropicLayers<Algebra = J>,
{
    validate_polarisation(reference, comparison)?;

    validate_mapping(reference, comparison)?;

    validate_layer_geometry(reference, comparison)
}

fn validate_polarisation<J, I, ML, MR, WL, WR>(
    reference: &PlaneWaveExcitation<'_, J, I, ML, WL>,
    comparison: &PlaneWaveExcitation<'_, J, I, MR, WR>,
) -> Result<(), PlaneWavePairError>
where
    J: Jet<Dimension = Ix0> + JetMapping,
    J::Scalar: ComplexField,
    I: ComplexField,
{
    let reference_polarisation = reference.state().polarisation();

    let comparison_polarisation = comparison.state().polarisation();

    if reference_polarisation != comparison_polarisation {
        return Err(PlaneWavePairError::PolarisationMismatch {
            reference: reference_polarisation,
            comparison: comparison_polarisation,
        });
    }

    Ok(())
}

fn validate_mapping<J, I, ML, MR, WL, WR>(
    reference: &PlaneWaveExcitation<'_, J, I, ML, WL>,
    comparison: &PlaneWaveExcitation<'_, J, I, MR, WR>,
) -> Result<(), PlaneWavePairError>
where
    J: Jet<Dimension = Ix0> + JetMapping,
    J::Scalar: ComplexField,
    I: ComplexField,
    J::Mapping: PartialEq,
{
    let reference_mapping = reference.state().context().mapping();

    let comparison_mapping = comparison.state().context().mapping();

    if reference_mapping != comparison_mapping {
        return Err(PlaneWavePairError::DifferentialMappingMismatch);
    }

    Ok(())
}

fn validate_layer_geometry<J, I, ML, MR, WL, WR>(
    reference: &PlaneWaveExcitation<'_, J, I, ML, WL>,
    comparison: &PlaneWaveExcitation<'_, J, I, MR, WR>,
) -> Result<(), PlaneWavePairError>
where
    J: Jet<Dimension = Ix0> + JetMapping + PartialEq,
    J::Scalar: ComplexField,
    I: ComplexField,
    WL: RetainedIsotropicLayers<Algebra = J>,
    WR: RetainedIsotropicLayers<Algebra = J>,
{
    let reference_workspace = reference.state().workspace();

    let comparison_workspace = comparison.state().workspace();

    let reference_count = reference_workspace.retained_layer_count().ok_or(
        PlaneWavePairError::LayersNotRetained {
            operand: PairOperand::Reference,
        },
    )?;

    let comparison_count = comparison_workspace.retained_layer_count().ok_or(
        PlaneWavePairError::LayersNotRetained {
            operand: PairOperand::Comparison,
        },
    )?;

    if reference_count != comparison_count {
        return Err(PlaneWavePairError::LayerCountMismatch {
            reference_count,
            comparison_count,
        });
    }

    for index in 0..reference_count {
        let layer_index = FiniteLayerIndex(index);

        let reference_thickness = reference_workspace.layer_thickness(index).ok_or(
            PlaneWavePairError::MissingLayerData {
                operand: PairOperand::Reference,
                index: layer_index,
                layer_count: reference_count,
            },
        )?;

        let comparison_thickness = comparison_workspace.layer_thickness(index).ok_or(
            PlaneWavePairError::MissingLayerData {
                operand: PairOperand::Comparison,
                index: layer_index,
                layer_count: comparison_count,
            },
        )?;

        if reference_thickness != comparison_thickness {
            return Err(PlaneWavePairError::LayerThicknessMismatch { index: layer_index });
        }
    }

    Ok(())
}
