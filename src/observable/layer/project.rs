use ndarray::Dimension;
use thiserror::Error;

use crate::{
    ComplexScalar,
    algebra::{RealScalarAlgebra, ScalarAlgebraExpRelExt},
    backend::{IsotropicLayerQuantities, RetainedIsotropicLayers},
    observable::{BoundaryProjectionError, BoundaryWaves, LayerBoundaries, LayerBoundaryWaves},
};

use super::{
    IntegratedStateProducts, Layers, integrate_hermitian_wave_products,
    integration::project_integrated_state_products,
};

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RetainedLayerDatum {
    Quantities,
    Thickness,
}

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum LayerProjectionError {
    #[error("error in boundary projection {0}")]
    Boundary(#[from] BoundaryProjectionError),

    #[error(
        "boundary-wave count {wave_count} does not match retained layer count \
         {layer_count}"
    )]
    LayerCountMismatch {
        wave_count: usize,
        layer_count: usize,
    },

    #[error("missing")]
    MissingRetainedLayerDatum {
        datum: RetainedLayerDatum,
        index: usize,
        layer_count: usize,
    },
}

/// Retained data required to analytically integrate over a homogeneous layer
///
/// Directional waves are expressed at the left-boundary of the layer
pub(crate) struct LayerIntegrationInput<A> {
    waves: BoundaryWaves<A>,
    quantities: IsotropicLayerQuantities<A>,
    thickness: A,
}

impl<A> LayerIntegrationInput<A> {
    pub(crate) const fn new(
        waves: BoundaryWaves<A>,
        quantities: IsotropicLayerQuantities<A>,
        thickness: A,
    ) -> Self {
        Self {
            waves,
            quantities,
            thickness,
        }
    }

    pub(crate) fn waves(&self) -> &BoundaryWaves<A> {
        &self.waves
    }

    pub(crate) fn quantities(&self) -> &IsotropicLayerQuantities<A> {
        &self.quantities
    }

    pub(crate) fn thickness(&self) -> &A {
        &self.thickness
    }

    pub(crate) fn into_parts(self) -> (BoundaryWaves<A>, IsotropicLayerQuantities<A>, A) {
        (self.waves, self.quantities, self.thickness)
    }
}

/// Analytically integrated canonical-state products and constitutive quantities for one finite-layer
pub(crate) struct IntegratedLayerData<A> {
    state_products: IntegratedStateProducts<A>,
    quantities: IsotropicLayerQuantities<A>,
}

impl<A> IntegratedLayerData<A> {
    pub(crate) const fn new(
        state_products: IntegratedStateProducts<A>,
        quantities: IsotropicLayerQuantities<A>,
    ) -> Self {
        Self {
            state_products,
            quantities,
        }
    }

    pub(crate) fn state_products(&self) -> &IntegratedStateProducts<A> {
        &self.state_products
    }

    pub(crate) fn quantities(&self) -> &IsotropicLayerQuantities<A> {
        &self.quantities
    }

    pub(crate) fn into_parts(self) -> (IntegratedStateProducts<A>, IsotropicLayerQuantities<A>) {
        (self.state_products, self.quantities)
    }
}

/// Combine retained boundary waves, medium quantities, and thicknesses.
///
/// Records are returned in physical left-to-right finite-layer order.
pub(crate) fn assemble_layer_integration_inputs<W>(
    workspace: &W,
    boundary_waves: LayerBoundaries<LayerBoundaryWaves<W::Algebra>>,
) -> Result<Layers<LayerIntegrationInput<W::Algebra>>, LayerProjectionError>
where
    W: RetainedIsotropicLayers,
    W::Algebra: Clone,
{
    let layer_count = workspace
        .retained_layer_count()
        .ok_or(BoundaryProjectionError::LayersNotRetained)?;

    if boundary_waves.len() != layer_count {
        return Err(LayerProjectionError::LayerCountMismatch {
            wave_count: boundary_waves.len(),
            layer_count,
        });
    }

    let mut layers = Vec::with_capacity(layer_count);

    for (index, boundaries) in boundary_waves.into_inner().into_iter().enumerate() {
        let quantities = workspace
            .layer_quantities(index)
            .ok_or(LayerProjectionError::MissingRetainedLayerDatum {
                datum: RetainedLayerDatum::Quantities,
                index,
                layer_count,
            })?
            .clone();

        let thickness = workspace
            .layer_thickness(index)
            .ok_or(LayerProjectionError::MissingRetainedLayerDatum {
                datum: RetainedLayerDatum::Thickness,
                index,
                layer_count,
            })?
            .clone();

        let (left, _right) = boundaries.into_parts();

        layers.push(LayerIntegrationInput::new(left, quantities, thickness));
    }

    Ok(Layers::new(layers))
}

impl<A> LayerIntegrationInput<A> {
    fn integrate(self) -> IntegratedLayerData<A>
    where
        A: RealScalarAlgebra + ScalarAlgebraExpRelExt,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        let (waves, quantities, thickness) = self.into_parts();

        let products = integrate_hermitian_wave_products(&waves, quantities.kappa(), &thickness);

        let admittance = quantities.clone().into_admittance().into_inner();

        let state_products = project_integrated_state_products(&products, &admittance);

        IntegratedLayerData::new(state_products, quantities)
    }
}

impl<A> Layers<LayerIntegrationInput<A>> {
    pub(crate) fn integrate(self) -> Layers<IntegratedLayerData<A>>
    where
        A: RealScalarAlgebra + ScalarAlgebraExpRelExt,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        self.map(|each| each.integrate())
    }
}
