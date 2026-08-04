use ndarray::Dimension;
use thiserror::Error;

use crate::{
    ComplexScalar,
    algebra::{RealScalarAlgebra, ScalarAlgebraExpRelExt},
    backend::{IsotropicLayerQuantities, RetainedIsotropicLayers},
    observable::{BoundaryProjectionError, BoundaryWaves, LayerBoundaries, LayerBoundaryWaves},
};

use super::{
    IntegratedHermitianStateProducts, Layers, integrate_hermitian_wave_products,
    integration::project_integrated_hermitian_state_products,
};

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

    #[error(
        "retained quantity is unavailable for finite layer {index}; \
     retained layer count is {layer_count}"
    )]
    MissingLayerQuantities { index: usize, layer_count: usize },

    #[error(
        "retained thickness is unavailable for finite layer {index}; \
     retained layer count is {layer_count}"
    )]
    MissingLayerThickness { index: usize, layer_count: usize },

    /// The backend result does not contain retained internal-layer data.
    #[error("the backend result does not retain internal layer data")]
    LayersNotRetained,
}

/// Retained data required to analytically integrate over a homogeneous layer.
///
/// Directional waves are expressed at the layer's left boundary.
#[derive(Clone, Debug)]
pub(crate) struct LayerIntegrationInput<A> {
    waves: BoundaryWaves<A>,
    quantities: IsotropicLayerQuantities<A>,
    thickness: A,
}

impl<A> LayerIntegrationInput<A> {
    const fn new(
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

    fn into_parts(self) -> (BoundaryWaves<A>, IsotropicLayerQuantities<A>, A) {
        (self.waves, self.quantities, self.thickness)
    }
}

/// Analytically integrated Hermitian canonical-state products and
/// constitutive quantities for one finite layer.
///
/// This type is specific to real-input Hermitian analysis. The original wave
/// products and thickness are discarded after the canonical-state products
/// have been constructed.
#[derive(Clone, Debug)]
pub(crate) struct IntegratedLayerData<A> {
    state_products: IntegratedHermitianStateProducts<A>,
    quantities: IsotropicLayerQuantities<A>,
}

impl<A> IntegratedLayerData<A> {
    pub(super) const fn new(
        state_products: IntegratedHermitianStateProducts<A>,
        quantities: IsotropicLayerQuantities<A>,
    ) -> Self {
        Self {
            state_products,
            quantities,
        }
    }

    pub(crate) fn state_products(&self) -> &IntegratedHermitianStateProducts<A> {
        &self.state_products
    }

    pub(crate) fn quantities(&self) -> &IsotropicLayerQuantities<A> {
        &self.quantities
    }

    pub(super) fn into_parts(
        self,
    ) -> (
        IntegratedHermitianStateProducts<A>,
        IsotropicLayerQuantities<A>,
    ) {
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
        .ok_or(LayerProjectionError::LayersNotRetained)?;

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
            .ok_or(LayerProjectionError::MissingLayerQuantities { index, layer_count })?
            .clone();

        let thickness = workspace
            .layer_thickness(index)
            .ok_or(LayerProjectionError::MissingLayerThickness { index, layer_count })?
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

        let admittance = quantities.admittance().into_inner();

        let state_products = project_integrated_hermitian_state_products(&products, &admittance);

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
        self.map(LayerIntegrationInput::integrate)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        Polarisation,
        algebra::{ArrayJet0, Jet0, RealParameter},
        backend::IsotropicLayerQuantities,
        observable::{BoundaryWaves, LayerBoundaryWaves},
        test_support::{C, TOLERANCE, assertions::assert_complex_close},
    };

    type A = ArrayJet0<Complex64, Ix0, RealParameter>;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn jet(value: C) -> A {
        Jet0::new(arr0(value))
    }

    fn scalar(value: &A) -> C {
        value.value()[()]
    }

    fn waves(offset: f64) -> LayerBoundaryWaves<A> {
        LayerBoundaryWaves::new(
            BoundaryWaves::new(jet(c(offset + 1.0, 0.0)), jet(c(offset + 2.0, 0.0))),
            BoundaryWaves::new(jet(c(offset + 3.0, 0.0)), jet(c(offset + 4.0, 0.0))),
        )
    }

    fn quantities(offset: f64) -> IsotropicLayerQuantities<A> {
        IsotropicLayerQuantities::test_fixture(
            jet(c(offset + 2.0, 0.1)),
            jet(c(offset + 3.0, 0.2)),
            jet(c(offset + 5.0, 0.3)),
            Polarisation::TransverseElectric,
        )
    }

    #[derive(Clone, Debug)]
    struct TestWorkspace {
        retained_count: Option<usize>,
        quantities: Vec<Option<IsotropicLayerQuantities<A>>>,
        thicknesses: Vec<Option<A>>,
    }

    impl TestWorkspace {
        fn consistent(layer_count: usize) -> Self {
            Self {
                retained_count: Some(layer_count),
                quantities: (0..layer_count)
                    .map(|index| Some(quantities(index as f64 * 10.0)))
                    .collect(),
                thicknesses: (0..layer_count)
                    .map(|index| Some(jet(c(0.4 + index as f64, 0.0))))
                    .collect(),
            }
        }
    }

    impl RetainedIsotropicLayers for TestWorkspace {
        type Algebra = A;

        fn retained_layer_count(&self) -> Option<usize> {
            self.retained_count
        }

        fn layer_quantities(
            &self,
            index: usize,
        ) -> Option<&IsotropicLayerQuantities<Self::Algebra>> {
            self.quantities.get(index).and_then(Option::as_ref)
        }

        fn layer_thickness(&self, index: usize) -> Option<&Self::Algebra> {
            self.thicknesses.get(index).and_then(Option::as_ref)
        }
    }

    #[test]
    fn layer_integration_input_into_parts_preserves_order() {
        let input = LayerIntegrationInput::new(
            BoundaryWaves::new(1, 2),
            IsotropicLayerQuantities::test_fixture(3, 4, 5, Polarisation::TransverseElectric),
            6,
        );

        let (waves, quantities, thickness) = input.into_parts();

        assert_eq!(waves.into_parts(), (1, 2));
        assert_eq!(quantities.kappa(), &3);
        assert_eq!(thickness, 6);
    }

    #[test]
    fn assembly_returns_one_input_per_finite_layer() {
        let workspace = TestWorkspace::consistent(2);

        let result = assemble_layer_integration_inputs(
            &workspace,
            LayerBoundaries::new(vec![waves(0.0), waves(10.0)]),
        )
        .expect("consistent retained data should assemble");

        assert_eq!(result.len(), 2);
    }

    #[test]
    fn assembly_preserves_physical_layer_order() {
        let workspace = TestWorkspace::consistent(2);

        let result = assemble_layer_integration_inputs(
            &workspace,
            LayerBoundaries::new(vec![waves(0.0), waves(10.0)]),
        )
        .unwrap();

        let first = result.first().unwrap();
        let second = result.last().unwrap();

        assert_eq!(scalar(first.waves().forward()), c(1.0, 0.0),);

        assert_eq!(scalar(second.waves().forward()), c(11.0, 0.0),);

        assert_eq!(scalar(first.thickness()), c(0.4, 0.0),);

        assert_eq!(scalar(second.thickness()), c(1.4, 0.0),);
    }

    #[test]
    fn assembly_uses_left_boundary_waves() {
        let workspace = TestWorkspace::consistent(1);

        let result =
            assemble_layer_integration_inputs(&workspace, LayerBoundaries::new(vec![waves(20.0)]))
                .unwrap();

        let input = result.first().unwrap();

        assert_eq!(scalar(input.waves().forward()), c(21.0, 0.0),);

        assert_eq!(scalar(input.waves().backward()), c(22.0, 0.0),);

        /*
         * The right-boundary markers are 23 and 24.
         */
        assert_ne!(scalar(input.waves().forward()), c(23.0, 0.0),);
    }

    #[test]
    fn empty_retained_stack_assembles_empty_sequence() {
        let workspace = TestWorkspace::consistent(0);

        let result =
            assemble_layer_integration_inputs(&workspace, LayerBoundaries::new(Vec::new()))
                .unwrap();

        assert!(result.is_empty());
    }

    #[test]
    fn assembly_rejects_unavailable_retention() {
        let workspace = TestWorkspace {
            retained_count: None,
            quantities: Vec::new(),
            thicknesses: Vec::new(),
        };

        let error = assemble_layer_integration_inputs(&workspace, LayerBoundaries::new(Vec::new()))
            .expect_err("missing retained analysis data must fail");

        assert_eq!(error, LayerProjectionError::LayersNotRetained,);
    }

    #[test]
    fn assembly_rejects_too_few_boundary_wave_records() {
        let workspace = TestWorkspace::consistent(2);

        let error =
            assemble_layer_integration_inputs(&workspace, LayerBoundaries::new(vec![waves(0.0)]))
                .expect_err("wave count must match retained layer count");

        assert_eq!(
            error,
            LayerProjectionError::LayerCountMismatch {
                wave_count: 1,
                layer_count: 2,
            },
        );
    }

    #[test]
    fn assembly_rejects_too_many_boundary_wave_records() {
        let workspace = TestWorkspace::consistent(1);

        let error = assemble_layer_integration_inputs(
            &workspace,
            LayerBoundaries::new(vec![waves(0.0), waves(10.0)]),
        )
        .expect_err("wave count must match retained layer count");

        assert_eq!(
            error,
            LayerProjectionError::LayerCountMismatch {
                wave_count: 2,
                layer_count: 1,
            },
        );
    }

    #[test]
    fn assembly_rejects_missing_layer_quantities() {
        let mut workspace = TestWorkspace::consistent(2);

        workspace.quantities[1] = None;

        let error = assemble_layer_integration_inputs(
            &workspace,
            LayerBoundaries::new(vec![waves(0.0), waves(10.0)]),
        )
        .expect_err("missing retained quantities must fail");

        assert_eq!(
            error,
            LayerProjectionError::MissingLayerQuantities {
                index: 1,
                layer_count: 2,
            },
        );
    }

    #[test]
    fn assembly_rejects_missing_layer_thickness() {
        let mut workspace = TestWorkspace::consistent(2);

        workspace.thicknesses[0] = None;

        let error = assemble_layer_integration_inputs(
            &workspace,
            LayerBoundaries::new(vec![waves(0.0), waves(10.0)]),
        )
        .expect_err("missing retained thickness must fail");

        assert_eq!(
            error,
            LayerProjectionError::MissingLayerThickness {
                index: 0,
                layer_count: 2,
            },
        );
    }

    #[test]
    fn one_input_integrates_to_one_record() {
        let input = LayerIntegrationInput::new(
            BoundaryWaves::new(jet(c(0.8, 0.3)), jet(c(-0.2, 0.5))),
            quantities(0.0),
            jet(c(1.7, 0.0)),
        );

        let integrated = input.integrate();

        assert_eq!(
            integrated.quantities().polarisation(),
            Polarisation::TransverseElectric,
        );
    }

    #[test]
    fn integration_matches_direct_wave_and_state_projection() {
        let input_for_actual = LayerIntegrationInput::new(
            BoundaryWaves::new(jet(c(0.8, 0.3)), jet(c(-0.2, 0.5))),
            quantities(0.0),
            jet(c(1.7, 0.0)),
        );

        let input_for_expected = LayerIntegrationInput::new(
            BoundaryWaves::new(jet(c(0.8, 0.3)), jet(c(-0.2, 0.5))),
            quantities(0.0),
            jet(c(1.7, 0.0)),
        );

        let (waves, quantities, thickness) = input_for_expected.into_parts();

        let wave_products =
            integrate_hermitian_wave_products(&waves, quantities.kappa(), &thickness);

        let admittance = quantities.admittance().into_inner();

        let expected = project_integrated_hermitian_state_products(&wave_products, &admittance);

        let actual = input_for_actual.integrate();

        let actual_state = actual.state_products();

        for (actual, expected) in [
            (actual_state.field_field(), expected.field_field()),
            (
                actual_state.secondary_secondary(),
                expected.secondary_secondary(),
            ),
            (actual_state.field_secondary(), expected.field_secondary()),
            (actual_state.secondary_field(), expected.secondary_field()),
        ] {
            assert_complex_close(scalar(actual), scalar(expected), TOLERANCE);
        }
    }

    #[test]
    fn integration_sequence_preserves_count_and_order() {
        let inputs = Layers::new(vec![
            LayerIntegrationInput::new(
                BoundaryWaves::new(jet(c(1.0, 0.0)), jet(c(0.0, 0.0))),
                quantities(0.0),
                jet(c(0.5, 0.0)),
            ),
            LayerIntegrationInput::new(
                BoundaryWaves::new(jet(c(2.0, 0.0)), jet(c(0.0, 0.0))),
                quantities(10.0),
                jet(c(0.5, 0.0)),
            ),
        ]);

        let integrated = inputs.integrate();

        assert_eq!(integrated.len(), 2);

        let first = integrated.first().unwrap();
        let second = integrated.last().unwrap();

        assert!(
            scalar(first.state_products().field_field(),).re
                < scalar(second.state_products().field_field(),).re,
            "larger forward amplitude should remain in the second record",
        );
    }
}
