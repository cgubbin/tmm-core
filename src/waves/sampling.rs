use nalgebra::ComplexField;
use ndarray::Dimension;

use crate::{
    ComplexScalar, FiniteLayerIndex,
    algebra::ScalarAlgebra,
    spatial::{CanonicalFieldPosition, CompiledFieldSampling},
    waves::{BidirectionalWaves, ExteriorBoundaryWaves, LayerBoundaryWaves},
};

use super::{PropagateLayerWaves, PropagateWaves};

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub(crate) enum WaveSamplingError {
    #[error(
        "sampling requested finite layer {requested}, \
         but propagation data contain {layer_count} layers"
    )]
    LayerOutOfBounds {
        requested: usize,
        layer_count: usize,
    },
}

/// Propagation data for one finite layer.
#[derive(Clone, Copy, Debug)]
pub(crate) struct LayerPropagationData<'a, A> {
    waves: &'a LayerBoundaryWaves<A>,
    longitudinal_wavevector: &'a A,
    thickness: &'a A,
}

impl<'a, A> LayerPropagationData<'a, A> {
    pub(crate) const fn new(
        waves: &'a LayerBoundaryWaves<A>,
        longitudinal_wavevector: &'a A,
        thickness: &'a A,
    ) -> Self {
        Self {
            waves,
            longitudinal_wavevector,
            thickness,
        }
    }

    pub(crate) const fn waves(&self) -> &'a LayerBoundaryWaves<A> {
        self.waves
    }

    pub(crate) const fn longitudinal_wavevector(&self) -> &'a A {
        self.longitudinal_wavevector
    }

    pub(crate) const fn thickness(&self) -> &'a A {
        self.thickness
    }
}

/// Wave data required to reconstruct amplitudes throughout a complete stack.
#[derive(Clone, Copy, Debug)]
pub(crate) struct WavePropagationData<'a, A> {
    exterior_waves: &'a ExteriorBoundaryWaves<A>,
    left_longitudinal_wavevector: &'a A,
    layers: &'a [LayerPropagationData<'a, A>],
    right_longitudinal_wavevector: &'a A,
}

impl<'a, A> WavePropagationData<'a, A> {
    pub(crate) const fn new(
        exterior_waves: &'a ExteriorBoundaryWaves<A>,
        left_longitudinal_wavevector: &'a A,
        layers: &'a [LayerPropagationData<'a, A>],
        right_longitudinal_wavevector: &'a A,
    ) -> Self {
        Self {
            exterior_waves,
            left_longitudinal_wavevector,
            layers,
            right_longitudinal_wavevector,
        }
    }

    pub(crate) const fn exterior_waves(&self) -> &'a ExteriorBoundaryWaves<A> {
        self.exterior_waves
    }

    pub(crate) fn layer(&self, index: FiniteLayerIndex) -> Option<&LayerPropagationData<'a, A>> {
        self.layers.get(index.0)
    }

    pub(crate) const fn left_longitudinal_wavevector(&self) -> &'a A {
        self.left_longitudinal_wavevector
    }

    pub(crate) const fn right_longitudinal_wavevector(&self) -> &'a A {
        self.right_longitudinal_wavevector
    }

    pub(crate) const fn layer_count(&self) -> usize {
        self.layers.len()
    }
}

pub(crate) fn propagate_sampling<A>(
    sampling: &CompiledFieldSampling<<A::Scalar as ComplexField>::RealField>,
    data: &WavePropagationData<'_, A>,
) -> Result<Vec<BidirectionalWaves<A>>, WaveSamplingError>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    <A::Scalar as ComplexField>::RealField:
        Copy + std::ops::Neg<Output = <A::Scalar as ComplexField>::RealField>,
    A::Dimension: Dimension,
{
    sampling
        .positions()
        .iter()
        .copied()
        .map(|position| propagate_at_position(position, data))
        .collect()
}

pub(crate) fn propagate_at_position<A>(
    position: CanonicalFieldPosition<<A::Scalar as ComplexField>::RealField>,
    data: &WavePropagationData<'_, A>,
) -> Result<BidirectionalWaves<A>, WaveSamplingError>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    <A::Scalar as ComplexField>::RealField:
        Copy + std::ops::Neg<Output = <A::Scalar as ComplexField>::RealField>,
    A::Dimension: Dimension,
{
    match position {
        CanonicalFieldPosition::LeftExterior { distance } => Ok(data
            .exterior_waves()
            .left()
            .propagate(data.left_longitudinal_wavevector(), -distance)),

        CanonicalFieldPosition::Layer { index, offset } => {
            let layer = data
                .layer(index)
                .ok_or(WaveSamplingError::LayerOutOfBounds {
                    requested: index.0,
                    layer_count: data.layers.len(),
                })?;

            Ok(layer.waves.propagate_to_offset(
                layer.longitudinal_wavevector(),
                layer.thickness(),
                offset,
            ))
        }

        CanonicalFieldPosition::RightExterior { distance } => Ok(data
            .exterior_waves()
            .right()
            .propagate(data.right_longitudinal_wavevector(), distance)),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        algebra::{ArrayJet0, Jet0, RealParameter},
        parameter::FiniteLayerIndex,
        spatial::{CanonicalFieldPosition, CompiledFieldSampling},
        test_support::{TOLERANCE, assertions::assert_complex_close},
        waves::{BidirectionalWaves, ExteriorBoundaryWaves, LayerBoundaryWaves},
    };

    type C = Complex64;
    type A = ArrayJet0<C, Ix0, RealParameter>;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn jet(value: C) -> A {
        Jet0::new(arr0(value))
    }

    fn real_jet(value: f64) -> A {
        jet(c(value, 0.0))
    }

    fn waves(forward: C, backward: C) -> BidirectionalWaves<A> {
        BidirectionalWaves::new(jet(forward), jet(backward))
    }

    fn layer_waves(offset: f64) -> LayerBoundaryWaves<A> {
        LayerBoundaryWaves::new(
            waves(c(offset + 1.0, 0.1), c(offset + 2.0, 0.2)),
            waves(c(offset + 3.0, 0.3), c(offset + 4.0, 0.4)),
        )
    }

    fn assert_jet_close(actual: &A, expected: &A) {
        assert_complex_close(actual.value()[()], expected.value()[()], TOLERANCE);
    }

    fn assert_waves_close(actual: &BidirectionalWaves<A>, expected: &BidirectionalWaves<A>) {
        assert_jet_close(actual.forward(), expected.forward());
        assert_jet_close(actual.backward(), expected.backward());
    }

    struct Fixture {
        exterior: ExteriorBoundaryWaves<A>,

        left_kappa: A,
        right_kappa: A,

        layer0_waves: LayerBoundaryWaves<A>,
        layer0_kappa: A,
        layer0_thickness: A,

        layer1_waves: LayerBoundaryWaves<A>,
        layer1_kappa: A,
        layer1_thickness: A,
    }

    impl Fixture {
        fn new() -> Self {
            Self {
                exterior: ExteriorBoundaryWaves::new(
                    waves(c(1.0, 0.1), c(2.0, 0.2)),
                    waves(c(3.0, 0.3), c(4.0, 0.4)),
                ),

                left_kappa: real_jet(1.1),
                right_kappa: real_jet(1.7),

                layer0_waves: layer_waves(10.0),
                layer0_kappa: real_jet(2.1),
                layer0_thickness: real_jet(0.4),

                layer1_waves: layer_waves(20.0),
                layer1_kappa: real_jet(2.7),
                layer1_thickness: real_jet(0.8),
            }
        }

        fn layers(&self) -> [LayerPropagationData<'_, A>; 2] {
            [
                LayerPropagationData::new(
                    &self.layer0_waves,
                    &self.layer0_kappa,
                    &self.layer0_thickness,
                ),
                LayerPropagationData::new(
                    &self.layer1_waves,
                    &self.layer1_kappa,
                    &self.layer1_thickness,
                ),
            ]
        }
    }

    #[test]
    fn left_exterior_dispatches_to_left_boundary() {
        let fixture = Fixture::new();
        let layers = fixture.layers();

        let data = WavePropagationData::new(
            &fixture.exterior,
            &fixture.left_kappa,
            &layers,
            &fixture.right_kappa,
        );

        let distance = 0.25;

        let actual =
            propagate_at_position(CanonicalFieldPosition::LeftExterior { distance }, &data)
                .unwrap();

        let expected = fixture
            .exterior
            .left()
            .propagate(&fixture.left_kappa, -distance);

        assert_waves_close(&actual, &expected);
    }

    #[test]
    fn right_exterior_dispatches_to_right_boundary() {
        let fixture = Fixture::new();
        let layers = fixture.layers();

        let data = WavePropagationData::new(
            &fixture.exterior,
            &fixture.left_kappa,
            &layers,
            &fixture.right_kappa,
        );

        let distance = 0.35;

        let actual =
            propagate_at_position(CanonicalFieldPosition::RightExterior { distance }, &data)
                .unwrap();

        let expected = fixture
            .exterior
            .right()
            .propagate(&fixture.right_kappa, distance);

        assert_waves_close(&actual, &expected);
    }

    #[test]
    fn finite_layer_dispatches_by_layer_index() {
        let fixture = Fixture::new();
        let layers = fixture.layers();

        let data = WavePropagationData::new(
            &fixture.exterior,
            &fixture.left_kappa,
            &layers,
            &fixture.right_kappa,
        );

        let offset = 0.3;

        let actual = propagate_at_position(
            CanonicalFieldPosition::Layer {
                index: FiniteLayerIndex(1),
                offset,
            },
            &data,
        )
        .unwrap();

        let expected = fixture.layer1_waves.propagate_to_offset(
            &fixture.layer1_kappa,
            &fixture.layer1_thickness,
            offset,
        );

        assert_waves_close(&actual, &expected);
    }

    #[test]
    fn finite_layer_rejects_missing_layer_data() {
        let fixture = Fixture::new();
        let layers = fixture.layers();

        let data = WavePropagationData::new(
            &fixture.exterior,
            &fixture.left_kappa,
            &layers,
            &fixture.right_kappa,
        );

        let error = propagate_at_position(
            CanonicalFieldPosition::Layer {
                index: FiniteLayerIndex(2),
                offset: 0.1,
            },
            &data,
        )
        .unwrap_err();

        assert_eq!(
            error,
            WaveSamplingError::LayerOutOfBounds {
                requested: 2,
                layer_count: 2,
            },
        );
    }

    #[test]
    fn empty_sampling_produces_empty_wave_sequence() {
        let fixture = Fixture::new();
        let layers = fixture.layers();

        let data = WavePropagationData::new(
            &fixture.exterior,
            &fixture.left_kappa,
            &layers,
            &fixture.right_kappa,
        );

        let sampling = CompiledFieldSampling::new(Vec::new());

        let sampled = propagate_sampling(&sampling, &data).unwrap();

        assert!(sampled.is_empty());
    }

    #[test]
    fn sampling_preserves_requested_order_and_duplicates() {
        let fixture = Fixture::new();
        let layers = fixture.layers();

        let data = WavePropagationData::new(
            &fixture.exterior,
            &fixture.left_kappa,
            &layers,
            &fixture.right_kappa,
        );

        let positions = vec![
            CanonicalFieldPosition::RightExterior { distance: 0.2 },
            CanonicalFieldPosition::Layer {
                index: FiniteLayerIndex(1),
                offset: 0.3,
            },
            CanonicalFieldPosition::Layer {
                index: FiniteLayerIndex(0),
                offset: 0.1,
            },
            CanonicalFieldPosition::LeftExterior { distance: 0.4 },
            CanonicalFieldPosition::Layer {
                index: FiniteLayerIndex(1),
                offset: 0.3,
            },
        ];

        let expected = positions
            .iter()
            .copied()
            .map(|position| propagate_at_position(position, &data).unwrap())
            .collect::<Vec<_>>();

        let sampling = CompiledFieldSampling::new(positions);

        let actual = propagate_sampling(&sampling, &data).unwrap();

        assert_eq!(actual.len(), expected.len());

        for (actual, expected) in actual.iter().zip(&expected) {
            assert_waves_close(actual, expected);
        }
    }
}
