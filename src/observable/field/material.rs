use std::marker::PhantomData;

use ndarray::Dimension;

use crate::{
    algebra::JetStack,
    backend::{
        ExteriorContextProvider, PlaneWaveEntries, PlaneWaveSolutionSource, RetainedIsotropicLayers,
    },
    observable::{FieldReconstructionError, IsotropicConstitutiveParameters},
    spatial::{FieldPosition, FieldSamplingError, ResolvedFieldSampling},
};

#[derive(Debug, thiserror::Error)]
pub enum ConstitutiveFieldReconstructionError<R> {
    #[error(transparent)]
    Field(#[from] FieldReconstructionError<R>),

    #[error(transparent)]
    Constitutive(#[from] ConstitutiveSamplingError),

    #[error(transparent)]
    FieldSampling(#[from] FieldSamplingError<R>),
}

/// Errors produced while sampling constitutive parameters over a resolved
/// spatial field request.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum ConstitutiveSamplingError {
    /// A finite-layer position referred to retained data which were absent.
    #[error("retained constitutive data are missing for finite layer {index}")]
    MissingLayerData { index: usize },

    /// No spatial positions were supplied.
    #[error("constitutive sampling request is empty")]
    EmptySampling,

    /// The sampled constitutive jets could not be stacked along the spatial
    /// axis.
    #[error("failed to stack sampled constitutive parameters")]
    Shape(#[from] ndarray::ShapeError),
}

/// Samples the constitutive parameters used by a retained plane-wave solve.
///
/// Sampling does not re-evaluate material models. Exterior quantities are read
/// from the retained exterior context and finite-layer quantities are read
/// from [`RetainedIsotropicLayers`]. Consequently, sampled constitutive
/// parameters carry exactly the same value and derivative information as the
/// corresponding solved problem.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ConstitutiveSamplingContext<'a, W, A> {
    workspace: &'a W,
    algebra: PhantomData<fn() -> A>,
}

impl<'a, W, A> ConstitutiveSamplingContext<'a, W, A> {
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

impl<'a, W, A> ConstitutiveSamplingContext<'a, W, A>
where
    W: PlaneWaveSolutionSource + RetainedIsotropicLayers<Algebra = A>,
    <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = A>,
    A: JetStack + Clone,
    A::Dimension: Dimension,
{
    /// Sample relative permittivity and permeability at the requested
    /// resolved spatial positions.
    ///
    /// The returned stacked algebra has one additional spatial axis whose
    /// order exactly matches `sampling.positions()`.
    pub(crate) fn sample<R>(
        &self,
        sampling: &ResolvedFieldSampling<R>,
    ) -> Result<IsotropicConstitutiveParameters<A::Stacked>, ConstitutiveSamplingError> {
        let solution = self.workspace.solution();
        let exterior = solution.context();

        let mut sampled = ConstitutiveSequences::with_capacity(sampling.len());

        for position in sampling.positions() {
            match position {
                FieldPosition::LeftExterior { .. } => {
                    sampled.push(exterior.left_epsilon(), exterior.left_mu());
                }

                FieldPosition::Layer { index, .. } => {
                    let quantities = self
                        .workspace
                        .layer_quantities(index.0)
                        .ok_or(ConstitutiveSamplingError::MissingLayerData { index: index.0 })?;

                    sampled.push(quantities.epsilon(), quantities.mu());
                }

                FieldPosition::RightExterior { .. } => {
                    sampled.push(exterior.right_epsilon(), exterior.right_mu());
                }
            }
        }

        sampled.stack()
    }
}

struct ConstitutiveSequences<A> {
    epsilon: Vec<A>,
    mu: Vec<A>,
}

impl<A> ConstitutiveSequences<A> {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            epsilon: Vec::with_capacity(capacity),
            mu: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, epsilon: &A, mu: &A)
    where
        A: Clone,
    {
        self.epsilon.push(epsilon.clone());
        self.mu.push(mu.clone());
    }

    fn stack(self) -> Result<IsotropicConstitutiveParameters<A::Stacked>, ConstitutiveSamplingError>
    where
        A: JetStack,
        A::Dimension: Dimension,
    {
        let epsilon = A::stack(self.epsilon)?.ok_or(ConstitutiveSamplingError::EmptySampling)?;

        let mu = A::stack(self.mu)?.ok_or(ConstitutiveSamplingError::EmptySampling)?;

        Ok(IsotropicConstitutiveParameters::new(epsilon, mu))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::Ix1;
    use num_complex::Complex64;

    use crate::{
        FiniteLayerIndex, Parameter, PlaneWaveEvaluator, Polarisation,
        backend::{
            ExteriorContextProvider, PlaneWaveEntries, PlaneWaveSolutionSource,
            RetainedIsotropicLayers, scatter2::Scatter2, transfer2::Transfer2,
        },
        spatial::{FieldPosition, Length, ResolvedFieldSampling, ResolvedLayerPosition},
        test_support::{
            assertions::assert_complex_close,
            finite_difference::{FIRST_DERIVATIVE_TOLERANCE, VALUE_TOLERANCE},
            planar::{scalar_real_input, two_layer_stack},
        },
    };

    use super::{ConstitutiveSamplingContext, ConstitutiveSamplingError};

    type C = Complex64;

    /*
     * Deliberately use a non-geometric order and repeat layer 0.
     *
     * This checks that constitutive sampling follows the resolved request
     * exactly rather than grouping, sorting, or deduplicating regions.
     */
    fn mixed_sampling() -> ResolvedFieldSampling<f64> {
        ResolvedFieldSampling::new(vec![
            FieldPosition::RightExterior {
                distance: Length::zero(),
            },
            FieldPosition::Layer {
                index: FiniteLayerIndex(0),
                position: ResolvedLayerPosition::Fraction(0.25),
            },
            FieldPosition::LeftExterior {
                distance: Length::zero(),
            },
            FieldPosition::Layer {
                index: FiniteLayerIndex(1),
                position: ResolvedLayerPosition::Fraction(0.75),
            },
            FieldPosition::Layer {
                index: FiniteLayerIndex(0),
                position: ResolvedLayerPosition::Fraction(0.75),
            },
        ])
    }

    fn repeated_layer_sampling() -> ResolvedFieldSampling<f64> {
        ResolvedFieldSampling::new(vec![
            FieldPosition::Layer {
                index: FiniteLayerIndex(0),
                position: ResolvedLayerPosition::Fraction(0.0),
            },
            FieldPosition::Layer {
                index: FiniteLayerIndex(0),
                position: ResolvedLayerPosition::Fraction(0.25),
            },
            FieldPosition::Layer {
                index: FiniteLayerIndex(0),
                position: ResolvedLayerPosition::Fraction(0.5),
            },
            FieldPosition::Layer {
                index: FiniteLayerIndex(0),
                position: ResolvedLayerPosition::Fraction(0.75),
            },
            FieldPosition::Layer {
                index: FiniteLayerIndex(0),
                position: ResolvedLayerPosition::Fraction(1.0),
            },
        ])
    }

    /*
     * Execute the same context-level contract test against both retained
     * backends.
     */
    macro_rules! for_each_backend {
        ($evaluator:ident, $body:block) => {{
            {
                let $evaluator = PlaneWaveEvaluator::new(Scatter2::new());

                $body
            }

            {
                let $evaluator = PlaneWaveEvaluator::new(Transfer2::new());

                $body
            }
        }};
    }

    #[test]
    fn constitutive_sampling_preserves_position_order_and_duplicates() {
        let stack = two_layer_stack();
        let sampling = mixed_sampling();

        for_each_backend!(evaluator, {
            let state = evaluator
                .retain(
                    scalar_real_input(2.5, 0.31),
                    &stack,
                    Polarisation::TransverseElectric,
                )
                .unwrap();

            /*
             * Constitutive sampling is a point operation, like field
             * reconstruction, so use the projected retained workspace.
             */
            let point = state.project_point(&()).unwrap();
            let workspace = point.workspace();

            let sampled = ConstitutiveSamplingContext::new(workspace)
                .sample(&sampling)
                .unwrap();

            let solution = workspace.solution();
            let exterior = solution.context();

            let layer_0 = workspace
                .layer_quantities(0)
                .expect("layer 0 quantities should be retained");

            let layer_1 = workspace
                .layer_quantities(1)
                .expect("layer 1 quantities should be retained");

            /*
             * Requested order:
             *
             *   right exterior,
             *   layer 0,
             *   left exterior,
             *   layer 1,
             *   layer 0.
             */
            let expected_epsilon = [
                exterior.right_epsilon().value()[()],
                layer_0.epsilon().value()[()],
                exterior.left_epsilon().value()[()],
                layer_1.epsilon().value()[()],
                layer_0.epsilon().value()[()],
            ];

            let expected_mu = [
                exterior.right_mu().value()[()],
                layer_0.mu().value()[()],
                exterior.left_mu().value()[()],
                layer_1.mu().value()[()],
                layer_0.mu().value()[()],
            ];

            assert_eq!(sampled.epsilon().value().shape(), &[sampling.len()],);

            assert_eq!(sampled.mu().value().shape(), &[sampling.len()],);

            for (&actual, expected) in sampled.epsilon().value().iter().zip(expected_epsilon) {
                assert_complex_close(actual, expected, VALUE_TOLERANCE);
            }

            for (&actual, expected) in sampled.mu().value().iter().zip(expected_mu) {
                assert_complex_close(actual, expected, VALUE_TOLERANCE);
            }
        });
    }

    #[test]
    fn constitutive_sampling_is_constant_within_homogeneous_layer() {
        let stack = two_layer_stack();
        let sampling = repeated_layer_sampling();

        for_each_backend!(evaluator, {
            let state = evaluator
                .retain(
                    scalar_real_input(2.5, 0.31),
                    &stack,
                    Polarisation::TransverseMagnetic,
                )
                .unwrap();

            let point = state.project_point(&()).unwrap();
            let workspace = point.workspace();

            let sampled = ConstitutiveSamplingContext::new(workspace)
                .sample(&sampling)
                .unwrap();

            let layer = workspace
                .layer_quantities(0)
                .expect("layer 0 quantities should be retained");

            let expected_epsilon = layer.epsilon().value()[()];

            let expected_mu = layer.mu().value()[()];

            for &actual in sampled.epsilon().value() {
                assert_complex_close(actual, expected_epsilon, VALUE_TOLERANCE);
            }

            for &actual in sampled.mu().value() {
                assert_complex_close(actual, expected_mu, VALUE_TOLERANCE);
            }
        });
    }

    #[test]
    fn constitutive_sampling_preserves_first_derivatives() {
        let stack = two_layer_stack();
        let sampling = mixed_sampling();

        /*
         * Spectral differentiation is useful here because dispersive
         * constitutive quantities may have nonzero first derivatives.
         */
        let parameter = Parameter::Spectral;

        for_each_backend!(evaluator, {
            let state = evaluator
                .retain_first(
                    scalar_real_input(2.5, 0.31),
                    &stack,
                    Polarisation::TransverseElectric,
                    parameter,
                )
                .unwrap();

            let point = state.project_point(&()).unwrap();
            let workspace = point.workspace();

            let sampled = ConstitutiveSamplingContext::new(workspace)
                .sample(&sampling)
                .unwrap();

            let solution = workspace.solution();
            let exterior = solution.context();

            let layer_0 = workspace
                .layer_quantities(0)
                .expect("layer 0 quantities should be retained");

            let layer_1 = workspace
                .layer_quantities(1)
                .expect("layer 1 quantities should be retained");

            let expected_epsilon_value = [
                exterior.right_epsilon().value()[()],
                layer_0.epsilon().value()[()],
                exterior.left_epsilon().value()[()],
                layer_1.epsilon().value()[()],
                layer_0.epsilon().value()[()],
            ];

            let expected_epsilon_first = [
                exterior.right_epsilon().first()[()],
                layer_0.epsilon().first()[()],
                exterior.left_epsilon().first()[()],
                layer_1.epsilon().first()[()],
                layer_0.epsilon().first()[()],
            ];

            let expected_mu_value = [
                exterior.right_mu().value()[()],
                layer_0.mu().value()[()],
                exterior.left_mu().value()[()],
                layer_1.mu().value()[()],
                layer_0.mu().value()[()],
            ];

            let expected_mu_first = [
                exterior.right_mu().first()[()],
                layer_0.mu().first()[()],
                exterior.left_mu().first()[()],
                layer_1.mu().first()[()],
                layer_0.mu().first()[()],
            ];

            for (&actual, expected) in sampled.epsilon().value().iter().zip(expected_epsilon_value)
            {
                assert_complex_close(actual, expected, VALUE_TOLERANCE);
            }

            for (&actual, expected) in sampled.epsilon().first().iter().zip(expected_epsilon_first)
            {
                assert_complex_close(actual, expected, FIRST_DERIVATIVE_TOLERANCE);
            }

            for (&actual, expected) in sampled.mu().value().iter().zip(expected_mu_value) {
                assert_complex_close(actual, expected, VALUE_TOLERANCE);
            }

            for (&actual, expected) in sampled.mu().first().iter().zip(expected_mu_first) {
                assert_complex_close(actual, expected, FIRST_DERIVATIVE_TOLERANCE);
            }
        });
    }

    #[test]
    fn constitutive_sampling_preserves_thickness_jet_structure() {
        let stack = two_layer_stack();
        let sampling = mixed_sampling();

        let parameter = Parameter::LayerThickness(FiniteLayerIndex(1));

        for_each_backend!(evaluator, {
            let state = evaluator
                .retain_first(
                    scalar_real_input(2.5, 0.31),
                    &stack,
                    Polarisation::TransverseMagnetic,
                    parameter,
                )
                .unwrap();

            let point = state.project_point(&()).unwrap();
            let workspace = point.workspace();

            let sampled = ConstitutiveSamplingContext::new(workspace)
                .sample(&sampling)
                .unwrap();

            let solution = workspace.solution();
            let exterior = solution.context();

            let layer_0 = workspace.layer_quantities(0).unwrap();

            let layer_1 = workspace.layer_quantities(1).unwrap();

            let expected_epsilon_first = [
                exterior.right_epsilon().first()[()],
                layer_0.epsilon().first()[()],
                exterior.left_epsilon().first()[()],
                layer_1.epsilon().first()[()],
                layer_0.epsilon().first()[()],
            ];

            let expected_mu_first = [
                exterior.right_mu().first()[()],
                layer_0.mu().first()[()],
                exterior.left_mu().first()[()],
                layer_1.mu().first()[()],
                layer_0.mu().first()[()],
            ];

            for (&actual, expected) in sampled.epsilon().first().iter().zip(expected_epsilon_first)
            {
                assert_complex_close(actual, expected, FIRST_DERIVATIVE_TOLERANCE);
            }

            for (&actual, expected) in sampled.mu().first().iter().zip(expected_mu_first) {
                assert_complex_close(actual, expected, FIRST_DERIVATIVE_TOLERANCE);
            }
        });
    }

    #[test]
    fn empty_constitutive_sampling_is_rejected() {
        let stack = two_layer_stack();
        let sampling = ResolvedFieldSampling::<f64>::new(Vec::new());

        for_each_backend!(evaluator, {
            let state = evaluator
                .retain(
                    scalar_real_input(2.5, 0.31),
                    &stack,
                    Polarisation::TransverseElectric,
                )
                .unwrap();

            let point = state.project_point(&()).unwrap();

            let error = ConstitutiveSamplingContext::new(point.workspace())
                .sample(&sampling)
                .unwrap_err();

            assert_eq!(error, ConstitutiveSamplingError::EmptySampling,);
        });
    }

    #[test]
    fn transfer_and_scatter_constitutive_sampling_agree() {
        let stack = two_layer_stack();
        let sampling = mixed_sampling();

        let scatter_state = PlaneWaveEvaluator::new(Scatter2::new())
            .retain_first(
                scalar_real_input(2.5, 0.31),
                &stack,
                Polarisation::TransverseElectric,
                Parameter::Spectral,
            )
            .unwrap();

        let transfer_state = PlaneWaveEvaluator::new(Transfer2::new())
            .retain_first(
                scalar_real_input(2.5, 0.31),
                &stack,
                Polarisation::TransverseElectric,
                Parameter::Spectral,
            )
            .unwrap();

        let scatter_point = scatter_state.project_point(&()).unwrap();

        let transfer_point = transfer_state.project_point(&()).unwrap();

        let scatter = ConstitutiveSamplingContext::new(scatter_point.workspace())
            .sample(&sampling)
            .unwrap();

        let transfer = ConstitutiveSamplingContext::new(transfer_point.workspace())
            .sample(&sampling)
            .unwrap();

        assert_eq!(
            scatter.epsilon().value().shape(),
            transfer.epsilon().value().shape(),
        );

        for (&actual, &expected) in scatter
            .epsilon()
            .value()
            .iter()
            .zip(transfer.epsilon().value())
        {
            assert_complex_close(actual, expected, VALUE_TOLERANCE);
        }

        for (&actual, &expected) in scatter
            .epsilon()
            .first()
            .iter()
            .zip(transfer.epsilon().first())
        {
            assert_complex_close(actual, expected, FIRST_DERIVATIVE_TOLERANCE);
        }

        for (&actual, &expected) in scatter.mu().value().iter().zip(transfer.mu().value()) {
            assert_complex_close(actual, expected, VALUE_TOLERANCE);
        }

        for (&actual, &expected) in scatter.mu().first().iter().zip(transfer.mu().first()) {
            assert_complex_close(actual, expected, FIRST_DERIVATIVE_TOLERANCE);
        }
    }
}
