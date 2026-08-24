use std::marker::PhantomData;

use ndarray::Dimension;

use crate::{
    ComplexScalar,
    algebra::{Jet, JetStack},
    backend::{
        ExteriorContextProvider, PlaneWaveEntries, PlaneWaveSolutionSource, RetainedIsotropicLayers,
    },
    input::CanonicalStack,
    material::{ConstitutiveDerivativeEvaluator, ConstitutiveSpectralFirstLift},
    observable::{
        FieldReconstructionError, IsotropicConstitutiveParameters,
        IsotropicConstitutiveSpectralData, field::constitutive::IsotropicConstitutiveSpectralFirst,
    },
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
                    let quantities = self.workspace.layer_quantities(index.get()).ok_or(
                        ConstitutiveSamplingError::MissingLayerData { index: index.get() },
                    )?;

                    sampled.push(quantities.epsilon(), quantities.mu());
                }

                FieldPosition::RightExterior { .. } => {
                    sampled.push(exterior.right_epsilon(), exterior.right_mu());
                }
            }
        }

        sampled.stack()
    }

    pub(crate) fn sample_spectral_first<R, M, E>(
        &self,
        sampling: &ResolvedFieldSampling<R>,
        stack: &CanonicalStack<M, A>,
    ) -> Result<IsotropicConstitutiveSpectralData<A::Stacked>, ConstitutiveSamplingError>
    where
        M: 'a,
        A: Jet + ConstitutiveSpectralFirstLift<E, M> + Clone,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
        E: ConstitutiveDerivativeEvaluator<A::Scalar, A::Dimension, M>,
    {
        let solution = self.workspace.solution();
        let exterior = solution.context();

        let mut sampled = ConstitutiveSequences::with_capacity(sampling.len());
        let mut sampled_derivatives = ConstitutiveSequences::with_capacity(sampling.len());

        for position in sampling.positions() {
            match position {
                FieldPosition::LeftExterior { .. } => {
                    sampled.push(exterior.left_epsilon(), exterior.left_mu());

                    let epsilon_spectral_first = A::relative_permittivity_spectral_first(
                        stack.left_exterior(),
                        exterior.vacuum_angular_wavenumber(),
                    );

                    let mu_spectral_first = A::relative_permeability_spectral_first(
                        stack.left_exterior(),
                        exterior.vacuum_angular_wavenumber(),
                    );

                    sampled_derivatives.push(&epsilon_spectral_first, &mu_spectral_first);
                }

                FieldPosition::Layer { index, .. } => {
                    let quantities = self.workspace.layer_quantities(index.get()).ok_or(
                        ConstitutiveSamplingError::MissingLayerData { index: index.get() },
                    )?;

                    sampled.push(quantities.epsilon(), quantities.mu());

                    let layer =
                        stack
                            .layer(*index)
                            .ok_or(ConstitutiveSamplingError::MissingLayerData {
                                index: index.get(),
                            })?;

                    let epsilon_spectral_first = A::relative_permittivity_spectral_first(
                        layer.material(),
                        exterior.vacuum_angular_wavenumber(),
                    );

                    let mu_spectral_first = A::relative_permeability_spectral_first(
                        layer.material(),
                        exterior.vacuum_angular_wavenumber(),
                    );

                    sampled_derivatives.push(&epsilon_spectral_first, &mu_spectral_first);
                }

                FieldPosition::RightExterior { .. } => {
                    sampled.push(exterior.right_epsilon(), exterior.right_mu());

                    let epsilon_spectral_first = A::relative_permittivity_spectral_first(
                        stack.right_exterior(),
                        exterior.vacuum_angular_wavenumber(),
                    );

                    let mu_spectral_first = A::relative_permeability_spectral_first(
                        stack.right_exterior(),
                        exterior.vacuum_angular_wavenumber(),
                    );

                    sampled_derivatives.push(&epsilon_spectral_first, &mu_spectral_first);
                }
            }
        }

        let parameters = sampled.stack()?;

        let spectral_first = sampled_derivatives.stack()?;

        let (epsilon_spectral_first, mu_spectral_first) = spectral_first.into_parts();

        let spectral_first =
            IsotropicConstitutiveSpectralFirst::new(epsilon_spectral_first, mu_spectral_first);

        let k0 = A::stack(
            std::iter::repeat_n(exterior.vacuum_angular_wavenumber().clone(), sampling.len())
                .collect(),
        )?
        .ok_or(ConstitutiveSamplingError::EmptySampling)?;

        Ok(IsotropicConstitutiveSpectralData::new(
            parameters,
            spectral_first,
            k0,
        ))
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
    use lamina_units::Length;
    use num_complex::Complex64;

    use crate::{
        FiniteLayerIndex, Parameter, Polarisation, RealAxisEvaluator,
        backend::{
            ExteriorContextProvider, RetainedIsotropicLayers, scatter2::Scatter2,
            transfer2::Transfer2,
        },
        spatial::{FieldPosition, ResolvedFieldSampling, ResolvedLayerPosition},
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
                index: FiniteLayerIndex::new(0),
                position: ResolvedLayerPosition::Fraction(0.25),
            },
            FieldPosition::LeftExterior {
                distance: Length::zero(),
            },
            FieldPosition::Layer {
                index: FiniteLayerIndex::new(1),
                position: ResolvedLayerPosition::Fraction(0.75),
            },
            FieldPosition::Layer {
                index: FiniteLayerIndex::new(0),
                position: ResolvedLayerPosition::Fraction(0.75),
            },
        ])
    }

    fn repeated_layer_sampling() -> ResolvedFieldSampling<f64> {
        ResolvedFieldSampling::new(vec![
            FieldPosition::Layer {
                index: FiniteLayerIndex::new(0),
                position: ResolvedLayerPosition::Fraction(0.0),
            },
            FieldPosition::Layer {
                index: FiniteLayerIndex::new(0),
                position: ResolvedLayerPosition::Fraction(0.25),
            },
            FieldPosition::Layer {
                index: FiniteLayerIndex::new(0),
                position: ResolvedLayerPosition::Fraction(0.5),
            },
            FieldPosition::Layer {
                index: FiniteLayerIndex::new(0),
                position: ResolvedLayerPosition::Fraction(0.75),
            },
            FieldPosition::Layer {
                index: FiniteLayerIndex::new(0),
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
                let $evaluator = RealAxisEvaluator::new(Scatter2::new());

                $body
            }

            {
                let $evaluator = RealAxisEvaluator::new(Transfer2::new());

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

        let parameter = Parameter::LayerThickness(FiniteLayerIndex::new(1));

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

        let scatter_state = RealAxisEvaluator::new(Scatter2::new())
            .retain_first(
                scalar_real_input(2.5, 0.31),
                &stack,
                Polarisation::TransverseElectric,
                Parameter::Spectral,
            )
            .unwrap();

        let transfer_state = RealAxisEvaluator::new(Transfer2::new())
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

#[cfg(test)]
mod spectral_first_tests {
    use lamina_units::Length;
    use num_complex::Complex64;

    use crate::{
        FiniteLayerIndex, Parameter, Polarisation, RealAxis, RealAxisEvaluator,
        backend::{scatter2::Scatter2, transfer2::Transfer2},
        material::ConstitutiveSpectralFirstLift,
        spatial::{FieldPosition, ResolvedFieldSampling, ResolvedLayerPosition},
        test_support::{
            assertions::assert_complex_close,
            finite_difference::{
                FIRST_DERIVATIVE_TOLERANCE, SECOND_DERIVATIVE_TOLERANCE, VALUE_TOLERANCE,
            },
            jet::{RealJ1, RealJ2},
            planar::{scalar_real_input, two_layer_stack},
        },
    };

    use super::ConstitutiveSamplingContext;

    type C = Complex64;

    fn mixed_sampling() -> ResolvedFieldSampling<f64> {
        ResolvedFieldSampling::new(vec![
            FieldPosition::RightExterior {
                distance: Length::zero(),
            },
            FieldPosition::Layer {
                index: FiniteLayerIndex::new(0),
                position: ResolvedLayerPosition::Fraction(0.25),
            },
            FieldPosition::LeftExterior {
                distance: Length::zero(),
            },
            FieldPosition::Layer {
                index: FiniteLayerIndex::new(1),
                position: ResolvedLayerPosition::Fraction(0.75),
            },
            FieldPosition::Layer {
                index: FiniteLayerIndex::new(0),
                position: ResolvedLayerPosition::Fraction(0.75),
            },
        ])
    }

    macro_rules! for_each_backend {
        ($evaluator:ident, $body:block) => {{
            {
                let $evaluator = RealAxisEvaluator::new(Scatter2::new());
                $body
            }

            {
                let $evaluator = RealAxisEvaluator::new(Transfer2::new());
                $body
            }
        }};
    }

    fn assert_complex_slice_close(
        actual: impl IntoIterator<Item = C>,
        expected: impl IntoIterator<Item = C>,
        tolerance: f64,
    ) {
        let actual = actual.into_iter().collect::<Vec<_>>();
        let expected = expected.into_iter().collect::<Vec<_>>();

        assert_eq!(actual.len(), expected.len());

        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_complex_close(actual, expected, tolerance);
        }
    }

    #[test]
    fn spectral_first_sampling_preserves_region_order_and_duplicates() {
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

            let point = state.project_point(&()).unwrap();

            let workspace = point.workspace();
            let canonical_stack = point.problem().stack();

            let k0 = point.problem().coordinates().vacuum_angular_wavenumber();

            let sampled = ConstitutiveSamplingContext::new(workspace)
                .sample_spectral_first::<_, _, RealAxis>(&sampling, canonical_stack)
                .unwrap();

            let layer_0 = canonical_stack.layer(FiniteLayerIndex::new(0)).unwrap();

            let layer_1 = canonical_stack.layer(FiniteLayerIndex::new(1)).unwrap();

            /*
             * Requested order:
             *
             * right exterior
             * layer 0
             * left exterior
             * layer 1
             * layer 0
             */
            let expected_epsilon_first = [
                <_ as ConstitutiveSpectralFirstLift<
                    RealAxis,
                    _,
                >>::relative_permittivity_spectral_first(
                    canonical_stack.right_exterior(),
                    k0,
                ),
                <_ as ConstitutiveSpectralFirstLift<
                    RealAxis,
                    _,
                >>::relative_permittivity_spectral_first(
                    layer_0.material(),
                    k0,
                ),
                <_ as ConstitutiveSpectralFirstLift<
                    RealAxis,
                    _,
                >>::relative_permittivity_spectral_first(
                    canonical_stack.left_exterior(),
                    k0,
                ),
                <_ as ConstitutiveSpectralFirstLift<
                    RealAxis,
                    _,
                >>::relative_permittivity_spectral_first(
                    layer_1.material(),
                    k0,
                ),
                <_ as ConstitutiveSpectralFirstLift<
                    RealAxis,
                    _,
                >>::relative_permittivity_spectral_first(
                    layer_0.material(),
                    k0,
                ),
            ];

            let expected_mu_first = [
                <_ as ConstitutiveSpectralFirstLift<
                    RealAxis,
                    _,
                >>::relative_permeability_spectral_first(
                    canonical_stack.right_exterior(),
                    k0,
                ),
                <_ as ConstitutiveSpectralFirstLift<
                    RealAxis,
                    _,
                >>::relative_permeability_spectral_first(
                    layer_0.material(),
                    k0,
                ),
                <_ as ConstitutiveSpectralFirstLift<
                    RealAxis,
                    _,
                >>::relative_permeability_spectral_first(
                    canonical_stack.left_exterior(),
                    k0,
                ),
                <_ as ConstitutiveSpectralFirstLift<
                    RealAxis,
                    _,
                >>::relative_permeability_spectral_first(
                    layer_1.material(),
                    k0,
                ),
                <_ as ConstitutiveSpectralFirstLift<
                    RealAxis,
                    _,
                >>::relative_permeability_spectral_first(
                    layer_0.material(),
                    k0,
                ),
            ];

            assert_eq!(
                sampled.epsilon_spectral_first().value().shape(),
                &[sampling.len()],
            );

            assert_eq!(
                sampled.mu_spectral_first().value().shape(),
                &[sampling.len()],
            );

            assert_complex_slice_close(
                sampled.epsilon_spectral_first().value().iter().copied(),
                expected_epsilon_first.iter().map(|value| value.value()[()]),
                VALUE_TOLERANCE,
            );

            assert_complex_slice_close(
                sampled.mu_spectral_first().value().iter().copied(),
                expected_mu_first.iter().map(|value| value.value()[()]),
                VALUE_TOLERANCE,
            );
        });
    }

    #[test]
    fn spectral_first_sampling_contains_same_constitutive_parameters_as_basic_sampling() {
        let stack = two_layer_stack();
        let sampling = mixed_sampling();

        for_each_backend!(evaluator, {
            let state = evaluator
                .retain(
                    scalar_real_input(2.5, 0.31),
                    &stack,
                    Polarisation::TransverseMagnetic,
                )
                .unwrap();

            let point = state.project_point(&()).unwrap();

            let context = ConstitutiveSamplingContext::new(point.workspace());

            let basic = context.sample(&sampling).unwrap();

            let spectral = context
                .sample_spectral_first::<_, _, RealAxis>(&sampling, point.problem().stack())
                .unwrap();

            assert_complex_slice_close(
                spectral.parameters().epsilon().value().iter().copied(),
                basic.epsilon().value().iter().copied(),
                VALUE_TOLERANCE,
            );

            assert_complex_slice_close(
                spectral.parameters().mu().value().iter().copied(),
                basic.mu().value().iter().copied(),
                VALUE_TOLERANCE,
            );
        });
    }

    #[test]
    fn constant_materials_have_zero_spectral_first_constitutive_data() {
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

            let point = state.project_point(&()).unwrap();

            let sampled = ConstitutiveSamplingContext::new(point.workspace())
                .sample_spectral_first::<_, _, RealAxis>(&sampling, point.problem().stack())
                .unwrap();

            for &value in sampled.epsilon_spectral_first().value() {
                assert_complex_close(value, C::new(0.0, 0.0), VALUE_TOLERANCE);
            }

            for &value in sampled.mu_spectral_first().value() {
                assert_complex_close(value, C::new(0.0, 0.0), VALUE_TOLERANCE);
            }
        });
    }

    #[test]
    fn spectral_first_data_preserve_thickness_outer_jet() {
        let stack = two_layer_stack();
        let sampling = mixed_sampling();

        let parameter = Parameter::LayerThickness(FiniteLayerIndex::new(1));

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

            let canonical_stack = point.problem().stack();

            let k0 = point.problem().coordinates().vacuum_angular_wavenumber();

            let sampled = ConstitutiveSamplingContext::new(point.workspace())
                .sample_spectral_first::<_, _, RealAxis>(&sampling, canonical_stack)
                .unwrap();

            let layer_0 = canonical_stack.layer(FiniteLayerIndex::new(0)).unwrap();

            let layer_1 = canonical_stack.layer(FiniteLayerIndex::new(1)).unwrap();

            let expected_epsilon = [
                <RealJ1 as ConstitutiveSpectralFirstLift<RealAxis, _>>::relative_permittivity_spectral_first(canonical_stack.right_exterior(), k0),
                <RealJ1 as ConstitutiveSpectralFirstLift<RealAxis, _>>::relative_permittivity_spectral_first(layer_0.material(), k0),
                <RealJ1 as ConstitutiveSpectralFirstLift<RealAxis, _>>::relative_permittivity_spectral_first(canonical_stack.left_exterior(), k0),
                <RealJ1 as ConstitutiveSpectralFirstLift<RealAxis, _>>::relative_permittivity_spectral_first(layer_1.material(), k0),
                <RealJ1 as ConstitutiveSpectralFirstLift<RealAxis, _>>::relative_permittivity_spectral_first(layer_0.material(), k0),
            ];

            let expected_mu = [
                <RealJ1 as ConstitutiveSpectralFirstLift<RealAxis, _>>::relative_permeability_spectral_first(canonical_stack.right_exterior(), k0),
                <RealJ1 as ConstitutiveSpectralFirstLift<RealAxis, _>>::relative_permeability_spectral_first(layer_0.material(), k0),
                <RealJ1 as ConstitutiveSpectralFirstLift<RealAxis, _>>::relative_permeability_spectral_first(canonical_stack.left_exterior(), k0),
                <RealJ1 as ConstitutiveSpectralFirstLift<RealAxis, _>>::relative_permeability_spectral_first(layer_1.material(), k0),
                <RealJ1 as ConstitutiveSpectralFirstLift<RealAxis, _>>::relative_permeability_spectral_first(layer_0.material(), k0),
            ];

            /*
             * Value is ∂k0 ε.
             *
             * `.first()` is the caller's outer derivative:
             *
             *     ∂d (∂k0 ε).
             */
            assert_complex_slice_close(
                sampled.epsilon_spectral_first().value().iter().copied(),
                expected_epsilon.iter().map(|value| value.value()[()]),
                VALUE_TOLERANCE,
            );

            assert_complex_slice_close(
                sampled.epsilon_spectral_first().first().iter().copied(),
                expected_epsilon.iter().map(|value| value.first()[()]),
                FIRST_DERIVATIVE_TOLERANCE,
            );

            assert_complex_slice_close(
                sampled.mu_spectral_first().value().iter().copied(),
                expected_mu.iter().map(|value| value.value()[()]),
                VALUE_TOLERANCE,
            );

            assert_complex_slice_close(
                sampled.mu_spectral_first().first().iter().copied(),
                expected_mu.iter().map(|value| value.first()[()]),
                FIRST_DERIVATIVE_TOLERANCE,
            );
        });
    }

    #[test]
    fn spectral_first_data_preserve_second_order_outer_jet() {
        let stack = two_layer_stack();
        let sampling = mixed_sampling();

        for_each_backend!(evaluator, {
            let state = evaluator
                .retain_second(
                    scalar_real_input(2.5, 0.31),
                    &stack,
                    Polarisation::TransverseMagnetic,
                    Parameter::Spectral,
                )
                .unwrap();

            let point = state.project_point(&()).unwrap();

            let canonical_stack = point.problem().stack();

            let k0 = point.problem().coordinates().vacuum_angular_wavenumber();

            let sampled = ConstitutiveSamplingContext::new(point.workspace())
                .sample_spectral_first::<_, _, RealAxis>(&sampling, canonical_stack)
                .unwrap();

            /*
             * Pick one finite-layer position and compare the complete
             * outer jet with a direct spectral-first lift.
             *
             * mixed_sampling()[1] is layer 0.
             */
            let expected_epsilon = <RealJ2 as ConstitutiveSpectralFirstLift<RealAxis, _>>::relative_permittivity_spectral_first(
                canonical_stack
                    .layer(FiniteLayerIndex::new(0))
                    .unwrap()
                    .material(),
                k0,
            );

            let expected_mu = <RealJ2 as ConstitutiveSpectralFirstLift<RealAxis, _>>::relative_permeability_spectral_first(
                canonical_stack
                    .layer(FiniteLayerIndex::new(0))
                    .unwrap()
                    .material(),
                k0,
            );

            assert_complex_close(
                sampled.epsilon_spectral_first().value()[1],
                expected_epsilon.value()[()],
                VALUE_TOLERANCE,
            );

            assert_complex_close(
                sampled.epsilon_spectral_first().first()[1],
                expected_epsilon.first()[()],
                FIRST_DERIVATIVE_TOLERANCE,
            );

            assert_complex_close(
                sampled.epsilon_spectral_first().second()[1],
                expected_epsilon.second()[()],
                SECOND_DERIVATIVE_TOLERANCE,
            );

            assert_complex_close(
                sampled.mu_spectral_first().value()[1],
                expected_mu.value()[()],
                VALUE_TOLERANCE,
            );

            assert_complex_close(
                sampled.mu_spectral_first().first()[1],
                expected_mu.first()[()],
                FIRST_DERIVATIVE_TOLERANCE,
            );

            assert_complex_close(
                sampled.mu_spectral_first().second()[1],
                expected_mu.second()[()],
                SECOND_DERIVATIVE_TOLERANCE,
            );
        });
    }

    #[test]
    fn transfer_and_scatter_agree_on_spectral_first_constitutive_sampling() {
        let stack = two_layer_stack();
        let sampling = mixed_sampling();

        let scatter = RealAxisEvaluator::new(Scatter2::new())
            .retain_second(
                scalar_real_input(2.5, 0.31),
                &stack,
                Polarisation::TransverseElectric,
                Parameter::Spectral,
            )
            .unwrap();

        let transfer = RealAxisEvaluator::new(Transfer2::new())
            .retain_second(
                scalar_real_input(2.5, 0.31),
                &stack,
                Polarisation::TransverseElectric,
                Parameter::Spectral,
            )
            .unwrap();

        let scatter = scatter.project_point(&()).unwrap();
        let transfer = transfer.project_point(&()).unwrap();

        let scatter_data = ConstitutiveSamplingContext::new(scatter.workspace())
            .sample_spectral_first::<_, _, RealAxis>(&sampling, scatter.problem().stack())
            .unwrap();

        let transfer_data = ConstitutiveSamplingContext::new(transfer.workspace())
            .sample_spectral_first::<_, _, RealAxis>(&sampling, transfer.problem().stack())
            .unwrap();

        for (actual, expected) in [
            (
                scatter_data.parameters().epsilon(),
                transfer_data.parameters().epsilon(),
            ),
            (
                scatter_data.parameters().mu(),
                transfer_data.parameters().mu(),
            ),
            (
                scatter_data.epsilon_spectral_first(),
                transfer_data.epsilon_spectral_first(),
            ),
            (
                scatter_data.mu_spectral_first(),
                transfer_data.mu_spectral_first(),
            ),
        ] {
            assert_complex_slice_close(
                actual.value().iter().copied(),
                expected.value().iter().copied(),
                VALUE_TOLERANCE,
            );

            assert_complex_slice_close(
                actual.first().iter().copied(),
                expected.first().iter().copied(),
                FIRST_DERIVATIVE_TOLERANCE,
            );

            assert_complex_slice_close(
                actual.second().iter().copied(),
                expected.second().iter().copied(),
                SECOND_DERIVATIVE_TOLERANCE,
            );
        }
    }
}
