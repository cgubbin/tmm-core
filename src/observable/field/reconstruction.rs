use nalgebra::ComplexField;
use ndarray::Dimension;

use crate::{
    ComplexScalar, IncidentSide, Polarisation,
    algebra::{CartesianScalarAlgebra, JetStack, ScalarAlgebra},
    backend::{
        ExteriorContextProvider, PlaneWaveEntries, PlaneWaveSolutionSource, RetainedIsotropicLayers,
    },
    observable::{Amplitudes, BoundaryProjectionError, BoundaryState, ProjectAmplitudes},
    spatial::{CanonicalFieldPosition, CompiledFieldSampling, FieldSamplingError},
    waves::{
        ReconstructExteriorBoundaryWaves, ReconstructLayerBoundaryWaves, WaveSamplingContext,
        WaveSamplingError,
    },
};

use super::ElectromagneticFields;

#[derive(Debug, thiserror::Error)]
pub enum FieldReconstructionError<R> {
    #[error(transparent)]
    Wave(#[from] WaveSamplingError),

    #[error(transparent)]
    Boundary(#[from] BoundaryProjectionError),

    #[error("retained data are incomplete for finite layer {index}")]
    MissingLayerData { index: usize },

    #[error("field sampling request is empty")]
    EmptySampling,

    #[error(transparent)]
    FieldSampling(#[from] FieldSamplingError<R>),

    #[error("failed to stack sampled field components")]
    Shape(#[from] ndarray::ShapeError),
}

pub(crate) struct FieldSamplingContext<'a, W, A> {
    waves: WaveSamplingContext<'a, W, A>,
}

impl<'a, W, A> FieldSamplingContext<'a, W, A> {
    pub(crate) fn new(workspace: &'a W) -> Self
    where
        W: ReconstructLayerBoundaryWaves<Algebra = A>,
    {
        Self {
            waves: WaveSamplingContext::new(workspace),
        }
    }

    pub(crate) fn reconstruct(
        &self,
        incident_side: IncidentSide,
        sampling: &CompiledFieldSampling<<A::Scalar as ComplexField>::RealField>,
    ) -> Result<
        ElectromagneticFields<<A::Stacked as CartesianScalarAlgebra>::Vector>,
        FieldReconstructionError<<A::Scalar as ComplexField>::RealField>,
    >
    where
        A: ScalarAlgebra + JetStack + Clone,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
        A::Stacked: CartesianScalarAlgebra,
        <A::Scalar as ComplexField>::RealField: Copy,
        W: ReconstructExteriorBoundaryWaves<Algebra = A>
            + ReconstructLayerBoundaryWaves<Algebra = A>
            + RetainedIsotropicLayers<Algebra = A>
            + PlaneWaveSolutionSource,
        W::Entries: ProjectAmplitudes,
        <W::Entries as ProjectAmplitudes>::Amplitudes: Amplitudes<Algebra = A>,
        <W::Entries as PlaneWaveEntries>::ExteriorContext: ExteriorContextProvider<Algebra = A>,
    {
        let sampled_waves = self.waves.propagate_sampling(incident_side, sampling)?;

        let solution = self.waves.workspace().solution();
        let exterior = solution.context();

        let k0 = exterior.vacuum_angular_wavenumber();
        let beta = exterior.parallel_angular_wavenumber();
        let polarisation = exterior.polarisation();

        let mut components = FieldComponentSequences::with_capacity(sampling.len());

        for (position, waves) in sampling.positions().iter().copied().zip(sampled_waves) {
            let waves: crate::observable::BoundaryWaves<A> = waves.into();
            let projected = match position {
                CanonicalFieldPosition::LeftExterior { .. } => {
                    let state = waves.into_state(exterior.left_admittance());

                    project_isotropic_field_components(
                        state,
                        polarisation,
                        exterior.left_kappa(),
                        exterior.left_admittance(),
                        k0,
                        beta,
                    )
                }

                CanonicalFieldPosition::Layer { index, .. } => {
                    let quantities = self
                        .waves
                        .workspace()
                        .layer_quantities(index.0)
                        .ok_or(FieldReconstructionError::MissingLayerData { index: index.0 })?;

                    let admittance = quantities.admittance().into_inner();

                    let state = waves.into_state(&admittance);

                    project_isotropic_field_components(
                        state,
                        quantities.polarisation(),
                        quantities.kappa(),
                        &admittance,
                        k0,
                        beta,
                    )
                }

                CanonicalFieldPosition::RightExterior { .. } => {
                    let state = waves.into_state(exterior.right_admittance());

                    project_isotropic_field_components(
                        state,
                        polarisation,
                        exterior.right_kappa(),
                        exterior.right_admittance(),
                        k0,
                        beta,
                    )
                }
            };

            components.push(projected);
        }

        components.stack()
    }
}

struct FieldComponentSequences<A> {
    ex: Vec<A>,
    ey: Vec<A>,
    ez: Vec<A>,
    hx: Vec<A>,
    hy: Vec<A>,
    hz: Vec<A>,
}

impl<A> FieldComponentSequences<A> {
    fn with_capacity(capacity: usize) -> Self {
        Self {
            ex: Vec::with_capacity(capacity),
            ey: Vec::with_capacity(capacity),
            ez: Vec::with_capacity(capacity),
            hx: Vec::with_capacity(capacity),
            hy: Vec::with_capacity(capacity),
            hz: Vec::with_capacity(capacity),
        }
    }

    fn push(&mut self, fields: ElectromagneticFieldComponents<A>) {
        let (ex, ey, ez, hx, hy, hz) = fields.into_parts();

        self.ex.push(ex);
        self.ey.push(ey);
        self.ez.push(ez);

        self.hx.push(hx);
        self.hy.push(hy);
        self.hz.push(hz);
    }

    fn stack(
        self,
    ) -> Result<
        ElectromagneticFields<<A::Stacked as CartesianScalarAlgebra>::Vector>,
        FieldReconstructionError<<A::Scalar as ComplexField>::RealField>,
    >
    where
        A: JetStack,
        A::Scalar: ComplexField,
        A::Dimension: Dimension,
        A::Stacked: CartesianScalarAlgebra,
    {
        let ex = A::stack(self.ex)?.ok_or(FieldReconstructionError::EmptySampling)?;
        let ey = A::stack(self.ey)?.ok_or(FieldReconstructionError::EmptySampling)?;
        let ez = A::stack(self.ez)?.ok_or(FieldReconstructionError::EmptySampling)?;

        let hx = A::stack(self.hx)?.ok_or(FieldReconstructionError::EmptySampling)?;
        let hy = A::stack(self.hy)?.ok_or(FieldReconstructionError::EmptySampling)?;
        let hz = A::stack(self.hz)?.ok_or(FieldReconstructionError::EmptySampling)?;

        let electric = A::Stacked::cartesian_vector(ex, ey, ez);

        let magnetic = A::Stacked::cartesian_vector(hx, hy, hz);

        Ok(ElectromagneticFields::new(electric, magnetic))
    }
}

#[derive(Clone, Debug)]
pub(crate) struct ElectromagneticFieldComponents<A> {
    pub(crate) ex: A,
    pub(crate) ey: A,
    pub(crate) ez: A,
    pub(crate) hx: A,
    pub(crate) hy: A,
    pub(crate) hz: A,
}

impl<A> ElectromagneticFieldComponents<A> {
    fn new(ex: A, ey: A, ez: A, hx: A, hy: A, hz: A) -> Self {
        Self {
            ex,
            ey,
            ez,
            hx,
            hy,
            hz,
        }
    }

    fn into_parts(self) -> (A, A, A, A, A, A) {
        (self.ex, self.ey, self.ez, self.hx, self.hy, self.hz)
    }
}

pub(crate) fn project_isotropic_field_components<A>(
    state: BoundaryState<A>,
    polarisation: Polarisation,
    kappa: &A,
    admittance: &A,
    vacuum_angular_wavenumber: &A,
    parallel_angular_wavenumber: &A,
) -> ElectromagneticFieldComponents<A>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    let (field, secondary) = state.into_parts();

    let zero = field.zero_like();

    /*
     * The canonical state is
     *
     *     field     = ψ
     *     secondary = factor⁻¹ ∂z ψ,
     *
     * with
     *
     *     factor = μ  for TE,
     *     factor = ε  for TM.
     *
     * Since
     *
     *     Y = κ / factor,
     *
     * the longitudinal Cartesian component can be written without
     * separately retaining ε or μ:
     *
     *     β ψ / (k0 factor)
     *       = β Y ψ / (k0 κ).
     */
    let transverse = secondary.divide(vacuum_angular_wavenumber);

    let longitudinal = parallel_angular_wavenumber
        .multiply(admittance)
        .multiply(&field)
        .divide(&vacuum_angular_wavenumber.multiply(kappa));

    match polarisation {
        Polarisation::TransverseElectric => {
            /*
             * TE:
             *
             *     E = (0, ψ, 0)
             *
             *     Hx =  i secondary / k0
             *     Hz =  β ψ / (k0 μ)
             *
             * for the exp(-iωt) phasor convention.
             */
            let hx = transverse.scale(A::Scalar::i());

            ElectromagneticFieldComponents::new(
                zero.clone(),
                field,
                zero.clone(),
                hx,
                zero,
                longitudinal,
            )
        }

        Polarisation::TransverseMagnetic => {
            /*
             * TM:
             *
             *     H = (0, ψ, 0)
             *
             *     Ex = -i secondary / k0
             *     Ez = -β ψ / (k0 ε)
             *
             * for the exp(-iωt) phasor convention.
             */
            let ex = transverse.scale(-A::Scalar::i());
            let ez = longitudinal.negate();

            ElectromagneticFieldComponents::new(ex, zero.clone(), ez, zero.clone(), field, zero)
        }
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use crate::{
        Polarisation,
        algebra::{ArrayJet0, ArrayJet1, Jet, Jet0, Jet1, RealParameter},
        observable::BoundaryState,
        test_support::{TOLERANCE, assertions::assert_complex_close},
    };

    use super::*;

    type C = Complex64;
    type A = ArrayJet0<C, Ix0, RealParameter>;

    fn c(re: f64, im: f64) -> C {
        C::new(re, im)
    }

    fn jet(value: C) -> A {
        Jet0::new(arr0(value))
    }

    fn value(value: &A) -> C {
        value.value()[()]
    }

    #[test]
    fn te_projects_canonical_state_to_cartesian_fields() {
        let field = c(2.0, -0.5);
        let secondary = c(0.7, 0.3);

        let kappa = c(1.8, 0.2);
        let admittance = c(1.2, -0.1);

        let k0 = c(2.4, 0.0);
        let beta = c(0.6, 0.0);

        let state = BoundaryState::new(jet(field), jet(secondary));

        let result = project_isotropic_field_components(
            state,
            Polarisation::TransverseElectric,
            &jet(kappa),
            &jet(admittance),
            &jet(k0),
            &jet(beta),
        );

        let (ex, ey, ez, hx, hy, hz) = result.into_parts();

        assert_complex_close(value(&ex), C::ZERO, TOLERANCE);
        assert_complex_close(value(&ey), field, TOLERANCE);
        assert_complex_close(value(&ez), C::ZERO, TOLERANCE);

        assert_complex_close(value(&hx), C::i() * secondary / k0, TOLERANCE);

        assert_complex_close(value(&hy), C::ZERO, TOLERANCE);

        assert_complex_close(
            value(&hz),
            beta * admittance * field / (k0 * kappa),
            TOLERANCE,
        );
    }

    #[test]
    fn tm_projects_canonical_state_to_cartesian_fields() {
        let field = c(2.0, -0.5);
        let secondary = c(0.7, 0.3);

        let kappa = c(1.8, 0.2);
        let admittance = c(1.2, -0.1);

        let k0 = c(2.4, 0.0);
        let beta = c(0.6, 0.0);

        let state = BoundaryState::new(jet(field), jet(secondary));

        let result = project_isotropic_field_components(
            state,
            Polarisation::TransverseMagnetic,
            &jet(kappa),
            &jet(admittance),
            &jet(k0),
            &jet(beta),
        );

        let (ex, ey, ez, hx, hy, hz) = result.into_parts();

        assert_complex_close(value(&ex), -C::i() * secondary / k0, TOLERANCE);

        assert_complex_close(value(&ey), C::ZERO, TOLERANCE);

        assert_complex_close(
            value(&ez),
            -beta * admittance * field / (k0 * kappa),
            TOLERANCE,
        );

        assert_complex_close(value(&hx), C::ZERO, TOLERANCE);
        assert_complex_close(value(&hy), field, TOLERANCE);
        assert_complex_close(value(&hz), C::ZERO, TOLERANCE);
    }

    #[test]
    fn longitudinal_components_vanish_at_normal_incidence() {
        let state = BoundaryState::new(jet(c(1.3, 0.2)), jet(c(0.4, -0.1)));

        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            let result = project_isotropic_field_components(
                state.clone(),
                polarisation,
                &jet(c(1.7, 0.0)),
                &jet(c(1.1, 0.0)),
                &jet(c(2.0, 0.0)),
                &jet(C::ZERO),
            );

            let (_, _, ez, _, _, hz) = result.into_parts();

            assert_complex_close(value(&ez), C::ZERO, TOLERANCE);

            assert_complex_close(value(&hz), C::ZERO, TOLERANCE);
        }
    }

    #[test]
    fn te_cartesian_norm_matches_canonical_overlap_normalisation() {
        let field = c(1.3, 0.4);
        let secondary = c(-0.2, 0.7);

        let kappa = c(1.8, 0.1);
        let admittance = c(1.2, -0.15);
        let k0 = c(2.5, 0.0);
        let beta = c(0.7, 0.0);

        let projected = project_isotropic_field_components(
            BoundaryState::new(jet(field), jet(secondary)),
            Polarisation::TransverseElectric,
            &jet(kappa),
            &jet(admittance),
            &jet(k0),
            &jet(beta),
        );

        let (ex, ey, ez, hx, hy, hz) = projected.into_parts();

        let electric_norm = value(&ex).norm_sqr() + value(&ey).norm_sqr() + value(&ez).norm_sqr();

        let magnetic_norm = value(&hx).norm_sqr() + value(&hy).norm_sqr() + value(&hz).norm_sqr();

        let expected_electric = field.norm_sqr();

        let expected_magnetic =
            (secondary / k0).norm_sqr() + (beta * admittance * field / (k0 * kappa)).norm_sqr();

        assert!((electric_norm - expected_electric).abs() < TOLERANCE,);

        assert!((magnetic_norm - expected_magnetic).abs() < TOLERANCE,);
    }

    #[test]
    fn projection_preserves_jet_derivatives() {
        type A1 = ArrayJet1<C, Ix0, RealParameter>;

        fn jet1(value: C, first: C) -> A1 {
            Jet1::from_parts(arr0(value), arr0(first))
        }

        let secondary = c(0.7, 0.2);
        let secondary_first = c(-0.3, 0.4);
        let k0 = c(2.0, 0.0);

        let result = project_isotropic_field_components(
            BoundaryState::new(jet1(c(1.0, 0.0), C::ZERO), jet1(secondary, secondary_first)),
            Polarisation::TransverseElectric,
            &jet1(c(1.5, 0.0), C::ZERO),
            &jet1(c(1.1, 0.0), C::ZERO),
            &jet1(k0, C::ZERO),
            &jet1(c(0.2, 0.0), C::ZERO),
        );

        let (_, _, _, hx, _, _) = result.into_parts();

        assert_complex_close(hx.first()[()], C::i() * secondary_first / k0, TOLERANCE);
    }
}

#[cfg(test)]
mod stacking_tests {
    use ndarray::{Ix0, Ix1, arr0, arr1, array};
    use num_complex::Complex64;

    use crate::algebra::{ArrayJet0, Jet0, RealParameter};

    use super::{ElectromagneticFieldComponents, FieldComponentSequences};

    type C = Complex64;
    type J0<D> = ArrayJet0<C, D, RealParameter>;

    const TOLERANCE: f64 = 1.0e-12;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn scalar(value: f64) -> J0<Ix0> {
        Jet0::new(arr0(c(value)))
    }

    fn vector(values: &[f64]) -> J0<Ix1> {
        Jet0::new(arr1(&values.iter().copied().map(c).collect::<Vec<_>>()))
    }

    fn assert_complex_close(actual: C, expected: C) {
        let error = (actual - expected).norm();

        assert!(
            error <= TOLERANCE,
            "expected {expected:?}, \
             got {actual:?}; error = {error:e}",
        );
    }

    fn assert_complex_slice_close(
        actual: impl IntoIterator<Item = C>,
        expected: impl IntoIterator<Item = C>,
    ) {
        for (actual, expected) in actual.into_iter().zip(expected) {
            assert_complex_close(actual, expected);
        }
    }

    #[test]
    fn scalar_samples_stack_into_spatial_axis_without_component_permutation() {
        let mut sequences = FieldComponentSequences::with_capacity(2);

        sequences.push(ElectromagneticFieldComponents::new(
            scalar(1.0),
            scalar(2.0),
            scalar(3.0),
            scalar(4.0),
            scalar(5.0),
            scalar(6.0),
        ));

        sequences.push(ElectromagneticFieldComponents::new(
            scalar(11.0),
            scalar(12.0),
            scalar(13.0),
            scalar(14.0),
            scalar(15.0),
            scalar(16.0),
        ));

        let fields = sequences.stack().expect("field components should stack");

        let electric = fields.electric();
        let magnetic = fields.magnetic();

        assert_eq!(electric.x().shape(), &[2]);
        assert_eq!(electric.y().shape(), &[2]);
        assert_eq!(electric.z().shape(), &[2]);

        assert_eq!(magnetic.x().shape(), &[2]);
        assert_eq!(magnetic.y().shape(), &[2]);
        assert_eq!(magnetic.z().shape(), &[2]);

        assert_complex_slice_close(electric.x().iter().copied(), [c(1.0), c(11.0)]);

        assert_complex_slice_close(electric.y().iter().copied(), [c(2.0), c(12.0)]);

        assert_complex_slice_close(electric.z().iter().copied(), [c(3.0), c(13.0)]);

        assert_complex_slice_close(magnetic.x().iter().copied(), [c(4.0), c(14.0)]);

        assert_complex_slice_close(magnetic.y().iter().copied(), [c(5.0), c(15.0)]);

        assert_complex_slice_close(magnetic.z().iter().copied(), [c(6.0), c(16.0)]);
    }

    #[test]
    fn sampled_inputs_append_position_as_final_axis() {
        let mut sequences = FieldComponentSequences::with_capacity(3);

        sequences.push(ElectromagneticFieldComponents::new(
            vector(&[1.0, 2.0]),
            vector(&[11.0, 12.0]),
            vector(&[21.0, 22.0]),
            vector(&[31.0, 32.0]),
            vector(&[41.0, 42.0]),
            vector(&[51.0, 52.0]),
        ));

        sequences.push(ElectromagneticFieldComponents::new(
            vector(&[3.0, 4.0]),
            vector(&[13.0, 14.0]),
            vector(&[23.0, 24.0]),
            vector(&[33.0, 34.0]),
            vector(&[43.0, 44.0]),
            vector(&[53.0, 54.0]),
        ));

        sequences.push(ElectromagneticFieldComponents::new(
            vector(&[5.0, 6.0]),
            vector(&[15.0, 16.0]),
            vector(&[25.0, 26.0]),
            vector(&[35.0, 36.0]),
            vector(&[45.0, 46.0]),
            vector(&[55.0, 56.0]),
        ));

        let fields = sequences.stack().expect("field components should stack");

        let electric = fields.electric();
        let magnetic = fields.magnetic();

        assert_eq!(electric.x().shape(), &[2, 3]);
        assert_eq!(electric.y().shape(), &[2, 3]);
        assert_eq!(electric.z().shape(), &[2, 3]);

        assert_eq!(magnetic.x().shape(), &[2, 3]);
        assert_eq!(magnetic.y().shape(), &[2, 3]);
        assert_eq!(magnetic.z().shape(), &[2, 3]);

        assert_eq!(
            electric.x(),
            &array![[c(1.0), c(3.0), c(5.0)], [c(2.0), c(4.0), c(6.0)],],
        );

        assert_eq!(
            electric.y(),
            &array![[c(11.0), c(13.0), c(15.0)], [c(12.0), c(14.0), c(16.0)],],
        );

        assert_eq!(
            electric.z(),
            &array![[c(21.0), c(23.0), c(25.0)], [c(22.0), c(24.0), c(26.0)],],
        );

        assert_eq!(
            magnetic.x(),
            &array![[c(31.0), c(33.0), c(35.0)], [c(32.0), c(34.0), c(36.0)],],
        );

        assert_eq!(
            magnetic.y(),
            &array![[c(41.0), c(43.0), c(45.0)], [c(42.0), c(44.0), c(46.0)],],
        );

        assert_eq!(
            magnetic.z(),
            &array![[c(51.0), c(53.0), c(55.0)], [c(52.0), c(54.0), c(56.0)],],
        );
    }

    #[test]
    fn duplicate_samples_are_preserved() {
        let sample = || {
            ElectromagneticFieldComponents::new(
                scalar(1.0),
                scalar(2.0),
                scalar(3.0),
                scalar(4.0),
                scalar(5.0),
                scalar(6.0),
            )
        };

        let mut sequences = FieldComponentSequences::with_capacity(2);

        sequences.push(sample());
        sequences.push(sample());

        let fields = sequences.stack().expect("field components should stack");

        assert_eq!(fields.electric().x(), &arr1(&[c(1.0), c(1.0)]),);

        assert_eq!(fields.electric().y(), &arr1(&[c(2.0), c(2.0)]),);

        assert_eq!(fields.electric().z(), &arr1(&[c(3.0), c(3.0)]),);

        assert_eq!(fields.magnetic().x(), &arr1(&[c(4.0), c(4.0)]),);

        assert_eq!(fields.magnetic().y(), &arr1(&[c(5.0), c(5.0)]),);

        assert_eq!(fields.magnetic().z(), &arr1(&[c(6.0), c(6.0)]),);
    }
}

#[cfg(test)]
mod integration_tests {
    use ndarray::Ix0;

    use crate::{
        FiniteLayerIndex, IncidentSide, Polarisation, RealAxis,
        algebra::{ArrayJet0, Jet, RealParameter},
        backend::{
            ExteriorContextProvider, PlaneWaveSolutionSource, RetainedIsotropicLayers, RunMode,
            Scatter2,
        },
        input::{CanonicalCoordinates, CanonicalStack},
        material::Constant,
        spatial::{CanonicalFieldPosition, CanonicalLayerPosition, CompiledFieldSampling},
        test_support::{
            C, TOLERANCE, assertions::assert_complex_close, jet::zero_jet_from_real_value,
            planar::boundary_test_single_layer_stack,
        },
        waves::WaveSamplingContext,
    };

    use super::{FieldSamplingContext, project_isotropic_field_components};

    type A = ArrayJet0<C, Ix0, RealParameter>;

    fn coordinates() -> CanonicalCoordinates<A> {
        CanonicalCoordinates::new(
            zero_jet_from_real_value(2.3),
            zero_jet_from_real_value(0.37),
        )
    }

    fn build_workspace(
        stack: CanonicalStack<Constant<f64>, A>,
    ) -> crate::backend::scatter2::Scatter2Workspace<A> {
        Scatter2::new()
            .accumulate::<A, RealAxis, _>(
                &coordinates(),
                &stack,
                Polarisation::TransverseElectric,
                RunMode::InternalFields,
            )
            .expect("scatter workspace accumulation should succeed")
    }

    fn assert_zero_component(values: impl IntoIterator<Item = C>) {
        for value in values {
            assert_complex_close(value, C::new(0.0, 0.0), TOLERANCE);
        }
    }

    #[test]
    fn reconstructs_stacked_te_fields_from_retained_workspace() {
        let workspace = build_workspace(boundary_test_single_layer_stack());

        let positions = vec![
            CanonicalFieldPosition::LeftExterior { distance: 0.15 },
            CanonicalFieldPosition::Layer {
                index: FiniteLayerIndex(0),
                position: CanonicalLayerPosition::FromLeft(0.10),
            },
            CanonicalFieldPosition::RightExterior { distance: 0.20 },
            CanonicalFieldPosition::Layer {
                index: FiniteLayerIndex(0),
                position: CanonicalLayerPosition::FromLeft(0.10),
            },
        ];

        let sampling = CompiledFieldSampling::new(positions.clone());

        let context = FieldSamplingContext::new(&workspace);

        let fields = context
            .reconstruct(IncidentSide::Left, &sampling)
            .expect("field reconstruction should succeed");

        let electric = fields.electric();
        let magnetic = fields.magnetic();

        assert_eq!(electric.x().shape(), &[4],);

        assert_eq!(electric.y().shape(), &[4],);

        assert_eq!(electric.z().shape(), &[4],);

        assert_eq!(magnetic.x().shape(), &[4],);

        assert_eq!(magnetic.y().shape(), &[4],);

        assert_eq!(magnetic.z().shape(), &[4],);

        assert_zero_component(electric.x().iter().copied());

        assert_zero_component(electric.z().iter().copied());

        assert_zero_component(magnetic.y().iter().copied());

        let wave_context = WaveSamplingContext::new(&workspace);

        let sampled_waves = wave_context
            .propagate_sampling(IncidentSide::Left, &sampling)
            .expect("wave sampling should succeed");

        assert_eq!(sampled_waves.len(), positions.len(),);

        let solution = workspace.solution();
        let exterior = solution.context();

        let k0 = exterior.vacuum_angular_wavenumber();

        let beta = exterior.parallel_angular_wavenumber();

        let polarisation = exterior.polarisation();

        for (sample_index, (position, waves)) in
            positions.into_iter().zip(sampled_waves).enumerate()
        {
            let waves: crate::observable::BoundaryWaves<_> = waves.into();

            let expected = match position {
                CanonicalFieldPosition::LeftExterior { .. } => {
                    let state = waves.into_state(exterior.left_admittance());

                    project_isotropic_field_components(
                        state,
                        polarisation,
                        exterior.left_kappa(),
                        exterior.left_admittance(),
                        k0,
                        beta,
                    )
                }

                CanonicalFieldPosition::Layer { index, .. } => {
                    let quantities = workspace.layer_quantities(index.0).expect(
                        "sampled layer should have \
                             retained quantities",
                    );

                    let admittance = quantities.admittance().into_inner();

                    let state = waves.into_state(&admittance);

                    project_isotropic_field_components(
                        state,
                        quantities.polarisation(),
                        quantities.kappa(),
                        &admittance,
                        k0,
                        beta,
                    )
                }

                CanonicalFieldPosition::RightExterior { .. } => {
                    let state = waves.into_state(exterior.right_admittance());

                    project_isotropic_field_components(
                        state,
                        polarisation,
                        exterior.right_kappa(),
                        exterior.right_admittance(),
                        k0,
                        beta,
                    )
                }
            };

            let (expected_ex, expected_ey, expected_ez, expected_hx, expected_hy, expected_hz) =
                expected.into_parts();

            assert_complex_close(
                electric.x()[sample_index],
                expected_ex.value()[()],
                TOLERANCE,
            );

            assert_complex_close(
                electric.y()[sample_index],
                expected_ey.value()[()],
                TOLERANCE,
            );

            assert_complex_close(
                electric.z()[sample_index],
                expected_ez.value()[()],
                TOLERANCE,
            );

            assert_complex_close(
                magnetic.x()[sample_index],
                expected_hx.value()[()],
                TOLERANCE,
            );

            assert_complex_close(
                magnetic.y()[sample_index],
                expected_hy.value()[()],
                TOLERANCE,
            );

            assert_complex_close(
                magnetic.z()[sample_index],
                expected_hz.value()[()],
                TOLERANCE,
            );
        }

        assert_complex_close(electric.y()[1], electric.y()[3], TOLERANCE);

        assert_complex_close(magnetic.x()[1], magnetic.x()[3], TOLERANCE);

        assert_complex_close(magnetic.z()[1], magnetic.z()[3], TOLERANCE);
    }
}
