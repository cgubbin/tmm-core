//! Internal boundary-wave reconstruction for the scalar 2×2 scattering backend.
//!
//! This module implements [`PlaneWaveFieldBackend`] for [`Scatter2`].
//! The ordinary external plane-wave response and all finite-layer boundary
//! amplitudes are obtained from the same scattering workspace.
//!
//! When internal fields are requested, the scattering accumulation retains
//! each interface and propagation component. Prefix and suffix cascades are
//! then used to solve the forward- and backward-propagating amplitudes at both
//! boundaries of every finite layer.
//!
//! Returned wave directions are geometric:
//!
//! - forward means left to right;
//! - backward means right to left.
//!
//! These meanings do not change with the incident side.
//!
//! Value, first-derivative, and second-derivative paths share the same
//! reconstruction algebra. Jet-valued components carry derivatives through
//! the prefix, suffix, and cut-wave calculations automatically.
use crate::{
    ComplexScalar, DerivativeVariable, IncidentSide, PlanarInput, PlaneWaveInput, Stack,
    backend::{
        RealAxis,
        field::{
            BidirectionalWaveDifferential, BidirectionalWaves, BoundaryWaveDerivatives,
            BoundaryWaveSolution, BoundaryWaves, ExteriorBoundaryWaveDifferential,
            ExteriorBoundaryWaves, PlaneWaveFieldBackend, PlaneWaveFieldKxDerivativeBackend,
            PlaneWaveFieldResponse, PlaneWaveFieldSpectralDerivativeBackend,
            PlaneWaveFieldThicknessDerivativeBackend, first_order_fields_from_generic,
            second_order_fields_from_generic, value_fields_from_generic,
        },
        input::AlgebraicPlanarInput,
        jet::{ArrayJet, ArrayJetFirst},
        scatter2::{
            Scatter2,
            plane_wave::{
                plane_wave_from_amplitudes, plane_wave_from_first_jet_amplitudes,
                plane_wave_from_second_jet_amplitudes, plane_wave_from_spectral_jet_amplitudes,
            },
        },
    },
    material::{EvaluateDifferentiableMaterial, EvaluateMaterial},
};

use ndarray::{ArrayBase, Dimension, OwnedRepr};
use num_traits::Float;

impl<C, D, M> PlaneWaveFieldBackend<C, D, Stack<M, C::RealField>> for Scatter2
where
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
    M: EvaluateMaterial<C, Real = C::RealField>,
{
    fn solve_plane_wave_internal_fields(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error> {
        let workspace = self.accumulate_real_axis(stack, input.planar())?;

        let planar = input.planar().clone().to_complex();
        let generic_fields = retained_boundary_waves(&workspace, input.incident_side(), &planar);

        let layers = value_fields_from_generic(generic_fields);

        let total = workspace.into_total();

        let (reflection, transmission) = total.clone().amplitudes(input.incident_side());

        let exterior = ExteriorBoundaryWaves::from_values(
            reflection.clone(),
            transmission.clone(),
            input.incident_side(),
        );

        let planar = AlgebraicPlanarInput::values(&planar);

        let response = plane_wave_from_amplitudes(
            reflection,
            transmission,
            &planar,
            stack,
            input.incident_side(),
        );

        let boundary_waves = BoundaryWaves::new(exterior, layers);

        Ok(PlaneWaveFieldResponse::new(
            response,
            BoundaryWaveSolution::Values(boundary_waves),
        ))
    }
}

impl<C, D, M> PlaneWaveFieldThicknessDerivativeBackend<C, D, Stack<M, C::RealField>> for Scatter2
where
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
{
    fn solve_plane_wave_internal_fields_thickness_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<<C>::RealField>, D>>,
        layer: usize,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input::<C>();
        let workspace =
            self.accumulate_thickness_first_with::<RealAxis, _, _, _>(stack, &planar, layer)?;

        let generic_fields = retained_boundary_waves(&workspace, input.incident_side(), &planar);

        let (layers, first_layers) = first_order_fields_from_generic(generic_fields);

        let total = workspace.into_total();

        let (reflection, transmission) = total.amplitudes(input.incident_side());

        let (exterior, exterior_first, reflection, transmission) =
            exterior_waves_from_first_jets(reflection, transmission, input.incident_side());

        let planar = AlgebraicPlanarInput::new(
            ArrayJetFirst::constant(planar.vacuum_wavenumber().clone()),
            ArrayJetFirst::constant(planar.parallel_wavenumber().clone()),
            planar.polarisation(),
        );

        let response = plane_wave_from_first_jet_amplitudes(
            reflection,
            transmission,
            &planar,
            stack,
            input.incident_side(),
            DerivativeVariable::Thickness(layer),
        );

        let derivatives = BoundaryWaveDerivatives::new(
            DerivativeVariable::Thickness(layer),
            exterior_first,
            first_layers,
        );

        Ok(PlaneWaveFieldResponse::new(
            response,
            BoundaryWaveSolution::new_with_derivative(exterior, layers, derivatives),
        ))
    }

    fn solve_plane_wave_internal_fields_thickness_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<<C>::RealField>, D>>,
        layer: usize,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input::<C>();

        let workspace =
            self.accumulate_thickness_second_with::<RealAxis, _, _, _>(stack, &planar, layer)?;

        let generic_fields = retained_boundary_waves(&workspace, input.incident_side(), &planar);

        let (layers, first_layers, second_layers) =
            second_order_fields_from_generic(generic_fields);

        let total = workspace.into_total();

        let (reflection, transmission) = total.amplitudes(input.incident_side());

        let (exterior, exterior_first, exterior_second, reflection, transmission) =
            exterior_waves_from_second_jets(reflection, transmission, input.incident_side());

        let planar = AlgebraicPlanarInput::new(
            ArrayJet::constant(planar.vacuum_wavenumber().clone()),
            ArrayJet::constant(planar.parallel_wavenumber().clone()),
            planar.polarisation(),
        );

        let response = plane_wave_from_second_jet_amplitudes(
            reflection,
            transmission,
            &planar,
            stack,
            input.incident_side(),
            DerivativeVariable::Thickness(layer),
        );

        let derivatives = BoundaryWaveDerivatives::new(
            DerivativeVariable::Thickness(layer),
            exterior_first,
            first_layers,
        )
        .with_second(exterior_second, second_layers);

        Ok(PlaneWaveFieldResponse::new(
            response,
            BoundaryWaveSolution::new_with_derivative(exterior, layers, derivatives),
        ))
    }
}

impl<C, D, M> PlaneWaveFieldKxDerivativeBackend<C, D, Stack<M, C::RealField>> for Scatter2
where
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
{
    fn solve_plane_wave_internal_fields_kx_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<<C>::RealField>, D>>,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input::<C>();
        let workspace = self.accumulate_kx_first_with::<RealAxis, _, _, _>(stack, &planar)?;

        let generic_fields = retained_boundary_waves(&workspace, input.incident_side(), &planar);

        let (layers, first_layers) = first_order_fields_from_generic(generic_fields);

        let total = workspace.into_total();

        let (reflection, transmission) = total.amplitudes(input.incident_side());

        let (exterior, exterior_first, reflection, transmission) =
            exterior_waves_from_first_jets(reflection, transmission, input.incident_side());

        let planar = AlgebraicPlanarInput::new(
            ArrayJetFirst::constant(planar.vacuum_wavenumber().clone()),
            ArrayJetFirst::variable(planar.parallel_wavenumber().clone()),
            planar.polarisation(),
        );

        let response = plane_wave_from_first_jet_amplitudes(
            reflection,
            transmission,
            &planar,
            stack,
            input.incident_side(),
            DerivativeVariable::ParallelWavenumber,
        );

        let derivatives = BoundaryWaveDerivatives::new(
            DerivativeVariable::ParallelWavenumber,
            exterior_first,
            first_layers,
        );

        Ok(PlaneWaveFieldResponse::new(
            response,
            BoundaryWaveSolution::new_with_derivative(exterior, layers, derivatives),
        ))
    }

    fn solve_plane_wave_internal_fields_kx_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<<C>::RealField>, D>>,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input::<C>();

        let workspace = self.accumulate_kx_second_with::<RealAxis, _, _, _>(stack, &planar)?;

        let generic_fields = retained_boundary_waves(&workspace, input.incident_side(), &planar);

        let (layers, first_layers, second_layers) =
            second_order_fields_from_generic(generic_fields);

        let total = workspace.into_total();

        let (reflection, transmission) = total.amplitudes(input.incident_side());

        let (exterior, exterior_first, exterior_second, reflection, transmission) =
            exterior_waves_from_second_jets(reflection, transmission, input.incident_side());

        let planar = AlgebraicPlanarInput::new(
            ArrayJet::constant(planar.vacuum_wavenumber().clone()),
            ArrayJet::variable(planar.parallel_wavenumber().clone()),
            planar.polarisation(),
        );

        let response = plane_wave_from_second_jet_amplitudes(
            reflection,
            transmission,
            &planar,
            stack,
            input.incident_side(),
            DerivativeVariable::ParallelWavenumber,
        );

        let derivatives = BoundaryWaveDerivatives::new(
            DerivativeVariable::ParallelWavenumber,
            exterior_first,
            first_layers,
        )
        .with_second(exterior_second, second_layers);

        Ok(PlaneWaveFieldResponse::new(
            response,
            BoundaryWaveSolution::new_with_derivative(exterior, layers, derivatives),
        ))
    }
}

impl<C, D, M> PlaneWaveFieldSpectralDerivativeBackend<C, D, Stack<M, C::RealField>> for Scatter2
where
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
{
    fn solve_plane_wave_internal_fields_k0_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input::<C>();

        let workspace = self.accumulate_k0_first_with::<RealAxis, _, _, _>(stack, &planar)?;

        let generic_fields = retained_boundary_waves(&workspace, input.incident_side(), &planar);

        let (layers, first_layers) = first_order_fields_from_generic(generic_fields);

        let total = workspace.into_total();

        let (reflection, transmission) = total.amplitudes(input.incident_side());

        let (exterior, exterior_first, reflection, transmission) =
            exterior_waves_from_first_jets(reflection, transmission, input.incident_side());

        let planar = AlgebraicPlanarInput::new(
            ArrayJetFirst::variable(planar.vacuum_wavenumber().clone()),
            ArrayJetFirst::constant(planar.parallel_wavenumber().clone()),
            planar.polarisation(),
        );

        let response = plane_wave_from_first_jet_amplitudes(
            reflection,
            transmission,
            &planar,
            stack,
            input.incident_side(),
            DerivativeVariable::VacuumWavenumber,
        );

        let derivatives = BoundaryWaveDerivatives::new(
            DerivativeVariable::VacuumWavenumber,
            exterior_first,
            first_layers,
        );

        Ok(PlaneWaveFieldResponse::new(
            response,
            BoundaryWaveSolution::new_with_derivative(exterior, layers, derivatives),
        ))
    }

    fn solve_plane_wave_internal_fields_k0_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input::<C>();
        let workspace = self.accumulate_k0_second_with::<RealAxis, _, _, _>(stack, &planar)?;

        let generic_fields = retained_boundary_waves(&workspace, input.incident_side(), &planar);

        let (layers, first_layers, second_layers) =
            second_order_fields_from_generic(generic_fields);

        let total = workspace.into_total();

        let (reflection, transmission) = total.amplitudes(input.incident_side());

        let (exterior, exterior_first, exterior_second, reflection, transmission) =
            exterior_waves_from_second_jets(reflection, transmission, input.incident_side());

        let planar = AlgebraicPlanarInput::new(
            ArrayJet::variable(planar.vacuum_wavenumber().clone()),
            ArrayJet::constant(planar.parallel_wavenumber().clone()),
            planar.polarisation(),
        );

        let response = plane_wave_from_second_jet_amplitudes(
            reflection,
            transmission,
            &planar,
            stack,
            input.incident_side(),
            DerivativeVariable::VacuumWavenumber,
        );

        let derivatives = BoundaryWaveDerivatives::new(
            DerivativeVariable::VacuumWavenumber,
            exterior_first,
            first_layers,
        )
        .with_second(exterior_second, second_layers);

        Ok(PlaneWaveFieldResponse::new(
            response,
            BoundaryWaveSolution::new_with_derivative(exterior, layers, derivatives),
        ))
    }

    fn solve_plane_wave_internal_fields_full_spectral_hessian(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input::<C>();
        let workspace =
            self.accumulate_full_spectral_hessian::<RealAxis, _, _, _>(stack, &planar)?;

        todo!()
        // let generic_fields = retained_boundary_waves(&workspace, input.incident_side(), &planar);

        // let (layers, first_layers, second_layers) =
        //     second_order_fields_from_generic(generic_fields);

        // let total = workspace.into_total();

        // let (reflection, transmission) = total.amplitudes(input.incident_side());

        // let (exterior, exterior_first, exterior_second, reflection, transmission) =
        //     exterior_waves_from_second_jets(reflection, transmission, input.incident_side());

        // let planar = AlgebraicPlanarInput::new(
        //     ArrayJet::variable(planar.vacuum_wavenumber().clone()),
        //     ArrayJet::constant(planar.parallel_wavenumber().clone()),
        //     planar.polarisation(),
        // );

        // let response = plane_wave_from_second_jet_amplitudes(
        //     reflection,
        //     transmission,
        //     &planar,
        //     stack,
        //     input.incident_side(),
        //     DerivativeVariable::VacuumWavenumber,
        // );

        // let derivatives =
        //     BoundaryWaveDerivatives::new(DerivativeVariable::VacuumWavenumber, exterior_first, first_layers)
        //         .with_second(exterior_second, second_layers);

        // Ok(PlaneWaveFieldResponse::new(
        //     response,
        //     BoundaryWaveSolution::new_with_derivative(exterior, layers, derivatives),
        // ))
    }
}

pub(crate) fn retained_boundary_waves<C, D, A>(
    workspace: &crate::backend::scatter2::workspace::ScatterWorkspace<A>,
    incident_side: IncidentSide,
    planar: &PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
) -> Vec<crate::backend::field::LayerBoundaryWavesGeneric<A>>
where
    C: ComplexScalar,
    D: Dimension,
    A: crate::backend::algebra::ScalarAlgebra<C, D> + Clone,
{
    workspace
        .reconstruct_layer_boundary_waves(incident_side, planar.vacuum_wavenumber())
        .expect("LayerBoundaries was requested, so retained boundary waves must exist")
}

#[allow(clippy::type_complexity)]
fn exterior_waves_from_first_jets<C, D>(
    reflection: ArrayJetFirst<C, D>,
    transmission: ArrayJetFirst<C, D>,
    incident_side: IncidentSide,
) -> (
    ExteriorBoundaryWaves<C, D>,
    ExteriorBoundaryWaveDifferential<C, D>,
    ArrayJetFirst<C, D>,
    ArrayJetFirst<C, D>,
)
where
    C: ComplexScalar,
    D: Dimension,
{
    let reflection_for_response = reflection.clone();

    let transmission_for_response = transmission.clone();

    let (reflection, d_reflection) = reflection.into_parts();

    let (transmission, d_transmission) = transmission.into_parts();

    let one = reflection.mapv(|_| C::one());
    let zero = reflection.mapv(|_| C::zero());

    let derivative_zero = reflection.mapv(|_| C::zero());

    let (exterior, first) = match incident_side {
        IncidentSide::Left => (
            ExteriorBoundaryWaves::new(
                BidirectionalWaves::new(one, reflection),
                BidirectionalWaves::new(transmission, zero),
            ),
            ExteriorBoundaryWaveDifferential::new(
                BidirectionalWaveDifferential::new(derivative_zero.clone(), d_reflection),
                BidirectionalWaveDifferential::new(d_transmission, derivative_zero),
            ),
        ),

        IncidentSide::Right => (
            ExteriorBoundaryWaves::new(
                BidirectionalWaves::new(zero, transmission),
                BidirectionalWaves::new(reflection, one),
            ),
            ExteriorBoundaryWaveDifferential::new(
                BidirectionalWaveDifferential::new(derivative_zero.clone(), d_transmission),
                BidirectionalWaveDifferential::new(d_reflection, derivative_zero),
            ),
        ),
    };

    (
        exterior,
        first,
        reflection_for_response,
        transmission_for_response,
    )
}

#[allow(clippy::type_complexity)]
fn exterior_waves_from_second_jets<C, D>(
    reflection: ArrayJet<C, D>,
    transmission: ArrayJet<C, D>,
    incident_side: IncidentSide,
) -> (
    ExteriorBoundaryWaves<C, D>,
    ExteriorBoundaryWaveDifferential<C, D>,
    ExteriorBoundaryWaveDifferential<C, D>,
    ArrayJet<C, D>,
    ArrayJet<C, D>,
)
where
    C: ComplexScalar,
    D: Dimension,
{
    /*
     * Preserve the original jets for construction of the external
     * PlaneWaveResponse.
     */
    let reflection_for_response = reflection.clone();
    let transmission_for_response = transmission.clone();

    let (reflection, reflection_first, reflection_second) = reflection.into_parts();

    let (transmission, transmission_first, transmission_second) = transmission.into_parts();

    /*
     * The imposed incident amplitude is constant with respect to every
     * derivative variable.
     */
    let one = reflection.mapv(|_| C::one());
    let zero = reflection.mapv(|_| C::zero());

    let derivative_zero = reflection.mapv(|_| C::zero());

    match incident_side {
        IncidentSide::Left => {
            /*
             * Left incidence:
             *
             * left exterior:
             *     forward  = 1
             *     backward = r
             *
             * right exterior:
             *     forward  = t
             *     backward = 0
             */
            let exterior = ExteriorBoundaryWaves::new(
                BidirectionalWaves::new(one, reflection),
                BidirectionalWaves::new(transmission, zero),
            );

            let first = ExteriorBoundaryWaveDifferential::new(
                BidirectionalWaveDifferential::new(derivative_zero.clone(), reflection_first),
                BidirectionalWaveDifferential::new(transmission_first, derivative_zero.clone()),
            );

            let second = ExteriorBoundaryWaveDifferential::new(
                BidirectionalWaveDifferential::new(derivative_zero.clone(), reflection_second),
                BidirectionalWaveDifferential::new(transmission_second, derivative_zero),
            );

            (
                exterior,
                first,
                second,
                reflection_for_response,
                transmission_for_response,
            )
        }

        IncidentSide::Right => {
            /*
             * Right incidence:
             *
             * left exterior:
             *     forward  = 0
             *     backward = t
             *
             * right exterior:
             *     forward  = r
             *     backward = 1
             */
            let exterior = ExteriorBoundaryWaves::new(
                BidirectionalWaves::new(zero, transmission),
                BidirectionalWaves::new(reflection, one),
            );

            let first = ExteriorBoundaryWaveDifferential::new(
                BidirectionalWaveDifferential::new(derivative_zero.clone(), transmission_first),
                BidirectionalWaveDifferential::new(reflection_first, derivative_zero.clone()),
            );

            let second = ExteriorBoundaryWaveDifferential::new(
                BidirectionalWaveDifferential::new(derivative_zero.clone(), transmission_second),
                BidirectionalWaveDifferential::new(reflection_second, derivative_zero),
            );

            (
                exterior,
                first,
                second,
                reflection_for_response,
                transmission_for_response,
            )
        }
    }
}

// #[cfg(test)]
// mod tests {
//     use approx::assert_relative_eq;
//     use ndarray::{ArrayBase, Dimension, OwnedRepr, arr0, arr1};
//     use num_complex::Complex64;
//     use num_traits::{One, Zero};

//     use super::*;

//     use crate::{
//         IncidentSide, Polarisation, ValidationConfig,
//         backend::{DifferentiablePlaneWaveBackend, PlaneWaveBackend, field::PlaneWaveFieldBackend},
//         material::Constant,
//         stack::Thickness,
//     };

//     type C = Complex64;

//     const TOLERANCE: f64 = 1e-10;

//     fn c(value: f64) -> C {
//         C::new(value, 0.0)
//     }

//     fn assert_complex_close(actual: C, expected: C, tolerance: f64) {
//         assert_relative_eq!(
//             actual.re,
//             expected.re,
//             epsilon = tolerance,
//             max_relative = tolerance,
//         );

//         assert_relative_eq!(
//             actual.im,
//             expected.im,
//             epsilon = tolerance,
//             max_relative = tolerance,
//         );
//     }

//     fn assert_array_close<D>(
//         actual: &ArrayBase<OwnedRepr<C>, D>,
//         expected: &ArrayBase<OwnedRepr<C>, D>,
//         tolerance: f64,
//     ) where
//         D: Dimension,
//     {
//         assert_eq!(actual.raw_dim(), expected.raw_dim());

//         for (&actual, &expected) in actual.iter().zip(expected.iter()) {
//             assert_complex_close(actual, expected, tolerance);
//         }
//     }

//     fn uniform_one_layer_stack() -> Stack<Constant<f64>, f64> {
//         Stack::builder(Constant::dielectric(1.0), Constant::dielectric(1.0))
//             .layer(Constant::dielectric(1.0), Thickness::from_cm(0.25).unwrap())
//             .build()
//             .unwrap()
//     }

//     fn two_layer_stack() -> Stack<Constant<f64>, f64> {
//         Stack::builder(Constant::dielectric(1.0), Constant::dielectric(1.69))
//             .layer(
//                 Constant::dielectric(2.25),
//                 Thickness::from_cm(0.17).unwrap(),
//             )
//             .layer(
//                 Constant::dielectric(1.44),
//                 Thickness::from_cm(0.29).unwrap(),
//             )
//             .build()
//             .unwrap()
//     }

//     fn input(side: IncidentSide) -> PlaneWaveInput<ArrayBase<OwnedRepr<f64>, ndarray::Ix0>> {
//         PlaneWaveInput::new(
//             PlanarInput::new(arr0(3.0), arr0(0.4), Polarisation::TransverseElectric),
//             side,
//         )
//     }
//     #[test]
//     fn field_response_contains_same_external_response_as_plane_wave_backend() {
//         let backend = Scatter2::new();
//         let stack = two_layer_stack();

//         for side in [IncidentSide::Left, IncidentSide::Right] {
//             let input = input(side);

//             let ordinary = backend.solve_plane_wave(&stack, &input).unwrap();

//             let field = backend
//                 .solve_plane_wave_internal_fields(&stack, &input)
//                 .unwrap();

//             assert_array_close(
//                 field.response().reflection(),
//                 ordinary.reflection(),
//                 TOLERANCE,
//             );

//             assert_array_close(
//                 field.response().transmission(),
//                 ordinary.transmission(),
//                 TOLERANCE,
//             );

//             assert_eq!(field.boundary_waves().len(), stack.len(),);
//         }
//     }

//     #[test]
//     fn uniform_layer_has_only_forward_internal_wave_for_left_incidence() {
//         let backend = Scatter2::new();
//         let stack = uniform_one_layer_stack();

//         let input = PlaneWaveInput::new(
//             PlanarInput::new(arr0(2.0), arr0(0.0), Polarisation::TransverseElectric),
//             IncidentSide::Left,
//         );

//         let result = backend
//             .solve_plane_wave_internal_fields(&stack, &input)
//             .unwrap();

//         let layer = result.boundary_waves().layer(0).unwrap();

//         assert_complex_close(layer.left().forward()[()], C::one(), TOLERANCE);

//         assert_complex_close(layer.left().backward()[()], C::zero(), TOLERANCE);

//         assert_complex_close(layer.right().backward()[()], C::zero(), TOLERANCE);

//         assert_complex_close(
//             layer.right().forward()[()],
//             result.response().transmission()[()],
//             TOLERANCE,
//         );
//     }

//     #[test]
//     fn uniform_layer_has_only_backward_internal_wave_for_right_incidence() {
//         let backend = Scatter2::new();
//         let stack = uniform_one_layer_stack();

//         let input = PlaneWaveInput::new(
//             PlanarInput::new(arr0(2.0), arr0(0.0), Polarisation::TransverseElectric),
//             IncidentSide::Right,
//         );

//         let result = backend
//             .solve_plane_wave_internal_fields(&stack, &input)
//             .unwrap();

//         let layer = result.boundary_waves().layer(0).unwrap();

//         assert_complex_close(layer.right().forward()[()], C::zero(), TOLERANCE);

//         assert_complex_close(layer.right().backward()[()], C::one(), TOLERANCE);

//         assert_complex_close(layer.left().forward()[()], C::zero(), TOLERANCE);

//         assert_complex_close(
//             layer.left().backward()[()],
//             result.response().transmission()[()],
//             TOLERANCE,
//         );
//     }

//     #[test]
//     fn total_scalar_field_is_continuous_between_adjacent_layers() {
//         let backend = Scatter2::new();
//         let stack = two_layer_stack();

//         let result = backend
//             .solve_plane_wave_internal_fields(&stack, &input(IncidentSide::Left))
//             .unwrap();

//         let first = result.boundary_waves().layer(0).unwrap();
//         let second = result.boundary_waves().layer(1).unwrap();

//         let first_field = first.right().forward().clone() + first.right().backward();

//         let second_field = second.left().forward().clone() + second.left().backward();

//         assert_array_close(&first_field, &second_field, TOLERANCE);
//     }

//     #[test]
//     fn empty_stack_returns_no_internal_layers() {
//         let stack = Stack::builder(Constant::dielectric(1.0), Constant::dielectric(2.25))
//             .validation(ValidationConfig::permissive())
//             .build()
//             .unwrap();

//         let result: PlaneWaveFieldResponse<C, _> = Scatter2::new()
//             .solve_plane_wave_internal_fields(&stack, &input(IncidentSide::Left))
//             .unwrap();

//         assert!(result.boundary_waves().is_empty());
//         assert!(result.boundary_waves().derivatives().is_none());
//     }

//     #[test]
//     fn first_order_field_response_matches_external_derivative_backend() {
//         let backend = Scatter2::new();
//         let stack = two_layer_stack();
//         let input = input(IncidentSide::Left);

//         let variable = SpectralDerivativeVariable::VacuumWavenumber;

//         let ordinary = backend
//             .solve_plane_wave_spectral_first_derivative(&stack, &input, variable)
//             .unwrap();

//         let field = backend
//             .solve_plane_wave_internal_fields_spectral_first_derivative(&stack, &input, variable)
//             .unwrap();

//         let ordinary_derivatives = ordinary.derivatives().unwrap();

//         let field_derivatives = field.response().derivatives().unwrap();

//         assert_array_close(
//             field_derivatives.first().amplitudes().reflection(),
//             ordinary_derivatives.first().amplitudes().reflection(),
//             TOLERANCE,
//         );

//         assert_array_close(
//             field_derivatives.first().amplitudes().transmission(),
//             ordinary_derivatives.first().amplitudes().transmission(),
//             TOLERANCE,
//         );

//         let internal = field.boundary_waves().derivatives().unwrap();

//         assert_eq!(internal.variable(), variable.into());

//         assert_eq!(internal.first_layers().len(), stack.len(),);

//         assert!(internal.second_layers().is_none());
//     }

//     #[test]
//     fn second_order_field_response_matches_external_derivative_backend() {
//         let backend = Scatter2::new();
//         let stack = two_layer_stack();
//         let input = input(IncidentSide::Right);

//         let ordinary = backend
//             .solve_plane_wave_structural_second_derivative(
//                 &stack,
//                 &input,
//                 StructuralDerivativeVariable::ParallelWavenumber,
//             )
//             .unwrap();

//         let field = backend
//             .solve_plane_wave_internal_fields_structural_second_derivative(
//                 &stack,
//                 &input,
//                 StructuralDerivativeVariable::ParallelWavenumber,
//             )
//             .unwrap();

//         let ordinary_derivatives = ordinary.derivatives().unwrap();

//         let field_derivatives = field.response().derivatives().unwrap();

//         assert_array_close(
//             field_derivatives
//                 .second()
//                 .unwrap()
//                 .amplitudes()
//                 .reflection(),
//             ordinary_derivatives
//                 .second()
//                 .unwrap()
//                 .amplitudes()
//                 .reflection(),
//             1e-9,
//         );

//         assert_array_close(
//             field_derivatives
//                 .second()
//                 .unwrap()
//                 .amplitudes()
//                 .transmission(),
//             ordinary_derivatives
//                 .second()
//                 .unwrap()
//                 .amplitudes()
//                 .transmission(),
//             1e-9,
//         );

//         let internal = field.boundary_waves().derivatives().unwrap();

//         assert_eq!(internal.first_layers().len(), stack.len(),);

//         assert_eq!(internal.second_layers().unwrap().len(), stack.len(),);
//     }

//     #[test]
//     fn internal_wave_first_derivative_matches_finite_difference() {
//         let backend = Scatter2::new();
//         let stack = two_layer_stack();

//         let k0 = 3.0;
//         let h = 1e-6;

//         let make_input = |k0| {
//             PlaneWaveInput::new(
//                 PlanarInput::new(arr0(k0), arr0(0.4), Polarisation::TransverseElectric),
//                 IncidentSide::Left,
//             )
//         };

//         let analytic = backend
//             .solve_plane_wave_internal_fields_spectral_first_derivative(
//                 &stack,
//                 &make_input(k0),
//                 SpectralDerivativeVariable::VacuumWavenumber,
//             )
//             .unwrap();

//         let plus: PlaneWaveFieldResponse<C, _> = backend
//             .solve_plane_wave_internal_fields(&stack, &make_input(k0 + h))
//             .unwrap();

//         let minus: PlaneWaveFieldResponse<C, _> = backend
//             .solve_plane_wave_internal_fields(&stack, &make_input(k0 - h))
//             .unwrap();

//         let analytic_first = analytic
//             .boundary_waves()
//             .derivatives()
//             .unwrap()
//             .first_layer(0)
//             .unwrap()
//             .left()
//             .forward()[()];

//         let expected = (plus.boundary_waves().layer(0).unwrap().left().forward()[()]
//             - minus.boundary_waves().layer(0).unwrap().left().forward()[()])
//             / (2.0 * h);

//         assert_complex_close(analytic_first, expected, 1e-6);
//     }

//     #[test]
//     fn sampled_internal_fields_preserve_input_shape() {
//         let backend = Scatter2::new();
//         let stack = two_layer_stack();

//         let input = PlaneWaveInput::new(
//             PlanarInput::new(
//                 arr1(&[2.5, 3.0, 3.5]),
//                 arr1(&[0.2, 0.3, 0.4]),
//                 Polarisation::TransverseMagnetic,
//             ),
//             IncidentSide::Left,
//         );

//         let result: PlaneWaveFieldResponse<C, _> = backend
//             .solve_plane_wave_internal_fields_spectral_second_derivative(
//                 &stack,
//                 &input,
//                 SpectralDerivativeVariable::VacuumWavenumber,
//             )
//             .unwrap();

//         for layer in result.boundary_waves().layers() {
//             assert_eq!(
//                 layer.left().forward().raw_dim(),
//                 input.planar().vacuum_wavenumber().raw_dim(),
//             );

//             assert_eq!(
//                 layer.left().backward().raw_dim(),
//                 input.planar().vacuum_wavenumber().raw_dim(),
//             );

//             assert_eq!(
//                 layer.right().forward().raw_dim(),
//                 input.planar().vacuum_wavenumber().raw_dim(),
//             );

//             assert_eq!(
//                 layer.right().backward().raw_dim(),
//                 input.planar().vacuum_wavenumber().raw_dim(),
//             );
//         }
//     }
// }
