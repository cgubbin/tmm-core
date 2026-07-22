//! Internal boundary-wave reconstruction for the scalar 2×2 transfer backend.
//!
//! This module implements [`PlaneWaveFieldBackend`] for [`Transfer2`].
//! The external plane-wave response and finite-layer boundary amplitudes are
//! obtained from the same transfer workspace.
//!
//! When internal fields are requested, transfer accumulation retains each
//! finite-layer transfer matrix together with its evaluated isotropic layer
//! quantities. Reconstruction starts from the complete state at the right
//! exterior boundary and propagates through the retained layers from right to
//! left.
//!
//! Returned wave directions are geometric:
//!
//! - forward means left to right;
//! - backward means right to left.
//!
//! These meanings do not change with the incident side.
//!
//! Value, first-derivative, and second-derivative paths share the same
//! reconstruction algebra. Jet-valued matrices, layer quantities, amplitudes,
//! and boundary states carry derivatives through the reconstruction
//! automatically.

use crate::{
    ComplexScalar, IncidentSide, PlaneWaveInput, Stack,
    backend::{
        RealAxis,
        algebra::ScalarAlgebra,
        derivative::{SpectralDerivativeVariable, StructuralDerivativeVariable},
        field::{
            BidirectionalWaveDifferential, BidirectionalWaves, BoundaryWaveDerivatives,
            BoundaryWaveSolution, BoundaryWaves, DifferentiablePlaneWaveFieldBackend,
            ExteriorBoundaryWaveDifferential, ExteriorBoundaryWaves, InternalFieldRequest,
            LayerBoundaryWavesGeneric, PlaneWaveFieldBackend, PlaneWaveFieldResponse,
            first_order_fields_from_generic, second_order_fields_from_generic,
            value_fields_from_generic,
        },
        isotropic::IsotropicLayerQuantities,
        jet::{ArrayJet, ArrayJetFirst},
        scatter2::plane_wave::{
            plane_wave_from_amplitudes, plane_wave_from_first_jet_amplitudes_spectral,
            plane_wave_from_first_jet_amplitudes_structural,
            plane_wave_from_second_jet_amplitudes_spectral,
            plane_wave_from_second_jet_amplitudes_structural,
        },
        transfer2::{
            Transfer2, TransferMatrix2,
            workspace::{TransferState, TransferWorkspace},
        },
    },
    material::{EvaluateDifferentiableMaterial, EvaluateMaterial},
};

use ndarray::{ArrayBase, Dimension, OwnedRepr};

use num_traits::Float;

impl<C, D, M> PlaneWaveFieldBackend<C, D, Stack<M, C::RealField>> for Transfer2
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
        let planar = input.complex_planar_input::<C>();

        let workspace = self.accumulate_with::<RealAxis, _, _, _>(
            stack,
            &planar,
            InternalFieldRequest::LayerBoundaries,
        )?;

        let total = workspace.total();

        let left_admittance = IsotropicLayerQuantities::real_axis(stack.left_exterior(), &planar)
            .into_admittance()
            .into_inner();

        let right_admittance = IsotropicLayerQuantities::real_axis(stack.right_exterior(), &planar)
            .into_admittance()
            .into_inner();

        let matrix: TransferMatrix2<C, D> = total.clone().into();

        let (reflection, transmission) =
            matrix.amplitudes(&left_admittance, &right_admittance, input.incident_side());

        let generic_fields = retained_boundary_waves(
            &workspace,
            input.incident_side(),
            &reflection,
            &transmission,
            &right_admittance,
            planar.vacuum_wavenumber(),
        );

        let layers = value_fields_from_generic(generic_fields);

        let exterior = ExteriorBoundaryWaves::from_values(
            reflection.clone(),
            transmission.clone(),
            input.incident_side(),
        );

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

    fn solve_plane_wave_internal_fields_structural_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input::<C>();

        let workspace = self.accumulate_structural_first_with::<RealAxis, _, _, _>(
            stack,
            &planar,
            variable,
            InternalFieldRequest::LayerBoundaries,
        )?;

        let mut left_admittance = IsotropicLayerQuantities::evaluate_first_structural_real_axis(
            stack.left_exterior(),
            &planar,
            variable,
        )
        .into_admittance()
        .into_inner();

        let mut right_admittance = IsotropicLayerQuantities::evaluate_first_structural_real_axis(
            stack.right_exterior(),
            &planar,
            variable,
        )
        .into_admittance()
        .into_inner();

        if let Some(rule) = variable.chain_rule(&planar) {
            left_admittance = left_admittance.chain_rule(&rule);
            right_admittance = right_admittance.chain_rule(&rule);
        }

        let (reflection, transmission) = workspace.total().clone().amplitude_jets(
            &left_admittance,
            &right_admittance,
            input.incident_side(),
        );

        let generic_fields = retained_boundary_waves(
            &workspace,
            input.incident_side(),
            &reflection,
            &transmission,
            &right_admittance,
            planar.vacuum_wavenumber(),
        );

        let (layers, first_layers) = first_order_fields_from_generic(generic_fields);

        let (exterior, exterior_first, reflection, transmission) =
            exterior_waves_from_first_jets(reflection, transmission, input.incident_side());

        let response = plane_wave_from_first_jet_amplitudes_structural(
            reflection,
            transmission,
            &planar,
            stack,
            input.incident_side(),
            variable,
        );

        let derivatives =
            BoundaryWaveDerivatives::new(variable.into(), exterior_first, first_layers);

        Ok(PlaneWaveFieldResponse::new(
            response,
            BoundaryWaveSolution::new_with_derivative(exterior, layers, derivatives),
        ))
    }

    fn solve_plane_wave_internal_fields_structural_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input::<C>();

        let workspace = self.accumulate_structural_second_with::<RealAxis, _, _, _>(
            stack,
            &planar,
            variable,
            InternalFieldRequest::LayerBoundaries,
        )?;

        let mut left_admittance = IsotropicLayerQuantities::evaluate_second_structural_real_axis(
            stack.left_exterior(),
            &planar,
            variable,
        )
        .into_admittance()
        .into_inner();

        let mut right_admittance = IsotropicLayerQuantities::evaluate_second_structural_real_axis(
            stack.right_exterior(),
            &planar,
            variable,
        )
        .into_admittance()
        .into_inner();

        if let Some(rule) = variable.chain_rule(&planar) {
            left_admittance = left_admittance.chain_rule(&rule);
            right_admittance = right_admittance.chain_rule(&rule);
        }

        let (reflection, transmission) = workspace.total().clone().amplitude_jets(
            &left_admittance,
            &right_admittance,
            input.incident_side(),
        );

        let generic_fields = retained_boundary_waves(
            &workspace,
            input.incident_side(),
            &reflection,
            &transmission,
            &right_admittance,
            planar.vacuum_wavenumber(),
        );

        let (layers, first_layers, second_layers) =
            second_order_fields_from_generic(generic_fields);

        let (exterior, exterior_first, exterior_second, reflection, transmission) =
            exterior_waves_from_second_jets(reflection, transmission, input.incident_side());

        let response = plane_wave_from_second_jet_amplitudes_structural(
            reflection,
            transmission,
            &planar,
            stack,
            input.incident_side(),
            variable,
        );

        let derivatives =
            BoundaryWaveDerivatives::new(variable.into(), exterior_first, first_layers)
                .with_second(exterior_second, second_layers);

        Ok(PlaneWaveFieldResponse::new(
            response,
            BoundaryWaveSolution::new_with_derivative(exterior, layers, derivatives),
        ))
    }
}

impl<C, D, M> DifferentiablePlaneWaveFieldBackend<C, D, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
{
    fn solve_plane_wave_internal_fields_spectral_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input::<C>();

        let workspace = self.accumulate_spectral_first_with::<RealAxis, _, _, _>(
            stack,
            &planar,
            variable,
            InternalFieldRequest::LayerBoundaries,
        )?;

        let primitive = variable.primitive();

        let mut left_admittance = IsotropicLayerQuantities::evaluate_first_spectral_real_axis(
            stack.left_exterior(),
            &planar,
            primitive,
        )
        .into_admittance()
        .into_inner();

        let mut right_admittance = IsotropicLayerQuantities::evaluate_first_spectral_real_axis(
            stack.right_exterior(),
            &planar,
            primitive,
        )
        .into_admittance()
        .into_inner();

        if let Some(rule) = variable.chain_rule(&planar) {
            left_admittance = left_admittance.chain_rule(&rule);
            right_admittance = right_admittance.chain_rule(&rule);
        }

        let (reflection, transmission) = workspace.total().clone().amplitude_jets(
            &left_admittance,
            &right_admittance,
            input.incident_side(),
        );

        let generic_fields = retained_boundary_waves(
            &workspace,
            input.incident_side(),
            &reflection,
            &transmission,
            &right_admittance,
            planar.vacuum_wavenumber(),
        );

        let (layers, first_layers) = first_order_fields_from_generic(generic_fields);

        let (exterior, exterior_first, reflection, transmission) =
            exterior_waves_from_first_jets(reflection, transmission, input.incident_side());

        let response = plane_wave_from_first_jet_amplitudes_spectral(
            reflection,
            transmission,
            &planar,
            stack,
            input.incident_side(),
            variable,
        );

        let derivatives =
            BoundaryWaveDerivatives::new(variable.into(), exterior_first, first_layers);

        Ok(PlaneWaveFieldResponse::new(
            response,
            BoundaryWaveSolution::new_with_derivative(exterior, layers, derivatives),
        ))
    }

    fn solve_plane_wave_internal_fields_spectral_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: SpectralDerivativeVariable,
    ) -> Result<PlaneWaveFieldResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input::<C>();

        let workspace = self.accumulate_spectral_second_with::<RealAxis, _, _, _>(
            stack,
            &planar,
            variable,
            InternalFieldRequest::LayerBoundaries,
        )?;

        let primitive = variable.primitive();

        let mut left_admittance = IsotropicLayerQuantities::evaluate_second_spectral_real_axis(
            stack.left_exterior(),
            &planar,
            primitive,
        )
        .into_admittance()
        .into_inner();

        let mut right_admittance = IsotropicLayerQuantities::evaluate_second_spectral_real_axis(
            stack.right_exterior(),
            &planar,
            primitive,
        )
        .into_admittance()
        .into_inner();

        if let Some(rule) = variable.chain_rule(&planar) {
            left_admittance = left_admittance.chain_rule(&rule);
            right_admittance = right_admittance.chain_rule(&rule);
        }

        let (reflection, transmission) = workspace.total().clone().amplitude_jets(
            &left_admittance,
            &right_admittance,
            input.incident_side(),
        );

        let generic_fields = retained_boundary_waves(
            &workspace,
            input.incident_side(),
            &reflection,
            &transmission,
            &right_admittance,
            planar.vacuum_wavenumber(),
        );

        let (layers, first_layers, second_layers) =
            second_order_fields_from_generic(generic_fields);

        let (exterior, exterior_first, exterior_second, reflection, transmission) =
            exterior_waves_from_second_jets(reflection, transmission, input.incident_side());

        let response = plane_wave_from_second_jet_amplitudes_spectral(
            reflection,
            transmission,
            &planar,
            stack,
            input.incident_side(),
            variable,
        );

        let derivatives =
            BoundaryWaveDerivatives::new(variable.into(), exterior_first, first_layers)
                .with_second(exterior_second, second_layers);

        Ok(PlaneWaveFieldResponse::new(
            response,
            BoundaryWaveSolution::new_with_derivative(exterior, layers, derivatives),
        ))
    }
}

/// Reconstruct the boundary waves in all retained finite layers.
///
/// The transfer workspace stores matrices in physical left-to-right order.
/// Reconstruction starts from the complete state at the right exterior
/// boundary and propagates through the retained matrices in reverse order.
///
/// The right-boundary state is:
///
/// ```text
/// left incidence:
///     forward  = t
///     backward = 0
///
/// right incidence:
///     forward  = r
///     backward = 1
/// ```
pub(crate) fn retained_boundary_waves<C, D, A>(
    workspace: &TransferWorkspace<A>,
    incident_side: IncidentSide,
    reflection: &A,
    transmission: &A,
    right_admittance: &A,
    source: &ArrayBase<OwnedRepr<C>, D>,
) -> Vec<LayerBoundaryWavesGeneric<A>>
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Clone,
{
    let right_state = right_exterior_transfer_state::<C, D, A>(
        incident_side,
        reflection,
        transmission,
        right_admittance,
        source,
    );

    workspace.reconstruct_layer_boundary_waves::<C, D>(right_state)
}

/// Construct the complete transfer state at the right exterior boundary.
///
/// Directional-wave states use:
///
/// ```text
/// forward:  [1, -ξ]
/// backward: [1, +ξ].
/// ```
///
/// Hence:
///
/// ```text
/// field = forward + backward
/// slope = ξ(backward - forward).
/// ```
fn right_exterior_transfer_state<C, D, A>(
    incident_side: IncidentSide,
    reflection: &A,
    transmission: &A,
    right_admittance: &A,
    source: &ArrayBase<OwnedRepr<C>, D>,
) -> TransferState<A>
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D> + Clone,
{
    let zero = A::constant_like(source, C::zero());

    let one = A::constant_like(source, C::one());

    let (forward, backward) = match incident_side {
        IncidentSide::Left => (transmission.clone(), zero),

        IncidentSide::Right => (reflection.clone(), one),
    };

    transfer_state_from_waves::<C, D, A>(forward, backward, right_admittance)
}

/// Convert directional amplitudes into the corresponding transfer state.
///
/// The supplied admittance is converted to the characteristic transfer slope
/// through [`boundary_slope`].
fn transfer_state_from_waves<C, D, A>(forward: A, backward: A, admittance: &A) -> TransferState<A>
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D>,
{
    let characteristic_slope =
        crate::backend::transfer2::plane_wave::boundary_slope::<C, D, A>(admittance);

    let field = forward.add(&backward);

    let slope = characteristic_slope.multiply(&backward.subtract(&forward));

    TransferState::new(field, slope)
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
    let reflection_for_response = reflection.clone();

    let transmission_for_response = transmission.clone();

    let (reflection, reflection_first, reflection_second) = reflection.into_parts();

    let (transmission, transmission_first, transmission_second) = transmission.into_parts();

    let one = reflection.mapv(|_| C::one());

    let zero = reflection.mapv(|_| C::zero());

    let derivative_zero = reflection.mapv(|_| C::zero());

    match incident_side {
        IncidentSide::Left => {
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
