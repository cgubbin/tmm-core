use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar, DerivativeVariable, IncidentSide, PlaneWaveInput, SpectralDerivativeVariable,
    Stack, StructuralDerivativeVariable,
    backend::{
        IsotropicFieldState, PlaneWaveFieldError, PlaneWaveFieldResponse,
        algebra::ScalarAlgebra,
        field::{
            boundary::{
                BoundaryWavesGeneric, generic_boundary_first, generic_boundary_second,
                generic_boundary_values,
            },
            observables::{
                context::{
                    AlgebraicPowerBalanceContext, power_balance_spectral_first_context,
                    power_balance_spectral_second_context, power_balance_structural_first_context,
                    power_balance_structural_second_context, power_balance_value_context,
                },
                sample::validate_generic_layer_count,
            },
        },
        jet::{ArrayJet, ArrayJetFirst},
    },
    material::{EvaluateDifferentiableMaterial, EvaluateMaterial},
};

#[derive(Clone, Debug)]
pub struct PlaneWavePowerBalance<R, D>
where
    D: Dimension,
{
    pub(crate) incident_flux: ArrayBase<OwnedRepr<R>, D>,
    pub(crate) reflected_flux: ArrayBase<OwnedRepr<R>, D>,
    pub(crate) transmitted_flux: ArrayBase<OwnedRepr<R>, D>,

    pub(crate) layer_absorptance: Vec<ArrayBase<OwnedRepr<R>, D>>,
    pub(crate) total_layer_absorptance: ArrayBase<OwnedRepr<R>, D>,

    pub(crate) balance_residual: ArrayBase<OwnedRepr<R>, D>,

    pub(crate) derivatives: Option<PlaneWavePowerBalanceDerivatives<R, D>>,
}

impl<R, D> PlaneWavePowerBalance<R, D>
where
    D: Dimension,
{
    pub(crate) fn from_values(
        balance: AlgebraicPlaneWavePowerBalance<ArrayBase<OwnedRepr<R>, D>>,
    ) -> Self {
        Self {
            incident_flux: balance.incident_flux,
            reflected_flux: balance.reflected_flux,
            transmitted_flux: balance.transmitted_flux,
            layer_absorptance: balance.layer_absorptance,
            total_layer_absorptance: balance.total_layer_absorptance,
            balance_residual: balance.balance_residual,
            derivatives: None,
        }
    }

    pub(crate) fn from_first_order(
        variable: DerivativeVariable,
        balance: AlgebraicPlaneWavePowerBalance<ArrayJetFirst<R, D>>,
    ) -> Self {
        let (incident_flux, incident_flux_first) = balance.incident_flux.into_parts();

        let (reflected_flux, reflected_flux_first) = balance.reflected_flux.into_parts();

        let (transmitted_flux, transmitted_flux_first) = balance.transmitted_flux.into_parts();

        let (total_layer_absorptance, total_layer_absorptance_first) =
            balance.total_layer_absorptance.into_parts();

        let (balance_residual, balance_residual_first) = balance.balance_residual.into_parts();

        let (layer_absorptance, layer_absorptance_first): (Vec<_>, Vec<_>) = balance
            .layer_absorptance
            .into_iter()
            .map(|each| each.into_parts())
            .unzip();

        Self {
            incident_flux,
            reflected_flux,
            transmitted_flux,
            layer_absorptance,
            total_layer_absorptance,
            balance_residual,

            derivatives: Some(PlaneWavePowerBalanceDerivatives {
                variable,

                first: PlaneWavePowerBalanceDerivative {
                    incident_flux: incident_flux_first,
                    reflected_flux: reflected_flux_first,
                    transmitted_flux: transmitted_flux_first,

                    layer_absorptance: layer_absorptance_first,
                    total_layer_absorptance: total_layer_absorptance_first,
                    balance_residual: balance_residual_first,
                },

                second: None,
            }),
        }
    }

    pub(crate) fn from_second_order(
        variable: DerivativeVariable,
        balance: AlgebraicPlaneWavePowerBalance<ArrayJet<R, D>>,
    ) -> Self {
        let (incident_flux, incident_flux_first, incident_flux_second) =
            balance.incident_flux.into_parts();

        let (reflected_flux, reflected_flux_first, reflected_flux_second) =
            balance.reflected_flux.into_parts();

        let (transmitted_flux, transmitted_flux_first, transmitted_flux_second) =
            balance.transmitted_flux.into_parts();

        let (
            total_layer_absorptance,
            total_layer_absorptance_first,
            total_layer_absorptance_second,
        ) = balance.total_layer_absorptance.into_parts();

        let (balance_residual, balance_residual_first, balance_residual_second) =
            balance.balance_residual.into_parts();

        let (mut layer_absorptance, mut layer_absorptance_first, mut layer_absorptance_second): (
            Vec<_>,
            Vec<_>,
            Vec<_>,
        ) = (vec![], vec![], vec![]);

        for each in balance.layer_absorptance {
            let (value, first, second) = each.into_parts();
            layer_absorptance.push(value);
            layer_absorptance_first.push(first);
            layer_absorptance_second.push(second);
        }

        Self {
            incident_flux,
            reflected_flux,
            transmitted_flux,
            layer_absorptance,
            total_layer_absorptance,
            balance_residual,

            derivatives: Some(PlaneWavePowerBalanceDerivatives {
                variable,

                first: PlaneWavePowerBalanceDerivative {
                    incident_flux: incident_flux_first,
                    reflected_flux: reflected_flux_first,
                    transmitted_flux: transmitted_flux_first,

                    layer_absorptance: layer_absorptance_first,
                    total_layer_absorptance: total_layer_absorptance_first,
                    balance_residual: balance_residual_first,
                },

                second: Some(PlaneWavePowerBalanceDerivative {
                    incident_flux: incident_flux_second,
                    reflected_flux: reflected_flux_second,
                    transmitted_flux: transmitted_flux_second,

                    layer_absorptance: layer_absorptance_second,
                    total_layer_absorptance: total_layer_absorptance_second,
                    balance_residual: balance_residual_second,
                }),
            }),
        }
    }

    /// Return the positive incident-flux magnitude.
    pub fn incident_flux(&self) -> &ArrayBase<OwnedRepr<R>, D> {
        &self.incident_flux
    }

    /// Return the positive reflected-flux magnitude.
    pub fn reflected_flux(&self) -> &ArrayBase<OwnedRepr<R>, D> {
        &self.reflected_flux
    }

    /// Return the positive transmitted-flux magnitude.
    pub fn transmitted_flux(&self) -> &ArrayBase<OwnedRepr<R>, D> {
        &self.transmitted_flux
    }

    /// Return absorptance of every finite layer in geometric left-to-right
    /// order.
    pub fn layer_absorptance(&self) -> &[ArrayBase<OwnedRepr<R>, D>] {
        &self.layer_absorptance
    }

    pub fn layer(&self, index: usize) -> Option<&ArrayBase<OwnedRepr<R>, D>> {
        self.layer_absorptance.get(index)
    }

    /// Return the sum of all finite-layer absorptances.
    pub fn total_layer_absorptance(&self) -> &ArrayBase<OwnedRepr<R>, D> {
        &self.total_layer_absorptance
    }

    /// Return:
    ///
    /// ```text
    /// 1 - R - T - Σ A_layer.
    /// ```
    pub fn balance_residual(&self) -> &ArrayBase<OwnedRepr<R>, D> {
        &self.balance_residual
    }
    pub fn derivatives(&self) -> Option<&PlaneWavePowerBalanceDerivatives<R, D>> {
        self.derivatives.as_ref()
    }
}

#[derive(Clone, Debug)]
pub struct PlaneWavePowerBalanceDerivatives<R, D>
where
    D: Dimension,
{
    pub(crate) variable: DerivativeVariable,

    pub(crate) first: PlaneWavePowerBalanceDerivative<R, D>,

    pub(crate) second: Option<PlaneWavePowerBalanceDerivative<R, D>>,
}

impl<R, D> PlaneWavePowerBalanceDerivatives<R, D>
where
    D: Dimension,
{
    pub fn variable(&self) -> DerivativeVariable {
        self.variable
    }

    pub fn first(&self) -> &PlaneWavePowerBalanceDerivative<R, D> {
        &self.first
    }

    pub fn second(&self) -> Option<&PlaneWavePowerBalanceDerivative<R, D>> {
        self.second.as_ref()
    }
}

#[derive(Clone, Debug)]
pub struct PlaneWavePowerBalanceDerivative<R, D>
where
    D: Dimension,
{
    pub(crate) incident_flux: ArrayBase<OwnedRepr<R>, D>,
    pub(crate) reflected_flux: ArrayBase<OwnedRepr<R>, D>,
    pub(crate) transmitted_flux: ArrayBase<OwnedRepr<R>, D>,

    pub(crate) layer_absorptance: Vec<ArrayBase<OwnedRepr<R>, D>>,
    pub(crate) total_layer_absorptance: ArrayBase<OwnedRepr<R>, D>,

    pub(crate) balance_residual: ArrayBase<OwnedRepr<R>, D>,
}

impl<R, D> PlaneWavePowerBalanceDerivative<R, D>
where
    D: Dimension,
{
    pub fn incident_flux(&self) -> &ArrayBase<OwnedRepr<R>, D> {
        &self.incident_flux
    }

    pub fn reflected_flux(&self) -> &ArrayBase<OwnedRepr<R>, D> {
        &self.reflected_flux
    }

    pub fn transmitted_flux(&self) -> &ArrayBase<OwnedRepr<R>, D> {
        &self.transmitted_flux
    }

    pub fn layer_absorptance(&self) -> &[ArrayBase<OwnedRepr<R>, D>] {
        &self.layer_absorptance
    }

    pub fn layer(&self, index: usize) -> Option<&ArrayBase<OwnedRepr<R>, D>> {
        self.layer_absorptance.get(index)
    }

    pub fn total_layer_absorptance(&self) -> &ArrayBase<OwnedRepr<R>, D> {
        &self.total_layer_absorptance
    }

    pub fn balance_residual(&self) -> &ArrayBase<OwnedRepr<R>, D> {
        &self.balance_residual
    }
}

#[derive(Clone, Debug)]
pub(super) struct AlgebraicPlaneWavePowerBalance<A> {
    pub(super) incident_flux: A,
    pub(super) reflected_flux: A,
    pub(super) transmitted_flux: A,
    pub(super) layer_absorptance: Vec<A>,
    pub(super) total_layer_absorptance: A,
    pub(super) balance_residual: A,
}

struct AlgebraicPowerResponse<A> {
    reflectance: A,
    transmittance: A,
}

impl<R, D> AlgebraicPowerResponse<ArrayBase<OwnedRepr<R>, D>>
where
    D: Dimension,
{
    fn from_values<C>(response: &PlaneWaveFieldResponse<C, D>) -> Self
    where
        C: ComplexField<RealField = R>,
        R: Clone,
    {
        Self {
            reflectance: response.response().reflectance().clone(),
            transmittance: response.response().transmittance().clone(),
        }
    }
}

impl<R, D> AlgebraicPowerResponse<ArrayJetFirst<R, D>>
where
    D: Dimension,
{
    fn from_first_order<C>(
        response: &PlaneWaveFieldResponse<C, D>,
    ) -> Result<Self, PlaneWaveFieldError<R>>
    where
        C: ComplexField<RealField = R>,
        R: Clone,
    {
        let derivatives = response
            .response()
            .first_derivatives()
            .ok_or(PlaneWaveFieldError::MissingPowerDerivatives)?;

        Ok(Self {
            reflectance: ArrayJetFirst::from_parts(
                response.response().reflectance().clone(),
                derivatives.reflectance().clone(),
            ),
            transmittance: ArrayJetFirst::from_parts(
                response.response().transmittance().clone(),
                derivatives.transmittance().clone(),
            ),
        })
    }
}

impl<R, D> AlgebraicPowerResponse<ArrayJet<R, D>>
where
    D: Dimension,
{
    fn from_second_order<C>(
        response: &PlaneWaveFieldResponse<C, D>,
    ) -> Result<Self, PlaneWaveFieldError<C::RealField>>
    where
        C: ComplexField<RealField = R>,
        R: Clone,
    {
        let first = response
            .response()
            .first_derivatives()
            .ok_or(PlaneWaveFieldError::MissingPowerDerivatives)?;

        let second = response
            .response()
            .second_derivatives()
            .ok_or(PlaneWaveFieldError::MissingPowerDerivatives)?;

        Ok(Self {
            reflectance: ArrayJet::from_parts(
                response.response().reflectance().clone(),
                first.reflectance().clone(),
                second.reflectance().clone(),
            ),
            transmittance: ArrayJet::from_parts(
                response.response().transmittance().clone(),
                first.transmittance().clone(),
                second.transmittance().clone(),
            ),
        })
    }
}

pub(crate) fn plane_wave_power_balance_values<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    response: &PlaneWaveFieldResponse<C, D>,
) -> Result<PlaneWavePowerBalance<C::RealField, D>, PlaneWaveFieldError<C::RealField>>
where
    M: EvaluateMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy + ComplexField,
    D: Dimension,
{
    let context = power_balance_value_context(stack, input);
    let waves = generic_boundary_values(response.boundary_waves().values());

    let power_response = AlgebraicPowerResponse::from_values(response);

    let balance = plane_wave_power_balance_algebraic(&context, &waves, &power_response)?;

    Ok(PlaneWavePowerBalance::from_values(balance))
}

pub(super) fn plane_wave_power_balance_structural_first<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    response: &PlaneWaveFieldResponse<C, D>,
) -> Result<PlaneWavePowerBalance<C::RealField, D>, PlaneWaveFieldError<C::RealField>>
where
    M: EvaluateMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy + ComplexField,
    D: Dimension,
{
    let differentiated = response
        .boundary_waves()
        .structural()
        .ok_or(PlaneWaveFieldError::ExpectedStructuralDerivatives)?;

    let variable: StructuralDerivativeVariable = differentiated
        .variable()
        .try_into()
        .map_err(|_| PlaneWaveFieldError::ExpectedStructuralDerivatives)?;

    let context = power_balance_structural_first_context(stack, input, variable);
    let waves = generic_boundary_first(response.boundary_waves().values(), differentiated);

    let power_response = AlgebraicPowerResponse::from_first_order(response)?;

    let balance = plane_wave_power_balance_algebraic(&context, &waves, &power_response)?;

    Ok(PlaneWavePowerBalance::from_first_order(
        differentiated.variable(),
        balance,
    ))
}

pub(super) fn plane_wave_power_balance_structural_second<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    response: &PlaneWaveFieldResponse<C, D>,
) -> Result<PlaneWavePowerBalance<C::RealField, D>, PlaneWaveFieldError<C::RealField>>
where
    M: EvaluateMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy + ComplexField,
    D: Dimension,
{
    let differentiated = response
        .boundary_waves()
        .structural()
        .ok_or(PlaneWaveFieldError::ExpectedStructuralDerivatives)?;

    let variable: StructuralDerivativeVariable = differentiated
        .variable()
        .try_into()
        .map_err(|_| PlaneWaveFieldError::ExpectedStructuralDerivatives)?;

    let context = power_balance_structural_second_context(stack, input, variable);

    let waves = generic_boundary_second(response.boundary_waves().values(), differentiated)
        .ok_or(PlaneWaveFieldError::MissingSecondDerivatives)?;

    let power_response = AlgebraicPowerResponse::from_second_order(response)?;

    let balance = plane_wave_power_balance_algebraic(&context, &waves, &power_response)?;

    Ok(PlaneWavePowerBalance::from_second_order(
        differentiated.variable(),
        balance,
    ))
}

pub(super) fn plane_wave_power_balance_spectral_first<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    response: &PlaneWaveFieldResponse<C, D>,
) -> Result<PlaneWavePowerBalance<C::RealField, D>, PlaneWaveFieldError<C::RealField>>
where
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy + ComplexField,
    D: Dimension,
{
    let differentiated = response
        .boundary_waves()
        .spectral()
        .ok_or(PlaneWaveFieldError::ExpectedSpectralDerivatives)?;

    let variable: SpectralDerivativeVariable = differentiated
        .variable()
        .try_into()
        .map_err(|_| PlaneWaveFieldError::ExpectedSpectralDerivatives)?;

    let context = power_balance_spectral_first_context(stack, input, variable);
    let waves = generic_boundary_first(response.boundary_waves().values(), differentiated);

    let power_response = AlgebraicPowerResponse::from_first_order(response)?;

    let balance = plane_wave_power_balance_algebraic(&context, &waves, &power_response)?;

    Ok(PlaneWavePowerBalance::from_first_order(
        differentiated.variable(),
        balance,
    ))
}

pub(super) fn plane_wave_power_balance_spectral_second<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    response: &PlaneWaveFieldResponse<C, D>,
) -> Result<PlaneWavePowerBalance<C::RealField, D>, PlaneWaveFieldError<C::RealField>>
where
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy + ComplexField,
    D: Dimension,
{
    let differentiated = response
        .boundary_waves()
        .spectral()
        .ok_or(PlaneWaveFieldError::ExpectedSpectralDerivatives)?;

    let variable: SpectralDerivativeVariable = differentiated
        .variable()
        .try_into()
        .map_err(|_| PlaneWaveFieldError::ExpectedSpectralDerivatives)?;

    let context = power_balance_spectral_second_context(stack, input, variable);

    let waves = generic_boundary_second(response.boundary_waves().values(), differentiated)
        .ok_or(PlaneWaveFieldError::MissingSecondDerivatives)?;

    let power_response = AlgebraicPowerResponse::from_second_order(response)?;

    let balance = plane_wave_power_balance_algebraic(&context, &waves, &power_response)?;

    Ok(PlaneWavePowerBalance::from_second_order(
        differentiated.variable(),
        balance,
    ))
}

fn plane_wave_power_balance_algebraic<C, D, A>(
    context: &AlgebraicPowerBalanceContext<A>,
    waves: &BoundaryWavesGeneric<A>,
    power_response: &AlgebraicPowerResponse<A::RealField>,
) -> Result<AlgebraicPlaneWavePowerBalance<A::RealField>, PlaneWaveFieldError<C::RealField>>
where
    A: ScalarAlgebra<C, D> + Clone,
    A::RealField: ScalarAlgebra<C::RealField, D> + Clone,
    C: ComplexScalar,
    C::RealField: ComplexField,
    D: Dimension,
{
    validate_generic_layer_count(context.layers.len(), waves)?;

    let left_admittance_inner = context.left_admittance.clone().into_inner();
    let right_admittance_inner = context.right_admittance.clone().into_inner();
    let incident_flux = incident_flux_magnitude(
        context.incident_side,
        &left_admittance_inner,
        &right_admittance_inner,
    )?;

    let reflected_flux = power_response.reflectance.clone().multiply(&incident_flux);

    let transmitted_flux = power_response
        .transmittance
        .clone()
        .multiply(&incident_flux);

    let mut layer_absorptance = Vec::with_capacity(context.layers.len());

    for (index, admittance) in context.layers.iter().enumerate() {
        let boundary = waves
            .layer(index)
            .ok_or(PlaneWaveFieldError::LayerOutOfBounds {
                requested: index,
                layer_count: waves.len(),
            })?;

        let left_state = IsotropicFieldState::from_waves(boundary.left(), admittance);

        let right_state = IsotropicFieldState::from_waves(boundary.right(), admittance);

        let left_flux = left_state.normal_flux();
        let right_flux = right_state.normal_flux();

        /*
         * This expression is valid for both incidence sides because flux is
         * signed geometrically:
         *
         * left incidence:  P_left > P_right
         * right incidence: P_left is less negative than P_right.
         */
        let absorption = (left_flux.subtract(&right_flux)).divide(&incident_flux);

        layer_absorptance.push(absorption);
    }

    let mut total_layer_absorptance = incident_flux.zero_like();

    for absorption in &layer_absorptance {
        total_layer_absorptance = total_layer_absorptance.add(absorption);
    }

    let one = <A::RealField as ScalarAlgebra<C::RealField, D>>::constant_like(
        incident_flux.value(),
        C::one().real(),
    );

    let balance_residual = one
        .subtract(&power_response.reflectance)
        .subtract(&power_response.transmittance)
        .subtract(&total_layer_absorptance);

    Ok(AlgebraicPlaneWavePowerBalance {
        incident_flux,
        reflected_flux,
        transmitted_flux,
        layer_absorptance,
        total_layer_absorptance,
        balance_residual,
    })
}

fn incident_flux_magnitude<C, D, A>(
    side: IncidentSide,
    left_admittance: &A,
    right_admittance: &A,
) -> Result<A::RealField, PlaneWaveFieldError<C::RealField>>
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D>,
{
    let half = C::one() / (C::one() + C::one());

    let incident = match side {
        IncidentSide::Left => left_admittance,
        IncidentSide::Right => right_admittance,
    }
    .scale(half)
    .real_part();

    // TODO: Reinstate checks
    // if incident
    //     .iter()
    //     .any(|value| !value.is_finite() || *value <= C::zero().real())
    // {
    //     return Err(PlaneWaveFieldError::InvalidIncidentFlux);
    // }

    Ok(incident)
}

// #[cfg(test)]
// mod tests {

//     use super::*;
//     use crate::{
//         IncidentSide,
//         backend::{
//             field::{BidirectionalWavesGeneric, LayerBoundaryWavesGeneric},
//             isotropic::IsotropicLayerAdmittance,
//         },
//     };

//     use approx::assert_relative_eq;
//     use ndarray::{Array0, Ix0};
//     use num_complex::Complex64;

//     type C = Complex64;
//     type D = Ix0;
//     type A = Array0<C>;

//     const TOLERANCE: f64 = 1.0e-12;

//     fn c(re: f64, im: f64) -> C {
//         C::new(re, im)
//     }

//     fn scalar(re: f64, im: f64) -> A {
//         Array0::from_elem((), c(re, im))
//     }

//     fn real_scalar(value: f64) -> Array0<f64> {
//         Array0::from_elem((), value)
//     }

//     fn assert_complex_close(actual: &A, expected: C) {
//         assert_relative_eq!(actual[()].re, expected.re, epsilon = TOLERANCE);

//         assert_relative_eq!(actual[()].im, expected.im, epsilon = TOLERANCE);
//     }

//     fn assert_real_close(actual: &Array0<f64>, expected: f64) {
//         assert_relative_eq!(actual[()], expected, epsilon = TOLERANCE);
//     }

//     fn power_context(
//         incident_side: IncidentSide,
//         exterior_admittance: f64,
//         layer_admittances: &[f64],
//     ) -> AlgebraicPowerBalanceContext<A> {
//         AlgebraicPowerBalanceContext {
//             incident_side,

//             left_admittance: IsotropicLayerAdmittance::new(scalar(exterior_admittance, 0.0)),

//             right_admittance: IsotropicLayerAdmittance::new(scalar(exterior_admittance, 0.0)),

//             layers: layer_admittances
//                 .iter()
//                 .map(|value| IsotropicLayerAdmittance::new(scalar(*value, 0.0)))
//                 .collect(),
//         }
//     }

//     #[test]
//     fn empty_lossless_stack_has_zero_absorption_and_residual() {
//         let context = power_context(IncidentSide::Left, 2.0, &[]);

//         let waves = boundary_waves_without_layers(
//             BidirectionalWavesGeneric::new(scalar(1.0, 0.0), scalar(0.0, 0.0)),
//             BidirectionalWavesGeneric::new(scalar(1.0, 0.0), scalar(0.0, 0.0)),
//         );

//         let reflectance = Array0::from_elem((), 0.0);
//         let transmittance = Array0::from_elem((), 1.0);

//         let response = AlgebraicPowerResponse {
//             reflectance,
//             transmittance,
//         };

//         let result = plane_wave_power_balance_algebraic(&context, &waves, &response).unwrap();

//         assert_real_close(&result.incident_flux, 1.0);
//         assert_real_close(&result.reflected_flux, 0.0);
//         assert_real_close(&result.transmitted_flux, 1.0);
//         assert!(result.layer_absorptance.is_empty());
//         assert_real_close(&result.total_layer_absorptance, 0.0);
//         assert_real_close(&result.balance_residual, 0.0);
//     }

//     #[test]
//     fn reflected_and_transmitted_flux_are_scaled_by_incident_flux() {
//         let context = power_context(IncidentSide::Left, 4.0, &[]);

//         let waves = boundary_waves_without_layers(
//             BidirectionalWavesGeneric::new(scalar(1.0, 0.0), scalar(0.0, 0.0)),
//             BidirectionalWavesGeneric::new(scalar(1.0, 0.0), scalar(0.0, 0.0)),
//         );

//         let result = plane_wave_power_balance_algebraic(
//             &context,
//             &waves,
//             &AlgebraicPowerResponse {
//                 reflectance: Array0::from_elem((), 0.25),
//                 transmittance: Array0::from_elem((), 0.50),
//             },
//         )
//         .unwrap();

//         assert_real_close(&result.incident_flux, 2.0);
//         assert_real_close(&result.reflected_flux, 0.5);
//         assert_real_close(&result.transmitted_flux, 1.0);

//         assert_real_close(&result.balance_residual, 0.25);
//     }

//     #[test]
//     fn lossless_layer_with_equal_boundary_flux_has_zero_absorptance() {
//         let context = power_context(IncidentSide::Left, 2.0, &[2.0]);

//         let layer = LayerBoundaryWavesGeneric::new(
//             BidirectionalWavesGeneric::new(scalar(1.0, 0.0), scalar(0.0, 0.0)),
//             BidirectionalWavesGeneric::new(scalar(1.0, 0.0), scalar(0.0, 0.0)),
//         );

//         let waves = boundary_waves_with_layers(vec![layer]);

//         let result = plane_wave_power_balance_algebraic(
//             &context,
//             &waves,
//             &AlgebraicPowerResponse {
//                 reflectance: Array0::from_elem((), 0.0),
//                 transmittance: Array0::from_elem((), 1.0),
//             },
//         )
//         .unwrap();

//         assert_real_close(&result.layer_absorptance[0], 0.0);
//         assert_real_close(&result.total_layer_absorptance, 0.0);
//         assert_real_close(&result.balance_residual, 0.0);
//     }

//     #[test]
//     fn layer_absorptance_is_normalised_flux_drop() {
//         let context = power_context(IncidentSide::Left, 2.0, &[2.0]);

//         let layer = LayerBoundaryWavesGeneric::new(
//             BidirectionalWavesGeneric::new(scalar(1.0, 0.0), scalar(0.0, 0.0)),
//             BidirectionalWavesGeneric::new(scalar(0.6_f64.sqrt(), 0.0), scalar(0.0, 0.0)),
//         );

//         let waves = boundary_waves_with_layers(vec![layer]);

//         let result = plane_wave_power_balance_algebraic(
//             &context,
//             &waves,
//             &AlgebraicPowerResponse {
//                 reflectance: Array0::from_elem((), 0.1),
//                 transmittance: Array0::from_elem((), 0.5),
//             },
//         )
//         .unwrap();

//         assert_real_close(&result.layer_absorptance[0], 0.4);
//         assert_real_close(&result.total_layer_absorptance, 0.4);
//         assert_real_close(&result.balance_residual, 0.0);
//     }

//     #[test]
//     fn total_layer_absorptance_is_sum_of_layer_absorptances() {
//         let context = power_context(IncidentSide::Left, 2.0, &[2.0, 2.0]);

//         let first = layer_with_forward_fluxes(1.0, 0.8);
//         let second = layer_with_forward_fluxes(0.8, 0.5);

//         let waves = boundary_waves_with_layers(vec![first, second]);

//         let result = plane_wave_power_balance_algebraic(
//             &context,
//             &waves,
//             &AlgebraicPowerResponse {
//                 reflectance: Array0::from_elem((), 0.1),
//                 transmittance: Array0::from_elem((), 0.4),
//             },
//         )
//         .unwrap();

//         assert_real_close(&result.layer_absorptance[0], 0.2);
//         assert_real_close(&result.layer_absorptance[1], 0.3);
//         assert_real_close(&result.total_layer_absorptance, 0.5);
//         assert_real_close(&result.balance_residual, 0.0);
//     }

//     #[test]
//     fn right_incidence_uses_signed_flux_consistently() {
//         let context = power_context(IncidentSide::Right, 2.0, &[2.0]);

//         /*
//          * Right boundary incident flux: -1.0.
//          * Left boundary exiting flux:  -0.7.
//          *
//          * left_flux - right_flux = -0.7 - (-1.0) = 0.3.
//          */
//         let layer = LayerBoundaryWavesGeneric::new(
//             BidirectionalWavesGeneric::new(scalar(0.0, 0.0), scalar(0.7_f64.sqrt(), 0.0)),
//             BidirectionalWavesGeneric::new(scalar(0.0, 0.0), scalar(1.0, 0.0)),
//         );

//         let waves = boundary_waves_with_layers(vec![layer]);

//         let result = plane_wave_power_balance_algebraic(
//             &context,
//             &waves,
//             &AlgebraicPowerResponse {
//                 reflectance: Array0::from_elem((), 0.1),
//                 transmittance: Array0::from_elem((), 0.6),
//             },
//         )
//         .unwrap();

//         assert_real_close(&result.incident_flux, 1.0);
//         assert_real_close(&result.layer_absorptance[0], 0.3);
//         assert_real_close(&result.balance_residual, 0.0);
//     }

//     #[test]
//     fn power_balance_rejects_layer_count_mismatch() {
//         let context = power_context(IncidentSide::Left, 2.0, &[2.0, 2.0]);

//         let waves = boundary_waves_with_layers(vec![layer_with_forward_fluxes(1.0, 0.8)]);

//         let error = plane_wave_power_balance_algebraic(
//             &context,
//             &waves,
//             &AlgebraicPowerResponse {
//                 reflectance: Array0::from_elem((), 0.0),
//                 transmittance: Array0::from_elem((), 1.0),
//             },
//         )
//         .unwrap_err();

//         assert!(matches!(
//             error,
//             PlaneWaveFieldError::LayerCountMismatch {
//                 expected: 2,
//                 actual: 1,
//             }
//         ));
//     }

//     #[test]
//     fn first_order_power_balance_is_split_correctly() {
//         let balance = AlgebraicPlaneWavePowerBalance {
//             incident_flux: real_first_jet(1.0, 11.0),
//             reflected_flux: real_first_jet(2.0, 12.0),
//             transmitted_flux: real_first_jet(3.0, 13.0),

//             layer_absorptance: vec![real_first_jet(4.0, 14.0), real_first_jet(5.0, 15.0)],

//             total_layer_absorptance: real_first_jet(6.0, 16.0),
//             balance_residual: real_first_jet(7.0, 17.0),
//         };

//         let result =
//             PlaneWavePowerBalance::from_first_order(DerivativeVariable::Thickness(0), balance);

//         assert_real_close(result.incident_flux(), 1.0);
//         assert_real_close(result.reflected_flux(), 2.0);
//         assert_real_close(result.transmitted_flux(), 3.0);

//         assert_real_close(&result.layer_absorptance()[0], 4.0);
//         assert_real_close(&result.layer_absorptance()[1], 5.0);

//         assert_real_close(result.total_layer_absorptance(), 6.0);
//         assert_real_close(result.balance_residual(), 7.0);

//         let derivatives = result.derivatives().unwrap();

//         assert_eq!(derivatives.variable(), DerivativeVariable::Thickness(0),);

//         let first = derivatives.first();

//         assert_real_close(first.incident_flux(), 11.0);
//         assert_real_close(first.reflected_flux(), 12.0);
//         assert_real_close(first.transmitted_flux(), 13.0);

//         assert_real_close(&first.layer_absorptance()[0], 14.0);
//         assert_real_close(&first.layer_absorptance()[1], 15.0);

//         assert_real_close(first.total_layer_absorptance(), 16.0);
//         assert_real_close(first.balance_residual(), 17.0);

//         assert!(derivatives.second().is_none());
//     }

//     fn real_first_jet(value: f64, first: f64) -> ArrayJetFirst<f64, Ix0> {
//         ArrayJetFirst::from_parts(Array0::from_elem((), value), Array0::from_elem((), first))
//     }

//     #[test]
//     fn second_order_power_balance_is_split_correctly() {
//         let balance = AlgebraicPlaneWavePowerBalance {
//             incident_flux: real_second_jet(1.0, 11.0, 21.0),
//             reflected_flux: real_second_jet(2.0, 12.0, 22.0),
//             transmitted_flux: real_second_jet(3.0, 13.0, 23.0),

//             layer_absorptance: vec![
//                 real_second_jet(4.0, 14.0, 24.0),
//                 real_second_jet(5.0, 15.0, 25.0),
//             ],

//             total_layer_absorptance: real_second_jet(6.0, 16.0, 26.0),

//             balance_residual: real_second_jet(7.0, 17.0, 27.0),
//         };

//         let result =
//             PlaneWavePowerBalance::from_second_order(DerivativeVariable::Thickness(0), balance);

//         let derivatives = result.derivatives().unwrap();
//         let first = derivatives.first();
//         let second = derivatives.second().unwrap();

//         assert_real_close(first.incident_flux(), 11.0);
//         assert_real_close(second.incident_flux(), 21.0);

//         assert_real_close(first.reflected_flux(), 12.0);
//         assert_real_close(second.reflected_flux(), 22.0);

//         assert_real_close(first.transmitted_flux(), 13.0);
//         assert_real_close(second.transmitted_flux(), 23.0);

//         assert_real_close(&first.layer_absorptance()[0], 14.0);
//         assert_real_close(&second.layer_absorptance()[0], 24.0);

//         assert_real_close(first.total_layer_absorptance(), 16.0);
//         assert_real_close(second.total_layer_absorptance(), 26.0);

//         assert_real_close(first.balance_residual(), 17.0);
//         assert_real_close(second.balance_residual(), 27.0);
//     }

//     fn real_second_jet(value: f64, first: f64, second: f64) -> ArrayJet<f64, Ix0> {
//         ArrayJet::from_parts(
//             Array0::from_elem((), value),
//             Array0::from_elem((), first),
//             Array0::from_elem((), second),
//         )
//     }

//     #[test]
//     fn first_order_balance_derivative_obeys_conservation_expression() {
//         type J = ArrayJetFirst<C, Ix0>;

//         let context = power_balance_spectral_first_context(
//             IncidentSide::Left,
//             /* Y = */ 2.0,
//             /* Y' = */ 0.0,
//             &[],
//         );

//         let waves = first_order_boundary_waves_without_layers();

//         let reflectance = real_first_jet(0.2, 0.03);
//         let transmittance = real_first_jet(0.7, -0.01);

//         let result = plane_wave_power_balance_algebraic::<C, D, J>(
//             &context,
//             &waves,
//             &reflectance,
//             &transmittance,
//         )
//         .unwrap();

//         let (residual, residual_first) = result.balance_residual.into_parts();

//         assert_real_close(&residual, 0.1);
//         assert_real_close(&residual_first, -0.02);
//     }

//     #[test]
//     fn second_order_balance_derivative_obeys_conservation_expression() {
//         type J = ArrayJet<C, Ix0>;

//         let context = second_order_power_context(IncidentSide::Left, 2.0, 0.0, 0.0, &[]);

//         let waves = second_order_boundary_waves_without_layers();

//         let reflectance = real_second_jet(0.2, 0.03, 0.04);
//         let transmittance = real_second_jet(0.7, -0.01, -0.02);

//         let result = plane_wave_power_balance_algebraic::<C, D, J>(
//             &context,
//             &waves,
//             &reflectance,
//             &transmittance,
//         )
//         .unwrap();

//         let (residual, first, second) = result.balance_residual.into_parts();

//         assert_real_close(&residual, 0.1);
//         assert_real_close(&first, -0.02);
//         assert_real_close(&second, -0.02);
//     }
// }
