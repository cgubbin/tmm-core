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
    ) -> Self
    where
        R: ComplexScalar,
        D: Dimension,
    {
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
    ) -> Self
    where
        R: ComplexScalar,
        D: Dimension,
    {
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
    ) -> Self
    where
        R: ComplexScalar,
        D: Dimension,
    {
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
        C: ComplexScalar<RealField = R>,
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
        C: ComplexScalar<RealField = R>,
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
        C: ComplexScalar<RealField = R>,
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
    C::RealField: Copy + ComplexScalar,
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
    C::RealField: Copy + ComplexScalar,
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
    C::RealField: Copy + ComplexScalar,
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
    C::RealField: Copy + ComplexScalar,
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
    C::RealField: Copy + ComplexScalar,
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
