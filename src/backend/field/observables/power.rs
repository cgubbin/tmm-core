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
    if !context.left_admittance.all_finite() || !context.right_admittance.all_finite() {
        return Err(PlaneWaveFieldError::NonFiniteFieldQuantity);
    }

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
    A::RealField: ScalarAlgebra<C::RealField, D>,
{
    let half = C::one() / (C::one() + C::one());

    let incident = match side {
        IncidentSide::Left => left_admittance,
        IncidentSide::Right => right_admittance,
    }
    .scale(half)
    .real_part();

    if incident
        .value()
        .iter()
        .any(|value| !value.is_finite() || *value <= C::zero().real())
    {
        return Err(PlaneWaveFieldError::InvalidIncidentFlux);
    }

    Ok(incident)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array0, Ix0};
    use num_complex::Complex64;

    use crate::{
        DerivativeVariable, IncidentSide,
        backend::{
            field::{
                IsotropicFieldState,
                boundary::{
                    BidirectionalWavesGeneric, BoundaryWavesGeneric, ExteriorBoundaryWavesGeneric,
                    LayerBoundaryWavesGeneric,
                },
            },
            isotropic::IsotropicLayerAdmittance,
            jet::{ArrayJet, ArrayJetFirst},
        },
    };

    use super::*;

    type C = Complex64;
    type D = Ix0;

    type ComplexArray = Array0<C>;
    type RealArray = Array0<f64>;

    type FirstComplex = ArrayJetFirst<C, D>;
    type FirstReal = ArrayJetFirst<f64, D>;

    type SecondComplex = ArrayJet<C, D>;
    type SecondReal = ArrayJet<f64, D>;

    const TOLERANCE: f64 = 1.0e-10;

    fn c(re: f64, im: f64) -> C {
        C::new(re, im)
    }

    fn scalar(re: f64, im: f64) -> ComplexArray {
        Array0::from_elem((), c(re, im))
    }

    fn real_scalar(value: f64) -> RealArray {
        Array0::from_elem((), value)
    }

    fn assert_complex_close(actual: &ComplexArray, expected: C) {
        assert_relative_eq!(
            actual[()].re,
            expected.re,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            actual[()].im,
            expected.im,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    fn assert_real_close(actual: &RealArray, expected: f64) {
        assert_relative_eq!(
            actual[()],
            expected,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    fn bidirectional<A>(forward: A, backward: A) -> BidirectionalWavesGeneric<A> {
        BidirectionalWavesGeneric::new(forward, backward)
    }

    fn zero_bidirectional<A>(zero: &A) -> BidirectionalWavesGeneric<A>
    where
        A: Clone,
    {
        bidirectional(zero.clone(), zero.clone())
    }

    fn zero_exterior<A>(zero: &A) -> ExteriorBoundaryWavesGeneric<A>
    where
        A: Clone,
    {
        ExteriorBoundaryWavesGeneric::new(zero_bidirectional(zero), zero_bidirectional(zero))
    }

    fn boundary_waves_without_layers<A>(zero: A) -> BoundaryWavesGeneric<A>
    where
        A: Clone,
    {
        BoundaryWavesGeneric::new(zero_exterior(&zero), Vec::new())
    }

    fn boundary_waves_with_layers<A>(
        zero: A,
        layers: Vec<LayerBoundaryWavesGeneric<A>>,
    ) -> BoundaryWavesGeneric<A>
    where
        A: Clone,
    {
        BoundaryWavesGeneric::new(zero_exterior(&zero), layers)
    }

    fn boundary_waves_with_exterior<A>(
        left: BidirectionalWavesGeneric<A>,
        right: BidirectionalWavesGeneric<A>,
        layers: Vec<LayerBoundaryWavesGeneric<A>>,
    ) -> BoundaryWavesGeneric<A> {
        BoundaryWavesGeneric::new(ExteriorBoundaryWavesGeneric::new(left, right), layers)
    }

    fn layer_waves<A>(
        left_forward: A,
        left_backward: A,
        right_forward: A,
        right_backward: A,
    ) -> LayerBoundaryWavesGeneric<A> {
        LayerBoundaryWavesGeneric::new(
            BidirectionalWavesGeneric::new(left_forward, left_backward),
            BidirectionalWavesGeneric::new(right_forward, right_backward),
        )
    }

    fn layer_with_forward_amplitudes(
        left_amplitude: f64,
        right_amplitude: f64,
    ) -> LayerBoundaryWavesGeneric<ComplexArray> {
        layer_waves(
            scalar(left_amplitude, 0.0),
            scalar(0.0, 0.0),
            scalar(right_amplitude, 0.0),
            scalar(0.0, 0.0),
        )
    }

    fn layer_with_forward_fluxes(
        left_flux: f64,
        right_flux: f64,
    ) -> LayerBoundaryWavesGeneric<ComplexArray> {
        assert!(left_flux >= 0.0);
        assert!(right_flux >= 0.0);

        layer_with_forward_amplitudes(left_flux.sqrt(), right_flux.sqrt())
    }

    fn forward_amplitude_for_flux(flux: f64, admittance: f64) -> f64 {
        assert!(flux >= 0.0);
        assert!(admittance > 0.0);

        (2.0 * flux / admittance).sqrt()
    }

    fn layer_with_forward_fluxes_for_admittance(
        left_flux: f64,
        right_flux: f64,
        admittance: f64,
    ) -> LayerBoundaryWavesGeneric<ComplexArray> {
        layer_with_forward_amplitudes(
            forward_amplitude_for_flux(left_flux, admittance),
            forward_amplitude_for_flux(right_flux, admittance),
        )
    }

    fn layer_with_backward_fluxes(
        left_flux_magnitude: f64,
        right_flux_magnitude: f64,
    ) -> LayerBoundaryWavesGeneric<ComplexArray> {
        assert!(left_flux_magnitude >= 0.0);
        assert!(right_flux_magnitude >= 0.0);

        layer_waves(
            scalar(0.0, 0.0),
            scalar(left_flux_magnitude.sqrt(), 0.0),
            scalar(0.0, 0.0),
            scalar(right_flux_magnitude.sqrt(), 0.0),
        )
    }

    fn power_context<A>(
        incident_side: IncidentSide,
        left_admittance: A,
        right_admittance: A,
        layer_admittances: Vec<A>,
    ) -> AlgebraicPowerBalanceContext<A> {
        AlgebraicPowerBalanceContext {
            incident_side,

            left_admittance: IsotropicLayerAdmittance::new(left_admittance),

            right_admittance: IsotropicLayerAdmittance::new(right_admittance),

            layers: layer_admittances
                .into_iter()
                .map(IsotropicLayerAdmittance::new)
                .collect(),
        }
    }

    fn value_power_context(
        incident_side: IncidentSide,
        left_admittance: f64,
        right_admittance: f64,
        layer_admittances: &[f64],
    ) -> AlgebraicPowerBalanceContext<ComplexArray> {
        power_context(
            incident_side,
            scalar(left_admittance, 0.0),
            scalar(right_admittance, 0.0),
            layer_admittances
                .iter()
                .map(|value| scalar(*value, 0.0))
                .collect(),
        )
    }

    fn symmetric_value_power_context(
        incident_side: IncidentSide,
        exterior_admittance: f64,
        layer_admittances: &[f64],
    ) -> AlgebraicPowerBalanceContext<ComplexArray> {
        value_power_context(
            incident_side,
            exterior_admittance,
            exterior_admittance,
            layer_admittances,
        )
    }

    fn complex_first_jet(value: C, first: C) -> FirstComplex {
        ArrayJetFirst::from_parts(Array0::from_elem((), value), Array0::from_elem((), first))
    }

    fn real_first_jet(value: f64, first: f64) -> FirstReal {
        ArrayJetFirst::from_parts(real_scalar(value), real_scalar(first))
    }

    fn zero_complex_first_jet() -> FirstComplex {
        complex_first_jet(c(0.0, 0.0), c(0.0, 0.0))
    }

    fn complex_second_jet(value: C, first: C, second: C) -> SecondComplex {
        ArrayJet::from_parts(
            Array0::from_elem((), value),
            Array0::from_elem((), first),
            Array0::from_elem((), second),
        )
    }

    fn real_second_jet(value: f64, first: f64, second: f64) -> SecondReal {
        ArrayJet::from_parts(real_scalar(value), real_scalar(first), real_scalar(second))
    }

    fn zero_complex_second_jet() -> SecondComplex {
        complex_second_jet(c(0.0, 0.0), c(0.0, 0.0), c(0.0, 0.0))
    }

    fn first_order_power_context(
        incident_side: IncidentSide,
        left_admittance: (C, C),
        right_admittance: (C, C),
        layer_admittances: &[(C, C)],
    ) -> AlgebraicPowerBalanceContext<FirstComplex> {
        power_context(
            incident_side,
            complex_first_jet(left_admittance.0, left_admittance.1),
            complex_first_jet(right_admittance.0, right_admittance.1),
            layer_admittances
                .iter()
                .map(|(value, first)| complex_first_jet(*value, *first))
                .collect(),
        )
    }

    fn first_order_boundary_waves_without_layers() -> BoundaryWavesGeneric<FirstComplex> {
        boundary_waves_without_layers(zero_complex_first_jet())
    }

    fn first_order_layer_waves(
        left_forward: (C, C),
        left_backward: (C, C),
        right_forward: (C, C),
        right_backward: (C, C),
    ) -> LayerBoundaryWavesGeneric<FirstComplex> {
        layer_waves(
            complex_first_jet(left_forward.0, left_forward.1),
            complex_first_jet(left_backward.0, left_backward.1),
            complex_first_jet(right_forward.0, right_forward.1),
            complex_first_jet(right_backward.0, right_backward.1),
        )
    }

    fn second_order_power_context(
        incident_side: IncidentSide,
        left_admittance: (C, C, C),
        right_admittance: (C, C, C),
        layer_admittances: &[(C, C, C)],
    ) -> AlgebraicPowerBalanceContext<SecondComplex> {
        power_context(
            incident_side,
            complex_second_jet(left_admittance.0, left_admittance.1, left_admittance.2),
            complex_second_jet(right_admittance.0, right_admittance.1, right_admittance.2),
            layer_admittances
                .iter()
                .map(|(value, first, second)| complex_second_jet(*value, *first, *second))
                .collect(),
        )
    }

    fn second_order_boundary_waves_without_layers() -> BoundaryWavesGeneric<SecondComplex> {
        boundary_waves_without_layers(zero_complex_second_jet())
    }

    fn second_order_layer_waves(
        left_forward: (C, C, C),
        left_backward: (C, C, C),
        right_forward: (C, C, C),
        right_backward: (C, C, C),
    ) -> LayerBoundaryWavesGeneric<SecondComplex> {
        layer_waves(
            complex_second_jet(left_forward.0, left_forward.1, left_forward.2),
            complex_second_jet(left_backward.0, left_backward.1, left_backward.2),
            complex_second_jet(right_forward.0, right_forward.1, right_forward.2),
            complex_second_jet(right_backward.0, right_backward.1, right_backward.2),
        )
    }

    fn power_response<A>(reflectance: A, transmittance: A) -> AlgebraicPowerResponse<A> {
        AlgebraicPowerResponse {
            reflectance,
            transmittance,
        }
    }

    #[test]
    fn empty_lossless_stack_has_zero_absorption_and_residual() {
        let context = symmetric_value_power_context(IncidentSide::Left, 2.0, &[]);

        let waves = boundary_waves_without_layers(scalar(0.0, 0.0));

        let pr = power_response(real_scalar(0.0), real_scalar(1.0));

        let result = plane_wave_power_balance_algebraic(&context, &waves, &pr).unwrap();

        assert_real_close(&result.incident_flux, 1.0);

        assert_real_close(&result.reflected_flux, 0.0);

        assert_real_close(&result.transmitted_flux, 1.0);

        assert!(result.layer_absorptance.is_empty());

        assert_real_close(&result.total_layer_absorptance, 0.0);

        assert_real_close(&result.balance_residual, 0.0);
    }

    #[test]
    fn layer_absorptance_is_normalised_flux_drop() {
        let context = symmetric_value_power_context(IncidentSide::Left, 2.0, &[2.0]);

        let waves =
            boundary_waves_with_layers(scalar(0.0, 0.0), vec![layer_with_forward_fluxes(1.0, 0.6)]);

        let result = plane_wave_power_balance_algebraic::<C, D, ComplexArray>(
            &context,
            &waves,
            &power_response(real_scalar(0.1), real_scalar(0.5)),
        )
        .unwrap();

        assert_real_close(&result.layer_absorptance[0], 0.4);

        assert_real_close(&result.total_layer_absorptance, 0.4);

        assert_real_close(&result.balance_residual, 0.0);
    }

    #[test]
    fn total_layer_absorptance_is_sum_of_layers() {
        let context = symmetric_value_power_context(IncidentSide::Left, 2.0, &[2.0, 2.0]);

        let waves = boundary_waves_with_layers(
            scalar(0.0, 0.0),
            vec![
                layer_with_forward_fluxes(1.0, 0.8),
                layer_with_forward_fluxes(0.8, 0.5),
            ],
        );

        let result = plane_wave_power_balance_algebraic::<C, D, ComplexArray>(
            &context,
            &waves,
            &power_response(real_scalar(0.1), real_scalar(0.4)),
        )
        .unwrap();

        assert_real_close(&result.layer_absorptance[0], 0.2);

        assert_real_close(&result.layer_absorptance[1], 0.3);

        assert_real_close(&result.total_layer_absorptance, 0.5);

        assert_real_close(&result.balance_residual, 0.0);
    }

    #[test]
    fn right_incidence_uses_signed_flux_consistently() {
        let context = symmetric_value_power_context(IncidentSide::Right, 2.0, &[2.0]);

        let waves = boundary_waves_with_layers(
            scalar(0.0, 0.0),
            vec![layer_with_backward_fluxes(0.7, 1.0)],
        );

        let result = plane_wave_power_balance_algebraic::<C, D, ComplexArray>(
            &context,
            &waves,
            &power_response(real_scalar(0.1), real_scalar(0.6)),
        )
        .unwrap();

        assert_real_close(&result.incident_flux, 1.0);

        assert_real_close(&result.layer_absorptance[0], 0.3);

        assert_real_close(&result.balance_residual, 0.0);
    }

    #[test]
    fn first_order_balance_derivative_obeys_conservation_expression() {
        let context = first_order_power_context(
            IncidentSide::Left,
            (c(2.0, 0.0), c(0.0, 0.0)),
            (c(2.0, 0.0), c(0.0, 0.0)),
            &[],
        );

        let waves = first_order_boundary_waves_without_layers();

        let result = plane_wave_power_balance_algebraic::<C, D, FirstComplex>(
            &context,
            &waves,
            &power_response(real_first_jet(0.2, 0.03), real_first_jet(0.7, -0.01)),
        )
        .unwrap();

        let (value, first) = result.balance_residual.into_parts();

        assert_real_close(&value, 0.1);
        assert_real_close(&first, -0.02);
    }

    #[test]
    fn second_order_balance_derivative_obeys_conservation_expression() {
        let context = second_order_power_context(
            IncidentSide::Left,
            (c(2.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)),
            (c(2.0, 0.0), c(0.0, 0.0), c(0.0, 0.0)),
            &[],
        );

        let waves = second_order_boundary_waves_without_layers();

        let result = plane_wave_power_balance_algebraic::<C, D, SecondComplex>(
            &context,
            &waves,
            &power_response(
                real_second_jet(0.2, 0.03, 0.04),
                real_second_jet(0.7, -0.01, -0.02),
            ),
        )
        .unwrap();

        let (value, first, second) = result.balance_residual.into_parts();

        assert_real_close(&value, 0.1);
        assert_real_close(&first, -0.02);
        assert_real_close(&second, -0.02);
    }
}
