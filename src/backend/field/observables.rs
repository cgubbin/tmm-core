//! Physical field and power post-processing for isotropic plane-wave solutions.
//!
//! This module converts [`BoundaryWaves`] into:
//!
//! - canonical tangential field states;
//! - signed normal power flux;
//! - fields sampled at arbitrary positions;
//! - per-layer absorptance;
//! - a complete plane-wave power balance.
//!
//! The calculations are backend-neutral once the boundary-wave solution has
//! been obtained.
//!
//! # Wave convention
//!
//! Geometric directions are fixed:
//!
//! - `forward` propagates from left to right;
//! - `backward` propagates from right to left.
//!
//! The finite-layer propagation convention is:
//!
//! ```text
//! p(d) = exp(-i κ d).
//! ```
//!
//! A forward wave referenced to a layer's left boundary therefore evolves as:
//!
//! ```text
//! a⁺(z) = a⁺(0) exp(-i κ z),
//! ```
//!
//! while a backward wave referenced to the layer's right boundary evolves as:
//!
//! ```text
//! a⁻(z) = a⁻(d) exp(-i κ (d - z)).
//! ```
//!
//! # Canonical field state
//!
//! For characteristic admittance `Y`, the canonical tangential pair is:
//!
//! ```text
//! primary = a⁺ + a⁻
//! dual    = Y (a⁺ - a⁻).
//! ```
//!
//! For TE polarisation, `primary` is the tangential electric-field amplitude.
//! For TM polarisation, `primary` is the tangential magnetic-field amplitude.
//! `dual` is the corresponding signed conjugate tangential field required for
//! the normal Poynting flux.
//!
//! The signed normal flux is:
//!
//! ```text
//! Pz = 1/2 Re(primary * dual*).
//! ```
//!
//! Positive flux is directed from left to right.

use ndarray::{ArrayBase, Dimension, OwnedRepr, Zip};
use num_traits::Float;

use crate::{
    ComplexScalar, IncidentSide, PlanarInput, PlaneWaveInput, Stack,
    backend::{
        field::{BidirectionalWaves, BoundaryWaves, LayerBoundaryWaves},
        isotropic::IsotropicLayerQuantities,
    },
    material::EvaluateMaterial,
};

use super::{
    FieldPosition, PlaneWaveFieldError, PlaneWaveFieldResponse,
    sampling::{validate_exterior_distance, validate_layer_offset},
};

/// Canonical tangential field pair at one spatial position.
///
/// The pair is selected so that the signed normal power flux is:
///
/// ```text
/// Pz = 1/2 Re(primary * dual*).
/// ```
#[derive(Clone, Debug, PartialEq)]
pub struct IsotropicFieldState<C, D>
where
    D: Dimension,
{
    primary: ArrayBase<OwnedRepr<C>, D>,
    dual: ArrayBase<OwnedRepr<C>, D>,
}

impl<C, D> IsotropicFieldState<C, D>
where
    D: Dimension,
{
    pub(crate) fn from_waves(
        waves: &BidirectionalWaves<C, D>,
        admittance: &ArrayBase<OwnedRepr<C>, D>,
    ) -> Self
    where
        C: ComplexScalar,
    {
        let primary = waves.forward().clone() + waves.backward().view();

        let dual = admittance.clone() * (waves.forward().clone() - waves.backward().view());

        Self::new(primary, dual)
    }

    pub(crate) fn new(
        primary: ArrayBase<OwnedRepr<C>, D>,
        dual: ArrayBase<OwnedRepr<C>, D>,
    ) -> Self {
        debug_assert_eq!(primary.raw_dim(), dual.raw_dim());

        Self { primary, dual }
    }

    /// Return the primary tangential field.
    ///
    /// This is the tangential electric field for TE and the tangential magnetic
    /// field for TM.
    pub fn primary(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.primary
    }

    /// Return the signed dual tangential field.
    pub fn dual(&self) -> &ArrayBase<OwnedRepr<C>, D> {
        &self.dual
    }

    /// Consume the state and return its canonical field pair.
    pub fn into_parts(self) -> (ArrayBase<OwnedRepr<C>, D>, ArrayBase<OwnedRepr<C>, D>) {
        (self.primary, self.dual)
    }

    /// Return the squared magnitude of the primary field.
    pub fn primary_intensity(&self) -> ArrayBase<OwnedRepr<C::RealField>, D>
    where
        C: ComplexScalar,
    {
        self.primary.mapv(|value| value.modulus_squared())
    }

    /// Return the signed normal time-averaged power flux.
    ///
    /// Positive values represent left-to-right flux.
    pub fn normal_flux(&self) -> ArrayBase<OwnedRepr<C::RealField>, D>
    where
        C: ComplexScalar,
        C::RealField: Float,
    {
        let half = C::one().real() / (C::one().real() + C::one().real());

        Zip::from(&self.primary)
            .and(&self.dual)
            .map_collect(|&primary, &dual| half * (primary * dual.conjugate()).real())
    }
}

/// Field state and signed normal flux at one requested position.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveFieldSample<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    position: FieldPosition<C::RealField>,

    /// Global coordinate measured rightward from the stack's left boundary.
    ///
    /// Left-exterior coordinates are negative.
    coordinate: C::RealField,

    state: IsotropicFieldState<C, D>,
    normal_flux: ArrayBase<OwnedRepr<C::RealField>, D>,
}

impl<C, D> PlaneWaveFieldSample<C, D>
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    pub(crate) fn new(
        position: FieldPosition<C::RealField>,
        coordinate: C::RealField,
        state: IsotropicFieldState<C, D>,
    ) -> Self
    where
        C::RealField: Float,
    {
        let normal_flux = state.normal_flux();

        Self {
            position,
            coordinate,
            state,
            normal_flux,
        }
    }

    pub fn position(&self) -> FieldPosition<C::RealField> {
        self.position
    }

    pub fn coordinate(&self) -> C::RealField {
        self.coordinate
    }

    pub fn state(&self) -> &IsotropicFieldState<C, D> {
        &self.state
    }

    pub fn normal_flux(&self) -> &ArrayBase<OwnedRepr<C::RealField>, D> {
        &self.normal_flux
    }
}

/// Fields sampled at a sequence of requested positions.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWaveFields<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    samples: Vec<PlaneWaveFieldSample<C, D>>,
}

impl<C, D> PlaneWaveFields<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(crate) fn new(samples: Vec<PlaneWaveFieldSample<C, D>>) -> Self {
        Self { samples }
    }

    pub fn samples(&self) -> &[PlaneWaveFieldSample<C, D>] {
        &self.samples
    }

    pub fn sample(&self, index: usize) -> Option<&PlaneWaveFieldSample<C, D>> {
        self.samples.get(index)
    }

    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    pub fn into_samples(self) -> Vec<PlaneWaveFieldSample<C, D>> {
        self.samples
    }
}

/// Per-layer and whole-stack physical power balance.
#[derive(Clone, Debug, PartialEq)]
pub struct PlaneWavePowerBalance<R, D>
where
    D: Dimension,
{
    incident_flux: ArrayBase<OwnedRepr<R>, D>,
    reflected_flux: ArrayBase<OwnedRepr<R>, D>,
    transmitted_flux: ArrayBase<OwnedRepr<R>, D>,
    layer_absorptance: Vec<ArrayBase<OwnedRepr<R>, D>>,
    total_layer_absorptance: ArrayBase<OwnedRepr<R>, D>,
    balance_residual: ArrayBase<OwnedRepr<R>, D>,
}

impl<R, D> PlaneWavePowerBalance<R, D>
where
    D: Dimension,
{
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
}

/// Sample physical field states at arbitrary exterior or finite-layer
/// positions.
pub fn sample_plane_wave_fields<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    waves: &BoundaryWaves<C, D>,
    positions: impl IntoIterator<Item = FieldPosition<C::RealField>>,
) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
where
    M: EvaluateMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
{
    validate_layer_count(stack, waves)?;

    let mut layer_origins = Vec::with_capacity(stack.len());

    let mut total_thickness = C::zero().real();

    for layer in stack.layers_left_to_right() {
        layer_origins.push(total_thickness);

        total_thickness = total_thickness + layer.thickness().as_cm();
    }

    let planar = complex_planar_input::<C, D>(input);

    let left_quantities = IsotropicLayerQuantities::real_axis(stack.left_exterior(), &planar);

    let right_quantities = IsotropicLayerQuantities::real_axis(stack.right_exterior(), &planar);

    let mut layer_data = Vec::with_capacity(stack.len());

    for layer in stack.layers_left_to_right() {
        let quantities = IsotropicLayerQuantities::real_axis(layer.material(), &planar);

        layer_data.push((quantities, layer.thickness().as_cm()));
    }

    let mut samples = Vec::new();

    for position in positions {
        let (coordinate, state) = match position {
            FieldPosition::LeftExterior { distance } => {
                validate_exterior_distance(distance)?;

                let local = sample_left_exterior_waves(
                    waves.exterior().left(),
                    left_quantities.kappa(),
                    distance,
                );

                (
                    -distance,
                    IsotropicFieldState::from_waves(&local, left_quantities.admittance().value()),
                )
            }

            FieldPosition::Layer { index, offset } => {
                let Some((quantities, thickness)) = layer_data.get(index) else {
                    return Err(PlaneWaveFieldError::LayerOutOfBounds {
                        requested: index,
                        layer_count: stack.len(),
                    });
                };

                validate_layer_offset(index, offset, *thickness)?;

                let boundary = waves
                    .layer(index)
                    .ok_or(PlaneWaveFieldError::LayerOutOfBounds {
                        requested: index,
                        layer_count: waves.len(),
                    })?;

                let local = sample_layer_waves(boundary, quantities.kappa(), offset, *thickness);

                (
                    layer_origins[index] + offset,
                    IsotropicFieldState::from_waves(&local, quantities.admittance().value()),
                )
            }

            FieldPosition::RightExterior { distance } => {
                validate_exterior_distance(distance)?;

                let local = sample_right_exterior_waves(
                    waves.exterior().right(),
                    right_quantities.kappa(),
                    distance,
                );

                (
                    total_thickness + distance,
                    IsotropicFieldState::from_waves(&local, right_quantities.admittance().value()),
                )
            }
        };

        samples.push(PlaneWaveFieldSample::new(position, coordinate, state));
    }

    Ok(PlaneWaveFields::new(samples))
}

/// Sample fields from a high-level spatial sampling specification.
pub fn sample_plane_wave_field_profile<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    waves: &BoundaryWaves<C, D>,
    sampling: &super::sampling::FieldSampling<C::RealField>,
) -> Result<PlaneWaveFields<C, D>, PlaneWaveFieldError<C::RealField>>
where
    M: EvaluateMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
{
    let positions = sampling.positions(stack)?;

    sample_plane_wave_fields(stack, input, waves, positions)
}

/// Calculate per-layer absorptance and the global power-balance residual.
pub fn plane_wave_power_balance<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    solution: &PlaneWaveFieldResponse<C, D>,
) -> Result<PlaneWavePowerBalance<C::RealField, D>, PlaneWaveFieldError<C::RealField>>
where
    M: EvaluateMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
{
    let waves = solution.boundary_waves();

    validate_layer_count(stack, waves.values())?;

    let planar = complex_planar_input::<C, D>(input);

    let left_quantities = IsotropicLayerQuantities::real_axis(stack.left_exterior(), &planar);

    let right_quantities = IsotropicLayerQuantities::real_axis(stack.right_exterior(), &planar);

    let left_exterior_state = IsotropicFieldState::from_waves(
        waves.exterior().left(),
        left_quantities.admittance().value(),
    );

    let right_exterior_state = IsotropicFieldState::from_waves(
        waves.exterior().right(),
        right_quantities.admittance().value(),
    );

    let left_total_flux = left_exterior_state.normal_flux();
    let right_total_flux = right_exterior_state.normal_flux();

    let incident_flux = incident_flux_magnitude(
        input.incident_side(),
        left_quantities.admittance().value(),
        right_quantities.admittance().value(),
    )?;

    let response = solution.response();

    let reflected_flux = response.power().reflectance().clone() * incident_flux.view();

    let transmitted_flux = response.power().transmittance().clone() * incident_flux.view();

    let mut layer_absorptance = Vec::with_capacity(stack.len());

    for (index, layer) in stack.layers_left_to_right().iter().enumerate() {
        let quantities = IsotropicLayerQuantities::real_axis(layer.material(), &planar);

        let boundary = waves
            .layer(index)
            .ok_or(PlaneWaveFieldError::LayerOutOfBounds {
                requested: index,
                layer_count: waves.len(),
            })?;

        let left_state =
            IsotropicFieldState::from_waves(boundary.left(), quantities.admittance().value());

        let right_state =
            IsotropicFieldState::from_waves(boundary.right(), quantities.admittance().value());

        let left_flux = left_state.normal_flux();
        let right_flux = right_state.normal_flux();

        /*
         * This expression is valid for both incidence sides because flux is
         * signed geometrically:
         *
         * left incidence:  P_left > P_right
         * right incidence: P_left is less negative than P_right.
         */
        let absorption = (left_flux - right_flux) / incident_flux.view();

        layer_absorptance.push(absorption);
    }

    let mut total_layer_absorptance = incident_flux.mapv(|_| C::zero().real());

    for absorption in &layer_absorptance {
        total_layer_absorptance = total_layer_absorptance + absorption.view();
    }

    let one = incident_flux.mapv(|_| C::one().real());

    let balance_residual = one
        - response.power().reflectance()
        - response.power().transmittance()
        - total_layer_absorptance.view();

    /*
     * Keep these computations in scope during early development. They provide
     * useful debugging checks for the external boundary convention.
     */
    debug_assert_eq!(left_total_flux.raw_dim(), incident_flux.raw_dim(),);
    debug_assert_eq!(right_total_flux.raw_dim(), incident_flux.raw_dim(),);

    Ok(PlaneWavePowerBalance {
        incident_flux,
        reflected_flux,
        transmitted_flux,
        layer_absorptance,
        total_layer_absorptance,
        balance_residual,
    })
}

fn complex_planar_input<C, D>(
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
) -> PlanarInput<ArrayBase<OwnedRepr<C>, D>>
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    input.planar().map(|values| values.mapv(C::from_real))
}

fn validate_layer_count<C, D, M>(
    stack: &Stack<M, C::RealField>,
    waves: &BoundaryWaves<C, D>,
) -> Result<(), PlaneWaveFieldError<C::RealField>>
where
    C: ComplexScalar,
    D: Dimension,
{
    if stack.len() != waves.len() {
        return Err(PlaneWaveFieldError::LayerCountMismatch {
            expected: stack.len(),
            actual: waves.len(),
        });
    }

    Ok(())
}

/// Central spatial-phase convention.
///
/// Change this function if the scattering propagation component uses the
/// opposite exponential sign.
fn propagation_phase<C>(kappa: C, distance: C::RealField) -> C
where
    C: ComplexScalar,
{
    let distance = C::from_real(distance);

    (-C::i() * kappa * distance).exp()
}

fn sample_layer_waves<C, D>(
    boundary: &LayerBoundaryWaves<C, D>,
    kappa: &ArrayBase<OwnedRepr<C>, D>,
    offset: C::RealField,
    thickness: C::RealField,
) -> BidirectionalWaves<C, D>
where
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
{
    let forward_phase = kappa.mapv(|value| propagation_phase(value, offset));

    let backward_distance = thickness - offset;

    let backward_phase = kappa.mapv(|value| propagation_phase(value, backward_distance));

    BidirectionalWaves::new(
        boundary.left().forward().clone() * forward_phase,
        boundary.right().backward().clone() * backward_phase,
    )
}

fn sample_left_exterior_waves<C, D>(
    boundary: &BidirectionalWaves<C, D>,
    kappa: &ArrayBase<OwnedRepr<C>, D>,
    distance: C::RealField,
) -> BidirectionalWaves<C, D>
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    /*
     * The sampling point lies at geometric coordinate z = -distance relative
     * to the left stack boundary.
     */
    let forward_phase = kappa.mapv(|value| propagation_phase(value, -distance));

    let backward_phase = kappa.mapv(|value| propagation_phase(-value, -distance));

    BidirectionalWaves::new(
        boundary.forward().clone() * forward_phase,
        boundary.backward().clone() * backward_phase,
    )
}

fn sample_right_exterior_waves<C, D>(
    boundary: &BidirectionalWaves<C, D>,
    kappa: &ArrayBase<OwnedRepr<C>, D>,
    distance: C::RealField,
) -> BidirectionalWaves<C, D>
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let forward_phase = kappa.mapv(|value| propagation_phase(value, distance));

    let backward_phase = kappa.mapv(|value| propagation_phase(-value, distance));

    BidirectionalWaves::new(
        boundary.forward().clone() * forward_phase,
        boundary.backward().clone() * backward_phase,
    )
}

fn incident_flux_magnitude<C, D>(
    side: IncidentSide,
    left_admittance: &ArrayBase<OwnedRepr<C>, D>,
    right_admittance: &ArrayBase<OwnedRepr<C>, D>,
) -> Result<ArrayBase<OwnedRepr<C::RealField>, D>, PlaneWaveFieldError<C::RealField>>
where
    C: ComplexScalar,
    C::RealField: Float,
    D: Dimension,
{
    let half = C::one().real() / (C::one().real() + C::one().real());

    let incident = match side {
        IncidentSide::Left => left_admittance,
        IncidentSide::Right => right_admittance,
    }
    .mapv(|value| half * value.real());

    if incident
        .iter()
        .any(|value| !value.is_finite() || *value <= C::zero().real())
    {
        return Err(PlaneWaveFieldError::InvalidIncidentFlux);
    }

    Ok(incident)
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, Ix1, arr1};
    use num_complex::Complex64;

    use crate::{
        IncidentSide,
        backend::field::{BidirectionalWaves, BoundaryWaves, ExteriorBoundaryWaves},
    };

    use super::*;

    type C = Complex64;
    type D = Ix1;

    const TOLERANCE: f64 = 1.0e-12;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn r(value: f64) -> C {
        c(value, 0.0)
    }

    fn assert_real_close(actual: f64, expected: f64) {
        let error = (actual - expected).abs();

        assert!(
            error <= TOLERANCE,
            "expected {expected:e}, got {actual:e}; \
             absolute error = {error:e}",
        );
    }

    fn assert_complex_close(actual: C, expected: C) {
        let error = (actual - expected).norm();

        assert!(
            error <= TOLERANCE,
            "expected {expected:?}, got {actual:?}; \
             absolute error = {error:e}",
        );
    }

    fn assert_array_real_close(actual: &Array1<f64>, expected: &Array1<f64>) {
        assert_eq!(actual.raw_dim(), expected.raw_dim());

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_real_close(actual, expected);
        }
    }

    fn assert_array_complex_close(actual: &Array1<C>, expected: &Array1<C>) {
        assert_eq!(actual.raw_dim(), expected.raw_dim());

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_complex_close(actual, expected);
        }
    }

    fn waves(forward: &[C], backward: &[C]) -> BidirectionalWaves<C, D> {
        BidirectionalWaves::new(
            Array1::from_vec(forward.to_vec()),
            Array1::from_vec(backward.to_vec()),
        )
    }

    fn exterior_waves(
        left: BidirectionalWaves<C, D>,
        right: BidirectionalWaves<C, D>,
    ) -> ExteriorBoundaryWaves<C, D> {
        ExteriorBoundaryWaves::new(left, right)
    }

    // ---------------------------------------------------------------------
    // IsotropicFieldState
    // ---------------------------------------------------------------------

    #[test]
    fn isotropic_field_state_preserves_primary_and_dual_fields() {
        let primary = arr1(&[c(1.0, 2.0), c(3.0, 4.0)]);

        let dual = arr1(&[c(5.0, 6.0), c(7.0, 8.0)]);

        let state = IsotropicFieldState::new(primary.clone(), dual.clone());

        assert_eq!(state.primary(), &primary);
        assert_eq!(state.dual(), &dual);
    }

    #[test]
    fn isotropic_field_state_into_parts_returns_original_arrays() {
        let primary = arr1(&[c(1.0, 2.0), c(3.0, 4.0)]);

        let dual = arr1(&[c(5.0, 6.0), c(7.0, 8.0)]);

        let state = IsotropicFieldState::new(primary.clone(), dual.clone());

        let (actual_primary, actual_dual) = state.into_parts();

        assert_eq!(actual_primary, primary);
        assert_eq!(actual_dual, dual);
    }

    #[test]
    fn primary_intensity_returns_squared_modulus() {
        let state = IsotropicFieldState::new(
            arr1(&[c(3.0, 4.0), c(5.0, 12.0), c(0.0, 2.0)]),
            arr1(&[C::new(0.0, 0.0), C::new(0.0, 0.0), C::new(0.0, 0.0)]),
        );

        assert_array_real_close(&state.primary_intensity(), &arr1(&[25.0, 169.0, 4.0]));
    }

    #[test]
    fn normal_flux_uses_primary_times_conjugate_dual() {
        let state = IsotropicFieldState::new(
            arr1(&[c(2.0, 1.0), c(1.0, -2.0)]),
            arr1(&[c(3.0, -1.0), c(-2.0, 4.0)]),
        );

        let expected = arr1(&[
            0.5 * (c(2.0, 1.0) * c(3.0, -1.0).conj()).re,
            0.5 * (c(1.0, -2.0) * c(-2.0, 4.0).conj()).re,
        ]);

        assert_array_real_close(&state.normal_flux(), &expected);
    }

    // ---------------------------------------------------------------------
    // Wave-to-field reconstruction
    // ---------------------------------------------------------------------

    #[test]
    fn state_from_waves_reconstructs_primary_and_dual() {
        let local_waves = waves(&[c(2.0, 1.0), c(1.0, -2.0)], &[c(0.5, -1.0), c(-1.0, 0.5)]);

        let admittance = arr1(&[r(3.0), r(2.0)]);

        let state = IsotropicFieldState::from_waves(&local_waves, &admittance);

        let expected_primary = arr1(&[c(2.5, 0.0), c(0.0, -1.5)]);

        let expected_dual = arr1(&[r(3.0) * c(1.5, 2.0), r(2.0) * c(2.0, -2.5)]);

        assert_array_complex_close(state.primary(), &expected_primary);

        assert_array_complex_close(state.dual(), &expected_dual);
    }

    #[test]
    fn forward_wave_has_positive_flux_for_positive_real_admittance() {
        let local_waves = waves(&[r(2.0)], &[r(0.0)]);

        let state = IsotropicFieldState::from_waves(&local_waves, &arr1(&[r(3.0)]));

        // primary = 2
        // dual = 3 * 2 = 6
        // flux = 1/2 * 2 * 6 = 6
        assert_real_close(state.normal_flux()[0], 6.0);
    }

    #[test]
    fn backward_wave_has_negative_flux_for_positive_real_admittance() {
        let local_waves = waves(&[r(0.0)], &[r(2.0)]);

        let state = IsotropicFieldState::from_waves(&local_waves, &arr1(&[r(3.0)]));

        // primary = 2
        // dual = 3 * (0 - 2) = -6
        // flux = 1/2 * 2 * -6 = -6
        assert_real_close(state.normal_flux()[0], -6.0);
    }

    #[test]
    fn equal_counterpropagating_waves_have_zero_net_flux() {
        let local_waves = waves(&[r(1.0)], &[r(1.0)]);

        let state = IsotropicFieldState::from_waves(&local_waves, &arr1(&[r(2.5)]));

        assert_eq!(state.primary(), &arr1(&[r(2.0)]),);

        assert_eq!(state.dual(), &arr1(&[r(0.0)]),);

        assert_real_close(state.normal_flux()[0], 0.0);
    }

    #[test]
    fn lossless_flux_is_forward_power_minus_backward_power() {
        let forward = c(2.0, 1.0);
        let backward = c(0.5, -0.25);
        let admittance = 4.0;

        let local_waves = waves(&[forward], &[backward]);

        let state = IsotropicFieldState::from_waves(&local_waves, &arr1(&[r(admittance)]));

        let expected = 0.5 * admittance * (forward.norm_sqr() - backward.norm_sqr());

        assert_real_close(state.normal_flux()[0], expected);
    }

    // ---------------------------------------------------------------------
    // Propagation phase
    // ---------------------------------------------------------------------

    #[test]
    fn propagation_phase_is_unity_at_zero_distance() {
        let phase = propagation_phase(c(2.0, 3.0), 0.0);

        assert_complex_close(phase, C::new(1.0, 0.0));
    }

    #[test]
    fn propagation_phase_matches_negative_i_kappa_d_convention() {
        let kappa = c(2.0, 0.0);
        let distance = 0.25;

        let actual = propagation_phase(kappa, distance);

        let expected = c(0.0, -0.5).exp();

        assert_complex_close(actual, expected);
    }

    #[test]
    fn propagation_phase_decays_for_negative_imaginary_kappa() {
        /*
         * With p(d) = exp(-i κ d), decay in the positive-z direction
         * requires Im(κ) < 0.
         */
        let phase = propagation_phase(c(0.0, -2.0), 0.5);

        assert_complex_close(phase, r((-1.0_f64).exp()));
    }

    // ---------------------------------------------------------------------
    // Finite-layer sampling
    // ---------------------------------------------------------------------

    #[test]
    fn layer_sampling_at_left_boundary_uses_left_forward_reference() {
        let boundary = LayerBoundaryWaves::new(
            waves(&[c(2.0, 1.0)], &[c(99.0, 0.0)]),
            waves(&[c(98.0, 0.0)], &[c(0.5, -0.25)]),
        );

        let kappa = arr1(&[c(1.5, 0.0)]);
        let thickness = 0.4;

        let sampled = sample_layer_waves(&boundary, &kappa, 0.0, thickness);

        assert_complex_close(sampled.forward()[0], c(2.0, 1.0));

        let expected_backward = c(0.5, -0.25) * propagation_phase(kappa[0], thickness);

        assert_complex_close(sampled.backward()[0], expected_backward);
    }

    #[test]
    fn layer_sampling_at_right_boundary_uses_right_backward_reference() {
        let boundary = LayerBoundaryWaves::new(
            waves(&[c(2.0, 1.0)], &[c(99.0, 0.0)]),
            waves(&[c(98.0, 0.0)], &[c(0.5, -0.25)]),
        );

        let kappa = arr1(&[c(1.5, 0.0)]);
        let thickness = 0.4;

        let sampled = sample_layer_waves(&boundary, &kappa, thickness, thickness);

        let expected_forward = c(2.0, 1.0) * propagation_phase(kappa[0], thickness);

        assert_complex_close(sampled.forward()[0], expected_forward);

        assert_complex_close(sampled.backward()[0], c(0.5, -0.25));
    }

    #[test]
    fn layer_sampling_propagates_from_opposite_reference_boundaries() {
        let boundary =
            LayerBoundaryWaves::new(waves(&[r(2.0)], &[r(0.0)]), waves(&[r(0.0)], &[r(0.5)]));

        let kappa = arr1(&[r(2.0)]);
        let thickness = 1.0;
        let offset = 0.25;

        let sampled = sample_layer_waves(&boundary, &kappa, offset, thickness);

        let expected_forward = r(2.0) * c(0.0, -0.5).exp();

        let expected_backward = r(0.5) * c(0.0, -1.5).exp();

        assert_complex_close(sampled.forward()[0], expected_forward);

        assert_complex_close(sampled.backward()[0], expected_backward);
    }

    #[test]
    fn layer_sampling_is_vectorised() {
        let boundary = LayerBoundaryWaves::new(
            waves(&[r(1.0), r(2.0)], &[r(0.0), r(0.0)]),
            waves(&[r(0.0), r(0.0)], &[r(3.0), r(4.0)]),
        );

        let kappa = arr1(&[r(1.0), r(2.0)]);

        let sampled = sample_layer_waves(&boundary, &kappa, 0.25, 1.0);

        assert_complex_close(sampled.forward()[0], propagation_phase(r(1.0), 0.25));

        assert_complex_close(
            sampled.forward()[1],
            r(2.0) * propagation_phase(r(2.0), 0.25),
        );

        assert_complex_close(
            sampled.backward()[0],
            r(3.0) * propagation_phase(r(1.0), 0.75),
        );

        assert_complex_close(
            sampled.backward()[1],
            r(4.0) * propagation_phase(r(2.0), 0.75),
        );
    }

    // ---------------------------------------------------------------------
    // Exterior sampling
    // ---------------------------------------------------------------------

    #[test]
    fn left_exterior_sampling_returns_boundary_values_at_zero_distance() {
        let boundary = waves(&[c(2.0, 1.0)], &[c(0.5, -0.25)]);

        let sampled = sample_left_exterior_waves(&boundary, &arr1(&[r(2.0)]), 0.0);

        assert_eq!(sampled.forward(), boundary.forward(),);

        assert_eq!(sampled.backward(), boundary.backward(),);
    }

    #[test]
    fn right_exterior_sampling_returns_boundary_values_at_zero_distance() {
        let boundary = waves(&[c(2.0, 1.0)], &[c(0.5, -0.25)]);

        let sampled = sample_right_exterior_waves(&boundary, &arr1(&[r(2.0)]), 0.0);

        assert_eq!(sampled.forward(), boundary.forward(),);

        assert_eq!(sampled.backward(), boundary.backward(),);
    }

    #[test]
    fn left_exterior_sampling_uses_negative_geometric_coordinate() {
        let boundary = waves(&[r(2.0)], &[r(0.5)]);

        let kappa = arr1(&[r(3.0)]);
        let distance = 0.2;

        let sampled = sample_left_exterior_waves(&boundary, &kappa, distance);

        let expected_forward = r(2.0) * propagation_phase(kappa[0], -distance);

        let expected_backward = r(0.5) * propagation_phase(-kappa[0], -distance);

        assert_complex_close(sampled.forward()[0], expected_forward);

        assert_complex_close(sampled.backward()[0], expected_backward);
    }

    #[test]
    fn right_exterior_sampling_uses_positive_geometric_coordinate() {
        let boundary = waves(&[r(2.0)], &[r(0.5)]);

        let kappa = arr1(&[r(3.0)]);
        let distance = 0.2;

        let sampled = sample_right_exterior_waves(&boundary, &kappa, distance);

        let expected_forward = r(2.0) * propagation_phase(kappa[0], distance);

        let expected_backward = r(0.5) * propagation_phase(-kappa[0], distance);

        assert_complex_close(sampled.forward()[0], expected_forward);

        assert_complex_close(sampled.backward()[0], expected_backward);
    }

    // ---------------------------------------------------------------------
    // Position validation
    // ---------------------------------------------------------------------

    #[test]
    fn zero_exterior_distance_is_valid() {
        assert_eq!(validate_exterior_distance(0.0_f64), Ok(()),);
    }

    #[test]
    fn positive_exterior_distance_is_valid() {
        assert_eq!(validate_exterior_distance(1.25_f64), Ok(()),);
    }

    #[test]
    fn negative_exterior_distance_is_rejected() {
        let result = validate_exterior_distance(-0.25_f64);

        assert_eq!(
            result,
            Err(PlaneWaveFieldError::InvalidExteriorDistance { distance: -0.25 },),
        );
    }

    #[test]
    fn infinite_exterior_distance_is_rejected() {
        let result = validate_exterior_distance(f64::INFINITY);

        assert_eq!(
            result,
            Err(PlaneWaveFieldError::InvalidExteriorDistance {
                distance: f64::INFINITY,
            },),
        );
    }

    #[test]
    fn nan_exterior_distance_is_rejected() {
        let result = validate_exterior_distance(f64::NAN);

        assert!(matches!(
            result,
            Err(
                PlaneWaveFieldError::
                    InvalidExteriorDistance {
                        distance,
                    },
            ) if distance.is_nan()
        ));
    }

    // ---------------------------------------------------------------------
    // Incident flux
    // ---------------------------------------------------------------------

    #[test]
    fn left_incident_flux_uses_left_admittance() {
        let left = arr1(&[r(2.0), r(4.0)]);

        let right = arr1(&[r(10.0), r(20.0)]);

        let flux = incident_flux_magnitude(IncidentSide::Left, &left, &right).unwrap();

        assert_array_real_close(&flux, &arr1(&[1.0, 2.0]));
    }

    #[test]
    fn right_incident_flux_uses_right_admittance() {
        let left = arr1(&[r(10.0), r(20.0)]);

        let right = arr1(&[r(2.0), r(4.0)]);

        let flux = incident_flux_magnitude(IncidentSide::Right, &left, &right).unwrap();

        assert_array_real_close(&flux, &arr1(&[1.0, 2.0]));
    }

    #[test]
    fn incident_flux_uses_real_part_of_complex_admittance() {
        let left = arr1(&[c(2.0, 7.0), c(4.0, -9.0)]);

        let right = arr1(&[r(1.0), r(1.0)]);

        let flux = incident_flux_magnitude(IncidentSide::Left, &left, &right).unwrap();

        assert_array_real_close(&flux, &arr1(&[1.0, 2.0]));
    }

    #[test]
    fn zero_incident_admittance_is_rejected() {
        let result =
            incident_flux_magnitude(IncidentSide::Left, &arr1(&[r(0.0)]), &arr1(&[r(1.0)]));

        assert_eq!(result, Err(PlaneWaveFieldError::InvalidIncidentFlux,),);
    }

    #[test]
    fn negative_incident_admittance_is_rejected() {
        let result =
            incident_flux_magnitude(IncidentSide::Left, &arr1(&[r(-1.0)]), &arr1(&[r(1.0)]));

        assert_eq!(result, Err(PlaneWaveFieldError::InvalidIncidentFlux,),);
    }

    #[test]
    fn non_finite_incident_admittance_is_rejected() {
        let result = incident_flux_magnitude(
            IncidentSide::Left,
            &arr1(&[c(f64::NAN, 0.0)]),
            &arr1(&[r(1.0)]),
        );

        assert_eq!(result, Err(PlaneWaveFieldError::InvalidIncidentFlux,),);
    }

    #[test]
    fn any_invalid_sample_rejects_incident_flux_array() {
        let result = incident_flux_magnitude(
            IncidentSide::Left,
            &arr1(&[r(2.0), r(0.0), r(4.0)]),
            &arr1(&[r(1.0), r(1.0), r(1.0)]),
        );

        assert_eq!(result, Err(PlaneWaveFieldError::InvalidIncidentFlux,),);
    }

    // ---------------------------------------------------------------------
    // Field containers
    // ---------------------------------------------------------------------

    #[test]
    fn field_sample_caches_normal_flux() {
        let state = IsotropicFieldState::new(arr1(&[r(2.0)]), arr1(&[r(6.0)]));

        let sample = PlaneWaveFieldSample::new(
            FieldPosition::Layer {
                index: 3,
                offset: 0.25,
            },
            0.5,
            state,
        );

        assert_eq!(
            sample.position(),
            FieldPosition::Layer {
                index: 3,
                offset: 0.25,
            },
        );

        assert_real_close(sample.normal_flux()[0], 6.0);
    }

    #[test]
    fn plane_wave_fields_preserves_sample_order() {
        let first = PlaneWaveFieldSample::new(
            FieldPosition::LeftExterior { distance: 0.5 },
            -0.5,
            IsotropicFieldState::new(arr1(&[r(1.0)]), arr1(&[r(2.0)])),
        );

        let second = PlaneWaveFieldSample::new(
            FieldPosition::RightExterior { distance: 0.75 },
            0.75,
            IsotropicFieldState::new(arr1(&[r(3.0)]), arr1(&[r(4.0)])),
        );

        let fields = PlaneWaveFields::new(vec![first, second]);

        assert_eq!(fields.len(), 2);
        assert!(!fields.is_empty());

        assert_eq!(
            fields.sample(0).unwrap().position(),
            FieldPosition::LeftExterior { distance: 0.5 },
        );

        assert_eq!(
            fields.sample(1).unwrap().position(),
            FieldPosition::RightExterior { distance: 0.75 },
        );

        assert!(fields.sample(2).is_none());
    }

    #[test]
    fn empty_plane_wave_fields_reports_empty() {
        let fields: PlaneWaveFields<C, D> = PlaneWaveFields::new(Vec::new());

        assert!(fields.is_empty());
        assert_eq!(fields.len(), 0);
        assert!(fields.samples().is_empty());
    }

    #[test]
    fn into_samples_returns_owned_samples() {
        let sample = PlaneWaveFieldSample::new(
            FieldPosition::LeftExterior { distance: 0.0 },
            0.0,
            IsotropicFieldState::new(arr1(&[r(1.0)]), arr1(&[r(2.0)])),
        );

        let fields = PlaneWaveFields::new(vec![sample]);

        let samples = fields.into_samples();

        assert_eq!(samples.len(), 1);

        assert_eq!(
            samples[0].position(),
            FieldPosition::LeftExterior { distance: 0.0 },
        );
    }

    // ---------------------------------------------------------------------
    // Power-balance container
    // ---------------------------------------------------------------------

    #[test]
    fn power_balance_accessors_preserve_all_components() {
        let balance = PlaneWavePowerBalance {
            incident_flux: arr1(&[1.0, 2.0]),
            reflected_flux: arr1(&[0.1, 0.2]),
            transmitted_flux: arr1(&[0.7, 1.4]),
            layer_absorptance: vec![arr1(&[0.1, 0.2]), arr1(&[0.1, 0.2])],
            total_layer_absorptance: arr1(&[0.2, 0.4]),
            balance_residual: arr1(&[0.0, 0.0]),
        };

        assert_eq!(balance.incident_flux(), &arr1(&[1.0, 2.0]),);

        assert_eq!(balance.reflected_flux(), &arr1(&[0.1, 0.2]),);

        assert_eq!(balance.transmitted_flux(), &arr1(&[0.7, 1.4]),);

        assert_eq!(balance.layer_absorptance().len(), 2,);

        assert_eq!(balance.layer(0), Some(&arr1(&[0.1, 0.2])),);

        assert_eq!(balance.layer(1), Some(&arr1(&[0.1, 0.2])),);

        assert!(balance.layer(2).is_none());

        assert_eq!(balance.total_layer_absorptance(), &arr1(&[0.2, 0.4]),);

        assert_eq!(balance.balance_residual(), &arr1(&[0.0, 0.0]),);
    }

    // ---------------------------------------------------------------------
    // Boundary-wave aggregate sanity
    // ---------------------------------------------------------------------

    #[test]
    fn empty_boundary_wave_collection_retains_exterior_waves() {
        let exterior = exterior_waves(waves(&[r(1.0)], &[r(0.25)]), waves(&[r(0.75)], &[r(0.0)]));

        let boundary_waves = BoundaryWaves::new(exterior, Vec::new());

        assert_eq!(boundary_waves.len(), 0);
        assert!(boundary_waves.is_empty());

        assert_eq!(boundary_waves.exterior().left().forward(), &arr1(&[r(1.0)]),);

        assert_eq!(
            boundary_waves.exterior().right().forward(),
            &arr1(&[r(0.75)]),
        );
    }

    #[test]
    fn layer_offset_accepts_both_boundaries() {
        assert_eq!(validate_layer_offset(0, 0.0_f64, 1.0), Ok(()),);

        assert_eq!(validate_layer_offset(0, 1.0_f64, 1.0), Ok(()),);
    }

    #[test]
    fn layer_offset_rejects_negative_offset() {
        assert_eq!(
            validate_layer_offset(2, -0.1_f64, 1.0),
            Err(PlaneWaveFieldError::InvalidLayerOffset {
                layer: 2,
                offset: -0.1,
                thickness: 1.0,
            },),
        );
    }

    #[test]
    fn layer_offset_rejects_offset_beyond_thickness() {
        assert_eq!(
            validate_layer_offset(2, 1.1_f64, 1.0),
            Err(PlaneWaveFieldError::InvalidLayerOffset {
                layer: 2,
                offset: 1.1,
                thickness: 1.0,
            },),
        );
    }
}
