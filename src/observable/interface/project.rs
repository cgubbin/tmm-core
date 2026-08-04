//! Assembly of backend-independent interface reconstruction data.
//!
//! This module bridges retained backend quantities and interface observables.
//! It:
//!
//! - constructs directional waves in the exterior media;
//! - collects finite-layer characteristic admittances;
//! - assembles finite-layer and exterior quantities into physical interfaces;
//! - projects retained layer waves into canonical boundary states.
//!
//! All returned sequences use physical left-to-right order. A stack with `N`
//! finite layers produces `N + 1` interfaces.

use ndarray::{ArrayBase, Dimension, OwnedRepr};
use num_traits::{One, Zero};

use crate::{
    ComplexScalar, IncidentSide,
    algebra::ScalarAlgebra,
    backend::RetainedIsotropicLayers,
    observable::{
        BoundaryProjectionError, BoundaryWaves, Interfaces, LayerBoundaries, LayerBoundaryWaves,
        PlaneWaveAmplitudes,
    },
};

use super::{ExteriorBoundaryStates, ExteriorBoundaryWaves, InterfaceSide, InterfaceWaveData};

/// Failure to project interface observables from a retained backend result.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum InterfaceProjectionError {
    #[error("error in boundary projection {0}")]
    Boundary(#[from] BoundaryProjectionError),

    #[error(
        "finite-layer boundary data are inconsistent: \
     {wave_count} wave records, and \
     {admittance_count} admittances"
    )]
    LayerDataCountMismatch {
        wave_count: usize,
        admittance_count: usize,
    },
}

pub(crate) fn assemble_interface_wave_data<A>(
    layer_waves: LayerBoundaries<LayerBoundaryWaves<A>>,
    layer_admittances: Vec<A>,
    exterior_waves: ExteriorBoundaryWaves<A>,
    left_exterior_admittance: A,
    right_exterior_admittance: A,
) -> Result<Interfaces<InterfaceWaveData<A>>, InterfaceProjectionError>
where
    A: Clone,
{
    let wave_count = layer_waves.len();
    let admittance_count = layer_admittances.len();

    if wave_count != admittance_count {
        return Err(InterfaceProjectionError::LayerDataCountMismatch {
            wave_count,
            admittance_count,
        });
    }

    let (left_exterior_waves, right_exterior_waves) = exterior_waves.into_parts();

    let left_exterior = InterfaceSide::new(left_exterior_waves, left_exterior_admittance);

    let right_exterior = InterfaceSide::new(right_exterior_waves, right_exterior_admittance);

    let mut layers = layer_waves.into_inner().into_iter().zip(layer_admittances);

    let Some((first_waves, first_admittance)) = layers.next() else {
        return Ok(Interfaces::new(vec![InterfaceWaveData::new(
            left_exterior,
            right_exterior,
        )]));
    };

    let (first_left_waves, first_right_waves) = first_waves.into_parts();

    let mut interfaces = Vec::with_capacity(wave_count + 1);

    interfaces.push(InterfaceWaveData::new(
        left_exterior,
        InterfaceSide::new(first_left_waves, first_admittance.clone()),
    ));

    let mut previous_right = InterfaceSide::new(first_right_waves, first_admittance);

    for (layer_waves, layer_admittance) in layers {
        let (left_waves, right_waves) = layer_waves.into_parts();

        let current_left = InterfaceSide::new(left_waves, layer_admittance.clone());

        interfaces.push(InterfaceWaveData::new(previous_right, current_left));

        previous_right = InterfaceSide::new(right_waves, layer_admittance);
    }

    interfaces.push(InterfaceWaveData::new(previous_right, right_exterior));

    Ok(Interfaces::new(interfaces))
}

/// Construct the directional waves in both exterior media.
///
/// With global forward direction defined as left-to-right:
///
/// - left incidence gives `(1, r)` on the left and `(t, 0)` on the right;
/// - right incidence gives `(0, t)` on the left and `(r, 1)` on the right.
pub(crate) fn exterior_boundary_waves<A>(
    amplitudes: &PlaneWaveAmplitudes<A>,
    incident_side: IncidentSide,
    source: &ArrayBase<OwnedRepr<A::Scalar>, A::Dimension>,
) -> ExteriorBoundaryWaves<A>
where
    A: ScalarAlgebra + Clone,
    A::Scalar: One + Zero,
    A::Dimension: Dimension,
{
    let zero = A::filled_constant_like(source, <A::Scalar as Zero>::zero());

    let one = A::filled_constant_like(source, <A::Scalar as One>::one());

    match incident_side {
        IncidentSide::Left => ExteriorBoundaryWaves::new(
            BoundaryWaves::new(one, amplitudes.reflection().clone()),
            BoundaryWaves::new(amplitudes.transmission().clone(), zero),
        ),

        IncidentSide::Right => ExteriorBoundaryWaves::new(
            BoundaryWaves::new(zero, amplitudes.transmission().clone()),
            BoundaryWaves::new(amplitudes.reflection().clone(), one),
        ),
    }
}

pub(crate) fn exterior_boundary_states<A>(
    amplitudes: &PlaneWaveAmplitudes<A>,
    incident_side: IncidentSide,
    left_admittance: &A,
    right_admittance: &A,
) -> ExteriorBoundaryStates<A>
where
    A: ScalarAlgebra + Clone,
    A::Scalar: ComplexScalar + One + Zero,
    A::Dimension: Dimension,
{
    let (left_waves, right_waves) =
        exterior_boundary_waves(amplitudes, incident_side, left_admittance.value()).into_parts();

    ExteriorBoundaryStates {
        left: left_waves.into_state(left_admittance),
        right: right_waves.into_state(right_admittance),
    }
}

/// Collect the retained characteristic admittance of every finite layer.
///
/// Admittances are returned in physical left-to-right layer order.
pub(crate) fn project_layer_admittances<W>(
    workspace: &W,
) -> Result<Vec<W::Algebra>, BoundaryProjectionError>
where
    W: RetainedIsotropicLayers,
    W::Algebra: ScalarAlgebra,
{
    let layer_count = workspace
        .retained_layer_count()
        .ok_or(BoundaryProjectionError::LayersNotRetained)?;

    let mut admittances = Vec::with_capacity(layer_count);

    for index in 0..layer_count {
        let quantities =
            workspace
                .layer_quantities(index)
                .ok_or(BoundaryProjectionError::LayerOutOfBounds {
                    requested: index,
                    layer_count,
                })?;

        admittances.push(quantities.clone().into_admittance().into_inner());
    }

    Ok(admittances)
}

#[cfg(test)]
mod tests {
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use super::*;
    use crate::{
        algebra::{ArrayJet0, Jet0, RealParameter},
        observable::{BoundaryWaves, LayerBoundaries, LayerBoundaryWaves, PlaneWaveAmplitudes},
    };

    type C = Complex64;
    type A = ArrayJet0<C, Ix0, RealParameter>;

    fn jet(real: f64) -> A {
        Jet0::new(arr0(C::new(real, 0.0)))
    }

    fn boundary_waves(offset: f64) -> BoundaryWaves<A> {
        BoundaryWaves::new(jet(offset + 1.0), jet(offset + 2.0))
    }

    fn layer_waves(left_offset: f64, right_offset: f64) -> LayerBoundaryWaves<A> {
        LayerBoundaryWaves::new(boundary_waves(left_offset), boundary_waves(right_offset))
    }

    fn exterior_waves() -> ExteriorBoundaryWaves<A> {
        ExteriorBoundaryWaves::new(boundary_waves(1_000.0), boundary_waves(2_000.0))
    }

    fn scalar(value: &A) -> f64 {
        value.value()[()].re
    }

    fn value(value: &A) -> C {
        value.value()[()]
    }

    fn assert_complex_eq(actual: &A, expected: C) {
        assert_eq!(value(actual), expected);
    }

    fn assert_side(side: &InterfaceSide<A>, forward: C, backward: C, admittance: C) {
        assert_complex_eq(side.waves().forward(), forward);
        assert_complex_eq(side.waves().backward(), backward);
        assert_complex_eq(side.admittance(), admittance);

        let expected_field = forward + backward;
        let expected_secondary = -C::i() * admittance * (backward - forward);

        let state = side.state();

        assert_complex_eq(state.field(), expected_field);
        assert_complex_eq(state.secondary(), expected_secondary);
    }

    #[test]
    fn exterior_boundary_waves_use_left_incidence_convention() {
        let amplitudes = PlaneWaveAmplitudes::new(jet(0.25), jet(0.75));

        let source = jet(0.0);

        let waves = exterior_boundary_waves(&amplitudes, IncidentSide::Left, source.value());

        assert_eq!(scalar(waves.left().forward()), 1.0);
        assert_eq!(scalar(waves.left().backward()), 0.25);

        assert_eq!(scalar(waves.right().forward()), 0.75);
        assert_eq!(scalar(waves.right().backward()), 0.0);
    }

    #[test]
    fn exterior_boundary_waves_use_right_incidence_convention() {
        let amplitudes = PlaneWaveAmplitudes::new(jet(0.25), jet(0.75));

        let source = jet(0.0);

        let waves = exterior_boundary_waves(&amplitudes, IncidentSide::Right, source.value());

        assert_eq!(scalar(waves.left().forward()), 0.0);
        assert_eq!(scalar(waves.left().backward()), 0.75);

        assert_eq!(scalar(waves.right().forward()), 0.25);
        assert_eq!(scalar(waves.right().backward()), 1.0);
    }

    #[test]
    fn inconsistent_layer_counts_are_rejected() {
        let error = assemble_interface_wave_data(
            LayerBoundaries::new(vec![layer_waves(0.0, 10.0), layer_waves(20.0, 30.0)]),
            vec![jet(2.0)],
            exterior_waves(),
            jet(1.0),
            jet(4.0),
        )
        .expect_err("inconsistent layer data should be rejected");

        assert_eq!(
            error,
            InterfaceProjectionError::LayerDataCountMismatch {
                wave_count: 2,
                admittance_count: 1,
            },
        );
    }

    #[test]
    fn empty_finite_stack_produces_one_exterior_interface() {
        let interfaces = assemble_interface_wave_data(
            LayerBoundaries::new(Vec::new()),
            Vec::new(),
            exterior_waves(),
            jet(5.0),
            jet(6.0),
        )
        .expect("empty finite stack should be valid");

        assert_eq!(interfaces.len(), 1);

        let interface = interfaces
            .first()
            .expect("one exterior interface should exist");

        assert_eq!(scalar(interface.left().waves().forward()), 1_001.0,);

        assert_eq!(scalar(interface.left().waves().backward()), 1_002.0,);

        assert_eq!(scalar(interface.left().admittance()), 5.0,);

        assert_eq!(scalar(interface.right().waves().forward()), 2_001.0,);

        assert_eq!(scalar(interface.right().waves().backward()), 2_002.0,);

        assert_eq!(scalar(interface.right().admittance()), 6.0,);
    }

    #[test]
    fn one_layer_produces_two_interfaces() {
        let interfaces = assemble_interface_wave_data(
            LayerBoundaries::new(vec![layer_waves(0.0, 10.0)]),
            vec![jet(7.0)],
            exterior_waves(),
            jet(5.0),
            jet(6.0),
        )
        .expect("consistent one-layer data should assemble");

        assert_eq!(interfaces.len(), 2);

        let first = interfaces.get(0).unwrap();
        let second = interfaces.get(1).unwrap();

        assert_side(
            first.left(),
            C::new(1_001.0, 0.0),
            C::new(1_002.0, 0.0),
            C::new(5.0, 0.0),
        );

        assert_side(
            first.right(),
            C::new(1.0, 0.0),
            C::new(2.0, 0.0),
            C::new(7.0, 0.0),
        );

        assert_side(
            second.left(),
            C::new(11.0, 0.0),
            C::new(12.0, 0.0),
            C::new(7.0, 0.0),
        );

        assert_side(
            second.right(),
            C::new(2_001.0, 0.0),
            C::new(2_002.0, 0.0),
            C::new(6.0, 0.0),
        );
    }

    #[test]
    fn two_layers_produce_three_interfaces_in_physical_order() {
        let interfaces = assemble_interface_wave_data(
            LayerBoundaries::new(vec![layer_waves(0.0, 10.0), layer_waves(20.0, 30.0)]),
            vec![jet(7.0), jet(8.0)],
            exterior_waves(),
            jet(5.0),
            jet(6.0),
        )
        .expect("consistent two-layer data should assemble");

        assert_eq!(interfaces.len(), 3);

        let first = interfaces.get(0).unwrap();
        let internal = interfaces.get(1).unwrap();
        let final_interface = interfaces.get(2).unwrap();

        // Left exterior | layer 0.
        assert_side(
            first.right(),
            C::new(1.0, 0.0),
            C::new(2.0, 0.0),
            C::new(7.0, 0.0),
        );

        // Layer 0 right boundary.
        assert_side(
            internal.left(),
            C::new(11.0, 0.0),
            C::new(12.0, 0.0),
            C::new(7.0, 0.0),
        );

        // Layer 1 left boundary.
        assert_side(
            internal.right(),
            C::new(21.0, 0.0),
            C::new(22.0, 0.0),
            C::new(8.0, 0.0),
        );

        // Layer 1 right boundary.
        assert_side(
            final_interface.left(),
            C::new(31.0, 0.0),
            C::new(32.0, 0.0),
            C::new(8.0, 0.0),
        );

        // Layer 1 | right exterior.
        assert_side(
            final_interface.right(),
            C::new(2_001.0, 0.0),
            C::new(2_002.0, 0.0),
            C::new(6.0, 0.0),
        );
    }

    #[test]
    fn assembled_exterior_states_are_consistent_with_waves_and_admittance() {
        let interfaces = assemble_interface_wave_data(
            LayerBoundaries::new(Vec::new()),
            Vec::new(),
            exterior_waves(),
            jet(5.0),
            jet(6.0),
        )
        .unwrap();

        let interface = interfaces.first().unwrap();

        let expected_left = interface
            .left()
            .waves()
            .clone()
            .into_state(interface.left().admittance());

        let expected_right = interface
            .right()
            .waves()
            .clone()
            .into_state(interface.right().admittance());

        assert_eq!(interface.left().state(), expected_left);
        assert_eq!(interface.right().state(), expected_right);
    }
}

#[cfg(test)]
mod exterior_boundary_state_tests {
    use approx::assert_relative_eq;
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        algebra::{ArrayJet0, Jet0, RealParameter},
        observable::{BoundaryState, LayerBoundaries, LayerBoundaryStates, PlaneWaveAmplitudes},
    };

    type C = Complex64;
    type A = ArrayJet0<C, Ix0, RealParameter>;

    const TOLERANCE: f64 = 1.0e-12;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn jet(value: C) -> A {
        Jet0::new(arr0(value))
    }

    fn assert_complex_close(actual: C, expected: C) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    fn assert_jet_close(actual: &A, expected: C) {
        assert_complex_close(actual.value()[()], expected);
    }

    fn assert_state_close(actual: &BoundaryState<A>, expected_field: C, expected_secondary: C) {
        assert_jet_close(actual.field(), expected_field);
        assert_jet_close(actual.secondary(), expected_secondary);
    }

    #[test]
    fn left_incidence_uses_incident_and_reflected_waves_on_left() {
        let reflection = c(-0.2, 0.3);
        let transmission = c(0.7, -0.1);

        let amplitudes = PlaneWaveAmplitudes::new(jet(reflection), jet(transmission));

        let left_admittance = c(2.0, 0.0);
        let right_admittance = c(3.0, 0.0);

        let states = exterior_boundary_states(
            &amplitudes,
            IncidentSide::Left,
            &jet(left_admittance),
            &jet(right_admittance),
        );

        let left_xi = -C::i() * left_admittance;

        assert_state_close(
            &states.left,
            C::new(1.0, 0.0) + reflection,
            left_xi * (reflection - C::new(1.0, 0.0)),
        );
    }

    #[test]
    fn left_incidence_uses_only_transmitted_wave_on_right() {
        let reflection = c(-0.2, 0.3);
        let transmission = c(0.7, -0.1);

        let amplitudes = PlaneWaveAmplitudes::new(jet(reflection), jet(transmission));

        let left_admittance = c(2.0, 0.0);
        let right_admittance = c(3.0, 0.0);

        let states = exterior_boundary_states(
            &amplitudes,
            IncidentSide::Left,
            &jet(left_admittance),
            &jet(right_admittance),
        );

        let right_xi = -C::i() * right_admittance;

        assert_state_close(&states.right, transmission, -right_xi * transmission);
    }

    #[test]
    fn right_incidence_uses_only_transmitted_wave_on_left() {
        let reflection = c(0.15, -0.25);
        let transmission = c(0.6, 0.2);

        let amplitudes = PlaneWaveAmplitudes::new(jet(reflection), jet(transmission));

        let left_admittance = c(2.0, 0.0);
        let right_admittance = c(3.0, 0.0);

        let states = exterior_boundary_states(
            &amplitudes,
            IncidentSide::Right,
            &jet(left_admittance),
            &jet(right_admittance),
        );

        let left_xi = -C::i() * left_admittance;

        assert_state_close(&states.left, transmission, left_xi * transmission);
    }

    #[test]
    fn right_incidence_uses_incident_and_reflected_waves_on_right() {
        let reflection = c(0.15, -0.25);
        let transmission = c(0.6, 0.2);

        let amplitudes = PlaneWaveAmplitudes::new(jet(reflection), jet(transmission));

        let left_admittance = c(2.0, 0.0);
        let right_admittance = c(3.0, 0.0);

        let states = exterior_boundary_states(
            &amplitudes,
            IncidentSide::Right,
            &jet(left_admittance),
            &jet(right_admittance),
        );

        let right_xi = -C::i() * right_admittance;

        assert_state_close(
            &states.right,
            C::new(1.0, 0.0) + reflection,
            right_xi * (C::new(1.0, 0.0) - reflection),
        );
    }

    #[test]
    fn two_finite_layers_produce_three_interfaces_in_order() {
        let layers = LayerBoundaries::new(vec![
            LayerBoundaryStates::new(BoundaryState::new(10, 11), BoundaryState::new(12, 13)),
            LayerBoundaryStates::new(BoundaryState::new(20, 21), BoundaryState::new(22, 23)),
        ]);

        let interfaces =
            assemble_interface_states(layers, BoundaryState::new(0, 1), BoundaryState::new(30, 31));

        assert_eq!(interfaces.len(), 3);

        let actual: Vec<_> = interfaces
            .iter()
            .map(|interface| {
                (
                    *interface.left().field(),
                    *interface.left().secondary(),
                    *interface.right().field(),
                    *interface.right().secondary(),
                )
            })
            .collect();

        assert_eq!(
            actual,
            vec![(0, 1, 10, 11), (12, 13, 20, 21), (22, 23, 30, 31),],
        );
    }

    #[test]
    fn empty_finite_stack_produces_one_exterior_interface() {
        let interfaces = assemble_interface_states(
            LayerBoundaries::new(Vec::new()),
            BoundaryState::new(1, 2),
            BoundaryState::new(3, 4),
        );

        assert_eq!(interfaces.len(), 1);

        let interface = interfaces
            .first()
            .expect("one exterior interface should exist");

        assert_eq!(interface.left().clone().into_parts(), (1, 2));
        assert_eq!(interface.right().clone().into_parts(), (3, 4));
    }
}
