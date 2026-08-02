use num_traits::{One, Zero};

use crate::{
    ComplexScalar, IncidentSide,
    algebra::ScalarAlgebra,
    observable::{BoundaryState, BoundaryWaves, PlaneWaveAmplitudes},
};

use ndarray::Dimension;

pub(crate) struct ExteriorBoundaryStates<A> {
    pub(crate) left: BoundaryState<A>,
    pub(crate) right: BoundaryState<A>,
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
    let zero = A::filled_constant_like(left_admittance.value(), <A::Scalar as Zero>::zero());

    let one = A::filled_constant_like(left_admittance.value(), <A::Scalar as One>::one());

    let (left_waves, right_waves) = match incident_side {
        IncidentSide::Left => (
            BoundaryWaves::new(one, amplitudes.reflection().clone()),
            BoundaryWaves::new(amplitudes.transmission().clone(), zero),
        ),

        IncidentSide::Right => (
            BoundaryWaves::new(zero, amplitudes.transmission().clone()),
            BoundaryWaves::new(amplitudes.reflection().clone(), one),
        ),
    };

    ExteriorBoundaryStates {
        left: left_waves.into_state(left_admittance),
        right: right_waves.into_state(right_admittance),
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        algebra::{ArrayJet0, Jet0, RealParameter},
        observable::{
            LayerBoundaries, LayerBoundaryStates, PlaneWaveAmplitudes, assemble_interface_states,
        },
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
