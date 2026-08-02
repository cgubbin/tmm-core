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
