use crate::backend::PlaneWaveEntries;

pub trait ProjectPlaneWaveModeDeterminant: PlaneWaveEntries {
    type Determinant;

    fn project_determinant(&self, exterior: &Self::ExteriorContext) -> Self::Determinant;
}

pub struct PlaneWaveDeterminant<J> {
    value: J,
}

impl<J> PlaneWaveDeterminant<J> {
    pub fn new(value: J) -> Self {
        Self { value }
    }

    pub fn value(&self) -> &J {
        &self.value
    }

    pub fn map<J2>(self, transform: impl Fn(J) -> J2) -> PlaneWaveDeterminant<J2> {
        PlaneWaveDeterminant {
            value: transform(self.value),
        }
    }

    pub fn into_inner(self) -> J {
        self.value
    }
}

#[cfg(test)]
mod projection_tests {
    use crate::{
        algebra::ScalarAlgebra,
        backend::scatter2::{Scatter2ExteriorContext, Scatter2ProjectiveEntries},
        test_support::{
            C, TOLERANCE,
            assertions::assert_complex_close,
            jet::{J0, zero_jet_from_value},
        },
    };

    use super::*;

    use ndarray::Ix0;

    type Algebra = J0;

    fn scalar(value: impl Into<C>) -> Algebra {
        zero_jet_from_value(value.into())
    }

    fn value<J>(jet: &J) -> J::Scalar
    where
        J: ScalarAlgebra<Dimension = Ix0>,
        J::Scalar: Copy,
    {
        jet.value()[()]
    }

    fn entries(
        s11: impl Into<C>,
        s12: impl Into<C>,
        s21: impl Into<C>,
        s22: impl Into<C>,
    ) -> Scatter2ProjectiveEntries<Algebra> {
        Scatter2ProjectiveEntries::from_parts(
            scalar(C::new(1.0, 0.0)),
            scalar(s11),
            scalar(s12),
            scalar(s21),
            scalar(s22),
        )
    }

    fn exterior_context(
        left_admittance: impl Into<C>,
        right_admittance: impl Into<C>,
    ) -> Scatter2ExteriorContext<Algebra> {
        Scatter2ExteriorContext::from_parts(
            scalar(left_admittance),
            scalar(right_admittance),
            scalar(0.0),
            scalar(0.0),
            scalar(0.0),
            scalar(0.0),
            crate::Polarisation::TransverseElectric,
        )
    }

    #[test]
    fn project_determinant_matches_scattering_residual() {
        let entries = entries(
            1.0, 2.0, 4.0, // s21
            3.0,
        );

        let exterior = exterior_context(
            6.0,   // left admittance
            100.0, // deliberately unrelated right admittance
        );

        let determinant = entries.project_determinant(&exterior);

        // transfer_state_slope(Y) = -iY
        //
        // D = 2(-iY_left) / s21
        //   = 2(-i * 6) / 4
        //   = -3i
        assert_complex_close(value(determinant.value()), C::new(0.0, -3.0), TOLERANCE);
    }

    #[test]
    fn project_determinant_uses_complex_left_admittance() {
        let entries = entries(0.0, 0.0, C::new(2.0, -1.0), 0.0);

        let left_admittance = C::new(3.0, 4.0);
        let exterior = exterior_context(left_admittance, 50.0);

        let determinant = entries.project_determinant(&exterior);

        let expected = 2.0 * (-C::i() * left_admittance) / C::new(2.0, -1.0);

        assert_complex_close(value(determinant.value()), expected, TOLERANCE);
    }

    #[test]
    fn project_determinant_is_independent_of_right_context_for_fixed_entries() {
        let entries = entries(1.0, 2.0, C::new(3.0, -0.5), 4.0);

        let first_context = exterior_context(C::new(2.0, 0.25), C::new(5.0, 1.0));

        let second_context = exterior_context(C::new(2.0, 0.25), C::new(100.0, -30.0));

        let first = entries.project_determinant(&first_context);
        let second = entries.project_determinant(&second_context);

        assert_complex_close(value(first.value()), value(second.value()), TOLERANCE);
    }
}
