use ndarray::Dimension;

use crate::{
    ComplexScalar, Polarisation,
    algebra::ScalarAlgebra,
    backend::{IsotropicLayerQuantities, isotropic::IsotropicMediumQuantities},
    input::CanonicalCoordinates,
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};

/// Canonical longitudinal angular wavevectors in the exterior media.
///
/// `left` and `right` contain the branch-selected out-of-plane angular
/// wavevectors `κ` in the corresponding semi-infinite exterior media.
///
/// Branch selection is completed before this value reaches a numerical
/// backend. This allows advanced callers, such as complex-frequency mode
/// solvers, to supply branch choices determined by an external conformal map
/// rather than relying on the backend to reconstruct them.
#[derive(Clone, Debug, PartialEq)]
pub struct ExteriorWavevectors<A> {
    left: A,
    right: A,
}

impl<A> ExteriorWavevectors<A> {
    /// Construct exterior longitudinal wavevectors.
    pub fn new(left: A, right: A) -> Self {
        Self { left, right }
    }

    /// Return the longitudinal wavevector in the left exterior medium.
    pub fn left(&self) -> &A {
        &self.left
    }

    /// Return the longitudinal wavevector in the right exterior medium.
    pub fn right(&self) -> &A {
        &self.right
    }

    /// Consume the container into `(left, right)`.
    pub(crate) fn into_parts(self) -> (A, A) {
        (self.left, self.right)
    }
}

pub(crate) fn evaluate_exterior_wavevectors<E, M, J>(
    coordinates: &CanonicalCoordinates<J>,
    left_exterior: &M,
    right_exterior: &M,
) -> ExteriorWavevectors<J>
where
    J: ScalarAlgebra + ConstitutiveLift<E, M> + Clone,
    J::Scalar: ComplexScalar,
    J::Dimension: Dimension,
    E: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
{
    let left_kappa =
        IsotropicMediumQuantities::evaluate::<E, M>(left_exterior, coordinates).into_kappa();

    let right_kappa =
        IsotropicMediumQuantities::evaluate::<E, M>(right_exterior, coordinates).into_kappa();

    ExteriorWavevectors::new(left_kappa, right_kappa)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        Polarisation,
        algebra::{ArrayJet0, RealParameter},
        domain::RealAxis,
        input::CanonicalCoordinates,
        material::Constant,
    };

    type C = Complex64;
    type J = ArrayJet0<C, Ix0, RealParameter>;

    const TOLERANCE: f64 = 1.0e-12;

    fn jet(value: C) -> J {
        J::new(arr0(value))
    }

    fn coordinates(
        vacuum_angular_wavenumber: C,
        parallel_angular_wavenumber: C,
    ) -> CanonicalCoordinates<J> {
        CanonicalCoordinates::new(
            jet(vacuum_angular_wavenumber),
            jet(parallel_angular_wavenumber),
        )
    }

    fn assert_jet_close(actual: &J, expected: &J) {
        let actual = actual.value()[()];
        let expected = expected.value()[()];

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

    #[test]
    fn exterior_wavevectors_match_layer_quantities() {
        let coordinates = coordinates(C::new(2.5, 0.0), C::new(0.7, 0.0));

        let left = Constant::new(2.0, 1.0);
        let right = Constant::new(4.0, 1.5);

        let exterior = evaluate_exterior_wavevectors::<RealAxis, _, J>(&coordinates, &left, &right);

        let left_quantities = IsotropicLayerQuantities::evaluate::<RealAxis, _>(
            &left,
            &coordinates,
            Polarisation::TransverseElectric,
        );

        let right_quantities = IsotropicLayerQuantities::evaluate::<RealAxis, _>(
            &right,
            &coordinates,
            Polarisation::TransverseElectric,
        );

        assert_jet_close(exterior.left(), left_quantities.kappa());

        assert_jet_close(exterior.right(), right_quantities.kappa());
    }

    #[test]
    fn exterior_wavevectors_preserve_left_and_right_materials() {
        let coordinates = coordinates(C::new(2.5, 0.0), C::new(0.3, 0.0));

        let left = Constant::new(1.0, 1.0);
        let right = Constant::new(9.0, 1.0);

        let exterior = evaluate_exterior_wavevectors::<RealAxis, _, J>(&coordinates, &left, &right);

        let left_expected = IsotropicLayerQuantities::evaluate::<RealAxis, _>(
            &left,
            &coordinates,
            Polarisation::TransverseElectric,
        );

        let right_expected = IsotropicLayerQuantities::evaluate::<RealAxis, _>(
            &right,
            &coordinates,
            Polarisation::TransverseElectric,
        );

        assert_jet_close(exterior.left(), left_expected.kappa());

        assert_jet_close(exterior.right(), right_expected.kappa());

        assert_ne!(exterior.left().value()[()], exterior.right().value()[()],);
    }

    #[test]
    fn exterior_wavevectors_are_independent_of_polarisation() {
        let coordinates = coordinates(C::new(2.5, 0.0), C::new(0.7, 0.0));

        let left = Constant::new(2.0, 3.0);
        let right = Constant::new(4.0, 5.0);

        let te = evaluate_exterior_wavevectors::<RealAxis, _, J>(&coordinates, &left, &right);

        let tm = evaluate_exterior_wavevectors::<RealAxis, _, J>(&coordinates, &left, &right);

        assert_jet_close(te.left(), tm.left());
        assert_jet_close(te.right(), tm.right());
    }
}
