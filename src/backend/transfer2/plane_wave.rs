use ndarray::{ArrayBase, Dimension, OwnedRepr};
use num_traits::Float;

use crate::{
    ComplexScalar,
    backend::{
        PlaneWaveBackend, PlaneWaveInput, PlaneWaveResponse,
        algebra::ScalarAlgebra,
        derivative::StructuralDerivativeVariable,
        evaluator::RealAxis,
        input::IncidentSide,
        isotropic::IsotropicLayerAdmittance,
        jet::{ArrayJet, ArrayJetFirst},
        plane_wave::DifferentiablePlaneWaveBackend,
        transfer2::{
            jet::{Transfer2Jet, Transfer2JetFirst},
            response::Matrix2Entries,
        },
    },
    material::{EvaluateDifferentiableMaterial, EvaluateMaterial},
    stack::Stack,
};

use super::{Matrix2, Transfer2, Transfer2Error};

pub(super) fn amplitudes<C, D, A>(
    matrix: Matrix2Entries<A>,
    left_admittance: &A,
    right_admittance: &A,
    incident_side: IncidentSide,
) -> (A, A)
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D>,
{
    let left_slope = boundary_slope::<C, D, A>(left_admittance);

    let right_slope = boundary_slope::<C, D, A>(right_admittance);

    let terms = matrix.boundary_terms(&left_slope, &right_slope);

    let two = A::constant_like(matrix.m11.value(), C::one() + C::one());

    match incident_side {
        IncidentSide::Left => {
            let reflection = left_slope
                .multiply(&terms.u)
                .add(&terms.v)
                .divide(&terms.denominator);

            let transmission = two.multiply(&left_slope).divide(&terms.denominator);

            (reflection, transmission)
        }

        IncidentSide::Right => {
            let p = matrix.m11.add(&terms.b_right);

            let q = matrix.m21.add(&terms.d_right);

            let reflection = q
                .subtract(&left_slope.multiply(&p))
                .divide(&terms.denominator);

            let determinant = matrix
                .m11
                .multiply(&matrix.m22)
                .subtract(&matrix.m12.multiply(&matrix.m21));

            let transmission = two
                .multiply(&right_slope)
                .multiply(&determinant)
                .divide(&terms.denominator);

            (reflection, transmission)
        }
    }
}

impl<C, D> Matrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(super) fn amplitudes(
        self,
        left_admittance: &ArrayBase<OwnedRepr<C>, D>,
        right_admittance: &ArrayBase<OwnedRepr<C>, D>,
        incident_side: IncidentSide,
    ) -> (ArrayBase<OwnedRepr<C>, D>, ArrayBase<OwnedRepr<C>, D>) {
        let (m11, m12, m21, m22) = self.into_parts();

        amplitudes(
            Matrix2Entries { m11, m12, m21, m22 },
            left_admittance,
            right_admittance,
            incident_side,
        )
    }
}

impl<C, D> Transfer2JetFirst<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(super) fn into_entries(self) -> Matrix2Entries<ArrayJetFirst<C, D>> {
        let (value, first) = self.into_parts();

        let (a, b, c, d) = value.into_parts();
        let (da, db, dc, dd) = first.into_parts();

        Matrix2Entries {
            m11: ArrayJetFirst::from_parts(a, da),
            m12: ArrayJetFirst::from_parts(b, db),
            m21: ArrayJetFirst::from_parts(c, dc),
            m22: ArrayJetFirst::from_parts(d, dd),
        }
    }

    pub(super) fn amplitude_jets(
        self,
        left_admittance: &ArrayJetFirst<C, D>,
        right_admittance: &ArrayJetFirst<C, D>,
        incident_side: IncidentSide,
    ) -> (ArrayJetFirst<C, D>, ArrayJetFirst<C, D>) {
        amplitudes(
            self.into_entries(),
            left_admittance,
            right_admittance,
            incident_side,
        )
    }
}

impl<C, D> Transfer2Jet<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    pub(super) fn into_entries(self) -> Matrix2Entries<ArrayJet<C, D>> {
        let (value, first, second) = self.into_parts();

        let (a, b, c, d) = value.into_parts();
        let (da, db, dc, dd) = first.into_parts();
        let (dda, ddb, ddc, ddd) = second.into_parts();

        Matrix2Entries {
            m11: ArrayJet::from_parts(a, da, dda),
            m12: ArrayJet::from_parts(b, db, ddb),
            m21: ArrayJet::from_parts(c, dc, ddc),
            m22: ArrayJet::from_parts(d, dd, ddd),
        }
    }

    pub(super) fn amplitude_jets(
        self,
        left_admittance: &ArrayJet<C, D>,
        right_admittance: &ArrayJet<C, D>,
        incident_side: IncidentSide,
    ) -> (ArrayJet<C, D>, ArrayJet<C, D>) {
        amplitudes(
            self.into_entries(),
            left_admittance,
            right_admittance,
            incident_side,
        )
    }
}

impl<C, D, M> PlaneWaveBackend<C, D, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
    M: EvaluateMaterial<C, Real = C::RealField>,
{
    type Error = Transfer2Error;

    fn solve_plane_wave(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input();

        let matrix = self.evaluate_with::<RealAxis, _, _, _>(stack, &planar)?;

        let left_admittance =
            IsotropicLayerAdmittance::evaluate_real_axis(stack.left_exterior(), &planar)
                .into_inner();

        let right_admittance =
            IsotropicLayerAdmittance::evaluate_real_axis(stack.right_exterior(), &planar)
                .into_inner();

        let (reflection, transmission) =
            matrix.amplitudes(&left_admittance, &right_admittance, input.incident_side());

        let (incident_normalisation, transmitted_normalisation) = match input.incident_side() {
            IncidentSide::Left => (left_admittance, right_admittance),

            IncidentSide::Right => (right_admittance, left_admittance),
        };

        Ok(PlaneWaveResponse::from_values(
            reflection,
            transmission,
            incident_normalisation,
            transmitted_normalisation,
        ))
    }

    fn solve_plane_wave_structural_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input();

        let matrix =
            self.evaluate_structural_first_with::<RealAxis, _, _, _>(stack, &planar, variable)?;

        let left_admittance = IsotropicLayerAdmittance::evaluate_first_structural_real_axis(
            stack.left_exterior(),
            &planar,
            variable,
        );

        let right_admittance = IsotropicLayerAdmittance::evaluate_first_structural_real_axis(
            stack.right_exterior(),
            &planar,
            variable,
        );

        let (reflection, transmission) =
            matrix.amplitude_jets(&left_admittance, &right_admittance, input.incident_side());

        let (incident_normalisation, transmitted_normalisation) = match input.incident_side() {
            IncidentSide::Left => (left_admittance, right_admittance),

            IncidentSide::Right => (right_admittance, left_admittance),
        };

        Ok(PlaneWaveResponse::from_first_jets(
            reflection,
            transmission,
            incident_normalisation,
            transmitted_normalisation,
            variable.into(),
        ))
    }

    fn solve_plane_wave_structural_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<<C>::RealField>, D>>,
        variable: StructuralDerivativeVariable,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input();

        let matrix =
            self.evaluate_structural_second_with::<RealAxis, _, _, _>(stack, &planar, variable)?;

        let left_admittance = IsotropicLayerAdmittance::evaluate_second_structural_real_axis(
            stack.left_exterior(),
            &planar,
            variable,
        );

        let right_admittance = IsotropicLayerAdmittance::evaluate_second_structural_real_axis(
            stack.right_exterior(),
            &planar,
            variable,
        );

        let (reflection, transmission) =
            matrix.amplitude_jets(&left_admittance, &right_admittance, input.incident_side());

        let (incident_normalisation, transmitted_normalisation) = match input.incident_side() {
            IncidentSide::Left => (left_admittance, right_admittance),

            IncidentSide::Right => (right_admittance, left_admittance),
        };

        Ok(PlaneWaveResponse::from_second_jets(
            reflection,
            transmission,
            incident_normalisation,
            transmitted_normalisation,
            variable.into(),
        ))
    }
}

impl<C, D, M> DifferentiablePlaneWaveBackend<C, D, Stack<M, C::RealField>> for Transfer2
where
    C: ComplexScalar,
    C::RealField: Copy + Float,
    D: Dimension,
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
{
    fn solve_plane_wave_spectral_first_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<<C>::RealField>, D>>,
        variable: crate::backend::derivative::SpectralDerivativeVariable,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input();

        let matrix =
            self.evaluate_spectral_first_with::<RealAxis, _, _, _>(stack, &planar, variable)?;

        let left_admittance = IsotropicLayerAdmittance::evaluate_first_spectral_real_axis(
            stack.left_exterior(),
            &planar,
            variable,
        );

        let right_admittance = IsotropicLayerAdmittance::evaluate_first_spectral_real_axis(
            stack.right_exterior(),
            &planar,
            variable,
        );

        let (reflection, transmission) =
            matrix.amplitude_jets(&left_admittance, &right_admittance, input.incident_side());

        let (incident_normalisation, transmitted_normalisation) = match input.incident_side() {
            IncidentSide::Left => (left_admittance, right_admittance),

            IncidentSide::Right => (right_admittance, left_admittance),
        };

        Ok(PlaneWaveResponse::from_first_jets(
            reflection,
            transmission,
            incident_normalisation,
            transmitted_normalisation,
            variable.into(),
        ))
    }

    fn solve_plane_wave_spectral_second_derivative(
        &self,
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<<C>::RealField>, D>>,
        variable: crate::backend::derivative::SpectralDerivativeVariable,
    ) -> Result<PlaneWaveResponse<C, D>, Self::Error> {
        let planar = input.complex_planar_input();

        let matrix =
            self.evaluate_spectral_second_with::<RealAxis, _, _, _>(stack, &planar, variable)?;

        let left_admittance = IsotropicLayerAdmittance::evaluate_second_spectral_real_axis(
            stack.left_exterior(),
            &planar,
            variable,
        );

        let right_admittance = IsotropicLayerAdmittance::evaluate_second_spectral_real_axis(
            stack.right_exterior(),
            &planar,
            variable,
        );

        let (reflection, transmission) =
            matrix.amplitude_jets(&left_admittance, &right_admittance, input.incident_side());

        let (incident_normalisation, transmitted_normalisation) = match input.incident_side() {
            IncidentSide::Left => (left_admittance, right_admittance),

            IncidentSide::Right => (right_admittance, left_admittance),
        };

        Ok(PlaneWaveResponse::from_second_jets(
            reflection,
            transmission,
            incident_normalisation,
            transmitted_normalisation,
            variable.into(),
        ))
    }
}
/// Convert a physical characteristic admittance into the field-state slope
/// used by the transfer matrix.
///
/// For the matrix convention
///
/// ```text
/// M = [ cos(κd)    -sin(κd)/Y ]
///     [ Y sin(κd)   cos(κd)   ],
/// ```
///
/// travelling-wave states have derivative components `±iY`.
pub(super) fn boundary_slope<C, D, A>(admittance: &A) -> A
where
    C: ComplexScalar,
    D: Dimension,
    A: ScalarAlgebra<C, D>,
{
    // TODO: Should be minus for outgoing
    let imaginary_unit = A::constant_like(admittance.value(), -C::i());

    imaginary_unit.multiply(admittance)
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Array0, Ix0, arr0, array};
    use num_complex::Complex64;

    use super::*;
    use crate::backend::jet::{ArrayJet, ArrayJetFirst};

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn scalar_matrix(m11: f64, m12: f64, m21: f64, m22: f64) -> Matrix2<C, Ix0> {
        Matrix2::new(arr0(c(m11)), arr0(c(m12)), arr0(c(m21)), arr0(c(m22)))
    }

    fn zero_matrix() -> Matrix2<C, Ix0> {
        scalar_matrix(0.0, 0.0, 0.0, 0.0)
    }

    fn identity_matrix() -> Matrix2<C, Ix0> {
        scalar_matrix(1.0, 0.0, 0.0, 1.0)
    }

    fn assert_complex_close(actual: C, expected: C, tolerance: f64) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = tolerance,
            max_relative = tolerance,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = tolerance,
            max_relative = tolerance,
        );
    }

    fn assert_scalar_array_close(actual: &Array0<C>, expected: C, tolerance: f64) {
        assert_complex_close(actual[()], expected, tolerance);
    }

    fn value_amplitudes(
        matrix: Matrix2<C, Ix0>,
        left_admittance: f64,
        right_admittance: f64,
        incident_side: IncidentSide,
    ) -> (C, C) {
        let (reflection, transmission) = matrix.amplitudes(
            &arr0(c(left_admittance)),
            &arr0(c(right_admittance)),
            incident_side,
        );

        (reflection[()], transmission[()])
    }

    #[test]
    fn array_amplitude_algebra_adds_elementwise() {
        let left = array![c(1.0), c(2.0), c(3.0)];
        let right = array![c(4.0), c(5.0), c(6.0)];

        let result = <_ as ScalarAlgebra<C, ndarray::Ix1>>::add(&left, &right);

        assert_eq!(result, array![c(5.0), c(7.0), c(9.0)],);
    }

    #[test]
    fn array_amplitude_algebra_subtracts_elementwise() {
        let left = array![c(5.0), c(7.0), c(9.0)];
        let right = array![c(1.0), c(2.0), c(3.0)];

        let result = <_ as ScalarAlgebra<C, ndarray::Ix1>>::subtract(&left, &right);

        assert_eq!(result, array![c(4.0), c(5.0), c(6.0)],);
    }

    #[test]
    fn array_amplitude_algebra_multiplies_elementwise() {
        let left = array![c(2.0), c(3.0), c(4.0)];
        let right = array![c(5.0), c(7.0), c(11.0)];

        let result = <_ as ScalarAlgebra<C, ndarray::Ix1>>::multiply(&left, &right);

        assert_eq!(result, array![c(10.0), c(21.0), c(44.0)],);
    }

    #[test]
    fn array_amplitude_algebra_divides_elementwise() {
        let numerator = array![c(10.0), c(21.0), c(44.0)];
        let denominator = array![c(5.0), c(7.0), c(11.0)];

        let result = <_ as ScalarAlgebra<C, ndarray::Ix1>>::divide(&numerator, &denominator);

        assert_eq!(result, array![c(2.0), c(3.0), c(4.0)],);
    }

    #[test]
    fn array_amplitude_algebra_constructs_constant_with_source_shape() {
        let source = array![c(1.0), c(2.0), c(3.0)];

        let result: ArrayBase<OwnedRepr<C>, ndarray::Ix1> =
            <_ as ScalarAlgebra<C, ndarray::Ix1>>::constant_like(&source, c(7.0));

        assert_eq!(result, array![c(7.0), c(7.0), c(7.0)]);
        assert_eq!(result.raw_dim(), source.raw_dim());
    }

    #[test]
    fn identity_matrix_left_incidence_matches_fresnel_amplitudes() {
        let yl = 2.0;
        let yr = 3.0;

        let (reflection, transmission) =
            value_amplitudes(identity_matrix(), yl, yr, IncidentSide::Left);

        assert_complex_close(reflection, c((yl - yr) / (yl + yr)), 1e-12);

        assert_complex_close(transmission, c(2.0 * yl / (yl + yr)), 1e-12);
    }

    #[test]
    fn identity_matrix_right_incidence_matches_fresnel_amplitudes() {
        let yl = 2.0;
        let yr = 3.0;

        let (reflection, transmission) =
            value_amplitudes(identity_matrix(), yl, yr, IncidentSide::Right);

        assert_complex_close(reflection, c((yr - yl) / (yl + yr)), 1e-12);

        assert_complex_close(transmission, c(2.0 * yr / (yl + yr)), 1e-12);
    }

    #[test]
    fn equal_exterior_admittances_and_identity_matrix_are_transparent() {
        for incident_side in [IncidentSide::Left, IncidentSide::Right] {
            let (reflection, transmission) =
                value_amplitudes(identity_matrix(), 2.5, 2.5, incident_side);

            assert_complex_close(reflection, c(0.0), 1e-12);
            assert_complex_close(transmission, c(1.0), 1e-12);
        }
    }

    #[test]
    fn nontrivial_left_incidence_matches_direct_formula() {
        let a = 1.2;
        let b = 0.3;
        let c_entry = -0.4;
        let d = 0.8;

        let left_admittance = 1.7;
        let right_admittance = 2.1;
        let left_slope = -C::i() * left_admittance;
        let right_slope = -C::i() * right_admittance;

        let u = a - b * right_slope;
        let v = c_entry - d * right_slope;

        let denominator = left_slope * u - v;

        let expected_reflection = (left_slope * u + v) / denominator;

        let expected_transmission = c(2.0) * left_slope / denominator;

        let (reflection, transmission) = value_amplitudes(
            scalar_matrix(a, b, c_entry, d),
            left_admittance,
            right_admittance,
            IncidentSide::Left,
        );

        assert_complex_close(reflection, expected_reflection, 1e-12);

        assert_complex_close(transmission, expected_transmission, 1e-12);
    }

    #[test]
    fn nontrivial_right_incidence_matches_direct_formula() {
        let a = 1.2;
        let b = 0.3;
        let c_ = -0.4;
        let d = 0.8;

        let left_admittance = 1.7;
        let right_admittance = 2.1;

        let left_slope = -C::i() * left_admittance;
        let right_slope = -C::i() * right_admittance;

        let (reflection, transmission) = value_amplitudes(
            scalar_matrix(a, b, c_, d),
            left_admittance,
            right_admittance,
            IncidentSide::Right,
        );

        let u = a - b * right_slope;
        let v = c_ - d * right_slope;

        let p = a + b * right_slope;
        let q = c_ + d * right_slope;

        let denominator = left_slope * u - v;

        let expected_reflection = (q - left_slope * p) / denominator;

        let determinant = a * d - b * c_;

        let expected_transmission = c(2.0) * right_slope * determinant / denominator;

        assert_complex_close(reflection, expected_reflection, 1e-12);

        assert_complex_close(transmission, expected_transmission, 1e-12);
    }

    #[test]
    fn first_order_zero_derivatives_reproduce_value_path() {
        let matrix = scalar_matrix(1.2, 0.3, -0.4, 0.8);
        let matrix_for_value = matrix.clone();

        let matrix_jet = Transfer2JetFirst::from_parts(matrix, zero_matrix());

        let left = ArrayJetFirst::from_parts(arr0(c(1.7)), arr0(c(0.0)));

        let right = ArrayJetFirst::from_parts(arr0(c(2.1)), arr0(c(0.0)));

        let (jet_reflection, jet_transmission) =
            matrix_jet.amplitude_jets(&left, &right, IncidentSide::Left);

        let (value_reflection, value_transmission) =
            matrix_for_value.amplitudes(&arr0(c(1.7)), &arr0(c(2.1)), IncidentSide::Left);

        assert_complex_close(jet_reflection.value()[()], value_reflection[()], 1e-12);

        assert_complex_close(jet_transmission.value()[()], value_transmission[()], 1e-12);

        assert_scalar_array_close(jet_reflection.first(), c(0.0), 1e-12);

        assert_scalar_array_close(jet_transmission.first(), c(0.0), 1e-12);
    }

    #[test]
    fn second_order_zero_derivatives_reproduce_value_path() {
        let matrix = scalar_matrix(1.2, 0.3, -0.4, 0.8);
        let matrix_for_value = matrix.clone();

        let matrix_jet = Transfer2Jet::from_parts(matrix, zero_matrix(), zero_matrix());

        let left = ArrayJet::from_parts(arr0(c(1.7)), arr0(c(0.0)), arr0(c(0.0)));

        let right = ArrayJet::from_parts(arr0(c(2.1)), arr0(c(0.0)), arr0(c(0.0)));

        let (jet_reflection, jet_transmission) =
            matrix_jet.amplitude_jets(&left, &right, IncidentSide::Right);

        let (value_reflection, value_transmission) =
            matrix_for_value.amplitudes(&arr0(c(1.7)), &arr0(c(2.1)), IncidentSide::Right);

        assert_complex_close(jet_reflection.value()[()], value_reflection[()], 1e-12);

        assert_complex_close(jet_transmission.value()[()], value_transmission[()], 1e-12);

        assert_scalar_array_close(jet_reflection.first(), c(0.0), 1e-12);

        assert_scalar_array_close(jet_reflection.second(), c(0.0), 1e-12);

        assert_scalar_array_close(jet_transmission.first(), c(0.0), 1e-12);

        assert_scalar_array_close(jet_transmission.second(), c(0.0), 1e-12);
    }

    #[test]
    fn first_order_identity_fresnel_derivatives_include_both_admittances() {
        let yl = 2.0;
        let dyl = 0.4;

        let yr = 3.0;
        let dyr = -0.2;

        let matrix_jet = Transfer2JetFirst::from_parts(identity_matrix(), zero_matrix());

        let left = ArrayJetFirst::from_parts(arr0(c(yl)), arr0(c(dyl)));

        let right = ArrayJetFirst::from_parts(arr0(c(yr)), arr0(c(dyr)));

        let (reflection, transmission) =
            matrix_jet.amplitude_jets(&left, &right, IncidentSide::Left);

        let denominator = yl + yr;
        let ddenominator = dyl + dyr;

        let numerator_r = yl - yr;
        let dnumerator_r = dyl - dyr;

        let expected_dr =
            (dnumerator_r * denominator - numerator_r * ddenominator) / denominator.powi(2);

        let numerator_t = 2.0 * yl;
        let dnumerator_t = 2.0 * dyl;

        let expected_dt =
            (dnumerator_t * denominator - numerator_t * ddenominator) / denominator.powi(2);

        assert_scalar_array_close(reflection.first(), c(expected_dr), 1e-12);

        assert_scalar_array_close(transmission.first(), c(expected_dt), 1e-12);
    }

    #[test]
    fn first_order_amplitudes_match_finite_difference() {
        fn parameters(x: f64) -> (Matrix2<C, Ix0>, f64, f64) {
            let matrix = scalar_matrix(
                1.2 + 0.10 * x,
                0.3 - 0.04 * x,
                -0.4 + 0.07 * x,
                0.8 + 0.03 * x,
            );

            let yl = 1.7 + 0.05 * x;
            let yr = 2.1 - 0.08 * x;

            (matrix, yl, yr)
        }

        let (matrix, yl, yr) = parameters(0.0);

        let matrix_first = scalar_matrix(0.10, -0.04, 0.07, 0.03);

        let matrix_jet = Transfer2JetFirst::from_parts(matrix, matrix_first);

        let left = ArrayJetFirst::from_parts(arr0(c(yl)), arr0(c(0.05)));

        let right = ArrayJetFirst::from_parts(arr0(c(yr)), arr0(c(-0.08)));

        let (reflection, transmission) =
            matrix_jet.amplitude_jets(&left, &right, IncidentSide::Left);

        let h = 1e-6;

        let (plus_matrix, plus_yl, plus_yr) = parameters(h);
        let (minus_matrix, minus_yl, minus_yr) = parameters(-h);

        let (r_plus, t_plus) = value_amplitudes(plus_matrix, plus_yl, plus_yr, IncidentSide::Left);

        let (r_minus, t_minus) =
            value_amplitudes(minus_matrix, minus_yl, minus_yr, IncidentSide::Left);

        let expected_dr = (r_plus - r_minus) / (2.0 * h);
        let expected_dt = (t_plus - t_minus) / (2.0 * h);

        assert_complex_close(reflection.first()[()], expected_dr, 1e-7);

        assert_complex_close(transmission.first()[()], expected_dt, 1e-7);
    }

    #[test]
    fn right_incidence_first_order_amplitudes_match_finite_difference() {
        fn parameters(x: f64) -> (Matrix2<C, Ix0>, f64, f64) {
            let matrix = scalar_matrix(
                1.2 + 0.10 * x,
                0.3 - 0.04 * x,
                -0.4 + 0.07 * x,
                0.8 + 0.03 * x,
            );

            let yl = 1.7 + 0.05 * x;
            let yr = 2.1 - 0.08 * x;

            (matrix, yl, yr)
        }

        let (matrix, yl, yr) = parameters(0.0);

        let matrix_jet =
            Transfer2JetFirst::from_parts(matrix, scalar_matrix(0.10, -0.04, 0.07, 0.03));

        let left = ArrayJetFirst::from_parts(arr0(c(yl)), arr0(c(0.05)));

        let right = ArrayJetFirst::from_parts(arr0(c(yr)), arr0(c(-0.08)));

        let (reflection, transmission) =
            matrix_jet.amplitude_jets(&left, &right, IncidentSide::Right);

        let h = 1e-6;

        let (plus_matrix, plus_yl, plus_yr) = parameters(h);
        let (minus_matrix, minus_yl, minus_yr) = parameters(-h);

        let (r_plus, t_plus) = value_amplitudes(plus_matrix, plus_yl, plus_yr, IncidentSide::Right);

        let (r_minus, t_minus) =
            value_amplitudes(minus_matrix, minus_yl, minus_yr, IncidentSide::Right);

        let expected_dr = (r_plus - r_minus) / (2.0 * h);
        let expected_dt = (t_plus - t_minus) / (2.0 * h);

        assert_complex_close(reflection.first()[()], expected_dr, 1e-7);

        assert_complex_close(transmission.first()[()], expected_dt, 1e-7);
    }

    #[test]
    fn second_order_amplitudes_match_finite_difference() {
        fn parameters(x: f64) -> (Matrix2<C, Ix0>, f64, f64) {
            let matrix = scalar_matrix(
                1.2 + 0.10 * x + 0.015 * x * x,
                0.3 - 0.04 * x + 0.010 * x * x,
                -0.4 + 0.07 * x - 0.020 * x * x,
                0.8 + 0.03 * x + 0.025 * x * x,
            );

            let yl = 1.7 + 0.05 * x + 0.0125 * x * x;

            let yr = 2.1 - 0.08 * x + 0.0200 * x * x;

            (matrix, yl, yr)
        }

        let (matrix, yl, yr) = parameters(0.0);

        let first = scalar_matrix(0.10, -0.04, 0.07, 0.03);

        // Coefficients of x² above are half the second derivatives.
        let second = scalar_matrix(0.03, 0.02, -0.04, 0.05);

        let matrix_jet = Transfer2Jet::from_parts(matrix, first, second);

        let left = ArrayJet::from_parts(arr0(c(yl)), arr0(c(0.05)), arr0(c(0.025)));

        let right = ArrayJet::from_parts(arr0(c(yr)), arr0(c(-0.08)), arr0(c(0.04)));

        let (reflection, transmission) =
            matrix_jet.amplitude_jets(&left, &right, IncidentSide::Left);

        let h = 1e-4;

        let (plus_matrix, plus_yl, plus_yr) = parameters(h);
        let (zero_matrix, zero_yl, zero_yr) = parameters(0.0);
        let (minus_matrix, minus_yl, minus_yr) = parameters(-h);

        let (r_plus, t_plus) = value_amplitudes(plus_matrix, plus_yl, plus_yr, IncidentSide::Left);

        let (r_zero, t_zero) = value_amplitudes(zero_matrix, zero_yl, zero_yr, IncidentSide::Left);

        let (r_minus, t_minus) =
            value_amplitudes(minus_matrix, minus_yl, minus_yr, IncidentSide::Left);

        let expected_ddr = (r_plus - c(2.0) * r_zero + r_minus) / (h * h);

        let expected_ddt = (t_plus - c(2.0) * t_zero + t_minus) / (h * h);

        assert_complex_close(reflection.second()[()], expected_ddr, 2e-6);

        assert_complex_close(transmission.second()[()], expected_ddt, 2e-6);
    }

    #[test]
    fn right_incidence_second_order_amplitudes_match_finite_difference() {
        fn parameters(x: f64) -> (Matrix2<C, Ix0>, f64, f64) {
            let matrix = scalar_matrix(
                1.2 + 0.10 * x + 0.015 * x * x,
                0.3 - 0.04 * x + 0.010 * x * x,
                -0.4 + 0.07 * x - 0.020 * x * x,
                0.8 + 0.03 * x + 0.025 * x * x,
            );

            let yl = 1.7 + 0.05 * x + 0.0125 * x * x;

            let yr = 2.1 - 0.08 * x + 0.0200 * x * x;

            (matrix, yl, yr)
        }

        let (matrix, yl, yr) = parameters(0.0);

        let matrix_jet = Transfer2Jet::from_parts(
            matrix,
            scalar_matrix(0.10, -0.04, 0.07, 0.03),
            scalar_matrix(0.03, 0.02, -0.04, 0.05),
        );

        let left = ArrayJet::from_parts(arr0(c(yl)), arr0(c(0.05)), arr0(c(0.025)));

        let right = ArrayJet::from_parts(arr0(c(yr)), arr0(c(-0.08)), arr0(c(0.04)));

        let (reflection, transmission) =
            matrix_jet.amplitude_jets(&left, &right, IncidentSide::Right);

        let h = 1e-4;

        let (plus_matrix, plus_yl, plus_yr) = parameters(h);
        let (zero_matrix, zero_yl, zero_yr) = parameters(0.0);
        let (minus_matrix, minus_yl, minus_yr) = parameters(-h);

        let (r_plus, t_plus) = value_amplitudes(plus_matrix, plus_yl, plus_yr, IncidentSide::Right);

        let (r_zero, t_zero) = value_amplitudes(zero_matrix, zero_yl, zero_yr, IncidentSide::Right);

        let (r_minus, t_minus) =
            value_amplitudes(minus_matrix, minus_yl, minus_yr, IncidentSide::Right);

        let expected_ddr = (r_plus - c(2.0) * r_zero + r_minus) / (h * h);

        let expected_ddt = (t_plus - c(2.0) * t_zero + t_minus) / (h * h);

        assert_complex_close(reflection.second()[()], expected_ddr, 2e-6);

        assert_complex_close(transmission.second()[()], expected_ddt, 2e-6);
    }

    #[test]
    fn sampled_value_amplitudes_preserve_shape() {
        let matrix = Matrix2::new(
            array![c(1.0), c(1.1), c(1.2)],
            array![c(0.1), c(0.2), c(0.3)],
            array![c(-0.2), c(-0.1), c(0.0)],
            array![c(0.9), c(1.0), c(1.1)],
        );

        let left = array![c(1.5), c(1.6), c(1.7)];
        let right = array![c(2.0), c(2.1), c(2.2)];

        let (reflection, transmission) = matrix.amplitudes(&left, &right, IncidentSide::Left);

        assert_eq!(reflection.raw_dim(), left.raw_dim());
        assert_eq!(transmission.raw_dim(), left.raw_dim());
    }

    #[test]
    fn sampled_first_order_amplitudes_preserve_shape() {
        let value = Matrix2::new(
            array![c(1.0), c(1.1)],
            array![c(0.1), c(0.2)],
            array![c(-0.2), c(-0.1)],
            array![c(0.9), c(1.0)],
        );

        let first = Matrix2::new(
            array![c(0.01), c(0.02)],
            array![c(0.03), c(0.04)],
            array![c(0.05), c(0.06)],
            array![c(0.07), c(0.08)],
        );

        let matrix = Transfer2JetFirst::from_parts(value, first);

        let left = ArrayJetFirst::from_parts(array![c(1.5), c(1.6)], array![c(0.1), c(0.2)]);

        let right = ArrayJetFirst::from_parts(array![c(2.0), c(2.1)], array![c(-0.1), c(-0.2)]);

        let (reflection, transmission) = matrix.amplitude_jets(&left, &right, IncidentSide::Left);

        assert_eq!(reflection.value().raw_dim(), left.value().raw_dim(),);
        assert_eq!(reflection.first().raw_dim(), left.value().raw_dim(),);
        assert_eq!(transmission.value().raw_dim(), left.value().raw_dim(),);
        assert_eq!(transmission.first().raw_dim(), left.value().raw_dim(),);
    }

    #[test]
    fn sampled_second_order_amplitudes_preserve_shape() {
        let value = Matrix2::new(
            array![c(1.0), c(1.1)],
            array![c(0.1), c(0.2)],
            array![c(-0.2), c(-0.1)],
            array![c(0.9), c(1.0)],
        );

        let first = Matrix2::zeros_like(value.m11());
        let second = Matrix2::zeros_like(value.m11());

        let matrix = Transfer2Jet::from_parts(value, first, second);

        let left = ArrayJet::from_parts(
            array![c(1.5), c(1.6)],
            array![c(0.1), c(0.2)],
            array![c(0.01), c(0.02)],
        );

        let right = ArrayJet::from_parts(
            array![c(2.0), c(2.1)],
            array![c(-0.1), c(-0.2)],
            array![c(0.03), c(0.04)],
        );

        let (reflection, transmission) = matrix.amplitude_jets(&left, &right, IncidentSide::Right);

        let expected = left.value().raw_dim();

        assert_eq!(reflection.value().raw_dim(), expected);
        assert_eq!(reflection.first().raw_dim(), expected);
        assert_eq!(reflection.second().raw_dim(), expected);

        assert_eq!(transmission.value().raw_dim(), expected);
        assert_eq!(transmission.first().raw_dim(), expected);
        assert_eq!(transmission.second().raw_dim(), expected);
    }
}

#[cfg(test)]
mod transfer2_pw_tests {
    use approx::assert_relative_eq;
    use ndarray::{Array0, ArrayBase, Dimension, OwnedRepr, arr0, array};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        backend::{
            DerivativeVariable, PlanarInput, PlaneWaveBackend, PlaneWaveInput, Polarisation,
            derivative::SpectralDerivativeVariable,
            transfer2::{Transfer2, Transfer2Error},
        },
        material::Constant,
        stack::{Stack, Thickness, ValidationConfig},
    };

    type C = Complex64;

    fn c(value: f64) -> C {
        C::new(value, 0.0)
    }

    fn material(epsilon: f64, mu: f64) -> Constant<f64> {
        Constant::new(epsilon, mu)
    }

    fn thickness(value_cm: f64) -> Thickness<f64> {
        Thickness::from_cm(value_cm).unwrap()
    }

    fn planar_input(
        vacuum_wavenumber: f64,
        parallel_wavenumber: f64,
        polarisation: Polarisation,
    ) -> PlanarInput<Array0<f64>> {
        PlanarInput::new(
            arr0(vacuum_wavenumber),
            arr0(parallel_wavenumber),
            polarisation,
        )
    }

    fn plane_wave_input(
        vacuum_wavenumber: f64,
        parallel_wavenumber: f64,
        polarisation: Polarisation,
        side: IncidentSide,
    ) -> PlaneWaveInput<Array0<f64>> {
        PlaneWaveInput::new(
            planar_input(vacuum_wavenumber, parallel_wavenumber, polarisation),
            side,
        )
    }

    fn empty_stack(left_epsilon: f64, right_epsilon: f64) -> Stack<Constant<f64>, f64> {
        // Adapt to the concrete Stack constructor.
        Stack::builder(material(left_epsilon, 1.0), material(right_epsilon, 1.0))
            .validation(ValidationConfig::permissive())
            .build()
            .unwrap()
    }

    fn one_layer_stack(thickness_cm: f64) -> Stack<Constant<f64>, f64> {
        // Adapt to the concrete Stack constructor.
        Stack::builder(material(1.0, 1.0), material(1.44, 1.0))
            .layer(material(2.25, 1.0), thickness(thickness_cm))
            .build()
            .unwrap()
    }

    fn two_layer_stack(
        first_thickness_cm: f64,
        second_thickness_cm: f64,
    ) -> Stack<Constant<f64>, f64> {
        Stack::builder(material(1.0, 1.0), material(1.44, 1.0))
            .layer(material(2.25, 1.0), thickness(first_thickness_cm))
            .layer(material(3.24, 1.0), thickness(second_thickness_cm))
            .build()
            .unwrap()
    }

    fn assert_complex_close(actual: C, expected: C, tolerance: f64) {
        assert_relative_eq!(
            actual.re,
            expected.re,
            epsilon = tolerance,
            max_relative = tolerance,
        );

        assert_relative_eq!(
            actual.im,
            expected.im,
            epsilon = tolerance,
            max_relative = tolerance,
        );
    }

    fn assert_array_close<D>(
        actual: &ArrayBase<OwnedRepr<C>, D>,
        expected: &ArrayBase<OwnedRepr<C>, D>,
        tolerance: f64,
    ) where
        D: Dimension,
    {
        assert_eq!(actual.raw_dim(), expected.raw_dim());

        for (&actual, &expected) in actual.iter().zip(expected.iter()) {
            assert_complex_close(actual, expected, tolerance);
        }
    }

    #[test]
    fn empty_stack_left_incidence_matches_fresnel_amplitudes() {
        let stack = empty_stack(1.0, 2.25);

        let input = plane_wave_input(
            3.0,
            0.0,
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        );

        let response: PlaneWaveResponse<C, ndarray::Ix0> =
            Transfer2::new().solve_plane_wave(&stack, &input).unwrap();

        let left_admittance = 1.0;
        let right_admittance = 1.5;

        let expected_reflection =
            (left_admittance - right_admittance) / (left_admittance + right_admittance);

        let expected_transmission = 2.0 * left_admittance / (left_admittance + right_admittance);

        assert_complex_close(response.reflection()[()], c(expected_reflection), 1e-12);

        assert_complex_close(response.transmission()[()], c(expected_transmission), 1e-12);

        assert!(response.derivatives().is_none());
    }

    #[test]
    fn empty_stack_right_incidence_matches_fresnel_amplitudes() {
        let stack = empty_stack(1.0, 2.25);

        let input = plane_wave_input(
            3.0,
            0.0,
            Polarisation::TransverseElectric,
            IncidentSide::Right,
        );

        let response: PlaneWaveResponse<C, ndarray::Ix0> =
            Transfer2::new().solve_plane_wave(&stack, &input).unwrap();

        let left_admittance = 1.0;
        let right_admittance = 1.5;

        let expected_reflection =
            (right_admittance - left_admittance) / (left_admittance + right_admittance);

        let expected_transmission = 2.0 * right_admittance / (left_admittance + right_admittance);

        assert_complex_close(response.reflection()[()], c(expected_reflection), 1e-12);

        assert_complex_close(response.transmission()[()], c(expected_transmission), 1e-12);
    }

    #[test]
    fn identical_exterior_media_and_empty_stack_are_transparent() {
        let stack = empty_stack(1.0, 1.0);

        for side in [IncidentSide::Left, IncidentSide::Right] {
            let input = plane_wave_input(3.0, 0.4, Polarisation::TransverseMagnetic, side);

            let response: PlaneWaveResponse<C, ndarray::Ix0> =
                Transfer2::new().solve_plane_wave(&stack, &input).unwrap();

            assert_complex_close(response.reflection()[()], c(0.0), 1e-12);

            assert_complex_close(response.transmission()[()], c(1.0), 1e-12);
        }
    }

    #[test]
    fn first_derivative_response_contains_requested_variable() {
        let stack = one_layer_stack(0.2);

        let input = plane_wave_input(
            3.0,
            0.4,
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        );

        let response: PlaneWaveResponse<C, ndarray::Ix0> = Transfer2::new()
            .solve_plane_wave_structural_first_derivative(
                &stack,
                &input,
                StructuralDerivativeVariable::Thickness(0),
            )
            .unwrap();

        let derivatives = response.derivatives().unwrap();

        assert_eq!(derivatives.variable(), DerivativeVariable::Thickness(0),);

        assert!(derivatives.second().is_none());
    }

    #[test]
    fn second_derivative_response_contains_both_orders() {
        let stack = one_layer_stack(0.2);

        let input = plane_wave_input(
            3.0,
            0.4,
            Polarisation::TransverseMagnetic,
            IncidentSide::Right,
        );

        let response: PlaneWaveResponse<C, ndarray::Ix0> = Transfer2::new()
            .solve_plane_wave_structural_second_derivative(
                &stack,
                &input,
                StructuralDerivativeVariable::ParallelWavenumberSquared,
            )
            .unwrap();

        let derivatives = response.derivatives().unwrap();

        assert_eq!(
            derivatives.variable(),
            DerivativeVariable::ParallelWavenumberSquared,
        );

        assert!(derivatives.second().is_some());
    }

    #[test]
    fn thickness_first_derivatives_match_finite_difference() {
        let d = 0.2;
        let h = 1e-6;

        let input = plane_wave_input(
            3.0,
            0.4,
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        );

        let analytic = Transfer2::new()
            .solve_plane_wave_structural_first_derivative(
                &one_layer_stack(d),
                &input,
                StructuralDerivativeVariable::Thickness(0),
            )
            .unwrap();

        let plus: PlaneWaveResponse<C, ndarray::Ix0> = Transfer2::new()
            .solve_plane_wave(&one_layer_stack(d + h), &input)
            .unwrap();

        let minus: PlaneWaveResponse<C, ndarray::Ix0> = Transfer2::new()
            .solve_plane_wave(&one_layer_stack(d - h), &input)
            .unwrap();

        let expected_dr = (plus.reflection()[()] - minus.reflection()[()]) / (2.0 * h);

        let expected_dt = (plus.transmission()[()] - minus.transmission()[()]) / (2.0 * h);

        let first = analytic.derivatives().unwrap().first();

        assert_complex_close(first.reflection()[()], expected_dr, 2e-7);

        assert_complex_close(first.transmission()[()], expected_dt, 2e-7);
    }

    #[test]
    fn thickness_second_derivatives_match_finite_difference() {
        let d = 0.2;
        let h = 1e-4;

        let input = plane_wave_input(
            3.0,
            0.4,
            Polarisation::TransverseMagnetic,
            IncidentSide::Right,
        );

        let analytic = Transfer2::new()
            .solve_plane_wave_structural_second_derivative(
                &one_layer_stack(d),
                &input,
                StructuralDerivativeVariable::Thickness(0),
            )
            .unwrap();

        let plus: PlaneWaveResponse<C, ndarray::Ix0> = Transfer2::new()
            .solve_plane_wave(&one_layer_stack(d + h), &input)
            .unwrap();

        let zero: PlaneWaveResponse<C, ndarray::Ix0> = Transfer2::new()
            .solve_plane_wave(&one_layer_stack(d), &input)
            .unwrap();

        let minus: PlaneWaveResponse<C, ndarray::Ix0> = Transfer2::new()
            .solve_plane_wave(&one_layer_stack(d - h), &input)
            .unwrap();

        let expected_ddr = (plus.reflection()[()] - c(2.0) * zero.reflection()[()]
            + minus.reflection()[()])
            / (h * h);

        let expected_ddt = (plus.transmission()[()] - c(2.0) * zero.transmission()[()]
            + minus.transmission()[()])
            / (h * h);

        let second = analytic.derivatives().unwrap().second().unwrap();

        assert_complex_close(second.reflection()[()], expected_ddr, 3e-6);

        assert_complex_close(second.transmission()[()], expected_ddt, 3e-6);
    }

    #[test]
    fn vacuum_wavenumber_first_derivatives_match_finite_difference() {
        let stack = two_layer_stack(0.15, 0.23);

        let k0 = 3.0;
        let h = 1e-6;

        let input = plane_wave_input(
            k0,
            0.4,
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        );

        let analytic = Transfer2::new()
            .solve_plane_wave_spectral_first_derivative(
                &stack,
                &input,
                SpectralDerivativeVariable::VacuumWavenumber,
            )
            .unwrap();

        let plus_input = plane_wave_input(
            k0 + h,
            0.4,
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        );

        let minus_input = plane_wave_input(
            k0 - h,
            0.4,
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        );

        let plus: PlaneWaveResponse<C, ndarray::Ix0> = Transfer2::new()
            .solve_plane_wave(&stack, &plus_input)
            .unwrap();

        let minus: PlaneWaveResponse<C, ndarray::Ix0> = Transfer2::new()
            .solve_plane_wave(&stack, &minus_input)
            .unwrap();

        let expected_dr = (plus.reflection()[()] - minus.reflection()[()]) / (2.0 * h);

        let expected_dt = (plus.transmission()[()] - minus.transmission()[()]) / (2.0 * h);

        let first = analytic.derivatives().unwrap().first();

        assert_complex_close(first.reflection()[()], expected_dr, 3e-7);

        assert_complex_close(first.transmission()[()], expected_dt, 3e-7);
    }

    #[test]
    fn parallel_wavenumber_squared_first_derivatives_match_finite_difference() {
        let stack = two_layer_stack(0.15, 0.23);

        let parallel_squared: f64 = 0.16;
        let h = 1e-6;

        let input = plane_wave_input(
            3.0,
            parallel_squared.sqrt(),
            Polarisation::TransverseMagnetic,
            IncidentSide::Right,
        );

        let analytic = Transfer2::new()
            .solve_plane_wave_structural_first_derivative(
                &stack,
                &input,
                StructuralDerivativeVariable::ParallelWavenumberSquared,
            )
            .unwrap();

        let plus_input = plane_wave_input(
            3.0,
            (parallel_squared + h).sqrt(),
            Polarisation::TransverseMagnetic,
            IncidentSide::Right,
        );

        let minus_input = plane_wave_input(
            3.0,
            (parallel_squared - h).sqrt(),
            Polarisation::TransverseMagnetic,
            IncidentSide::Right,
        );

        let plus: PlaneWaveResponse<C, ndarray::Ix0> = Transfer2::new()
            .solve_plane_wave(&stack, &plus_input)
            .unwrap();

        let minus: PlaneWaveResponse<C, ndarray::Ix0> = Transfer2::new()
            .solve_plane_wave(&stack, &minus_input)
            .unwrap();

        let expected_dr = (plus.reflection()[()] - minus.reflection()[()]) / (2.0 * h);

        let expected_dt = (plus.transmission()[()] - minus.transmission()[()]) / (2.0 * h);

        let first = analytic.derivatives().unwrap().first();

        assert_complex_close(first.reflection()[()], expected_dr, 3e-7);

        assert_complex_close(first.transmission()[()], expected_dt, 3e-7);
    }

    #[test]
    fn linear_parallel_derivative_matches_squared_chain_rule() {
        let stack = two_layer_stack(0.15, 0.23);

        let parallel = 0.4;

        let input = plane_wave_input(
            3.0,
            parallel,
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        );

        let linear: PlaneWaveResponse<C, ndarray::Ix0> = Transfer2::new()
            .solve_plane_wave_structural_first_derivative(
                &stack,
                &input,
                StructuralDerivativeVariable::ParallelWavenumber,
            )
            .unwrap();

        let squared: PlaneWaveResponse<C, ndarray::Ix0> = Transfer2::new()
            .solve_plane_wave_structural_first_derivative(
                &stack,
                &input,
                StructuralDerivativeVariable::ParallelWavenumberSquared,
            )
            .unwrap();

        let linear = linear.derivatives().unwrap().first();
        let squared = squared.derivatives().unwrap().first();

        assert_complex_close(
            linear.reflection()[()],
            c(2.0 * parallel) * squared.reflection()[()],
            1e-11,
        );

        assert_complex_close(
            linear.transmission()[()],
            c(2.0 * parallel) * squared.transmission()[()],
            1e-11,
        );
    }

    #[test]
    fn invalid_thickness_index_is_returned_by_plane_wave_backend() {
        let stack = one_layer_stack(0.2);

        let input = plane_wave_input(
            3.0,
            0.4,
            Polarisation::TransverseElectric,
            IncidentSide::Left,
        );

        let error = <Transfer2 as PlaneWaveBackend<
            C,
            ndarray::Dim<[usize; 0]>,
            Stack<Constant<f64>, f64>,
        >>::solve_plane_wave_structural_first_derivative(
            &Transfer2::new(),
            &stack,
            &input,
            StructuralDerivativeVariable::Thickness(1),
        )
        .unwrap_err();

        assert_eq!(
            error,
            Transfer2Error::ThicknessLayerOutOfBounds {
                requested: 1,
                layer_count: 1,
            },
        );
    }

    #[test]
    fn sampled_plane_wave_response_preserves_input_shape() {
        let stack = one_layer_stack(0.2);

        let planar = PlanarInput::new(
            array![2.0, 2.5, 3.0],
            array![0.2, 0.3, 0.4],
            Polarisation::TransverseElectric,
        );

        let input = PlaneWaveInput::new(planar, IncidentSide::Left);

        let response: PlaneWaveResponse<C, ndarray::Ix1> = Transfer2::new()
            .solve_plane_wave_spectral_second_derivative(
                &stack,
                &input,
                SpectralDerivativeVariable::VacuumWavenumber,
            )
            .unwrap();

        let expected = input.planar().vacuum_wavenumber().raw_dim();

        assert_eq!(response.reflection().raw_dim(), expected);
        assert_eq!(response.transmission().raw_dim(), expected);

        let derivatives = response.derivatives().unwrap();

        assert_eq!(derivatives.first().reflection().raw_dim(), expected,);
        assert_eq!(derivatives.first().transmission().raw_dim(), expected,);

        let second = derivatives.second().unwrap();

        assert_eq!(second.reflection().raw_dim(), expected);
        assert_eq!(second.transmission().raw_dim(), expected);
    }

    fn assert_real_close(actual: f64, expected: f64, tolerance: f64) {
        assert_relative_eq!(
            actual,
            expected,
            epsilon = tolerance,
            max_relative = tolerance,
        );
    }

    #[test]
    fn assert_lossless_power_balance_transfer2() {
        let stack = two_layer_stack(1.0, 2.0);

        for polarisation in [
            Polarisation::TransverseElectric,
            Polarisation::TransverseMagnetic,
        ] {
            for incident_side in [IncidentSide::Left, IncidentSide::Right] {
                let input = PlaneWaveInput::new(
                    PlanarInput::new(arr0(3.0), arr0(0.4), polarisation),
                    incident_side,
                );

                let response: PlaneWaveResponse<C, ndarray::Ix0> =
                    Transfer2::new().solve_plane_wave(&stack, &input).unwrap();

                assert_real_close(
                    response.power().reflectance()[()] + response.power().transmittance()[()],
                    1.0,
                    1e-10,
                );

                assert_real_close(response.power().absorptance()[()], 0.0, 1e-10);
            }
        }
    }

    #[test]
    fn reflectance_and_transmittance_derivatives_match_finite_difference() {
        let stack = two_layer_stack(1.0, 2.0);
        let backend = Transfer2::new();

        let k0 = 3.0;
        let h = 1e-4;

        let input = |k0| {
            PlaneWaveInput::new(
                PlanarInput::new(arr0(k0), arr0(0.4), Polarisation::TransverseElectric),
                IncidentSide::Left,
            )
        };

        let analytic: PlaneWaveResponse<C, ndarray::Ix0> = backend
            .solve_plane_wave_spectral_second_derivative(
                &stack,
                &input(k0),
                SpectralDerivativeVariable::VacuumWavenumber,
            )
            .unwrap();

        let plus: PlaneWaveResponse<C, ndarray::Ix0> =
            backend.solve_plane_wave(&stack, &input(k0 + h)).unwrap();

        let zero: PlaneWaveResponse<C, ndarray::Ix0> =
            backend.solve_plane_wave(&stack, &input(k0)).unwrap();

        let minus: PlaneWaveResponse<C, ndarray::Ix0> =
            backend.solve_plane_wave(&stack, &input(k0 - h)).unwrap();

        let expected_reflectance_first =
            (plus.power().reflectance()[()] - minus.power().reflectance()[()]) / (2.0 * h);

        let expected_transmittance_first =
            (plus.power().transmittance()[()] - minus.power().transmittance()[()]) / (2.0 * h);

        let expected_reflectance_second = (plus.power().reflectance()[()]
            - 2.0 * zero.power().reflectance()[()]
            + minus.power().reflectance()[()])
            / (h * h);

        let expected_transmittance_second = (plus.power().transmittance()[()]
            - 2.0 * zero.power().transmittance()[()]
            + minus.power().transmittance()[()])
            / (h * h);

        let derivatives = analytic.derivatives().unwrap();

        assert_real_close(
            derivatives.first().power().reflectance()[()],
            expected_reflectance_first,
            1e-6,
        );

        assert_real_close(
            derivatives.first().power().transmittance()[()],
            expected_transmittance_first,
            1e-6,
        );

        assert_real_close(
            derivatives.second().unwrap().power().reflectance()[()],
            expected_reflectance_second,
            1e-4,
        );

        assert_real_close(
            derivatives.second().unwrap().power().transmittance()[()],
            expected_transmittance_second,
            1e-4,
        );
    }
}
