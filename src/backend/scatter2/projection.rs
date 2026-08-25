//! Projective scattering representation and modal projection.
//!
//! The ordinary scattering entries are represented by a common denominator
//! and four numerators. Redheffer composition is performed directly on this
//! homogeneous representation, avoiding division by intermediate scattering
//! denominators.
//!
//! Physical amplitudes and powers are obtained by returning to ordinary
//! scattering entries. Modal quantities are projected directly from the
//! projective representation so that poles of the ordinary scattering matrix
//! need not be materialised.

use ndarray::{ArrayBase, Dimension, OwnedRepr};
use num_traits::{One, Zero};

use crate::{
    ComplexScalar, IncidentSide, PlaneWaveAmplitudes, PlaneWaveDeterminant, PlaneWavePower,
    algebra::{Jet, RealScalarAlgebra, ScalarAlgebra},
    backend::{
        ExteriorContextProvider, PlaneWaveEntries, PlaneWaveModeCandidate,
        scatter2::{Scatter2ExteriorContext, entries::transfer_state_slope},
    },
    observable::{BoundaryState, ProjectAmplitudes, ProjectPlaneWaveModeDeterminant, ProjectPower},
};

use super::Scatter2Entries;

/// Projective representation of a two-port scattering matrix.
///
/// The represented scattering entries satisfy:
///
/// ```text
/// sᵢⱼ = nᵢⱼ / d.
/// ```
///
/// Multiplying `d` and every numerator by the same nonzero scalar leaves the
/// represented scattering matrix unchanged.
#[derive(Clone, Debug, PartialEq)]
pub struct Scatter2ProjectiveEntries<A> {
    denominator: A,
    n11: A,
    n12: A,
    n21: A,
    n22: A,
}

impl<A> PlaneWaveEntries for Scatter2ProjectiveEntries<A> {
    type ExteriorContext = Scatter2ExteriorContext<A>;
    type Algebra = A;
}

impl<A> Scatter2ProjectiveEntries<A> {
    pub(crate) const fn from_parts(denominator: A, n11: A, n12: A, n21: A, n22: A) -> Self {
        Self {
            denominator,
            n11,
            n12,
            n21,
            n22,
        }
    }

    pub(crate) fn denominator(&self) -> &A {
        &self.denominator
    }

    pub(crate) fn n11(&self) -> &A {
        &self.n11
    }

    pub(crate) fn n12(&self) -> &A {
        &self.n12
    }

    pub(crate) fn n21(&self) -> &A {
        &self.n21
    }

    pub(crate) fn n22(&self) -> &A {
        &self.n22
    }

    pub(crate) fn sample_source(&self) -> &ArrayBase<OwnedRepr<A::Scalar>, A::Dimension>
    where
        A: ScalarAlgebra,
    {
        self.n11.value()
    }
}

impl<A> Scatter2ProjectiveEntries<A>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar + One + Zero,
    A::Dimension: Dimension,
{
    pub(crate) fn identity_like(
        source: &ndarray::ArrayBase<ndarray::OwnedRepr<A::Scalar>, A::Dimension>,
    ) -> Self {
        let zero = A::filled_constant_like(source, A::Scalar::zero());

        let one = A::filled_constant_like(source, A::Scalar::one());

        Self {
            denominator: one.clone(),
            n11: zero.clone(),
            n12: one.clone(),
            n21: one,
            n22: zero,
        }
    }
}

impl<A> Scatter2ProjectiveEntries<A>
where
    A: ScalarAlgebra,
{
    pub(crate) fn entries(&self) -> Scatter2Entries<A> {
        Scatter2Entries::from_parts(
            self.n11.divide(&self.denominator),
            self.n12.divide(&self.denominator),
            self.n21.divide(&self.denominator),
            self.n22.divide(&self.denominator),
        )
    }
}

impl<A> Scatter2ProjectiveEntries<A>
where
    A: ScalarAlgebra,
    A::Scalar: One,
{
    pub(crate) fn from_entries(entries: &Scatter2Entries<A>) -> Self {
        let denominator = A::filled_constant_like(entries.sample_source(), A::Scalar::one());

        Self {
            denominator,
            n11: entries.s11().clone(),
            n12: entries.s12().clone(),
            n21: entries.s21().clone(),
            n22: entries.s22().clone(),
        }
    }
}

impl<A> Scatter2ProjectiveEntries<A>
where
    A: ScalarAlgebra,
    A::Scalar: ComplexScalar + One,
    A::Dimension: Dimension,
{
    pub(crate) fn right_gauged_mode_candidate(
        &self,
        left_admittance: &A,
    ) -> PlaneWaveModeCandidate<A> {
        let slope = transfer_state_slope(left_admittance);

        /*
         * Projectively scaled left waves:
         *
         * forward  = denominator
         * backward = n11
         */
        let field = self.denominator().add(self.n11());

        let secondary = self.n11().subtract(self.denominator()).multiply(&slope);

        let two = A::Scalar::one() + A::Scalar::one();

        let residual = slope.multiply(self.denominator()).scale(two);

        let right_outgoing = self.n21().clone();

        PlaneWaveModeCandidate::new(
            BoundaryState::new(field, secondary),
            right_outgoing,
            residual,
        )
    }
}

pub(crate) fn cascade_projection<A>(
    left: &Scatter2ProjectiveEntries<A>,
    right: &Scatter2ProjectiveEntries<A>,
) -> Scatter2ProjectiveEntries<A>
where
    A: ScalarAlgebra,
{
    let dl = left.denominator();
    let dr = right.denominator();

    let interaction = dl.multiply(dr).subtract(&left.n22().multiply(right.n11()));

    let denominator = dl.multiply(dr).multiply(&interaction);

    /*
     * s11 =
     *     L11 + L12 R11 L21 / (1 - L22 R11)
     */
    let n11 = left
        .n11()
        .multiply(&interaction)
        .add(&left.n12().multiply(right.n11()).multiply(left.n21()))
        .multiply(dr);

    /*
     * s12 =
     *     L12 R12 / (1 - L22 R11)
     */
    let n12 = dl.multiply(dr).multiply(left.n12()).multiply(right.n12());

    /*
     * s21 =
     *     R21 L21 / (1 - L22 R11)
     */
    let n21 = dl.multiply(dr).multiply(right.n21()).multiply(left.n21());

    /*
     * s22 =
     *     R22 + R21 L22 R12 / (1 - L22 R11)
     */
    let n22 = right
        .n22()
        .multiply(&interaction)
        .add(&right.n21().multiply(left.n22()).multiply(right.n12()))
        .multiply(dl);

    Scatter2ProjectiveEntries::from_parts(denominator, n11, n12, n21, n22)
}

impl<J> ProjectAmplitudes for Scatter2ProjectiveEntries<J>
where
    J: ScalarAlgebra,
{
    type Amplitudes = PlaneWaveAmplitudes<J>;

    fn project_amplitudes(
        &self,
        exterior: &Self::ExteriorContext,
        incident_side: IncidentSide,
    ) -> Self::Amplitudes {
        self.entries().project_amplitudes(exterior, incident_side)
    }
}

impl<J> ProjectPower for Scatter2ProjectiveEntries<J>
where
    J: Clone + RealScalarAlgebra,
    J::RealJet: ScalarAlgebra,
    <J::RealJet as Jet>::Scalar: One,
{
    type Power = PlaneWavePower<J::RealJet>;

    fn project_power(
        &self,
        exterior: &Self::ExteriorContext,
        incident_side: IncidentSide,
    ) -> Self::Power {
        self.entries().project_power(exterior, incident_side)
    }
}

/// The projection uses `n21` as its transmission chart and is therefore
/// singular where `n21 = 0`. Such points are distinct from poles represented
/// by `denominator = 0`.
impl<J> ProjectPlaneWaveModeDeterminant for Scatter2ProjectiveEntries<J>
where
    J: ScalarAlgebra,
    J::Scalar: ComplexScalar + One,
    J::Dimension: Dimension,
{
    type Determinant = PlaneWaveDeterminant<J>;

    fn project_determinant(&self, exterior: &Self::ExteriorContext) -> Self::Determinant {
        let left_slope = transfer_state_slope(exterior.left_admittance());

        let two = J::Scalar::one() + J::Scalar::one();

        let value = left_slope
            .multiply(self.denominator())
            .scale(two)
            .divide(self.n21());

        PlaneWaveDeterminant::new(value)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    use crate::{
        Polarisation, RealAxis,
        algebra::ScalarAlgebra,
        backend::{
            ExteriorWavevectors, IsotropicLayerQuantities, scatter2::Scatter2ExteriorContext,
        },
        input::CanonicalCoordinates,
        material::Constant,
        observable::BoundaryState,
        test_support::{
            C, TOLERANCE,
            assertions::{assert_array_close, assert_complex_close},
            c,
            jet::{RealJ0, real_j0},
        },
    };

    type Projection = Scatter2ProjectiveEntries<RealJ0>;

    fn jet(value: C) -> RealJ0 {
        real_j0(value)
    }

    fn scalar(value: &RealJ0) -> C {
        value.value()[()]
    }

    fn projection(denominator: C, n11: C, n12: C, n21: C, n22: C) -> Projection {
        Scatter2ProjectiveEntries::from_parts(
            jet(denominator),
            jet(n11),
            jet(n12),
            jet(n21),
            jet(n22),
        )
    }

    fn make_context() -> Scatter2ExteriorContext<RealJ0> {
        let source = jet(c(1.7));

        let coordinates = CanonicalCoordinates::new(source.clone(), source);

        Scatter2ExteriorContext::new::<RealAxis, _>(
            &coordinates,
            &Constant::vacuum(),
            &Constant::vacuum(),
            &ExteriorWavevectors::new(
                IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                    &Constant::vacuum(),
                    &coordinates,
                    Polarisation::TransverseElectric,
                )
                .kappa()
                .clone(),
                IsotropicLayerQuantities::evaluate::<RealAxis, _>(
                    &Constant::vacuum(),
                    &coordinates,
                    Polarisation::TransverseElectric,
                )
                .kappa()
                .clone(),
            ),
            Polarisation::TransverseElectric,
        )
    }

    fn scale_projection(data: &Projection, scale: C) -> Projection {
        Scatter2ProjectiveEntries::from_parts(
            data.denominator().scale(scale),
            data.n11().scale(scale),
            data.n12().scale(scale),
            data.n21().scale(scale),
            data.n22().scale(scale),
        )
    }

    fn assert_state_scaled(
        actual: &BoundaryState<RealJ0>,
        expected: &BoundaryState<RealJ0>,
        scale: C,
    ) {
        assert_complex_close(
            scalar(actual.field()),
            scale * scalar(expected.field()),
            TOLERANCE,
        );

        assert_complex_close(
            scalar(actual.secondary()),
            scale * scalar(expected.secondary()),
            TOLERANCE,
        );
    }

    #[test]
    fn entries_divides_every_numerator_by_common_denominator() {
        let data = projection(c(2.0), c(3.0), c(5.0), c(7.0), c(11.0));

        let entries = data.entries();

        assert_complex_close(scalar(entries.s11()), c(1.5), TOLERANCE);

        assert_complex_close(scalar(entries.s12()), c(2.5), TOLERANCE);

        assert_complex_close(scalar(entries.s21()), c(3.5), TOLERANCE);

        assert_complex_close(scalar(entries.s22()), c(5.5), TOLERANCE);
    }

    #[test]
    fn common_projective_scaling_does_not_change_into_scattering_entries() {
        let data = projection(c(2.0), c(3.0), c(5.0), c(7.0), c(11.0));

        let scale = c(1.7) + C::i() * c(-0.4);

        let scaled = scale_projection(&data, scale);

        let actual = scaled.entries();
        let expected = data.entries();

        assert_array_close(actual.s11().value(), expected.s11().value(), TOLERANCE);

        assert_array_close(actual.s12().value(), expected.s12().value(), TOLERANCE);

        assert_array_close(actual.s21().value(), expected.s21().value(), TOLERANCE);

        assert_array_close(actual.s22().value(), expected.s22().value(), TOLERANCE);
    }

    #[test]
    fn determinant_is_invariant_under_common_projective_scaling() {
        let data = projection(c(2.0), c(3.0), c(5.0), c(7.0), c(11.0));

        let scale = c(1.7) + C::i() * c(-0.4);

        let scaled = scale_projection(&data, scale);

        let context = make_context();

        let actual = scaled.project_determinant(&context);
        let expected = data.project_determinant(&context);

        assert_array_close(actual.value().value(), expected.value().value(), TOLERANCE);
    }

    #[test]
    fn projective_determinant_matches_inverse_transmission_characteristic() {
        let data = projection(c(2.0), c(3.0), c(5.0), c(7.0), c(11.0));

        let context = make_context();

        let determinant = data.project_determinant(&context);

        let entries = data.entries();

        let slope = transfer_state_slope(context.left_admittance());

        let two = C::one() + C::one();

        let expected = slope.scale(two).divide(entries.s21());

        assert_array_close(determinant.value().value(), expected.value(), TOLERANCE);
    }

    #[test]
    fn right_gauged_candidate_uses_denominator_and_n11_as_left_waves() {
        let data = projection(c(2.0), c(3.0), c(5.0), c(7.0), c(11.0));

        let context = make_context();

        let candidate = data.right_gauged_mode_candidate(context.left_admittance());

        let slope = transfer_state_slope(context.left_admittance());

        let expected_field = data.denominator().add(data.n11());

        let expected_secondary = data.n11().subtract(data.denominator()).multiply(&slope);

        assert_array_close(
            candidate.state().field().value(),
            expected_field.value(),
            TOLERANCE,
        );

        assert_array_close(
            candidate.state().secondary().value(),
            expected_secondary.value(),
            TOLERANCE,
        );
    }

    #[test]
    fn candidate_residual_is_left_outgoing_boundary_residual() {
        let data = projection(c(2.0), c(3.0), c(5.0), c(7.0), c(11.0));

        let context = make_context();

        let candidate = data.right_gauged_mode_candidate(context.left_admittance());

        let slope = transfer_state_slope(context.left_admittance());

        let expected = slope
            .multiply(candidate.state().field())
            .subtract(candidate.state().secondary());

        assert_array_close(candidate.residual().value(), expected.value(), TOLERANCE);
    }

    #[test]
    fn chart_normalized_candidate_residual_equals_determinant() {
        let data = projection(c(2.0), c(3.0), c(5.0), c(7.0), c(11.0));

        let context = make_context();

        let candidate = data.right_gauged_mode_candidate(context.left_admittance());

        let determinant = data.project_determinant(&context);

        let chart_normalized = candidate.residual().divide(data.n21());

        assert_array_close(
            chart_normalized.value(),
            determinant.value().value(),
            TOLERANCE,
        );
    }

    #[test]
    fn projective_candidate_scales_with_representation() {
        let data = projection(c(2.0), c(3.0), c(5.0), c(7.0), c(11.0));

        let scale = c(1.7) + C::i() * c(-0.4);

        let scaled = scale_projection(&data, scale);

        let context = make_context();

        let base = data.right_gauged_mode_candidate(context.left_admittance());

        let scaled = scaled.right_gauged_mode_candidate(context.left_admittance());

        assert_state_scaled(scaled.state(), base.state(), scale);

        assert_complex_close(
            scalar(scaled.residual()),
            scale * scalar(base.residual()),
            TOLERANCE,
        );
    }

    #[test]
    fn candidate_remains_finite_at_exact_projective_pole() {
        let data = projection(c(0.0), c(3.0), c(5.0), c(7.0), c(11.0));

        let context = make_context();

        let candidate = data.right_gauged_mode_candidate(context.left_admittance());

        assert!(scalar(candidate.state().field()).re.is_finite(),);

        assert!(scalar(candidate.state().field()).im.is_finite(),);

        assert!(scalar(candidate.state().secondary()).re.is_finite(),);

        assert!(scalar(candidate.state().secondary()).im.is_finite(),);

        assert_complex_close(scalar(candidate.residual()), c(0.0), TOLERANCE);

        let determinant = data.project_determinant(&context);

        assert_complex_close(scalar(determinant.value()), c(0.0), TOLERANCE);
    }

    #[test]
    fn regularized_candidate_is_n21_times_unit_right_outgoing_candidate() {
        let data = projection(c(2.0), c(3.0), c(5.0), c(7.0), c(11.0));

        let context = make_context();

        let regularized = data.right_gauged_mode_candidate(context.left_admittance());

        let entries = data.entries();

        let left_forward = entries.s21().reciprocal();

        let left_backward = entries.s11().divide(entries.s21());

        let slope = transfer_state_slope(context.left_admittance());

        let unit_state = BoundaryState::new(
            left_forward.add(&left_backward),
            left_backward.subtract(&left_forward).multiply(&slope),
        );

        assert_state_scaled(regularized.state(), &unit_state, scalar(data.n21()));
    }
}

#[cfg(test)]
mod cascade_tests {
    use super::*;

    use crate::{
        backend::scatter2::entries::cascade,
        test_support::{
            C, TOLERANCE,
            assertions::assert_array_close,
            c,
            jet::{RealJ0, real_j0},
        },
    };

    type Projection = Scatter2ProjectiveEntries<RealJ0>;
    type Entries = Scatter2Entries<RealJ0>;

    fn jet(value: C) -> RealJ0 {
        real_j0(value)
    }

    fn entries(s11: C, s12: C, s21: C, s22: C) -> Entries {
        Scatter2Entries::from_parts(jet(s11), jet(s12), jet(s21), jet(s22))
    }

    fn assert_entries_close(actual: &Entries, expected: &Entries) {
        assert_array_close(actual.s11().value(), expected.s11().value(), TOLERANCE);

        assert_array_close(actual.s12().value(), expected.s12().value(), TOLERANCE);

        assert_array_close(actual.s21().value(), expected.s21().value(), TOLERANCE);

        assert_array_close(actual.s22().value(), expected.s22().value(), TOLERANCE);
    }

    #[test]
    fn projective_cascade_reduces_to_ordinary_redheffer_cascade() {
        let left = entries(c(0.10), c(0.80), c(0.70), c(-0.20));

        let right = entries(c(0.30), c(0.60), c(0.50), c(-0.10));

        let expected = cascade(&left, &right);

        let actual = cascade_projection(
            &Projection::from_entries(&left),
            &Projection::from_entries(&right),
        )
        .entries();

        assert_entries_close(&actual, &expected);
    }

    #[test]
    fn projective_cascade_is_unchanged_by_operand_chart_scaling() {
        let left = Projection::from_entries(&entries(c(0.10), c(0.80), c(0.70), c(-0.20)));

        let right = Projection::from_entries(&entries(c(0.30), c(0.60), c(0.50), c(-0.10)));

        let left_scale = c(1.3) + C::i() * c(0.2);

        let right_scale = c(-0.7) + C::i() * c(0.4);

        let scaled_left = Scatter2ProjectiveEntries::from_parts(
            left.denominator().scale(left_scale),
            left.n11().scale(left_scale),
            left.n12().scale(left_scale),
            left.n21().scale(left_scale),
            left.n22().scale(left_scale),
        );

        let scaled_right = Scatter2ProjectiveEntries::from_parts(
            right.denominator().scale(right_scale),
            right.n11().scale(right_scale),
            right.n12().scale(right_scale),
            right.n21().scale(right_scale),
            right.n22().scale(right_scale),
        );

        let expected = cascade_projection(&left, &right).entries();

        let actual = cascade_projection(&scaled_left, &scaled_right).entries();

        assert_entries_close(&actual, &expected);
    }

    #[test]
    fn projective_identity_reduces_to_redheffer_identity() {
        let source = ndarray::arr0(c(0.0));

        let identity: Projection = Projection::identity_like(&source);

        let entries = identity.entries();

        assert_array_close(entries.s11().value(), &ndarray::arr0(c(0.0)), TOLERANCE);

        assert_array_close(entries.s12().value(), &ndarray::arr0(c(1.0)), TOLERANCE);

        assert_array_close(entries.s21().value(), &ndarray::arr0(c(1.0)), TOLERANCE);

        assert_array_close(entries.s22().value(), &ndarray::arr0(c(0.0)), TOLERANCE);
    }
}
