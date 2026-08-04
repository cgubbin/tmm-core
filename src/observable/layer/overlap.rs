//! Hermitian field overlaps between finite-layer solutions.
//!
//! A Hermitian overlap compares two independently reconstructed solutions.
//! The left solution is complex-conjugated while the right solution is not:
//!
//! ```text
//! electric = ∫ E_left* · E_right dz
//! magnetic = ∫ H_left* · H_right dz
//! ```
//!
//! The overlap pipeline is:
//!
//! 1. integrate directional-wave cross-products analytically through each
//!    homogeneous layer;
//! 2. transform them into canonical-state cross-products;
//! 3. reconstruct complete vector electric and magnetic field overlaps;
//! 4. aggregate the layer contributions;
//! 5. optionally normalize by the two self-overlaps.
//!
//! These are unweighted field overlaps. Constitutive perturbations, energy
//! weights, and coupled-mode operators are applied by later projections.
//!
//! This module implements the Hermitian form appropriate to real-input
//! physical analysis. Complex modal analysis uses a separate bilinear form.
//!

use ndarray::Dimension;

use crate::{
    ComplexScalar, Polarisation,
    algebra::{RealScalarAlgebra, ScalarAlgebra, ScalarAlgebraExpRelExt},
    backend::IsotropicLayerQuantities,
    observable::{BoundaryProjectionError, BoundaryWaves, LayerProjectionError},
};

use super::{
    LayerAggregateError, Layers,
    integration::{
        HermitianOverlapError, IntegratedHermitianFieldOverlap,
        integrate_hermitian_cross_wave_products, project_integrated_hermitian_cross_state_products,
        project_integrated_hermitian_field_overlap,
    },
};

/// Boundary waves and isotropic medium quantities for one overlap operand.
///
/// Waves are expressed at the physical layer's left boundary.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LayerOverlapOperand<A> {
    waves: BoundaryWaves<A>,
    quantities: IsotropicLayerQuantities<A>,
}

impl<A> LayerOverlapOperand<A> {
    pub(crate) const fn new(
        waves: BoundaryWaves<A>,
        quantities: IsotropicLayerQuantities<A>,
    ) -> Self {
        Self { waves, quantities }
    }

    fn into_parts(self) -> (BoundaryWaves<A>, IsotropicLayerQuantities<A>) {
        (self.waves, self.quantities)
    }
}

/// Matched left and right solution data for one physical finite layer.
///
/// The two operands must refer to the same physical layer. `thickness` is the
/// common physical integration interval.
#[derive(Clone, Debug, PartialEq)]
pub struct HermitianLayerOverlapInput<A> {
    left: LayerOverlapOperand<A>,
    right: LayerOverlapOperand<A>,
    thickness: A,
}

impl<A> HermitianLayerOverlapInput<A> {
    pub(crate) const fn new(
        left: LayerOverlapOperand<A>,
        right: LayerOverlapOperand<A>,
        thickness: A,
    ) -> Self {
        Self {
            left,
            right,
            thickness,
        }
    }

    fn into_parts(self) -> (LayerOverlapOperand<A>, LayerOverlapOperand<A>, A) {
        (self.left, self.right, self.thickness)
    }
}

/// Integrated Hermitian field overlap within one finite layer.
///
/// ```text
/// electric = ∫ E_left* · E_right dz
/// magnetic = ∫ H_left* · H_right dz
/// total    = electric + magnetic
/// ```
///
/// This is an unweighted field overlap. Constitutive or perturbative weights
/// are applied by later projections.
#[derive(Clone, Debug, PartialEq)]
pub struct HermitianLayerOverlap<C> {
    electric: C,
    magnetic: C,
    total: C,
}

impl<C> HermitianLayerOverlap<C> {
    pub(crate) fn new(electric: C, magnetic: C, total: C) -> Self {
        Self {
            electric,
            magnetic,
            total,
        }
    }

    /// Return the integrated electric-field overlap.
    pub fn electric(&self) -> &C {
        &self.electric
    }

    /// Return the integrated magnetic-field overlap.
    pub fn magnetic(&self) -> &C {
        &self.magnetic
    }

    /// Return the sum of the electric and magnetic overlaps.
    pub fn total(&self) -> &C {
        &self.total
    }

    /// Consume the overlap and return `(electric, magnetic, total)`.
    pub fn into_parts(self) -> (C, C, C) {
        (self.electric, self.magnetic, self.total)
    }

    /// Transform every overlap component.
    pub fn map<U>(self, mut map: impl FnMut(C) -> U) -> HermitianLayerOverlap<U> {
        HermitianLayerOverlap {
            electric: map(self.electric),
            magnetic: map(self.magnetic),
            total: map(self.total),
        }
    }
}

/// Hermitian field overlap aggregated over all finite layers.
///
/// Exterior media are not included.
#[derive(Clone, Debug, PartialEq)]
pub struct AggregateHermitianOverlap<C> {
    electric: C,
    magnetic: C,
    total: C,
}

impl<C> AggregateHermitianOverlap<C> {
    pub(crate) const fn new(electric: C, magnetic: C, total: C) -> Self {
        Self {
            electric,
            magnetic,
            total,
        }
    }

    /// Return the aggregate electric-field overlap.
    pub fn electric(&self) -> &C {
        &self.electric
    }

    /// Return the aggregate magnetic-field overlap.
    pub fn magnetic(&self) -> &C {
        &self.magnetic
    }

    /// Return the aggregate total field overlap.
    pub fn total(&self) -> &C {
        &self.total
    }

    /// Consume the aggregate and return `(electric, magnetic, total)`.
    pub fn into_parts(self) -> (C, C, C) {
        (self.electric, self.magnetic, self.total)
    }

    /// Transform every aggregate component.
    pub fn map<U>(self, mut map: impl FnMut(C) -> U) -> AggregateHermitianOverlap<U> {
        AggregateHermitianOverlap {
            electric: map(self.electric),
            magnetic: map(self.magnetic),
            total: map(self.total),
        }
    }
}

/// Componentwise normalized Hermitian overlap between two solutions.
///
/// Each component is normalized independently:
///
/// ```text
/// electric = cross_electric
///          / sqrt(left_self_electric right_self_electric)
///
/// magnetic = cross_magnetic
///          / sqrt(left_self_magnetic right_self_magnetic)
///
/// total    = cross_total
///          / sqrt(left_self_total right_self_total)
/// ```
///
/// For nonzero self-overlaps, identical solutions have unit normalized
/// overlap. Orthogonal solutions have zero overlap under the selected form.
#[derive(Clone, Debug, PartialEq)]
pub struct NormalizedHermitianOverlap<C> {
    electric: C,
    magnetic: C,
    total: C,
}

impl<C> NormalizedHermitianOverlap<C> {
    const fn new(electric: C, magnetic: C, total: C) -> Self {
        Self {
            electric,
            magnetic,
            total,
        }
    }

    /// Return the normalized electric overlap.
    pub fn electric(&self) -> &C {
        &self.electric
    }

    /// Return the normalized magnetic overlap.
    pub fn magnetic(&self) -> &C {
        &self.magnetic
    }

    /// Return the normalized total overlap.
    pub fn total(&self) -> &C {
        &self.total
    }

    /// Consume the result and return `(electric, magnetic, total)`.
    pub fn into_parts(self) -> (C, C, C) {
        (self.electric, self.magnetic, self.total)
    }

    /// Transform every normalized component.
    pub fn map<U>(self, mut map: impl FnMut(C) -> U) -> NormalizedHermitianOverlap<U> {
        NormalizedHermitianOverlap {
            electric: map(self.electric),
            magnetic: map(self.magnetic),
            total: map(self.total),
        }
    }
}

impl<A> HermitianLayerOverlapInput<A> {
    fn integrate(
        self,
        left_vacuum_angular_wavenumber: &A,
        right_vacuum_angular_wavenumber: &A,
        left_parallel_angular_wavenumber: &A,
        right_parallel_angular_wavenumber: &A,
    ) -> Result<HermitianLayerOverlap<A>, HermitianOverlapError>
    where
        A: RealScalarAlgebra + ScalarAlgebraExpRelExt,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        let (left, right, thickness) = self.into_parts();

        let (left_waves, left_quantities) = left.into_parts();

        let (right_waves, right_quantities) = right.into_parts();

        let products = integrate_hermitian_cross_wave_products(
            &left_waves,
            &right_waves,
            left_quantities.kappa(),
            right_quantities.kappa(),
            &thickness,
        );

        let left_admittance = left_quantities.admittance().into_inner();

        let right_admittance = right_quantities.admittance().into_inner();

        let state = project_integrated_hermitian_cross_state_products(
            &products,
            &left_admittance,
            &right_admittance,
        );

        let field = project_integrated_hermitian_field_overlap(
            &state,
            &left_quantities,
            &right_quantities,
            left_vacuum_angular_wavenumber,
            right_vacuum_angular_wavenumber,
            left_parallel_angular_wavenumber,
            right_parallel_angular_wavenumber,
        )?;

        Ok(HermitianLayerOverlap::new(
            field.electric().clone(),
            field.magnetic().clone(),
            field.electric().add(field.magnetic()),
        ))
    }
}

impl<A> Layers<HermitianLayerOverlapInput<A>>
where
    A: RealScalarAlgebra + ScalarAlgebraExpRelExt,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    pub(crate) fn integrate(
        self,
        left_vacuum_angular_wavenumber: &A,
        right_vacuum_angular_wavenumber: &A,
        left_parallel_angular_wavenumber: &A,
        right_parallel_angular_wavenumber: &A,
    ) -> Result<Layers<HermitianLayerOverlap<A>>, HermitianOverlapError> {
        self.into_inner()
            .into_iter()
            .map(|layer| {
                layer.integrate(
                    left_vacuum_angular_wavenumber,
                    right_vacuum_angular_wavenumber,
                    left_parallel_angular_wavenumber,
                    right_parallel_angular_wavenumber,
                )
            })
            .collect::<Result<Vec<_>, _>>()
            .map(Layers::new)
    }
}

impl<C> Layers<HermitianLayerOverlap<C>>
where
    C: ScalarAlgebra,
{
    pub fn aggregate(&self) -> Result<AggregateHermitianOverlap<C>, LayerAggregateError> {
        let mut layers = self.iter();

        let first = layers.next().ok_or(LayerAggregateError::EmptyLayers)?;

        let mut electric = first.electric().clone();

        let mut magnetic = first.magnetic().clone();

        let mut total = first.total().clone();

        for layer in layers {
            electric = electric.add(layer.electric());

            magnetic = magnetic.add(layer.magnetic());

            total = total.add(layer.total());
        }

        Ok(AggregateHermitianOverlap::new(electric, magnetic, total))
    }
}

impl<C> AggregateHermitianOverlap<C>
where
    C: ScalarAlgebra,
{
    /// Normalize this cross-overlap by the corresponding left and right
    /// self-overlaps.
    pub fn normalized(&self, left_self: &Self, right_self: &Self) -> NormalizedHermitianOverlap<C> {
        let electric_denominator = left_self.electric().multiply(right_self.electric()).sqrt();

        let magnetic_denominator = left_self.magnetic().multiply(right_self.magnetic()).sqrt();

        let total_denominator = left_self.total().multiply(right_self.total()).sqrt();

        NormalizedHermitianOverlap::new(
            self.electric().divide(&electric_denominator),
            self.magnetic().divide(&magnetic_denominator),
            self.total().divide(&total_denominator),
        )
    }

    /// Return the componentwise normalized overlap of this result with itself.
    ///
    /// This returns unity for every nonzero component.
    pub fn self_normalized(&self) -> NormalizedHermitianOverlap<C> {
        self.normalized(self, self)
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Ix0, arr0};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        Polarisation,
        algebra::{ArrayJet0, ArrayJet1, Jet0, RealParameter},
        backend::IsotropicLayerQuantities,
    };

    type C = Complex64;

    type A0 = ArrayJet0<C, Ix0, RealParameter>;

    type A1 = ArrayJet1<C, Ix0, RealParameter>;

    const TOLERANCE: f64 = 1.0e-12;

    fn c(real: f64, imaginary: f64) -> C {
        C::new(real, imaginary)
    }

    fn jet(value: C) -> A0 {
        Jet0::new(arr0(value))
    }

    fn scalar(value: &A0) -> C {
        value.value()[()]
    }

    fn jet1(value: C, first: C) -> A1 {
        A1::from_parts(arr0(value), arr0(first))
    }

    fn assert_complex_relative_eq(actual: C, expected: C) {
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

    fn overlap(electric: C, magnetic: C) -> HermitianLayerOverlap<A0> {
        HermitianLayerOverlap::new(jet(electric), jet(magnetic), jet(electric + magnetic))
    }

    fn quantities(
        kappa: C,
        polarisation: Polarisation,
        epsilon: C,
        mu: C,
    ) -> IsotropicLayerQuantities<A0> {
        IsotropicLayerQuantities::test_fixture(jet(kappa), jet(epsilon), jet(mu), polarisation)
    }

    fn operand(forward: C, backward: C, kappa: C) -> LayerOverlapOperand<A0> {
        LayerOverlapOperand::new(
            BoundaryWaves::new(jet(forward), jet(backward)),
            quantities(
                kappa,
                Polarisation::TransverseElectric,
                c(2.0, 0.0),
                c(3.0, 0.0),
            ),
        )
    }

    #[test]
    fn layer_overlap_constructs_total_from_components() {
        let overlap =
            HermitianLayerOverlap::new(jet(c(2.0, 3.0)), jet(c(5.0, -1.0)), jet(c(7.0, 2.0)));

        assert_complex_relative_eq(scalar(overlap.total()), c(7.0, 2.0));
    }

    #[test]
    fn layer_overlap_into_parts_preserves_order() {
        let overlap =
            HermitianLayerOverlap::new(jet(c(1.0, 0.0)), jet(c(2.0, 0.0)), jet(c(3.0, 0.0)));

        let (electric, magnetic, total) = overlap.into_parts();

        assert_complex_relative_eq(electric.value()[()], c(1.0, 0.0));
        assert_complex_relative_eq(magnetic.value()[()], c(2.0, 0.0));
        assert_complex_relative_eq(total.value()[()], c(3.0, 0.0));
    }

    #[test]
    fn layer_overlap_map_transforms_all_components() {
        let overlap =
            HermitianLayerOverlap::new(jet(c(1.0, 0.0)), jet(c(2.0, 0.0)), jet(c(3.0, 0.0)));

        let mapped = overlap.map(|value| value.scale_by(c(10.0, 0.0)));

        assert_complex_relative_eq(mapped.electric().value()[()], c(10.0, 0.0));
        assert_complex_relative_eq(mapped.magnetic().value()[()], c(20.0, 0.0));
        assert_complex_relative_eq(mapped.total().value()[()], c(30.0, 0.0));
    }

    #[test]
    fn aggregation_sums_every_layer_component() {
        let layers = Layers::new(vec![
            overlap(c(1.0, 2.0), c(3.0, 4.0)),
            overlap(c(5.0, -1.0), c(7.0, 2.0)),
        ]);

        let aggregate = layers.aggregate().unwrap();

        assert_complex_relative_eq(scalar(aggregate.electric()), c(6.0, 1.0));

        assert_complex_relative_eq(scalar(aggregate.magnetic()), c(10.0, 6.0));

        assert_complex_relative_eq(scalar(aggregate.total()), c(16.0, 7.0));
    }

    #[test]
    fn empty_overlap_sequence_cannot_be_aggregated() {
        let layers: Layers<HermitianLayerOverlap<A0>> = Layers::new(Vec::new());

        assert_eq!(layers.aggregate(), Err(LayerAggregateError::EmptyLayers),);
    }

    #[test]
    fn normalized_self_overlap_is_unity() {
        let aggregate =
            AggregateHermitianOverlap::new(jet(c(4.0, 0.0)), jet(c(9.0, 0.0)), jet(c(13.0, 0.0)));

        let normalized = aggregate.normalized(&aggregate, &aggregate);

        assert_complex_relative_eq(scalar(normalized.electric()), c(1.0, 0.0));

        assert_complex_relative_eq(scalar(normalized.magnetic()), c(1.0, 0.0));

        assert_complex_relative_eq(scalar(normalized.total()), c(1.0, 0.0));
    }

    #[test]
    fn normalization_uses_componentwise_self_overlaps() {
        let cross = AggregateHermitianOverlap::new(
            jet(c(6.0, 2.0)),
            jet(c(10.0, -4.0)),
            jet(c(16.0, -2.0)),
        );

        let left =
            AggregateHermitianOverlap::new(jet(c(4.0, 0.0)), jet(c(9.0, 0.0)), jet(c(13.0, 0.0)));

        let right =
            AggregateHermitianOverlap::new(jet(c(16.0, 0.0)), jet(c(25.0, 0.0)), jet(c(41.0, 0.0)));

        let normalized = cross.normalized(&left, &right);

        assert_complex_relative_eq(scalar(normalized.electric()), c(6.0, 2.0) / 8.0);

        assert_complex_relative_eq(scalar(normalized.magnetic()), c(10.0, -4.0) / 15.0);

        assert_complex_relative_eq(
            scalar(normalized.total()),
            c(16.0, -2.0) / (13.0_f64 * 41.0).sqrt(),
        );
    }

    #[test]
    fn normalization_propagates_first_derivatives() {
        /*
         * Cross:
         *   x = 2 + 3p
         *
         * Left self:
         *   l = 4 + 5p
         *
         * Right self:
         *   r = 9 + 7p
         *
         * n = x / sqrt(l r)
         *
         * n(0) = 1/3
         */
        let cross = AggregateHermitianOverlap::new(
            jet1(c(2.0, 0.0), c(3.0, 0.0)),
            jet1(c(2.0, 0.0), c(3.0, 0.0)),
            jet1(c(2.0, 0.0), c(3.0, 0.0)),
        );

        let left = AggregateHermitianOverlap::new(
            jet1(c(4.0, 0.0), c(5.0, 0.0)),
            jet1(c(4.0, 0.0), c(5.0, 0.0)),
            jet1(c(4.0, 0.0), c(5.0, 0.0)),
        );

        let right = AggregateHermitianOverlap::new(
            jet1(c(9.0, 0.0), c(7.0, 0.0)),
            jet1(c(9.0, 0.0), c(7.0, 0.0)),
            jet1(c(9.0, 0.0), c(7.0, 0.0)),
        );

        let normalized = cross.normalized(&left, &right);

        let expected_value = 2.0 / 6.0;

        let denominator_first = 0.5 * 6.0 * (5.0 / 4.0 + 7.0 / 9.0);

        let expected_first = 3.0 / 6.0 - 2.0 * denominator_first / 36.0;

        assert_relative_eq!(
            normalized.total().value()[()].re,
            expected_value,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            normalized.total().first()[()].re,
            expected_first,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn one_layer_integration_returns_finite_overlap() {
        let input = HermitianLayerOverlapInput::new(
            operand(c(0.8, 0.2), c(-0.1, 0.3), c(1.4, 0.1)),
            operand(c(0.6, -0.4), c(0.2, 0.1), c(1.1, 0.05)),
            jet(c(0.7, 0.0)),
        );

        let overlap = input
            .integrate(
                &jet(c(2.0, 0.0)),
                &jet(c(2.3, 0.0)),
                &jet(c(0.4, 0.0)),
                &jet(c(0.5, 0.0)),
            )
            .unwrap();

        for value in [
            scalar(overlap.electric()),
            scalar(overlap.magnetic()),
            scalar(overlap.total()),
        ] {
            assert!(value.re.is_finite());
            assert!(value.im.is_finite());
        }
    }

    #[test]
    fn sequence_integration_preserves_layer_count_and_order() {
        let first = HermitianLayerOverlapInput::new(
            operand(c(0.5, 0.0), c(0.0, 0.0), c(1.4, 0.0)),
            operand(c(0.7, 0.0), c(0.0, 0.0), c(1.1, 0.0)),
            jet(c(0.7, 0.0)),
        );

        let second = HermitianLayerOverlapInput::new(
            operand(c(1.5, 0.0), c(0.0, 0.0), c(1.4, 0.0)),
            operand(c(1.7, 0.0), c(0.0, 0.0), c(1.1, 0.0)),
            jet(c(0.7, 0.0)),
        );

        let overlaps = Layers::new(vec![first, second])
            .integrate(
                &jet(c(2.0, 0.0)),
                &jet(c(2.3, 0.0)),
                &jet(c(0.4, 0.0)),
                &jet(c(0.5, 0.0)),
            )
            .unwrap();

        assert_eq!(overlaps.len(), 2);

        assert!(
            scalar(overlaps.first().unwrap().total()).re
                < scalar(overlaps.last().unwrap().total()).re,
            "larger amplitudes must remain in the second layer",
        );
    }
}
