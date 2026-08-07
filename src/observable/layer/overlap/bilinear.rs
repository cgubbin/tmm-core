use ndarray::Dimension;

use crate::{
    ComplexScalar,
    algebra::{ScalarAlgebra, ScalarAlgebraExpRelExt},
    observable::layer::integration::{
        integrate_bilinear_cross_wave_products, project_integrated_bilinear_cross_state_products,
        project_integrated_bilinear_field_overlap,
    },
};

use super::{
    super::{LayerAggregateError, LayerOverlapOperand, Layers},
    OverlapError,
};

/// Dispersive bilinear contribution to a quasinormal-mode normalization.
///
/// ```text
/// electric = ∫ E · ∂(k0 ε)/∂k0 E dz
/// magnetic = ∫ H · ∂(k0 μ)/∂k0 H dz
/// total    = electric - magnetic
/// ```
///
/// No complex conjugation is applied.
#[derive(Clone, Debug, PartialEq)]
pub struct BilinearLayerNormalization<C> {
    electric: C,
    magnetic: C,
    total: C,
}

impl<C> BilinearLayerNormalization<C> {
    pub(crate) fn new(electric: C, magnetic: C) -> Self
    where
        C: ScalarAlgebra,
    {
        let total = electric.subtract(&magnetic);

        Self {
            electric,
            magnetic,
            total,
        }
    }

    pub(crate) const fn from_parts(electric: C, magnetic: C, total: C) -> Self {
        Self {
            electric,
            magnetic,
            total,
        }
    }

    pub fn electric(&self) -> &C {
        &self.electric
    }

    pub fn magnetic(&self) -> &C {
        &self.magnetic
    }

    pub fn total(&self) -> &C {
        &self.total
    }

    pub fn into_parts(self) -> (C, C, C) {
        (self.electric, self.magnetic, self.total)
    }

    pub fn normalise_by(mut self, total: &C) -> Self
    where
        C: ScalarAlgebra,
    {
        self.electric = self.electric.divide(total);
        self.magnetic = self.magnetic.divide(total);
        self.total = self.total.divide(total);

        self
    }

    pub fn map<U>(self, mut map: impl FnMut(C) -> U) -> BilinearLayerNormalization<U> {
        BilinearLayerNormalization {
            electric: map(self.electric),
            magnetic: map(self.magnetic),
            total: map(self.total),
        }
    }
}

pub struct AggregateBilinearNormalization<C> {
    electric: C,
    magnetic: C,
    total: C,
}

impl<C> AggregateBilinearNormalization<C> {
    pub(crate) const fn from_parts(electric: C, magnetic: C, total: C) -> Self {
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
    pub fn map<U>(self, mut map: impl FnMut(C) -> U) -> AggregateBilinearNormalization<U> {
        AggregateBilinearNormalization {
            electric: map(self.electric),
            magnetic: map(self.magnetic),
            total: map(self.total),
        }
    }
}

impl<A> Layers<BilinearLayerNormalization<A>> {
    pub(crate) fn aggregate(&self) -> Result<AggregateBilinearNormalization<A>, LayerAggregateError>
    where
        A: ScalarAlgebra,
    {
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

        Ok(AggregateBilinearNormalization::from_parts(
            electric, magnetic, total,
        ))
    }
}

fn bilinear_normalization_coefficients<A>(
    vacuum_angular_wavenumber: &A,
    epsilon: &A,
    epsilon_first: &A,
    mu: &A,
    mu_first: &A,
) -> (A, A)
where
    A: ScalarAlgebra,
{
    let electric = epsilon.add(&vacuum_angular_wavenumber.multiply(epsilon_first));

    let magnetic = mu.add(&vacuum_angular_wavenumber.multiply(mu_first));

    (electric, magnetic)
}

/// Matched left and right solution data for one physical finite layer.
///
/// The two operands must refer to the same physical layer. `thickness` is the
/// common physical integration interval.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BilinearLayerOverlapInput<A> {
    left: LayerOverlapOperand<A>,
    right: LayerOverlapOperand<A>,
    thickness: A,
}

impl<A> BilinearLayerOverlapInput<A> {
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

/// Integrated Bilinear field overlap within one finite layer.
///
/// ```text
/// electric = ∫ E_left · E_right dz
/// magnetic = ∫ H_left · H_right dz
/// total    = electric + magnetic
/// ```
///
/// This is an unweighted field overlap. Constitutive or perturbative weights
/// are applied by later projections.
#[derive(Clone, Debug, PartialEq)]
pub struct BilinearLayerOverlap<C> {
    electric: C,
    magnetic: C,
    total: C,
}

impl<C> BilinearLayerOverlap<C> {
    pub(crate) fn from_parts(electric: C, magnetic: C, total: C) -> Self {
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
    pub fn map<U>(self, mut map: impl FnMut(C) -> U) -> BilinearLayerOverlap<U> {
        BilinearLayerOverlap {
            electric: map(self.electric),
            magnetic: map(self.magnetic),
            total: map(self.total),
        }
    }
}

/// Bilinear field overlap aggregated over all finite layers.
///
/// Exterior media are not included.
#[derive(Clone, Debug, PartialEq)]
pub struct AggregateBilinearOverlap<C> {
    electric: C,
    magnetic: C,
    total: C,
}

impl<C> AggregateBilinearOverlap<C> {
    pub(crate) const fn from_parts(electric: C, magnetic: C, total: C) -> Self {
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
    pub fn map<U>(self, mut map: impl FnMut(C) -> U) -> AggregateBilinearOverlap<U> {
        AggregateBilinearOverlap {
            electric: map(self.electric),
            magnetic: map(self.magnetic),
            total: map(self.total),
        }
    }
}

impl<A> BilinearLayerOverlapInput<A> {
    fn integrate(
        self,
        left_vacuum_angular_wavenumber: &A,
        right_vacuum_angular_wavenumber: &A,
        left_parallel_angular_wavenumber: &A,
        right_parallel_angular_wavenumber: &A,
    ) -> Result<BilinearLayerOverlap<A>, OverlapError>
    where
        A: ScalarAlgebra + ScalarAlgebraExpRelExt,
        A::Scalar: ComplexScalar,
        A::Dimension: Dimension,
    {
        let (left, right, thickness) = self.into_parts();

        let (left_waves, left_quantities) = left.into_parts();

        let (right_waves, right_quantities) = right.into_parts();

        let products = integrate_bilinear_cross_wave_products(
            &left_waves,
            &right_waves,
            left_quantities.kappa(),
            right_quantities.kappa(),
            &thickness,
        );

        let left_admittance = left_quantities.admittance().into_inner();

        let right_admittance = right_quantities.admittance().into_inner();

        let left_polarisation = left_quantities.polarisation();

        let right_polarisation = right_quantities.polarisation();

        if left_polarisation != right_polarisation {
            return Err(OverlapError::PolarisationMismatch {
                reference: left_polarisation,
                comparison: right_polarisation,
            });
        }

        let state = project_integrated_bilinear_cross_state_products(
            &products,
            &left_admittance,
            &right_admittance,
        );

        let field = project_integrated_bilinear_field_overlap(
            &state,
            &left_quantities,
            &right_quantities,
            left_vacuum_angular_wavenumber,
            right_vacuum_angular_wavenumber,
            left_parallel_angular_wavenumber,
            right_parallel_angular_wavenumber,
        );

        Ok(BilinearLayerOverlap::from_parts(
            field.electric().clone(),
            field.magnetic().clone(),
            field.electric().add(field.magnetic()),
        ))
    }
}

impl<A> Layers<BilinearLayerOverlapInput<A>>
where
    A: ScalarAlgebra + ScalarAlgebraExpRelExt,
    A::Scalar: ComplexScalar,
    A::Dimension: Dimension,
{
    pub(crate) fn integrate(
        self,
        left_vacuum_angular_wavenumber: &A,
        right_vacuum_angular_wavenumber: &A,
        left_parallel_angular_wavenumber: &A,
        right_parallel_angular_wavenumber: &A,
    ) -> Result<Layers<BilinearLayerOverlap<A>>, OverlapError> {
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

impl<C> Layers<BilinearLayerOverlap<C>>
where
    C: ScalarAlgebra,
{
    pub fn aggregate(&self) -> Result<AggregateBilinearOverlap<C>, LayerAggregateError> {
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

        Ok(AggregateBilinearOverlap::from_parts(
            electric, magnetic, total,
        ))
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
        observable::Layers,
    };

    type A = ArrayJet0<Complex64, Ix0, RealParameter>;

    const TOLERANCE: f64 = 1.0e-12;

    fn jet(value: Complex64) -> A {
        Jet0::new(arr0(value))
    }

    fn scalar(value: &A) -> Complex64 {
        value.value()[()]
    }

    fn assert_complex_close(actual: Complex64, expected: Complex64) {
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
    fn bilinear_layer_overlap_into_parts_preserves_order() {
        let overlap = BilinearLayerOverlap::from_parts(
            jet(Complex64::new(1.0, 0.0)),
            jet(Complex64::new(2.0, 0.0)),
            jet(Complex64::new(3.0, 0.0)),
        );

        let (electric, magnetic, total) = overlap.into_parts();

        assert_eq!(scalar(&electric), Complex64::new(1.0, 0.0));
        assert_eq!(scalar(&magnetic), Complex64::new(2.0, 0.0));
        assert_eq!(scalar(&total), Complex64::new(3.0, 0.0));
    }

    #[test]
    fn bilinear_layer_overlap_map_transforms_every_component() {
        let overlap = BilinearLayerOverlap::from_parts(1, 2, 3);

        let mapped = overlap.map(|value| format!("value-{value}"));

        assert_eq!(mapped.electric(), "value-1");
        assert_eq!(mapped.magnetic(), "value-2");
        assert_eq!(mapped.total(), "value-3");
    }

    #[test]
    fn aggregate_sums_all_layers() {
        let layers = Layers::new(vec![
            BilinearLayerOverlap::from_parts(
                jet(Complex64::new(1.0, 1.0)),
                jet(Complex64::new(2.0, 0.0)),
                jet(Complex64::new(3.0, 1.0)),
            ),
            BilinearLayerOverlap::from_parts(
                jet(Complex64::new(3.0, -1.0)),
                jet(Complex64::new(4.0, 2.0)),
                jet(Complex64::new(7.0, 1.0)),
            ),
        ]);

        let aggregate = layers.aggregate().unwrap();

        assert_complex_close(scalar(aggregate.electric()), Complex64::new(4.0, 0.0));

        assert_complex_close(scalar(aggregate.magnetic()), Complex64::new(6.0, 2.0));

        assert_complex_close(scalar(aggregate.total()), Complex64::new(10.0, 2.0));
    }

    #[test]
    fn aggregate_rejects_empty_layers() {
        let layers: Layers<BilinearLayerOverlap<A>> = Layers::new(Vec::new());

        let error = layers.aggregate().unwrap_err();

        assert_eq!(error, LayerAggregateError::EmptyLayers);
    }

    #[test]
    fn aggregate_total_is_component_sum() {
        let layers = Layers::new(vec![
            BilinearLayerOverlap::from_parts(
                jet(Complex64::new(1.0, 1.0)),
                jet(Complex64::new(2.0, 0.0)),
                jet(Complex64::new(3.0, 1.0)),
            ),
            BilinearLayerOverlap::from_parts(
                jet(Complex64::new(3.0, -1.0)),
                jet(Complex64::new(4.0, 2.0)),
                jet(Complex64::new(7.0, 1.0)),
            ),
        ]);

        let aggregate = layers.aggregate().unwrap();

        assert_complex_close(
            scalar(aggregate.total()),
            scalar(aggregate.electric()) + scalar(aggregate.magnetic()),
        );
    }

    #[test]
    fn aggregate_map_transforms_every_component() {
        let aggregate = AggregateBilinearOverlap::from_parts(1, 2, 3);

        let mapped = aggregate.map(|value| value * 10);

        assert_eq!(mapped.electric(), &10);
        assert_eq!(mapped.magnetic(), &20);
        assert_eq!(mapped.total(), &30);
    }
}
