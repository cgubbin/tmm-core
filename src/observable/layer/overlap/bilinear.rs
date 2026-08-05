use ndarray::Dimension;

use crate::{
    ComplexScalar,
    algebra::{RealScalarAlgebra, ScalarAlgebra, ScalarAlgebraExpRelExt},
};

use super::super::{
    LayerAggregateError, LayerOverlapOperand, Layers, integration::BilinearOverlapError,
};

/// Matched left and right solution data for one physical finite layer.
///
/// The two operands must refer to the same physical layer. `thickness` is the
/// common physical integration interval.
#[derive(Clone, Debug, PartialEq)]
pub struct BilinearLayerOverlapInput<A> {
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
    pub fn map<U>(self, mut map: impl FnMut(C) -> U) -> AggregateBilinearOverlap<U> {
        AggregateBilinearOverlap {
            electric: map(self.electric),
            magnetic: map(self.magnetic),
            total: map(self.total),
        }
    }
}

/// Componentwise normalized Bilinear overlap between two solutions.
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
pub struct NormalizedBilinearOverlap<C> {
    electric: C,
    magnetic: C,
    total: C,
}

impl<C> NormalizedBilinearOverlap<C> {
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
    pub fn map<U>(self, mut map: impl FnMut(C) -> U) -> NormalizedBilinearOverlap<U> {
        NormalizedBilinearOverlap {
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
    ) -> Result<BilinearLayerOverlap<A>, HermitianOverlapError>
    where
        A: RealScalarAlgebra + ScalarAlgebraExpRelExt,
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
        )?;

        Ok(BilinearLayerOverlap::new(
            field.electric().clone(),
            field.magnetic().clone(),
            field.electric().add(field.magnetic()),
        ))
    }
}

impl<A> Layers<BilinearLayerOverlapInput<A>>
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
    ) -> Result<Layers<BilinearLayerOverlap<A>>, HermitianOverlapError> {
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

        Ok(AggregateBilinearOverlap::new(electric, magnetic, total))
    }
}

impl<C> AggregateBilinearOverlap<C>
where
    C: ScalarAlgebra,
{
    /// Normalize this cross-overlap by the corresponding left and right
    /// self-overlaps.
    pub fn normalized(&self, left_self: &Self, right_self: &Self) -> NormalizedBilinearOverlap<C> {
        let electric_denominator = left_self.electric().multiply(right_self.electric()).sqrt();

        let magnetic_denominator = left_self.magnetic().multiply(right_self.magnetic()).sqrt();

        let total_denominator = left_self.total().multiply(right_self.total()).sqrt();

        NormalizedBilinearOverlap::new(
            self.electric().divide(&electric_denominator),
            self.magnetic().divide(&magnetic_denominator),
            self.total().divide(&total_denominator),
        )
    }

    /// Return the componentwise normalized overlap of this result with itself.
    ///
    /// This returns unity for every nonzero component.
    pub fn self_normalized(&self) -> NormalizedBilinearOverlap<C> {
        self.normalized(self, self)
    }
}
