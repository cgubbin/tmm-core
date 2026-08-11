//! Finite-layer energy participation factors.

use crate::algebra::ScalarAlgebra;

use super::{LayerEnergy, LayerEnergyError, Layers, aggregate::LayerAggregateError};

#[derive(Debug, thiserror::Error)]
pub enum LayerParticipationError {
    #[error(transparent)]
    Energy(#[from] LayerEnergyError),

    #[error(transparent)]
    Aggregate(#[from] LayerAggregateError),
}

/// Fraction of aggregate finite-layer energy associated with one layer.
///
/// Every component is normalized independently:
///
/// ```text
/// electric = layer electric / aggregate electric
/// magnetic = layer magnetic / aggregate magnetic
/// total    = layer total    / aggregate total
/// ```
///
/// Therefore, each component sums to unity across all finite layers when its
/// corresponding aggregate denominator is nonzero.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerParticipation<R> {
    electric: R,
    magnetic: R,
    total: R,
}

impl<R> LayerParticipation<R> {
    pub(crate) const fn new(electric: R, magnetic: R, total: R) -> Self {
        Self {
            electric,
            magnetic,
            total,
        }
    }

    /// Return this layer's fraction of aggregate electric energy.
    pub fn electric(&self) -> &R {
        &self.electric
    }

    /// Return this layer's fraction of aggregate magnetic energy.
    pub fn magnetic(&self) -> &R {
        &self.magnetic
    }

    /// Return this layer's fraction of aggregate total energy.
    pub fn total(&self) -> &R {
        &self.total
    }

    /// Consume the result and return `(electric, magnetic, total)`.
    pub fn into_parts(self) -> (R, R, R) {
        (self.electric, self.magnetic, self.total)
    }

    /// Transform every participation component.
    pub fn map<U>(self, mut map: impl FnMut(R) -> U) -> LayerParticipation<U> {
        LayerParticipation {
            electric: map(self.electric),
            magnetic: map(self.magnetic),
            total: map(self.total),
        }
    }
}

impl<R> Layers<LayerEnergy<R>>
where
    R: ScalarAlgebra,
{
    /// Calculate componentwise finite-layer participation factors.
    pub fn participation(&self) -> Result<Layers<LayerParticipation<R>>, LayerAggregateError> {
        let aggregate = self.aggregate()?;

        Ok(Layers::new(
            self.iter()
                .map(|layer| {
                    LayerParticipation::new(
                        layer.electric().divide(aggregate.electric()),
                        layer.magnetic().divide(aggregate.magnetic()),
                        layer.total().divide(aggregate.total()),
                    )
                })
                .collect(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Ix0, arr0};

    use super::*;

    use crate::{
        FiniteLayerIndex,
        algebra::{ArrayJet0, ArrayJet1, Jet0, RealParameter},
        observable::LayerEnergy,
    };

    type R0 = ArrayJet0<f64, Ix0, RealParameter>;

    type R1 = ArrayJet1<f64, Ix0, RealParameter>;

    const TOLERANCE: f64 = 1.0e-12;

    fn jet(value: f64) -> R0 {
        Jet0::new(arr0(value))
    }

    fn scalar(value: &R0) -> f64 {
        value.value()[()]
    }

    fn jet1(value: f64, first: f64) -> R1 {
        R1::from_parts(arr0(value), arr0(first))
    }

    fn energy(electric: f64, magnetic: f64) -> LayerEnergy<R0> {
        LayerEnergy::new(jet(electric), jet(magnetic), jet(electric + magnetic))
    }

    #[test]
    fn layer_participation_preserves_component_order() {
        let participation = LayerParticipation::new(1, 2, 3);

        assert_eq!(participation.electric(), &1);
        assert_eq!(participation.magnetic(), &2);
        assert_eq!(participation.total(), &3);

        assert_eq!(participation.into_parts(), (1, 2, 3),);
    }

    #[test]
    fn layer_participation_map_transforms_all_components() {
        let participation = LayerParticipation::new(1, 2, 3);

        let mapped = participation.map(|value| value * 10);

        assert_eq!(mapped.electric(), &10);
        assert_eq!(mapped.magnetic(), &20);
        assert_eq!(mapped.total(), &30);
    }

    #[test]
    fn layer_participation_map_supports_non_clone_storage() {
        #[derive(Debug, PartialEq)]
        struct NonClone(i32);

        let participation = LayerParticipation::new(NonClone(1), NonClone(2), NonClone(3));

        let mapped = participation.map(|value| value.0 * 10);

        assert_eq!(mapped.electric(), &10);
        assert_eq!(mapped.magnetic(), &20);
        assert_eq!(mapped.total(), &30);
    }

    #[test]
    fn one_layer_has_unit_participation() {
        let layers = Layers::new(vec![energy(2.0, 3.0)]);

        let participation = layers.participation().unwrap();

        assert_eq!(participation.len(), 1);

        let layer = participation.first().unwrap();

        assert_relative_eq!(
            scalar(layer.electric()),
            1.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(layer.magnetic()),
            1.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(layer.total()),
            1.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn participation_is_normalized_componentwise() {
        let layers = Layers::new(vec![energy(2.0, 3.0), energy(6.0, 9.0)]);

        let participation = layers.participation().unwrap();

        let first = participation.first().unwrap();

        let second = participation.last().unwrap();

        assert_relative_eq!(
            scalar(first.electric()),
            0.25,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(second.electric()),
            0.75,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(first.magnetic()),
            0.25,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(second.magnetic()),
            0.75,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(first.total()),
            0.25,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(second.total()),
            0.75,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn participation_components_sum_to_unity() {
        let layers = Layers::new(vec![energy(2.0, 5.0), energy(3.0, 7.0), energy(11.0, 13.0)]);

        let participation = layers.participation().unwrap();

        let electric_sum: f64 = participation
            .iter()
            .map(|layer| scalar(layer.electric()))
            .sum();

        let magnetic_sum: f64 = participation
            .iter()
            .map(|layer| scalar(layer.magnetic()))
            .sum();

        let total_sum: f64 = participation
            .iter()
            .map(|layer| scalar(layer.total()))
            .sum();

        assert_relative_eq!(
            electric_sum,
            1.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            magnetic_sum,
            1.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            total_sum,
            1.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn participation_preserves_physical_layer_order() {
        let layers = Layers::new(vec![energy(1.0, 1.0), energy(3.0, 3.0), energy(6.0, 6.0)]);

        let participation = layers.participation().unwrap();

        assert_relative_eq!(
            scalar(participation.first().unwrap().total(),),
            0.1,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(participation.get(FiniteLayerIndex::new(1)).unwrap().total(),),
            0.3,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(participation.last().unwrap().total(),),
            0.6,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn empty_layer_sequence_cannot_produce_participation() {
        let layers: Layers<LayerEnergy<R0>> = Layers::new(Vec::new());

        assert_eq!(
            layers.participation(),
            Err(LayerAggregateError::EmptyLayers),
        );
    }

    #[test]
    fn participation_propagates_first_derivatives() {
        /*
         * Layer energies:
         *
         * E0 = 2 + 3p
         * E1 = 6 + 5p
         *
         * aggregate = 8 + 8p
         *
         * P0 = E0 / aggregate
         *
         * P0(0)  = 1/4
         * P0'(0) = (3*8 - 2*8) / 8²
         *        = 1/8
         */
        let layers = Layers::new(vec![
            LayerEnergy::new(jet1(2.0, 3.0), jet1(2.0, 3.0), jet1(2.0, 3.0)),
            LayerEnergy::new(jet1(6.0, 5.0), jet1(6.0, 5.0), jet1(6.0, 5.0)),
        ]);

        let participation = layers.participation().unwrap();

        let first = participation.first().unwrap();

        assert_relative_eq!(
            first.total().value()[()],
            0.25,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            first.total().first()[()],
            0.125,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        /*
         * Participation derivatives must sum to zero because
         * the participation values sum identically to one.
         */
        let derivative_sum: f64 = participation
            .iter()
            .map(|layer| layer.total().first()[()])
            .sum();

        assert_relative_eq!(
            derivative_sum,
            0.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }
}
