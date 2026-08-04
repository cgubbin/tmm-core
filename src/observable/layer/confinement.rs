//! Energy confinement within selected finite layers.

use crate::{FiniteLayerIndex, algebra::ScalarAlgebra};

use super::{LayerEnergy, LayerEnergyError, Layers, aggregate::LayerAggregateError};

#[derive(Debug, thiserror::Error)]
pub enum LayerConfinementError {
    #[error(transparent)]
    Energy(#[from] LayerEnergyError),

    #[error(transparent)]
    Aggregate(#[from] LayerAggregateError),
}

/// Fraction of aggregate finite-layer energy confined to a selected region.
///
/// Each component is normalized independently against the corresponding
/// aggregate over all finite layers.
#[derive(Clone, Debug, PartialEq)]
pub struct EnergyConfinement<R> {
    electric: R,
    magnetic: R,
    total: R,
}

impl<R> EnergyConfinement<R> {
    pub(crate) const fn new(electric: R, magnetic: R, total: R) -> Self {
        Self {
            electric,
            magnetic,
            total,
        }
    }

    /// Return the selected region's electric-energy confinement.
    pub fn electric(&self) -> &R {
        &self.electric
    }

    /// Return the selected region's magnetic-energy confinement.
    pub fn magnetic(&self) -> &R {
        &self.magnetic
    }

    /// Return the selected region's total-energy confinement.
    pub fn total(&self) -> &R {
        &self.total
    }

    pub fn into_parts(self) -> (R, R, R) {
        (self.electric, self.magnetic, self.total)
    }
}

impl<R> Layers<LayerEnergy<R>>
where
    R: ScalarAlgebra,
{
    /// Calculate energy confinement in the finite layers selected by
    /// `include`.
    ///
    /// The selector receives each typed finite-layer index and its energy
    /// record. It is evaluated once per layer in physical order.
    pub fn confinement_by(
        &self,
        mut include: impl FnMut(FiniteLayerIndex, &LayerEnergy<R>) -> bool,
    ) -> Result<EnergyConfinement<R>, LayerAggregateError> {
        let aggregate = self.aggregate()?;

        let mut selected = self.iter().enumerate().filter_map(|(index, layer)| {
            let index = FiniteLayerIndex(index);

            include(index, layer).then_some(layer)
        });

        let first = selected.next().ok_or(LayerAggregateError::EmptySelection)?;

        let mut electric = first.electric().clone();

        let mut magnetic = first.magnetic().clone();

        let mut total = first.total().clone();

        for layer in selected {
            electric = electric.add(layer.electric());

            magnetic = magnetic.add(layer.magnetic());

            total = total.add(layer.total());
        }

        Ok(EnergyConfinement::new(
            electric.divide(aggregate.electric()),
            magnetic.divide(aggregate.magnetic()),
            total.divide(aggregate.total()),
        ))
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{Ix0, arr0};

    use super::*;

    use crate::{
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
    fn confinement_preserves_component_order() {
        let confinement = EnergyConfinement::new(1, 2, 3);

        assert_eq!(confinement.electric(), &1);
        assert_eq!(confinement.magnetic(), &2);
        assert_eq!(confinement.total(), &3);

        assert_eq!(confinement.into_parts(), (1, 2, 3),);
    }

    #[test]
    fn selecting_every_layer_gives_unit_confinement() {
        let layers = Layers::new(vec![energy(2.0, 3.0), energy(5.0, 7.0), energy(11.0, 13.0)]);

        let confinement = layers.confinement_by(|_, _| true).unwrap();

        assert_relative_eq!(
            scalar(confinement.electric()),
            1.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(confinement.magnetic()),
            1.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(confinement.total()),
            1.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn selecting_one_layer_matches_its_participation() {
        let layers = Layers::new(vec![energy(2.0, 5.0), energy(3.0, 7.0), energy(11.0, 13.0)]);

        let participation = layers.participation().unwrap();

        let confinement = layers
            .confinement_by(|index, _| index == FiniteLayerIndex(1))
            .unwrap();

        let expected = participation.get(FiniteLayerIndex(1)).unwrap();

        assert_eq!(confinement.electric(), expected.electric(),);

        assert_eq!(confinement.magnetic(), expected.magnetic(),);

        assert_eq!(confinement.total(), expected.total(),);
    }

    #[test]
    fn confinement_sums_selected_layer_participations() {
        let layers = Layers::new(vec![energy(1.0, 2.0), energy(3.0, 5.0), energy(6.0, 11.0)]);

        let participation = layers.participation().unwrap();

        let confinement = layers
            .confinement_by(|index, _| index == FiniteLayerIndex(0) || index == FiniteLayerIndex(2))
            .unwrap();

        let expected_electric = scalar(participation.get(FiniteLayerIndex(0)).unwrap().electric())
            + scalar(participation.get(FiniteLayerIndex(2)).unwrap().electric());

        let expected_magnetic = scalar(participation.get(FiniteLayerIndex(0)).unwrap().magnetic())
            + scalar(participation.get(FiniteLayerIndex(2)).unwrap().magnetic());

        let expected_total = scalar(participation.get(FiniteLayerIndex(0)).unwrap().total())
            + scalar(participation.get(FiniteLayerIndex(2)).unwrap().total());

        assert_relative_eq!(
            scalar(confinement.electric()),
            expected_electric,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(confinement.magnetic()),
            expected_magnetic,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(confinement.total()),
            expected_total,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn selector_receives_physical_layer_indices() {
        let layers = Layers::new(vec![energy(1.0, 1.0), energy(2.0, 2.0), energy(3.0, 3.0)]);

        let mut visited = Vec::new();

        let _ = layers
            .confinement_by(|index, _| {
                visited.push(index);
                index == FiniteLayerIndex(1)
            })
            .unwrap();

        assert_eq!(
            visited,
            vec![
                FiniteLayerIndex(0),
                FiniteLayerIndex(1),
                FiniteLayerIndex(2),
            ],
        );
    }

    #[test]
    fn empty_selection_is_rejected() {
        let layers = Layers::new(vec![energy(2.0, 3.0), energy(5.0, 7.0)]);

        assert_eq!(
            layers.confinement_by(|_, _| false),
            Err(LayerAggregateError::EmptySelection),
        );
    }

    #[test]
    fn empty_layer_sequence_is_rejected_before_selection() {
        let layers: Layers<LayerEnergy<R0>> = Layers::new(Vec::new());

        let mut called = false;

        let result = layers.confinement_by(|_, _| {
            called = true;
            true
        });

        assert_eq!(result, Err(LayerAggregateError::EmptyLayers),);

        assert!(!called, "the selector must not run for an empty sequence",);
    }

    #[test]
    fn confinement_propagates_first_derivatives() {
        /*
         * Selected layer:
         *   S = 2 + 3p
         *
         * Unselected layer:
         *   U = 6 + 5p
         *
         * Aggregate:
         *   A = 8 + 8p
         *
         * C = S/A
         *
         * C(0)  = 1/4
         * C'(0) = (3*8 - 2*8)/64 = 1/8
         */
        let layers = Layers::new(vec![
            LayerEnergy::new(jet1(2.0, 3.0), jet1(2.0, 3.0), jet1(2.0, 3.0)),
            LayerEnergy::new(jet1(6.0, 5.0), jet1(6.0, 5.0), jet1(6.0, 5.0)),
        ]);

        let confinement = layers
            .confinement_by(|index, _| index == FiniteLayerIndex(0))
            .unwrap();

        assert_relative_eq!(
            confinement.total().value()[()],
            0.25,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            confinement.total().first()[()],
            0.125,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }

    #[test]
    fn complementary_regions_have_confinements_summing_to_one() {
        let layers = Layers::new(vec![energy(2.0, 3.0), energy(5.0, 7.0), energy(11.0, 13.0)]);

        let left = layers.confinement_by(|index, _| index.0 < 2).unwrap();

        let right = layers.confinement_by(|index, _| index.0 >= 2).unwrap();

        assert_relative_eq!(
            scalar(left.electric()) + scalar(right.electric()),
            1.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(left.magnetic()) + scalar(right.magnetic()),
            1.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );

        assert_relative_eq!(
            scalar(left.total()) + scalar(right.total()),
            1.0,
            epsilon = TOLERANCE,
            max_relative = TOLERANCE,
        );
    }
}
