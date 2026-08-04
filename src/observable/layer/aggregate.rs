//! Aggregate finite-layer observables.

use crate::algebra::ScalarAlgebra;

use super::{LayerEnergy, Layers};

/// Total electromagnetic energy integrated over all finite layers.
///
/// Each component is the sum of the corresponding component from every
/// finite-layer record:
///
/// ```text
/// electric = sum_i electric_i
/// magnetic = sum_i magnetic_i
/// total    = sum_i total_i
/// ```
///
/// Exterior media are not included.
#[derive(Clone, Debug, PartialEq)]
pub struct AggregateEnergy<R> {
    electric: R,
    magnetic: R,
    total: R,
}

impl<R> AggregateEnergy<R> {
    pub(crate) const fn new(electric: R, magnetic: R, total: R) -> Self {
        Self {
            electric,
            magnetic,
            total,
        }
    }

    /// Return the total finite-layer electric energy.
    pub fn electric(&self) -> &R {
        &self.electric
    }

    /// Return the total finite-layer magnetic energy.
    pub fn magnetic(&self) -> &R {
        &self.magnetic
    }

    /// Return the total finite-layer electromagnetic energy.
    pub fn total(&self) -> &R {
        &self.total
    }

    /// Consume the aggregate and return `(electric, magnetic, total)`.
    pub fn into_parts(self) -> (R, R, R) {
        (self.electric, self.magnetic, self.total)
    }

    /// Transform every aggregate component.
    pub fn map<U>(self, mut map: impl FnMut(R) -> U) -> AggregateEnergy<U> {
        AggregateEnergy {
            electric: map(self.electric),
            magnetic: map(self.magnetic),
            total: map(self.total),
        }
    }
}

/// Error produced while aggregating finite-layer quantities.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum LayerAggregateError {
    #[error("cannot aggregate an empty finite-layer sequence")]
    EmptyLayers,

    #[error("the confinement selection contains no finite layers")]
    EmptySelection,
}

impl<R> Layers<LayerEnergy<R>>
where
    R: ScalarAlgebra,
{
    /// Sum the energy stored in all finite layers.
    ///
    /// The returned aggregate excludes the exterior media and preserves the
    /// full algebra representation, including derivative components.
    pub fn aggregate(&self) -> Result<AggregateEnergy<R>, LayerAggregateError> {
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

        Ok(AggregateEnergy::new(electric, magnetic, total))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Ix0, arr0};

    use super::*;

    use crate::algebra::{ArrayJet0, ArrayJet1, Jet0, RealParameter};

    type R0 = ArrayJet0<f64, Ix0, RealParameter>;

    fn jet(value: f64) -> R0 {
        Jet0::new(arr0(value))
    }

    fn scalar(value: &R0) -> f64 {
        value.value()[()]
    }

    fn energy(electric: f64, magnetic: f64) -> LayerEnergy<R0> {
        LayerEnergy::new(jet(electric), jet(magnetic), jet(electric + magnetic))
    }

    #[test]
    fn aggregate_energy_preserves_component_order() {
        let aggregate = AggregateEnergy::new(1, 2, 3);

        assert_eq!(aggregate.electric(), &1);
        assert_eq!(aggregate.magnetic(), &2);
        assert_eq!(aggregate.total(), &3);

        assert_eq!(aggregate.into_parts(), (1, 2, 3),);
    }

    #[test]
    fn aggregate_energy_map_transforms_every_component() {
        let aggregate = AggregateEnergy::new(1, 2, 3);

        let mapped = aggregate.map(|value| value * 10);

        assert_eq!(mapped.electric(), &10);
        assert_eq!(mapped.magnetic(), &20);
        assert_eq!(mapped.total(), &30);
    }

    #[test]
    fn one_layer_aggregate_equals_that_layer() {
        let layers = Layers::new(vec![energy(2.0, 3.0)]);

        let aggregate = layers.aggregate().unwrap();

        assert_eq!(scalar(aggregate.electric()), 2.0,);

        assert_eq!(scalar(aggregate.magnetic()), 3.0,);

        assert_eq!(scalar(aggregate.total()), 5.0,);
    }

    #[test]
    fn aggregate_sums_all_layers_in_physical_sequence() {
        let layers = Layers::new(vec![energy(2.0, 3.0), energy(5.0, 7.0), energy(11.0, 13.0)]);

        let aggregate = layers.aggregate().unwrap();

        assert_eq!(scalar(aggregate.electric()), 18.0,);

        assert_eq!(scalar(aggregate.magnetic()), 23.0,);

        assert_eq!(scalar(aggregate.total()), 41.0,);
    }

    #[test]
    fn empty_sequence_cannot_be_aggregated() {
        let layers: Layers<LayerEnergy<R0>> = Layers::new(Vec::new());

        assert_eq!(layers.aggregate(), Err(LayerAggregateError::EmptyLayers),);
    }

    #[test]
    fn aggregation_propagates_first_derivatives() {
        type R = ArrayJet1<f64, Ix0, RealParameter>;

        fn first(value: f64, derivative: f64) -> R {
            R::from_parts(arr0(value), arr0(derivative))
        }

        let layers = Layers::new(vec![
            LayerEnergy::new(first(2.0, 3.0), first(5.0, 7.0), first(7.0, 10.0)),
            LayerEnergy::new(first(11.0, 13.0), first(17.0, 19.0), first(28.0, 32.0)),
        ]);

        let aggregate = layers.aggregate().unwrap();

        assert_eq!(aggregate.electric().value()[()], 13.0,);

        assert_eq!(aggregate.electric().first()[()], 16.0,);

        assert_eq!(aggregate.magnetic().value()[()], 22.0,);

        assert_eq!(aggregate.magnetic().first()[()], 26.0,);

        assert_eq!(aggregate.total().value()[()], 35.0,);

        assert_eq!(aggregate.total().first()[()], 42.0,);
    }
}
