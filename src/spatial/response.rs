use crate::spatial::ResolvedFieldSampling;

/// A quantity sampled at resolved positions throughout a planar stack.
///
/// The final ndarray axis of `quantity` corresponds, in order, to
/// [`Self::sampling`].
#[derive(Clone, Debug, PartialEq)]
pub struct SpatialResponse<Q, R> {
    quantity: Q,
    sampling: ResolvedFieldSampling<R>,
}

impl<Q, R> SpatialResponse<Q, R> {
    pub(crate) fn new(quantity: Q, sampling: ResolvedFieldSampling<R>) -> Self {
        Self { quantity, sampling }
    }

    /// Return the sampled quantity.
    pub fn quantity(&self) -> &Q {
        &self.quantity
    }

    /// Return the resolved spatial sampling metadata.
    pub fn sampling(&self) -> &ResolvedFieldSampling<R> {
        &self.sampling
    }

    /// Consume the response and return its components.
    pub fn into_parts(self) -> (Q, ResolvedFieldSampling<R>) {
        (self.quantity, self.sampling)
    }

    /// Transform the sampled quantity while preserving its spatial metadata.
    pub fn map_quantity<U>(self, f: impl FnOnce(Q) -> U) -> SpatialResponse<U, R> {
        SpatialResponse {
            quantity: f(self.quantity),
            sampling: self.sampling,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stores_quantity_and_sampling() {
        let sampling: ResolvedFieldSampling<f64> = ResolvedFieldSampling::new(vec![]);

        let response = SpatialResponse::new(42, sampling.clone());

        assert_eq!(response.quantity(), &42);
        assert_eq!(response.sampling(), &sampling);
    }

    #[test]
    fn into_parts_preserves_both_components() {
        let sampling: ResolvedFieldSampling<f64> = ResolvedFieldSampling::new(vec![]);

        let response = SpatialResponse::new(42, sampling.clone());

        assert_eq!(response.into_parts(), (42, sampling),);
    }

    #[test]
    fn map_quantity_preserves_sampling() {
        let sampling: ResolvedFieldSampling<f64> = ResolvedFieldSampling::new(vec![]);

        let response = SpatialResponse::new(2, sampling.clone()).map_quantity(|value| value * 10);

        assert_eq!(response.quantity(), &20);
        assert_eq!(response.sampling(), &sampling);
    }
}
