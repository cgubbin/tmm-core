//! Scalar and ndarray sampling abstractions.
//!
//! Material traits are generic over [`Sampled`], allowing one implementation
//! to support both pointwise and ndarray evaluation without coupling material
//! models to a fixed sampled dimension.

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct Scalar<C>(pub C);

impl<C> Scalar<C> {
    pub const fn new(value: C) -> Self {
        Self(value)
    }

    pub fn into_inner(self) -> C {
        self.0
    }
}

impl<C> From<C> for Scalar<C> {
    fn from(value: C) -> Self {
        Self::new(value)
    }
}

/// A scalar or sampled collection that can be mapped elementwise.
///
/// Material model traits use this abstraction so the same implementation can
/// evaluate either one spectral coordinate or an ndarray of coordinates.
pub trait Sampled {
    /// Scalar element type.
    type Elem;

    /// Output representation produced by mapping each element to `T`.
    type Mapped<T>;

    /// Map every sampled value through `f`.
    fn map<T, F>(self, f: F) -> Self::Mapped<T>
    where
        F: FnMut(Self::Elem) -> T;
}

impl<C> Sampled for Scalar<C>
where
    C: Copy,
{
    type Elem = C;
    type Mapped<T> = T;

    fn map<T, F>(self, mut f: F) -> T
    where
        F: FnMut(C) -> T,
    {
        f(self.0)
    }
}

impl<C, S, D> Sampled for ndarray::ArrayBase<S, D>
where
    C: Copy,
    S: ndarray::Data<Elem = C>,
    D: ndarray::Dimension,
{
    type Elem = C;
    type Mapped<T> = ndarray::Array<T, D>;

    fn map<T, F>(self, f: F) -> Self::Mapped<T>
    where
        F: FnMut(C) -> T,
    {
        self.mapv(f)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use super::*;

    #[test]
    fn scalar_maps_to_plain_value() {
        let sampled = Scalar::new(2.0);

        let mapped = sampled.map(|value| value * 3.0);

        assert_eq!(mapped, 6.0);
    }

    #[test]
    fn ndarray_mapping_preserves_shape() {
        let sampled = array![[1.0, 2.0], [3.0, 4.0]];

        let mapped = sampled.map(|value| value * 2.0);

        assert_eq!(mapped, array![[2.0, 4.0], [6.0, 8.0]],);
    }

    #[test]
    fn scalar_round_trips_through_inner_value() {
        let scalar = Scalar::from(3.5);

        assert_eq!(scalar.into_inner(), 3.5);
    }
}
