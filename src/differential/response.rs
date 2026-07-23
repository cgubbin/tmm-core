#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct NoDerivatives;

#[derive(Clone, Debug, PartialEq)]
pub struct DifferentialResponse<V, D = NoDerivatives> {
    values: V,
    derivatives: D,
}

impl<V, D> DifferentialResponse<V, D> {
    pub(crate) fn new(values: V, derivatives: D) -> Self {
        Self {
            values,
            derivatives,
        }
    }

    pub fn values(&self) -> &V {
        &self.values
    }

    pub fn derivatives(&self) -> &D {
        &self.derivatives
    }

    pub fn into_parts(self) -> (V, D) {
        (self.values, self.derivatives)
    }

    pub fn map_values<U>(self, f: impl FnOnce(V) -> U) -> DifferentialResponse<U, D> {
        DifferentialResponse {
            values: f(self.values),
            derivatives: self.derivatives,
        }
    }

    pub fn map_derivatives<E>(self, f: impl FnOnce(D) -> E) -> DifferentialResponse<V, E> {
        DifferentialResponse {
            values: self.values,
            derivatives: f(self.derivatives),
        }
    }
}
