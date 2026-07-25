use crate::algebra::{Jet1, Jet2, JetBivariate};

/// Value and one first directional derivative.
#[derive(Clone, Debug)]
pub(crate) struct DirectionalFirstParts<T> {
    pub(crate) value: T,
    pub(crate) first: T,
}

impl<T> DirectionalFirstParts<T> {
    pub(crate) fn new(value: T, first: T) -> Self {
        Self { value, first }
    }

    pub(crate) fn into_parts(self) -> (T, T) {
        (self.value, self.first)
    }
}

impl<I, P> From<Jet1<I, P>> for DirectionalFirstParts<I> {
    fn from(jet: Jet1<I, P>) -> Self {
        let (value, first) = jet.into_parts();
        DirectionalFirstParts::new(value, first)
    }
}

impl<I, P> From<Jet2<I, P>> for DirectionalFirstParts<I> {
    fn from(jet: Jet2<I, P>) -> Self {
        let (value, first, ..) = jet.into_parts();
        DirectionalFirstParts::new(value, first)
    }
}

/// Value, first derivative, and repeated second derivative.
#[derive(Clone, Debug)]
pub(crate) struct DirectionalSecondParts<T> {
    pub(crate) value: T,
    pub(crate) first: T,
    pub(crate) second: T,
}

impl<T> DirectionalSecondParts<T> {
    pub(crate) fn new(value: T, first: T, second: T) -> Self {
        Self {
            value,
            first,
            second,
        }
    }

    pub(crate) fn into_parts(self) -> (T, T, T) {
        (self.value, self.first, self.second)
    }
}

impl<I, P> From<Jet2<I, P>> for DirectionalSecondParts<I> {
    fn from(jet: Jet2<I, P>) -> Self {
        let (value, first, second) = jet.into_parts();
        DirectionalSecondParts::new(value, first, second)
    }
}

/// Value and canonical spectral derivatives through second order.
#[derive(Clone, Debug)]
pub(crate) struct SpectralFirstParts<T> {
    pub(crate) value: T,

    pub(crate) vacuum_wavenumber: T,
    pub(crate) parallel_wavenumber: T,
}

impl<T> SpectralFirstParts<T> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(value: T, vacuum_wavenumber: T, parallel_wavenumber: T) -> Self {
        Self {
            value,
            vacuum_wavenumber,
            parallel_wavenumber,
        }
    }
}

impl<I, P> From<JetBivariate<I, P>> for SpectralFirstParts<I> {
    fn from(jet: JetBivariate<I, P>) -> Self {
        let (value, gradient, ..) = jet.into_parts();
        let (vacuum_wavenumber, parallel_wavenumber) = gradient.into_parts();
        SpectralFirstParts::new(value, vacuum_wavenumber, parallel_wavenumber)
    }
}

/// Value and canonical spectral derivatives through second order.
#[derive(Clone, Debug)]
pub(crate) struct SpectralSecondParts<T> {
    pub(crate) value: T,

    pub(crate) vacuum_wavenumber: T,
    pub(crate) parallel_wavenumber: T,

    pub(crate) vacuum_wavenumber_vacuum_wavenumber: T,
    pub(crate) vacuum_wavenumber_parallel_wavenumber: T,
    pub(crate) parallel_wavenumber_parallel_wavenumber: T,
}

impl<T> SpectralSecondParts<T> {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        value: T,
        vacuum_wavenumber: T,
        parallel_wavenumber: T,
        vacuum_wavenumber_vacuum_wavenumber: T,
        vacuum_wavenumber_parallel_wavenumber: T,
        parallel_wavenumber_parallel_wavenumber: T,
    ) -> Self {
        Self {
            value,
            vacuum_wavenumber,
            parallel_wavenumber,
            vacuum_wavenumber_vacuum_wavenumber,
            vacuum_wavenumber_parallel_wavenumber,
            parallel_wavenumber_parallel_wavenumber,
        }
    }
}

impl<I, P> From<JetBivariate<I, P>> for SpectralSecondParts<I> {
    fn from(jet: JetBivariate<I, P>) -> Self {
        let (value, gradient, hessian) = jet.into_parts();
        let (vacuum_wavenumber, parallel_wavenumber) = gradient.into_parts();
        let (
            vacuum_wavenumber_vacuum_wavenumber,
            vacuum_wavenumber_parallel_wavenumber,
            parallel_wavenumber_parallel_wavenumber,
        ) = hessian.into_parts();
        SpectralSecondParts::new(
            value,
            vacuum_wavenumber,
            parallel_wavenumber,
            vacuum_wavenumber_vacuum_wavenumber,
            vacuum_wavenumber_parallel_wavenumber,
            parallel_wavenumber_parallel_wavenumber,
        )
    }
}
