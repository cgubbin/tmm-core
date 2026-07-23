#[derive(Clone, Debug, PartialEq)]
pub struct SpectralSecond<T> {
    gradient: SpectralGradient<T>,
    hessian: SpectralHessian<T>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpectralGradient<T> {
    vacuum_wavenumber: T,
    parallel_wavenumber: T,
}

impl<T> SpectralGradient<T> {
    pub(crate) fn new(vacuum_wavenumber: T, parallel_wavenumber: T) -> Self {
        Self {
            vacuum_wavenumber,
            parallel_wavenumber,
        }
    }

    pub fn vacuum_wavenumber(&self) -> &T {
        &self.vacuum_wavenumber
    }

    pub fn parallel_wavenumber(&self) -> &T {
        &self.parallel_wavenumber
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> SpectralGradient<U> {
        SpectralGradient {
            vacuum_wavenumber: f(self.vacuum_wavenumber),
            parallel_wavenumber: f(self.parallel_wavenumber),
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct SpectralHessian<T> {
    vacuum_wavenumber_vacuum_wavenumber: T,
    vacuum_wavenumber_parallel_wavenumber: T,
    parallel_wavenumber_parallel_wavenumber: T,
}

impl<T> SpectralHessian<T> {
    pub(crate) fn new(
        vacuum_wavenumber_vacuum_wavenumber: T,
        vacuum_wavenumber_parallel_wavenumber: T,
        parallel_wavenumber_parallel_wavenumber: T,
    ) -> Self {
        Self {
            vacuum_wavenumber_vacuum_wavenumber,
            vacuum_wavenumber_parallel_wavenumber,
            parallel_wavenumber_parallel_wavenumber,
        }
    }

    pub fn vacuum_wavenumber_vacuum_wavenumber(&self) -> &T {
        &self.vacuum_wavenumber_vacuum_wavenumber
    }

    pub fn vacuum_wavenumber_parallel_wavenumber(&self) -> &T {
        &self.vacuum_wavenumber_parallel_wavenumber
    }

    pub fn parallel_wavenumber_parallel_wavenumber(&self) -> &T {
        &self.parallel_wavenumber_parallel_wavenumber
    }

    pub fn map<U>(self, mut f: impl FnMut(T) -> U) -> SpectralHessian<U> {
        SpectralHessian {
            vacuum_wavenumber_vacuum_wavenumber: f(self.vacuum_wavenumber_vacuum_wavenumber),
            vacuum_wavenumber_parallel_wavenumber: f(self.vacuum_wavenumber_parallel_wavenumber),
            parallel_wavenumber_parallel_wavenumber: f(self.parallel_wavenumber_parallel_wavenumber),
        }
    }
}
