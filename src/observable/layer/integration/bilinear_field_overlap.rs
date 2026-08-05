pub(crate) struct IntegratedBilinearFieldOverlap<A> {
    electric: A,
    magnetic: A,
}

impl<A> IntegratedBilinearFieldOverlap<A> {
    pub(crate) const fn new(electric: A, magnetic: A) -> Self {
        Self { electric, magnetic }
    }

    pub(crate) fn electric(&self) -> &A {
        &self.electric
    }

    pub(crate) fn magnetic(&self) -> &A {
        &self.magnetic
    }

    pub(crate) fn into_parts(self) -> (A, A) {
        (self.electric, self.magnetic)
    }
}
