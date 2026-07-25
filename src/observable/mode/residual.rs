pub struct ModeResidual<C> {
    residual: C,
}

impl<C> ModeResidual<C> {
    pub(crate) fn new(residual: C) -> Self {
        Self { residual }
    }

    pub(crate) fn residual(&self) -> &C {
        &self.residual
    }

    pub(crate) fn map<U>(self, f: impl Fn(C) -> U) -> ModeResidual<U> {
        ModeResidual {
            residual: f(self.residual),
        }
    }

    pub(crate) fn into_inner(self) -> C {
        self.residual
    }
}
