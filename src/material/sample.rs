#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Scalar<C>(pub C);

impl<C> From<C> for Scalar<C> {
    fn from(value: C) -> Self {
        Self(value)
    }
}

pub trait Sampled {
    type Elem;
    type Mapped<T>;

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

#[cfg(feature = "ndarray")]
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
