use crate::{ComplexScalar, tensor::Tensor3};

use ndarray::Dimension;

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

pub trait TensorSampled: Sized {
    type Elem;
    type TensorOutput<T>;

    fn map_tensor3<T, F>(self, f: F) -> Self::TensorOutput<T>
    where
        T: ComplexScalar,
        F: FnMut(Self::Elem) -> Tensor3<T>;
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

impl<C> TensorSampled for Scalar<C>
where
    C: Copy,
{
    type Elem = C;
    type TensorOutput<T> = Tensor3<T>;

    fn map_tensor3<T, F>(self, mut f: F) -> Self::TensorOutput<T>
    where
        T: ComplexScalar,
        F: FnMut(C) -> Tensor3<T>,
    {
        f(self.0)
    }
}

impl<C, S, D> TensorSampled for ndarray::ArrayBase<S, D>
where
    C: Copy,
    S: ndarray::Data<Elem = C>,
    D: Dimension,
{
    type Elem = C;
    type TensorOutput<T> = ndarray::Array<T, <<D as Dimension>::Larger as Dimension>::Larger>;

    fn map_tensor3<T, F>(self, mut f: F) -> Self::TensorOutput<T>
    where
        T: ComplexScalar,
        F: FnMut(C) -> Tensor3<T>,
    {
        use ndarray::{Array, Axis};

        let input_dim = self.raw_dim();
        let mut output_dim = input_dim
            .insert_axis(Axis(input_dim.ndim()))
            .insert_axis(Axis(input_dim.ndim() + 1));

        output_dim[input_dim.ndim()] = 3;
        output_dim[input_dim.ndim() + 1] = 3;

        let mut out = Array::from_elem(output_dim, T::zero());
        {
            let input_dyn = self.view().into_dyn();
            let mut out_dyn = out.view_mut().into_dyn();

            for (idx, value) in input_dyn.indexed_iter() {
                let tensor = f(*value);

                for a in 0..3 {
                    for b in 0..3 {
                        let mut out_idx = idx.slice().to_vec();
                        out_idx.push(a);
                        out_idx.push(b);

                        out_dyn[ndarray::IxDyn(&out_idx)] = tensor[[a, b]];
                    }
                }
            }
        }
        out
    }
}
