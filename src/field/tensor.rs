//! Cartesian rank-two tensor-valued array fields.

use crate::{
    SpatialProfileError,
    field::FieldShapeError,
    spatial::array_profile,
};
use ndarray::{Array, ArrayView1, Dimension, Ix1};

/// One Cartesian rank-two tensor value.
///
/// Components are stored in row-major order:
///
/// ```text
/// [ xx  xy  xz ]
/// [ yx  yy  yz ]
/// [ zx  zy  zz ]
/// ```
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorValue<C> {
    pub xx: C,
    pub xy: C,
    pub xz: C,
    pub yx: C,
    pub yy: C,
    pub yz: C,
    pub zx: C,
    pub zy: C,
    pub zz: C,
}

impl<C> TensorValue<C> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(xx: C, xy: C, xz: C, yx: C, yy: C, yz: C, zx: C, zy: C, zz: C) -> Self {
        Self {
            xx,
            xy,
            xz,
            yx,
            yy,
            yz,
            zx,
            zy,
            zz,
        }
    }

    /// Convert into row-major nested arrays.
    pub fn into_rows(self) -> [[C; 3]; 3] {
        [
            [self.xx, self.xy, self.xz],
            [self.yx, self.yy, self.yz],
            [self.zx, self.zy, self.zz],
        ]
    }
}

impl<C> From<[[C; 3]; 3]> for TensorValue<C> {
    fn from([[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]]: [[C; 3]; 3]) -> Self {
        Self::new(xx, xy, xz, yx, yy, yz, zx, zy, zz)
    }
}

/// A Cartesian rank-two tensor field.
///
/// All nine component arrays must have the same shape.
#[derive(Clone, Debug, PartialEq)]
pub struct TensorField<C, D>
where
    D: Dimension,
{
    xx: Array<C, D>,
    xy: Array<C, D>,
    xz: Array<C, D>,
    yx: Array<C, D>,
    yy: Array<C, D>,
    yz: Array<C, D>,
    zx: Array<C, D>,
    zy: Array<C, D>,
    zz: Array<C, D>,
}

/// A borrowed one-dimensional Cartesian tensor-field view.
#[derive(Clone, Copy, Debug)]
pub struct TensorFieldView1<'a, C> {
    xx: ArrayView1<'a, C>,
    xy: ArrayView1<'a, C>,
    xz: ArrayView1<'a, C>,
    yx: ArrayView1<'a, C>,
    yy: ArrayView1<'a, C>,
    yz: ArrayView1<'a, C>,
    zx: ArrayView1<'a, C>,
    zy: ArrayView1<'a, C>,
    zz: ArrayView1<'a, C>,
}

impl<C, D> TensorField<C, D>
where
    D: Dimension,
{
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        xx: Array<C, D>,
        xy: Array<C, D>,
        xz: Array<C, D>,
        yx: Array<C, D>,
        yy: Array<C, D>,
        yz: Array<C, D>,
        zx: Array<C, D>,
        zy: Array<C, D>,
        zz: Array<C, D>,
    ) -> Result<Self, FieldShapeError> {
        let expected = xx.shape();

        for (name, shape) in [
            ("xy", xy.shape()),
            ("xz", xz.shape()),
            ("yx", yx.shape()),
            ("yy", yy.shape()),
            ("yz", yz.shape()),
            ("zx", zx.shape()),
            ("zy", zy.shape()),
            ("zz", zz.shape()),
        ] {
            if shape != expected {
                return Err(FieldShapeError::new(name, expected, shape));
            }
        }

        Ok(Self {
            xx,
            xy,
            xz,
            yx,
            yy,
            yz,
            zx,
            zy,
            zz,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_unchecked(
        xx: Array<C, D>,
        xy: Array<C, D>,
        xz: Array<C, D>,
        yx: Array<C, D>,
        yy: Array<C, D>,
        yz: Array<C, D>,
        zx: Array<C, D>,
        zy: Array<C, D>,
        zz: Array<C, D>,
    ) -> Self {
        debug_assert_eq!(xx.shape(), xy.shape());
        debug_assert_eq!(xx.shape(), xz.shape());
        debug_assert_eq!(xx.shape(), yx.shape());
        debug_assert_eq!(xx.shape(), yy.shape());
        debug_assert_eq!(xx.shape(), yz.shape());
        debug_assert_eq!(xx.shape(), zx.shape());
        debug_assert_eq!(xx.shape(), zy.shape());
        debug_assert_eq!(xx.shape(), zz.shape());

        Self {
            xx,
            xy,
            xz,
            yx,
            yy,
            yz,
            zx,
            zy,
            zz,
        }
    }

    pub fn xx(&self) -> &Array<C, D> {
        &self.xx
    }

    pub fn xy(&self) -> &Array<C, D> {
        &self.xy
    }

    pub fn xz(&self) -> &Array<C, D> {
        &self.xz
    }

    pub fn yx(&self) -> &Array<C, D> {
        &self.yx
    }

    pub fn yy(&self) -> &Array<C, D> {
        &self.yy
    }

    pub fn yz(&self) -> &Array<C, D> {
        &self.yz
    }

    pub fn zx(&self) -> &Array<C, D> {
        &self.zx
    }

    pub fn zy(&self) -> &Array<C, D> {
        &self.zy
    }

    pub fn zz(&self) -> &Array<C, D> {
        &self.zz
    }

    pub fn shape(&self) -> &[usize] {
        self.xx.shape()
    }

    pub fn raw_dim(&self) -> D {
        self.xx.raw_dim()
    }

    pub fn ndim(&self) -> usize {
        self.xx.ndim()
    }

    pub fn len(&self) -> usize {
        self.xx.len()
    }

    pub fn is_empty(&self) -> bool {
        self.xx.is_empty()
    }

    pub fn get<I>(&self, index: I) -> Option<TensorValue<&C>>
    where
        I: ndarray::NdIndex<D> + Clone,
    {
        Some(TensorValue {
            xx: self.xx.get(index.clone())?,
            xy: self.xy.get(index.clone())?,
            xz: self.xz.get(index.clone())?,
            yx: self.yx.get(index.clone())?,
            yy: self.yy.get(index.clone())?,
            yz: self.yz.get(index.clone())?,
            zx: self.zx.get(index.clone())?,
            zy: self.zy.get(index.clone())?,
            zz: self.zz.get(index)?,
        })
    }

    pub fn map<B, F>(&self, mut map: F) -> TensorField<B, D>
    where
        F: FnMut(&C) -> B,
    {
        TensorField::new_unchecked(
            self.xx.map(&mut map),
            self.xy.map(&mut map),
            self.xz.map(&mut map),
            self.yx.map(&mut map),
            self.yy.map(&mut map),
            self.yz.map(&mut map),
            self.zx.map(&mut map),
            self.zy.map(&mut map),
            self.zz.map(map),
        )
    }

    pub(crate) fn profile_last_axis(
        &self,
        excitation_index: &D::Smaller,
    ) -> Result<TensorFieldView1<'_, C>, SpatialProfileError>
    where
        D::Smaller: Dimension<Larger = D>,
    {
        Ok(TensorFieldView1::new_unchecked(
            array_profile(self.xx.view(), excitation_index)?,
            array_profile(self.xy.view(), excitation_index)?,
            array_profile(self.xz.view(), excitation_index)?,
            array_profile(self.yx.view(), excitation_index)?,
            array_profile(self.yy.view(), excitation_index)?,
            array_profile(self.yz.view(), excitation_index)?,
            array_profile(self.zx.view(), excitation_index)?,
            array_profile(self.zy.view(), excitation_index)?,
            array_profile(self.zz.view(), excitation_index)?,
        ))
    }
}

impl<C> TensorField<C, Ix1> {
    pub fn view1(&self) -> TensorFieldView1<'_, C> {
        TensorFieldView1::new_unchecked(
            self.xx.view(),
            self.xy.view(),
            self.xz.view(),
            self.yx.view(),
            self.yy.view(),
            self.yz.view(),
            self.zx.view(),
            self.zy.view(),
            self.zz.view(),
        )
    }
}

impl<'a, C> TensorFieldView1<'a, C> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        xx: ArrayView1<'a, C>,
        xy: ArrayView1<'a, C>,
        xz: ArrayView1<'a, C>,
        yx: ArrayView1<'a, C>,
        yy: ArrayView1<'a, C>,
        yz: ArrayView1<'a, C>,
        zx: ArrayView1<'a, C>,
        zy: ArrayView1<'a, C>,
        zz: ArrayView1<'a, C>,
    ) -> Result<Self, FieldShapeError> {
        let expected = xx.shape();

        for (name, shape) in [
            ("xy", xy.shape()),
            ("xz", xz.shape()),
            ("yx", yx.shape()),
            ("yy", yy.shape()),
            ("yz", yz.shape()),
            ("zx", zx.shape()),
            ("zy", zy.shape()),
            ("zz", zz.shape()),
        ] {
            if shape != expected {
                return Err(FieldShapeError::new(name, expected, shape));
            }
        }

        Ok(Self {
            xx,
            xy,
            xz,
            yx,
            yy,
            yz,
            zx,
            zy,
            zz,
        })
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_unchecked(
        xx: ArrayView1<'a, C>,
        xy: ArrayView1<'a, C>,
        xz: ArrayView1<'a, C>,
        yx: ArrayView1<'a, C>,
        yy: ArrayView1<'a, C>,
        yz: ArrayView1<'a, C>,
        zx: ArrayView1<'a, C>,
        zy: ArrayView1<'a, C>,
        zz: ArrayView1<'a, C>,
    ) -> Self {
        Self {
            xx,
            xy,
            xz,
            yx,
            yy,
            yz,
            zx,
            zy,
            zz,
        }
    }

    pub fn xx(&self) -> ArrayView1<'a, C> {
        self.xx
    }

    pub fn xy(&self) -> ArrayView1<'a, C> {
        self.xy
    }

    pub fn xz(&self) -> ArrayView1<'a, C> {
        self.xz
    }

    pub fn yx(&self) -> ArrayView1<'a, C> {
        self.yx
    }

    pub fn yy(&self) -> ArrayView1<'a, C> {
        self.yy
    }

    pub fn yz(&self) -> ArrayView1<'a, C> {
        self.yz
    }

    pub fn zx(&self) -> ArrayView1<'a, C> {
        self.zx
    }

    pub fn zy(&self) -> ArrayView1<'a, C> {
        self.zy
    }

    pub fn zz(&self) -> ArrayView1<'a, C> {
        self.zz
    }

    pub fn len(&self) -> usize {
        self.xx.len()
    }

    pub fn is_empty(&self) -> bool {
        self.xx.is_empty()
    }

    pub fn get(&self, index: usize) -> Option<TensorValue<&C>> {
        Some(TensorValue {
            xx: self.xx.get(index)?,
            xy: self.xy.get(index)?,
            xz: self.xz.get(index)?,
            yx: self.yx.get(index)?,
            yy: self.yy.get(index)?,
            yz: self.yz.get(index)?,
            zx: self.zx.get(index)?,
            zy: self.zy.get(index)?,
            zz: self.zz.get(index)?,
        })
    }

    pub fn to_owned(&self) -> TensorField<C, Ix1>
    where
        C: Clone,
    {
        TensorField::new_unchecked(
            self.xx.to_owned(),
            self.xy.to_owned(),
            self.xz.to_owned(),
            self.yx.to_owned(),
            self.yy.to_owned(),
            self.yz.to_owned(),
            self.zx.to_owned(),
            self.zy.to_owned(),
            self.zz.to_owned(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn tensor_field() -> TensorField<i32, Ix1> {
        TensorField::new(
            array![1, 2],
            array![3, 4],
            array![5, 6],
            array![7, 8],
            array![9, 10],
            array![11, 12],
            array![13, 14],
            array![15, 16],
            array![17, 18],
        )
        .unwrap()
    }

    #[test]
    fn constructor_accepts_matching_shapes() {
        let field = tensor_field();

        assert_eq!(field.shape(), &[2]);
        assert_eq!(field.len(), 2);
    }

    #[test]
    fn constructor_rejects_mismatched_shape() {
        let error = TensorField::new(
            array![1, 2],
            array![3],
            array![5, 6],
            array![7, 8],
            array![9, 10],
            array![11, 12],
            array![13, 14],
            array![15, 16],
            array![17, 18],
        )
        .unwrap_err();

        assert_eq!(error.component(), "xy");
        assert_eq!(error.expected(), &[2]);
        assert_eq!(error.actual(), &[1]);
    }

    #[test]
    fn get_returns_complete_tensor() {
        let field = tensor_field();

        assert_eq!(
            field.get(1),
            Some(TensorValue::new(&2, &4, &6, &8, &10, &12, &14, &16, &18,)),
        );
    }

    #[test]
    fn component_mapping_preserves_structure() {
        let field = tensor_field();
        let mapped = field.map(|value| value * 2);

        assert_eq!(mapped.xx(), &array![2, 4]);
        assert_eq!(mapped.xy(), &array![6, 8]);
        assert_eq!(mapped.zz(), &array![34, 36]);
    }

    #[test]
    fn view1_borrows_all_components() {
        let field = tensor_field();
        let view = field.view1();

        assert_eq!(view.len(), 2);
        assert_eq!(
            view.get(0),
            Some(TensorValue::new(&1, &3, &5, &7, &9, &11, &13, &15, &17,)),
        );
    }

    #[test]
    fn view1_can_be_copied_to_owned_field() {
        let field = tensor_field();

        assert_eq!(field.view1().to_owned(), field);
    }
}
