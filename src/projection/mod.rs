mod context;
mod problem;
mod quantities;
mod workspace;

use ndarray::{Array, Dimension, NdIndex, arr0};
use thiserror::Error;

use crate::{
    algebra::{
        ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, Jet, Jet0, Jet1,
        Jet2, JetBivariate1, JetBivariate2,
    },
    differential::{BivariateGradient, BivariateHessian},
};

#[derive(Clone, Debug, PartialEq, Eq, Error)]
pub enum PointProjectionError {
    #[error("requested sample index not found in source shape {shape:?}")]
    OutOfBounds { shape: Vec<usize> },
}

/// Project a sampled structure into its owned zero-dimensional representation.
///
/// Implementations recursively project every sampled jet contained by the
/// structure while preserving non-sampled metadata.
pub(crate) trait ProjectPoint {
    type Dimension: Dimension;
    type Point;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: NdIndex<Self::Dimension> + Clone;

    fn at<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: ndarray::NdIndex<Self::Dimension> + Clone,
    {
        self.project_point(index)
    }
}

impl<J> ProjectPoint for J
where
    J: JetPointProjection,
    J::Dimension: Dimension,
{
    type Dimension = J::Dimension;
    type Point = J::PointJet;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: NdIndex<Self::Dimension> + Clone,
    {
        JetPointProjection::project_point(self, index)
    }
}

/// Extract one sampled point from a jet as an owned zero-dimensional jet.
///
/// The projection preserves the scalar type, derivative order, parameter
/// markers, and every derivative component. Only the sampled ndarray
/// dimension changes to [`Ix0`].
pub trait JetPointProjection: Jet {
    /// Project the value and all derivative components at `index`.
    fn project_point<Idx>(&self, index: &Idx) -> Result<Self::PointJet, PointProjectionError>
    where
        Idx: NdIndex<Self::Dimension> + Clone;
}

impl<C, D, P> JetPointProjection for ArrayJet0<C, D, P>
where
    C: Copy,
    D: Dimension,
{
    fn project_point<I>(&self, index: &I) -> Result<Self::PointJet, PointProjectionError>
    where
        I: NdIndex<D> + Clone,
    {
        let value = project_array_point(self.value(), index)?;

        Ok(Jet0::new(arr0(value)))
    }
}

impl<C, D, P> JetPointProjection for ArrayJet1<C, D, P>
where
    C: Copy,
    D: Dimension,
{
    fn project_point<I>(&self, index: &I) -> Result<Self::PointJet, PointProjectionError>
    where
        I: NdIndex<D> + Clone,
    {
        let value = project_array_point(self.value(), index)?;
        let first = project_array_point(self.first(), index)?;

        Ok(Jet1::from_parts(arr0(value), arr0(first)))
    }
}

impl<C, D, P> JetPointProjection for ArrayJet2<C, D, P>
where
    C: Copy,
    D: Dimension,
{
    fn project_point<I>(&self, index: &I) -> Result<Self::PointJet, PointProjectionError>
    where
        I: NdIndex<D> + Clone,
    {
        let value = project_array_point(self.value(), index)?;
        let first = project_array_point(self.first(), index)?;
        let second = project_array_point(self.second(), index)?;

        Ok(Jet2::from_parts(arr0(value), arr0(first), arr0(second)))
    }
}

impl<C, D, P> JetPointProjection for ArrayJetBivariate1<C, D, P>
where
    C: Copy,
    D: Dimension,
{
    fn project_point<I>(&self, index: &I) -> Result<Self::PointJet, PointProjectionError>
    where
        I: NdIndex<D> + Clone,
    {
        let value = project_array_point(self.value(), index)?;
        let axis0 = project_array_point(self.axis0(), index)?;
        let axis1 = project_array_point(self.axis1(), index)?;

        Ok(JetBivariate1::from_parts(
            arr0(value),
            BivariateGradient::new(arr0(axis0), arr0(axis1)),
        ))
    }
}

impl<C, D, P> JetPointProjection for ArrayJetBivariate2<C, D, P>
where
    C: Copy,
    D: Dimension,
{
    fn project_point<I>(&self, index: &I) -> Result<Self::PointJet, PointProjectionError>
    where
        I: NdIndex<D> + Clone,
    {
        let value = project_array_point(self.value(), index)?;
        let axis0 = project_array_point(self.axis0(), index)?;
        let axis1 = project_array_point(self.axis1(), index)?;
        let axis0_axis0 = project_array_point(self.axis0_axis0(), index)?;
        let axis0_axis1 = project_array_point(self.axis0_axis1(), index)?;
        let axis1_axis1 = project_array_point(self.axis1_axis1(), index)?;

        Ok(JetBivariate2::from_parts(
            arr0(value),
            BivariateGradient::new(arr0(axis0), arr0(axis1)),
            BivariateHessian::new(arr0(axis0_axis0), arr0(axis0_axis1), arr0(axis1_axis1)),
        ))
    }
}

fn project_array_point<T, D, I>(array: &Array<T, D>, index: &I) -> Result<T, PointProjectionError>
where
    T: Copy,
    D: Dimension,
    I: NdIndex<D> + Clone,
{
    array
        .get(index.clone())
        .copied()
        .ok_or_else(|| PointProjectionError::OutOfBounds {
            shape: array.shape().to_vec(),
        })
}

#[cfg(test)]
mod tests {
    use ndarray::{Ix0, Ix1, Ix2, arr0, arr1, arr2};

    use super::*;

    use crate::{
        algebra::{
            ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, Jet, Jet0,
            Jet1, Jet2, RealParameter, ScalarAlgebra,
        },
        differential::{BivariateGradient, BivariateHessian},
    };

    type J0D1 = ArrayJet0<f64, Ix1, RealParameter>;

    type J1D1 = ArrayJet1<f64, Ix1, RealParameter>;

    type J2D1 = ArrayJet2<f64, Ix1, RealParameter>;

    type JB1D1 = ArrayJetBivariate1<f64, Ix1, RealParameter>;

    type JB2D1 = ArrayJetBivariate2<f64, Ix1, RealParameter>;

    fn scalar<J>(jet: &J) -> J::Scalar
    where
        J: Jet<Dimension = Ix0> + ScalarAlgebra,
        J::Scalar: Copy,
    {
        jet.value()[()]
    }

    #[test]
    fn jet0_projects_selected_value() {
        let jet: Jet0<_, RealParameter> = Jet0::new(arr1(&[2.0, 3.0, 5.0]));

        let point = ProjectPoint::project_point(&jet, &[1]).unwrap();

        assert_eq!(scalar(&point), 3.0);
        assert_eq!(point.value().ndim(), 0);
    }

    #[test]
    fn jet1_projects_value_and_first_derivative() {
        let jet = J1D1::from_parts(arr1(&[2.0, 3.0, 5.0]), arr1(&[7.0, 11.0, 13.0]));

        let point = ProjectPoint::project_point(&jet, &[1]).unwrap();

        assert_eq!(point.value()[()], 3.0);
        assert_eq!(point.first()[()], 11.0);
    }

    #[test]
    fn jet2_projects_every_derivative_component() {
        let jet = J2D1::from_parts(
            arr1(&[2.0, 3.0, 5.0]),
            arr1(&[7.0, 11.0, 13.0]),
            arr1(&[17.0, 19.0, 23.0]),
        );

        let point = ProjectPoint::project_point(&jet, &2).unwrap();

        assert_eq!(point.value()[()], 5.0);
        assert_eq!(point.first()[()], 13.0);
        assert_eq!(point.second()[()], 23.0);
    }

    #[test]
    fn bivariate_first_projects_both_gradient_axes() {
        let jet = JB1D1::from_parts(
            arr1(&[2.0, 3.0, 5.0]),
            BivariateGradient::new(arr1(&[7.0, 11.0, 13.0]), arr1(&[17.0, 19.0, 23.0])),
        );

        let point = ProjectPoint::project_point(&jet, &1).unwrap();

        assert_eq!(point.value()[()], 3.0);
        assert_eq!(point.axis0()[()], 11.0);
        assert_eq!(point.axis1()[()], 19.0);
    }

    #[test]
    fn bivariate_second_projects_gradient_and_hessian() {
        let jet = JB2D1::from_parts(
            arr1(&[2.0, 3.0, 5.0]),
            BivariateGradient::new(arr1(&[7.0, 11.0, 13.0]), arr1(&[17.0, 19.0, 23.0])),
            BivariateHessian::new(
                arr1(&[29.0, 31.0, 37.0]),
                arr1(&[41.0, 43.0, 47.0]),
                arr1(&[53.0, 59.0, 61.0]),
            ),
        );

        let point = ProjectPoint::project_point(&jet, &1).unwrap();

        assert_eq!(point.value()[()], 3.0);

        assert_eq!(point.axis0()[()], 11.0);
        assert_eq!(point.axis1()[()], 19.0);

        assert_eq!(point.axis0_axis0()[()], 31.0);
        assert_eq!(point.axis0_axis1()[()], 43.0);
        assert_eq!(point.axis1_axis1()[()], 59.0);
    }

    #[test]
    fn projection_supports_multidimensional_indices() {
        let jet: ArrayJet2<f64, Ix2, RealParameter> = Jet2::from_parts(
            arr2(&[[2.0, 3.0], [5.0, 7.0]]),
            arr2(&[[11.0, 13.0], [17.0, 19.0]]),
            arr2(&[[23.0, 29.0], [31.0, 37.0]]),
        );

        let point = ProjectPoint::project_point(&jet, &(1, 0)).unwrap();

        assert_eq!(point.value()[()], 5.0);
        assert_eq!(point.first()[()], 17.0);
        assert_eq!(point.second()[()], 31.0);
    }

    #[test]
    fn projection_of_ix0_jet_is_identity_in_value() {
        let jet: ArrayJet2<f64, Ix0, RealParameter> =
            Jet2::from_parts(arr0(2.0), arr0(3.0), arr0(5.0));

        let point = ProjectPoint::project_point(&jet, &()).unwrap();

        assert_eq!(point, jet);
    }

    #[test]
    fn invalid_index_reports_source_shape() {
        let jet: J0D1 = Jet0::new(arr1(&[2.0, 3.0]));

        let error = ProjectPoint::project_point(&jet, &2).unwrap_err();

        assert_eq!(error, PointProjectionError::OutOfBounds { shape: vec![2] },);
    }

    #[test]
    fn invalid_multidimensional_index_reports_source_shape() {
        let jet: ArrayJet0<f64, Ix2, RealParameter> = Jet0::new(arr2(&[[2.0, 3.0], [5.0, 7.0]]));

        let error = ProjectPoint::project_point(&jet, &(2, 0)).unwrap_err();

        assert_eq!(
            error,
            PointProjectionError::OutOfBounds { shape: vec![2, 2] },
        );
    }

    #[test]
    fn structural_blanket_impl_delegates_to_jet_projection() {
        let jet: J1D1 = Jet1::from_parts(arr1(&[2.0, 3.0]), arr1(&[5.0, 7.0]));

        let point = ProjectPoint::project_point(&jet, &1).unwrap();

        assert_eq!(point.value()[()], 3.0);
        assert_eq!(point.first()[()], 7.0);
    }
}
