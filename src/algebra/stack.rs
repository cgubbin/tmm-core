use ndarray::{Axis, Dimension, ShapeError};

use crate::{
    algebra::{
        ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, Jet, Jet0, Jet1,
        Jet2, JetBivariate1, JetBivariate2,
    },
    differential::{BivariateGradient, BivariateHessian},
};

pub trait JetStack: Sized + Jet
where
    Self::Dimension: Dimension,
{
    type Stacked: Jet<Scalar = Self::Scalar, Dimension = <Self::Dimension as Dimension>::Larger>;

    fn stack(stack: Vec<Self>) -> Result<Option<Self::Stacked>, ShapeError>;
}

impl<C, D, P> JetStack for ArrayJet0<C, D, P>
where
    C: Clone,
    D: Dimension,
    P: Clone,
{
    type Stacked = ArrayJet0<C, D::Larger, P>;

    fn stack(values: Vec<Self>) -> Result<Option<Self::Stacked>, ShapeError> {
        if values.is_empty() {
            return Ok(None);
        }
        let views = values
            .iter()
            .map(Jet0::value)
            .map(|each| each.view())
            .collect::<Vec<_>>();

        let ndim = values[0].ndim();

        ndarray::stack(Axis(ndim), &views[..])
            .map(Jet0::new)
            .map(Option::Some)
    }
}

impl<C, D, P> JetStack for ArrayJet1<C, D, P>
where
    C: Clone,
    D: Dimension,
    P: Clone,
{
    type Stacked = ArrayJet1<C, D::Larger, P>;

    fn stack(values: Vec<Self>) -> Result<Option<Self::Stacked>, ShapeError> {
        if values.is_empty() {
            return Ok(None);
        }

        let axis = Axis(values[0].value().ndim());

        let value_views = values
            .iter()
            .map(Jet1::value)
            .map(|each| each.view())
            .collect::<Vec<_>>();

        let first_views = values
            .iter()
            .map(Jet1::first)
            .map(|each| each.view())
            .collect::<Vec<_>>();

        let value = ndarray::stack(axis, &value_views)?;
        let first = ndarray::stack(axis, &first_views)?;

        Ok(Some(Jet1::from_parts(value, first)))
    }
}

impl<C, D, P> JetStack for ArrayJet2<C, D, P>
where
    C: Clone,
    D: Dimension,
    P: Clone,
{
    type Stacked = ArrayJet2<C, D::Larger, P>;

    fn stack(values: Vec<Self>) -> Result<Option<Self::Stacked>, ShapeError> {
        if values.is_empty() {
            return Ok(None);
        }

        let axis = Axis(values[0].value().ndim());

        let value_views = values
            .iter()
            .map(Jet2::value)
            .map(|each| each.view())
            .collect::<Vec<_>>();

        let first_views = values
            .iter()
            .map(Jet2::first)
            .map(|each| each.view())
            .collect::<Vec<_>>();

        let second_views = values
            .iter()
            .map(Jet2::second)
            .map(|each| each.view())
            .collect::<Vec<_>>();

        let value = ndarray::stack(axis, &value_views)?;
        let first = ndarray::stack(axis, &first_views)?;
        let second = ndarray::stack(axis, &second_views)?;

        Ok(Some(Jet2::from_parts(value, first, second)))
    }
}

impl<C, D, P> JetStack for ArrayJetBivariate1<C, D, P>
where
    C: Clone,
    D: Dimension,
    P: Clone,
{
    type Stacked = ArrayJetBivariate1<C, D::Larger, P>;

    fn stack(values: Vec<Self>) -> Result<Option<Self::Stacked>, ShapeError> {
        if values.is_empty() {
            return Ok(None);
        }

        let axis = Axis(values[0].value().ndim());

        let value_views = values
            .iter()
            .map(JetBivariate1::value)
            .map(|each| each.view())
            .collect::<Vec<_>>();

        let axis0_views = values
            .iter()
            .map(JetBivariate1::axis0)
            .map(|each| each.view())
            .collect::<Vec<_>>();

        let axis1_views = values
            .iter()
            .map(JetBivariate1::axis1)
            .map(|each| each.view())
            .collect::<Vec<_>>();

        let value = ndarray::stack(axis, &value_views)?;
        let axis0 = ndarray::stack(axis, &axis0_views)?;
        let axis1 = ndarray::stack(axis, &axis1_views)?;

        Ok(Some(JetBivariate1::from_parts(
            value,
            BivariateGradient::new(axis0, axis1),
        )))
    }
}

impl<C, D, P> JetStack for ArrayJetBivariate2<C, D, P>
where
    C: Clone,
    D: Dimension,
    P: Clone,
{
    type Stacked = ArrayJetBivariate2<C, D::Larger, P>;

    fn stack(values: Vec<Self>) -> Result<Option<Self::Stacked>, ShapeError> {
        if values.is_empty() {
            return Ok(None);
        }

        let axis = Axis(values[0].value().ndim());

        let value_views = values
            .iter()
            .map(JetBivariate2::value)
            .map(|each| each.view())
            .collect::<Vec<_>>();

        let axis0_views = values
            .iter()
            .map(JetBivariate2::axis0)
            .map(|each| each.view())
            .collect::<Vec<_>>();

        let axis1_views = values
            .iter()
            .map(JetBivariate2::axis1)
            .map(|each| each.view())
            .collect::<Vec<_>>();

        let axis0_axis0_views = values
            .iter()
            .map(JetBivariate2::axis0_axis0)
            .map(|each| each.view())
            .collect::<Vec<_>>();

        let axis0_axis1_views = values
            .iter()
            .map(JetBivariate2::axis0_axis1)
            .map(|each| each.view())
            .collect::<Vec<_>>();

        let axis1_axis1_views = values
            .iter()
            .map(JetBivariate2::axis1_axis1)
            .map(|each| each.view())
            .collect::<Vec<_>>();

        let value = ndarray::stack(axis, &value_views)?;
        let axis0 = ndarray::stack(axis, &axis0_views)?;
        let axis1 = ndarray::stack(axis, &axis1_views)?;
        let axis0_axis0 = ndarray::stack(axis, &axis0_axis0_views)?;
        let axis0_axis1 = ndarray::stack(axis, &axis0_axis1_views)?;
        let axis1_axis1 = ndarray::stack(axis, &axis1_axis1_views)?;

        Ok(Some(JetBivariate2::from_parts(
            value,
            BivariateGradient::new(axis0, axis1),
            BivariateHessian::new(axis0_axis0, axis0_axis1, axis1_axis1),
        )))
    }
}

#[cfg(test)]
mod jet0_tests {
    use crate::algebra::RealParameter;

    use super::*;

    use ndarray::{Ix1, arr0, arr1, arr2, array};

    #[test]
    fn empty_stack_returns_none() {
        let values = Vec::<ArrayJet0<f64, Ix1, ()>>::new();

        assert!(JetStack::stack(values).unwrap().is_none());
    }

    #[test]
    fn stacks_scalar_arrays_into_vector() {
        let values = vec![
            Jet0::<_, RealParameter>::new(arr0(1.0)),
            Jet0::new(arr0(2.0)),
            Jet0::new(arr0(3.0)),
        ];

        let stacked = JetStack::stack(values).unwrap().unwrap();

        assert_eq!(stacked.into_inner(), arr1(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn stacks_vectors_along_final_axis() {
        let values = vec![
            Jet0::<_, RealParameter>::new(arr1(&[1.0, 2.0])),
            Jet0::new(arr1(&[3.0, 4.0])),
            Jet0::new(arr1(&[5.0, 6.0])),
        ];

        let stacked = JetStack::stack(values).unwrap().unwrap();

        assert_eq!(
            stacked.into_inner(),
            array![[1.0, 3.0, 5.0], [2.0, 4.0, 6.0],],
        );
    }

    #[test]
    fn stacks_matrices_along_final_axis() {
        let values = vec![
            Jet0::<_, RealParameter>::new(arr2(&[[1.0, 2.0], [3.0, 4.0]])),
            Jet0::new(arr2(&[[5.0, 6.0], [7.0, 8.0]])),
        ];

        let stacked = JetStack::stack(values).unwrap().unwrap();

        assert_eq!(stacked.clone().into_inner().shape(), &[2, 2, 2]);

        assert_eq!(stacked.into_inner()[[1, 0, 1]], 7.0);
    }

    #[test]
    fn rejects_incompatible_shapes() {
        let values = vec![
            Jet0::<_, RealParameter>::new(arr1(&[1.0, 2.0])),
            Jet0::new(arr1(&[3.0])),
        ];

        assert!(JetStack::stack(values).is_err());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::{Ix0, Ix1, arr0, arr1, array};

    type J0<D> = ArrayJet0<f64, D, ()>;
    type J1<D> = ArrayJet1<f64, D, ()>;
    type J2<D> = ArrayJet2<f64, D, ()>;
    type JB1<D> = ArrayJetBivariate1<f64, D, ()>;
    type JB2<D> = ArrayJetBivariate2<f64, D, ()>;

    // ---------------------------------------------------------------------
    // ArrayJet0
    // ---------------------------------------------------------------------

    #[test]
    fn jet0_empty_stack_returns_none() {
        let values = Vec::<J0<Ix0>>::new();

        assert!(J0::stack(values).unwrap().is_none());
    }

    #[test]
    fn jet0_stacks_scalar_values() {
        let values = vec![J0::new(arr0(1.0)), J0::new(arr0(2.0)), J0::new(arr0(3.0))];

        let stacked = J0::stack(values).unwrap().unwrap();

        assert_eq!(stacked.value(), &arr1(&[1.0, 2.0, 3.0]));
    }

    #[test]
    fn jet0_stacks_along_new_final_axis() {
        let values = vec![
            J0::new(arr1(&[1.0, 2.0])),
            J0::new(arr1(&[3.0, 4.0])),
            J0::new(arr1(&[5.0, 6.0])),
        ];

        let stacked = J0::stack(values).unwrap().unwrap();

        assert_eq!(stacked.value(), &array![[1.0, 3.0, 5.0], [2.0, 4.0, 6.0],],);
    }

    #[test]
    fn jet0_rejects_mismatched_shapes() {
        let values = vec![J0::new(arr1(&[1.0, 2.0])), J0::new(arr1(&[3.0]))];

        assert!(J0::stack(values).is_err());
    }

    // ---------------------------------------------------------------------
    // ArrayJet1
    // ---------------------------------------------------------------------

    #[test]
    fn jet1_empty_stack_returns_none() {
        let values = Vec::<J1<Ix0>>::new();

        assert!(J1::stack(values).unwrap().is_none());
    }

    #[test]
    fn jet1_stacks_value_and_first_derivative() {
        let values = vec![
            J1::from_parts(arr0(1.0), arr0(10.0)),
            J1::from_parts(arr0(2.0), arr0(20.0)),
            J1::from_parts(arr0(3.0), arr0(30.0)),
        ];

        let stacked = J1::stack(values).unwrap().unwrap();

        assert_eq!(stacked.value(), &arr1(&[1.0, 2.0, 3.0]));
        assert_eq!(stacked.first(), &arr1(&[10.0, 20.0, 30.0]));
    }

    #[test]
    fn jet1_stacks_every_part_along_final_axis() {
        let values = vec![
            J1::from_parts(arr1(&[1.0, 2.0]), arr1(&[10.0, 20.0])),
            J1::from_parts(arr1(&[3.0, 4.0]), arr1(&[30.0, 40.0])),
        ];

        let stacked = J1::stack(values).unwrap().unwrap();

        assert_eq!(stacked.value(), &array![[1.0, 3.0], [2.0, 4.0],],);

        assert_eq!(stacked.first(), &array![[10.0, 30.0], [20.0, 40.0],],);
    }

    #[test]
    fn jet1_rejects_mismatched_shapes() {
        let values = vec![
            J1::from_parts(arr1(&[1.0, 2.0]), arr1(&[10.0, 20.0])),
            J1::from_parts(arr1(&[3.0]), arr1(&[30.0])),
        ];

        assert!(J1::stack(values).is_err());
    }

    // ---------------------------------------------------------------------
    // ArrayJet2
    // ---------------------------------------------------------------------

    #[test]
    fn jet2_empty_stack_returns_none() {
        let values = Vec::<J2<Ix0>>::new();

        assert!(J2::stack(values).unwrap().is_none());
    }

    #[test]
    fn jet2_stacks_all_directional_coefficients() {
        let values = vec![
            J2::from_parts(arr0(1.0), arr0(10.0), arr0(100.0)),
            J2::from_parts(arr0(2.0), arr0(20.0), arr0(200.0)),
            J2::from_parts(arr0(3.0), arr0(30.0), arr0(300.0)),
        ];

        let stacked = J2::stack(values).unwrap().unwrap();

        assert_eq!(stacked.value(), &arr1(&[1.0, 2.0, 3.0]));
        assert_eq!(stacked.first(), &arr1(&[10.0, 20.0, 30.0]));
        assert_eq!(stacked.second(), &arr1(&[100.0, 200.0, 300.0]));
    }

    #[test]
    fn jet2_stacks_every_part_along_final_axis() {
        let values = vec![
            J2::from_parts(
                arr1(&[1.0, 2.0]),
                arr1(&[10.0, 20.0]),
                arr1(&[100.0, 200.0]),
            ),
            J2::from_parts(
                arr1(&[3.0, 4.0]),
                arr1(&[30.0, 40.0]),
                arr1(&[300.0, 400.0]),
            ),
        ];

        let stacked = J2::stack(values).unwrap().unwrap();

        assert_eq!(stacked.value(), &array![[1.0, 3.0], [2.0, 4.0]]);
        assert_eq!(stacked.first(), &array![[10.0, 30.0], [20.0, 40.0]]);
        assert_eq!(stacked.second(), &array![[100.0, 300.0], [200.0, 400.0]]);
    }

    #[test]
    fn jet2_rejects_mismatched_shapes() {
        let values = vec![
            J2::from_parts(
                arr1(&[1.0, 2.0]),
                arr1(&[10.0, 20.0]),
                arr1(&[100.0, 200.0]),
            ),
            J2::from_parts(arr1(&[3.0]), arr1(&[30.0]), arr1(&[300.0])),
        ];

        assert!(J2::stack(values).is_err());
    }

    // ---------------------------------------------------------------------
    // ArrayJetBivariate1
    // ---------------------------------------------------------------------

    #[test]
    fn bivariate1_empty_stack_returns_none() {
        let values = Vec::<JB1<Ix0>>::new();

        assert!(JB1::stack(values).unwrap().is_none());
    }

    #[test]
    fn bivariate1_stacks_value_and_gradient() {
        let values = vec![
            JB1::from_parts(arr0(1.0), BivariateGradient::new(arr0(10.0), arr0(100.0))),
            JB1::from_parts(arr0(2.0), BivariateGradient::new(arr0(20.0), arr0(200.0))),
            JB1::from_parts(arr0(3.0), BivariateGradient::new(arr0(30.0), arr0(300.0))),
        ];

        let stacked = JB1::stack(values).unwrap().unwrap();

        assert_eq!(stacked.value(), &arr1(&[1.0, 2.0, 3.0]));
        assert_eq!(stacked.axis0(), &arr1(&[10.0, 20.0, 30.0]));
        assert_eq!(stacked.axis1(), &arr1(&[100.0, 200.0, 300.0]));
    }

    #[test]
    fn bivariate1_stacks_every_part_along_final_axis() {
        let values = vec![
            JB1::from_parts(
                arr1(&[1.0, 2.0]),
                BivariateGradient::new(arr1(&[10.0, 20.0]), arr1(&[100.0, 200.0])),
            ),
            JB1::from_parts(
                arr1(&[3.0, 4.0]),
                BivariateGradient::new(arr1(&[30.0, 40.0]), arr1(&[300.0, 400.0])),
            ),
        ];

        let stacked = JB1::stack(values).unwrap().unwrap();

        assert_eq!(stacked.value(), &array![[1.0, 3.0], [2.0, 4.0]]);
        assert_eq!(stacked.axis0(), &array![[10.0, 30.0], [20.0, 40.0]]);
        assert_eq!(stacked.axis1(), &array![[100.0, 300.0], [200.0, 400.0]]);
    }

    #[test]
    fn bivariate1_rejects_mismatched_shapes() {
        let values = vec![
            JB1::from_parts(
                arr1(&[1.0, 2.0]),
                BivariateGradient::new(arr1(&[10.0, 20.0]), arr1(&[100.0, 200.0])),
            ),
            JB1::from_parts(
                arr1(&[3.0]),
                BivariateGradient::new(arr1(&[30.0]), arr1(&[300.0])),
            ),
        ];

        assert!(JB1::stack(values).is_err());
    }

    // ---------------------------------------------------------------------
    // ArrayJetBivariate2
    // ---------------------------------------------------------------------

    #[test]
    fn bivariate2_empty_stack_returns_none() {
        let values = Vec::<JB2<Ix0>>::new();

        assert!(JB2::stack(values).unwrap().is_none());
    }

    #[test]
    fn bivariate2_stacks_value_gradient_and_hessian() {
        let values = vec![
            JB2::from_parts(
                arr0(1.0),
                BivariateGradient::new(arr0(10.0), arr0(20.0)),
                BivariateHessian::new(arr0(100.0), arr0(200.0), arr0(300.0)),
            ),
            JB2::from_parts(
                arr0(2.0),
                BivariateGradient::new(arr0(30.0), arr0(40.0)),
                BivariateHessian::new(arr0(400.0), arr0(500.0), arr0(600.0)),
            ),
        ];

        let stacked = JB2::stack(values).unwrap().unwrap();

        assert_eq!(stacked.value(), &arr1(&[1.0, 2.0]));

        assert_eq!(stacked.axis0(), &arr1(&[10.0, 30.0]));
        assert_eq!(stacked.axis1(), &arr1(&[20.0, 40.0]));

        assert_eq!(stacked.axis0_axis0(), &arr1(&[100.0, 400.0]));
        assert_eq!(stacked.axis0_axis1(), &arr1(&[200.0, 500.0]));
        assert_eq!(stacked.axis1_axis1(), &arr1(&[300.0, 600.0]));
    }

    #[test]
    fn bivariate2_stacks_every_part_along_final_axis() {
        let values = vec![
            JB2::from_parts(
                arr1(&[1.0, 2.0]),
                BivariateGradient::new(arr1(&[10.0, 20.0]), arr1(&[30.0, 40.0])),
                BivariateHessian::new(
                    arr1(&[100.0, 200.0]),
                    arr1(&[300.0, 400.0]),
                    arr1(&[500.0, 600.0]),
                ),
            ),
            JB2::from_parts(
                arr1(&[3.0, 4.0]),
                BivariateGradient::new(arr1(&[50.0, 60.0]), arr1(&[70.0, 80.0])),
                BivariateHessian::new(
                    arr1(&[700.0, 800.0]),
                    arr1(&[900.0, 1_000.0]),
                    arr1(&[1_100.0, 1_200.0]),
                ),
            ),
        ];

        let stacked = JB2::stack(values).unwrap().unwrap();

        assert_eq!(stacked.value(), &array![[1.0, 3.0], [2.0, 4.0]]);

        assert_eq!(stacked.axis0(), &array![[10.0, 50.0], [20.0, 60.0]]);
        assert_eq!(stacked.axis1(), &array![[30.0, 70.0], [40.0, 80.0]]);

        assert_eq!(
            stacked.axis0_axis0(),
            &array![[100.0, 700.0], [200.0, 800.0]]
        );
        assert_eq!(
            stacked.axis0_axis1(),
            &array![[300.0, 900.0], [400.0, 1_000.0]]
        );
        assert_eq!(
            stacked.axis1_axis1(),
            &array![[500.0, 1_100.0], [600.0, 1_200.0]]
        );
    }

    #[test]
    fn bivariate2_rejects_mismatched_shapes() {
        let values = vec![
            JB2::from_parts(
                arr1(&[1.0, 2.0]),
                BivariateGradient::new(arr1(&[10.0, 20.0]), arr1(&[30.0, 40.0])),
                BivariateHessian::new(
                    arr1(&[100.0, 200.0]),
                    arr1(&[300.0, 400.0]),
                    arr1(&[500.0, 600.0]),
                ),
            ),
            JB2::from_parts(
                arr1(&[3.0]),
                BivariateGradient::new(arr1(&[50.0]), arr1(&[70.0])),
                BivariateHessian::new(arr1(&[700.0]), arr1(&[900.0]), arr1(&[1_100.0])),
            ),
        ];

        assert!(JB2::stack(values).is_err());
    }
}
