use nalgebra::ComplexField;
use ndarray::{Dimension, Ix0, arr0};

use crate::input::{CompilationContext, CoordinateContext, CoordinateValues};

use super::{PointProjectionError, ProjectPoint, project_array_point};

impl<C, D, M> ProjectPoint for CompilationContext<C, D, M>
where
    C: ComplexField + Copy,
    D: Dimension,
    M: Clone,
{
    type Point = CompilationContext<C, Ix0, M>;
    type Dimension = D;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: ndarray::NdIndex<Self::Dimension> + Clone,
    {
        Ok(CompilationContext::new(
            self.coordinates().project_point(index)?,
            self.stack().clone(),
            self.mapping().clone(),
            self.projection_constraint(),
        ))
    }
}

impl<C, D> ProjectPoint for CoordinateContext<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Point = CoordinateContext<C, Ix0>;
    type Dimension = D;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: ndarray::NdIndex<Self::Dimension> + Clone,
    {
        Ok(CoordinateContext::new(
            self.coordinates(),
            self.values().project_point(index)?,
        ))
    }
}

impl<C, D> ProjectPoint for CoordinateValues<C, D>
where
    C: ComplexField + Copy,
    D: Dimension,
{
    type Point = CoordinateValues<C, Ix0>;
    type Dimension = D;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: ndarray::NdIndex<Self::Dimension> + Clone,
    {
        Ok(CoordinateValues::new(
            arr0(project_array_point(self.spectral(), index)?),
            arr0(project_array_point(self.in_plane(), index)?),
        ))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Ix1, arr1};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        Coordinates, InPlaneCoordinate, SpectralCoordinate,
        derivative_parts::ValueOnly,
        input::{
            CompilationContext, CoordinateContext, CoordinateValues, ProjectionConstraint,
            compile::StackContext,
        },
    };

    fn values() -> CoordinateValues<Complex64, Ix1> {
        CoordinateValues::new(
            arr1(&[
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(5.0, 0.0),
            ]),
            arr1(&[
                Complex64::new(7.0, 0.0),
                Complex64::new(11.0, 0.0),
                Complex64::new(13.0, 0.0),
            ]),
        )
    }

    #[test]
    fn coordinate_values_project_both_arrays() {
        let point = values().project_point(&1).unwrap();

        assert_eq!(point.spectral()[()], Complex64::new(3.0, 0.0),);

        assert_eq!(point.in_plane()[()], Complex64::new(11.0, 0.0),);
    }

    #[test]
    fn coordinate_values_reject_invalid_index() {
        let error = values().project_point(&3).unwrap_err();

        assert_eq!(error, PointProjectionError::OutOfBounds { shape: vec![3] },);
    }

    #[test]
    fn coordinate_context_preserves_coordinate_metadata() {
        let context = CoordinateContext::new(
            Coordinates::new(
                SpectralCoordinate::Frequency(lamina_units::FrequencyUnit::Hertz),
                InPlaneCoordinate::ParallelWavenumber(lamina_units::InverseLengthUnit::PerMetre),
            ),
            values(),
        );

        let point = context.project_point(&2).unwrap();

        assert_eq!(point.coordinates(), context.coordinates(),);

        assert_eq!(point.values().spectral()[()], Complex64::new(5.0, 0.0),);
    }

    #[test]
    fn compilation_context_preserves_mapping_and_constraint() {
        let context = CompilationContext::new(
            CoordinateContext::new(
                Coordinates::new(
                    SpectralCoordinate::Frequency(lamina_units::FrequencyUnit::Hertz),
                    InPlaneCoordinate::ParallelWavenumber(
                        lamina_units::InverseLengthUnit::PerMetre,
                    ),
                ),
                values(),
            ),
            StackContext::new(vec![]),
            ValueOnly,
            ProjectionConstraint::Free,
        );

        let point = context.project_point(&1).unwrap();

        assert_eq!(point.mapping(), context.mapping(),);

        assert_eq!(
            point.projection_constraint(),
            context.projection_constraint(),
        );

        assert_eq!(
            point.coordinates().values().spectral()[()],
            Complex64::new(3.0, 0.0),
        );
    }
}
