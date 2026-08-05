use ndarray::{Dimension, NdIndex};

use crate::input::{
    CanonicalCoordinates, CanonicalProblem, CanonicalStack, canonical::CanonicalLayer,
};

use super::{JetPointProjection, PointProjectionError, ProjectPoint};

impl<M, J> ProjectPoint for CanonicalProblem<M, J>
where
    J: JetPointProjection,
    J::Dimension: Dimension,
    M: Clone,
{
    type Dimension = J::Dimension;
    type Point = CanonicalProblem<M, J::PointJet>;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: NdIndex<Self::Dimension> + Clone,
    {
        Ok(CanonicalProblem::new(
            self.coordinates().project_point(index)?,
            self.stack().project_point(index)?,
        ))
    }
}

impl<J> ProjectPoint for CanonicalCoordinates<J>
where
    J: JetPointProjection,
    J::Dimension: Dimension,
{
    type Dimension = J::Dimension;
    type Point = CanonicalCoordinates<J::PointJet>;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: NdIndex<Self::Dimension> + Clone,
    {
        Ok(CanonicalCoordinates::new(
            self.vacuum_angular_wavenumber().project_point(index)?,
            self.parallel_angular_wavenumber().project_point(index)?,
        ))
    }
}

impl<M, J> ProjectPoint for CanonicalStack<M, J>
where
    J: JetPointProjection,
    J::Dimension: Dimension,
    M: Clone,
{
    type Dimension = J::Dimension;
    type Point = CanonicalStack<M, J::PointJet>;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: NdIndex<Self::Dimension> + Clone,
    {
        Ok(CanonicalStack::new(
            self.left_exterior().clone(),
            self.right_exterior().clone(),
            self.layers()
                .iter()
                .map(|each| each.project_point(index))
                .collect::<Result<Vec<_>, _>>()?,
        ))
    }
}

impl<M, J> ProjectPoint for CanonicalLayer<M, J>
where
    J: JetPointProjection,
    J::Dimension: Dimension,
    M: Clone,
{
    type Dimension = J::Dimension;
    type Point = CanonicalLayer<M, J::PointJet>;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: NdIndex<Self::Dimension> + Clone,
    {
        Ok(CanonicalLayer::new(
            self.material().clone(),
            self.thickness_cm().project_point(index)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Ix0, Ix1, arr1};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        algebra::{ArrayJet0, Jet0, RealParameter},
        input::{CanonicalCoordinates, CanonicalProblem, CanonicalStack},
        material::MaterialHandle,
        test_support::materials::vacuum,
    };

    type J = ArrayJet0<Complex64, Ix1, RealParameter>;

    fn jet(values: &[f64]) -> J {
        Jet0::new(arr1(
            &values
                .iter()
                .map(|value| Complex64::new(*value, 0.0))
                .collect::<Vec<_>>(),
        ))
    }

    fn material_fixture() -> MaterialHandle<Complex64> {
        MaterialHandle::new(vacuum())
    }

    fn problem_fixture() -> CanonicalProblem<MaterialHandle<Complex64>, J> {
        let coordinates = CanonicalCoordinates::new(jet(&[2.0, 3.0, 5.0]), jet(&[7.0, 11.0, 13.0]));

        let stack = CanonicalStack::new(
            material_fixture(),
            material_fixture(),
            vec![
                CanonicalLayer::new(material_fixture(), jet(&[17.0, 19.0, 23.0])),
                CanonicalLayer::new(material_fixture(), jet(&[29.0, 31.0, 37.0])),
            ],
        );

        CanonicalProblem::new(coordinates, stack)
    }

    #[test]
    fn coordinates_project_both_canonical_axes() {
        let coordinates = problem_fixture().coordinates().project_point(&1).unwrap();

        assert_eq!(
            coordinates.vacuum_angular_wavenumber().value()[()],
            Complex64::new(3.0, 0.0),
        );

        assert_eq!(
            coordinates.parallel_angular_wavenumber().value()[()],
            Complex64::new(11.0, 0.0),
        );
    }

    #[test]
    fn canonical_layer_projects_thickness() {
        let point = problem_fixture().stack().layers()[0]
            .project_point(&2)
            .unwrap();

        assert_eq!(point.thickness_cm().value()[()], Complex64::new(23.0, 0.0),);
    }

    #[test]
    fn canonical_stack_preserves_layer_count_and_order() {
        let point = problem_fixture().stack().project_point(&1).unwrap();

        assert_eq!(point.layers().len(), 2);

        assert_eq!(
            point.layers()[0].thickness_cm().value()[()],
            Complex64::new(19.0, 0.0),
        );

        assert_eq!(
            point.layers()[1].thickness_cm().value()[()],
            Complex64::new(31.0, 0.0),
        );
    }

    #[test]
    fn canonical_stack_preserves_material_handles() {
        let source = problem_fixture();

        let point = source.stack().project_point(&1).unwrap();

        /*
         * Prefer pointer identity if MaterialHandle is Arc-backed.
         * Otherwise use its existing identity/equality assertion.
         */
        assert_eq!(point.left_exterior(), source.stack().left_exterior(),);

        assert_eq!(point.right_exterior(), source.stack().right_exterior(),);

        assert_eq!(
            point.layers()[0].material(),
            source.stack().layers()[0].material(),
        );
    }

    #[test]
    fn canonical_problem_projects_coordinates_and_stack() {
        let point = problem_fixture().project_point(&2).unwrap();

        assert_eq!(
            point.coordinates().vacuum_angular_wavenumber().value()[()],
            Complex64::new(5.0, 0.0),
        );

        assert_eq!(point.stack().layers().len(), 2);

        assert_eq!(
            point.stack().layers()[1].thickness_cm().value()[()],
            Complex64::new(37.0, 0.0),
        );
    }

    #[test]
    fn canonical_problem_projection_rejects_invalid_index() {
        let error = problem_fixture().project_point(&3).unwrap_err();

        assert_eq!(error, PointProjectionError::OutOfBounds { shape: vec![3] },);
    }
}
