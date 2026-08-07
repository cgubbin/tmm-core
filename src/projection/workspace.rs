use crate::backend::{
    ExteriorContextProvider, IsotropicLayerQuantities, PlaneWaveEntries, PlaneWaveSolution,
    scatter2::{
        RetainedScatterComponents, Scatter2Entries, Scatter2ExteriorContext,
        Scatter2ProjectiveEntries, Scatter2Workspace,
    },
    transfer2::{
        RetainedTransferLayer, RetainedTransferLayers, Transfer2Entries, Transfer2ExteriorContext,
        Transfer2Workspace,
    },
};

use super::{PointProjectionError, ProjectPoint};

impl<J> ProjectPoint for Scatter2Workspace<J>
where
    J: ProjectPoint,
{
    type Dimension = J::Dimension;
    type Point = Scatter2Workspace<J::Point>;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: ndarray::NdIndex<Self::Dimension> + Clone,
    {
        Ok(Scatter2Workspace::from_parts(
            self.solution().project_point(index)?,
            self.retained()
                .map(|retained| retained.project_point(index))
                .transpose()?,
        ))
    }
}

impl<J> ProjectPoint for Transfer2Workspace<J>
where
    J: ProjectPoint,
{
    type Dimension = J::Dimension;
    type Point = Transfer2Workspace<J::Point>;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: ndarray::NdIndex<Self::Dimension> + Clone,
    {
        Ok(Transfer2Workspace::from_parts(
            self.solution().project_point(index)?,
            self.retained()
                .map(|retained| retained.project_point(index))
                .transpose()?,
        ))
    }
}

impl<X> ProjectPoint for PlaneWaveSolution<X>
where
    X: ProjectPoint + PlaneWaveEntries,
    X::ExteriorContext: ProjectPoint<
            Dimension = X::Dimension,
            Point = <X::Point as PlaneWaveEntries>::ExteriorContext,
        >,
    X::Point: PlaneWaveEntries,
{
    type Dimension = X::Dimension;
    type Point = PlaneWaveSolution<X::Point>;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: ndarray::NdIndex<Self::Dimension> + Clone,
    {
        Ok(PlaneWaveSolution::new(
            self.entries().project_point(index)?,
            self.context().project_point(index)?,
        ))
    }
}

impl<J> ProjectPoint for Scatter2ProjectiveEntries<J>
where
    J: ProjectPoint,
{
    type Dimension = J::Dimension;
    type Point = Scatter2ProjectiveEntries<J::Point>;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: ndarray::NdIndex<Self::Dimension> + Clone,
    {
        Ok(Scatter2ProjectiveEntries::from_parts(
            self.denominator().project_point(index)?,
            self.n11().project_point(index)?,
            self.n12().project_point(index)?,
            self.n21().project_point(index)?,
            self.n22().project_point(index)?,
        ))
    }
}

impl<J> ProjectPoint for Scatter2Entries<J>
where
    J: ProjectPoint,
{
    type Dimension = J::Dimension;
    type Point = Scatter2Entries<J::Point>;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: ndarray::NdIndex<Self::Dimension> + Clone,
    {
        Ok(Scatter2Entries::from_parts(
            self.s11().project_point(index)?,
            self.s12().project_point(index)?,
            self.s21().project_point(index)?,
            self.s22().project_point(index)?,
        ))
    }
}

impl<J> ProjectPoint for Scatter2ExteriorContext<J>
where
    J: ProjectPoint,
{
    type Dimension = J::Dimension;
    type Point = Scatter2ExteriorContext<J::Point>;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: ndarray::NdIndex<Self::Dimension> + Clone,
    {
        Ok(Scatter2ExteriorContext::from_parts(
            self.left_admittance().project_point(index)?,
            self.right_admittance().project_point(index)?,
            self.left_kappa().project_point(index)?,
            self.right_kappa().project_point(index)?,
        ))
    }
}

impl<J> ProjectPoint for Transfer2Entries<J>
where
    J: ProjectPoint,
{
    type Dimension = J::Dimension;
    type Point = Transfer2Entries<J::Point>;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: ndarray::NdIndex<Self::Dimension> + Clone,
    {
        Ok(Transfer2Entries::new(
            self.m11().project_point(index)?,
            self.m12().project_point(index)?,
            self.m21().project_point(index)?,
            self.m22().project_point(index)?,
        ))
    }
}

impl<J> ProjectPoint for Transfer2ExteriorContext<J>
where
    J: ProjectPoint,
{
    type Dimension = J::Dimension;
    type Point = Transfer2ExteriorContext<J::Point>;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: ndarray::NdIndex<Self::Dimension> + Clone,
    {
        Ok(Transfer2ExteriorContext::from_parts(
            self.left_admittance().project_point(index)?,
            self.right_admittance().project_point(index)?,
            self.left_kappa().project_point(index)?,
            self.right_kappa().project_point(index)?,
        ))
    }
}

impl<J> ProjectPoint for RetainedScatterComponents<J>
where
    J: ProjectPoint,
{
    type Dimension = J::Dimension;
    type Point = RetainedScatterComponents<J::Point>;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: ndarray::NdIndex<Self::Dimension> + Clone,
    {
        Ok(RetainedScatterComponents::from_parts(
            self.components()
                .iter()
                .map(|each| each.project_point(index))
                .collect::<Result<_, _>>()?,
            self.layer_cuts().clone(),
            self.quantities()
                .iter()
                .map(|each| each.project_point(index))
                .collect::<Result<_, _>>()?,
            self.thicknesses()
                .iter()
                .map(|each| each.project_point(index))
                .collect::<Result<_, _>>()?,
        ))
    }
}

impl<J> ProjectPoint for RetainedTransferLayers<J>
where
    J: ProjectPoint,
{
    type Dimension = J::Dimension;
    type Point = RetainedTransferLayers<J::Point>;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: ndarray::NdIndex<Self::Dimension> + Clone,
    {
        Ok(RetainedTransferLayers::from_layers(
            self.layers()
                .iter()
                .map(|each| each.project_point(index))
                .collect::<Result<_, _>>()?,
        ))
    }
}

impl<J> ProjectPoint for RetainedTransferLayer<J>
where
    J: ProjectPoint,
{
    type Dimension = J::Dimension;
    type Point = RetainedTransferLayer<J::Point>;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: ndarray::NdIndex<Self::Dimension> + Clone,
    {
        Ok(RetainedTransferLayer::new(
            self.matrix().project_point(index)?,
            self.quantities().project_point(index)?,
            self.thickness().project_point(index)?,
        ))
    }
}

impl<J> ProjectPoint for IsotropicLayerQuantities<J>
where
    J: ProjectPoint,
{
    type Dimension = J::Dimension;
    type Point = IsotropicLayerQuantities<J::Point>;

    fn project_point<I>(&self, index: &I) -> Result<Self::Point, PointProjectionError>
    where
        I: ndarray::NdIndex<Self::Dimension> + Clone,
    {
        Ok(IsotropicLayerQuantities::from_parts(
            self.epsilon().project_point(index)?,
            self.mu().project_point(index)?,
            self.kappa().project_point(index)?,
            self.polarisation(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Ix1, arr1};
    use num_complex::Complex64;

    use super::*;

    use crate::{
        Polarisation,
        algebra::{ArrayJet0, Jet0, RealParameter},
        backend::{
            IsotropicLayerQuantities,
            scatter2::{Scatter2Entries, Scatter2ExteriorContext},
            transfer2::{Transfer2Entries, Transfer2ExteriorContext},
        },
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

    #[test]
    fn scatter_entries_project_all_matrix_elements() {
        let entries = Scatter2Entries::from_parts(
            jet(&[2.0, 3.0]),
            jet(&[5.0, 7.0]),
            jet(&[11.0, 13.0]),
            jet(&[17.0, 19.0]),
        );

        let point = entries.project_point(&1).unwrap();

        assert_eq!(point.s11().value()[()], Complex64::new(3.0, 0.0),);

        assert_eq!(point.s12().value()[()], Complex64::new(7.0, 0.0),);

        assert_eq!(point.s21().value()[()], Complex64::new(13.0, 0.0),);

        assert_eq!(point.s22().value()[()], Complex64::new(19.0, 0.0),);
    }

    #[test]
    fn transfer_entries_project_all_matrix_elements() {
        let entries = Transfer2Entries::new(
            jet(&[2.0, 3.0]),
            jet(&[5.0, 7.0]),
            jet(&[11.0, 13.0]),
            jet(&[17.0, 19.0]),
        );

        let point = entries.project_point(&0).unwrap();

        assert_eq!(point.m11().value()[()], Complex64::new(2.0, 0.0),);

        assert_eq!(point.m12().value()[()], Complex64::new(5.0, 0.0),);

        assert_eq!(point.m21().value()[()], Complex64::new(11.0, 0.0),);

        assert_eq!(point.m22().value()[()], Complex64::new(17.0, 0.0),);
    }

    #[test]
    fn scatter_exterior_context_projects_both_admittances() {
        let context = Scatter2ExteriorContext::from_parts(
            jet(&[2.0, 3.0]),
            jet(&[5.0, 7.0]),
            jet(&[0.0, 0.0]),
            jet(&[0.0, 0.0]),
        );

        let point = context.project_point(&1).unwrap();

        assert_eq!(
            point.left_admittance().value()[()],
            Complex64::new(3.0, 0.0),
        );

        assert_eq!(
            point.right_admittance().value()[()],
            Complex64::new(7.0, 0.0),
        );
    }

    #[test]
    fn transfer_exterior_context_projects_both_admittances() {
        let context = Transfer2ExteriorContext::from_parts(
            jet(&[2.0, 3.0]),
            jet(&[5.0, 7.0]),
            jet(&[0.0, 0.0]),
            jet(&[0.0, 0.0]),
        );

        let point = context.project_point(&0).unwrap();

        assert_eq!(
            point.left_admittance().value()[()],
            Complex64::new(2.0, 0.0),
        );

        assert_eq!(
            point.right_admittance().value()[()],
            Complex64::new(5.0, 0.0),
        );
    }

    #[test]
    fn isotropic_quantities_project_every_sampled_quantity() {
        let quantities = IsotropicLayerQuantities::from_parts(
            jet(&[2.0, 3.0]),
            jet(&[5.0, 7.0]),
            jet(&[11.0, 13.0]),
            Polarisation::TransverseElectric,
        );

        let point = quantities.project_point(&1).unwrap();

        assert_eq!(point.epsilon().value()[()], Complex64::new(3.0, 0.0),);

        assert_eq!(point.mu().value()[()], Complex64::new(7.0, 0.0),);

        assert_eq!(point.kappa().value()[()], Complex64::new(13.0, 0.0),);

        assert_eq!(point.polarisation(), Polarisation::TransverseElectric,);
    }
}
