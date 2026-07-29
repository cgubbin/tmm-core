use crate::{
    ComplexScalar, Polarisation,
    algebra::ScalarAlgebra,
    backend::{RunMode, isotropic::IsotropicLayerQuantities},
    input::{CanonicalCoordinates, CanonicalSolverInput, CanonicalStack},
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};

use super::{
    Scatter2, Scatter2Entries, Scatter2Error, Scatter2Workspace,
    component::{interface, propagation_from_exponent},
};

use nalgebra::ComplexField;
use ndarray::Dimension;

impl<J> Scatter2<J> {
    pub(crate) fn accumulate<E, M>(
        &self,
        coordinates: &CanonicalCoordinates<J>,
        stack: &CanonicalStack<M, J>,
        polarisation: Polarisation,
        request: RunMode,
    ) -> Result<Scatter2Workspace<J>, Scatter2Error>
    where
        J: ScalarAlgebra + ConstitutiveLift<E, M> + Clone,
        J::Scalar: ComplexScalar,
        <J::Scalar as ComplexField>::RealField: Copy,
        J::Dimension: Dimension,
        E: ConstitutiveEvaluator<J::Scalar, J::Dimension, M>,
    {
        let mut workspace = Scatter2Workspace::new(
            coordinates.vacuum_angular_wavenumber().value(),
            request,
            stack.layer_count(),
        );

        let left_quantities = IsotropicLayerQuantities::evaluate::<E, M>(
            stack.left_exterior(),
            coordinates,
            polarisation,
        );

        let mut current_admittance = left_quantities.into_admittance().into_inner();

        for layer in stack.layers() {
            let quantities = IsotropicLayerQuantities::evaluate::<E, M>(
                layer.material(),
                coordinates,
                polarisation,
            );

            let imaginary_unit = J::filled_constant_like(
                coordinates.vacuum_angular_wavenumber().value(),
                <J::Scalar as ComplexScalar>::i(),
            );

            let exponent = quantities
                .kappa()
                .multiply(&imaginary_unit)
                .multiply(layer.thickness_cm());

            let layer_admittance = quantities.into_admittance().into_inner();

            let interface = interface(&current_admittance, &layer_admittance);

            let propagation = propagation_from_exponent(exponent);

            workspace.append_layer(interface, propagation);

            current_admittance = layer_admittance;
        }

        let right_quantities = IsotropicLayerQuantities::evaluate::<E, M>(
            stack.right_exterior(),
            coordinates,
            polarisation,
        );

        let right_admittance = right_quantities.into_admittance().into_inner();

        let final_interface = interface(&current_admittance, &right_admittance);

        workspace.append(final_interface);

        Ok(workspace)
    }
}
