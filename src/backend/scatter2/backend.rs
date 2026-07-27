use crate::{
    ComplexScalar,
    algebra::ScalarAlgebra,
    backend::{InternalFieldRequest, IntoEntries, isotropic::IsotropicLayerQuantities},
    input::{CanonicalSolverInput, CanonicalStack},
    material::{ConstitutiveEvaluator, ConstitutiveLift},
};

use super::{
    Scatter2, Scatter2Entries, Scatter2Error, Scatter2Workspace,
    component::{interface, propagation_from_exponent},
};

use ndarray::Dimension;

impl<J> Scatter2<J> {
    pub(super) fn evaluate<E, M, C, D>(
        &self,
        input: &CanonicalSolverInput<J>,
        stack: &CanonicalStack<M, J>,
    ) -> Result<Scatter2Entries<J>, Scatter2Error>
    where
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
        E: ConstitutiveEvaluator<C, D, M>,
        J: ScalarAlgebra<C, D> + ConstitutiveLift<C, D, E, M> + Clone,
    {
        let workspace = self.accumulate::<E, M, C, D>(input, stack, InternalFieldRequest::None)?;

        Ok(workspace.into_entries())
    }

    pub(crate) fn accumulate<E, M, C, D>(
        &self,
        input: &CanonicalSolverInput<J>,
        stack: &CanonicalStack<M, J>,
        request: InternalFieldRequest,
    ) -> Result<Scatter2Workspace<J>, Scatter2Error>
    where
        C: ComplexScalar,
        C::RealField: Copy,
        D: Dimension,
        E: ConstitutiveEvaluator<C, D, M>,
        J: ScalarAlgebra<C, D> + ConstitutiveLift<C, D, E, M> + Clone,
    {
        let mut workspace = Scatter2Workspace::new(
            input.vacuum_angular_wavenumber().value(),
            request,
            stack.layer_count(),
        );

        let left_quantities =
            IsotropicLayerQuantities::evaluate::<C, D, E, M>(stack.left_exterior(), input);

        let mut current_admittance = left_quantities.into_admittance().into_inner();

        for (index, layer) in stack.layers().iter().enumerate() {
            let quantities =
                IsotropicLayerQuantities::evaluate::<C, D, E, M>(layer.material(), input);

            let imaginary_unit =
                J::filled_constant_like(input.vacuum_angular_wavenumber().value(), C::i());

            let exponent = quantities
                .kappa()
                .multiply(&imaginary_unit)
                .multiply(layer.thickness_cm());

            let layer_admittance = quantities.into_admittance().into_inner();

            let interface = interface::<C, D, J>(&current_admittance, &layer_admittance);

            let propagation = propagation_from_exponent::<C, D, J>(exponent);

            workspace.append_layer::<C, D>(interface, propagation);

            current_admittance = layer_admittance;
        }

        let right_quantities =
            IsotropicLayerQuantities::evaluate::<C, D, E, M>(stack.right_exterior(), input);

        let right_admittance = right_quantities.into_admittance().into_inner();

        let final_interface = interface::<C, D, J>(&current_admittance, &right_admittance);

        workspace.append(final_interface);

        Ok(workspace)
    }
}
