use crate::{
    ComplexScalar, DerivativeVariable, IncidentSide, PlanarInput, PlaneWaveInput, Polarisation,
    SpectralDerivativeVariable, Stack, StructuralDerivativeVariable,
    backend::{
        algebra::ScalarAlgebra,
        isotropic::{IsotropicLayerAdmittance, IsotropicLayerQuantities},
        jet::{ArrayJet, ArrayJetFirst},
    },
    material::{EvaluateDifferentiableMaterial, EvaluateMaterial},
};

use nalgebra::ComplexField;
use ndarray::{ArrayBase, Dimension, OwnedRepr};

#[derive(Clone, Debug)]
pub(super) struct AlgebraicLayerFieldData<C, A>
where
    C: ComplexField,
{
    pub(super) origin: C::RealField,
    pub(super) thickness: C::RealField,
    pub(super) algebraic_thickness: A,
    pub(super) quantities: IsotropicLayerQuantities<A>,
}

#[derive(Clone, Debug)]
pub(super) struct AlgebraicFieldContext<C, D, A>
where
    C: ComplexField,
    D: Dimension,
    A: ScalarAlgebra<C, D>,
{
    pub(super) planar: PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    pub(super) polarisation: Polarisation,
    pub(super) left: IsotropicLayerQuantities<A>,
    pub(super) right: IsotropicLayerQuantities<A>,
    pub(super) layers: Vec<AlgebraicLayerFieldData<C, A>>,
    pub(super) total_thickness: C::RealField,
}

impl<C, D, A> AlgebraicFieldContext<C, D, A>
where
    C: ComplexField,
    C::RealField: Copy + ComplexField,
    D: Dimension,
    A: ScalarAlgebra<C, D>,
    A::RealField: ScalarAlgebra<C::RealField, D>,
{
    pub(super) fn evaluate<M, F>(
        stack: &Stack<M, C::RealField>,
        planar: PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
        derivative: Option<DerivativeVariable>,
        mut evaluate: F,
    ) -> Self
    where
        F: FnMut(&M, &PlanarInput<ArrayBase<OwnedRepr<C>, D>>) -> IsotropicLayerQuantities<A>,
    {
        let left = evaluate(stack.left_exterior(), &planar);
        let right = evaluate(stack.right_exterior(), &planar);

        let mut origin = C::zero().real();
        let mut layers = Vec::with_capacity(stack.layers_left_to_right().len());

        for (idx, layer) in stack.layers_left_to_right().iter().enumerate() {
            let thickness = layer.thickness().as_cm();

            let algebraic_thickness: A =
                if let Some(DerivativeVariable::Thickness(index)) = derivative {
                    if index == idx {
                        A::structural_like(planar.vacuum_wavenumber(), C::from_real(thickness))
                    } else {
                        A::constant_like(planar.vacuum_wavenumber(), C::from_real(thickness))
                    }
                } else {
                    A::constant_like(planar.vacuum_wavenumber(), C::from_real(thickness))
                };

            layers.push(AlgebraicLayerFieldData {
                origin,
                thickness,
                algebraic_thickness,
                quantities: evaluate(layer.material(), &planar),
            });

            origin = origin + thickness;
        }

        Self {
            polarisation: planar.polarisation(),
            planar,
            left,
            right,
            layers,
            total_thickness: origin,
        }
    }
}

pub(super) fn value_context<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
) -> AlgebraicFieldContext<C, D, ArrayBase<OwnedRepr<C>, D>>
where
    M: EvaluateMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let input = input.to_complex();

    AlgebraicFieldContext::evaluate(stack, input.planar().clone(), None, |material, planar| {
        IsotropicLayerQuantities::real_axis(material, planar)
    })
}

pub(super) fn structural_first_context<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    variable: StructuralDerivativeVariable,
) -> AlgebraicFieldContext<C, D, ArrayJetFirst<C, D>>
where
    M: EvaluateMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let input = input.to_complex();

    AlgebraicFieldContext::evaluate(
        stack,
        input.planar().clone(),
        Some(variable.into()),
        |material, planar| {
            IsotropicLayerQuantities::evaluate_first_structural_real_axis(
                material, planar, variable,
            )
        },
    )
}

pub(super) fn structural_second_context<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    variable: StructuralDerivativeVariable,
) -> AlgebraicFieldContext<C, D, ArrayJet<C, D>>
where
    M: EvaluateMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let input = input.to_complex();

    AlgebraicFieldContext::evaluate(
        stack,
        input.planar().clone(),
        Some(variable.into()),
        |material, planar| {
            IsotropicLayerQuantities::evaluate_second_structural_real_axis(
                material, planar, variable,
            )
        },
    )
}

pub(super) fn spectral_first_context<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    variable: SpectralDerivativeVariable,
) -> AlgebraicFieldContext<C, D, ArrayJetFirst<C, D>>
where
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let input = input.to_complex();

    AlgebraicFieldContext::evaluate(
        stack,
        input.planar().clone(),
        Some(variable.into()),
        |material, planar| {
            IsotropicLayerQuantities::evaluate_first_spectral_real_axis(material, planar, variable)
        },
    )
}

pub(super) fn spectral_second_context<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    variable: SpectralDerivativeVariable,
) -> AlgebraicFieldContext<C, D, ArrayJet<C, D>>
where
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let input = input.to_complex();

    AlgebraicFieldContext::evaluate(
        stack,
        input.planar().clone(),
        Some(variable.into()),
        |material, planar| {
            IsotropicLayerQuantities::evaluate_second_spectral_real_axis(material, planar, variable)
        },
    )
}

#[derive(Clone, Debug)]
pub(super) struct AlgebraicPowerBalanceContext<A> {
    pub(super) incident_side: IncidentSide,

    pub(super) left_admittance: IsotropicLayerAdmittance<A>,
    pub(super) right_admittance: IsotropicLayerAdmittance<A>,

    pub(super) layers: Vec<IsotropicLayerAdmittance<A>>,
}

impl<A> AlgebraicPowerBalanceContext<A> {
    fn evaluate<C, D, M, F>(
        stack: &Stack<M, C::RealField>,
        input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
        mut evaluate: F,
    ) -> Self
    where
        C: ComplexField,
        C::RealField: Copy,
        D: Dimension,
        F: FnMut(&M, &PlanarInput<ArrayBase<OwnedRepr<C>, D>>) -> IsotropicLayerAdmittance<A>,
    {
        let input = input.to_complex();
        let planar = input.planar();

        let left_admittance = evaluate(stack.left_exterior(), planar);
        let right_admittance = evaluate(stack.right_exterior(), planar);

        let layers = stack
            .layers_left_to_right()
            .iter()
            .map(|layer| evaluate(layer.material(), planar))
            .collect();

        Self {
            incident_side: input.incident_side(),
            left_admittance,
            right_admittance,
            layers,
        }
    }
}

pub(super) fn power_balance_value_context<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
) -> AlgebraicPowerBalanceContext<ArrayBase<OwnedRepr<C>, D>>
where
    M: EvaluateMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    AlgebraicPowerBalanceContext::evaluate(stack, input, |material, planar| {
        IsotropicLayerQuantities::real_axis(material, planar).into_admittance()
    })
}

pub(super) fn power_balance_structural_first_context<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    variable: StructuralDerivativeVariable,
) -> AlgebraicPowerBalanceContext<ArrayJetFirst<C, D>>
where
    M: EvaluateMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    AlgebraicPowerBalanceContext::evaluate(stack, input, |material, planar| {
        IsotropicLayerQuantities::evaluate_first_structural_real_axis(material, planar, variable)
            .into_admittance()
    })
}

pub(super) fn power_balance_spectral_first_context<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    variable: SpectralDerivativeVariable,
) -> AlgebraicPowerBalanceContext<ArrayJetFirst<C, D>>
where
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    AlgebraicPowerBalanceContext::evaluate(stack, input, |material, planar| {
        IsotropicLayerQuantities::evaluate_first_spectral_real_axis(material, planar, variable)
            .into_admittance()
    })
}

pub(super) fn power_balance_structural_second_context<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    variable: StructuralDerivativeVariable,
) -> AlgebraicPowerBalanceContext<ArrayJet<C, D>>
where
    M: EvaluateMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    AlgebraicPowerBalanceContext::evaluate(stack, input, |material, planar| {
        IsotropicLayerQuantities::evaluate_second_structural_real_axis(material, planar, variable)
            .into_admittance()
    })
}

pub(super) fn power_balance_spectral_second_context<M, C, D>(
    stack: &Stack<M, C::RealField>,
    input: &PlaneWaveInput<ArrayBase<OwnedRepr<C::RealField>, D>>,
    variable: SpectralDerivativeVariable,
) -> AlgebraicPowerBalanceContext<ArrayJet<C, D>>
where
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    AlgebraicPowerBalanceContext::evaluate(stack, input, |material, planar| {
        IsotropicLayerQuantities::evaluate_second_spectral_real_axis(material, planar, variable)
            .into_admittance()
    })
}
