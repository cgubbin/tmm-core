use crate::{
    ComplexScalar, IncidentSide, PlanarInput, PlaneWaveInput, Polarisation,
    SpectralDerivativeVariable, Stack, StructuralDerivativeVariable,
    backend::{
        algebra::ScalarAlgebra,
        input::AlgebraicPlanarInput,
        isotropic::{IsotropicLayerAdmittance, IsotropicLayerQuantities},
        jet::{ArrayJet, ArrayJetFirst, ArraySpectralJet},
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
pub(super) struct AlgebraicFieldContext<C, A>
where
    C: ComplexField,
    // D: Dimension,
    // A: ScalarAlgebra<C, D>,
{
    pub(super) planar: AlgebraicPlanarInput<A>,
    pub(super) polarisation: Polarisation,
    pub(super) left: IsotropicLayerQuantities<A>,
    pub(super) right: IsotropicLayerQuantities<A>,
    pub(super) layers: Vec<AlgebraicLayerFieldData<C, A>>,
    pub(super) total_thickness: C::RealField,
}

impl<C, A> AlgebraicFieldContext<C, A>
where
    C: ComplexField,
    C::RealField: Copy + ComplexField,
{
    pub(super) fn evaluate<D, M, F, T>(
        stack: &Stack<M, C::RealField>,
        planar: AlgebraicPlanarInput<A>,
        mut evaluate_material: F,
        mut seed_thickness: T,
    ) -> Self
    where
        D: Dimension,
        A: ScalarAlgebra<C, D> + Clone,
        A::RealField: ScalarAlgebra<C::RealField, D>,
        F: FnMut(&M, &AlgebraicPlanarInput<A>) -> IsotropicLayerQuantities<A>,
        T: FnMut(usize, C::RealField, &A) -> A,
    {
        let left = evaluate_material(stack.left_exterior(), &planar);

        let right = evaluate_material(stack.right_exterior(), &planar);

        let mut origin = C::zero().real();
        let mut layers = Vec::with_capacity(stack.layers_left_to_right().len());

        for (index, layer) in stack.layers_left_to_right().iter().enumerate() {
            let thickness = layer.thickness().as_cm();

            let algebraic_thickness = seed_thickness(index, thickness, planar.vacuum_wavenumber());

            let quantities = evaluate_material(layer.material(), &planar);

            layers.push(AlgebraicLayerFieldData {
                origin,
                thickness,
                algebraic_thickness,
                quantities,
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
    planar: PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
) -> AlgebraicFieldContext<C, ArrayBase<OwnedRepr<C>, D>>
where
    C: ComplexScalar,
    D: Dimension,
    M: EvaluateMaterial<C, Real = C::RealField>,
    C::RealField: Copy,
{
    let planar = AlgebraicPlanarInput::values(&planar);

    AlgebraicFieldContext::evaluate(
        stack,
        planar,
        |material, planar| IsotropicLayerQuantities::real_axis(material, planar),
        |_index, thickness, source| {
            <ArrayBase<OwnedRepr<C>, D> as ScalarAlgebra<C, D>>::constant_like(
                source.value(),
                C::from_real(thickness),
            )
        },
    )
}

pub(super) fn thickness_first_context<M, C, D>(
    stack: &Stack<M, C::RealField>,
    planar: PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    layer: usize,
) -> AlgebraicFieldContext<C, ArrayJetFirst<C, D>>
where
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let planar = AlgebraicPlanarInput::new(
        ArrayJetFirst::constant(planar.vacuum_wavenumber().clone()),
        ArrayJetFirst::constant(planar.parallel_wavenumber().clone()),
        planar.polarisation(),
    );

    AlgebraicFieldContext::evaluate(
        stack,
        planar,
        |material, planar| IsotropicLayerQuantities::real_axis(material, planar),
        move |index, value, source| {
            let values = source.value().mapv(|_| C::from_real(value));
            if index == layer {
                ArrayJetFirst::variable(values)
            } else {
                ArrayJetFirst::constant(values)
            }
        },
    )
}

pub(super) fn thickness_second_context<M, C, D>(
    stack: &Stack<M, C::RealField>,
    planar: PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
    layer: usize,
) -> AlgebraicFieldContext<C, ArrayJet<C, D>>
where
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let planar = AlgebraicPlanarInput::new(
        ArrayJet::constant(planar.vacuum_wavenumber().clone()),
        ArrayJet::constant(planar.parallel_wavenumber().clone()),
        planar.polarisation(),
    );

    AlgebraicFieldContext::evaluate(
        stack,
        planar,
        |material, planar| IsotropicLayerQuantities::real_axis(material, planar),
        move |index, value, source| {
            let values = source.value().mapv(|_| C::from_real(value));
            if index == layer {
                ArrayJet::variable(values)
            } else {
                ArrayJet::constant(values)
            }
        },
    )
}

pub(super) fn kx_first_context<M, C, D>(
    stack: &Stack<M, C::RealField>,
    planar: PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
) -> AlgebraicFieldContext<C, ArrayJetFirst<C, D>>
where
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let planar = AlgebraicPlanarInput::new(
        ArrayJetFirst::constant(planar.vacuum_wavenumber().clone()),
        ArrayJetFirst::variable(planar.parallel_wavenumber().clone()),
        planar.polarisation(),
    );

    AlgebraicFieldContext::evaluate(
        stack,
        planar,
        |material, planar| IsotropicLayerQuantities::real_axis(material, planar),
        |_, value, source| {
            let values = source.value().mapv(|_| C::from_real(value));
            ArrayJetFirst::constant(values)
        },
    )
}

pub(super) fn kx_second_context<M, C, D>(
    stack: &Stack<M, C::RealField>,
    planar: PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
) -> AlgebraicFieldContext<C, ArrayJet<C, D>>
where
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let planar = AlgebraicPlanarInput::new(
        ArrayJet::constant(planar.vacuum_wavenumber().clone()),
        ArrayJet::variable(planar.parallel_wavenumber().clone()),
        planar.polarisation(),
    );

    AlgebraicFieldContext::evaluate(
        stack,
        planar,
        |material, planar| IsotropicLayerQuantities::real_axis(material, planar),
        |_, value, source| {
            let values = source.value().mapv(|_| C::from_real(value));
            ArrayJet::constant(values)
        },
    )
}

pub(super) fn k0_first_context<M, C, D>(
    stack: &Stack<M, C::RealField>,
    planar: PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
) -> AlgebraicFieldContext<C, ArrayJetFirst<C, D>>
where
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let planar = AlgebraicPlanarInput::new(
        ArrayJetFirst::variable(planar.vacuum_wavenumber().clone()),
        ArrayJetFirst::constant(planar.parallel_wavenumber().clone()),
        planar.polarisation(),
    );

    AlgebraicFieldContext::evaluate(
        stack,
        planar,
        |material, planar| IsotropicLayerQuantities::real_axis(material, planar),
        |_, value, source| {
            let values = source.value().mapv(|_| C::from_real(value));
            ArrayJetFirst::constant(values)
        },
    )
}

pub(super) fn k0_second_context<M, C, D>(
    stack: &Stack<M, C::RealField>,
    planar: PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
) -> AlgebraicFieldContext<C, ArrayJet<C, D>>
where
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let planar = AlgebraicPlanarInput::new(
        ArrayJet::variable(planar.vacuum_wavenumber().clone()),
        ArrayJet::constant(planar.parallel_wavenumber().clone()),
        planar.polarisation(),
    );

    AlgebraicFieldContext::evaluate(
        stack,
        planar,
        |material, planar| IsotropicLayerQuantities::real_axis(material, planar),
        |_, value, source| {
            let values = source.value().mapv(|_| C::from_real(value));
            ArrayJet::constant(values)
        },
    )
}

pub(super) fn spectral_hessian_context<M, C, D>(
    stack: &Stack<M, C::RealField>,
    planar: PlanarInput<ArrayBase<OwnedRepr<C>, D>>,
) -> AlgebraicFieldContext<C, ArraySpectralJet<C, D>>
where
    M: EvaluateDifferentiableMaterial<C, Real = C::RealField>,
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let planar = AlgebraicPlanarInput::new(
        ArraySpectralJet::vacuum_wavenumber(planar.vacuum_wavenumber().clone()),
        ArraySpectralJet::parallel_wavenumber(planar.parallel_wavenumber().clone()),
        planar.polarisation(),
    );

    AlgebraicFieldContext::evaluate(
        stack,
        planar,
        |material, planar| IsotropicLayerQuantities::real_axis(material, planar),
        |_, value, source| {
            let values = source.value().mapv(|_| C::from_real(value));
            ArraySpectralJet::constant(values)
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
