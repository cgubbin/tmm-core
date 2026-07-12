use crate::{
    ComplexScalar,
    backend::{
        DerivativeVariable, MatrixDerivatives, MatrixEvaluation,
        derivative::ChainRule,
        transfer2::{BoundaryMode, BoundaryModeDerivatives, FieldState, Matrix2},
    },
};
use ndarray::{ArrayBase, Dimension, OwnedRepr, ScalarOperand};
use num_traits::One;

impl<C, D> MatrixEvaluation<Matrix2<C, D>>
where
    C: ComplexScalar,
    D: Dimension,
{
    /// Complex reflection amplitude, assuming the standard convention:
    ///
    /// incident-side field = incident + reflected
    /// transmission-side field = transmitted only
    pub fn reflection_amplitude(&self) -> ArrayBase<OwnedRepr<C>, D> {
        -self.matrix().m21().clone() / self.matrix().m22().clone()
    }

    pub fn transmission_amplitude(&self) -> ArrayBase<OwnedRepr<C>, D> {
        let one = self.matrix().m22().mapv(|_| C::one());
        one / self.matrix().m22().clone()
    }

    pub fn reflectance(&self) -> ArrayBase<OwnedRepr<C::RealField>, D> {
        self.reflection_amplitude().mapv(|r| r.modulus_squared())
    }

    pub fn transmittance_unscaled(&self) -> ArrayBase<OwnedRepr<C::RealField>, D> {
        self.transmission_amplitude().mapv(|t| t.modulus_squared())
    }

    pub fn outgoing_state(&self, boundary: &BoundaryMode<C, D>) -> FieldState<C, D> {
        self.matrix().apply_state(&boundary.outgoing_state())
    }

    pub fn outgoing_residual(
        &self,
        incident: &BoundaryMode<C, D>,
        transmission: &BoundaryMode<C, D>,
    ) -> ArrayBase<OwnedRepr<C>, D> {
        let state = self.outgoing_state(transmission);

        state.derivative - state.value * incident.gamma.view() / incident.factor.view()
    }

    pub fn outgoing_residual_derivative(
        &self,
        incident: &BoundaryMode<C, D>,
        incident_derivatives: &BoundaryModeDerivatives<C, D>,
        transmission: &BoundaryMode<C, D>,
        transmission_derivatives: &BoundaryModeDerivatives<C, D>,
    ) -> Option<ArrayBase<OwnedRepr<C>, D>> {
        let dm = self.derivatives.as_ref()?.first();

        let s0 = transmission.outgoing_state();
        let ds0 = transmission.outgoing_state_derivative(transmission_derivatives);

        let state = self.matrix().apply_state(&s0);
        let dstate = dm.apply_state(&s0).add(&self.matrix().apply_state(&ds0));

        Some(
            dstate.derivative
                - dstate.value * incident.gamma.view() / incident.factor.view()
                - state.value.clone() * incident_derivatives.gamma_first.view()
                    / incident.factor.view()
                + state.value * incident.gamma.view() * incident_derivatives.factor_first.view()
                    / incident.factor.view()
                    / incident.factor.view(),
        )
    }

    pub fn outgoing_residual_second_derivative(
        &self,
        incident: &BoundaryMode<C, D>,
        incident_derivatives: &BoundaryModeDerivatives<C, D>,
        transmission: &BoundaryMode<C, D>,
        transmission_derivatives: &BoundaryModeDerivatives<C, D>,
    ) -> Option<ArrayBase<OwnedRepr<C>, D>>
    where
        C: ScalarOperand,
    {
        let derivatives = self.derivatives()?;
        let dm = derivatives.first();
        let ddm = derivatives.second()?;

        let s0 = transmission.outgoing_state();
        let ds0 = transmission.outgoing_state_derivative(transmission_derivatives);
        let dds0 = transmission.outgoing_state_second_derivative(transmission_derivatives)?;

        let state = self.matrix().apply_state(&s0);

        let dstate = dm.apply_state(&s0).add(&self.matrix().apply_state(&ds0));

        let ddstate = ddm
            .apply_state(&s0)
            .add(&dm.apply_state(&ds0).scale(C::one() + C::one()))
            .add(&self.matrix().apply_state(&dds0));

        let g = incident.gamma.clone();
        let f = incident.factor.clone();

        let dg = incident_derivatives.gamma_first.clone();
        let df = incident_derivatives.factor_first.clone();

        let incident_second = incident_derivatives.second()?;
        let ddg = incident_second.gamma.clone();
        let ddf = incident_second.factor.clone();

        let two = C::one() + C::one();

        let f2 = f.mapv(|x| x * x);
        let f3 = f.mapv(|x| x * x * x);

        let q = g.clone() / f.view();

        let dq = dg.clone() / f.view() - g.clone() * df.view() / f2.view();

        let ddq = ddg.clone() / f.view()
            - dg.clone() * df.view() * two / f2.view()
            - g.clone() * ddf / f2.view()
            + g.clone() * df.mapv(|x| x * x) * two / f3;

        Some(
            ddstate.derivative
                - ddstate.value * q
                - dstate.value * dq.mapv(|x| x * two)
                - state.value * ddq,
        )
    }

    pub fn determinant(&self) -> ArrayBase<OwnedRepr<C>, D> {
        self.matrix().determinant()
    }

    pub fn determinant_derivative(&self) -> Option<ArrayBase<OwnedRepr<C>, D>>
    where
        C: ScalarOperand,
    {
        let dm = self.derivatives.as_ref()?.first();

        Some(
            dm.m11().clone() * self.matrix().m22().view()
                + self.matrix().m11().clone() * dm.m22().view()
                - dm.m12().clone() * self.matrix().m21().view()
                - self.matrix().m12().clone() * dm.m21().view(),
        )
    }

    pub fn determinant_second_derivative(&self) -> Option<ArrayBase<OwnedRepr<C>, D>>
    where
        C: ScalarOperand,
    {
        let dm = self.derivatives.as_ref()?.first();
        let ddm = self.derivatives.as_ref()?.second()?;

        let two = C::one() + C::one();

        Some(
            ddm.m11().clone() * self.matrix().m22().view()
                + dm.m11().mapv(|each| each * two) * dm.m22().view()
                + self.matrix().m11().clone() * ddm.m22().view()
                - ddm.m12().clone() * self.matrix().m21().view()
                - dm.m12().mapv(|each| each * two) * dm.m21().view()
                - self.matrix().m12().clone() * ddm.m21().view(),
        )
    }

    pub(crate) fn chain_rule(
        mut self,
        variable: DerivativeVariable,
        chain_rule: ChainRule<ArrayBase<OwnedRepr<C>, D>>,
    ) -> Self {
        let Some(derivatives) = self.derivatives.take() else {
            return self;
        };

        let primitive_first = derivatives.first().clone();

        let first_derivative = derivatives.first().scale_by_array(&chain_rule.first);

        let second_derivative = derivatives.second().map(|primitive_second| {
            let first_squared = chain_rule.first.mapv(|x| x * x);

            &primitive_second.scale_by_array(&first_squared)
                + &primitive_first.scale_by_array(&chain_rule.second)
        });

        let mut transformed = MatrixDerivatives::new(variable, first_derivative);

        if let Some(second_derivative) = second_derivative {
            transformed = transformed.with_second(second_derivative);
        }

        self.derivatives = Some(transformed);
        self
    }
}

impl<C, D> MatrixEvaluation<Matrix2<C, D>>
where
    C: ComplexScalar,
    C::RealField: One,
    D: Dimension,
{
    pub fn reflection_amplitude_derivative(&self) -> Option<ArrayBase<OwnedRepr<C>, D>> {
        let dm = self.derivatives()?.first();

        let m21 = self.matrix().m21().clone();
        let m22 = self.matrix().m22().clone();

        let dm21 = dm.m21().clone();
        let dm22 = dm.m22().clone();

        Some(-(dm21 * m22.clone() - m21 * dm22) / m22.mapv(|x| x * x))
    }

    pub fn transmission_amplitude_derivative(&self) -> Option<ArrayBase<OwnedRepr<C>, D>> {
        let dm = self.derivatives()?.first();

        let m22 = self.matrix().m22().clone();
        let dm22 = dm.m22().clone();

        Some(-dm22 / m22.mapv(|x| x * x))
    }

    pub fn reflectance_derivative(&self) -> Option<ArrayBase<OwnedRepr<C::RealField>, D>> {
        let r = self.reflection_amplitude();
        let dr = self.reflection_amplitude_derivative()?;

        Some((r.mapv(|x| x.conjugate()) * dr).mapv(|x| {
            let two = C::RealField::one() + C::RealField::one();
            two * x.real()
        }))
    }
}
