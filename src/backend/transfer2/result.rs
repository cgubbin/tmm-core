use crate::{
    ComplexScalar,
    backend::{
        DerivativeVariable, MatrixDerivatives, MatrixEvaluation,
        derivative::ChainRule,
        input::IncidentSide,
        jet::ArrayJet,
        transfer2::{BoundaryMode, BoundaryModeDerivatives, FieldState, Matrix2},
    },
};
use ndarray::{ArrayBase, Dimension, OwnedRepr, ScalarOperand};
use num_traits::One;

// impl<C, D> MatrixEvaluation<Matrix2<C, D>>
// where
//     C: ComplexScalar,
//     D: Dimension,
// {
//     fn jets(
//         &self,
//     ) -> (
//         ArrayJet<C, D>,
//         ArrayJet<C, D>,
//         ArrayJet<C, D>,
//         ArrayJet<C, D>,
//     )
//     where
//         C: ComplexScalar,
//         D: Dimension,
//     {
//         let matrix = self.matrix();

//         match self.derivatives() {
//             None => (
//                 ArrayJet::value_only(matrix.m11().clone()),
//                 ArrayJet::value_only(matrix.m12().clone()),
//                 ArrayJet::value_only(matrix.m21().clone()),
//                 ArrayJet::value_only(matrix.m22().clone()),
//             ),

//             Some(derivatives) => {
//                 let first = derivatives.first();

//                 match derivatives.second() {
//                     None => (
//                         ArrayJet::with_first(matrix.m11().clone(), first.m11().clone()),
//                         ArrayJet::with_first(matrix.m12().clone(), first.m12().clone()),
//                         ArrayJet::with_first(matrix.m21().clone(), first.m21().clone()),
//                         ArrayJet::with_first(matrix.m22().clone(), first.m22().clone()),
//                     ),

//                     Some(second) => (
//                         ArrayJet::with_second(
//                             matrix.m11().clone(),
//                             first.m11().clone(),
//                             second.m11().clone(),
//                         ),
//                         ArrayJet::with_second(
//                             matrix.m12().clone(),
//                             first.m12().clone(),
//                             second.m12().clone(),
//                         ),
//                         ArrayJet::with_second(
//                             matrix.m21().clone(),
//                             first.m21().clone(),
//                             second.m21().clone(),
//                         ),
//                         ArrayJet::with_second(
//                             matrix.m22().clone(),
//                             first.m22().clone(),
//                             second.m22().clone(),
//                         ),
//                     ),
//                 }
//             }
//         }
//     }

//     pub(super) fn amplitude_jets(
//         &self,
//         left_admittance: &ArrayJet<C, D>,
//         right_admittance: &ArrayJet<C, D>,
//         incident_side: IncidentSide,
//     ) -> (ArrayJet<C, D>, ArrayJet<C, D>)
//     where
//         C: ComplexScalar,
//         D: Dimension,
//     {
//         let (a, b, c, d) = self.jets();

//         let two = ArrayJet::constant_like(self.matrix().m11(), C::one() + C::one());

//         let b_yr = b.multiply(right_admittance);
//         let d_yr = d.multiply(right_admittance);

//         let u = a.subtract(&b_yr);
//         let v = c.subtract(&d_yr);

//         let denominator = left_admittance.multiply(&u).subtract(&v);

//         match incident_side {
//             IncidentSide::Left => {
//                 let reflection = left_admittance.multiply(&u).add(&v).divide(&denominator);

//                 let transmission = two.multiply(left_admittance).divide(&denominator);

//                 (reflection, transmission)
//             }

//             IncidentSide::Right => {
//                 let p = a.add(&b_yr);
//                 let q = c.add(&d_yr);

//                 let reflection = q
//                     .subtract(&left_admittance.multiply(&p))
//                     .divide(&denominator);

//                 let determinant = a.multiply(&d).subtract(&b.multiply(&c));

//                 let transmission = two
//                     .multiply(right_admittance)
//                     .multiply(&determinant)
//                     .divide(&denominator);

//                 (reflection, transmission)
//             }
//         }
//     }
// }

impl<C, D> MatrixEvaluation<Matrix2<C, D>>
where
    C: ComplexScalar,
    D: Dimension,
{
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
    /// Complex reflection amplitude, assuming the standard convention:
    ///
    /// incident-side field = incident + reflected
    /// transmission-side field = transmitted only
    pub fn reflection_amplitude(&self, incident_side: IncidentSide) -> ArrayBase<OwnedRepr<C>, D> {
        match incident_side {
            IncidentSide::Left => -self.matrix().m21().clone() / self.matrix().m22().clone(),
            IncidentSide::Right => todo!(),
        }
    }

    pub fn transmission_amplitude(
        &self,
        incident_side: IncidentSide,
    ) -> ArrayBase<OwnedRepr<C>, D> {
        let one = self.matrix().m22().mapv(|_| C::one());
        match incident_side {
            IncidentSide::Left => one / self.matrix().m22().clone(),
            IncidentSide::Right => one / self.matrix().m11().clone(),
        }
    }

    pub fn reflection_amplitude_derivative(
        &self,
        incident_side: IncidentSide,
    ) -> Option<ArrayBase<OwnedRepr<C>, D>> {
        let dm = self.derivatives()?.first();

        match incident_side {
            IncidentSide::Left => {
                let m21 = self.matrix().m21().clone();
                let m22 = self.matrix().m22().clone();

                let dm21 = dm.m21().clone();
                let dm22 = dm.m22().clone();

                Some(-(dm21 * m22.clone() - m21 * dm22) / m22.mapv(|x| x * x))
            }
            IncidentSide::Right => {
                let m12 = self.matrix().m12().clone();
                let m11 = self.matrix().m11().clone();

                let dm12 = dm.m12().clone();
                let dm11 = dm.m11().clone();

                Some(-(dm12 * m11.clone() - m12 * dm11) / m11.mapv(|x| x * x))
            }
        }
    }

    pub fn reflection_amplitude_second_derivative(
        &self,
        incident_side: IncidentSide,
    ) -> Option<ArrayBase<OwnedRepr<C>, D>> {
        let dm = self.derivatives()?.first();
        let ddm = self.derivatives()?.second()?;

        match incident_side {
            IncidentSide::Left => {
                let m21 = self.matrix().m21().clone();
                let m22 = self.matrix().m22().clone();

                let dm21 = dm.m21().clone();
                let dm22 = dm.m22().clone();

                let ddm21 = ddm.m21().clone();
                let ddm22 = ddm.m22().clone();

                todo!()
            }
            IncidentSide::Right => {
                let m12 = self.matrix().m12().clone();
                let m11 = self.matrix().m11().clone();

                let dm12 = dm.m12().clone();
                let dm11 = dm.m11().clone();

                let ddm12 = ddm.m12().clone();
                let ddm11 = ddm.m11().clone();

                todo!()
            }
        }
    }

    pub fn transmission_amplitude_derivative(
        &self,
        incident_side: IncidentSide,
    ) -> Option<ArrayBase<OwnedRepr<C>, D>> {
        let dm = self.derivatives()?.first();

        match incident_side {
            IncidentSide::Left => {
                let m22 = self.matrix().m22().clone();
                let dm22 = dm.m22().clone();
                Some(-dm22 / m22.mapv(|x| x * x))
            }
            IncidentSide::Right => {
                let m11 = self.matrix().m11().clone();
                let dm11 = dm.m11().clone();
                Some(-dm11 / m11.mapv(|x| x * x))
            }
        }
    }

    pub fn transmission_amplitude_second_derivative(
        &self,
        incident_side: IncidentSide,
    ) -> Option<ArrayBase<OwnedRepr<C>, D>> {
        let dm = self.derivatives()?.first();
        let ddm = self.derivatives()?.second()?;

        match incident_side {
            IncidentSide::Left => {
                let m22 = self.matrix().m22().clone();
                let dm22 = dm.m22().clone();
                let ddm22 = ddm.m22().clone();
                Some(
                    -ddm22 / m22.mapv(|x| x * x)
                        + dm22.mapv(|x| x * x + x * x) / m22.mapv(|x| x * x * x),
                )
            }
            IncidentSide::Right => {
                let m11 = self.matrix().m11().clone();
                let dm11 = dm.m11().clone();
                let ddm11 = ddm.m11().clone();
                Some(
                    -ddm11 / m11.mapv(|x| x * x)
                        + dm11.mapv(|x| x * x + x * x) / m11.mapv(|x| x * x * x),
                )
            }
        }
    }
}
