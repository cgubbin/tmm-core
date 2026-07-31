use crate::{
    crystallise::{
        BivariateFirstParts, BivariateSecondParts, DirectionalFirstParts, DirectionalSecondParts,
    },
    differential::{
        BivariateFirst, BivariateGradient, BivariateHessian, BivariateSecond, DifferentialResponse,
        DirectionalFirst, DirectionalSecond, NoDerivatives,
    },
    parameter::{BivariateMapping, DirectionalMapping, ValueMapping},
};

use super::ValueParts;

pub trait AssembleDifferentialResponse<A> {
    type Response;

    fn assemble(self, assignment: &A) -> Self::Response;
}

impl<T> AssembleDifferentialResponse<ValueMapping> for ValueParts<T> {
    type Response = DifferentialResponse<T, NoDerivatives>;

    fn assemble(self, _assignment: &ValueMapping) -> Self::Response {
        DifferentialResponse::new(self.into_inner(), NoDerivatives)
    }
}

impl<T> AssembleDifferentialResponse<DirectionalMapping> for DirectionalFirstParts<T> {
    type Response = DifferentialResponse<T, DirectionalFirst<T>>;

    fn assemble(self, assignment: &DirectionalMapping) -> Self::Response {
        let (value, first) = self.into_parts();
        DifferentialResponse::new(
            value,
            DirectionalFirst::new(assignment.parameter().clone(), first),
        )
    }
}

impl<T> AssembleDifferentialResponse<DirectionalMapping> for DirectionalSecondParts<T> {
    type Response = DifferentialResponse<T, DirectionalSecond<T>>;

    fn assemble(self, assignment: &DirectionalMapping) -> Self::Response {
        let (value, first, second) = self.into_parts();
        DifferentialResponse::new(
            value,
            DirectionalSecond::new(assignment.parameter().clone(), first, second),
        )
    }
}

impl<T> AssembleDifferentialResponse<BivariateMapping> for BivariateFirstParts<T> {
    type Response = DifferentialResponse<T, BivariateFirst<T>>;

    fn assemble(self, assignment: &BivariateMapping) -> Self::Response {
        let (value, axis0, axis1) = self.into_parts();

        let (parameter0, parameter1) = assignment.parameters();
        DifferentialResponse::new(
            value,
            BivariateFirst::new([parameter0.clone(), parameter1.clone()], axis0, axis1),
        )
    }
}

impl<T> AssembleDifferentialResponse<BivariateMapping> for BivariateSecondParts<T> {
    type Response = DifferentialResponse<T, BivariateSecond<T>>;

    fn assemble(self, assignment: &BivariateMapping) -> Self::Response {
        let (value, axis0, axis1, axis0_axis0, axis0_axis1, axis1_axis1) = self.into_parts();

        let gradient = BivariateGradient::new(axis0, axis1);
        let hessian = BivariateHessian::new(axis0_axis0, axis0_axis1, axis1_axis1);

        let (parameter0, parameter1) = assignment.parameters();

        DifferentialResponse::new(
            value,
            BivariateSecond::new([parameter0.clone(), parameter1.clone()], gradient, hessian),
        )
    }
}
