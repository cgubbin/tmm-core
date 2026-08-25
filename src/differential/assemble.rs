//! Assembly of public differential responses.
//!
//! Derivative extraction and response assembly are deliberately separate
//! stages.
//!
//! Internal jet-valued quantities are first converted into coordinate-free
//! derivative-parts containers by the policies in [`crate::derivative_parts`].
//! This module then combines those parts with a typed derivative mapping to
//! construct a caller-facing [`DifferentialResponse`].
//!
//! The typed mapping determines the physical meaning of each derivative axis:
//!
//! - [`ValueMapping`] carries no derivative axes;
//! - [`DirectionalMapping`] identifies one caller-facing parameter;
//! - [`BivariateMapping`] identifies two ordered caller-facing parameters.
//!
//! Only valid combinations of parts and mappings implement
//! [`AssembleDifferentialResponse`]. For example,
//! [`DirectionalFirstParts`] can be assembled with a
//! [`DirectionalMapping`], but not with a [`BivariateMapping`].
//!
//! [`IntoDifferentialResponse`] provides the complete generic pipeline:
//!
//! ```text
//! internal derivative-bearing value
//!     -> derivative-parts extraction
//!     -> typed response assembly
//!     -> DifferentialResponse
//! ```
//!
//! This module does not depend on input compilation or on concrete jet types.

use crate::{
    derivative_parts::{
        BivariateFirstParts, BivariateSecondParts, DerivativePartsPolicy, DirectionalFirstParts,
        DirectionalSecondParts, IntoDerivativeParts, ValuePart,
    },
    differential::{
        BivariateFirst, BivariateGradient, BivariateHessian, BivariateSecond, DifferentialResponse,
        DirectionalFirst, DirectionalSecond,
    },
    parameter::{BivariateMapping, DirectionalMapping, ValueMapping},
};

/// Assembles coordinate-free derivative parts into a public differential
/// response using typed parameter metadata.
///
/// Implementations exist only for compatible pairs of derivative-parts and
/// mapping types. This makes mismatched derivative cardinalities
/// unrepresentable at the assembly boundary.
#[doc(hidden)]
pub trait AssembleDifferentialResponse<M>: Sized {
    /// Public response produced by the assembly.
    type Output;

    /// Attach the parameter metadata in `mapping` to `self`.
    fn assemble(self, mapping: &M) -> Self::Output;
}

/// Converts an internal derivative-bearing quantity directly into a public
/// differential response.
///
/// This combines derivative-parts extraction through `P` with response
/// assembly through `M`.
#[doc(hidden)]
pub trait IntoDifferentialResponse<P, M>: Sized
where
    P: DerivativePartsPolicy<Self>,
{
    /// Public response produced by extraction and assembly.
    type Output;

    /// Extract derivative parts with `policy`, then attach the parameter
    /// metadata in `mapping`.
    fn into_differential_response(self, policy: &P, mapping: &M) -> Self::Output;
}

impl<T> AssembleDifferentialResponse<ValueMapping> for ValuePart<T> {
    type Output = T;

    fn assemble(self, _mapping: &ValueMapping) -> Self::Output {
        self.into_inner()
    }
}

impl<T> AssembleDifferentialResponse<DirectionalMapping> for DirectionalFirstParts<T> {
    type Output = DifferentialResponse<T, DirectionalFirst<T>>;

    fn assemble(self, mapping: &DirectionalMapping) -> Self::Output {
        let (value, first) = self.into_parts();
        DifferentialResponse::new(value, DirectionalFirst::new(mapping.parameter(), first))
    }
}

impl<T> AssembleDifferentialResponse<DirectionalMapping> for DirectionalSecondParts<T> {
    type Output = DifferentialResponse<T, DirectionalSecond<T>>;

    fn assemble(self, mapping: &DirectionalMapping) -> Self::Output {
        let (value, first, second) = self.into_parts();
        DifferentialResponse::new(
            value,
            DirectionalSecond::new(mapping.parameter(), first, second),
        )
    }
}

impl<T> AssembleDifferentialResponse<BivariateMapping> for BivariateFirstParts<T> {
    type Output = DifferentialResponse<T, BivariateFirst<T>>;

    fn assemble(self, mapping: &BivariateMapping) -> Self::Output {
        let (value, axis0, axis1) = self.into_parts();

        let (parameter0, parameter1) = mapping.parameters();
        DifferentialResponse::new(
            value,
            BivariateFirst::new([parameter0, parameter1], axis0, axis1),
        )
    }
}

impl<T> AssembleDifferentialResponse<BivariateMapping> for BivariateSecondParts<T> {
    type Output = DifferentialResponse<T, BivariateSecond<T>>;

    fn assemble(self, mapping: &BivariateMapping) -> Self::Output {
        let (value, axis0, axis1, axis0_axis0, axis0_axis1, axis1_axis1) = self.into_parts();

        let gradient = BivariateGradient::new(axis0, axis1);
        let hessian = BivariateHessian::new(axis0_axis0, axis0_axis1, axis1_axis1);

        let (parameter0, parameter1) = mapping.parameters();

        DifferentialResponse::new(
            value,
            BivariateSecond::new([parameter0, parameter1], gradient, hessian),
        )
    }
}

impl<T, P, M> IntoDifferentialResponse<P, M> for T
where
    P: DerivativePartsPolicy<T>,
    P::Output: AssembleDifferentialResponse<M>,
{
    type Output = <P::Output as AssembleDifferentialResponse<M>>::Output;

    fn into_differential_response(self, policy: &P, mapping: &M) -> Self::Output {
        self.into_derivative_parts(policy).assemble(mapping)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::{
        algebra::{Jet0, Jet1, Jet2, JetBivariate1, JetBivariate2},
        derivative_parts::{
            FirstBivariate, FirstDirectional, SecondBivariate, SecondDirectional, ValueOnly,
        },
        parameter::{DerivativeMapping, FiniteLayerIndex, Parameter},
    };

    fn spectral() -> Parameter {
        Parameter::Spectral
    }

    fn in_plane() -> Parameter {
        Parameter::InPlane
    }

    fn thickness(layer: usize) -> Parameter {
        Parameter::LayerThickness(FiniteLayerIndex::new(layer))
    }

    fn value_mapping() -> ValueMapping {
        ValueMapping
    }

    fn directional_mapping(parameter: Parameter) -> DirectionalMapping {
        let mapping = DerivativeMapping::new([parameter]).unwrap();

        DirectionalMapping::try_from_mapping(&mapping).unwrap()
    }

    fn bivariate_mapping(first: Parameter, second: Parameter) -> BivariateMapping {
        let mapping = DerivativeMapping::new([first, second]).unwrap();

        BivariateMapping::try_from_mapping(&mapping).unwrap()
    }

    #[test]
    fn value_parts_assemble_without_derivatives() {
        let parts = ValuePart::new(10);
        let mapping = value_mapping();

        let response = parts.assemble(&mapping);

        assert_eq!(response, 10);
    }

    #[test]
    fn directional_first_assembly_attaches_parameter() {
        let parts = DirectionalFirstParts::new(10, 20);

        let mapping = directional_mapping(spectral());

        let response = parts.assemble(&mapping);

        assert_eq!(response.value(), &10);
        assert_eq!(response.derivatives().parameter(), spectral(),);
        assert_eq!(response.derivatives().first(), &20,);
    }

    #[test]
    fn directional_second_assembly_preserves_component_order() {
        let parts = DirectionalSecondParts::new(10, 20, 30);

        let mapping = directional_mapping(thickness(3));

        let response = parts.assemble(&mapping);
        let derivatives = response.derivatives();

        assert_eq!(derivatives.parameter(), thickness(3),);
        assert_eq!(derivatives.first(), &20);
        assert_eq!(derivatives.second(), &30);
    }

    #[test]
    fn bivariate_first_assembly_preserves_axis_order() {
        let parts = BivariateFirstParts::new(10, 20, 30);

        let mapping = bivariate_mapping(spectral(), in_plane());

        let response = parts.assemble(&mapping);
        let derivatives = response.derivatives();

        assert_eq!(derivatives.parameters(), [spectral(), in_plane()],);
        assert_eq!(derivatives.axis0(), &20);
        assert_eq!(derivatives.axis1(), &30);
    }

    #[test]
    fn bivariate_second_assembly_preserves_gradient_and_hessian_order() {
        let parts = BivariateSecondParts::new(10, 20, 30, 40, 50, 60);

        let mapping = bivariate_mapping(thickness(2), spectral());

        let response = parts.assemble(&mapping);
        let derivatives = response.derivatives();

        assert_eq!(derivatives.parameters(), [thickness(2), spectral()],);

        assert_eq!(derivatives.gradient().axis0(), &20,);
        assert_eq!(derivatives.gradient().axis1(), &30,);

        assert_eq!(derivatives.hessian().axis0_axis0(), &40,);
        assert_eq!(derivatives.hessian().axis0_axis1(), &50,);
        assert_eq!(derivatives.hessian().axis1_axis1(), &60,);
    }

    #[test]
    fn value_pipeline_extracts_and_assembles() {
        let jet = Jet0::<_, ()>::new(10);
        let mapping = value_mapping();

        let response = jet.into_differential_response(&ValueOnly, &mapping);

        assert_eq!(response, 10);
    }

    #[test]
    fn directional_first_pipeline_extracts_and_assembles() {
        let jet = Jet1::<_, ()>::from_parts(10, 20);

        let mapping = directional_mapping(spectral());

        let response = jet.into_differential_response(&FirstDirectional, &mapping);

        assert_eq!(response.value(), &10);
        assert_eq!(response.derivatives().parameter(), spectral(),);
        assert_eq!(response.derivatives().first(), &20,);
    }

    #[test]
    fn directional_second_pipeline_extracts_and_assembles() {
        let jet = Jet2::<_, ()>::from_parts(10, 20, 30);

        let mapping = directional_mapping(in_plane());

        let response = jet.into_differential_response(&SecondDirectional, &mapping);

        assert_eq!(response.value(), &10);
        assert_eq!(response.derivatives().parameter(), in_plane(),);
        assert_eq!(response.derivatives().first(), &20,);
        assert_eq!(response.derivatives().second(), &30,);
    }

    #[test]
    fn bivariate_first_pipeline_extracts_and_assembles() {
        let jet = JetBivariate1::<_, ()>::from_parts(10, BivariateGradient::new(20, 30));

        let mapping = bivariate_mapping(spectral(), thickness(1));

        let response = jet.into_differential_response(&FirstBivariate, &mapping);

        let derivatives = response.derivatives();

        assert_eq!(derivatives.parameters(), [spectral(), thickness(1)],);
        assert_eq!(derivatives.axis0(), &20);
        assert_eq!(derivatives.axis1(), &30);
    }

    #[test]
    fn bivariate_second_pipeline_extracts_and_assembles() {
        let jet = JetBivariate2::<_, ()>::from_parts(
            10,
            BivariateGradient::new(20, 30),
            BivariateHessian::new(40, 50, 60),
        );

        let mapping = bivariate_mapping(in_plane(), spectral());

        let response = jet.into_differential_response(&SecondBivariate, &mapping);

        let derivatives = response.derivatives();

        assert_eq!(derivatives.parameters(), [in_plane(), spectral()],);

        assert_eq!(derivatives.gradient().axis0(), &20,);
        assert_eq!(derivatives.gradient().axis1(), &30,);

        assert_eq!(derivatives.hessian().axis0_axis0(), &40,);
        assert_eq!(derivatives.hessian().axis0_axis1(), &50,);
        assert_eq!(derivatives.hessian().axis1_axis1(), &60,);
    }

    #[test]
    fn assembly_supports_non_copy_values() {
        let parts = DirectionalFirstParts::new(String::from("value"), String::from("first"));

        let mapping = directional_mapping(spectral());

        let response = parts.assemble(&mapping);

        assert_eq!(response.value(), "value");
        assert_eq!(response.derivatives().first(), "first",);
    }
}
