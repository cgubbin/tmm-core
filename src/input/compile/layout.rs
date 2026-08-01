use crate::{
    algebra::{Jet0, Jet1, Jet2, JetBivariate1, JetBivariate2},
    derivative_parts::{
        FirstBivariate, FirstDirectional, SecondBivariate, SecondDirectional, ValueOnly,
    },
    parameter::{
        BivariateMapping, DerivativeMapping, DerivativeMappingError, DirectionalMapping, Parameter,
        ValueMapping,
    },
};

pub(crate) trait JetMapping {
    type Policy: Default;
    type Mapping;

    fn compile_mapping(
        mapping: &DerivativeMapping,
    ) -> Result<Self::Mapping, DerivativeMappingError>;
}

impl<I, P> JetMapping for Jet0<I, P> {
    type Policy = ValueOnly;
    type Mapping = ValueMapping;

    fn compile_mapping(
        mapping: &DerivativeMapping,
    ) -> Result<Self::Mapping, DerivativeMappingError> {
        ValueMapping::try_from_mapping(mapping)
    }
}

impl<I, P> JetMapping for Jet1<I, P> {
    type Policy = FirstDirectional;
    type Mapping = DirectionalMapping;

    fn compile_mapping(
        mapping: &DerivativeMapping,
    ) -> Result<Self::Mapping, DerivativeMappingError> {
        DirectionalMapping::try_from_mapping(mapping)
    }
}

impl<I, P> JetMapping for Jet2<I, P> {
    type Policy = SecondDirectional;
    type Mapping = DirectionalMapping;

    fn compile_mapping(
        mapping: &DerivativeMapping,
    ) -> Result<Self::Mapping, DerivativeMappingError> {
        DirectionalMapping::try_from_mapping(mapping)
    }
}

impl<I, P> JetMapping for JetBivariate1<I, P> {
    type Policy = FirstBivariate;
    type Mapping = BivariateMapping;

    fn compile_mapping(
        mapping: &DerivativeMapping,
    ) -> Result<Self::Mapping, DerivativeMappingError> {
        BivariateMapping::try_from_mapping(mapping)
    }
}

impl<I, P> JetMapping for JetBivariate2<I, P> {
    type Policy = SecondBivariate;
    type Mapping = BivariateMapping;

    fn compile_mapping(
        mapping: &DerivativeMapping,
    ) -> Result<Self::Mapping, DerivativeMappingError> {
        BivariateMapping::try_from_mapping(mapping)
    }
}
