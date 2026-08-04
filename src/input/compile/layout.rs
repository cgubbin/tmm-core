use crate::{
    algebra::{Jet0, Jet1, Jet2, JetBivariate1, JetBivariate2},
    derivative_parts::{
        FirstBivariate, FirstDirectional, SecondBivariate, SecondDirectional, ValueOnly,
    },
    evaluate::PairMappingCompatibility,
    parameter::{
        BivariateMapping, DerivativeMapping, DerivativeMappingError, DirectionalMapping,
        ValueMapping,
    },
};

mod sealed {
    pub trait Sealed {}
}

#[doc(hidden)]
pub trait JetMapping: sealed::Sealed {
    type Policy: Default;
    type Mapping;

    fn compile_mapping(
        mapping: &DerivativeMapping,
    ) -> Result<Self::Mapping, DerivativeMappingError>;
}

impl<I, P> sealed::Sealed for Jet0<I, P> {}

impl<I, P> JetMapping for Jet0<I, P> {
    type Policy = ValueOnly;
    type Mapping = ValueMapping;

    fn compile_mapping(
        mapping: &DerivativeMapping,
    ) -> Result<Self::Mapping, DerivativeMappingError> {
        ValueMapping::try_from_mapping(mapping)
    }
}

impl<I, P> sealed::Sealed for Jet1<I, P> {}

impl<I, P> JetMapping for Jet1<I, P> {
    type Policy = FirstDirectional;
    type Mapping = DirectionalMapping;

    fn compile_mapping(
        mapping: &DerivativeMapping,
    ) -> Result<Self::Mapping, DerivativeMappingError> {
        DirectionalMapping::try_from_mapping(mapping)
    }
}

impl<I, P> sealed::Sealed for Jet2<I, P> {}

impl<I, P> JetMapping for Jet2<I, P> {
    type Policy = SecondDirectional;
    type Mapping = DirectionalMapping;

    fn compile_mapping(
        mapping: &DerivativeMapping,
    ) -> Result<Self::Mapping, DerivativeMappingError> {
        DirectionalMapping::try_from_mapping(mapping)
    }
}

impl<I, P> sealed::Sealed for JetBivariate1<I, P> {}

impl<I, P> JetMapping for JetBivariate1<I, P> {
    type Policy = FirstBivariate;
    type Mapping = BivariateMapping;

    fn compile_mapping(
        mapping: &DerivativeMapping,
    ) -> Result<Self::Mapping, DerivativeMappingError> {
        BivariateMapping::try_from_mapping(mapping)
    }
}

impl<I, P> sealed::Sealed for JetBivariate2<I, P> {}

impl<I, P> JetMapping for JetBivariate2<I, P> {
    type Policy = SecondBivariate;
    type Mapping = BivariateMapping;

    fn compile_mapping(
        mapping: &DerivativeMapping,
    ) -> Result<Self::Mapping, DerivativeMappingError> {
        BivariateMapping::try_from_mapping(mapping)
    }
}

impl PairMappingCompatibility for ValueMapping {
    fn pair_mapping_compatible(&self, _other: &Self) -> bool {
        true
    }
}

impl PairMappingCompatibility for DirectionalMapping {
    fn pair_mapping_compatible(&self, other: &Self) -> bool {
        self == other
    }
}

impl PairMappingCompatibility for BivariateMapping {
    fn pair_mapping_compatible(&self, other: &Self) -> bool {
        self == other
    }
}
