use std::fmt::Debug;

use crate::{
    algebra::{Jet0, Jet1, Jet2, JetBivariate1, JetBivariate2},
    derivative_parts::{
        FirstBivariate, FirstDirectional, SecondBivariate, SecondDirectional, ValueOnly,
    },
    parameter::{
        BivariateMapping, DerivativeMapping, DerivativeMappingError, DirectionalMapping,
        ValueMapping,
    },
};

mod sealed {
    pub trait Sealed {}
}

/// Associates a jet algebra with its crystallisation policy and
/// caller-facing derivative mapping.
///
/// This trait is sealed and implemented only by Lamina's supported jet
/// families. It is public solely because it appears in bounds on public
/// evaluator APIs.
#[doc(hidden)]
pub trait JetMapping: sealed::Sealed {
    type Policy: Default;
    type Mapping: Debug + PartialEq;

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

#[cfg(test)]
mod tests {
    use super::*;

    use ndarray::Ix0;
    use num_complex::Complex64;

    use crate::{
        algebra::{
            ArrayJet0, ArrayJet1, ArrayJet2, ArrayJetBivariate1, ArrayJetBivariate2, RealParameter,
        },
        parameter::{DerivativeMapping, Parameter, ValueMapping},
    };

    type C = Complex64;

    type Value = ArrayJet0<C, Ix0, RealParameter>;

    type First = ArrayJet1<C, Ix0, RealParameter>;

    type Second = ArrayJet2<C, Ix0, RealParameter>;

    type BivariateFirst = ArrayJetBivariate1<C, Ix0, RealParameter>;

    type BivariateSecond = ArrayJetBivariate2<C, Ix0, RealParameter>;

    #[test]
    fn value_jet_compiles_empty_mapping() {
        let mapping = DerivativeMapping::none();

        let compiled = <Value as JetMapping>::compile_mapping(&mapping).unwrap();

        assert_eq!(compiled, ValueMapping);
    }

    #[test]
    fn directional_jets_preserve_parameter() {
        let mapping = DerivativeMapping::new([Parameter::Spectral]).unwrap();

        let first = <First as JetMapping>::compile_mapping(&mapping).unwrap();

        let second = <Second as JetMapping>::compile_mapping(&mapping).unwrap();

        assert_eq!(first.parameter(), Parameter::Spectral,);

        assert_eq!(second.parameter(), Parameter::Spectral,);
    }

    #[test]
    fn bivariate_jets_preserve_axis_order() {
        let mapping = DerivativeMapping::new([Parameter::InPlane, Parameter::Spectral]).unwrap();

        let first = <BivariateFirst as JetMapping>::compile_mapping(&mapping).unwrap();

        let second = <BivariateSecond as JetMapping>::compile_mapping(&mapping).unwrap();

        assert_eq!(
            first.parameters(),
            (Parameter::InPlane, Parameter::Spectral,),
        );

        assert_eq!(
            second.parameters(),
            (Parameter::InPlane, Parameter::Spectral,),
        );
    }

    #[test]
    fn value_jet_rejects_directional_mapping() {
        let mapping = DerivativeMapping::new([Parameter::Spectral]).unwrap();

        assert!(<Value as JetMapping>::compile_mapping(&mapping,).is_err());
    }

    #[test]
    fn directional_jet_rejects_bivariate_mapping() {
        let mapping = DerivativeMapping::new([Parameter::Spectral, Parameter::InPlane]).unwrap();

        assert!(<First as JetMapping>::compile_mapping(&mapping,).is_err());
    }

    #[test]
    fn bivariate_jet_rejects_directional_mapping() {
        let mapping = DerivativeMapping::new([Parameter::Spectral]).unwrap();

        assert!(<BivariateFirst as JetMapping>::compile_mapping(&mapping,).is_err());
    }
}
