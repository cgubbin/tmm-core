use super::{
    Parameter,
    derivative_mapping::{DerivativeMapping, DerivativeMappingError},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ValueMapping;

impl ValueMapping {
    pub(crate) fn try_from_mapping(
        mapping: &DerivativeMapping,
    ) -> Result<Self, DerivativeMappingError> {
        if !mapping.is_empty() {
            return Err(DerivativeMappingError::IncompatibleShape {
                derivative_dimension: 0,
                assigned_slots: mapping.parameter_count(),
            });
        }

        Ok(Self)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectionalMapping {
    parameter: Parameter,
}

impl DirectionalMapping {
    pub(crate) fn try_from_mapping(
        mapping: &DerivativeMapping,
    ) -> Result<Self, DerivativeMappingError> {
        let [parameter]: [Parameter; 1] =
            mapping
                .slots()
                .try_into()
                .map_err(|_| DerivativeMappingError::IncompatibleShape {
                    derivative_dimension: 1,
                    assigned_slots: mapping.parameter_count(),
                })?;

        Ok(Self { parameter })
    }

    pub fn parameter(&self) -> Parameter {
        self.parameter
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BivariateMapping {
    first: Parameter,
    second: Parameter,
}

impl BivariateMapping {
    pub(crate) fn try_from_mapping(
        mapping: &DerivativeMapping,
    ) -> Result<Self, DerivativeMappingError> {
        let [first, second]: [Parameter; 2] =
            mapping
                .slots()
                .try_into()
                .map_err(|_| DerivativeMappingError::IncompatibleShape {
                    derivative_dimension: 2,
                    assigned_slots: mapping.parameter_count(),
                })?;

        Ok(Self { first, second })
    }

    pub fn parameters(&self) -> (Parameter, Parameter) {
        (self.first, self.second)
    }
}
