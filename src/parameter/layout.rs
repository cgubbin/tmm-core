use super::{
    Parameter,
    derivative_mapping::{DerivativeMapping, DerivativeMappingError},
};

/// Parameter mapping for a value-only response.
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

/// Parameter mapping for a one-direction derivative response.
///
/// `parameter()` identifies the caller-facing parameter represented by the
/// derivative direction.
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

    pub(crate) fn parameter(&self) -> Parameter {
        self.parameter
    }
}

/// Parameter mapping for a two-direction derivative response.
///
/// The returned parameter order matches the first and second jet directions.
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

    pub(crate) fn parameters(&self) -> (Parameter, Parameter) {
        (self.first, self.second)
    }
}
