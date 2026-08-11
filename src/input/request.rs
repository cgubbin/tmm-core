use crate::parameter::{
    DerivativeMapping, DerivativeMappingError, Parameter, ParameterValidationError,
};

/// The numerical information requested from the solver.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SolveRequest {
    /// Compute values only.
    Value,

    /// Compute the value and first derivative with respect to one
    /// caller-facing parameter.
    UnivariateFirst { parameter: Parameter },

    /// Compute the value, first derivative, and second derivative with respect
    /// to one caller-facing parameter.
    UnivariateSecond { parameter: Parameter },

    /// Compute the value and first derivatives with respect to two
    /// caller-facing parameters.
    BivariateFirst { axis0: Parameter, axis1: Parameter },

    /// Compute the value, first derivatives, and full second-order derivative
    /// information with respect to two caller-facing parameters.
    ///
    /// This includes both pure second derivatives and the mixed derivative.
    BivariateSecond { axis0: Parameter, axis1: Parameter },
}

impl SolveRequest {
    pub(crate) fn parameters(&self) -> impl Iterator<Item = Parameter> {
        let parameters = match self {
            Self::Value => [None, None],
            Self::UnivariateFirst { parameter } => [Some(*parameter), None],
            Self::UnivariateSecond { parameter } => [Some(*parameter), None],
            Self::BivariateFirst { axis0, axis1 } => [Some(*axis0), Some(*axis1)],
            Self::BivariateSecond { axis0, axis1 } => [Some(*axis0), Some(*axis1)],
        };

        parameters.into_iter().flatten()
    }

    pub(crate) fn derivative_mapping(&self) -> Result<DerivativeMapping, DerivativeMappingError> {
        DerivativeMapping::new(self.parameters())
    }

    pub(crate) fn validate(
        &self,
        finite_layer_count: usize,
    ) -> Result<(), ParameterValidationError> {
        for parameter in self.parameters() {
            parameter.validate(finite_layer_count)?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::parameter::FiniteLayerIndex;

    #[test]
    fn value_request_has_no_parameters() {
        let request = SolveRequest::Value;

        let parameters = request.parameters().collect::<Vec<_>>();

        assert!(parameters.is_empty());
    }

    #[test]
    fn univariate_requests_preserve_parameter() {
        for request in [
            SolveRequest::UnivariateFirst {
                parameter: Parameter::Spectral,
            },
            SolveRequest::UnivariateSecond {
                parameter: Parameter::Spectral,
            },
        ] {
            let parameters = request.parameters().collect::<Vec<_>>();

            assert_eq!(parameters, vec![Parameter::Spectral]);
        }
    }

    #[test]
    fn bivariate_requests_preserve_parameter_order() {
        let first = Parameter::LayerThickness(FiniteLayerIndex::new(2));
        let second = Parameter::InPlane;

        for request in [
            SolveRequest::BivariateFirst {
                axis0: first,
                axis1: second,
            },
            SolveRequest::BivariateSecond {
                axis0: first,
                axis1: second,
            },
        ] {
            let parameters = request.parameters().collect::<Vec<_>>();

            assert_eq!(parameters, vec![first, second]);
        }
    }

    #[test]
    fn derivative_mapping_preserves_request_order() {
        let first = Parameter::Spectral;
        let second = Parameter::LayerThickness(FiniteLayerIndex::new(1));

        let request = SolveRequest::BivariateSecond {
            axis0: first,
            axis1: second,
        };

        let mapping = request
            .derivative_mapping()
            .expect("distinct parameters should produce a mapping");

        assert_eq!(mapping.parameter(0), Some(first));
        assert_eq!(mapping.parameter(1), Some(second));

        assert_eq!(mapping.slot_for(first), Some(0));
        assert_eq!(mapping.slot_for(second), Some(1));
    }

    #[test]
    fn duplicate_bivariate_parameter_is_rejected() {
        let request = SolveRequest::BivariateFirst {
            axis0: Parameter::Spectral,
            axis1: Parameter::Spectral,
        };

        let error = request
            .derivative_mapping()
            .expect_err("duplicate parameters must be rejected");

        assert_eq!(
            error,
            DerivativeMappingError::DuplicateVariable {
                variable: Parameter::Spectral,
                first_slot: 0,
                second_slot: 1,
            },
        );
    }

    #[test]
    fn coordinate_parameters_validate_for_empty_stack() {
        for request in [
            SolveRequest::UnivariateFirst {
                parameter: Parameter::Spectral,
            },
            SolveRequest::UnivariateFirst {
                parameter: Parameter::InPlane,
            },
            SolveRequest::BivariateFirst {
                axis0: Parameter::Spectral,
                axis1: Parameter::InPlane,
            },
        ] {
            assert!(request.validate(0).is_ok());
        }
    }

    #[test]
    fn valid_layer_thickness_parameter_passes_validation() {
        let request = SolveRequest::UnivariateSecond {
            parameter: Parameter::LayerThickness(FiniteLayerIndex::new(2)),
        };

        assert!(request.validate(3).is_ok());
    }

    #[test]
    fn layer_index_equal_to_layer_count_is_rejected() {
        let request = SolveRequest::UnivariateFirst {
            parameter: Parameter::LayerThickness(FiniteLayerIndex::new(3)),
        };

        let error = request
            .validate(3)
            .expect_err("layer index equal to layer count is out of bounds");

        assert_eq!(
            error,
            ParameterValidationError::LayerOutOfBounds {
                index: 3,
                finite_layer_count: 3,
            },
        );
    }

    #[test]
    fn bivariate_validation_checks_both_parameters() {
        let request = SolveRequest::BivariateSecond {
            axis0: Parameter::Spectral,
            axis1: Parameter::LayerThickness(FiniteLayerIndex::new(4)),
        };

        let error = request
            .validate(2)
            .expect_err("invalid second parameter should be detected");

        assert_eq!(
            error,
            ParameterValidationError::LayerOutOfBounds {
                index: 4,
                finite_layer_count: 2,
            },
        );
    }
}
