use crate::input::{
    DerivativeParameter, PlaneWaveCoordinates,
    compile::assignment::{ParameterAssignment, ParameterAssignmentError, ProblemVariable},
    parameter::SolveRequest,
};

use thiserror::Error;

pub(crate) enum CompilationPlan {
    Value {
        assignment: ParameterAssignment,
    },

    UnivariateFirst {
        assignment: ParameterAssignment,
        parameter: DerivativeParameter,
    },

    UnivariateSecond {
        assignment: ParameterAssignment,
        parameter: DerivativeParameter,
    },

    CoordinateGradient {
        assignment: ParameterAssignment,
    },

    CoordinateHessian {
        assignment: ParameterAssignment,
    },
}

#[derive(Debug, Error)]
pub enum RequestError {
    #[error(
        "layer {layer} does not exist \
         (stack contains {layer_count} layers)"
    )]
    LayerOutOfBounds { layer: usize, layer_count: usize },

    #[error("error in parameter assignment: {0}")]
    Assignment(#[from] ParameterAssignmentError),
}

pub(crate) fn plan_compilation(
    request: SolveRequest,
    layer_count: usize,
) -> Result<CompilationPlan, RequestError> {
    match request {
        SolveRequest::Value => Ok(CompilationPlan::Value {
            assignment: ParameterAssignment::none(),
        }),

        SolveRequest::First { parameter } => {
            validate_parameter(parameter, layer_count)?;

            Ok(CompilationPlan::UnivariateFirst {
                assignment: assignment_for_parameter(parameter),
                parameter,
            })
        }

        SolveRequest::Second { parameter } => {
            validate_parameter(parameter, layer_count)?;

            Ok(CompilationPlan::UnivariateSecond {
                assignment: assignment_for_parameter(parameter),
                parameter,
            })
        }

        SolveRequest::CoordinateGradient => Ok(CompilationPlan::CoordinateGradient {
            assignment: ParameterAssignment::new([
                ProblemVariable::Spectral,
                ProblemVariable::InPlane,
            ])?,
        }),

        SolveRequest::CoordinateHessian => Ok(CompilationPlan::CoordinateHessian {
            assignment: ParameterAssignment::new([
                ProblemVariable::Spectral,
                ProblemVariable::InPlane,
            ])?,
        }),
    }
}

fn assignment_for_parameter(parameter: DerivativeParameter) -> ParameterAssignment {
    match parameter {
        DerivativeParameter::Spectral => ParameterAssignment::spectral(),

        DerivativeParameter::InPlane => ParameterAssignment::in_plane(),

        DerivativeParameter::LayerThickness { layer } => {
            ParameterAssignment::layer_thickness(layer)
        }
    }
}

fn validate_parameter(
    parameter: DerivativeParameter,
    layer_count: usize,
) -> Result<(), RequestError> {
    if let DerivativeParameter::LayerThickness { layer } = parameter {
        if layer >= layer_count {
            return Err(RequestError::LayerOutOfBounds { layer, layer_count });
        }
    }

    Ok(())
}
