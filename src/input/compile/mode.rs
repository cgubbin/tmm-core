use crate::input::compile::assignment::ProblemVariable;

/// The actual jet representation required to satisfy a caller request.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CompilationMode {
    Value,

    /// One independent variable, first order.
    UnivariateFirst {
        variable: ProblemVariable,
    },

    /// One independent variable, second order.
    UnivariateSecond {
        variable: ProblemVariable,
    },

    /// Two independent variables, first order.
    BivariateFirst {
        first: ProblemVariable,
        second: ProblemVariable,
    },

    /// Two independent variables with full second-order information.
    BivariateSecond {
        first: ProblemVariable,
        second: ProblemVariable,
    },
}
