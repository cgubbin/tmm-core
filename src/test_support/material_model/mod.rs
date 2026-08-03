mod delegate;
pub(crate) mod drude;
pub(crate) mod drude_lorentz;
pub(crate) mod lorentz;

pub(crate) use drude::Drude;
pub(crate) use drude_lorentz::DrudeLorentz;
pub(crate) use lorentz::Lorentz;

/// Validation errors shared by analytical models.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum MaterialModelError<R>
where
    R: std::fmt::Debug,
{
    /// A parameter is NaN or infinite.
    #[error("material parameter `{name}` must be finite, found {value:?}")]
    NonFiniteParameter {
        /// Parameter name.
        name: &'static str,
        /// Invalid value.
        value: R,
    },

    /// A parameter required to be nonnegative is negative.
    #[error("material parameter `{name}` must be nonnegative, found {value:?}")]
    NegativeParameter {
        /// Parameter name.
        name: &'static str,
        /// Invalid value.
        value: R,
    },

    /// A parameter required to be positive is zero or negative.
    #[error("material parameter `{name}` must be positive, found {value:?}")]
    NonPositiveParameter {
        /// Parameter name.
        name: &'static str,
        /// Invalid value.
        value: R,
    },

    /// A coefficient list is empty.
    #[error("material parameter `{name}` must contain at least one coefficient")]
    EmptyCoefficients {
        /// Parameter name.
        name: &'static str,
    },
}
