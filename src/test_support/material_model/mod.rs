mod delegate;
pub(crate) mod drude;
pub(crate) mod drude_lorentz;
pub(crate) mod lorentz;
pub(crate) mod magnetic_loss;

pub(crate) use drude::Drude;
pub(crate) use drude_lorentz::DrudeLorentz;

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
}
