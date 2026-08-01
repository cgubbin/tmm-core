//! Caller-facing differential responses.
//!
//! This module defines the public representations used to return values and
//! derivatives from evaluated observables.
//!
//! Differential responses are assembled in three stages:
//!
//! 1. backend quantities retain derivatives in internal jet-valued forms;
//! 2. derivative-parts policies extract coordinate-free values and derivative
//!    components;
//! 3. typed parameter mappings attach caller-facing parameter metadata and
//!    construct a [`DifferentialResponse`].
//!
//! The public derivative representations are:
//!
//! - [`NoDerivatives`];
//! - [`DirectionalFirst`];
//! - [`DirectionalSecond`];
//! - [`BivariateFirst`];
//! - [`BivariateSecond`].
//!
//! Bivariate axes are ordered but not assumed to be spatial. They may represent
//! any supported caller-facing [`crate::parameter::Parameter`].

mod assemble;
mod bivariate;
mod directional;
mod response;

pub(crate) use assemble::{AssembleDifferentialResponse, IntoDifferentialResponse};

pub use bivariate::{BivariateFirst, BivariateGradient, BivariateHessian, BivariateSecond};

pub use directional::{DirectionalFirst, DirectionalSecond};

pub use response::{DifferentialResponse, NoDerivatives};
