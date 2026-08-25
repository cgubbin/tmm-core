//! Conversion of internal algebraic results into public differential responses.
//!
//! Plane-wave evaluation propagates values and derivatives together using jet
//! algebra. For the ordinary evaluator, those jet-valued structures are
//! crystallised into backend-independent response types before being returned
//! to the caller.
//!
//! Crystallisation separates an evaluated structure into:
//!
//! - its physical values;
//! - any requested first derivatives;
//! - any requested second derivatives.
//!
//! The resulting components are stored in types from [`crate::differential`].
//! Advanced evaluation paths may instead retain the raw algebraic
//! representation and therefore bypass this module.
//!
//! # Delayed crystallisation
//!
//! Crystallisation is performed only after all requested physical quantities
//! have been evaluated. This allows algebraic operations to act on complete
//! jet-valued structures before the final result is transposed from structures
//! containing jets into differential responses containing structures.
//!
//! Conceptually, a directional first-order result is transformed from
//!
//! ```text
//! Quantity<Jet1<Value>>
//! ```
//!
//! into
//!
//! ```text
//! DifferentialResponse<
//!     Quantity<Value>,
//!     DirectionalFirst<Quantity<Value>>,
//! >
//! ```
//!
//! The same process applies recursively to composite quantities such as
//! plane-wave amplitudes, powers, and complete observable sets.
//!
//! # Responsibilities
//!
//! This module:
//!
//! - extracts value and derivative components from internal algebraic types;
//! - recursively transposes composite evaluated quantities;
//! - selects the public derivative representation associated with the active
//!   jet family and compiled derivative mapping.
//!
//! It does not perform differentiation, coordinate transformation, or physical
//! interpretation of derivative coordinates. Those responsibilities belong to
//! the algebra, compilation, and evaluator layers.
//!
//! The principal internal components are:
//!
//! - decomposition traits in `decompose`;
//! - derivative-part structures in `parts`;
//! - crystallisation policies in `policy`;
//! - recursive implementations for evaluated quantities in `quantity`.

mod decompose;
mod parts;
mod policy;
mod quantity;

pub(crate) use decompose::{
    IntoBivariateFirst, IntoBivariateSecond, IntoFirst, IntoSecond, IntoValue,
};
pub(crate) use parts::{
    BivariateFirstParts, BivariateSecondParts, DirectionalFirstParts, DirectionalSecondParts,
    ValuePart,
};
pub(crate) use policy::{
    DerivativePartsPolicy, FirstBivariate, FirstDirectional, IntoDerivativeParts, SecondBivariate,
    SecondDirectional, ValueOnly,
};
