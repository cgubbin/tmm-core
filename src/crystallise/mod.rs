//! Conversion of internal algebraic results into public differential responses.
//!
//! Backend calculations propagate values and derivatives together using jet
//! algebra. Those jet-valued structures are convenient during evaluation, but
//! they are not intended to form part of the public response API.
//!
//! Crystallisation separates an evaluated structure into:
//!
//! - its physical values;
//! - any requested first derivatives;
//! - any requested second derivatives.
//!
//! The resulting components are stored in backend-independent types from the
//! [`crate::differential`] module.
//!
//! # Delayed crystallisation
//!
//! Crystallisation is performed only after all requested physical quantities
//! have been evaluated. This allows algebraic operations to act on complete
//! jet-valued structures and then transposes the final result from structures
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
//! - selects the public derivative representation requested by the evaluator.
//!
//! It does not perform differentiation, apply coordinate transformations, or
//! interpret the physical meaning of bivariate coordinates. Those operations
//! belong to the algebra, coordinate, and evaluator layers respectively.
//!
//! The principal internal components are:
//!
//! - decomposition traits in `parts`;
//! - crystallisation policies in `policy`;
//! - recursive implementations for evaluated quantities in `quantity`.

mod parts;
mod policy;
mod quantity;

pub(crate) use parts::{
    BivariateGradientParts, BivariateHessianParts, DirectionalFirstParts, DirectionalSecondParts,
    IntoFirst, IntoGradient, IntoHessian, IntoSecond, IntoValue,
};
pub(crate) use policy::{
    Crystallise, CrystallisePolicy, FirstBivariate, FirstDirectional, SecondBivariate,
    SecondDirectional, ValueOnly,
};
