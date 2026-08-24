//! Analytic integration primitives for homogeneous finite layers.
//!
//! This internal module separates three stages:
//!
//! 1. directional-wave products are integrated analytically;
//! 2. those products are transformed into canonical-state products;
//! 3. canonical-state products are projected into electromagnetic field
//!    norms or overlaps.
//!
//! Two algebraic paths are supported:
//!
//! - Hermitian products conjugate the reference field and are used for
//!   real-frequency norms, energy, dissipation, and Hermitian overlaps;
//! - bilinear products apply no conjugation and preserve holomorphic
//!   dependence for complex modal overlaps and normalization.
//!
//! The same directional-wave integration machinery underlies both paths.

mod bilinear_field_overlap;
mod bilinear_state_products;
mod field_norm;
mod hermitian_field_overlap;
mod hermitian_state_products;
pub(crate) mod wave_products;

pub(crate) use wave_products::{
    IntegratedWaveProducts, integrate_bilinear_cross_wave_products,
    integrate_bilinear_wave_products, integrate_hermitian_cross_wave_products,
    integrate_hermitian_wave_products,
};

pub(crate) use bilinear_state_products::IntegratedBilinearCrossStateProducts;
pub(super) use bilinear_state_products::{
    project_integrated_bilinear_cross_state_products, project_integrated_bilinear_state_products,
};

pub(crate) use hermitian_state_products::IntegratedHermitianCrossStateProducts;
pub(super) use hermitian_state_products::{
    project_integrated_hermitian_cross_state_products, project_integrated_hermitian_state_products,
};

pub(crate) use bilinear_field_overlap::project_integrated_bilinear_field_overlap;
pub(crate) use field_norm::project_integrated_field_norms;
pub(crate) use hermitian_field_overlap::project_integrated_hermitian_field_overlap;
