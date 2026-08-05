//! Analytic integration primitives for homogeneous finite layers.
//!
//! This internal module separates three stages:
//!
//! 1. directional-wave products are integrated analytically;
//! 2. those products are transformed into canonical-state products;
//! 3. Hermitian state products are projected into complete electric and
//!    magnetic field norms.
//!
//! The wave-product layer supports both Hermitian real-frequency analysis and
//! bilinear complex-modal analysis. State products and field norms currently
//! implement the Hermitian path only.

mod bilinear_state_products;
mod field_norm;
mod hermitian_field_overlap;
mod hermitian_state_products;
mod wave_products;

pub(crate) use wave_products::{
    IntegratedWaveProducts, integrate_hermitian_cross_wave_products,
    integrate_hermitian_wave_products,
};

pub(crate) use hermitian_field_overlap::{
    HermitianOverlapError, IntegratedHermitianFieldOverlap, PairOperand,
    project_integrated_hermitian_field_overlap,
};

pub(crate) use hermitian_state_products::IntegratedHermitianStateProducts;

pub(crate) use field_norm::project_integrated_field_norms;

pub(super) use hermitian_state_products::{
    project_integrated_hermitian_cross_state_products, project_integrated_hermitian_state_products,
};
