//! Spatial coordinates, sampling requests, and profile extraction.
//!
//! This module defines the spatial layer used when evaluating quantities
//! through a planar stack. It is responsible for describing where quantities
//! are sampled, resolving those locations against a particular stack, and
//! converting resolved positions into the canonical spatial coordinate used by
//! the numerical backends.
//!
//! Caller-facing distances are represented by [`Length`]. A sampling request
//! may contain positions in either exterior region or inside finite layers.
//! During resolution, each position is assigned an unambiguous physical region
//! and finite-layer identity where applicable. Compilation then converts its
//! distance or layer offset into backend canonical units.
//!
//! Spatially sampled response arrays use their final ndarray axis for position.
//! Every preceding axis describes an excitation coordinate, such as vacuum
//! wavenumber or in-plane wavenumber. Profile extraction selects one index on
//! each excitation axis while retaining the final spatial axis as a borrowed
//! one-dimensional view.
//!
//! The sampling machinery deliberately preserves region identity at interfaces.
//! Two samples may therefore occupy the same geometric coordinate while
//! referring to opposite sides of an interface. This is required for quantities
//! whose Cartesian components are discontinuous across material boundaries.

mod compiled;
mod length;
mod resolved;
mod response;
mod sampling;

pub use length::Length;
pub use response::SpatialResponse;

pub(crate) use compiled::{CanonicalFieldPosition, CanonicalLayerPosition, CompiledFieldSampling};
pub(crate) use resolved::ResolvedFieldSampling;
pub(crate) use sampling::{
    ExteriorSampling, FieldPosition, FieldSampling, FieldSamplingError, LayerSampling,
    ResolvedLayerPosition,
};
