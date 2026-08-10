//! Typed results returned by observable evaluations.
//!
//! This module defines the response types used to return computed observables,
//! their derivatives, and the metadata required to interpret them.
//!
//! # Response structure
//!
//! A response consists of three parts:
//!
//! - the computed observable values;
//! - any requested derivatives;
//! - metadata describing the coordinates on which the values were evaluated.
//!
//! The generic representation is [`Response`]:
//!
//! ```text
//! Response<O, D, M>
//! ├── observables: O
//! ├── derivatives: D
//! └── metadata: M
//! ```
//!
//! `O` is the observable payload, `D` is either a derivative payload or
//! [`NoDerivatives`](crate::differential::NoDerivatives), and `M` contains the
//! response-specific metadata.
//!
//! Most users interact with concrete response aliases rather than constructing
//! [`Response`] directly. These include responses for:
//!
//! - plane-wave observables;
//! - electromagnetic and constitutive fields;
//! - energy and dissipation densities;
//! - interface-resolved quantities;
//! - layer-resolved quantities.
//!
//! # Differential responses
//!
//! Observable values and their derivatives are grouped by
//! [`DifferentialResponse`](crate::differential::DifferentialResponse).
//!
//! This preserves the same distinction throughout the API:
//!
//! ```text
//! DifferentialResponse<O, D>
//! ├── observables: O
//! └── derivatives: D
//! ```
//!
//! When no derivatives were requested, `D` is
//! [`NoDerivatives`](crate::differential::NoDerivatives). Consequently, callers
//! can inspect values through the same response API regardless of whether the
//! calculation included derivatives.
//!
//! # Excitation and semantic axes
//!
//! Array-valued observables may contain several excitation dimensions followed
//! by one semantic result dimension.
//!
//! For example, a field response evaluated on a two-dimensional excitation grid
//! has the conceptual shape
//!
//! ```text
//! (spectral, in-plane, position)
//! ```
//!
//! An interface response may instead have shape
//!
//! ```text
//! (spectral, in-plane, interface)
//! ```
//!
//! and a layer response may have shape
//!
//! ```text
//! (spectral, in-plane, layer)
//! ```
//!
//! The final axis is therefore not another excitation coordinate. It identifies
//! positions, interfaces, or layers, depending on the response type.
//!
//! # Profiling
//!
//! Spatially, interface-, and layer-resolved responses support profiling at one
//! excitation point.
//!
//! Profiling selects every excitation axis while retaining the final semantic
//! axis. For example:
//!
//! ```text
//! (spectral, in-plane, position)
//!             │       │
//!             └───────┴── select `(i, j)`
//!
//!                   ↓
//!
//!               (position)
//! ```
//!
//! The public profiling methods accept any index implementing
//! [`ndarray::IntoDimension`] for the excitation dimension. This gives natural
//! syntax for each dimensionality:
//!
//! ```ignore
//! point_response.profile(())?;
//! sample_response.profile(4)?;
//! grid_response.profile((2, 7))?;
//! ```
//!
//! Profiling is a borrowing operation. It returns views into the original
//! response rather than allocating or copying observable arrays.
//!
//! # Recursive profiling
//!
//! Profiling is implemented recursively through
//! [`SpatialProfile`](crate::SpatialProfile).
//!
//! Observable containers and derivative containers implement the trait by
//! profiling their constituent fields or quantities. Leaf field types perform
//! the underlying array operation through their `profile_last_axis` methods.
//!
//! This separation is intentional:
//!
//! - leaf field types know how to retain their final array axis;
//! - observable types know that the operation represents a physical profile;
//! - derivative wrappers preserve their structure while profiling their
//!   contents;
//! - response types attach the corresponding excitation point and metadata.
//!
//! As a result, profiling preserves the complete differential response. A
//! profiled response contains both the selected observable values and the
//! corresponding selected derivatives.
//!
//! # Field profiles
//!
//! A field profile contains:
//!
//! - the profiled observable and derivative payload;
//! - the canonical excitation point;
//! - sampled positions in centimetres;
//! - the stack region associated with each position.
//!
//! The positions and region arrays share the retained spatial axis.
//!
//! # Interface and layer profiles
//!
//! Interface and layer responses follow the same model.
//!
//! An interface profile retains the interface axis and its associated interface
//! metadata. A layer profile retains the layer axis and its associated layer
//! metadata. Their observable and derivative payloads are profiled recursively
//! in the same way as field responses.
//!
//! # Shape invariants
//!
//! Profile-capable response types enforce the relationship between excitation
//! and stored observable dimensions through `ndarray` dimension types.
//!
//! If `ED` is the excitation dimension, the stored field dimension is
//! `ED::Larger`. Selecting all axes in `ED` must therefore leave exactly one
//! axis.
//!
//! Violations of user-supplied indices are reported as
//! [`SpatialProfileError`](crate::SpatialProfileError). Internal failures to
//! reduce `ED::Larger` to one retained axis indicate a broken type or
//! construction invariant rather than a recoverable runtime condition.
//!
//! # Ownership
//!
//! Response objects own their computed arrays. Profiles borrow from those
//! arrays and cannot outlive the source response.
//!
//! This makes extraction of individual profiles inexpensive, including for
//! large excitation grids and derivative payloads.

mod field;
mod interface;
mod layer;
mod metadata;
mod response_type;

pub use metadata::{
    FieldMetadata, InterfaceLocation, InterfaceMetadata, LayerLocation, LayerMetadata, StackRegion,
};
pub use response_type::Response;
