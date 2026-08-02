//! Metadata for interpreting core response values.
//!
//! The core solver works exclusively with canonical coordinates:
//!
//! - vacuum wavenumber `k₀` in cm⁻¹;
//! - conserved parallel wavenumber `k∥` in cm⁻¹;
//! - spatial positions and layer thicknesses in cm.
//!
//! [`crate::input::PlaneWaveInput`] is used directly as the metadata for
//! ordinary externally excited plane-wave responses. The types in this module
//! add information for responses defined over an additional spatial,
//! interface, or layer domain.

mod field;
mod interface;
mod layer;

pub use field::{FieldMetadata, StackRegion};
pub use interface::{InterfaceLocation, InterfaceMetadata};
pub use layer::{LayerIndex, LayerLocation, LayerMetadata};
