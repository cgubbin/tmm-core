//! # lamina-core
//!
//! `lamina-core` is a transfer- and scattering-matrix library for planar
//! electromagnetic systems.
//!
//! It evaluates plane-wave response, electromagnetic fields, derivatives, and
//! outgoing modal solutions for stacks of homogeneous planar layers. The
//! library is designed as the numerical core of the Lamina ecosystem: material
//! models, coordinate systems, plotting, and mode-finding algorithms can be
//! built around the canonical calculations provided here.
//!
//! The crate provides two complementary evaluation paths:
//!
//! - [`RealAxisEvaluator`] evaluates driven plane-wave problems in caller-facing
//!   physical coordinates and exposes derivatives with respect to physical
//!   parameters;
//! - [`ComplexPlaneEvaluator`] evaluates determinants and outgoing modes directly
//!   in canonical complex coordinates, with caller-controlled holomorphic
//!   derivative algebra and explicit exterior-wavevector branches.
//!
//! Two isotropic 2×2 numerical backends are provided:
//!
//! - [`Scatter2`] uses scattering-matrix composition and is the recommended
//!   default;
//! - [`Transfer2`] uses transfer matrices and is useful for comparison,
//!   diagnostics, and problems where the transfer formulation is well
//!   conditioned.
//!
//! ## Plane-wave response
//!
//! A real-axis evaluation describes a driven scattering problem. The evaluator
//! converts caller-facing coordinates into the canonical representation,
//! evaluates the stack, and projects the result into physical observables.
//!
//! For a plane wave incident on a planar stack, the basic response contains the
//! complex reflection and transmission amplitudes together with reflectance,
//! transmittance, and absorptance:
//!
//! ```text
//! amplitudes:  r, t
//! powers:      R, T, A
//! ```
//!
//! The real-axis API is intended to be the normal entry point for spectroscopy,
//! angular scans, parameter sweeps, sensitivity calculations, and field
//! reconstruction.
//!
//! ```ignore
//! use lamina_core::{Polarisation, RealAxisEvaluator, Scatter2};
//!
//! let evaluator = RealAxisEvaluator::new(Scatter2::new());
//!
//! let response = evaluator.evaluate(
//!     coordinates,
//!     &stack,
//!     Polarisation::TransverseElectric,
//! )?;
//!
//! let amplitudes = response.amplitudes();
//! let power = response.power();
//!
//! println!("r = {:?}", amplitudes.reflection());
//! println!("t = {:?}", amplitudes.transmission());
//! println!("R = {:?}", power.reflectance());
//! println!("T = {:?}", power.transmittance());
//! println!("A = {:?}", power.absorptance());
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! The exact construction of `coordinates` and `stack` depends on the
//! caller-facing coordinate and material types. See the examples shipped with
//! the crate for complete executable problems.
//!
//! ## Derivatives
//!
//! Derivatives are propagated through the electromagnetic calculation using
//! forward-mode jet algebra rather than finite differences.
//!
//! Real-axis evaluators attach derivative coordinates to physical
//! [`Parameter`] values. A parameter may, for example, represent the spectral
//! coordinate, the in-plane coordinate, or the thickness of a particular
//! finite layer.
//!
//! First- and second-order directional derivatives are available, as are
//! bivariate derivatives when mixed derivatives are required. The resulting
//! public differential responses separate the physical value from its
//! derivatives.
//!
//! Conceptually,
//!
//! ```text
//! evaluate
//!     -> value
//!
//! evaluate_first(parameter)
//!     -> value
//!      + ∂/∂parameter
//!
//! evaluate_second(parameter)
//!     -> value
//!      + ∂/∂parameter
//!      + ∂²/∂parameter²
//!
//! evaluate_bivariate_second(parameter0, parameter1)
//!     -> value
//!      + gradient
//!      + symmetric 2×2 Hessian
//! ```
//!
//! Layer thicknesses and coordinate transformations are differentiated as part
//! of the compiled problem, so returned derivatives correspond to the
//! caller-facing parameters rather than merely to the backend's internal
//! canonical variables.
//!
//! ## Retained solutions and fields
//!
//! Ordinary evaluation retains only the state required to construct the final
//! plane-wave response. When internal fields or layer observables are required,
//! use the retained evaluation path.
//!
//! A retained solution preserves the intermediate layer data required to
//! reconstruct directional waves and electromagnetic fields throughout the
//! stack.
//!
//! Field sampling is specified independently of the backend calculation. A
//! [`FieldSampling`] request can select positions within finite layers, after
//! which the retained solution can reconstruct quantities such as:
//!
//! - electric and magnetic fields;
//! - electric displacement and magnetic induction;
//! - field intensities and Poynting vectors;
//! - energy and dissipation densities;
//! - interface and layer power;
//! - integrated layer energy and dissipation;
//! - layer participation and confinement;
//! - Hermitian and bilinear overlaps.
//!
//! This separation allows the electromagnetic solve to be performed once and
//! subsequently sampled or integrated in several different ways.
//!
//! ## Complex-plane evaluation
//!
//! Outgoing modes require a different contract from driven real-axis
//! scattering.
//!
//! [`ComplexPlaneEvaluator`] therefore operates directly on the canonical
//! problem. The caller supplies:
//!
//! - [`CanonicalCoordinates`] containing complex spectral and in-plane
//!   coordinates;
//! - the compiled canonical stack;
//! - [`ExteriorWavevectors`] specifying the selected exterior branches;
//! - the polarization.
//!
//! For example, the determinant interface has the form
//!
//! ```ignore
//! use lamina_core::{ComplexPlaneEvaluator, Polarisation, Scatter2};
//!
//! let evaluator =
//!     ComplexPlaneEvaluator::compile(&stack, Scatter2::new())?;
//!
//! let determinant = evaluator.determinant(
//!     coordinates,
//!     exterior_wavevectors,
//!     Polarisation::TransverseElectric,
//! )?;
//!
//! println!("D = {:?}", determinant);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! This lower-level interface is deliberate. In the complex plane there is no
//! globally correct automatic choice of longitudinal-wavevector branch.
//! Mode-finding and continuation algorithms can therefore track branches
//! explicitly instead of having branch selection hidden inside a convenience
//! coordinate conversion.
//!
//! ## Outgoing modes
//!
//! A zero of the complex-plane determinant corresponds to a non-trivial
//! solution of the homogeneous outgoing boundary problem.
//!
//! Retaining a complex-plane evaluation provides the backend state required to
//! reconstruct that solution. The modal reconstruction:
//!
//! 1. constructs a nonzero null vector of the outgoing boundary system;
//! 2. reconstructs exterior and finite-layer directional waves;
//! 3. propagates the modal state throughout the stack;
//! 4. applies bilinear quasinormal-mode normalization.
//!
//! The resulting mode can then be sampled using the same field machinery used
//! by retained driven solutions.
//!
//! Complex modal normalization is **bilinear**, not Hermitian. In particular,
//! field products have the form
//!
//! ```text
//! E · E
//! ```
//!
//! rather than
//!
//! ```text
//! E* · E.
//! ```
//!
//! For dispersive materials, the normalization includes the corresponding
//! spectral constitutive derivatives.
//!
//! ## Canonical coordinates
//!
//! Backend calculations use canonical angular-wavenumber coordinates:
//!
//! ```text
//! k₀       vacuum angular wavenumber
//! k∥       in-plane angular wavenumber
//! κ        longitudinal angular wavenumber
//! ```
//!
//! Real-axis callers do not normally need to work with this representation
//! directly. Caller-facing coordinates are resolved and transformed before
//! reaching the backend.
//!
//! Complex-plane calculations expose the canonical representation intentionally:
//! it provides the stable coordinate contract needed by root finders,
//! continuation algorithms, conformal mappings, and other modal solvers.
//!
//! ## Materials
//!
//! Material models are evaluated through constitutive traits rather than being
//! hard-coded into the numerical backends. The same stack representation can
//! therefore be sampled on either the real spectral axis or in the complex
//! plane when its material models support the corresponding domain.
//!
//! Isotropic layer calculations propagate the constitutive parameters
//!
//! ```text
//! ε(k₀), μ(k₀)
//! ```
//!
//! together with any derivatives required by the selected jet algebra.
//!
//! This makes dispersive material response part of the differentiated
//! electromagnetic problem rather than a separate post-processing correction.
//!
//! ## Scattering and transfer backends
//!
//! [`Scatter2`] and [`Transfer2`] solve the same isotropic scalar-channel
//! problem and expose the same backend-independent response and reconstruction
//! interfaces.
//!
//! [`Scatter2`] composes interfaces and propagation regions using scattering
//! matrices. It avoids the exponentially growing intermediate states that make
//! ordinary transfer matrices unreliable for strongly evanescent, absorbing,
//! or optically thick structures, and should normally be preferred.
//!
//! [`Transfer2`] propagates a two-component field/slope state directly through
//! the stack. It is compact and useful as an independent formulation, but its
//! matrices can become poorly conditioned or overflow for difficult stacks.
//! [`TransferStabilityCheck`] controls explicit non-finite checks performed by
//! the transfer backend.
//!
//! ## Array-valued evaluation
//!
//! The numerical core is generic over [`ndarray`] dimensions. Coordinates and
//! the associated jet algebra may therefore represent a scalar point or sampled
//! arrays without changing the underlying electromagnetic formulation.
//!
//! This is useful for spectral and angular sweeps: material evaluation,
//! propagation, differentiation, and observable projection operate directly on
//! the sampled arrays.
//!
//! ## Architecture
//!
//! The main calculation pipeline is:
//!
//! ```text
//! caller-facing problem
//!        │
//!        ▼
//! coordinate resolution and validation
//!        │
//!        ▼
//! canonical coordinates + canonical stack
//!        │
//!        ▼
//! constitutive evaluation
//!        │
//!        ▼
//! jet-valued isotropic layer quantities
//!        │
//!        ▼
//! Scatter2 / Transfer2
//!        │
//!        ├──────────────► plane-wave response
//!        │
//!        ▼
//! retained workspace
//!        │
//!        ├──────────────► fields and layer observables
//!        │
//!        └──────────────► outgoing modal reconstruction
//! ```
//!
//! Values and derivatives remain together in the internal algebra throughout
//! the physical calculation. They are separated into public differential
//! responses only after the requested observable has been evaluated. This
//! delayed crystallisation keeps differentiation independent of the particular
//! observable being computed.
//!
//! ## Scope
//!
//! `lamina-core` supplies the electromagnetic primitives needed to analyse
//! planar systems. Higher-level tasks such as mode searches, branch tracking,
//! continuation, optimization, plotting, and experiment-facing parameterization
//! can be implemented by other crates without being coupled to a particular
//! backend representation.
//!
//! The current numerical backends implement isotropic 2×2 transfer and
//! scattering formulations.

pub(crate) mod algebra;
pub mod backend;
pub(crate) mod derivative_parts;
mod differential;
mod domain;
mod evaluate;
pub mod field;
mod input;
pub mod material;
mod observable;
mod parameter;
mod projection;
mod scalar;
mod spatial;
pub mod stack;
mod waves;

#[cfg(test)]
mod test_support;

pub use algebra::{Jet, ModeJet1, ScalarAlgebra, SeedJet};

pub use backend::{Backend, ExteriorWavevectors, Scatter2, Transfer2};

pub use domain::{ComplexPlane, RealAxis};

pub use input::{
    CanonicalCoordinates, CanonicalStack, CoordinateGrid, CoordinateInput, CoordinateReference,
    CoordinateSamples, Coordinates, InPlaneCoordinate, IncidentSide, Polarisation,
    SpectralCoordinate, StackCompileError, StackThicknessJet, compile_canonical_constant_stack,
};

pub use evaluate::{ComplexPlaneEvaluator, RealAxisEvaluator};

pub use material::{
    AnalyticalMaterialHandle, Constant, ConstitutiveEvaluator, ConstitutiveLift, DerivativeOrder,
    DifferentiableMaterial, DifferentiableMeromorphicMaterial, EvaluateDifferentiableMaterial,
    EvaluateDifferentiableMeromorphicMaterial, EvaluateMaterial, EvaluateMeromorphicMaterial,
    Material, MeromorphicMaterial, Sampled, Scalar,
};

pub use field::VectorField;

pub use observable::{
    AggregateBilinearNormalization, ConstitutiveFields, DirectedPower, ElectromagneticDissipation,
    ElectromagneticEnergy, ElectromagneticFields, ElectromagneticIntensities, FieldIndexError,
    InterfacePower, LayerDissipation, LayerPower, PlaneWaveAmplitudes, PlaneWaveDeterminant,
    PlaneWavePower, ProjectPlaneWaveModeDeterminant,
};
pub use parameter::{FiniteLayerIndex, Parameter};
pub use scalar::ComplexScalar;

pub use stack::{
    AnalyticalMaterialStack, DifferentiableMaterialStack, Layer, MaterialStack,
    MeromorphicMaterialStack, Stack,
};
