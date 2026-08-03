//! # Material capabilities
//!
//! Optical calculations in `tmm-core` distinguish between real-axis material
//! evaluation, real-axis differentiation, and continuation into the complex
//! spectral plane.
//!
//! These are separate mathematical capabilities. In particular, the ability to
//! evaluate or differentiate measured optical data on the real axis does not
//! imply that the data possesses a unique or physically meaningful
//! continuation to complex frequency.
//!
//! The material API therefore consists of four traits:
//!
//! ```text
//! Material
//! ├── real-axis constitutive properties
//! │
//! ├──► DifferentiableMaterial
//! │      real-axis derivatives
//! │
//! └── supplied automatically for MeromorphicMaterial
//!
//! MeromorphicMaterial
//! ├── complex-frequency constitutive properties
//! │
//! └──► DifferentiableMeromorphicMaterial
//!        complex-frequency derivatives
//!        and automatic real-axis derivative support
//! ```
//!
//! ## [`Material`]
//!
//! [`Material`] represents an optical material evaluated on the real vacuum
//! angular-wavenumber axis.
//!
//! It is sufficient for calculations such as:
//!
//! - reflection and transmission amplitudes;
//! - reflectance, transmittance, and absorptance;
//! - phase and ellipsometry;
//! - real-frequency field reconstruction.
//!
//! Implementations may be analytical, tabulated, interpolated, or externally
//! supplied. No differentiability or complex-frequency continuation is
//! implied.
//!
//! ## [`DifferentiableMaterial`]
//!
//! [`DifferentiableMaterial`] adds first- and second-order derivatives on the
//! real spectral axis.
//!
//! The derivatives may be obtained from:
//!
//! - an analytical model;
//! - automatic differentiation;
//! - derivatives of an interpolation spline;
//! - finite differences;
//! - another explicit numerical differentiation scheme.
//!
//! This capability supports derivatives of real-frequency observables such as
//! reflectance, transmission phase, group delay, and ellipsometric quantities.
//!
//! Implementing [`DifferentiableMaterial`] does not imply that the material can
//! be evaluated at complex frequency.
//!
//! ## [`MeromorphicMaterial`]
//!
//! [`MeromorphicMaterial`] represents a constitutive model with a defined
//! meromorphic continuation into the complex vacuum-angular-wavenumber plane.
//!
//! It is intended for:
//!
//! - complex-frequency mode finding;
//! - argument-principle methods;
//! - contour integration;
//! - complex root searches;
//! - outgoing-boundary determinant calculations.
//!
//! Typical implementations include Drude, Lorentz, and Drude–Lorentz
//! oscillator models.
//!
//! A type implementing [`MeromorphicMaterial`] automatically implements
//! [`Material`] by evaluating the complex model on the real axis.
//!
//! Tabulated data and arbitrary real-axis interpolation should generally not
//! implement [`MeromorphicMaterial`], because they do not define a unique
//! physically meaningful complex continuation.
//!
//! ## [`DifferentiableMeromorphicMaterial`]
//!
//! [`DifferentiableMeromorphicMaterial`] adds derivatives of the same
//! meromorphic continuation throughout the complex spectral plane.
//!
//! This capability is required for:
//!
//! - derivatives of complex outgoing-boundary determinants;
//! - Newton refinement of complex modes;
//! - contour-integral and argument-principle methods using logarithmic
//!   derivatives;
//! - analytical complex modal sensitivities.
//!
//! A type implementing [`DifferentiableMeromorphicMaterial`] automatically
//! implements [`DifferentiableMaterial`] by restricting its complex
//! derivatives to the real axis.
//!
//! ## Typical implementations
//!
//! | Material representation | `Material` | `DifferentiableMaterial` | `MeromorphicMaterial` | `DifferentiableMeromorphicMaterial` |
//! |---|:---:|:---:|:---:|:---:|
//! | Constant material | yes | yes | yes | yes |
//! | Drude or Lorentz model | automatic | automatic | yes | yes |
//! | Sellmeier fit | automatic | automatic | possibly | possibly |
//! | Cubic-spline optical data | yes | possibly | no | no |
//! | Piecewise-linear table | yes | possibly | no | no |
//! | Numerically differentiated table | yes | yes | no | no |
//!
//! Whether a Sellmeier or other empirical fit should implement the
//! complex-plane traits depends on the model's documented domain and whether
//! its mathematical continuation is considered physically appropriate for the
//! intended calculation.
//!
//! ## Numerical differentiation
//!
//! Numerical differentiation should be represented explicitly by an adapter,
//! for example:
//!
//! ```text
//! NumericallyDifferentiated<M>
//! ```
//!
//! Such an adapter may implement [`Material`] and
//! [`DifferentiableMaterial`], but must not implement
//! [`MeromorphicMaterial`] merely because it can estimate derivatives along the
//! real axis.
//!
//! This separation prevents real-axis numerical differentiation from being
//! used accidentally as though it were a valid analytic continuation in a
//! complex mode calculation.

pub mod erased;
pub mod evaluate;
pub mod handle;
pub mod lifting;
pub mod model;
pub mod sample;
// pub mod tensor;

pub use evaluate::{
    EvaluateDifferentiableMaterial, EvaluateDifferentiableMeromorphicMaterial, EvaluateMaterial,
    EvaluateMeromorphicMaterial,
};
pub use handle::{
    AnalyticalMaterialHandle, DifferentiableMaterialHandle, MaterialHandle,
    MeromorphicMaterialHandle,
};
pub(crate) use lifting::{ConstitutiveEvaluator, ConstitutiveLift, ConstitutiveSpectralFirstLift};
pub use model::Constant;
pub use sample::Scalar;

pub use sample::{Sampled, TensorSampled};
// use tensor::{DiagonalTensorMaterial, TensorMaterial};

use num_traits::{One, Zero};

use crate::ComplexScalar;

/// Highest derivative order requested.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum DerivativeOrder {
    First,
    Second,
    Third,
}

/// Pointwise optical material model defined on the real spectral axis.
///
/// A material provides the constitutive properties required by the transfer-
/// and scattering-matrix backends for real-frequency calculations.
///
/// The independent variable is the vacuum angular wavenumber
///
/// ```text
/// k₀ = ω / c = 2π / λ₀
/// ```
///
/// represented numerically in inverse centimetres.
///
/// This trait intentionally makes **no** guarantees about
///
/// - analytical derivatives,
/// - differentiability,
/// - complex-frequency continuation.
///
/// Consequently it can be implemented by
///
/// - analytical dispersion models,
/// - tabulated measurements,
/// - spline interpolation,
/// - externally supplied material databases,
/// - user-defined black-box models.
///
/// This capability is sufficient for
///
/// - reflectance,
/// - transmittance,
/// - absorptance,
/// - phase,
/// - ellipsometry,
/// - field reconstruction
///
/// evaluated on the real spectral axis.
pub trait Material {
    /// Real scalar type used by the material.
    type Real: One + Zero;

    /// Return the relative permittivity.
    fn relative_permittivity<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real>,
        I: Sampled<Elem = C::RealField>;

    /// Return the relative permeability.
    ///
    /// The default implementation represents a non-magnetic material.
    fn relative_permeability<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real>,
        I: Sampled<Elem = C::RealField>,
    {
        vacuum_wavenumber.map(|_| C::from_real(Self::Real::one()))
    }
}

/// Material model with derivatives on the real spectral axis.
///
/// This trait extends [`Material`] with spectral derivatives evaluated along
/// the real vacuum angular wavenumber axis.
///
/// The derivatives may be obtained by any mathematically appropriate method,
/// including
///
/// - symbolic differentiation,
/// - automatic differentiation,
/// - spline differentiation,
/// - finite differences,
/// - complex-step differentiation.
///
/// Consequently, implementing this trait **does not** imply that the material
/// possesses a meaningful continuation into the complex-frequency plane.
///
/// This capability is sufficient for analytical derivatives of
///
/// - reflection,
/// - transmission,
/// - absorptance,
/// - phase,
/// - group delay,
/// - ellipsometric quantities
///
/// evaluated on the real axis.
pub trait DifferentiableMaterial: Material {
    /// Return a derivative of the relative permittivity.
    fn relative_permittivity_derivative<I, C>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real>,
        I: Sampled<Elem = C::RealField>;

    /// Return a derivative of the relative permeability.
    ///
    /// The default implementation represents a non-magnetic material.
    fn relative_permeability_derivative<I, C>(
        &self,
        vacuum_wavenumber: I,
        _order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real>,
        I: Sampled<Elem = C::RealField>,
    {
        vacuum_wavenumber.map(|_| C::from_real(Self::Real::zero()))
    }
}

/// Material model possessing a meromorphic continuation into the complex
/// spectral plane.
///
/// Implementations define the constitutive response for complex vacuum angular
/// wavenumbers.
///
/// This continuation is intended for algorithms operating away from the real
/// axis, such as
///
/// - complex mode finding,
/// - contour integration,
/// - argument-principle methods,
/// - complex-frequency root searches.
///
/// Implementations must define a single-valued meromorphic continuation
/// consistent with the underlying physical dispersion model.
///
/// Typical implementations include
///
/// - Drude,
/// - Lorentz,
/// - Drude–Lorentz,
/// - oscillator models.
///
/// Tabulated measurements, piecewise interpolation and arbitrary numerical
/// fits should generally **not** implement this trait because they do not
/// possess a unique physically meaningful continuation away from the real
/// spectral axis.
pub trait MeromorphicMaterial: Material {
    /// Return the complex relative permittivity.
    fn relative_permittivity_complex<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real>,
        I: Sampled<Elem = C>;

    /// Return the complex relative permeability.
    fn relative_permeability_complex<I, C>(&self, vacuum_wavenumber: I) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real>,
        I: Sampled<Elem = C>;
}

/// Meromorphic material model with analytical complex derivatives.
///
/// This trait extends [`MeromorphicMaterial`] with derivatives throughout the
/// complex spectral plane.
///
/// These derivatives are required by algorithms which perform differentiation
/// away from the real axis, including
///
/// - Newton refinement of complex roots,
/// - argument-principle mode solvers,
/// - contour-integral methods,
/// - analytical modal sensitivities.
///
/// Implementations must return derivatives of the same meromorphic
/// continuation defined by [`MeromorphicMaterial`].
///
/// Every implementation of this trait also provides exact derivatives on the
/// real axis and may therefore implement [`DifferentiableMaterial`] directly.
pub trait DifferentiableMeromorphicMaterial: MeromorphicMaterial + DifferentiableMaterial {
    /// Return a complex derivative of the relative permittivity.
    fn relative_permittivity_complex_derivative<I, C>(
        &self,
        vacuum_wavenumber: I,
        order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real>,
        I: Sampled<Elem = C>;

    /// Return a complex derivative of the relative permeability.
    fn relative_permeability_complex_derivative<I, C>(
        &self,
        vacuum_wavenumber: I,
        _order: DerivativeOrder,
    ) -> I::Mapped<C>
    where
        C: ComplexScalar<RealField = Self::Real>,
        I: Sampled<Elem = C>,
    {
        vacuum_wavenumber.map(|_| C::from_real(Self::Real::zero()))
    }
}
