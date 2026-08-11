/// Marker selecting constitutive evaluation on the real spectral axis.
#[doc(hidden)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct RealAxis;

/// Marker selecting meromorphic constitutive evaluation in the complex
/// spectral plane.
#[doc(hidden)]
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct ComplexPlane;
