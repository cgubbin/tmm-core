pub(crate) mod assertions;
pub(crate) mod expected;
pub(crate) mod finite_difference;
pub(crate) mod jet;
pub(crate) mod material_model;
pub(crate) mod materials;
pub(crate) mod planar;
pub(crate) mod stack;

use num_complex::Complex64;

pub const TOLERANCE: f64 = 1.0e-12;
pub type C = Complex64;

pub fn c(x: f64) -> C {
    C::new(x, 0.0)
}
