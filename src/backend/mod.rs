use crate::{ComplexScalar, stack::Stack};
use ndarray::{ArrayBase, OwnedRepr};

mod derivative;
mod input;
mod isotropic;
mod jet;
mod matrix;
mod mode;
mod plane_wave;

mod scatter2;
mod transfer2;

pub use derivative::{DerivativeRequest, DerivativeVariable};
pub use input::{PlanarInput, PlaneWaveInput, Polarisation};
pub use matrix::{MatrixDerivatives, MatrixEvaluation, RawMatrixBackend};
pub use mode::{AnalyticResidual, OutgoingModeBackend, ResidualDerivatives};
pub use plane_wave::{PlaneWaveBackend, PlaneWaveResponse};

use isotropic::IsotropicLayerQuantities;
