use ndarray::Dimension;

use crate::ComplexScalar;
use crate::backend::isotropic::IsotropicLayerQuantities;
use crate::stack::Thickness;

use super::ScatterMatrix2;

pub(crate) fn propagation_scattering_matrix<C, D>(
    quantities: &IsotropicLayerQuantities<C, D>,
    thickness: Thickness<C::RealField>,
) -> ScatterMatrix2<C, D>
where
    C: ComplexScalar,
    C::RealField: Copy,
    D: Dimension,
{
    let d = C::from_real(thickness.as_cm());

    let phase = quantities.kappa.mapv(|kappa| (C::i() * kappa * d).exp());

    let zero = phase.mapv(|_| C::zero());

    ScatterMatrix2::new(zero.clone(), phase.clone(), phase, zero)
}
