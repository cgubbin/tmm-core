use ndarray::Dimension;

use crate::ComplexScalar;

use super::ScatterMatrix2;
use crate::backend::isotropic::IsotropicLayerQuantities;

pub(crate) fn interface_scattering_matrix<C, D>(
    left: &IsotropicLayerQuantities<C, D>,
    right: &IsotropicLayerQuantities<C, D>,
) -> ScatterMatrix2<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    let left_admittance = left.admittance();
    let right_admittance = right.admittance();

    let denominator = left_admittance.value().to_owned() + right_admittance.value().view();

    let two = C::one() + C::one();

    let reflection_left =
        (left_admittance.value().to_owned() - right_admittance.value().view()) / denominator.view();

    let transmission_left = left_admittance.value().mapv(|x| two * x) / denominator.view();

    let transmission_right = right_admittance.value().mapv(|x| two * x) / denominator;

    ScatterMatrix2::new(
        reflection_left.clone(),
        transmission_right,
        transmission_left,
        -reflection_left,
    )
}
