use ndarray::{ArrayBase, Dimension, OwnedRepr};

use crate::{
    ComplexScalar,
    backend::Polarisation,
    backend::transfer2::quantities::IsotropicLayerQuantities,
    material::{DerivativeOrder, Material, Scalar, SpectralVariable},
};

#[derive(Clone, Debug, PartialEq)]
pub struct LayerFirstDerivatives<C, D>
where
    D: Dimension,
{
    pub dkappa: ArrayBase<OwnedRepr<C>, D>,
    pub dfactor: ArrayBase<OwnedRepr<C>, D>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct LayerSecondDerivatives<C, D>
where
    D: Dimension,
{
    pub first: LayerFirstDerivatives<C, D>,
    pub ddkappa: ArrayBase<OwnedRepr<C>, D>,
    pub ddfactor: ArrayBase<OwnedRepr<C>, D>,
}

pub fn vacuum_wavenumber_squared_derivatives<M, C, D>(
    material: &M,
    q: &IsotropicLayerQuantities<C, D>,
    vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    polarisation: Polarisation,
) -> LayerFirstDerivatives<C, D>
where
    M: Material<Real = C::RealField>,
    C: ComplexScalar,
    D: Dimension,
{
    let epsilon = &q.epsilon;
    let mu = &q.mu;
    let kappa = &q.kappa;

    let deps = vacuum_wavenumber.mapv(|k0| {
        material.relative_permittivity_derivative(
            Scalar(k0),
            DerivativeOrder::First,
            SpectralVariable::FrequencySquared,
        )
    });

    let dmu = vacuum_wavenumber.mapv(|k0| {
        material.relative_permeability_derivative(
            Scalar(k0),
            DerivativeOrder::First,
            SpectralVariable::FrequencySquared,
        )
    });

    let vacuum_wavenumber2 = vacuum_wavenumber.mapv(|k0| k0 * k0);

    let dq = (deps.clone() * mu.view() + epsilon.clone() * dmu.view()) * vacuum_wavenumber2
        + epsilon.clone() * mu.view();

    let two = C::one() + C::one();
    let dkappa = dq / kappa.mapv(|k| two * k);

    let dfactor = match polarisation {
        Polarisation::TransverseElectric => dmu,
        Polarisation::TransverseMagnetic => deps,
    };

    LayerFirstDerivatives { dkappa, dfactor }
}

pub fn parallel_wavenumber_squared_derivatives<C, D>(
    q: &IsotropicLayerQuantities<C, D>,
) -> LayerFirstDerivatives<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    let two = C::one() + C::one();

    let dkappa = q.kappa.mapv(|k| -C::one() / (two * k));
    let dfactor = q.factor.mapv(|_| C::zero());

    LayerFirstDerivatives { dkappa, dfactor }
}

pub fn vacuum_wavenumber_squared_second_derivatives<M, C, D>(
    material: &M,
    q: &IsotropicLayerQuantities<C, D>,
    vacuum_wavenumber: &ArrayBase<OwnedRepr<C>, D>,
    polarisation: Polarisation,
) -> LayerSecondDerivatives<C, D>
where
    M: Material<Real = C::RealField>,
    C: ComplexScalar,
    D: Dimension,
{
    let vacuum_wavenumber2 = vacuum_wavenumber.mapv(|k0| k0 * k0);

    let deps = vacuum_wavenumber.mapv(|k0| {
        material.relative_permittivity_derivative(
            Scalar(k0),
            DerivativeOrder::First,
            SpectralVariable::FrequencySquared,
        )
    });

    let ddeps = vacuum_wavenumber.mapv(|k0| {
        material.relative_permittivity_derivative(
            Scalar(k0),
            DerivativeOrder::Second,
            SpectralVariable::FrequencySquared,
        )
    });

    let dmu = vacuum_wavenumber.mapv(|k0| {
        material.relative_permeability_derivative(
            Scalar(k0),
            DerivativeOrder::First,
            SpectralVariable::FrequencySquared,
        )
    });

    let ddmu = vacuum_wavenumber.mapv(|k0| {
        material.relative_permeability_derivative(
            Scalar(k0),
            DerivativeOrder::Second,
            SpectralVariable::FrequencySquared,
        )
    });

    // q = ε μ ω² - β²
    //
    // dq/dω² =
    //     (ε' μ + ε μ') ω² + ε μ
    //
    // d²q/d(ω²)² =
    //     (ε'' μ + 2 ε' μ' + ε μ'') ω²
    //     + 2(ε' μ + ε μ')
    let a = deps.clone() * q.mu.view() + q.epsilon.clone() * dmu.view();

    let da = ddeps.clone() * q.mu.view()
        + deps.clone() * dmu.view()
        + deps.clone() * dmu.view()
        + q.epsilon.clone() * ddmu.view();

    let dq = a.clone() * vacuum_wavenumber2.view() + q.epsilon.clone() * q.mu.view();
    let ddq = da * vacuum_wavenumber2 + a.mapv(|x| x + x);

    let two = C::one() + C::one();

    let dkappa = dq.clone() / q.kappa.mapv(|k| two * k);

    // κ = sqrt(q)
    // κ'  = q' / (2κ)
    // κ'' = q'' / (2κ) - (q')² / (4κ³)
    let four = two + two;
    let ddkappa =
        ddq / q.kappa.mapv(|k| two * k) - dq.mapv(|x| x * x) / q.kappa.mapv(|k| four * k * k * k);

    let (dfactor, ddfactor) = match polarisation {
        Polarisation::TransverseElectric => (dmu, ddmu),
        Polarisation::TransverseMagnetic => (deps, ddeps),
    };

    LayerSecondDerivatives {
        ddkappa,
        ddfactor,
        first: LayerFirstDerivatives { dkappa, dfactor },
    }
}

pub fn parallel_wavenumber_squared_second_derivatives<C, D>(
    q: &IsotropicLayerQuantities<C, D>,
) -> LayerSecondDerivatives<C, D>
where
    C: ComplexScalar,
    D: Dimension,
{
    let two = C::one() + C::one();
    let four = two + two;

    // q = ε μ ω² - β², with x = β²
    // dq/dx = -1, d²q/dx² = 0
    //
    // κ'  = -1/(2κ)
    // κ'' = -1/(4κ³)
    let dkappa = q.kappa.mapv(|k| -C::one() / (two * k));
    let ddkappa = q.kappa.mapv(|k| -C::one() / (four * k * k * k));

    let zero = q.factor.mapv(|_| C::zero());

    LayerSecondDerivatives {
        ddkappa,
        ddfactor: zero.clone(),
        first: LayerFirstDerivatives {
            dkappa,
            dfactor: zero,
        },
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::{ArrayBase, Dimension, OwnedRepr, arr0};
    use num_complex::Complex64;

    use super::*;
    use crate::{
        backend::{
            PlanarInput, Polarisation,
            transfer2::{Matrix2, quantities::isotropic_layer_quantities},
        },
        material::Constant,
        stack::Thickness,
    };

    type C = Complex64;

    fn c(x: f64) -> C {
        C::new(x, 0.0)
    }

    fn assert_array_close<D>(
        actual: &ArrayBase<OwnedRepr<C>, D>,
        expected: &ArrayBase<OwnedRepr<C>, D>,
        tolerance: f64,
    ) where
        D: Dimension,
    {
        assert_eq!(actual.shape(), expected.shape());

        for (actual, expected) in actual.iter().zip(expected.iter()) {
            assert_relative_eq!(
                actual.re,
                expected.re,
                max_relative = tolerance,
                epsilon = tolerance
            );
            assert_relative_eq!(
                actual.im,
                expected.im,
                max_relative = tolerance,
                epsilon = tolerance
            );
        }
    }

    fn assert_matrix_close<D>(actual: &Matrix2<C, D>, expected: &Matrix2<C, D>, tolerance: f64)
    where
        D: Dimension,
    {
        assert_array_close(actual.m11(), expected.m11(), tolerance);
        assert_array_close(actual.m12(), expected.m12(), tolerance);
        assert_array_close(actual.m21(), expected.m21(), tolerance);
        assert_array_close(actual.m22(), expected.m22(), tolerance);
    }

    #[test]
    fn vacuum_wavenumber_squared_derivative_matches_finite_difference() {
        let material = Constant::new(2.25);
        let thickness = Thickness::from_nm(100.0).unwrap();

        let vacuum_wavenumber2 = 1000.0_f64.powi(2);
        let h = 1e-2 * vacuum_wavenumber2;

        let parallel_wavenumber2 = 10.0_f64.powi(2);

        let q = isotropic_layer_quantities(
            &material,
            &PlanarInput::new(
                arr0(c(vacuum_wavenumber2.sqrt())),
                arr0(c(parallel_wavenumber2.sqrt())),
                Polarisation::TransverseElectric,
            ),
        );

        let dq = vacuum_wavenumber_squared_derivatives(
            &material,
            &q,
            &arr0(c(vacuum_wavenumber2.sqrt())),
            Polarisation::TransverseElectric,
        );

        let analytical = Matrix2::spectral_derivative(&q, thickness, &dq);

        let plus = Matrix2::from_layer(
            &isotropic_layer_quantities(
                &material,
                &PlanarInput::new(
                    arr0(c((vacuum_wavenumber2 + h).sqrt())),
                    arr0(c(parallel_wavenumber2.sqrt())),
                    Polarisation::TransverseElectric,
                ),
            ),
            thickness,
        );

        let minus = Matrix2::from_layer(
            &isotropic_layer_quantities(
                &material,
                &PlanarInput::new(
                    arr0(c((vacuum_wavenumber2 - h).sqrt())),
                    arr0(c(parallel_wavenumber2.sqrt())),
                    Polarisation::TransverseElectric,
                ),
            ),
            thickness,
        );

        let expected = (&plus.add(&(&minus).scale(c(-1.0)))).scale(c(1.0 / (2.0 * h)));

        assert_matrix_close(&analytical, &expected, 1e-8);
    }

    #[test]
    fn parallel_wavenumber_squared_derivative_matches_finite_difference() {
        let material = Constant::new(2.25);
        let thickness = Thickness::from_nm(100.0).unwrap();

        let vacuum_wavenumber = arr0(c(1000.0));
        let parallel_wavenumber2 = 10.0_f64.powi(2);
        let h = 1e-3;

        let q = isotropic_layer_quantities(
            &material,
            &PlanarInput::new(
                vacuum_wavenumber.clone(),
                arr0(c(parallel_wavenumber2.sqrt())),
                Polarisation::TransverseElectric,
            ),
        );

        let dq = parallel_wavenumber_squared_derivatives(&q);

        let analytical = Matrix2::spectral_derivative(&q, thickness, &dq);

        let plus = Matrix2::from_layer(
            &isotropic_layer_quantities(
                &material,
                &PlanarInput::new(
                    vacuum_wavenumber.clone(),
                    arr0(c((parallel_wavenumber2 + h).sqrt())),
                    Polarisation::TransverseElectric,
                ),
            ),
            thickness,
        );

        let minus = Matrix2::from_layer(
            &isotropic_layer_quantities(
                &material,
                &PlanarInput::new(
                    vacuum_wavenumber,
                    arr0(c((parallel_wavenumber2 - h).sqrt())),
                    Polarisation::TransverseElectric,
                ),
            ),
            thickness,
        );

        let expected = (&plus.add(&(&minus).scale(c(-1.0)))).scale(c(1.0 / (2.0 * h)));

        assert_matrix_close(&analytical, &expected, 1e-8);
    }

    #[test]
    fn vacuum_wavenumber_squared_second_derivative_matches_finite_difference() {
        let material = Constant::new(2.25);
        let thickness = Thickness::from_nm(100.0).unwrap();

        let vacuum_wavenumber2 = 1000.0_f64.powi(2);
        let h = 1e-2 * vacuum_wavenumber2;
        let parallel_wavenumber2 = 100.0_f64;

        let q = isotropic_layer_quantities(
            &material,
            &PlanarInput::new(
                arr0(c(vacuum_wavenumber2.sqrt())),
                arr0(c(parallel_wavenumber2.sqrt())),
                Polarisation::TransverseElectric,
            ),
        );

        let ddq = vacuum_wavenumber_squared_second_derivatives(
            &material,
            &q,
            &arr0(c(vacuum_wavenumber2.sqrt())),
            Polarisation::TransverseElectric,
        );

        let analytical = Matrix2::spectral_second_derivative(&q, thickness, &ddq);

        let plus = Matrix2::from_layer(
            &isotropic_layer_quantities(
                &material,
                &PlanarInput::new(
                    arr0(c((vacuum_wavenumber2 + h).sqrt())),
                    arr0(c(parallel_wavenumber2.sqrt())),
                    Polarisation::TransverseElectric,
                ),
            ),
            thickness,
        );

        let zero = Matrix2::from_layer(
            &isotropic_layer_quantities(
                &material,
                &PlanarInput::new(
                    arr0(c(vacuum_wavenumber2.sqrt())),
                    arr0(c(parallel_wavenumber2.sqrt())),
                    Polarisation::TransverseElectric,
                ),
            ),
            thickness,
        );

        let minus = Matrix2::from_layer(
            &isotropic_layer_quantities(
                &material,
                &PlanarInput::new(
                    arr0(c((vacuum_wavenumber2 - h).sqrt())),
                    arr0(c(parallel_wavenumber2.sqrt())),
                    Polarisation::TransverseElectric,
                ),
            ),
            thickness,
        );

        let expected = (&plus.add(&(&zero).scale(c(-2.0))).add(&minus)).scale(c(1.0 / (h * h)));

        assert_matrix_close(&analytical, &expected, 1e-4);
    }

    #[test]
    fn parallel_wavenumber_squared_second_derivative_matches_finite_difference() {
        let material = Constant::new(2.25);
        let thickness = Thickness::from_nm(100.0).unwrap();

        let vacuum_wavenumber = 1000.0_f64;
        let parallel_wavenumber2 = 100.0_f64;
        let h = 1e-2;
        let q = isotropic_layer_quantities(
            &material,
            &PlanarInput::new(
                arr0(c(vacuum_wavenumber)),
                arr0(c(parallel_wavenumber2.sqrt())),
                Polarisation::TransverseElectric,
            ),
        );

        let ddq = parallel_wavenumber_squared_second_derivatives(&q);
        let analytical = Matrix2::spectral_second_derivative(&q, thickness, &ddq);

        let plus = Matrix2::from_layer(
            &isotropic_layer_quantities(
                &material,
                &PlanarInput::new(
                    arr0(c(vacuum_wavenumber)),
                    arr0(c((parallel_wavenumber2 + h).sqrt())),
                    Polarisation::TransverseElectric,
                ),
            ),
            thickness,
        );

        let zero = Matrix2::from_layer(
            &isotropic_layer_quantities(
                &material,
                &PlanarInput::new(
                    arr0(c(vacuum_wavenumber)),
                    arr0(c((parallel_wavenumber2).sqrt())),
                    Polarisation::TransverseElectric,
                ),
            ),
            thickness,
        );

        let minus = Matrix2::from_layer(
            &isotropic_layer_quantities(
                &material,
                &PlanarInput::new(
                    arr0(c(vacuum_wavenumber)),
                    arr0(c((parallel_wavenumber2 - h).sqrt())),
                    Polarisation::TransverseElectric,
                ),
            ),
            thickness,
        );

        let expected = (&plus.add(&(&zero).scale(c(-2.0))).add(&minus)).scale(c(1.0 / (h * h)));

        assert_matrix_close(&analytical, &expected, 1e-5);
    }
}
